"""
RAG (Retrieval-Augmented Generation) Retriever Module

Handles:
- Query expansion and rephrasing
- Multi-step retrieval strategies
- Context ranking and filtering
- Confidence scoring
- Integration with vector store for semantic search
"""

from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from tqdm import tqdm

from src.vector_store import VectorStore
from src.embeddings import EmbeddingGenerator
from src.config import (
    RAG_K_RETRIEVAL, RAG_SIMILARITY_THRESHOLD, RAG_RELEVANCE_THRESHOLD,
    EMBEDDING_DIMENSION
)
from src.logger import logger


@dataclass
class RetrievedDocument:
    """A retrieved document with scoring metadata."""
    chunk_id: str
    text: str
    source: str
    similarity_score: float
    rank: int
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'source': self.source,
            'similarity_score': self.similarity_score,
            'rank': self.rank,
            'metadata': self.metadata,
        }


@dataclass
class RetrievalContext:
    """Complete retrieval context for RAG."""
    original_query: str
    retrieved_documents: List[RetrievedDocument]
    confidence_score: float
    retrieval_method: str
    query_variations: List[str] = field(default_factory=list)
    total_documents: int = 0
    min_similarity: float = 0.0
    max_similarity: float = 0.0
    formatted_context: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'original_query': self.original_query,
            'retrieved_documents': [d.to_dict() for d in self.retrieved_documents],
            'confidence_score': self.confidence_score,
            'retrieval_method': self.retrieval_method,
            'query_variations': self.query_variations,
            'total_documents': self.total_documents,
            'min_similarity': self.min_similarity,
            'max_similarity': self.max_similarity,
            'formatted_context': self.formatted_context,
        }


class RAGRetriever:
    """
    Retrieval-Augmented Generation (RAG) retriever.
    
    Features:
    - Query expansion (paraphrasing)
    - Multi-step retrieval
    - Context ranking
    - Confidence scoring
    - Cross-document synthesis
    """
    
    def __init__(self, vector_store: VectorStore, embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize RAG retriever.
        
        Args:
            vector_store: VectorStore instance
            embedding_generator: EmbeddingGenerator instance (uses store's if None)
        """
        self.logger = logger
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator or vector_store.embedding_generator
        
        self.stats = {
            'queries_processed': 0,
            'documents_retrieved': 0,
            'avg_confidence': 0.0,
            'avg_top_similarity': 0.0,
        }
    
    # ============= QUERY EXPANSION =============
    
    def expand_query(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generate query variations for multi-angle retrieval.
        
        Strategies:
        - Original query
        - Add synonyms
        - Rephrase as statement
        - Add context keywords
        - Simplify technical terms
        
        Args:
            query: Original query
            num_variations: Number of variations to generate
            
        Returns:
            List of query variations
        """
        variations = [query]  # Always include original
        
        # Variation 1: Question to statement
        statement = self._question_to_statement(query)
        if statement != query:
            variations.append(statement)
        
        # Variation 2: Add related terms
        expanded = self._expand_with_related_terms(query)
        if expanded != query:
            variations.append(expanded)
        
        # Variation 3: Technical to simple
        simplified = self._simplify_technical_terms(query)
        if simplified != query:
            variations.append(simplified)
        
        # Variation 4: Short form
        short = self._create_short_form(query)
        if short != query:
            variations.append(short)
        
        # Return requested number
        return variations[:num_variations]
    
    @staticmethod
    def _question_to_statement(query: str) -> str:
        """Convert question to statement."""
        # Remove common question markers
        statement = query
        
        # Remove question marks
        statement = statement.rstrip('?').strip()
        
        # Convert "How do I X?" to "How to X" or "X instructions"
        if statement.lower().startswith('how do i '):
            statement = statement[len('how do i '):].strip()
            statement = f"how to {statement}"
        elif statement.lower().startswith('how can i '):
            statement = statement[len('how can i '):].strip()
            statement = f"how to {statement}"
        elif statement.lower().startswith('what is '):
            statement = statement[len('what is '):].strip()
            statement = f"definition of {statement}"
        elif statement.lower().startswith('why '):
            # Keep similar structure
            statement = statement[len('why '):].strip()
            statement = f"reason for {statement}"
        
        return statement
    
    @staticmethod
    def _expand_with_related_terms(query: str) -> str:
        """Expand query with related terms."""
        # Simple keyword expansion mappings
        expansions = {
            'password': 'password reset authentication login credentials',
            'payment': 'payment billing charge transaction',
            'api': 'API endpoints integration technical',
            'account': 'account profile user credentials',
            'refund': 'refund money back reimbursement credit',
            'error': 'error bug issue problem exception',
            'timeout': 'timeout latency slow performance',
            '2fa': '2fa two-factor authentication security',
        }
        
        expanded = query
        for key, expansion in expansions.items():
            if key.lower() in query.lower():
                expanded = query + " " + expansion
                break
        
        return expanded
    
    @staticmethod
    def _simplify_technical_terms(query: str) -> str:
        """Simplify technical terms."""
        simplifications = {
            'api': 'application programming interface',
            'http': 'web request',
            'ssl': 'security certificate',
            'tcp': 'network connection',
            'database': 'data storage',
            '401': 'unauthorized access',
            '500': 'server error',
            'json': 'data format',
            'rest': 'web service',
        }
        
        simplified = query
        for tech, simple in simplifications.items():
            if tech.lower() in query.lower():
                simplified = query.replace(tech, simple)
                break
        
        return simplified
    
    @staticmethod
    def _create_short_form(query: str) -> str:
        """Create short form of query."""
        # Extract key terms
        words = query.split()
        
        # Remove common words
        stopwords = {'how', 'do', 'i', 'can', 'what', 'is', 'a', 'the', 'to', 'my', 'me'}
        important_words = [w for w in words if w.lower() not in stopwords and len(w) > 2]
        
        short = " ".join(important_words[:4]) if important_words else query
        return short if short != query else query
    
    # ============= MULTI-STEP RETRIEVAL =============
    
    def retrieve_multi_step(
        self,
        query: str,
        k: int = RAG_K_RETRIEVAL,
        threshold: float = RAG_SIMILARITY_THRESHOLD,
    ) -> RetrievalContext:
        """
        Multi-step retrieval strategy:
        1. Expand query into variations
        2. Retrieve for each variation
        3. Merge and rank results
        4. Calculate confidence
        
        Args:
            query: User query
            k: Number of documents per variation
            threshold: Similarity threshold
            
        Returns:
            RetrievalContext with ranked documents
        """
        self.logger.info(f"Multi-step retrieval for: {query}")
        
        # Step 1: Expand query
        variations = self.expand_query(query, num_variations=3)
        self.logger.debug(f"Query variations: {variations}")
        
        # Step 2: Retrieve for each variation
        all_results = {}
        for variation in variations:
            results = self.vector_store.search_by_text(variation, k=k, similarity_threshold=threshold)
            for doc in results:
                doc_id = doc['id']
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        'text': doc['text'],
                        'source': doc['metadata'].get('source_type', 'unknown'),
                        'metadata': doc['metadata'],
                        'scores': [],
                    }
                all_results[doc_id]['scores'].append(doc['similarity'])
        
        # Step 3: Merge and rank
        ranked_docs = self._rank_merged_results(all_results)
        
        # Step 4: Calculate confidence
        confidence = self._calculate_confidence(ranked_docs)
        
        # Build context
        context = RetrievalContext(
            original_query=query,
            retrieved_documents=ranked_docs[:k],
            confidence_score=confidence,
            retrieval_method='multi-step',
            query_variations=variations,
            total_documents=len(ranked_docs),
            min_similarity=min([d.similarity_score for d in ranked_docs], default=0),
            max_similarity=max([d.similarity_score for d in ranked_docs], default=0),
        )
        
        # Format context
        context.formatted_context = self._format_context(context.retrieved_documents)
        
        self._update_stats(context)
        
        return context
    
    def _rank_merged_results(self, results: Dict) -> List[RetrievedDocument]:
        """
        Rank merged results from multiple queries.
        
        Scoring strategy:
        - Average similarity across all queries
        - Prefer documents that matched multiple variations
        - Boost based on source type
        """
        ranked = []
        
        for doc_id, data in results.items():
            # Average similarity
            avg_similarity = np.mean(data['scores'])
            
            # Boost for multiple matches
            match_boost = min(len(data['scores']) * 0.1, 0.3)
            
            # Boost for ticket sources (historical context)
            source_boost = 0.1 if data['source'] == 'ticket' else 0.0
            
            # Final score
            final_score = min(avg_similarity + match_boost + source_boost, 1.0)
            
            doc = RetrievedDocument(
                chunk_id=doc_id,
                text=data['text'],
                source=data['source'],
                similarity_score=float(final_score),
                rank=0,
                metadata=data['metadata'],
            )
            ranked.append(doc)
        
        # Sort by similarity (descending)
        ranked.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks
        for idx, doc in enumerate(ranked, 1):
            doc.rank = idx
        
        return ranked
    
    # ============= CONTEXT RANKING & FILTERING =============
    
    def rank_by_relevance(
        self,
        query: str,
        documents: List[Dict],
        category: Optional[str] = None,
    ) -> List[RetrievedDocument]:
        """
        Rank documents by relevance to query.
        
        Considers:
        - Semantic similarity
        - Document category
        - Freshness
        - Source type
        
        Args:
            query: Query text
            documents: List of retrieved documents
            category: Optional category filter
            
        Returns:
            Ranked list of documents
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_text(query)
        
        ranked = []
        for idx, doc in enumerate(documents):
            retrieved_doc = RetrievedDocument(
                chunk_id=doc.get('id', f'doc_{idx}'),
                text=doc['text'],
                source=doc['metadata'].get('source_type', 'unknown'),
                similarity_score=doc.get('similarity', 0.5),
                rank=0,
                metadata=doc.get('metadata', {}),
            )
            
            # Apply category filter
            if category and doc['metadata'].get('category', '').lower() != category.lower():
                # Reduce score for category mismatch
                retrieved_doc.similarity_score *= 0.8
            
            ranked.append(retrieved_doc)
        
        # Sort by score
        ranked.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks
        for idx, doc in enumerate(ranked, 1):
            doc.rank = idx
        
        return ranked
    
    def filter_by_threshold(
        self,
        documents: List[RetrievedDocument],
        threshold: float = RAG_RELEVANCE_THRESHOLD,
    ) -> List[RetrievedDocument]:
        """
        Filter documents by relevance threshold.
        
        Args:
            documents: List of retrieved documents
            threshold: Minimum relevance score
            
        Returns:
            Filtered documents above threshold
        """
        filtered = [d for d in documents if d.similarity_score >= threshold]
        
        self.logger.debug(
            f"Filtered {len(documents)} → {len(filtered)} documents "
            f"(threshold: {threshold})"
        )
        
        return filtered
    
    # ============= CONFIDENCE SCORING =============
    
    def _calculate_confidence(self, documents: List[RetrievedDocument]) -> float:
        """
        Calculate confidence score for retrieved context.
        
        Factors:
        - Top document similarity
        - Number of relevant documents
        - Consistency across results
        - Diversity of sources
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Confidence score (0-1)
        """
        if not documents:
            return 0.0
        
        scores = [d.similarity_score for d in documents]
        
        # Factor 1: Top document quality
        top_score = max(scores) if scores else 0
        
        # Factor 2: Number of relevant docs
        relevant_count = sum(1 for s in scores if s >= 0.5)
        num_factor = min(relevant_count / 5, 1.0)  # Max at 5 docs
        
        # Factor 3: Score consistency (variance)
        if len(scores) > 1:
            variance = np.var(scores)
            consistency_factor = 1.0 - min(variance, 1.0)
        else:
            consistency_factor = 1.0
        
        # Factor 4: Source diversity
        sources = set(d.source for d in documents)
        diversity_factor = min(len(sources) / 3, 1.0)  # Boost for diverse sources
        
        # Weighted confidence
        confidence = (
            0.4 * top_score +
            0.25 * num_factor +
            0.2 * consistency_factor +
            0.15 * diversity_factor
        )
        
        return float(min(max(confidence, 0.0), 1.0))
    
    # ============= CONTEXT FORMATTING =============
    
    @staticmethod
    def _format_context(documents: List[RetrievedDocument], max_length: int = 4000) -> str:
        """
        Format retrieved documents as context string for LLM.
        
        Args:
            documents: Retrieved documents
            max_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        lines = ["=== RELEVANT CONTEXT ===\n"]
        current_length = 0
        
        for doc in documents:
            # Document header
            header = f"[{doc.rank}] {doc.source.upper()} ({doc.similarity_score:.1%})"
            header_text = f"\n{header}\n{'-' * len(header)}\n"
            
            # Document text (truncate if too long)
            doc_text = doc.text
            if len(doc_text) > 300:
                doc_text = doc_text[:300] + "..."
            
            # Combine
            section = header_text + doc_text + "\n"
            
            # Check length
            if current_length + len(section) > max_length:
                # Add truncation notice
                lines.append(f"\n[... truncated, {len(documents) - len(lines)} more documents ...]")
                break
            
            lines.append(section)
            current_length += len(section)
        
        lines.append("\n=== END CONTEXT ===")
        
        return "\n".join(lines)
    
    # ============= CATEGORY-AWARE RETRIEVAL =============
    
    def retrieve_by_category(
        self,
        query: str,
        category: Optional[str] = None,
        k: int = RAG_K_RETRIEVAL,
    ) -> RetrievalContext:
        """
        Retrieve documents filtered by category.
        
        Args:
            query: User query
            category: Document category (technical, billing, account, etc.)
            k: Number of documents
            
        Returns:
            Filtered retrieval context
        """
        self.logger.info(f"Retrieving by category: {category} for query: {query}")
        
        # Retrieve without category filter first
        all_docs = self.vector_store.search_by_text(query, k=k*2)
        
        # Filter by category
        if category:
            filtered_docs = [
                d for d in all_docs
                if d['metadata'].get('category', '').lower() == category.lower()
            ]
            if not filtered_docs:
                self.logger.warning(f"No documents for category: {category}, using all")
                filtered_docs = all_docs
        else:
            filtered_docs = all_docs
        
        # Rank
        ranked = self.rank_by_relevance(query, filtered_docs[:k], category)
        
        # Calculate confidence
        confidence = self._calculate_confidence(ranked)
        
        # Build context
        context = RetrievalContext(
            original_query=query,
            retrieved_documents=ranked,
            confidence_score=confidence,
            retrieval_method=f'category-aware ({category})',
            total_documents=len(ranked),
            min_similarity=min([d.similarity_score for d in ranked], default=0),
            max_similarity=max([d.similarity_score for d in ranked], default=0),
        )
        
        context.formatted_context = self._format_context(ranked)
        
        self._update_stats(context)
        
        return context
    
    # ============= STATISTICS =============
    
    def _update_stats(self, context: RetrievalContext) -> None:
        """Update statistics."""
        self.stats['queries_processed'] += 1
        self.stats['documents_retrieved'] += len(context.retrieved_documents)
        
        # Update averages
        all_confidences = getattr(self, '_confidences', [])
        all_confidences.append(context.confidence_score)
        self._confidences = all_confidences
        self.stats['avg_confidence'] = np.mean(all_confidences)
        
        all_similarities = getattr(self, '_top_similarities', [])
        if context.retrieved_documents:
            all_similarities.append(context.retrieved_documents[0].similarity_score)
            self._top_similarities = all_similarities
            self.stats['avg_top_similarity'] = np.mean(all_similarities)
    
    def get_stats(self) -> Dict:
        """Get retrieval statistics."""
        return self.stats.copy()
    
    def print_stats(self) -> None:
        """Print statistics to logger."""
        stats = self.get_stats()
        self.logger.info("RAG Retriever Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.2f}")
            else:
                self.logger.info(f"  {key}: {value}")


def main():
    """Example usage."""
    from src.data_processor import DataProcessor
    from src.vector_store import VectorStore
    
    # Initialize components
    print("Initializing components...")
    store = VectorStore()
    store.get_or_create_collection()
    
    # Load and ingest data
    processor = DataProcessor()
    chunks, _ = processor.process_all_data(
        tickets_csv="data/sample/sample_tickets.csv",
        docs_dir="data/sample",
    )
    
    print(f"Ingesting {len(chunks)} chunks...")
    store.add_documents(chunks, show_progress=False)
    
    # Create retriever
    print("\nInitializing RAG retriever...")
    retriever = RAGRetriever(store)
    
    # Test queries
    test_queries = [
        "How do I reset my password?",
        "I was charged twice for my subscription",
        "API timeout errors",
        "Enable two-factor authentication",
        "Refund request process",
    ]
    
    print("\n" + "="*60)
    print("RAG RETRIEVAL EXAMPLES")
    print("="*60)
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 60)
        
        # Retrieve with multi-step
        context = retriever.retrieve_multi_step(query, k=3)
        
        print(f"Confidence: {context.confidence_score:.1%}")
        print(f"Documents: {context.total_documents}")
        print(f"Variations: {context.query_variations}")
        print(f"\nContext:\n{context.formatted_context}")
    
    # Statistics
    print("\n" + "="*60)
    print("📊 Statistics:")
    retriever.print_stats()


if __name__ == "__main__":
    main()
