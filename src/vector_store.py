"""
Vector Store Module for Support Bot

Handles:
- Chroma vector database integration
- Document storage with metadata
- Similarity search
- Knowledge base management
"""

from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

from src.config import (
    CHROMA_DB_PATH, CHROMA_COLLECTION_NAME,
    RAG_K_RETRIEVAL, RAG_SIMILARITY_THRESHOLD
)
from src.logger import logger
from src.embeddings import EmbeddingGenerator
from src.data_processor import DocumentChunk


class VectorStore:
    """
    Vector store for knowledge base using Chroma.
    
    Features:
    - SQLite-backed persistent storage
    - Semantic search
    - Metadata filtering
    - Collection management
    """
    
    def __init__(self, db_path: str = CHROMA_DB_PATH, embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize vector store.
        
        Args:
            db_path: Path to Chroma database
            embedding_generator: EmbeddingGenerator instance (creates new if None)
        """
        self.logger = logger
        self.db_path = db_path
        
        if not HAS_CHROMA:
            self.logger.error("chromadb not installed")
            raise ImportError("chromadb package required")
        
        # Initialize embedding generator
        if embedding_generator is None:
            self.embedding_generator = EmbeddingGenerator()
        else:
            self.embedding_generator = embedding_generator
        
        # Initialize Chroma client
        self.logger.info(f"Initializing Chroma at {db_path}")
        
        self.client = chromadb.PersistentClient(
            path=db_path
        )

        self.collection = None
        
        self.stats = {
            'documents_added': 0,
            'documents_removed': 0,
            'total_documents': 0,
            'searches_performed': 0,
        }
    
    # ============= COLLECTION MANAGEMENT =============
    
    def get_or_create_collection(self, name: str = CHROMA_COLLECTION_NAME) -> None:
        """
        Get or create a collection.
        
        Args:
            name: Collection name
        """
        self.logger.info(f"Getting or creating collection: {name}")
        
        # Delete existing if needed (fresh start)
        try:
            self.client.delete_collection(name=name)
            self.logger.info(f"Deleted existing collection: {name}")
        except Exception:
            pass
        
        # Create new collection
        self.collection = self.client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        self.logger.info(f"Collection ready: {name}")
    
    def delete_collection(self, name: str = CHROMA_COLLECTION_NAME) -> None:
        """Delete a collection."""
        self.logger.info(f"Deleting collection: {name}")
        try:
            self.client.delete_collection(name=name)
            self.logger.info(f"Collection deleted: {name}")
        except Exception as e:
            self.logger.warning(f"Error deleting collection: {e}")
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        try:
            collections = self.client.list_collections()
            return [c.name for c in collections]
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            return []
    
    # ============= DOCUMENT ADDITION =============
    
    def add_documents(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            Number of documents added
        """
        if self.collection is None:
            self.logger.error("Collection not initialized. Call get_or_create_collection() first.")
            return 0
        
        self.logger.info(f"Adding {len(chunks)} chunks to collection")
        
        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_generator.embed_batch(texts, show_progress=show_progress)
        
        # Validate embeddings
        if not self.embedding_generator.validate_embeddings(embeddings):
            self.logger.error("Embedding validation failed")
            return 0
        
        # Add to collection in batches
        added_count = 0
        iterator = tqdm(
            range(0, len(chunks), batch_size),
            desc="Adding to vector store",
            disable=not show_progress
        ) if show_progress else range(0, len(chunks), batch_size)
        
        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            batch_embeddings = embeddings[batch_start:batch_end]
            
            try:
                self.collection.add(
                    ids=[chunk.chunk_id for chunk in batch_chunks],
                    embeddings=batch_embeddings,
                    metadatas=[chunk.metadata for chunk in batch_chunks],
                    documents=[chunk.text for chunk in batch_chunks],
                )
                added_count += len(batch_chunks)
            except Exception as e:
                self.logger.error(f"Error adding batch: {e}")
                continue
        
        self.stats['documents_added'] += added_count
        self.stats['total_documents'] = self.count_documents()
        
        self.logger.info(f"Successfully added {added_count} chunks")
        return added_count
    
    def delete_documents(self, chunk_ids: List[str]) -> int:
        """
        Delete documents by ID.
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Returns:
            Number of documents deleted
        """
        if self.collection is None:
            self.logger.error("Collection not initialized")
            return 0
        
        try:
            self.collection.delete(ids=chunk_ids)
            deleted_count = len(chunk_ids)
            self.stats['documents_removed'] += deleted_count
            self.stats['total_documents'] = self.count_documents()
            return deleted_count
        except Exception as e:
            self.logger.error(f"Error deleting documents: {e}")
            return 0
    
    def count_documents(self) -> int:
        """Get total number of documents in collection."""
        if self.collection is None:
            return 0
        try:
            return self.collection.count()
        except Exception as e:
            self.logger.warning(f"Error counting documents: {e}")
            return 0
    
    # ============= SIMILARITY SEARCH =============
    
    def search_by_text(
        self,
        query: str,
        k: int = RAG_K_RETRIEVAL,
        similarity_threshold: float = RAG_SIMILARITY_THRESHOLD,
    ) -> List[Dict]:
        """
        Search for documents similar to a query text.
        
        Args:
            query: Query text
            k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of matching documents with scores
        """
        if self.collection is None:
            self.logger.error("Collection not initialized")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.embed_text(query)
            
            # Search in Chroma
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
            )
            
            # Format results
            documents = []
            if results and results['documents'] and len(results['documents']) > 0:
                for idx, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][idx]
                    # Convert distance to similarity (1 - distance for cosine)
                    similarity = 1 - distance
                    
                    # Filter by threshold
                    if similarity >= similarity_threshold:
                        doc_result = {
                            'id': results['ids'][0][idx],
                            'text': doc,
                            'similarity': float(similarity),
                            'metadata': results['metadatas'][0][idx] if results['metadatas'] else {},
                        }
                        documents.append(doc_result)
            
            self.stats['searches_performed'] += 1
            self.logger.debug(f"Search returned {len(documents)} results for query: {query[:50]}...")
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            return []
    
    def search_by_embedding(
        self,
        embedding: np.ndarray,
        k: int = RAG_K_RETRIEVAL,
        similarity_threshold: float = RAG_SIMILARITY_THRESHOLD,
    ) -> List[Dict]:
        """
        Search for documents similar to a query embedding.
        
        Args:
            embedding: Query embedding
            k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of matching documents with scores
        """
        if self.collection is None:
            self.logger.error("Collection not initialized")
            return []
        
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=k,
            )
            
            # Format results
            documents = []
            if results and results['documents'] and len(results['documents']) > 0:
                for idx, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][idx]
                    similarity = 1 - distance
                    
                    if similarity >= similarity_threshold:
                        doc_result = {
                            'id': results['ids'][0][idx],
                            'text': doc,
                            'similarity': float(similarity),
                            'metadata': results['metadatas'][0][idx] if results['metadatas'] else {},
                        }
                        documents.append(doc_result)
            
            self.stats['searches_performed'] += 1
            return documents
            
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            return []
    
    def search_by_metadata(self, metadata_filter: Dict, k: int = RAG_K_RETRIEVAL) -> List[Dict]:
        """
        Search documents by metadata filter.
        
        Args:
            metadata_filter: Metadata filter conditions
            k: Number of results
            
        Returns:
            List of matching documents
        """
        if self.collection is None:
            self.logger.error("Collection not initialized")
            return []
        
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=k,
            )
            
            documents = []
            if results and results['documents']:
                for idx, doc in enumerate(results['documents']):
                    doc_result = {
                        'id': results['ids'][idx],
                        'text': doc,
                        'metadata': results['metadatas'][idx] if results['metadatas'] else {},
                    }
                    documents.append(doc_result)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error during metadata search: {e}")
            return []
    
    # ============= RETRIEVAL CONTEXT =============
    
    def get_context(
        self,
        query: str,
        k: int = RAG_K_RETRIEVAL,
        similarity_threshold: float = RAG_SIMILARITY_THRESHOLD,
    ) -> Dict:
        """
        Get formatted context for a query (for RAG).
        
        Args:
            query: Query text
            k: Number of documents
            similarity_threshold: Minimum similarity
            
        Returns:
            Context dict with documents and metadata
        """
        documents = self.search_by_text(query, k=k, similarity_threshold=similarity_threshold)
        
        context = {
            'query': query,
            'document_count': len(documents),
            'min_similarity': min([d['similarity'] for d in documents], default=0),
            'max_similarity': max([d['similarity'] for d in documents], default=0),
            'documents': documents,
            'formatted_text': self._format_context(documents),
        }
        
        return context
    
    @staticmethod
    def _format_context(documents: List[Dict]) -> str:
        """Format documents as readable context string."""
        if not documents:
            return "No relevant documents found."
        
        lines = []
        for idx, doc in enumerate(documents, 1):
            source = doc['metadata'].get('source_type', 'unknown')
            similarity = doc['similarity']
            text = doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
            
            lines.append(f"[{idx}] ({source}, {similarity:.2%}) {text}")
        
        return "\n".join(lines)
    
    # ============= STATISTICS & PERSISTENCE =============
    
    def persist(self) -> None:
        """Persist database to disk."""
        try:
            self.logger.info("Persistence handled automatically")
            self.logger.info("Vector store persisted")
        except Exception as e:
            self.logger.warning(f"Error persisting: {e}")
    
    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        return {
            **self.stats,
            'db_path': self.db_path,
            'collection_name': CHROMA_COLLECTION_NAME,
            'current_documents': self.count_documents(),
        }
    
    def print_stats(self) -> None:
        """Print statistics to logger."""
        stats = self.get_stats()
        self.logger.info("Vector Store Statistics:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")


def main():
    """Example usage."""
    from src.data_processor import DataProcessor
    
    # Create vector store
    store = VectorStore()
    store.get_or_create_collection()
    
    # Load and process data
    processor = DataProcessor()
    chunks, tickets = processor.process_all_data(
        tickets_csv="data/sample/sample_tickets.csv",
        docs_dir="data/sample",
    )
    
    # Add to vector store
    print(f"\nAdding {len(chunks)} chunks to vector store...")
    added = store.add_documents(chunks)
    print(f"✅ Added {added} documents")
    
    # Search examples
    queries = [
        "How do I reset my password?",
        "I was charged twice",
        "API timeout errors",
    ]
    
    print(f"\n📚 Search Examples:")
    for query in queries:
        context = store.get_context(query, k=3)
        print(f"\nQuery: {query}")
        print(f"Found: {context['document_count']} documents")
        print(f"Context:\n{context['formatted_text']}")
    
    # Stats
    print(f"\n📊 Stats:")
    store.print_stats()
    
    # Persist
    store.persist()


if __name__ == "__main__":
    main()
