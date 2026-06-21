"""
Embeddings Module for Support Bot

Handles:
- Loading pre-trained embedding models (Sentence Transformers)
- Generating embeddings for text
- Batch processing with caching
- Embedding statistics and quality checks
"""

import os
from typing import List, Union, Optional, Dict
import logging
import hashlib
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from src.config import (
    EMBEDDING_MODEL, EMBEDDING_DIMENSION, BATCH_EMBEDDING_SIZE,
    KNOWLEDGE_BASE_DIR, CACHE_EMBEDDINGS
)
from src.logger import logger


class EmbeddingGenerator:
    """
    Generates embeddings using Sentence Transformers.
    
    Supports:
    - Single text embeddings
    - Batch embeddings
    - Caching for repeated queries
    - Dimension verification
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Sentence Transformer model name (default: all-MiniLM-L6-v2)
        """
        self.logger = logger
        self.model_name = model_name
        self.cache_dir = KNOWLEDGE_BASE_DIR / "embeddings_cache"
        self.cache_enabled = CACHE_EMBEDDINGS
        
        # Create cache directory
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        if not HAS_SENTENCE_TRANSFORMERS:
            self.logger.error("sentence-transformers not installed")
            raise ImportError("sentence-transformers package required")
        
        self.logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.logger.info(f"Model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")
        
        self.stats = {
            'total_embeddings': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_operations': 0,
        }
    
    # ============= SINGLE TEXT EMBEDDING =============
    
    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            use_cache: Whether to use cached embedding if available
            
        Returns:
            Numpy array of shape (embedding_dimension,)
        """
        # Check cache first
        if use_cache and self.cache_enabled:
            cached = self._get_from_cache(text)
            if cached is not None:
                self.stats['cache_hits'] += 1
                return cached
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Verify dimension
        if embedding.shape[0] != EMBEDDING_DIMENSION:
            self.logger.warning(
                f"Unexpected embedding dimension: {embedding.shape[0]} "
                f"(expected {EMBEDDING_DIMENSION})"
            )
        
        # Cache it
        if use_cache and self.cache_enabled:
            self._save_to_cache(text, embedding)
        
        self.stats['cache_misses'] += 1
        self.stats['total_embeddings'] += 1
        
        return embedding
    
    # ============= BATCH EMBEDDINGS =============
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = BATCH_EMBEDDING_SIZE,
        show_progress: bool = True,
        use_cache: bool = True,
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Show progress bar
            use_cache: Use cached embeddings
            
        Returns:
            List of embedding arrays
        """
        self.logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
        
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache and separate texts that need embedding
        if use_cache and self.cache_enabled:
            for idx, text in enumerate(texts):
                cached = self._get_from_cache(text)
                if cached is not None:
                    embeddings.append(cached)
                    self.stats['cache_hits'] += 1
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(idx)
            
            self.logger.info(f"Cache hits: {len(texts) - len(texts_to_embed)} / {len(texts)}")
        else:
            texts_to_embed = texts
            indices_to_embed = list(range(len(texts)))
        
        # Process remaining texts in batches
        iterator = tqdm(
            range(0, len(texts_to_embed), batch_size),
            desc="Embedding batches",
            disable=not show_progress
        ) if show_progress else range(0, len(texts_to_embed), batch_size)
        
        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, len(texts_to_embed))
            batch_texts = texts_to_embed[batch_start:batch_end]
            
            # Generate embeddings for batch
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Cache and store
            for text, embedding in zip(batch_texts, batch_embeddings):
                if use_cache and self.cache_enabled:
                    self._save_to_cache(text, embedding)
                embeddings.append(embedding)
                self.stats['cache_misses'] += 1
            
            self.stats['batch_operations'] += 1
        
        # Reorder embeddings to match original order
        if use_cache and self.cache_enabled:
            # Create a mapping of indices
            ordered_embeddings = [None] * len(texts)
            embedding_idx = 0
            
            for idx in range(len(texts)):
                if self._get_from_cache(texts[idx]) is not None:
                    # Find from cache
                    ordered_embeddings[idx] = self._get_from_cache(texts[idx])
                else:
                    # From batch processing
                    ordered_embeddings[idx] = embeddings[embedding_idx]
                    embedding_idx += 1
            
            embeddings = ordered_embeddings
        
        self.stats['total_embeddings'] += len(texts)
        self.logger.info(f"Generated {len(embeddings)} embeddings. Stats: {self.stats}")
        
        return embeddings
    
    # ============= EMBEDDING SIMILARITY =============
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        # Normalize embeddings
        norm1 = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
        norm2 = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
        
        # Cosine similarity
        return float(np.dot(norm1, norm2))
    
    def similarity_batch(
        self,
        query_embedding: np.ndarray,
        embeddings: List[np.ndarray],
        top_k: Optional[int] = None,
    ) -> List[tuple]:
        """
        Find most similar embeddings to a query.
        
        Args:
            query_embedding: Query embedding
            embeddings: List of embeddings to compare
            top_k: Return top K results (None = all)
            
        Returns:
            List of (index, similarity_score) tuples, sorted by score
        """
        similarities = []
        
        for idx, embedding in enumerate(embeddings):
            score = self.similarity(query_embedding, embedding)
            similarities.append((idx, score))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            similarities = similarities[:top_k]
        
        return similarities
    
    # ============= CACHING =============
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache if available."""
        if not self.cache_enabled:
            return None
        
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Error reading cache: {e}")
        
        return None
    
    def _save_to_cache(self, text: str, embedding: np.ndarray) -> None:
        """Save embedding to cache."""
        if not self.cache_enabled:
            return
        
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            self.logger.warning(f"Error writing cache: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        if not self.cache_enabled:
            return
        
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self.logger.info("Cache cleared")
        except Exception as e:
            self.logger.warning(f"Error clearing cache: {e}")
    
    # ============= STATISTICS & INFO =============
    
    def get_stats(self) -> Dict:
        """Get embedding statistics."""
        return {
            **self.stats,
            'model': self.model_name,
            'dimension': EMBEDDING_DIMENSION,
            'cache_enabled': self.cache_enabled,
        }
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def validate_embeddings(self, embeddings: List[np.ndarray]) -> bool:
        """
        Validate that all embeddings have correct dimension.
        
        Args:
            embeddings: List of embeddings
            
        Returns:
            True if all valid, False otherwise
        """
        if not embeddings:
            return True
        
        expected_dim = EMBEDDING_DIMENSION
        
        for embedding in embeddings:
            if embedding.shape[0] != expected_dim:
                self.logger.error(
                    f"Invalid embedding dimension: {embedding.shape[0]} "
                    f"(expected {expected_dim})"
                )
                return False
        
        return True
    
    def print_stats(self) -> None:
        """Print statistics to logger."""
        stats = self.get_stats()
        self.logger.info("Embedding Statistics:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")


def main():
    """Example usage."""
    generator = EmbeddingGenerator()
    
    # Single embedding
    text1 = "How do I reset my password?"
    embedding1 = generator.embed_text(text1)
    print(f"Text: {text1}")
    print(f"Embedding shape: {embedding1.shape}")
    print(f"First 5 values: {embedding1[:5]}")
    
    # Batch embeddings
    texts = [
        "How do I reset my password?",
        "I was charged twice for my subscription",
        "API timeout errors when calling endpoints",
        "How do I enable two-factor authentication?",
        "I need a refund for my purchase",
    ]
    
    embeddings = generator.embed_batch(texts)
    print(f"\n✅ Generated {len(embeddings)} embeddings")
    
    # Similarity
    print(f"\nSimilarity between first two texts:")
    similarity = generator.similarity(embeddings[0], embeddings[1])
    print(f"  Score: {similarity:.4f}")
    
    # Top-K similarity
    print(f"\nTop 3 similar to first text:")
    top_k = generator.similarity_batch(embeddings[0], embeddings, top_k=3)
    for idx, score in top_k:
        print(f"  - {texts[idx]}: {score:.4f}")
    
    # Stats
    print(f"\n📊 Stats:")
    generator.print_stats()


if __name__ == "__main__":
    main()
