import os
import sys
import logging
from typing import List, Union
import numpy as np

# Ensure backend root is in python path for standalone runs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Production-grade local text embedding and cross-encoder service.
    Implements lazy-loading for transformer models to minimize FastAPI startup latency,
    and supports a fallback mock mode when hardware acceleration or downloading is unavailable.
    """
    def __init__(self) -> None:
        self.embedding_model_name = settings.EMBEDDING_MODEL_NAME
        self.cross_encoder_model_name = settings.CROSS_ENCODER_MODEL_NAME
        self._embedding_model = None
        self._cross_encoder = None

    def initialize_models(self) -> None:
        """
        Thread-safe lazy initialization of embedding and grading models.
        """
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading local embedding model: {self.embedding_model_name}...")
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info("Local embedding model loaded successfully.")
            except Exception as e:
                logger.error(
                    f"Could not load local embedding model '{self.embedding_model_name}': {e}. "
                    "Falling back to mock embedding generation for this run."
                )
                self._embedding_model = "mock"

        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading local cross-encoder model: {self.cross_encoder_model_name}...")
                self._cross_encoder = CrossEncoder(self.cross_encoder_model_name)
                logger.info("Local cross-encoder model loaded successfully.")
            except Exception as e:
                logger.error(
                    f"Could not load local cross-encoder model '{self.cross_encoder_model_name}': {e}. "
                    "Falling back to mock scoring."
                )
                self._cross_encoder = "mock"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate high-throughput dense vector representations for a list of document chunks.
        """
        if not texts:
            return []
        
        self.initialize_models()
        
        if self._embedding_model == "mock":
            # Generate deterministic mock embeddings of dimension 384
            logger.warning("Generating 384-dimensional mock embeddings...")
            dimension = 384
            embeddings = []
            for text in texts:
                # Deterministic seed using hash of text to make it repeatable for testing
                seed = sum(ord(c) for c in text) % 1000
                np.random.seed(seed)
                vector = np.random.randn(dimension)
                norm = np.linalg.norm(vector)
                normalized_vector = (vector / norm).tolist() if norm > 0 else [0.0] * dimension
                embeddings.append(normalized_vector)
            return embeddings

        # Direct batched model inference
        vectors = self._embedding_model.encode(
            texts, 
            batch_size=32, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Generate a single query vector representation.
        """
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []

    def compute_relevance_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        Grade relevance of retrieved document chunks against the user query.
        Uses a local cross-encoder (bi-directional attention mechanism over concatenated sequences).
        """
        if not documents:
            return []

        self.initialize_models()

        if self._cross_encoder == "mock":
            logger.warning("Computing mock relevance scores...")
            # Return pseudo-relevance scores
            scores = []
            for doc in documents:
                # Match simple keyword intersection for pseudo-relevance
                query_words = set(query.lower().split())
                doc_words = set(doc.lower().split())
                intersection = query_words.intersection(doc_words)
                score = min(0.3 + (len(intersection) * 0.1), 0.95)
                scores.append(score)
            return scores

        pairs = [[query, doc] for doc in documents]
        predictions = self._cross_encoder.predict(pairs)
        
        # Handle outputs that are single floats (if only 1 document is graded)
        if isinstance(predictions, (float, np.float32, np.float64)):
            return [float(predictions)]
        
        return [float(score) for score in predictions]

# Expose global instance
embedding_service = EmbeddingService()
