"""
Embeddings Module for Supply Chain RAG Pipeline

Provides embedding model wrappers for text vectorization,
supporting multiple embedding backends.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np

from .chunker import Chunk

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        pass
    
    def embed_chunk(self, chunk: Chunk) -> List[float]:
        """Embed a chunk."""
        return self.embed_text(chunk.content)
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        """Embed multiple chunks."""
        texts = [chunk.content for chunk in chunks]
        return self.embed_texts(texts)


class SentenceTransformerEmbeddings(EmbeddingModel):
    """
    Embedding model using sentence-transformers.
    
    Recommended models:
    - sentence-transformers/all-MiniLM-L6-v2 (fast, good quality)
    - sentence-transformers/all-mpnet-base-v2 (slower, better quality)
    - BAAI/bge-small-en-v1.5 (good for retrieval)
    - BAAI/bge-large-en-v1.5 (best quality, slower)
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
    ):
        """
        Initialize sentence-transformer embeddings.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cpu' or 'cuda')
            normalize: Whether to normalize embeddings
        """
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None
        self._dimension = None
    
    @property
    def model(self):
        """Lazy-load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Loaded model with dimension {self._dimension}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension is None:
            _ = self.model  # Load model to get dimension
        return self._dimension
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 10,
            batch_size=32,
        )
        return embeddings.tolist()


class HuggingFaceEmbeddings(EmbeddingModel):
    """
    Embedding model using HuggingFace transformers directly.
    
    Alternative to sentence-transformers for more control.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._tokenizer = None
        self._model = None
        self._dimension = None
    
    def _load_model(self):
        """Load model and tokenizer."""
        if self._model is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
                
                logger.info(f"Loading HuggingFace model: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.to(self.device)
                self._model.eval()
                
                # Get dimension from config
                self._dimension = self._model.config.hidden_size
                logger.info(f"Loaded model with dimension {self._dimension}")
                
            except ImportError:
                raise ImportError(
                    "transformers and torch are required. "
                    "Install with: pip install transformers torch"
                )
    
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension
    
    def _mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on token embeddings."""
        import torch
        
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        import torch
        
        self._load_model()
        
        # Tokenize
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self._model(**encoded)
        
        # Mean pooling
        embeddings = self._mean_pooling(outputs, encoded["attention_mask"])
        
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().tolist()


class OpenAIEmbeddings(EmbeddingModel):
    """
    Embedding model using OpenAI API.
    
    Requires OPENAI_API_KEY environment variable or api_key parameter.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key
        self._client = None
        
        # Embedding dimensions by model
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
    
    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                import os

                from openai import OpenAI
                
                api_key = self._api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key required")
                
                self._client = OpenAI(api_key=api_key)
                
            except ImportError:
                raise ImportError(
                    "openai is required. Install with: pip install openai"
                )
        return self._client
    
    @property
    def dimension(self) -> int:
        return self._dimensions.get(self.model, 1536)
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]


class MockEmbeddings(EmbeddingModel):
    """
    Mock embedding model for testing.
    
    Generates random embeddings of specified dimension.
    """
    
    def __init__(self, dimension: int = 384, seed: int = 42):
        self._dimension = dimension
        self._seed = seed
        self._rng = np.random.default_rng(seed)
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed_text(self, text: str) -> List[float]:
        """Generate a mock embedding."""
        # Use text hash for deterministic embedding
        text_hash = hash(text) % (2**32)
        rng = np.random.default_rng(text_hash)
        embedding = rng.random(self._dimension).astype(np.float32)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for multiple texts."""
        return [self.embed_text(text) for text in texts]


class CachedEmbeddings(EmbeddingModel):
    """
    Wrapper that adds caching to any embedding model.
    
    Uses an in-memory cache to avoid re-embedding duplicate texts.
    """
    
    def __init__(self, base_model: EmbeddingModel, max_cache_size: int = 10000):
        self.base_model = base_model
        self.max_cache_size = max_cache_size
        self._cache = {}
    
    @property
    def dimension(self) -> int:
        return self.base_model.dimension
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed_text(self, text: str) -> List[float]:
        """Embed with caching."""
        key = self._get_cache_key(text)
        
        if key in self._cache:
            return self._cache[key]
        
        embedding = self.base_model.embed_text(text)
        
        # Add to cache (with simple eviction if needed)
        if len(self._cache) >= self.max_cache_size:
            # Remove oldest 10%
            keys_to_remove = list(self._cache.keys())[:self.max_cache_size // 10]
            for k in keys_to_remove:
                del self._cache[k]
        
        self._cache[key] = embedding
        return embedding
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts with caching."""
        results = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            key = self._get_cache_key(text)
            if key in self._cache:
                results.append(self._cache[key])
            else:
                results.append(None)  # Placeholder
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Embed uncached texts
        if uncached_texts:
            new_embeddings = self.base_model.embed_texts(uncached_texts)
            
            for idx, embedding in zip(uncached_indices, new_embeddings):
                results[idx] = embedding
                key = self._get_cache_key(texts[idx])
                
                if len(self._cache) >= self.max_cache_size:
                    keys_to_remove = list(self._cache.keys())[:self.max_cache_size // 10]
                    for k in keys_to_remove:
                        del self._cache[k]
                
                self._cache[key] = embedding
        
        return results
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()


class SupplyChainEmbeddings:
    """
    Main embedding interface for the Supply Chain RAG pipeline.
    
    Provides a unified interface with automatic model selection
    and optional caching.
    """
    
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __init__(
        self,
        model_name: str = None,
        provider: str = "sentence-transformers",
        device: str = "cpu",
        use_cache: bool = True,
        cache_size: int = 10000,
        api_key: Optional[str] = None,
    ):
        """
        Initialize embeddings.
        
        Args:
            model_name: Model name/path
            provider: Provider type ('sentence-transformers', 'huggingface', 'openai', 'mock')
            device: Device for local models ('cpu' or 'cuda')
            use_cache: Whether to cache embeddings
            cache_size: Maximum cache size
            api_key: API key for cloud providers
        """
        model_name = model_name or self.DEFAULT_MODEL
        
        # Create base model
        if provider == "sentence-transformers":
            base_model = SentenceTransformerEmbeddings(
                model_name=model_name,
                device=device,
            )
        elif provider == "huggingface":
            base_model = HuggingFaceEmbeddings(
                model_name=model_name,
                device=device,
            )
        elif provider == "openai":
            base_model = OpenAIEmbeddings(
                model=model_name,
                api_key=api_key,
            )
        elif provider == "mock":
            base_model = MockEmbeddings()
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Wrap with cache if requested
        if use_cache:
            self.model = CachedEmbeddings(base_model, max_cache_size=cache_size)
        else:
            self.model = base_model
        
        self.provider = provider
        self.model_name = model_name
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.model.dimension
    
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Embed text(s).
        
        Args:
            text: Single text or list of texts
            
        Returns:
            Single embedding or list of embeddings
        """
        if isinstance(text, str):
            return self.model.embed_text(text)
        return self.model.embed_texts(text)
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        """Embed chunks."""
        return self.model.embed_chunks(chunks)


# =============================================================================
# Factory Functions
# =============================================================================

def create_embeddings(
    provider: str = "sentence-transformers",
    model_name: str = None,
    **kwargs,
) -> SupplyChainEmbeddings:
    """
    Create an embeddings instance.
    
    Args:
        provider: Provider type
        model_name: Model name
        **kwargs: Additional arguments
        
    Returns:
        SupplyChainEmbeddings instance
    """
    return SupplyChainEmbeddings(
        model_name=model_name,
        provider=provider,
        **kwargs,
    )


def get_default_embeddings() -> SupplyChainEmbeddings:
    """Get default embeddings instance (singleton pattern)."""
    if not hasattr(get_default_embeddings, "_instance"):
        get_default_embeddings._instance = create_embeddings()
    return get_default_embeddings._instance
