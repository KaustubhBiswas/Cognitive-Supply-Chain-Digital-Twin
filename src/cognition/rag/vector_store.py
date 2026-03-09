"""
Vector Store Module for Supply Chain RAG Pipeline

Implements ChromaDB-based vector storage with specialized collections
for different supply chain document types.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .chunker import Chunk
from .embeddings import EmbeddingModel, MockEmbeddings
from .ingestion import (DisruptionType, DocumentType, Region,
                        SupplyChainDocument)

logger = logging.getLogger(__name__)


class CollectionType(Enum):
    """Pre-defined collections for supply chain documents."""
    DISRUPTIONS = "sc_disruptions"
    BEST_PRACTICES = "sc_best_practices"
    CASE_STUDIES = "sc_case_studies"
    REGULATIONS = "sc_regulations"
    NEWS = "sc_news"
    RESEARCH = "sc_research"
    GENERAL = "sc_general"


# Mapping from DocumentType to CollectionType
DOCUMENT_TO_COLLECTION: Dict[DocumentType, CollectionType] = {
    DocumentType.NEWS: CollectionType.NEWS,
    DocumentType.REPORT: CollectionType.GENERAL,
    DocumentType.BEST_PRACTICE: CollectionType.BEST_PRACTICES,
    DocumentType.CASE_STUDY: CollectionType.CASE_STUDIES,
    DocumentType.REGULATION: CollectionType.REGULATIONS,
    DocumentType.INTERNAL: CollectionType.GENERAL,
    DocumentType.RESEARCH: CollectionType.RESEARCH,
}


@dataclass
class SearchResult:
    """A single search result from vector store."""
    chunk_id: str
    content: str
    score: float  # Similarity score (higher is better)
    metadata: Dict[str, Any]
    doc_id: str
    collection: str
    
    def __repr__(self) -> str:
        return (
            f"SearchResult(score={self.score:.4f}, "
            f"doc_id='{self.doc_id}', collection='{self.collection}')"
        )


@dataclass
class SearchResults:
    """Collection of search results with metadata."""
    results: List[SearchResult]
    query: str
    collections_searched: List[str]
    total_results: int
    search_time_ms: float
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __iter__(self):
        return iter(self.results)
    
    def top(self, n: int = 5) -> List[SearchResult]:
        """Get top N results."""
        return self.results[:n]
    
    def by_collection(self, collection: str) -> List[SearchResult]:
        """Filter results by collection."""
        return [r for r in self.results if r.collection == collection]
    
    def above_threshold(self, threshold: float) -> List[SearchResult]:
        """Filter results above similarity threshold."""
        return [r for r in self.results if r.score >= threshold]


class MetadataFilter:
    """Build metadata filters for ChromaDB queries."""
    
    def __init__(self):
        self._filters: Dict[str, Any] = {}
        self._where_clauses: List[Dict] = []
    
    def disruption_type(self, dtype: Union[DisruptionType, str]) -> "MetadataFilter":
        """Filter by disruption type."""
        value = dtype.value if isinstance(dtype, DisruptionType) else dtype
        self._where_clauses.append({"disruption_type": {"$eq": value}})
        return self
    
    def region(self, region: Union[Region, str]) -> "MetadataFilter":
        """Filter by region."""
        value = region.value if isinstance(region, Region) else region
        self._where_clauses.append({"region": {"$eq": value}})
        return self
    
    def doc_type(self, doc_type: Union[DocumentType, str]) -> "MetadataFilter":
        """Filter by document type."""
        value = doc_type.value if isinstance(doc_type, DocumentType) else doc_type
        self._where_clauses.append({"doc_type": {"$eq": value}})
        return self
    
    def severity(self, severity: str) -> "MetadataFilter":
        """Filter by severity level."""
        self._where_clauses.append({"severity": {"$eq": severity}})
        return self
    
    def industry(self, industry: str) -> "MetadataFilter":
        """Filter by industry."""
        self._where_clauses.append({"industry": {"$eq": industry}})
        return self
    
    def since(self, timestamp: datetime) -> "MetadataFilter":
        """Filter documents after timestamp."""
        self._where_clauses.append({"timestamp": {"$gte": timestamp.isoformat()}})
        return self
    
    def before(self, timestamp: datetime) -> "MetadataFilter":
        """Filter documents before timestamp."""
        self._where_clauses.append({"timestamp": {"$lte": timestamp.isoformat()}})
        return self
    
    def contains_entity(self, entity: str) -> "MetadataFilter":
        """Filter documents containing a specific entity."""
        self._where_clauses.append({"entities": {"$contains": entity}})
        return self
    
    def custom(self, field: str, operator: str, value: Any) -> "MetadataFilter":
        """Add a custom filter clause."""
        self._where_clauses.append({field: {operator: value}})
        return self
    
    def build(self) -> Optional[Dict[str, Any]]:
        """Build the ChromaDB where clause."""
        if not self._where_clauses:
            return None
        if len(self._where_clauses) == 1:
            return self._where_clauses[0]
        return {"$and": self._where_clauses}


class ChromaVectorStore:
    """
    ChromaDB-based vector store for supply chain documents.
    
    Provides:
    - Multiple collections for different document types
    - Metadata filtering
    - Similarity search with configurable parameters
    - Batch operations for efficient ingestion
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        collection_prefix: str = "",
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data. 
                              If None, uses in-memory storage.
            embedding_model: Embedding model to use. If None, uses MockEmbeddings.
            collection_prefix: Prefix for collection names (useful for testing).
        """
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model or MockEmbeddings()
        self.collection_prefix = collection_prefix
        self._client = None
        self._collections: Dict[str, Any] = {}
        
    @property
    def client(self):
        """Lazy-load ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                if self.persist_directory:
                    # Persistent storage
                    Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
                    self._client = chromadb.PersistentClient(
                        path=self.persist_directory,
                        settings=Settings(anonymized_telemetry=False),
                    )
                    logger.info(f"ChromaDB initialized with persistence at {self.persist_directory}")
                else:
                    # In-memory storage
                    self._client = chromadb.Client(
                        settings=Settings(anonymized_telemetry=False),
                    )
                    logger.info("ChromaDB initialized with in-memory storage")
                    
            except ImportError:
                raise ImportError(
                    "chromadb is required. Install with: pip install chromadb"
                )
        return self._client
    
    def _get_collection_name(self, collection_type: CollectionType) -> str:
        """Get full collection name with prefix."""
        return f"{self.collection_prefix}{collection_type.value}"
    
    def get_or_create_collection(self, collection_type: CollectionType):
        """Get or create a collection by type."""
        name = self._get_collection_name(collection_type)
        
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={
                    "description": f"Supply chain {collection_type.name} documents",
                    "created_at": datetime.now().isoformat(),
                    "hnsw:space": "cosine",  # Use cosine similarity
                }
            )
            logger.info(f"Collection '{name}' ready")
            
        return self._collections[name]
    
    def get_collection_for_document(self, doc: SupplyChainDocument):
        """Get appropriate collection for a document based on its type."""
        collection_type = DOCUMENT_TO_COLLECTION.get(
            doc.metadata.doc_type, CollectionType.GENERAL
        )
        return self.get_or_create_collection(collection_type)
    
    def _prepare_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """Prepare chunk metadata for ChromaDB storage."""
        # ChromaDB only supports str, int, float, bool in metadata
        metadata = {}
        
        for key, value in chunk.metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                metadata[key] = ",".join(str(v) for v in value)
            elif isinstance(value, datetime):
                metadata[key] = value.isoformat()
            elif isinstance(value, Enum):
                metadata[key] = value.value
            else:
                metadata[key] = str(value)
        
        # Add chunk-specific metadata
        metadata["doc_id"] = chunk.doc_id
        metadata["chunk_index"] = chunk.chunk_index
        metadata["total_chunks"] = chunk.total_chunks
        
        return metadata
    
    def add_chunks(
        self,
        chunks: List[Chunk],
        collection_type: Optional[CollectionType] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Add chunks to vector store.
        
        Args:
            chunks: List of chunks to add
            collection_type: Target collection. If None, infers from chunk metadata.
            batch_size: Batch size for embedding and insertion
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Determine target collection
        if collection_type:
            collection = self.get_or_create_collection(collection_type)
        else:
            # Try to infer from first chunk's doc_type
            doc_type_str = chunks[0].metadata.get("doc_type")
            if doc_type_str:
                doc_type = DocumentType(doc_type_str)
                collection_type = DOCUMENT_TO_COLLECTION.get(doc_type, CollectionType.GENERAL)
            else:
                collection_type = CollectionType.GENERAL
            collection = self.get_or_create_collection(collection_type)
        
        total_added = 0
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            ids = [chunk.id for chunk in batch]
            contents = [chunk.content for chunk in batch]
            metadatas = [self._prepare_metadata(chunk) for chunk in batch]
            
            # Generate embeddings
            embeddings = self.embedding_model.embed_texts(contents)
            
            # Add to collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
            )
            
            total_added += len(batch)
            logger.debug(f"Added batch of {len(batch)} chunks to {collection_type.value}")
        
        logger.info(f"Added {total_added} chunks to collection '{collection_type.value}'")
        return total_added
    
    def add_document(
        self,
        doc: SupplyChainDocument,
        chunks: List[Chunk],
    ) -> int:
        """
        Add a document with its chunks to the appropriate collection.
        
        Args:
            doc: The source document
            chunks: Pre-chunked document chunks
            
        Returns:
            Number of chunks added
        """
        collection_type = DOCUMENT_TO_COLLECTION.get(
            doc.metadata.doc_type, CollectionType.GENERAL
        )
        return self.add_chunks(chunks, collection_type)
    
    def search(
        self,
        query: str,
        collections: Optional[List[CollectionType]] = None,
        n_results: int = 10,
        filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> SearchResults:
        """
        Search for similar documents across collections.
        
        Args:
            query: Search query text
            collections: Collections to search. If None, searches all.
            n_results: Maximum results per collection
            filter: Metadata filter to apply
            include_embeddings: Whether to include embeddings in results
            
        Returns:
            SearchResults object with ranked results
        """
        import time
        start_time = time.time()
        
        # Embed query
        query_embedding = self.embedding_model.embed_text(query)
        
        # Determine collections to search
        if collections is None:
            collections = list(CollectionType)
        
        all_results = []
        collections_searched = []
        
        # Build where clause
        where_clause = filter.build() if filter else None
        
        # Search each collection
        for collection_type in collections:
            collection_name = self._get_collection_name(collection_type)
            
            # Skip if collection doesn't exist
            if collection_name not in self._collections:
                try:
                    collection = self.client.get_collection(collection_name)
                    self._collections[collection_name] = collection
                except Exception:
                    continue
            
            collection = self._collections[collection_name]
            collections_searched.append(collection_type.value)
            
            try:
                # Perform search
                include_list = ["documents", "metadatas", "distances"]
                if include_embeddings:
                    include_list.append("embeddings")
                
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_clause,
                    include=include_list,
                )
                
                # Process results
                if results["ids"] and results["ids"][0]:
                    for idx, chunk_id in enumerate(results["ids"][0]):
                        # Convert distance to similarity score
                        # ChromaDB returns L2 distance, convert to similarity
                        distance = results["distances"][0][idx]
                        score = 1 / (1 + distance)  # Convert to 0-1 similarity
                        
                        metadata = results["metadatas"][0][idx] if results["metadatas"] else {}
                        
                        all_results.append(SearchResult(
                            chunk_id=chunk_id,
                            content=results["documents"][0][idx],
                            score=score,
                            metadata=metadata,
                            doc_id=metadata.get("doc_id", ""),
                            collection=collection_type.value,
                        ))
                        
            except Exception as e:
                logger.warning(f"Error searching collection {collection_name}: {e}")
        
        # Sort by score (highest first)
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        search_time = (time.time() - start_time) * 1000
        
        return SearchResults(
            results=all_results[:n_results],
            query=query,
            collections_searched=collections_searched,
            total_results=len(all_results),
            search_time_ms=search_time,
        )
    
    def search_by_disruption(
        self,
        query: str,
        disruption_type: DisruptionType,
        n_results: int = 10,
        include_best_practices: bool = True,
    ) -> SearchResults:
        """
        Search for documents related to a specific disruption type.
        
        Args:
            query: Search query
            disruption_type: Type of disruption to filter for
            n_results: Maximum results
            include_best_practices: Whether to include best practices collection
            
        Returns:
            Relevant search results
        """
        filter = MetadataFilter().disruption_type(disruption_type)
        
        collections = [CollectionType.DISRUPTIONS, CollectionType.NEWS]
        if include_best_practices:
            collections.append(CollectionType.BEST_PRACTICES)
        
        return self.search(
            query=query,
            collections=collections,
            n_results=n_results,
            filter=filter,
        )
    
    def search_best_practices(
        self,
        query: str,
        region: Optional[Region] = None,
        industry: Optional[str] = None,
        n_results: int = 5,
    ) -> SearchResults:
        """
        Search for best practices with optional filters.
        
        Args:
            query: Search query
            region: Optional region filter
            industry: Optional industry filter
            n_results: Maximum results
            
        Returns:
            Best practice search results
        """
        filter = MetadataFilter()
        if region:
            filter.region(region)
        if industry:
            filter.industry(industry)
        
        return self.search(
            query=query,
            collections=[CollectionType.BEST_PRACTICES],
            n_results=n_results,
            filter=filter if (region or industry) else None,
        )
    
    def search_recent(
        self,
        query: str,
        days: int = 7,
        n_results: int = 10,
    ) -> SearchResults:
        """
        Search for recent documents.
        
        Args:
            query: Search query
            days: Number of days to look back
            n_results: Maximum results
            
        Returns:
            Recent search results
        """
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        filter = MetadataFilter().since(cutoff)
        
        return self.search(
            query=query,
            collections=[CollectionType.NEWS, CollectionType.DISRUPTIONS],
            n_results=n_results,
            filter=filter,
        )
    
    def get_document_chunks(
        self,
        doc_id: str,
        collection_type: Optional[CollectionType] = None,
    ) -> List[SearchResult]:
        """
        Retrieve all chunks for a specific document.
        
        Args:
            doc_id: Document ID to retrieve
            collection_type: Collection to search. If None, searches all.
            
        Returns:
            List of chunks for the document
        """
        collections = [collection_type] if collection_type else list(CollectionType)
        
        results = []
        for ctype in collections:
            collection_name = self._get_collection_name(ctype)
            
            try:
                collection = self.client.get_collection(collection_name)
                
                query_result = collection.get(
                    where={"doc_id": {"$eq": doc_id}},
                    include=["documents", "metadatas"],
                )
                
                if query_result["ids"]:
                    for idx, chunk_id in enumerate(query_result["ids"]):
                        metadata = query_result["metadatas"][idx] if query_result["metadatas"] else {}
                        results.append(SearchResult(
                            chunk_id=chunk_id,
                            content=query_result["documents"][idx],
                            score=1.0,  # Perfect match for retrieval by ID
                            metadata=metadata,
                            doc_id=doc_id,
                            collection=ctype.value,
                        ))
                        
            except Exception:
                continue
        
        # Sort by chunk index
        results.sort(key=lambda x: x.metadata.get("chunk_index", 0))
        return results
    
    def delete_document(
        self,
        doc_id: str,
        collection_type: Optional[CollectionType] = None,
    ) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            doc_id: Document ID to delete
            collection_type: Collection to delete from. If None, deletes from all.
            
        Returns:
            Number of chunks deleted
        """
        collections = [collection_type] if collection_type else list(CollectionType)
        
        total_deleted = 0
        for ctype in collections:
            collection_name = self._get_collection_name(ctype)
            
            try:
                collection = self.client.get_collection(collection_name)
                
                # Get IDs to delete
                query_result = collection.get(
                    where={"doc_id": {"$eq": doc_id}},
                )
                
                if query_result["ids"]:
                    collection.delete(ids=query_result["ids"])
                    total_deleted += len(query_result["ids"])
                    logger.info(f"Deleted {len(query_result['ids'])} chunks from {collection_name}")
                    
            except Exception as e:
                logger.warning(f"Error deleting from {collection_name}: {e}")
        
        return total_deleted
    
    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all collections."""
        stats = {}
        
        for collection_type in CollectionType:
            collection_name = self._get_collection_name(collection_type)
            
            try:
                collection = self.client.get_collection(collection_name)
                count = collection.count()
                
                stats[collection_type.value] = {
                    "count": count,
                    "collection_name": collection_name,
                }
                
            except Exception:
                stats[collection_type.value] = {
                    "count": 0,
                    "collection_name": collection_name,
                    "exists": False,
                }
        
        return stats
    
    def clear_collection(self, collection_type: CollectionType) -> bool:
        """
        Clear all documents from a collection.
        
        Args:
            collection_type: Collection to clear
            
        Returns:
            True if successful
        """
        collection_name = self._get_collection_name(collection_type)
        
        try:
            self.client.delete_collection(collection_name)
            if collection_name in self._collections:
                del self._collections[collection_name]
            logger.info(f"Cleared collection '{collection_name}'")
            return True
        except Exception as e:
            logger.warning(f"Error clearing collection {collection_name}: {e}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all collections."""
        success = True
        for collection_type in CollectionType:
            if not self.clear_collection(collection_type):
                success = False
        return success


class InMemoryVectorStore:
    """
    In-memory vector store for testing or small deployments.
    
    Does not require ChromaDB but provides similar functionality
    using numpy for similarity calculations.
    """
    
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        """
        Initialize in-memory vector store.
        
        Args:
            embedding_model: Embedding model to use
        """
        self.embedding_model = embedding_model or MockEmbeddings()
        self._chunks: Dict[str, Chunk] = {}  # chunk_id -> chunk
        self._embeddings: Dict[str, List[float]] = {}  # chunk_id -> embedding
        self._collections: Dict[str, Set[str]] = {}  # collection -> set of chunk_ids
    
    def add_chunks(
        self,
        chunks: List[Chunk],
        collection: str = "default",
    ) -> int:
        """Add chunks to vector store."""
        import numpy as np
        
        if not chunks:
            return 0
        
        # Initialize collection
        if collection not in self._collections:
            self._collections[collection] = set()
        
        # Generate embeddings
        contents = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(contents)
        
        # Store chunks and embeddings
        for chunk, embedding in zip(chunks, embeddings):
            self._chunks[chunk.id] = chunk
            self._embeddings[chunk.id] = embedding
            self._collections[collection].add(chunk.id)
        
        return len(chunks)
    
    def search(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        n_results: int = 10,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks.
        
        Returns:
            List of (chunk, similarity_score) tuples
        """
        import numpy as np
        
        if not self._chunks:
            return []
        
        # Get query embedding
        query_embedding = np.array(self.embedding_model.embed_text(query))
        
        # Filter by collections
        if collections:
            valid_ids = set()
            for col in collections:
                valid_ids.update(self._collections.get(col, set()))
        else:
            valid_ids = set(self._chunks.keys())
        
        # Calculate similarities
        results = []
        for chunk_id in valid_ids:
            if chunk_id not in self._embeddings:
                continue
                
            chunk_embedding = np.array(self._embeddings[chunk_id])
            
            # Cosine similarity
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding) + 1e-8
            )
            
            results.append((self._chunks[chunk_id], float(similarity)))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:n_results]
    
    def count(self, collection: Optional[str] = None) -> int:
        """Count chunks in store."""
        if collection:
            return len(self._collections.get(collection, set()))
        return len(self._chunks)
    
    def clear(self):
        """Clear all data."""
        self._chunks.clear()
        self._embeddings.clear()
        self._collections.clear()


# Convenience functions
def create_vector_store(
    persist_directory: Optional[str] = None,
    embedding_model: Optional[EmbeddingModel] = None,
    use_chromadb: bool = True,
) -> Union[ChromaVectorStore, InMemoryVectorStore]:
    """
    Create a vector store instance.
    
    Args:
        persist_directory: Directory for persistence (ChromaDB only)
        embedding_model: Embedding model to use
        use_chromadb: Whether to use ChromaDB (requires installation)
        
    Returns:
        Vector store instance
    """
    if use_chromadb:
        return ChromaVectorStore(
            persist_directory=persist_directory,
            embedding_model=embedding_model,
        )
    else:
        return InMemoryVectorStore(embedding_model=embedding_model)


def get_default_vector_store(
    persist_directory: str = "./data/vectorstore",
) -> ChromaVectorStore:
    """
    Get a default vector store with standard configuration.
    
    Args:
        persist_directory: Where to persist data
        
    Returns:
        Configured ChromaVectorStore instance
    """
    from .embeddings import get_default_embeddings
    
    return ChromaVectorStore(
        persist_directory=persist_directory,
        embedding_model=get_default_embeddings(),
    )
