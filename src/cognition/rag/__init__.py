"""
RAG (Retrieval-Augmented Generation) Pipeline for Supply Chain Cognitive Digital Twin

This module provides a complete RAG pipeline for ingesting, chunking, embedding,
storing, and retrieving supply chain documents to augment LLM decision-making.

Components:
- ingestion: Document ingestion from URLs, PDFs, RSS feeds, and text files
- chunker: Smart chunking strategies for different document types
- embeddings: Text embedding models for vectorization
- vector_store: ChromaDB-based vector storage with semantic search
- retrieval: Query routing, hybrid retrieval, and reranking strategies

Usage:
    from src.cognition.rag import (
        DocumentIngester,
        SupplyChainChunker,
        ChromaVectorStore,
        SupplyChainRetriever,
        DocumentType,
    )
    
    # Ingest documents
    ingester = DocumentIngester()
    doc = ingester.ingest_url("https://example.com/article", DocumentType.NEWS)
    
    # Chunk documents
    chunker = SupplyChainChunker()
    chunks = chunker.chunk_document(doc)
    
    # Store in vector database
    vector_store = ChromaVectorStore(persist_directory="./data/vectorstore")
    vector_store.add_document(doc, chunks)
    
    # Retrieve with intelligent routing and reranking
    retriever = SupplyChainRetriever(vector_store)
    results = retriever.retrieve("semiconductor shortage impact", n_results=5)
    
    # Get formatted context for LLM
    context = retriever.get_context_for_llm("how to handle stockout")
"""

from .chunker import (Chunk, ChunkingStrategy, EventBasedChunker,
                      FixedSizeChunker, ParagraphChunker, RecursiveChunker,
                      SectionChunker, SupplyChainChunker, chunk_document,
                      chunk_text)
from .embeddings import (CachedEmbeddings, EmbeddingModel,
                         HuggingFaceEmbeddings, MockEmbeddings,
                         OpenAIEmbeddings, SentenceTransformerEmbeddings,
                         SupplyChainEmbeddings, create_embeddings,
                         get_default_embeddings)
from .ingestion import (DisruptionType, DocumentIngester, DocumentMetadata,
                        DocumentType, EntityExtractor, Region,
                        SupplyChainDocument, ingest_from_pdf, ingest_from_text,
                        ingest_from_url)
from .retrieval import (BM25Retriever, CombinedReranker, DiversityReranker,
                        HybridRetriever, QueryAnalysis, QueryIntent,
                        QueryRouter, RecencyReranker, RerankerStrategy,
                        RetrievalConfig, SeverityReranker,
                        SupplyChainRetriever, create_retriever,
                        get_default_retriever)
from .vector_store import (ChromaVectorStore, CollectionType,
                           InMemoryVectorStore, MetadataFilter, SearchResult,
                           SearchResults, create_vector_store,
                           get_default_vector_store)

__all__ = [
    # Ingestion
    "DocumentType",
    "DisruptionType",
    "Region",
    "DocumentMetadata",
    "SupplyChainDocument",
    "DocumentIngester",
    "EntityExtractor",
    "ingest_from_url",
    "ingest_from_pdf",
    "ingest_from_text",
    # Chunking
    "Chunk",
    "ChunkingStrategy",
    "FixedSizeChunker",
    "ParagraphChunker",
    "SectionChunker",
    "EventBasedChunker",
    "SupplyChainChunker",
    "RecursiveChunker",
    "chunk_document",
    "chunk_text",
    # Embeddings
    "EmbeddingModel",
    "SentenceTransformerEmbeddings",
    "HuggingFaceEmbeddings",
    "OpenAIEmbeddings",
    "MockEmbeddings",
    "CachedEmbeddings",
    "SupplyChainEmbeddings",
    "create_embeddings",
    "get_default_embeddings",
    # Vector Store
    "CollectionType",
    "SearchResult",
    "SearchResults",
    "MetadataFilter",
    "ChromaVectorStore",
    "InMemoryVectorStore",
    "create_vector_store",
    "get_default_vector_store",
    # Retrieval
    "QueryIntent",
    "QueryAnalysis",
    "QueryRouter",
    "RerankerStrategy",
    "RecencyReranker",
    "SeverityReranker",
    "DiversityReranker",
    "CombinedReranker",
    "BM25Retriever",
    "HybridRetriever",
    "RetrievalConfig",
    "SupplyChainRetriever",
    "create_retriever",
    "get_default_retriever",
]
