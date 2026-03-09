"""
Retrieval Strategy Module for Supply Chain RAG Pipeline

Implements intelligent retrieval with:
- Context-aware query routing to appropriate collections
- Reranking strategies (recency, relevance, combined)
- Hybrid retrieval (dense embeddings + sparse BM25)
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .chunker import Chunk
from .embeddings import EmbeddingModel, MockEmbeddings
from .ingestion import DisruptionType, DocumentType, Region
from .vector_store import (ChromaVectorStore, CollectionType,
                           InMemoryVectorStore, MetadataFilter, SearchResult,
                           SearchResults)

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Detected intent of a user query."""
    DISRUPTION_INFO = "disruption_info"
    BEST_PRACTICE = "best_practice"
    CASE_STUDY = "case_study"
    REGULATION = "regulation"
    RECENT_NEWS = "recent_news"
    RESEARCH = "research"
    GENERAL = "general"
    MULTI_INTENT = "multi_intent"


# Intent to collection mapping
INTENT_TO_COLLECTIONS: Dict[QueryIntent, List[CollectionType]] = {
    QueryIntent.DISRUPTION_INFO: [
        CollectionType.DISRUPTIONS,
        CollectionType.NEWS,
    ],
    QueryIntent.BEST_PRACTICE: [
        CollectionType.BEST_PRACTICES,
        CollectionType.CASE_STUDIES,
    ],
    QueryIntent.CASE_STUDY: [
        CollectionType.CASE_STUDIES,
        CollectionType.BEST_PRACTICES,
    ],
    QueryIntent.REGULATION: [
        CollectionType.REGULATIONS,
        CollectionType.GENERAL,
    ],
    QueryIntent.RECENT_NEWS: [
        CollectionType.NEWS,
        CollectionType.DISRUPTIONS,
    ],
    QueryIntent.RESEARCH: [
        CollectionType.RESEARCH,
        CollectionType.GENERAL,
    ],
    QueryIntent.GENERAL: [
        CollectionType.GENERAL,
        CollectionType.NEWS,
        CollectionType.BEST_PRACTICES,
    ],
    QueryIntent.MULTI_INTENT: list(CollectionType),
}


@dataclass
class QueryAnalysis:
    """Analysis of a user query."""
    original_query: str
    normalized_query: str
    intent: QueryIntent
    confidence: float
    detected_entities: List[str]
    detected_disruption_type: Optional[DisruptionType]
    detected_region: Optional[Region]
    is_urgent: bool
    suggested_collections: List[CollectionType]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "intent": self.intent.value,
            "confidence": self.confidence,
            "detected_entities": self.detected_entities,
            "detected_disruption_type": (
                self.detected_disruption_type.value
                if self.detected_disruption_type else None
            ),
            "detected_region": (
                self.detected_region.value if self.detected_region else None
            ),
            "is_urgent": self.is_urgent,
            "suggested_collections": [c.value for c in self.suggested_collections],
        }


class QueryRouter:
    """
    Routes queries to appropriate collections based on intent detection.
    
    Uses keyword patterns and entity recognition to determine
    which collections are most relevant for a given query.
    """
    
    # Intent detection patterns
    INTENT_PATTERNS: Dict[QueryIntent, List[str]] = {
        QueryIntent.DISRUPTION_INFO: [
            r"\bdisrupt(ion|ed|ing)?\b",
            r"\bshortage\b",
            r"\bstockout\b",
            r"\bdelay(s|ed)?\b",
            r"\bcrisis\b",
            r"\bimpact\b",
            r"\bfailure\b",
            r"\bproblem\b",
            r"\brisk\b",
            r"\bthreat\b",
        ],
        QueryIntent.BEST_PRACTICE: [
            r"\bbest practice\b",
            r"\bhow (to|do|should)\b",
            r"\brecommend(ation|ed)?\b",
            r"\bstrateg(y|ies)\b",
            r"\bmitigat(e|ion)\b",
            r"\bprevent(ion)?\b",
            r"\boptimiz(e|ation)\b",
            r"\bimprov(e|ement)\b",
        ],
        QueryIntent.CASE_STUDY: [
            r"\bcase study\b",
            r"\bexample\b",
            r"\blearned\b",
            r"\bexperience\b",
            r"\bcompany\b",
            r"\borganization\b",
            r"\bhow did\b",
            r"\bwhat happened\b",
        ],
        QueryIntent.REGULATION: [
            r"\bregulation\b",
            r"\bcompliance\b",
            r"\blegal\b",
            r"\brequirement\b",
            r"\bstandard\b",
            r"\bpolicy\b",
            r"\blaw\b",
            r"\bmandat(e|ory)\b",
        ],
        QueryIntent.RECENT_NEWS: [
            r"\bnews\b",
            r"\brecent\b",
            r"\blatest\b",
            r"\btoday\b",
            r"\bthis week\b",
            r"\bcurrent\b",
            r"\bupdate\b",
            r"\breport(ed|s)?\b",
        ],
        QueryIntent.RESEARCH: [
            r"\bresearch\b",
            r"\bstudy\b",
            r"\banalysis\b",
            r"\bdata\b",
            r"\bstatistic\b",
            r"\btrend\b",
            r"\bforecast\b",
            r"\bprediction\b",
        ],
    }
    
    # Disruption type patterns
    DISRUPTION_PATTERNS: Dict[DisruptionType, List[str]] = {
        DisruptionType.STOCKOUT: [r"\bstockout\b", r"\bout of stock\b", r"\binventory (shortage|depletion)\b"],
        DisruptionType.DEMAND_SPIKE: [r"\bdemand (spike|surge)\b", r"\bhigh demand\b", r"\bdemand increase\b"],
        DisruptionType.SUPPLIER_FAILURE: [r"\bsupplier (failure|issue|problem)\b", r"\bvendor\b"],
        DisruptionType.LOGISTICS_DELAY: [r"\blogistics\b", r"\bshipping\b", r"\bport\b", r"\btransport\b", r"\bfreight\b"],
        DisruptionType.QUALITY_ISSUE: [r"\bquality\b", r"\bdefect\b", r"\brecall\b"],
        DisruptionType.NATURAL_DISASTER: [r"\bearthquake\b", r"\bflood\b", r"\bhurricane\b", r"\btyphoon\b", r"\bdisaster\b"],
        DisruptionType.GEOPOLITICAL: [r"\btariff\b", r"\bsanction\b", r"\btrade war\b", r"\bpolitical\b", r"\bgeopolitical\b"],
        DisruptionType.CYBER_ATTACK: [r"\bcyber\b", r"\bhack\b", r"\bransomware\b", r"\bsecurity breach\b"],
        DisruptionType.LABOR_SHORTAGE: [r"\blabor (shortage|strike)\b", r"\bworkforce\b", r"\bstrike\b"],
        DisruptionType.RAW_MATERIAL_SHORTAGE: [r"\braw material\b", r"\bsemiconductor\b", r"\bchip\b", r"\bcomponent\b"],
    }
    
    # Region patterns
    REGION_PATTERNS: Dict[Region, List[str]] = {
        Region.ASIA: [r"\basia\b", r"\bchina\b", r"\bjapan\b", r"\bkorea\b", r"\btaiwan\b", r"\bindia\b", r"\bvietnam\b"],
        Region.EUROPE: [r"\beurope\b", r"\bgermany\b", r"\bfrance\b", r"\buk\b", r"\bitaly\b", r"\bspain\b", r"\beu\b"],
        Region.AMERICAS: [r"\bamerica\b", r"\bus\b", r"\busa\b", r"\bcanada\b", r"\bmexico\b", r"\bbrazil\b"],
        Region.AFRICA: [r"\bafrica\b", r"\bsouth africa\b", r"\bnigeria\b", r"\bkenya\b"],
        Region.MIDDLE_EAST: [r"\bmiddle east\b", r"\bsaudi\b", r"\buae\b", r"\bisrael\b", r"\biran\b"],
        Region.OCEANIA: [r"\baustralia\b", r"\bnew zealand\b", r"\boceania\b"],
        Region.GLOBAL: [r"\bglobal\b", r"\bworldwide\b", r"\binternational\b"],
    }
    
    # Urgency indicators
    URGENCY_PATTERNS = [
        r"\burgent\b",
        r"\bimmediately\b",
        r"\bcritical\b",
        r"\bemergency\b",
        r"\basap\b",
        r"\bright now\b",
        r"\btime.?sensitive\b",
    ]
    
    def __init__(self, custom_patterns: Optional[Dict[QueryIntent, List[str]]] = None):
        """
        Initialize query router.
        
        Args:
            custom_patterns: Additional patterns to add to intent detection
        """
        self.intent_patterns = dict(self.INTENT_PATTERNS)
        if custom_patterns:
            for intent, patterns in custom_patterns.items():
                if intent in self.intent_patterns:
                    self.intent_patterns[intent].extend(patterns)
                else:
                    self.intent_patterns[intent] = patterns
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a query to determine routing.
        
        Args:
            query: User query string
            
        Returns:
            QueryAnalysis with detected intent and suggestions
        """
        normalized = query.lower().strip()
        
        # Detect intent
        intent, confidence = self._detect_intent(normalized)
        
        # Detect entities
        entities = self._extract_entities(normalized)
        
        # Detect disruption type
        disruption_type = self._detect_disruption_type(normalized)
        
        # Detect region
        region = self._detect_region(normalized)
        
        # Check urgency
        is_urgent = self._check_urgency(normalized)
        
        # Get suggested collections
        suggested_collections = self._get_suggested_collections(
            intent, disruption_type, region
        )
        
        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            intent=intent,
            confidence=confidence,
            detected_entities=entities,
            detected_disruption_type=disruption_type,
            detected_region=region,
            is_urgent=is_urgent,
            suggested_collections=suggested_collections,
        )
    
    def _detect_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Detect query intent using pattern matching."""
        scores: Dict[QueryIntent, int] = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            if score > 0:
                scores[intent] = score
        
        if not scores:
            return QueryIntent.GENERAL, 0.5
        
        # Check for multi-intent
        if len(scores) > 2:
            return QueryIntent.MULTI_INTENT, 0.7
        
        # Return highest scoring intent
        best_intent = max(scores, key=scores.get)
        max_score = scores[best_intent]
        confidence = min(0.5 + (max_score * 0.15), 0.95)
        
        return best_intent, confidence
    
    def _detect_disruption_type(self, query: str) -> Optional[DisruptionType]:
        """Detect disruption type from query."""
        for dtype, patterns in self.DISRUPTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return dtype
        return None
    
    def _detect_region(self, query: str) -> Optional[Region]:
        """Detect region from query."""
        for region, patterns in self.REGION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return region
        return None
    
    def _check_urgency(self, query: str) -> bool:
        """Check if query indicates urgency."""
        for pattern in self.URGENCY_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entities from query."""
        entities = []
        
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        
        # Extract capitalized words (potential proper nouns)
        words = query.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                clean = re.sub(r'[^\w]', '', word)
                if clean and clean not in entities:
                    entities.append(clean)
        
        return entities
    
    def _get_suggested_collections(
        self,
        intent: QueryIntent,
        disruption_type: Optional[DisruptionType],
        region: Optional[Region],
    ) -> List[CollectionType]:
        """Get collections to search based on analysis."""
        collections = list(INTENT_TO_COLLECTIONS.get(intent, [CollectionType.GENERAL]))
        
        # Add disruptions collection if disruption detected
        if disruption_type and CollectionType.DISRUPTIONS not in collections:
            collections.insert(0, CollectionType.DISRUPTIONS)
        
        return collections
    
    def route(self, query: str) -> List[CollectionType]:
        """
        Route query to appropriate collections.
        
        Args:
            query: User query
            
        Returns:
            List of collections to search
        """
        analysis = self.analyze_query(query)
        return analysis.suggested_collections


class RerankerStrategy(ABC):
    """Abstract base class for reranking strategies."""
    
    @abstractmethod
    def rerank(
        self,
        results: List[SearchResult],
        query: str,
        **kwargs,
    ) -> List[SearchResult]:
        """Rerank search results."""
        pass


class RecencyReranker(RerankerStrategy):
    """
    Reranks results by recency, boosting newer documents.
    """
    
    def __init__(
        self,
        recency_weight: float = 0.3,
        decay_days: int = 30,
    ):
        """
        Initialize recency reranker.
        
        Args:
            recency_weight: Weight for recency score (0-1)
            decay_days: Days after which recency boost decays to ~0.37
        """
        self.recency_weight = recency_weight
        self.decay_days = decay_days
    
    def rerank(
        self,
        results: List[SearchResult],
        query: str,
        **kwargs,
    ) -> List[SearchResult]:
        """Rerank by combining similarity and recency."""
        now = datetime.now()
        reranked = []
        
        for result in results:
            timestamp_str = result.metadata.get("timestamp")
            recency_score = 0.5  # Default if no timestamp
            
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    days_old = (now - timestamp).days
                    # Exponential decay
                    import math
                    recency_score = math.exp(-days_old / self.decay_days)
                except (ValueError, TypeError):
                    pass
            
            # Combine scores
            combined_score = (
                (1 - self.recency_weight) * result.score +
                self.recency_weight * recency_score
            )
            
            # Create new result with updated score
            reranked.append(SearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=combined_score,
                metadata=result.metadata,
                doc_id=result.doc_id,
                collection=result.collection,
            ))
        
        # Sort by new score
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked


class SeverityReranker(RerankerStrategy):
    """
    Reranks results by severity, boosting critical items.
    """
    
    SEVERITY_SCORES = {
        "critical": 1.0,
        "high": 0.8,
        "medium": 0.5,
        "low": 0.2,
    }
    
    def __init__(self, severity_weight: float = 0.2):
        """
        Initialize severity reranker.
        
        Args:
            severity_weight: Weight for severity boost (0-1)
        """
        self.severity_weight = severity_weight
    
    def rerank(
        self,
        results: List[SearchResult],
        query: str,
        **kwargs,
    ) -> List[SearchResult]:
        """Rerank by combining similarity and severity."""
        reranked = []
        
        for result in results:
            severity = result.metadata.get("severity", "medium")
            severity_score = self.SEVERITY_SCORES.get(severity.lower(), 0.5)
            
            combined_score = (
                (1 - self.severity_weight) * result.score +
                self.severity_weight * severity_score
            )
            
            reranked.append(SearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=combined_score,
                metadata=result.metadata,
                doc_id=result.doc_id,
                collection=result.collection,
            ))
        
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked


class DiversityReranker(RerankerStrategy):
    """
    Reranks to ensure diversity across documents and collections.
    Uses Maximal Marginal Relevance (MMR) style approach.
    """
    
    def __init__(
        self,
        diversity_weight: float = 0.3,
        max_per_document: int = 2,
    ):
        """
        Initialize diversity reranker.
        
        Args:
            diversity_weight: Trade-off between relevance and diversity
            max_per_document: Maximum chunks from same document
        """
        self.diversity_weight = diversity_weight
        self.max_per_document = max_per_document
    
    def rerank(
        self,
        results: List[SearchResult],
        query: str,
        **kwargs,
    ) -> List[SearchResult]:
        """Rerank with diversity constraint."""
        if not results:
            return results
        
        reranked = []
        doc_counts: Dict[str, int] = {}
        collection_counts: Dict[str, int] = {}
        
        for result in results:
            doc_id = result.doc_id
            collection = result.collection
            
            # Check document limit
            if doc_counts.get(doc_id, 0) >= self.max_per_document:
                continue
            
            # Apply diversity penalty
            doc_penalty = doc_counts.get(doc_id, 0) * 0.2
            collection_penalty = (collection_counts.get(collection, 0) * 0.1)
            
            adjusted_score = result.score * (1 - self.diversity_weight * (doc_penalty + collection_penalty))
            
            reranked.append(SearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=adjusted_score,
                metadata=result.metadata,
                doc_id=result.doc_id,
                collection=result.collection,
            ))
            
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            collection_counts[collection] = collection_counts.get(collection, 0) + 1
        
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked


class CombinedReranker(RerankerStrategy):
    """
    Combines multiple reranking strategies.
    """
    
    def __init__(self, rerankers: List[Tuple[RerankerStrategy, float]]):
        """
        Initialize with weighted rerankers.
        
        Args:
            rerankers: List of (reranker, weight) tuples
        """
        self.rerankers = rerankers
        total_weight = sum(w for _, w in rerankers)
        # Normalize weights
        self.rerankers = [(r, w / total_weight) for r, w in rerankers]
    
    def rerank(
        self,
        results: List[SearchResult],
        query: str,
        **kwargs,
    ) -> List[SearchResult]:
        """Apply all rerankers and combine scores."""
        if not results:
            return results
        
        # Collect scores from each reranker
        all_scores: Dict[str, List[float]] = {r.chunk_id: [] for r in results}
        
        for reranker, weight in self.rerankers:
            reranked = reranker.rerank(results, query, **kwargs)
            for r in reranked:
                all_scores[r.chunk_id].append(r.score * weight)
        
        # Create results with combined scores
        combined = []
        for result in results:
            scores = all_scores[result.chunk_id]
            combined_score = sum(scores)
            
            combined.append(SearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=combined_score,
                metadata=result.metadata,
                doc_id=result.doc_id,
                collection=result.collection,
            ))
        
        combined.sort(key=lambda x: x.score, reverse=True)
        return combined


class BM25Retriever:
    """
    Sparse retrieval using BM25 algorithm.
    
    Provides lexical matching to complement dense embeddings.
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self._corpus: List[Chunk] = []
        self._bm25 = None
        self._tokenized_corpus: List[List[str]] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Remove very short tokens
        return [t for t in tokens if len(t) > 2]
    
    def index(self, chunks: List[Chunk]):
        """
        Index chunks for BM25 search.
        
        Args:
            chunks: List of chunks to index
        """
        self._corpus = chunks
        self._tokenized_corpus = [
            self._tokenize(chunk.content)
            for chunk in chunks
        ]
        
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(
                self._tokenized_corpus,
                k1=self.k1,
                b=self.b,
            )
            logger.info(f"BM25 indexed {len(chunks)} chunks")
        except ImportError:
            logger.warning(
                "rank_bm25 not installed. BM25 search unavailable. "
                "Install with: pip install rank-bm25"
            )
            self._bm25 = None
    
    def search(
        self,
        query: str,
        n_results: int = 10,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search using BM25.
        
        Args:
            query: Search query
            n_results: Maximum results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        if self._bm25 is None or not self._corpus:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top results
        scored_chunks = list(zip(self._corpus, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize scores to 0-1 range
        if scored_chunks:
            max_score = max(s for _, s in scored_chunks)
            if max_score > 0:
                scored_chunks = [
                    (c, s / max_score) for c, s in scored_chunks
                ]
        
        return scored_chunks[:n_results]


class HybridRetriever:
    """
    Combines dense (embedding) and sparse (BM25) retrieval.
    
    Uses reciprocal rank fusion to merge results from both methods.
    """
    
    def __init__(
        self,
        vector_store: Union[ChromaVectorStore, InMemoryVectorStore],
        bm25_retriever: Optional[BM25Retriever] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store for dense retrieval
            bm25_retriever: BM25 retriever for sparse retrieval
            dense_weight: Weight for dense (embedding) results
            sparse_weight: Weight for sparse (BM25) results
            rrf_k: Reciprocal Rank Fusion constant
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
    
    def index_for_bm25(self, chunks: List[Chunk]):
        """Index chunks for BM25 retrieval."""
        self.bm25_retriever.index(chunks)
    
    def search(
        self,
        query: str,
        collections: Optional[List[CollectionType]] = None,
        n_results: int = 10,
        filter: Optional[MetadataFilter] = None,
        use_bm25: bool = True,
    ) -> SearchResults:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            collections: Collections to search
            n_results: Maximum results
            filter: Metadata filter
            use_bm25: Whether to include BM25 results
            
        Returns:
            Combined search results
        """
        import time
        start_time = time.time()
        
        # Dense retrieval
        dense_results = self.vector_store.search(
            query=query,
            collections=collections,
            n_results=n_results * 2,  # Over-fetch for fusion
            filter=filter,
        )
        
        # Sparse retrieval (if enabled and indexed)
        sparse_results = []
        if use_bm25 and self.bm25_retriever._corpus:
            sparse_raw = self.bm25_retriever.search(query, n_results * 2)
            # Convert to SearchResult format
            sparse_results = [
                SearchResult(
                    chunk_id=chunk.id,
                    content=chunk.content,
                    score=score,
                    metadata=chunk.metadata,
                    doc_id=chunk.doc_id,
                    collection=chunk.metadata.get("doc_type", "general"),
                )
                for chunk, score in sparse_raw
            ]
        
        # Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            dense_results.results,
            sparse_results,
        )
        
        search_time = (time.time() - start_time) * 1000
        
        return SearchResults(
            results=fused_results[:n_results],
            query=query,
            collections_searched=dense_results.collections_searched,
            total_results=len(fused_results),
            search_time_ms=search_time,
        )
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) for each result list
        """
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, SearchResult] = {}
        
        # Score dense results
        for rank, result in enumerate(dense_results, 1):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (
                self.dense_weight / (self.rrf_k + rank)
            )
            result_map[chunk_id] = result
        
        # Score sparse results
        for rank, result in enumerate(sparse_results, 1):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (
                self.sparse_weight / (self.rrf_k + rank)
            )
            if chunk_id not in result_map:
                result_map[chunk_id] = result
        
        # Create fused results
        fused = []
        for chunk_id, rrf_score in sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            original = result_map[chunk_id]
            fused.append(SearchResult(
                chunk_id=chunk_id,
                content=original.content,
                score=rrf_score,
                metadata=original.metadata,
                doc_id=original.doc_id,
                collection=original.collection,
            ))
        
        return fused


@dataclass
class RetrievalConfig:
    """Configuration for the unified retriever."""
    n_results: int = 10
    use_query_routing: bool = True
    use_reranking: bool = True
    use_hybrid: bool = True
    recency_weight: float = 0.2
    severity_weight: float = 0.1
    diversity_weight: float = 0.2
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    min_score_threshold: float = 0.3


class SupplyChainRetriever:
    """
    Unified retriever for supply chain knowledge.
    
    Combines query routing, hybrid retrieval, and reranking
    for optimal context retrieval.
    """
    
    def __init__(
        self,
        vector_store: Union[ChromaVectorStore, InMemoryVectorStore],
        embedding_model: Optional[EmbeddingModel] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        """
        Initialize retriever.
        
        Args:
            vector_store: Vector store backend
            embedding_model: Embedding model (for BM25 index building)
            config: Retrieval configuration
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.config = config or RetrievalConfig()
        
        # Initialize components
        self.query_router = QueryRouter()
        self.bm25_retriever = BM25Retriever()
        self.hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_retriever=self.bm25_retriever,
            dense_weight=self.config.dense_weight,
            sparse_weight=self.config.sparse_weight,
        )
        
        # Initialize rerankers
        self.reranker = CombinedReranker([
            (RecencyReranker(recency_weight=self.config.recency_weight), 0.4),
            (SeverityReranker(severity_weight=self.config.severity_weight), 0.3),
            (DiversityReranker(diversity_weight=self.config.diversity_weight), 0.3),
        ])
    
    def index_chunks(self, chunks: List[Chunk]):
        """Index chunks for BM25 retrieval."""
        self.bm25_retriever.index(chunks)
    
    def retrieve(
        self,
        query: str,
        n_results: Optional[int] = None,
        collections: Optional[List[CollectionType]] = None,
        filter: Optional[MetadataFilter] = None,
        override_config: Optional[Dict[str, Any]] = None,
    ) -> SearchResults:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            n_results: Override number of results
            collections: Override collections to search
            filter: Additional metadata filters
            override_config: Override specific config values
            
        Returns:
            Ranked search results
        """
        import time
        start_time = time.time()
        
        config = self.config
        if override_config:
            # Create modified config
            config = RetrievalConfig(**{
                **self.config.__dict__,
                **override_config,
            })
        
        n_results = n_results or config.n_results
        
        # Step 1: Route query to collections
        if config.use_query_routing and collections is None:
            query_analysis = self.query_router.analyze_query(query)
            collections = query_analysis.suggested_collections
            
            # Add filter from query analysis
            if query_analysis.detected_disruption_type and filter is None:
                filter = MetadataFilter().disruption_type(
                    query_analysis.detected_disruption_type
                )
        
        # Step 2: Retrieve results
        if config.use_hybrid:
            results = self.hybrid_retriever.search(
                query=query,
                collections=collections,
                n_results=n_results * 2,  # Over-fetch for reranking
                filter=filter,
            )
        else:
            results = self.vector_store.search(
                query=query,
                collections=collections,
                n_results=n_results * 2,
                filter=filter,
            )
        
        # Step 3: Rerank results
        if config.use_reranking and results.results:
            reranked = self.reranker.rerank(results.results, query)
            results = SearchResults(
                results=reranked,
                query=query,
                collections_searched=results.collections_searched,
                total_results=len(reranked),
                search_time_ms=results.search_time_ms,
            )
        
        # Step 4: Filter by threshold and limit
        filtered = [
            r for r in results.results
            if r.score >= config.min_score_threshold
        ][:n_results]
        
        search_time = (time.time() - start_time) * 1000
        
        return SearchResults(
            results=filtered,
            query=query,
            collections_searched=results.collections_searched,
            total_results=len(filtered),
            search_time_ms=search_time,
        )
    
    def retrieve_for_disruption(
        self,
        disruption_description: str,
        disruption_type: Optional[DisruptionType] = None,
        include_best_practices: bool = True,
        include_case_studies: bool = True,
        n_results: int = 10,
    ) -> SearchResults:
        """
        Retrieve context for a specific disruption.
        
        Args:
            disruption_description: Description of the disruption
            disruption_type: Known disruption type (or auto-detect)
            include_best_practices: Include mitigation strategies
            include_case_studies: Include relevant case studies
            n_results: Maximum results
            
        Returns:
            Relevant context for handling the disruption
        """
        # Detect disruption type if not provided
        if disruption_type is None:
            analysis = self.query_router.analyze_query(disruption_description)
            disruption_type = analysis.detected_disruption_type
        
        # Determine collections
        collections = [CollectionType.DISRUPTIONS, CollectionType.NEWS]
        if include_best_practices:
            collections.append(CollectionType.BEST_PRACTICES)
        if include_case_studies:
            collections.append(CollectionType.CASE_STUDIES)
        
        # Create filter
        filter = None
        if disruption_type:
            filter = MetadataFilter().disruption_type(disruption_type)
        
        return self.retrieve(
            query=disruption_description,
            n_results=n_results,
            collections=collections,
            filter=filter,
        )
    
    def retrieve_best_practices(
        self,
        topic: str,
        region: Optional[Region] = None,
        industry: Optional[str] = None,
        n_results: int = 5,
    ) -> SearchResults:
        """
        Retrieve best practices for a topic.
        
        Args:
            topic: Topic to find best practices for
            region: Optional region filter
            industry: Optional industry filter
            n_results: Maximum results
            
        Returns:
            Relevant best practices
        """
        filter = MetadataFilter()
        if region:
            filter.region(region)
        if industry:
            filter.industry(industry)
        
        return self.retrieve(
            query=f"best practice {topic}",
            n_results=n_results,
            collections=[CollectionType.BEST_PRACTICES, CollectionType.CASE_STUDIES],
            filter=filter if (region or industry) else None,
        )
    
    def retrieve_recent_context(
        self,
        topic: str,
        days: int = 7,
        n_results: int = 10,
    ) -> SearchResults:
        """
        Retrieve recent context about a topic.
        
        Args:
            topic: Topic to search for
            days: Number of days to look back
            n_results: Maximum results
            
        Returns:
            Recent relevant context
        """
        cutoff = datetime.now() - timedelta(days=days)
        filter = MetadataFilter().since(cutoff)
        
        return self.retrieve(
            query=topic,
            n_results=n_results,
            collections=[CollectionType.NEWS, CollectionType.DISRUPTIONS],
            filter=filter,
            override_config={"recency_weight": 0.5},  # Boost recency
        )
    
    def get_context_for_llm(
        self,
        query: str,
        max_tokens: int = 2000,
        n_results: int = 5,
    ) -> str:
        """
        Get formatted context for LLM consumption.
        
        Args:
            query: User query
            max_tokens: Approximate token limit (chars / 4)
            n_results: Maximum results to include
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, n_results=n_results)
        
        if not results.results:
            return ""
        
        context_parts = []
        total_chars = 0
        char_limit = max_tokens * 4  # Rough estimate
        
        for i, result in enumerate(results.results, 1):
            # Format context entry
            entry = f"[Source {i}] {result.content}"
            
            # Add metadata hints
            if result.metadata.get("doc_type"):
                entry += f"\n(Type: {result.metadata['doc_type']}"
                if result.metadata.get("severity"):
                    entry += f", Severity: {result.metadata['severity']}"
                entry += ")"
            
            entry_chars = len(entry)
            if total_chars + entry_chars > char_limit:
                break
            
            context_parts.append(entry)
            total_chars += entry_chars
        
        return "\n\n".join(context_parts)


# Convenience functions
def create_retriever(
    vector_store: Union[ChromaVectorStore, InMemoryVectorStore],
    config: Optional[RetrievalConfig] = None,
) -> SupplyChainRetriever:
    """
    Create a configured retriever.
    
    Args:
        vector_store: Vector store backend
        config: Optional configuration
        
    Returns:
        Configured SupplyChainRetriever
    """
    return SupplyChainRetriever(
        vector_store=vector_store,
        config=config,
    )


def get_default_retriever(
    persist_directory: str = "./data/vectorstore",
) -> SupplyChainRetriever:
    """
    Get a default retriever with standard configuration.
    
    Args:
        persist_directory: Vector store persistence path
        
    Returns:
        Configured retriever
    """
    from .vector_store import get_default_vector_store
    
    vector_store = get_default_vector_store(persist_directory)
    return SupplyChainRetriever(vector_store=vector_store)
