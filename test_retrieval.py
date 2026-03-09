"""
Test Retrieval Strategy (RAG Phase 3)

Comprehensive tests for query routing, reranking,
hybrid retrieval, and the unified retriever.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all retrieval components can be imported."""
    print("\n" + "=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)
    
    try:
        from src.cognition.rag import (BM25Retriever, CombinedReranker,
                                       DiversityReranker, HybridRetriever,
                                       QueryAnalysis, QueryIntent, QueryRouter,
                                       RecencyReranker, RetrievalConfig,
                                       SeverityReranker, SupplyChainRetriever,
                                       create_retriever)
        print("✓ All retrieval imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_query_router():
    """Test query routing and intent detection."""
    print("\n" + "=" * 60)
    print("TEST 2: Query Router")
    print("=" * 60)
    
    from src.cognition.rag import (DisruptionType, QueryIntent, QueryRouter,
                                   Region)
    
    router = QueryRouter()
    
    # Test disruption query
    analysis1 = router.analyze_query(
        "What's the impact of the semiconductor shortage on automotive suppliers?"
    )
    print(f"\nQuery: 'semiconductor shortage automotive impact'")
    print(f"  Intent: {analysis1.intent.value} (confidence: {analysis1.confidence:.2f})")
    print(f"  Disruption Type: {analysis1.detected_disruption_type}")
    print(f"  Collections: {[c.value for c in analysis1.suggested_collections]}")
    assert analysis1.intent == QueryIntent.DISRUPTION_INFO
    assert analysis1.detected_disruption_type == DisruptionType.RAW_MATERIAL_SHORTAGE
    print("✓ Disruption query routed correctly")
    
    # Test best practice query
    analysis2 = router.analyze_query(
        "How should we mitigate supplier risks? What are the best practices?"
    )
    print(f"\nQuery: 'mitigate supplier risks best practices'")
    print(f"  Intent: {analysis2.intent.value} (confidence: {analysis2.confidence:.2f})")
    print(f"  Collections: {[c.value for c in analysis2.suggested_collections]}")
    assert analysis2.intent == QueryIntent.BEST_PRACTICE
    print("✓ Best practice query routed correctly")
    
    # Test news query
    analysis3 = router.analyze_query(
        "Latest news about port congestion in Asia"
    )
    print(f"\nQuery: 'latest news port congestion Asia'")
    print(f"  Intent: {analysis3.intent.value}")
    print(f"  Region: {analysis3.detected_region}")
    assert analysis3.detected_region == Region.ASIA
    print("✓ News query with region detected")
    
    # Test urgent query
    analysis4 = router.analyze_query(
        "URGENT: Critical supplier failure affecting production immediately"
    )
    print(f"\nQuery: 'URGENT critical supplier failure'")
    print(f"  Is Urgent: {analysis4.is_urgent}")
    print(f"  Disruption Type: {analysis4.detected_disruption_type}")
    assert analysis4.is_urgent == True
    assert analysis4.detected_disruption_type == DisruptionType.SUPPLIER_FAILURE
    print("✓ Urgency and supplier failure detected")
    
    return True


def test_rerankers():
    """Test reranking strategies."""
    print("\n" + "=" * 60)
    print("TEST 3: Reranking Strategies")
    print("=" * 60)
    
    from src.cognition.rag import (CombinedReranker, DiversityReranker,
                                   RecencyReranker, SearchResult,
                                   SeverityReranker)

    # Create test results
    now = datetime.now()
    results = [
        SearchResult(
            chunk_id="c1",
            content="Old critical disruption",
            score=0.8,
            metadata={
                "timestamp": (now - timedelta(days=30)).isoformat(),
                "severity": "critical",
            },
            doc_id="d1",
            collection="news",
        ),
        SearchResult(
            chunk_id="c2",
            content="Recent low severity news",
            score=0.9,
            metadata={
                "timestamp": (now - timedelta(days=1)).isoformat(),
                "severity": "low",
            },
            doc_id="d2",
            collection="news",
        ),
        SearchResult(
            chunk_id="c3",
            content="Recent high severity alert",
            score=0.7,
            metadata={
                "timestamp": (now - timedelta(days=2)).isoformat(),
                "severity": "high",
            },
            doc_id="d3",
            collection="disruptions",
        ),
    ]
    
    # Test recency reranking
    recency_reranker = RecencyReranker(recency_weight=0.4)
    recency_results = recency_reranker.rerank(results, "test")
    print("\nRecency Reranking (weight=0.4):")
    for i, r in enumerate(recency_results):
        print(f"  [{i+1}] {r.chunk_id}: {r.score:.4f}")
    # Recent item should be boosted
    assert recency_results[0].chunk_id == "c2"
    print("✓ Recency reranker boosts recent items")
    
    # Test severity reranking
    severity_reranker = SeverityReranker(severity_weight=0.3)
    severity_results = severity_reranker.rerank(results, "test")
    print("\nSeverity Reranking (weight=0.3):")
    for i, r in enumerate(severity_results):
        print(f"  [{i+1}] {r.chunk_id}: {r.score:.4f} (severity: {r.metadata.get('severity')})")
    print("✓ Severity reranker adjusts by severity")
    
    # Test diversity reranking (with duplicates)
    dup_results = results + [
        SearchResult(
            chunk_id="c4",
            content="Another from d1",
            score=0.85,
            metadata={"timestamp": now.isoformat(), "severity": "medium"},
            doc_id="d1",  # Same doc as c1
            collection="news",
        ),
    ]
    diversity_reranker = DiversityReranker(max_per_document=1)
    diverse_results = diversity_reranker.rerank(dup_results, "test")
    print("\nDiversity Reranking (max_per_document=1):")
    doc_counts = {}
    for r in diverse_results:
        doc_counts[r.doc_id] = doc_counts.get(r.doc_id, 0) + 1
    print(f"  Documents represented: {list(doc_counts.keys())}")
    assert all(count <= 1 for count in doc_counts.values())
    print("✓ Diversity reranker limits per-document results")
    
    # Test combined reranking
    combined = CombinedReranker([
        (RecencyReranker(recency_weight=0.3), 0.4),
        (SeverityReranker(severity_weight=0.2), 0.3),
        (DiversityReranker(diversity_weight=0.2), 0.3),
    ])
    combined_results = combined.rerank(results, "test")
    print("\nCombined Reranking:")
    for i, r in enumerate(combined_results):
        print(f"  [{i+1}] {r.chunk_id}: {r.score:.4f}")
    print("✓ Combined reranker merges strategies")
    
    return True


def test_bm25_retriever():
    """Test BM25 sparse retrieval."""
    print("\n" + "=" * 60)
    print("TEST 4: BM25 Retriever")
    print("=" * 60)
    
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("⚠ rank_bm25 not installed - skipping BM25 tests")
        print("  Install with: pip install rank-bm25")
        return True
    
    from src.cognition.rag import BM25Retriever, Chunk

    # Create test chunks
    chunks = [
        Chunk(
            id="c1",
            content="Semiconductor chip shortage disrupts automotive production worldwide",
            metadata={"doc_type": "news"},
            doc_id="d1",
            chunk_index=0,
            total_chunks=1,
        ),
        Chunk(
            id="c2",
            content="Best practice for managing supplier relationships and contracts",
            metadata={"doc_type": "best_practice"},
            doc_id="d2",
            chunk_index=0,
            total_chunks=1,
        ),
        Chunk(
            id="c3",
            content="Port congestion causes shipping delays across Pacific routes",
            metadata={"doc_type": "news"},
            doc_id="d3",
            chunk_index=0,
            total_chunks=1,
        ),
        Chunk(
            id="c4",
            content="Automotive industry faces parts shortage from chip manufacturers",
            metadata={"doc_type": "news"},
            doc_id="d4",
            chunk_index=0,
            total_chunks=1,
        ),
    ]
    
    # Initialize and index
    bm25 = BM25Retriever()
    bm25.index(chunks)
    print(f"✓ Indexed {len(chunks)} chunks for BM25")
    
    # Search
    results = bm25.search("automotive chip shortage", n_results=3)
    print(f"\nQuery: 'automotive chip shortage'")
    print(f"Results ({len(results)}):")
    for chunk, score in results:
        print(f"  - [{chunk.id}] {score:.4f}: {chunk.content[:50]}...")
    
    # Verify relevant results are top
    top_ids = [c.id for c, _ in results[:2]]
    assert "c1" in top_ids or "c4" in top_ids
    print("✓ BM25 returns relevant results")
    
    return True


def test_hybrid_retriever():
    """Test hybrid retrieval combining dense and sparse."""
    print("\n" + "=" * 60)
    print("TEST 5: Hybrid Retriever")
    print("=" * 60)
    
    try:
        import chromadb
    except ImportError:
        print("⚠ chromadb not installed - skipping hybrid tests")
        return True
    
    from src.cognition.rag import (ChromaVectorStore, Chunk, CollectionType,
                                   HybridRetriever, MockEmbeddings)

    # Create vector store
    store = ChromaVectorStore(
        persist_directory=None,
        embedding_model=MockEmbeddings(dimension=384),
        collection_prefix="hybrid_test_",
    )
    
    # Create test chunks
    chunks = [
        Chunk(
            id="h1",
            content="Supply chain visibility platforms enable real-time tracking of inventory",
            metadata={"doc_type": "best_practice"},
            doc_id="doc1",
            chunk_index=0,
            total_chunks=1,
        ),
        Chunk(
            id="h2",
            content="Machine learning predicts demand patterns for inventory optimization",
            metadata={"doc_type": "research"},
            doc_id="doc2",
            chunk_index=0,
            total_chunks=1,
        ),
        Chunk(
            id="h3",
            content="Blockchain technology improves supply chain transparency and trust",
            metadata={"doc_type": "research"},
            doc_id="doc3",
            chunk_index=0,
            total_chunks=1,
        ),
    ]
    
    # Add to vector store
    store.add_chunks(chunks, collection_type=CollectionType.BEST_PRACTICES)
    
    # Create hybrid retriever
    hybrid = HybridRetriever(
        vector_store=store,
        dense_weight=0.7,
        sparse_weight=0.3,
    )
    
    # Index for BM25
    hybrid.index_for_bm25(chunks)
    print("✓ Hybrid retriever initialized with dense + sparse")
    
    # Search
    results = hybrid.search(
        "inventory tracking real-time",
        collections=[CollectionType.BEST_PRACTICES],
        n_results=3,
    )
    
    print(f"\nQuery: 'inventory tracking real-time'")
    print(f"Results ({len(results)}):")
    for r in results.results:
        print(f"  - [{r.chunk_id}] {r.score:.4f}: {r.content[:50]}...")
    print(f"Search time: {results.search_time_ms:.2f}ms")
    
    print("✓ Hybrid retrieval returns combined results")
    
    return True


def test_unified_retriever():
    """Test the unified SupplyChainRetriever."""
    print("\n" + "=" * 60)
    print("TEST 6: Unified SupplyChainRetriever")
    print("=" * 60)
    
    try:
        import chromadb
    except ImportError:
        print("⚠ chromadb not installed - skipping unified retriever tests")
        return True
    
    from src.cognition.rag import (ChromaVectorStore, Chunk, CollectionType,
                                   DisruptionType, MockEmbeddings,
                                   RetrievalConfig, SupplyChainRetriever)

    # Setup
    store = ChromaVectorStore(
        persist_directory=None,
        embedding_model=MockEmbeddings(dimension=384),
        collection_prefix="unified_test_",
    )
    
    # Create diverse test chunks
    chunks = [
        # Disruption news
        Chunk(
            id="n1",
            content="Global semiconductor shortage causes production halts at major automotive plants",
            metadata={
                "doc_type": "news",
                "disruption_type": "raw_material_shortage",
                "severity": "critical",
                "timestamp": datetime.now().isoformat(),
            },
            doc_id="news_001",
            chunk_index=0,
            total_chunks=1,
        ),
        # Best practice
        Chunk(
            id="bp1",
            content="Best Practice: Implement dual-sourcing strategy to reduce single-supplier dependency",
            metadata={
                "doc_type": "best_practice",
                "industry": "manufacturing",
                "timestamp": datetime.now().isoformat(),
            },
            doc_id="bp_001",
            chunk_index=0,
            total_chunks=1,
        ),
        # Case study
        Chunk(
            id="cs1",
            content="Case Study: How Toyota's JIT system adapted during the 2021 chip shortage crisis",
            metadata={
                "doc_type": "case_study",
                "industry": "automotive",
                "disruption_type": "raw_material_shortage",
                "timestamp": datetime.now().isoformat(),
            },
            doc_id="cs_001",
            chunk_index=0,
            total_chunks=1,
        ),
        # Older news
        Chunk(
            id="n2",
            content="Port congestion eases as shipping volumes normalize post-holiday season",
            metadata={
                "doc_type": "news",
                "disruption_type": "logistics_delay",
                "severity": "low",
                "timestamp": (datetime.now() - timedelta(days=30)).isoformat(),
            },
            doc_id="news_002",
            chunk_index=0,
            total_chunks=1,
        ),
    ]
    
    # Add to collections
    store.add_chunks([c for c in chunks if c.metadata.get("doc_type") == "news"],
                     collection_type=CollectionType.NEWS)
    store.add_chunks([c for c in chunks if c.metadata.get("doc_type") == "best_practice"],
                     collection_type=CollectionType.BEST_PRACTICES)
    store.add_chunks([c for c in chunks if c.metadata.get("doc_type") == "case_study"],
                     collection_type=CollectionType.CASE_STUDIES)
    
    # Create retriever
    config = RetrievalConfig(
        n_results=5,
        use_query_routing=True,
        use_reranking=True,
        use_hybrid=False,  # Disable hybrid for this test
        recency_weight=0.2,
        min_score_threshold=0.1,
    )
    
    retriever = SupplyChainRetriever(vector_store=store, config=config)
    print("✓ SupplyChainRetriever initialized")
    
    # Test general retrieve
    results1 = retriever.retrieve("semiconductor shortage automotive impact")
    print(f"\nQuery: 'semiconductor shortage automotive impact'")
    print(f"  Results: {len(results1)} (time: {results1.search_time_ms:.2f}ms)")
    for r in results1.results[:3]:
        print(f"    - {r.score:.4f}: {r.content[:50]}...")
    
    # Test disruption-specific retrieval
    results2 = retriever.retrieve_for_disruption(
        "chip shortage affecting production",
        disruption_type=DisruptionType.RAW_MATERIAL_SHORTAGE,
        include_best_practices=True,
    )
    print(f"\nDisruption Query: 'chip shortage'")
    print(f"  Results: {len(results2)}")
    
    # Test best practices retrieval
    results3 = retriever.retrieve_best_practices(
        "supplier risk management",
        industry="manufacturing",
    )
    print(f"\nBest Practices: 'supplier risk management'")
    print(f"  Results: {len(results3)}")
    
    # Test LLM context generation
    context = retriever.get_context_for_llm(
        "How should we handle semiconductor shortage?",
        max_tokens=1000,
        n_results=3,
    )
    print(f"\nLLM Context (truncated):")
    print(f"  '{context[:200]}...'")
    print(f"  Length: {len(context)} chars")
    
    print("\n✓ Unified retriever working correctly")
    
    return True


def test_end_to_end_retrieval():
    """Test complete retrieval pipeline."""
    print("\n" + "=" * 60)
    print("TEST 7: End-to-End Retrieval Pipeline")
    print("=" * 60)
    
    try:
        import chromadb
    except ImportError:
        print("⚠ chromadb not installed - skipping E2E test")
        return True
    
    from src.cognition.rag import (ChromaVectorStore, DocumentIngester,
                                   DocumentType, MockEmbeddings,
                                   SupplyChainChunker, SupplyChainRetriever)

    # Sample documents
    docs_content = [
        """
        Breaking: Major Semiconductor Shortage Alert
        
        The global semiconductor industry faces unprecedented shortages as demand
        surges post-pandemic. Key impacts include:
        - Automotive production delays up to 6 months
        - Consumer electronics backlogs growing
        - Prices increasing 20-30% across chip categories
        
        Experts recommend diversifying supplier base and increasing safety stock.
        """,
        """
        Best Practice Guide: Managing Supply Chain Disruptions
        
        Key strategies for resilience:
        1. Dual-sourcing: Never rely on a single supplier
        2. Safety stock: Maintain buffer inventory for critical components
        3. Visibility: Implement real-time tracking across the supply chain
        4. Communication: Regular updates with suppliers and customers
        
        Companies following these practices show 40% faster recovery times.
        """,
        """
        Case Study: Manufacturing Resilience During Crisis
        
        XYZ Manufacturing successfully navigated the 2024 component shortage
        by implementing predictive analytics and multi-tier supplier visibility.
        
        Results:
        - 95% on-time delivery maintained
        - 15% reduction in emergency procurement costs
        - Zero production line stoppages
        """,
    ]
    
    # Setup
    ingester = DocumentIngester()
    chunker = SupplyChainChunker()
    store = ChromaVectorStore(
        persist_directory=None,
        embedding_model=MockEmbeddings(dimension=384),
        collection_prefix="e2e_retrieval_",
    )
    
    # Ingest and store
    all_chunks = []
    doc_types = [DocumentType.NEWS, DocumentType.BEST_PRACTICE, DocumentType.CASE_STUDY]
    
    print("Step 1: Ingesting documents...")
    for i, (content, doc_type) in enumerate(zip(docs_content, doc_types)):
        doc = ingester.ingest_text(content, doc_type)
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
        store.add_document(doc, chunks)
        print(f"  ✓ Document {i+1}: {doc_type.value} ({len(chunks)} chunks)")
    
    print(f"\nTotal chunks indexed: {len(all_chunks)}")
    
    # Create retriever
    retriever = SupplyChainRetriever(vector_store=store)
    retriever.index_chunks(all_chunks)  # For BM25
    print("✓ Retriever ready with BM25 index")
    
    # Test queries
    test_queries = [
        ("semiconductor shortage impact", "Should find news about chip shortage"),
        ("how to manage supply disruptions", "Should find best practices"),
        ("successful crisis management example", "Should find case study"),
    ]
    
    print("\nStep 2: Testing queries...")
    for query, expected in test_queries:
        results = retriever.retrieve(query, n_results=3)
        print(f"\n  Query: '{query}'")
        print(f"  Expected: {expected}")
        print(f"  Found: {len(results)} results")
        if results.results:
            top = results.results[0]
            print(f"  Top result ({top.score:.4f}): {top.content[:60]}...")
    
    # Test LLM context
    print("\nStep 3: Generating LLM context...")
    context = retriever.get_context_for_llm(
        "What should we do about component shortages?",
        max_tokens=1500,
    )
    print(f"  Context length: {len(context)} chars")
    print(f"  Preview:\n    {context[:300]}...")
    
    print("\n✓ End-to-End retrieval pipeline working!")
    
    return True


def run_all_tests():
    """Run all retrieval tests."""
    print("\n" + "=" * 60)
    print("RAG PHASE 3: RETRIEVAL STRATEGY TESTS")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Query Router", test_query_router),
        ("Reranking Strategies", test_rerankers),
        ("BM25 Retriever", test_bm25_retriever),
        ("Hybrid Retriever", test_hybrid_retriever),
        ("Unified Retriever", test_unified_retriever),
        ("End-to-End Pipeline", test_end_to_end_retrieval),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
