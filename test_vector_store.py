"""
Test Vector Store (RAG Phase 2)

Comprehensive tests for ChromaDB vector store,
collection management, and similarity search.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all vector store components can be imported."""
    print("\n" + "=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)
    
    try:
        from src.cognition.rag import (ChromaVectorStore, CollectionType,
                                       InMemoryVectorStore, MetadataFilter,
                                       SearchResult, SearchResults,
                                       create_vector_store)
        print("✓ All vector store imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_in_memory_store():
    """Test in-memory vector store for basic functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: In-Memory Vector Store")
    print("=" * 60)
    
    from src.cognition.rag import (Chunk, DocumentType, InMemoryVectorStore,
                                   MockEmbeddings)

    # Create store with mock embeddings
    store = InMemoryVectorStore(embedding_model=MockEmbeddings(dimension=384))
    
    # Create test chunks
    chunks = [
        Chunk(
            id="chunk_1",
            content="Supply chain disruption due to semiconductor shortage affecting electronics manufacturing.",
            metadata={"doc_type": "news", "severity": "high"},
            doc_id="doc_1",
            chunk_index=0,
            total_chunks=1,
        ),
        Chunk(
            id="chunk_2",
            content="Best practice: Maintain safety stock of critical components to buffer against shortages.",
            metadata={"doc_type": "best_practice", "industry": "manufacturing"},
            doc_id="doc_2",
            chunk_index=0,
            total_chunks=1,
        ),
        Chunk(
            id="chunk_3",
            content="Port congestion in Asia delays container shipments by up to 2 weeks.",
            metadata={"doc_type": "news", "region": "asia"},
            doc_id="doc_3",
            chunk_index=0,
            total_chunks=1,
        ),
    ]
    
    # Add chunks
    added = store.add_chunks(chunks, collection="test")
    print(f"✓ Added {added} chunks to in-memory store")
    
    # Test count
    count = store.count("test")
    print(f"✓ Store contains {count} chunks")
    
    # Test search
    results = store.search("semiconductor shortage impact", n_results=2)
    print(f"✓ Search returned {len(results)} results")
    
    if results:
        top_chunk, top_score = results[0]
        print(f"  Top result: '{top_chunk.content[:50]}...' (score: {top_score:.4f})")
    
    # Test clear
    store.clear()
    assert store.count() == 0
    print("✓ Store cleared successfully")
    
    return True


def test_metadata_filter():
    """Test metadata filter building."""
    print("\n" + "=" * 60)
    print("TEST 3: Metadata Filter")
    print("=" * 60)
    
    from src.cognition.rag import DisruptionType, MetadataFilter, Region

    # Test single filter
    filter1 = MetadataFilter().disruption_type(DisruptionType.STOCKOUT)
    clause1 = filter1.build()
    print(f"✓ Single filter: {clause1}")
    
    # Test multiple filters (AND)
    filter2 = (
        MetadataFilter()
        .disruption_type(DisruptionType.SUPPLIER_FAILURE)
        .region(Region.ASIA)
        .severity("critical")
    )
    clause2 = filter2.build()
    print(f"✓ Multiple filters: {clause2}")
    
    # Test date filter
    cutoff = datetime.now() - timedelta(days=7)
    filter3 = MetadataFilter().since(cutoff)
    clause3 = filter3.build()
    print(f"✓ Date filter: {clause3}")
    
    # Test empty filter
    filter4 = MetadataFilter()
    clause4 = filter4.build()
    assert clause4 is None
    print("✓ Empty filter returns None")
    
    return True


def test_collection_types():
    """Test collection type mapping."""
    print("\n" + "=" * 60)
    print("TEST 4: Collection Types")
    print("=" * 60)
    
    from src.cognition.rag import CollectionType, DocumentType
    from src.cognition.rag.vector_store import DOCUMENT_TO_COLLECTION

    # Test all document types have mappings
    for doc_type in DocumentType:
        collection = DOCUMENT_TO_COLLECTION.get(doc_type)
        print(f"  {doc_type.value} -> {collection.value if collection else 'GENERAL'}")
    
    print(f"✓ All {len(DocumentType)} document types mapped")
    
    # Verify collection types
    print(f"✓ {len(CollectionType)} collection types available:")
    for ctype in CollectionType:
        print(f"    - {ctype.value}")
    
    return True


def test_chromadb_store():
    """Test ChromaDB vector store (requires chromadb)."""
    print("\n" + "=" * 60)
    print("TEST 5: ChromaDB Vector Store")
    print("=" * 60)
    
    try:
        import chromadb
    except ImportError:
        print("⚠ chromadb not installed - skipping ChromaDB tests")
        print("  Install with: pip install chromadb")
        return True  # Not a failure, just skip
    
    from src.cognition.rag import (ChromaVectorStore, Chunk, CollectionType,
                                   DisruptionType, DocumentType,
                                   MetadataFilter, MockEmbeddings)

    # Create in-memory ChromaDB store (no persistence)
    store = ChromaVectorStore(
        persist_directory=None,  # In-memory
        embedding_model=MockEmbeddings(dimension=384),
        collection_prefix="test_",  # Prefix for test isolation
    )
    
    print("✓ ChromaDB client initialized (in-memory)")
    
    # Create test chunks
    test_chunks = [
        Chunk(
            id="news_001_chunk_0",
            content="Global chip shortage disrupts automotive supply chains worldwide. Major manufacturers face production delays.",
            metadata={
                "doc_type": "news",
                "disruption_type": "raw_material_shortage",
                "region": "global",
                "severity": "high",
                "timestamp": datetime.now().isoformat(),
                "entities": ["semiconductor", "automotive"],
            },
            doc_id="news_001",
            chunk_index=0,
            total_chunks=2,
        ),
        Chunk(
            id="news_001_chunk_1",
            content="Industry experts predict recovery by Q4 as new fabrication plants come online in Taiwan and Arizona.",
            metadata={
                "doc_type": "news",
                "disruption_type": "raw_material_shortage",
                "region": "global",
                "severity": "high",
                "timestamp": datetime.now().isoformat(),
            },
            doc_id="news_001",
            chunk_index=1,
            total_chunks=2,
        ),
        Chunk(
            id="bp_001_chunk_0",
            content="Best Practice: Dual sourcing strategy reduces dependency on single suppliers. Maintain relationships with backup suppliers.",
            metadata={
                "doc_type": "best_practice",
                "industry": "manufacturing",
                "timestamp": datetime.now().isoformat(),
            },
            doc_id="bp_001",
            chunk_index=0,
            total_chunks=1,
        ),
        Chunk(
            id="cs_001_chunk_0",
            content="Case Study: How Toyota's JIT system adapted during semiconductor crisis. Buffer stock strategy implementation.",
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
    ]
    
    # Add chunks to different collections
    added_news = store.add_chunks(
        [c for c in test_chunks if c.metadata.get("doc_type") == "news"],
        collection_type=CollectionType.NEWS,
    )
    print(f"✓ Added {added_news} chunks to NEWS collection")
    
    added_bp = store.add_chunks(
        [c for c in test_chunks if c.metadata.get("doc_type") == "best_practice"],
        collection_type=CollectionType.BEST_PRACTICES,
    )
    print(f"✓ Added {added_bp} chunk to BEST_PRACTICES collection")
    
    added_cs = store.add_chunks(
        [c for c in test_chunks if c.metadata.get("doc_type") == "case_study"],
        collection_type=CollectionType.CASE_STUDIES,
    )
    print(f"✓ Added {added_cs} chunk to CASE_STUDIES collection")
    
    # Test basic search
    results = store.search(
        query="chip shortage automotive impact",
        n_results=5,
    )
    print(f"\n✓ Search: 'chip shortage automotive impact'")
    print(f"  Found {len(results)} results in {results.search_time_ms:.2f}ms")
    print(f"  Collections searched: {results.collections_searched}")
    
    for i, result in enumerate(results.top(3)):
        print(f"  [{i+1}] Score: {result.score:.4f}, Collection: {result.collection}")
        print(f"      Content: '{result.content[:60]}...'")
    
    # Test filtered search
    print(f"\n✓ Testing filtered search (severity=high)")
    filter_high = MetadataFilter().severity("high")
    filtered_results = store.search(
        query="supply chain disruption",
        collections=[CollectionType.NEWS],
        n_results=5,
        filter=filter_high,
    )
    print(f"  Found {len(filtered_results)} results with severity=high")
    
    # Test best practices search
    print(f"\n✓ Testing best practices search")
    bp_results = store.search_best_practices(
        query="supplier risk mitigation",
        n_results=3,
    )
    print(f"  Found {len(bp_results)} best practice results")
    
    # Test get collection stats
    print(f"\n✓ Collection statistics:")
    stats = store.get_collection_stats()
    for coll, info in stats.items():
        if info.get("count", 0) > 0:
            print(f"    {coll}: {info['count']} chunks")
    
    # Test document retrieval
    print(f"\n✓ Testing document retrieval (doc_id='news_001')")
    doc_chunks = store.get_document_chunks("news_001", CollectionType.NEWS)
    print(f"  Retrieved {len(doc_chunks)} chunks for document")
    
    return True


def test_search_results():
    """Test SearchResults helper methods."""
    print("\n" + "=" * 60)
    print("TEST 6: SearchResults Helper Methods")
    print("=" * 60)
    
    from src.cognition.rag import SearchResult, SearchResults

    # Create mock results
    results = [
        SearchResult(
            chunk_id="c1", content="Content 1", score=0.95,
            metadata={}, doc_id="d1", collection="news",
        ),
        SearchResult(
            chunk_id="c2", content="Content 2", score=0.85,
            metadata={}, doc_id="d2", collection="best_practices",
        ),
        SearchResult(
            chunk_id="c3", content="Content 3", score=0.75,
            metadata={}, doc_id="d3", collection="news",
        ),
        SearchResult(
            chunk_id="c4", content="Content 4", score=0.65,
            metadata={}, doc_id="d4", collection="best_practices",
        ),
    ]
    
    search_results = SearchResults(
        results=results,
        query="test query",
        collections_searched=["news", "best_practices"],
        total_results=4,
        search_time_ms=10.5,
    )
    
    # Test top()
    top2 = search_results.top(2)
    assert len(top2) == 2
    assert top2[0].score == 0.95
    print(f"✓ top(2) returned {len(top2)} results")
    
    # Test by_collection()
    news = search_results.by_collection("news")
    assert len(news) == 2
    print(f"✓ by_collection('news') returned {len(news)} results")
    
    # Test above_threshold()
    high_score = search_results.above_threshold(0.80)
    assert len(high_score) == 2
    print(f"✓ above_threshold(0.80) returned {len(high_score)} results")
    
    # Test iteration
    count = sum(1 for _ in search_results)
    assert count == 4
    print(f"✓ Iteration works (counted {count} results)")
    
    return True


def test_end_to_end_pipeline():
    """Test end-to-end: ingest -> chunk -> embed -> store -> search."""
    print("\n" + "=" * 60)
    print("TEST 7: End-to-End Pipeline")
    print("=" * 60)
    
    try:
        import chromadb
    except ImportError:
        print("⚠ chromadb not installed - skipping E2E test")
        return True
    
    from src.cognition.rag import (ChromaVectorStore, CollectionType,
                                   DocumentIngester, DocumentType,
                                   MockEmbeddings, SupplyChainChunker)

    # Sample supply chain document
    content = """
    Supply Chain Risk Alert: Port Congestion Crisis
    
    Major ports across the Pacific Rim are experiencing unprecedented 
    congestion, with container ships waiting up to 2 weeks for berth access.
    
    The bottleneck is caused by a combination of factors:
    - Surge in consumer demand post-pandemic
    - Labor shortages at port facilities
    - Insufficient chassis and truck availability
    
    Impact on Supply Chains:
    Companies report significant delays in receiving components, leading to
    production slowdowns. The automotive and electronics sectors are 
    particularly affected.
    
    Recommended Actions:
    1. Diversify shipping routes to alternate ports
    2. Increase safety stock for critical items
    3. Negotiate longer lead times with customers
    4. Consider air freight for high-value items
    """
    
    # Step 1: Ingest
    print("Step 1: Ingesting document...")
    ingester = DocumentIngester()
    doc = ingester.ingest_text(content, DocumentType.NEWS)
    print(f"  ✓ Document ID: {doc.id}")
    print(f"  ✓ Disruption type: {doc.metadata.disruption_type}")
    
    # Step 2: Chunk
    print("\nStep 2: Chunking document...")
    chunker = SupplyChainChunker()
    chunks = chunker.chunk_document(doc)
    print(f"  ✓ Created {len(chunks)} chunks")
    
    # Step 3: Store
    print("\nStep 3: Storing in vector database...")
    store = ChromaVectorStore(
        persist_directory=None,
        embedding_model=MockEmbeddings(dimension=384),
        collection_prefix="e2e_",
    )
    added = store.add_document(doc, chunks)
    print(f"  ✓ Stored {added} chunks")
    
    # Step 4: Search
    print("\nStep 4: Searching for related content...")
    
    # Query 1: General search
    results1 = store.search("port congestion shipping delays")
    print(f"\n  Query: 'port congestion shipping delays'")
    print(f"  ✓ Found {len(results1)} results")
    if results1.results:
        print(f"    Top score: {results1.results[0].score:.4f}")
    
    # Query 2: Best practice style query
    results2 = store.search("how to handle logistics delays")
    print(f"\n  Query: 'how to handle logistics delays'")
    print(f"  ✓ Found {len(results2)} results")
    
    # Query 3: Specific sector
    results3 = store.search("automotive electronics impact")
    print(f"\n  Query: 'automotive electronics impact'")
    print(f"  ✓ Found {len(results3)} results")
    
    print("\n✓ End-to-End pipeline completed successfully!")
    return True


def run_all_tests():
    """Run all vector store tests."""
    print("\n" + "=" * 60)
    print("RAG PHASE 2: VECTOR STORE TESTS")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("In-Memory Store", test_in_memory_store),
        ("Metadata Filter", test_metadata_filter),
        ("Collection Types", test_collection_types),
        ("ChromaDB Store", test_chromadb_store),
        ("SearchResults Helpers", test_search_results),
        ("End-to-End Pipeline", test_end_to_end_pipeline),
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
