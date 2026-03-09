"""Test RAG Pipeline Phase 1: Ingestion, Chunking, and Embeddings."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all RAG modules can be imported."""
    print("=" * 70)
    print("TEST: Module Imports")
    print("=" * 70)
    
    try:
        from src.cognition.rag import (Chunk, DisruptionType, DocumentIngester,
                                       DocumentMetadata, DocumentType,
                                       EntityExtractor, MockEmbeddings, Region,
                                       SupplyChainChunker, SupplyChainDocument,
                                       SupplyChainEmbeddings)
        print("✓ All RAG modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_entity_extraction():
    """Test entity extraction from supply chain text."""
    print("\n" + "=" * 70)
    print("TEST: Entity Extraction")
    print("=" * 70)
    
    from src.cognition.rag import DisruptionType, EntityExtractor, Region
    
    extractor = EntityExtractor()
    
    # Test text with various entities
    text = """
    Toyota announced a major production halt at its Japanese factories due to a 
    semiconductor chip shortage. The automotive giant expects this disruption to 
    affect vehicle production across Asia and Europe for the next quarter. 
    Industry analysts warn this could lead to stockouts at dealerships.
    """
    
    # Test disruption type extraction
    disruption = extractor.extract_disruption_type(text)
    print(f"Disruption type: {disruption}")
    # The text mentions stockout, chip shortage - either is valid
    valid_types = [DisruptionType.RAW_MATERIAL_SHORTAGE, DisruptionType.STOCKOUT]
    assert disruption in valid_types, f"Expected one of {valid_types}, got {disruption}"
    
    # Test region extraction
    region = extractor.extract_region(text)
    print(f"Region: {region}")
    assert region in [Region.ASIA, Region.EUROPE], f"Expected ASIA or EUROPE, got {region}"
    
    # Test industry extraction
    industry = extractor.extract_industry(text)
    print(f"Industry: {industry}")
    assert industry == "automotive", f"Expected automotive, got {industry}"
    
    # Test entity extraction
    entities = extractor.extract_entities(text)
    print(f"Entities: {entities}")
    assert "Toyota" in entities, f"Expected Toyota in entities"
    
    # Test severity assessment
    severity = extractor.assess_severity(text)
    print(f"Severity: {severity}")
    assert severity in ["medium", "high", "critical"], f"Expected medium/high/critical severity"
    
    print("✓ Entity extraction tests passed")
    return True


def test_document_ingestion():
    """Test document ingestion from text."""
    print("\n" + "=" * 70)
    print("TEST: Document Ingestion")
    print("=" * 70)
    
    from src.cognition.rag import DocumentIngester, DocumentType
    
    ingester = DocumentIngester()
    
    # Test text ingestion
    content = """
    Supply Chain Crisis Alert: Port Congestion Worsens
    
    Major shipping delays continue at ports across the Americas as container 
    backlogs reach critical levels. The logistics bottleneck is causing 
    significant disruptions for retail and manufacturing sectors.
    
    Companies are advised to increase safety stock levels and consider 
    alternative shipping routes. Air freight remains an option for critical 
    components despite higher costs.
    """
    
    doc = ingester.ingest_text(
        content=content,
        doc_type=DocumentType.NEWS,
        source="test",
        title="Supply Chain Crisis Alert",
    )
    
    print(f"Document ID: {doc.id}")
    print(f"Doc Type: {doc.metadata.doc_type}")
    print(f"Disruption Type: {doc.metadata.disruption_type}")
    print(f"Region: {doc.metadata.region}")
    print(f"Industry: {doc.metadata.industry}")
    print(f"Severity: {doc.metadata.severity}")
    print(f"Entities: {doc.metadata.entities[:5]}...")
    
    assert doc.id is not None
    assert doc.metadata.doc_type == DocumentType.NEWS
    assert doc.metadata.disruption_type is not None
    
    print("✓ Document ingestion tests passed")
    return True


def test_chunking():
    """Test document chunking strategies."""
    print("\n" + "=" * 70)
    print("TEST: Document Chunking")
    print("=" * 70)
    
    from src.cognition.rag import (DocumentIngester, DocumentType,
                                   FixedSizeChunker, ParagraphChunker,
                                   SectionChunker, SupplyChainChunker)

    # Create a longer document for chunking
    content = """
    # Supply Chain Risk Management Best Practices
    
    ## 1. Inventory Management
    
    Effective inventory management is crucial for supply chain resilience. 
    Companies should maintain appropriate safety stock levels based on demand 
    variability and lead time uncertainty. Regular inventory audits help 
    identify discrepancies and optimize holding costs.
    
    ## 2. Supplier Diversification
    
    Relying on a single supplier creates significant risk. Organizations should 
    develop relationships with multiple suppliers across different geographic 
    regions. This diversification provides backup options during disruptions 
    and increases negotiating leverage.
    
    ## 3. Demand Forecasting
    
    Accurate demand forecasting reduces inventory costs and prevents stockouts. 
    Modern approaches combine statistical methods with machine learning to 
    improve prediction accuracy. Real-time data integration enables faster 
    response to demand changes.
    
    ## 4. Logistics Optimization
    
    Transportation costs typically represent a significant portion of supply 
    chain expenses. Route optimization, carrier selection, and mode switching 
    can significantly reduce costs while maintaining service levels.
    """
    
    ingester = DocumentIngester()
    doc = ingester.ingest_text(content, DocumentType.BEST_PRACTICE, title="SC Best Practices")
    
    chunker = SupplyChainChunker()
    chunks = chunker.chunk_document(doc)
    
    print(f"Document length: {len(content)} chars")
    print(f"Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  ID: {chunk.id}")
        print(f"  Length: {len(chunk.content)} chars")
        print(f"  Preview: {chunk.content[:100]}...")
    
    assert len(chunks) >= 2, "Expected at least 2 chunks"
    assert all(chunk.doc_id == doc.id for chunk in chunks)
    
    # Test different chunking strategies
    print("\n--- Testing Different Chunking Strategies ---")
    
    fixed_chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)
    fixed_chunks = fixed_chunker.chunk(content)
    print(f"Fixed chunker: {len(fixed_chunks)} chunks")
    
    para_chunker = ParagraphChunker(min_chunk_size=50, max_chunk_size=500)
    para_chunks = para_chunker.chunk(content)
    print(f"Paragraph chunker: {len(para_chunks)} chunks")
    
    section_chunker = SectionChunker(max_chunk_size=800)
    section_chunks = section_chunker.chunk(content)
    print(f"Section chunker: {len(section_chunks)} chunks")
    
    print("✓ Chunking tests passed")
    return True


def test_embeddings():
    """Test embedding generation."""
    print("\n" + "=" * 70)
    print("TEST: Embeddings")
    print("=" * 70)
    
    from src.cognition.rag import Chunk, MockEmbeddings, SupplyChainEmbeddings

    # Test mock embeddings (no dependencies required)
    print("\n--- Testing Mock Embeddings ---")
    mock = MockEmbeddings(dimension=384)
    
    text = "Supply chain disruption caused by port congestion"
    embedding = mock.embed_text(text)
    
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    assert len(embedding) == 384
    assert isinstance(embedding[0], float)
    
    # Test batch embedding
    texts = [
        "Inventory stockout at distribution center",
        "Demand spike in consumer electronics",
        "Supplier delivery delay notification",
    ]
    embeddings = mock.embed_texts(texts)
    
    print(f"Batch embeddings: {len(embeddings)} vectors")
    assert len(embeddings) == 3
    
    # Test deterministic embeddings (same text = same embedding)
    emb1 = mock.embed_text("test")
    emb2 = mock.embed_text("test")
    assert emb1 == emb2, "Mock embeddings should be deterministic"
    
    # Test different texts give different embeddings
    emb3 = mock.embed_text("different text")
    assert emb1 != emb3, "Different texts should have different embeddings"
    
    print("✓ Mock embedding tests passed")
    
    # Test SupplyChainEmbeddings with mock provider
    print("\n--- Testing SupplyChainEmbeddings (mock provider) ---")
    sc_embeddings = SupplyChainEmbeddings(provider="mock")
    
    embedding = sc_embeddings.embed("Supply chain resilience strategy")
    print(f"SupplyChainEmbeddings dimension: {sc_embeddings.dimension}")
    
    assert len(embedding) == sc_embeddings.dimension
    
    # Test chunk embedding
    chunk = Chunk(
        id="test_chunk",
        content="Test content for embedding",
        metadata={},
        doc_id="test_doc",
        chunk_index=0,
        total_chunks=1,
    )
    chunk_embedding = sc_embeddings.embed_chunks([chunk])
    print(f"Chunk embedding: {len(chunk_embedding)} vectors")
    
    print("✓ Embedding tests passed")
    return True


def test_end_to_end_pipeline():
    """Test complete ingestion -> chunking -> embedding pipeline."""
    print("\n" + "=" * 70)
    print("TEST: End-to-End Pipeline")
    print("=" * 70)
    
    from src.cognition.rag import (DocumentIngester, DocumentType,
                                   SupplyChainChunker, SupplyChainEmbeddings)

    # Step 1: Ingest documents
    print("\n1. Ingesting documents...")
    ingester = DocumentIngester()
    
    documents = []
    
    # News article
    news_content = """
    Semiconductor shortage continues to impact global automotive industry. 
    Major manufacturers including Toyota, Ford, and Volkswagen have announced 
    production cuts. The chip shortage, initially caused by pandemic-related 
    factory shutdowns, has been exacerbated by increased demand for electronics.
    """
    doc1 = ingester.ingest_text(news_content, DocumentType.NEWS, title="Chip Shortage Impact")
    documents.append(doc1)
    
    # Case study
    case_content = """
    Case Study: Toyota's Response to the 2011 Thailand Floods
    
    The floods in Thailand disrupted Toyota's supply chain severely. As a result, 
    the company lost production of approximately 150,000 vehicles. In response, 
    Toyota implemented a new supplier risk assessment program.
    
    The outcome was a more resilient supply chain with better visibility into 
    sub-tier suppliers. This led to faster recovery times in subsequent disruptions.
    """
    doc2 = ingester.ingest_text(case_content, DocumentType.CASE_STUDY, title="Toyota Floods Response")
    documents.append(doc2)
    
    print(f"   Ingested {len(documents)} documents")
    
    # Step 2: Chunk documents
    print("\n2. Chunking documents...")
    chunker = SupplyChainChunker()
    
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"   {doc.metadata.title}: {len(chunks)} chunks")
    
    print(f"   Total chunks: {len(all_chunks)}")
    
    # Step 3: Generate embeddings
    print("\n3. Generating embeddings...")
    embeddings = SupplyChainEmbeddings(provider="mock")
    
    vectors = embeddings.embed_chunks(all_chunks)
    print(f"   Generated {len(vectors)} embeddings of dimension {embeddings.dimension}")
    
    # Verify results
    assert len(vectors) == len(all_chunks)
    assert all(len(v) == embeddings.dimension for v in vectors)
    
    # Print summary
    print("\n" + "-" * 40)
    print("PIPELINE SUMMARY")
    print("-" * 40)
    print(f"Documents ingested: {len(documents)}")
    print(f"Chunks created: {len(all_chunks)}")
    print(f"Embeddings generated: {len(vectors)}")
    print(f"Embedding dimension: {embeddings.dimension}")
    
    print("\nChunk details:")
    for chunk in all_chunks:
        print(f"  - {chunk.id[:30]}... ({len(chunk.content)} chars)")
    
    print("\n✓ End-to-end pipeline test passed")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RAG PIPELINE PHASE 1 TESTS")
    print("=" * 70)
    
    results = []
    
    results.append(("Module Imports", test_imports()))
    results.append(("Entity Extraction", test_entity_extraction()))
    results.append(("Document Ingestion", test_document_ingestion()))
    results.append(("Chunking", test_chunking()))
    results.append(("Embeddings", test_embeddings()))
    results.append(("End-to-End Pipeline", test_end_to_end_pipeline()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - RAG Phase 1 implementation complete!")
    else:
        print("\n✗ Some tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
