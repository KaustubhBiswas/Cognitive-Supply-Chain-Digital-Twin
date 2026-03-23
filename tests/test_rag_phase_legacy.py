"""Run legacy root-level RAG phase tests as part of default pytest suite."""

from __future__ import annotations

import importlib


def _run(module_name: str, function_name: str) -> None:
    module = importlib.import_module(module_name)
    func = getattr(module, function_name)
    result = func()
    if result is not None:
        assert result is True


def test_rag_pipeline_phase_1_2_legacy_suite() -> None:
    for fn_name in [
        "test_imports",
        "test_entity_extraction",
        "test_document_ingestion",
        "test_chunking",
        "test_embeddings",
        "test_end_to_end_pipeline",
    ]:
        _run("test_rag_pipeline", fn_name)


def test_rag_pipeline_phase_3_legacy_suite() -> None:
    for fn_name in [
        "test_imports",
        "test_query_router",
        "test_rerankers",
        "test_bm25_retriever",
        "test_hybrid_retriever",
        "test_unified_retriever",
        "test_end_to_end_retrieval",
    ]:
        _run("test_retrieval", fn_name)


def test_rag_pipeline_phase_2_vector_store_legacy_suite() -> None:
    for fn_name in [
        "test_imports",
        "test_in_memory_store",
        "test_metadata_filter",
        "test_collection_types",
        "test_chromadb_store",
        "test_search_results",
        "test_end_to_end_pipeline",
    ]:
        _run("test_vector_store", fn_name)
