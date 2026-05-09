"""
Build a persisted RAG vector store from a local source directory.

Usage:
    python scripts/build_rag_store.py --source-dir "RAG source" --doc-type report
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cognition.rag import ChromaVectorStore  # noqa: E402
from src.cognition.rag import (DocumentIngester, DocumentType,
                               SupplyChainChunker, SupplyChainEmbeddings)

logger = logging.getLogger(__name__)

DOC_TYPE_MAP = {
    "news": DocumentType.NEWS,
    "report": DocumentType.REPORT,
    "best_practice": DocumentType.BEST_PRACTICE,
    "case_study": DocumentType.CASE_STUDY,
    "regulation": DocumentType.REGULATION,
    "internal": DocumentType.INTERNAL,
    "research": DocumentType.RESEARCH,
}


def _parse_doc_type(value: str) -> DocumentType:
    key = str(value or "").strip().lower()
    if key not in DOC_TYPE_MAP:
        valid = ", ".join(sorted(DOC_TYPE_MAP.keys()))
        raise ValueError(f"Unknown doc type '{value}'. Valid: {valid}")
    return DOC_TYPE_MAP[key]


def build_vector_store(
    source_dir: Path,
    persist_dir: Path,
    doc_type: DocumentType,
    provider: str,
    device: str,
    use_cache: bool,
) -> int:
    ingester = DocumentIngester()
    docs = ingester.ingest_directory(
        directory=str(source_dir),
        doc_type=doc_type,
        recursive=True,
        auto_extract=True,
    )

    if not docs:
        logger.error("No documents ingested from %s", source_dir)
        return 0

    chunker = SupplyChainChunker()
    embeddings = SupplyChainEmbeddings(
        provider=provider,
        device=device,
        use_cache=use_cache,
    )
    vector_store = ChromaVectorStore(
        persist_directory=str(persist_dir),
        embedding_model=embeddings.model,
    )

    total_chunks = 0
    for doc in docs:
        chunks = chunker.chunk_document(doc)
        if not chunks:
            logger.warning("No chunks generated for doc %s", doc.id)
            continue
        total_chunks += vector_store.add_document(doc, chunks)

    logger.info("Ingested %d docs and stored %d chunks", len(docs), total_chunks)
    return total_chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RAG vector store from local docs")
    parser.add_argument("--source-dir", required=True, help="Folder with PDFs/TXT/MD sources")
    parser.add_argument(
        "--persist-dir",
        default=str(Path("data") / "vectorstore"),
        help="Persisted ChromaDB directory",
    )
    parser.add_argument(
        "--doc-type",
        default="report",
        help="Document type: news|report|best_practice|case_study|regulation|internal|research",
    )
    parser.add_argument(
        "--provider",
        default="sentence-transformers",
        help="Embedding provider: sentence-transformers|huggingface|openai|mock",
    )
    parser.add_argument("--device", default="cpu", help="Embedding device: cpu|cuda")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache",
    )

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")

    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    doc_type = _parse_doc_type(args.doc_type)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    total_chunks = build_vector_store(
        source_dir=source_dir,
        persist_dir=persist_dir,
        doc_type=doc_type,
        provider=args.provider,
        device=args.device,
        use_cache=not args.no_cache,
    )

    if total_chunks == 0:
        logger.error("No chunks stored. Check source files and dependencies.")
        return 1

    logger.info("Vector store ready at %s", persist_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
