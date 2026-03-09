"""
Smart Chunking Module for Supply Chain RAG Pipeline

Implements domain-aware chunking strategies that preserve
semantic coherence for supply chain documents.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ingestion import DocumentMetadata, DocumentType, SupplyChainDocument

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of text ready for embedding."""
    id: str
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    chunk_index: int
    total_chunks: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }


class ChunkingStrategy:
    """Base class for chunking strategies."""
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """Split text into chunks. Override in subclasses."""
        raise NotImplementedError


class FixedSizeChunker(ChunkingStrategy):
    """
    Fixed-size chunking with overlap.
    
    Good for general-purpose text where structure doesn't matter.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[str]:
        """Split text into fixed-size chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in ['. ', '.\n', '? ', '?\n', '! ', '!\n']:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start < 0:
                start = end
        
        return chunks


class ParagraphChunker(ChunkingStrategy):
    """
    Paragraph-based chunking.
    
    Good for news articles and reports where paragraphs are meaningful units.
    """
    
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 1000):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[str]:
        """Split text by paragraphs, merging small ones."""
        # Split by double newlines or multiple newlines
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return [text] if text.strip() else []
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If paragraph itself is too long, use fixed chunking
            if len(para) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split long paragraph
                sub_chunker = FixedSizeChunker(
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=50
                )
                chunks.extend(sub_chunker.chunk(para))
                continue
            
            # Try to add to current chunk
            potential = f"{current_chunk}\n\n{para}" if current_chunk else para
            
            if len(potential) <= self.max_chunk_size:
                current_chunk = potential
            else:
                # Save current chunk and start new one
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        elif current_chunk and chunks:
            # Append to last chunk if too small
            chunks[-1] = f"{chunks[-1]}\n\n{current_chunk}".strip()
        elif current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


class SectionChunker(ChunkingStrategy):
    """
    Section-based chunking using headers.
    
    Good for structured documents like reports and documentation.
    """
    
    def __init__(self, max_chunk_size: int = 1500):
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[str]:
        """Split text by section headers."""
        # Pattern for markdown headers or numbered sections
        header_pattern = r'^(?:#{1,6}\s+.+|(?:\d+\.)+\s+[A-Z].+|[A-Z][A-Z\s]+:)$'
        
        lines = text.split('\n')
        sections = []
        current_section = []
        current_header = ""
        
        for line in lines:
            if re.match(header_pattern, line.strip(), re.MULTILINE):
                if current_section:
                    section_text = '\n'.join(current_section).strip()
                    if section_text:
                        if current_header:
                            section_text = f"{current_header}\n{section_text}"
                        sections.append(section_text)
                current_header = line.strip()
                current_section = []
            else:
                current_section.append(line)
        
        # Last section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                if current_header:
                    section_text = f"{current_header}\n{section_text}"
                sections.append(section_text)
        
        # Chunk large sections
        chunks = []
        for section in sections:
            if len(section) > self.max_chunk_size:
                para_chunker = ParagraphChunker(max_chunk_size=self.max_chunk_size)
                chunks.extend(para_chunker.chunk(section))
            else:
                chunks.append(section)
        
        return chunks if chunks else [text]  # Fallback to whole text


class EventBasedChunker(ChunkingStrategy):
    """
    Event-based chunking for case studies.
    
    Preserves cause-effect relationships by chunking around event narratives.
    """
    
    EVENT_MARKERS = [
        r'(?i)as a result',
        r'(?i)consequently',
        r'(?i)this led to',
        r'(?i)the impact was',
        r'(?i)in response',
        r'(?i)the outcome',
        r'(?i)following this',
        r'(?i)due to this',
        r'(?i)because of',
        r'(?i)which caused',
    ]
    
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[str]:
        """Split text by event markers while preserving cause-effect context."""
        # First try paragraph chunking as base
        para_chunker = ParagraphChunker(max_chunk_size=self.max_chunk_size)
        paragraphs = para_chunker.chunk(text)
        
        # Group paragraphs that contain event markers with previous paragraph
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # Check if paragraph starts with event marker
            starts_with_marker = any(
                re.match(marker, para.strip())
                for marker in self.EVENT_MARKERS
            )
            
            if starts_with_marker and current_chunk:
                # Append to current chunk (this is the effect/result)
                potential = f"{current_chunk}\n\n{para}"
                if len(potential) <= self.max_chunk_size * 1.5:
                    current_chunk = potential
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


class SupplyChainChunker:
    """
    Domain-aware chunker that selects strategy based on document type.
    
    Strategies:
    - NEWS: Paragraph-based (preserve article structure)
    - REPORT: Section-based (respect headers/subheaders)
    - CASE_STUDY: Event-based (preserve cause-effect relationships)
    - BEST_PRACTICE: Section or paragraph-based
    - REGULATION: Section-based (preserve legal structure)
    - INTERNAL/RESEARCH: Section-based
    """
    
    def __init__(
        self,
        default_chunk_size: int = 512,
        default_overlap: int = 50,
    ):
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
        
        # Strategy instances
        self.strategies = {
            DocumentType.NEWS: ParagraphChunker(min_chunk_size=100, max_chunk_size=800),
            DocumentType.REPORT: SectionChunker(max_chunk_size=1200),
            DocumentType.CASE_STUDY: EventBasedChunker(max_chunk_size=1000),
            DocumentType.BEST_PRACTICE: SectionChunker(max_chunk_size=1000),
            DocumentType.REGULATION: SectionChunker(max_chunk_size=1000),
            DocumentType.INTERNAL: ParagraphChunker(min_chunk_size=100, max_chunk_size=800),
            DocumentType.RESEARCH: SectionChunker(max_chunk_size=1500),
        }
        
        self.default_strategy = FixedSizeChunker(
            chunk_size=default_chunk_size,
            chunk_overlap=default_overlap
        )
    
    def chunk_document(self, document: SupplyChainDocument) -> List[Chunk]:
        """
        Chunk a document using the appropriate strategy.
        
        Args:
            document: The document to chunk
            
        Returns:
            List of Chunk objects
        """
        doc_type = document.metadata.doc_type
        strategy = self.strategies.get(doc_type, self.default_strategy)
        
        # Get text chunks
        text_chunks = strategy.chunk(document.content, document.metadata.to_dict())
        
        if not text_chunks:
            return []
        
        # Create Chunk objects with metadata
        chunks = []
        base_metadata = document.metadata.to_dict()
        
        for i, text in enumerate(text_chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(text_chunks)
            
            chunk = Chunk(
                id=f"{document.id}_chunk_{i}",
                content=text,
                metadata=chunk_metadata,
                doc_id=document.id,
                chunk_index=i,
                total_chunks=len(text_chunks),
            )
            chunks.append(chunk)
        
        logger.debug(f"Chunked document {document.id} into {len(chunks)} chunks")
        return chunks
    
    def chunk_text(
        self,
        text: str,
        doc_type: DocumentType = DocumentType.INTERNAL,
        doc_id: str = "unknown",
    ) -> List[Chunk]:
        """
        Chunk raw text using the appropriate strategy.
        
        Args:
            text: Text to chunk
            doc_type: Type of document
            doc_id: Document ID for chunk IDs
            
        Returns:
            List of Chunk objects
        """
        strategy = self.strategies.get(doc_type, self.default_strategy)
        text_chunks = strategy.chunk(text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = Chunk(
                id=f"{doc_id}_chunk_{i}",
                content=chunk_text,
                metadata={"doc_type": doc_type.value},
                doc_id=doc_id,
                chunk_index=i,
                total_chunks=len(text_chunks),
            )
            chunks.append(chunk)
        
        return chunks


class RecursiveChunker:
    """
    Recursive character text splitter for fallback.
    
    Uses multiple separators to find best split points.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if not separator:
            # Character-level split as last resort
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
        
        splits = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            test_chunk = f"{current_chunk}{separator}{split}" if current_chunk else split
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if len(split) > self.chunk_size:
                    # Recursively split with next separator
                    sub_chunks = self._split_text(split, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def chunk(self, text: str) -> List[str]:
        """Split text recursively."""
        return self._split_text(text, self.separators)


# =============================================================================
# Convenience Functions
# =============================================================================

def chunk_document(document: SupplyChainDocument) -> List[Chunk]:
    """Convenience function to chunk a document."""
    chunker = SupplyChainChunker()
    return chunker.chunk_document(document)


def chunk_text(
    text: str,
    doc_type: DocumentType = DocumentType.INTERNAL,
    chunk_size: int = 512,
) -> List[Chunk]:
    """Convenience function to chunk raw text."""
    chunker = SupplyChainChunker(default_chunk_size=chunk_size)
    return chunker.chunk_text(text, doc_type)
