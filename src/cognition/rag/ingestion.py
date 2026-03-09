"""
Document Ingestion Module for Supply Chain RAG Pipeline

Handles ingestion of documents from various sources including:
- Web URLs (news articles, reports)
- PDF files
- RSS feeds
- APIs (news services)
- Local text/markdown files
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Types of supply chain documents."""
    NEWS = "news"
    REPORT = "report"
    BEST_PRACTICE = "best_practice"
    CASE_STUDY = "case_study"
    REGULATION = "regulation"
    INTERNAL = "internal"
    RESEARCH = "research"


class DisruptionType(Enum):
    """Types of supply chain disruptions for categorization."""
    STOCKOUT = "stockout"
    DEMAND_SPIKE = "demand_spike"
    SUPPLIER_FAILURE = "supplier_failure"
    LOGISTICS_DELAY = "logistics_delay"
    QUALITY_ISSUE = "quality_issue"
    NATURAL_DISASTER = "natural_disaster"
    GEOPOLITICAL = "geopolitical"
    CYBER_ATTACK = "cyber_attack"
    LABOR_SHORTAGE = "labor_shortage"
    RAW_MATERIAL_SHORTAGE = "raw_material_shortage"
    GENERAL = "general"


class Region(Enum):
    """Geographic regions for metadata."""
    ASIA = "asia"
    EUROPE = "europe"
    AMERICAS = "americas"
    AFRICA = "africa"
    MIDDLE_EAST = "middle_east"
    OCEANIA = "oceania"
    GLOBAL = "global"


@dataclass
class DocumentMetadata:
    """Metadata associated with an ingested document."""
    source: str
    source_type: str  # url, pdf, rss, api, file
    doc_type: DocumentType
    timestamp: datetime
    region: Optional[Region] = None
    industry: Optional[str] = None
    disruption_type: Optional[DisruptionType] = None
    severity: Optional[str] = None  # low, medium, high, critical
    entities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    url: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for storage."""
        return {
            "source": self.source,
            "source_type": self.source_type,
            "doc_type": self.doc_type.value,
            "timestamp": self.timestamp.isoformat(),
            "region": self.region.value if self.region else None,
            "industry": self.industry,
            "disruption_type": self.disruption_type.value if self.disruption_type else None,
            "severity": self.severity,
            "entities": self.entities,
            "tags": self.tags,
            "url": self.url,
            "author": self.author,
            "title": self.title,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Create metadata from dictionary."""
        return cls(
            source=data["source"],
            source_type=data["source_type"],
            doc_type=DocumentType(data["doc_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            region=Region(data["region"]) if data.get("region") else None,
            industry=data.get("industry"),
            disruption_type=DisruptionType(data["disruption_type"]) if data.get("disruption_type") else None,
            severity=data.get("severity"),
            entities=data.get("entities", []),
            tags=data.get("tags", []),
            url=data.get("url"),
            author=data.get("author"),
            title=data.get("title"),
        )


@dataclass
class SupplyChainDocument:
    """A document ingested for the RAG pipeline."""
    id: str
    content: str
    metadata: DocumentMetadata
    
    @classmethod
    def create(
        cls,
        content: str,
        metadata: DocumentMetadata,
    ) -> "SupplyChainDocument":
        """Create a document with auto-generated ID."""
        # Generate deterministic ID from content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        timestamp_str = metadata.timestamp.strftime("%Y%m%d%H%M%S")
        doc_id = f"{metadata.doc_type.value}_{timestamp_str}_{content_hash}"
        
        return cls(id=doc_id, content=content, metadata=metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
        }


class EntityExtractor:
    """Extract supply chain entities from text."""
    
    # Common supply chain keywords for classification
    DISRUPTION_KEYWORDS = {
        DisruptionType.STOCKOUT: ["stockout", "out of stock", "inventory shortage", "empty shelves"],
        DisruptionType.DEMAND_SPIKE: ["demand surge", "demand spike", "panic buying", "hoarding"],
        DisruptionType.SUPPLIER_FAILURE: ["supplier bankruptcy", "supplier failure", "vendor collapse"],
        DisruptionType.LOGISTICS_DELAY: ["shipping delay", "port congestion", "logistics bottleneck", "freight"],
        DisruptionType.QUALITY_ISSUE: ["recall", "quality defect", "contamination", "safety issue"],
        DisruptionType.NATURAL_DISASTER: ["earthquake", "hurricane", "flood", "typhoon", "wildfire"],
        DisruptionType.GEOPOLITICAL: ["tariff", "sanction", "trade war", "embargo", "border"],
        DisruptionType.CYBER_ATTACK: ["ransomware", "cyber attack", "data breach", "hacking"],
        DisruptionType.LABOR_SHORTAGE: ["labor shortage", "strike", "worker shortage", "staffing"],
        DisruptionType.RAW_MATERIAL_SHORTAGE: ["raw material", "commodity shortage", "chip shortage"],
    }
    
    REGION_KEYWORDS = {
        Region.ASIA: ["china", "japan", "korea", "india", "vietnam", "taiwan", "asia", "asian"],
        Region.EUROPE: ["germany", "france", "uk", "britain", "europe", "european", "eu"],
        Region.AMERICAS: ["usa", "united states", "america", "canada", "mexico", "brazil"],
        Region.AFRICA: ["africa", "african", "nigeria", "south africa", "egypt"],
        Region.MIDDLE_EAST: ["middle east", "saudi", "uae", "dubai", "iran", "israel"],
        Region.OCEANIA: ["australia", "new zealand", "oceania"],
    }
    
    INDUSTRY_KEYWORDS = {
        "automotive": ["automotive", "car", "vehicle", "auto", "ev", "electric vehicle"],
        "electronics": ["semiconductor", "chip", "electronics", "tech", "computer"],
        "pharmaceutical": ["pharma", "pharmaceutical", "drug", "medicine", "vaccine"],
        "retail": ["retail", "store", "shopping", "consumer goods"],
        "food": ["food", "agriculture", "grocery", "beverage"],
        "manufacturing": ["manufacturing", "factory", "production", "industrial"],
        "aerospace": ["aerospace", "aviation", "aircraft", "airline"],
        "energy": ["oil", "gas", "energy", "renewable", "solar", "wind"],
    }
    
    def extract_disruption_type(self, text: str) -> Optional[DisruptionType]:
        """Identify the type of disruption mentioned in text."""
        text_lower = text.lower()
        
        for disruption_type, keywords in self.DISRUPTION_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return disruption_type
        
        return DisruptionType.GENERAL
    
    def extract_region(self, text: str) -> Optional[Region]:
        """Identify the geographic region mentioned in text."""
        text_lower = text.lower()
        
        for region, keywords in self.REGION_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return region
        
        return Region.GLOBAL
    
    def extract_industry(self, text: str) -> Optional[str]:
        """Identify the industry mentioned in text."""
        text_lower = text.lower()
        
        for industry, keywords in self.INDUSTRY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return industry
        
        return None
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities (simplified pattern-based extraction)."""
        entities = []
        
        # Extract capitalized phrases (potential company names)
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(pattern, text)
        
        # Filter common words and keep potential entities
        common_words = {
            "The", "This", "That", "These", "There", "When", "Where", "What",
            "How", "Why", "Which", "January", "February", "March", "April",
            "May", "June", "July", "August", "September", "October", "November",
            "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday", "According", "However", "Meanwhile", "Furthermore",
        }
        
        for match in matches:
            if match not in common_words and len(match) > 2:
                entities.append(match)
        
        return list(set(entities))[:20]  # Limit to 20 entities
    
    def assess_severity(self, text: str) -> str:
        """Assess severity of the situation described in text."""
        text_lower = text.lower()
        
        critical_keywords = ["crisis", "catastrophic", "emergency", "shutdown", "halt", "collapse"]
        high_keywords = ["major", "significant", "severe", "disruption", "shortage", "delay"]
        medium_keywords = ["moderate", "concern", "impact", "affected", "challenge"]
        
        if any(kw in text_lower for kw in critical_keywords):
            return "critical"
        elif any(kw in text_lower for kw in high_keywords):
            return "high"
        elif any(kw in text_lower for kw in medium_keywords):
            return "medium"
        
        return "low"


class DocumentIngester:
    """
    Ingest documents from various sources for the RAG pipeline.
    
    Supports:
    - Web URLs (HTML pages)
    - PDF files
    - RSS feeds
    - Plain text / Markdown files
    - Directory scanning
    """
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if optional dependencies are available."""
        self._has_beautifulsoup = False
        self._has_feedparser = False
        self._has_pypdf = False
        
        try:
            import bs4
            self._has_beautifulsoup = True
        except ImportError:
            logger.warning("beautifulsoup4 not installed. URL ingestion will be limited.")
        
        try:
            import feedparser
            self._has_feedparser = True
        except ImportError:
            logger.warning("feedparser not installed. RSS feed ingestion unavailable.")
        
        try:
            import pypdf
            self._has_pypdf = True
        except ImportError:
            logger.warning("pypdf not installed. PDF ingestion unavailable.")
    
    def ingest_text(
        self,
        content: str,
        doc_type: DocumentType,
        source: str = "manual",
        title: Optional[str] = None,
        auto_extract: bool = True,
    ) -> SupplyChainDocument:
        """
        Ingest raw text content.
        
        Args:
            content: The text content to ingest
            doc_type: Type of document
            source: Source identifier
            title: Optional title
            auto_extract: Whether to auto-extract metadata from content
            
        Returns:
            SupplyChainDocument ready for chunking
        """
        metadata = DocumentMetadata(
            source=source,
            source_type="text",
            doc_type=doc_type,
            timestamp=datetime.now(),
            title=title,
        )
        
        if auto_extract:
            metadata.disruption_type = self.entity_extractor.extract_disruption_type(content)
            metadata.region = self.entity_extractor.extract_region(content)
            metadata.industry = self.entity_extractor.extract_industry(content)
            metadata.entities = self.entity_extractor.extract_entities(content)
            metadata.severity = self.entity_extractor.assess_severity(content)
        
        return SupplyChainDocument.create(content=content, metadata=metadata)
    
    def ingest_url(
        self,
        url: str,
        doc_type: DocumentType,
        auto_extract: bool = True,
    ) -> Optional[SupplyChainDocument]:
        """
        Ingest content from a web URL.
        
        Args:
            url: The URL to fetch and ingest
            doc_type: Type of document
            auto_extract: Whether to auto-extract metadata
            
        Returns:
            SupplyChainDocument or None if ingestion fails
        """
        if not self._has_beautifulsoup:
            logger.error("beautifulsoup4 required for URL ingestion")
            return None
        
        try:
            import requests
            from bs4 import BeautifulSoup
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else None
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text(separator="\n", strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            content = "\n".join(lines)
            
            if not content:
                logger.warning(f"No content extracted from {url}")
                return None
            
            metadata = DocumentMetadata(
                source=urlparse(url).netloc,
                source_type="url",
                doc_type=doc_type,
                timestamp=datetime.now(),
                url=url,
                title=title,
            )
            
            if auto_extract:
                metadata.disruption_type = self.entity_extractor.extract_disruption_type(content)
                metadata.region = self.entity_extractor.extract_region(content)
                metadata.industry = self.entity_extractor.extract_industry(content)
                metadata.entities = self.entity_extractor.extract_entities(content)
                metadata.severity = self.entity_extractor.assess_severity(content)
            
            logger.info(f"Successfully ingested URL: {url}")
            return SupplyChainDocument.create(content=content, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Failed to ingest URL {url}: {e}")
            return None
    
    def ingest_pdf(
        self,
        path: str,
        doc_type: DocumentType,
        auto_extract: bool = True,
    ) -> Optional[SupplyChainDocument]:
        """
        Ingest content from a PDF file.
        
        Args:
            path: Path to the PDF file
            doc_type: Type of document
            auto_extract: Whether to auto-extract metadata
            
        Returns:
            SupplyChainDocument or None if ingestion fails
        """
        if not self._has_pypdf:
            logger.error("pypdf required for PDF ingestion")
            return None
        
        try:
            from pypdf import PdfReader
            
            file_path = Path(path)
            if not file_path.exists():
                logger.error(f"PDF file not found: {path}")
                return None
            
            reader = PdfReader(file_path)
            
            # Extract text from all pages
            content_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content_parts.append(text)
            
            content = "\n\n".join(content_parts)
            
            if not content:
                logger.warning(f"No content extracted from PDF: {path}")
                return None
            
            # Get PDF metadata
            pdf_metadata = reader.metadata or {}
            title = pdf_metadata.get("/Title", file_path.stem)
            author = pdf_metadata.get("/Author")
            
            metadata = DocumentMetadata(
                source=file_path.name,
                source_type="pdf",
                doc_type=doc_type,
                timestamp=datetime.now(),
                title=title,
                author=author,
            )
            
            if auto_extract:
                metadata.disruption_type = self.entity_extractor.extract_disruption_type(content)
                metadata.region = self.entity_extractor.extract_region(content)
                metadata.industry = self.entity_extractor.extract_industry(content)
                metadata.entities = self.entity_extractor.extract_entities(content)
                metadata.severity = self.entity_extractor.assess_severity(content)
            
            logger.info(f"Successfully ingested PDF: {path}")
            return SupplyChainDocument.create(content=content, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Failed to ingest PDF {path}: {e}")
            return None
    
    def ingest_rss_feed(
        self,
        feed_url: str,
        doc_type: DocumentType = DocumentType.NEWS,
        max_entries: int = 20,
        auto_extract: bool = True,
    ) -> List[SupplyChainDocument]:
        """
        Ingest articles from an RSS feed.
        
        Args:
            feed_url: URL of the RSS feed
            doc_type: Type of documents (default: NEWS)
            max_entries: Maximum number of entries to fetch
            auto_extract: Whether to auto-extract metadata
            
        Returns:
            List of SupplyChainDocuments
        """
        if not self._has_feedparser:
            logger.error("feedparser required for RSS feed ingestion")
            return []
        
        try:
            import feedparser
            
            feed = feedparser.parse(feed_url)
            
            if feed.bozo:
                logger.warning(f"RSS feed parsing warning: {feed.bozo_exception}")
            
            documents = []
            
            for entry in feed.entries[:max_entries]:
                # Get content
                content = ""
                if hasattr(entry, 'content'):
                    content = entry.content[0].value if entry.content else ""
                elif hasattr(entry, 'summary'):
                    content = entry.summary
                elif hasattr(entry, 'description'):
                    content = entry.description
                
                # Strip HTML tags if present
                if content and self._has_beautifulsoup:
                    from bs4 import BeautifulSoup
                    content = BeautifulSoup(content, 'html.parser').get_text()
                
                if not content:
                    continue
                
                # Parse timestamp
                timestamp = datetime.now()
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try:
                        timestamp = datetime(*entry.published_parsed[:6])
                    except:
                        pass
                
                title = getattr(entry, 'title', None)
                link = getattr(entry, 'link', None)
                author = getattr(entry, 'author', None)
                
                metadata = DocumentMetadata(
                    source=urlparse(feed_url).netloc,
                    source_type="rss",
                    doc_type=doc_type,
                    timestamp=timestamp,
                    url=link,
                    title=title,
                    author=author,
                )
                
                if auto_extract:
                    full_text = f"{title or ''} {content}"
                    metadata.disruption_type = self.entity_extractor.extract_disruption_type(full_text)
                    metadata.region = self.entity_extractor.extract_region(full_text)
                    metadata.industry = self.entity_extractor.extract_industry(full_text)
                    metadata.entities = self.entity_extractor.extract_entities(full_text)
                    metadata.severity = self.entity_extractor.assess_severity(full_text)
                
                doc = SupplyChainDocument.create(content=content, metadata=metadata)
                documents.append(doc)
            
            logger.info(f"Successfully ingested {len(documents)} entries from RSS feed: {feed_url}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to ingest RSS feed {feed_url}: {e}")
            return []
    
    def ingest_file(
        self,
        path: str,
        doc_type: DocumentType,
        auto_extract: bool = True,
    ) -> Optional[SupplyChainDocument]:
        """
        Ingest a local text or markdown file.
        
        Args:
            path: Path to the file
            doc_type: Type of document
            auto_extract: Whether to auto-extract metadata
            
        Returns:
            SupplyChainDocument or None if ingestion fails
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                logger.error(f"File not found: {path}")
                return None
            
            content = file_path.read_text(encoding="utf-8")
            
            if not content:
                logger.warning(f"Empty file: {path}")
                return None
            
            metadata = DocumentMetadata(
                source=file_path.name,
                source_type="file",
                doc_type=doc_type,
                timestamp=datetime.fromtimestamp(file_path.stat().st_mtime),
                title=file_path.stem,
            )
            
            if auto_extract:
                metadata.disruption_type = self.entity_extractor.extract_disruption_type(content)
                metadata.region = self.entity_extractor.extract_region(content)
                metadata.industry = self.entity_extractor.extract_industry(content)
                metadata.entities = self.entity_extractor.extract_entities(content)
                metadata.severity = self.entity_extractor.assess_severity(content)
            
            logger.info(f"Successfully ingested file: {path}")
            return SupplyChainDocument.create(content=content, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Failed to ingest file {path}: {e}")
            return None
    
    def ingest_directory(
        self,
        directory: str,
        doc_type: DocumentType,
        extensions: List[str] = None,
        recursive: bool = True,
        auto_extract: bool = True,
    ) -> List[SupplyChainDocument]:
        """
        Ingest all matching files from a directory.
        
        Args:
            directory: Path to directory
            doc_type: Type of documents
            extensions: File extensions to process (default: .txt, .md)
            recursive: Whether to search subdirectories
            auto_extract: Whether to auto-extract metadata
            
        Returns:
            List of SupplyChainDocuments
        """
        if extensions is None:
            extensions = [".txt", ".md", ".markdown"]
        
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return []
        
        documents = []
        pattern = "**/*" if recursive else "*"
        
        for ext in extensions:
            for file_path in dir_path.glob(f"{pattern}{ext}"):
                if file_path.is_file():
                    doc = self.ingest_file(str(file_path), doc_type, auto_extract)
                    if doc:
                        documents.append(doc)
        
        # Also try to find PDFs
        if self._has_pypdf:
            for file_path in dir_path.glob(f"{pattern}.pdf"):
                if file_path.is_file():
                    doc = self.ingest_pdf(str(file_path), doc_type, auto_extract)
                    if doc:
                        documents.append(doc)
        
        logger.info(f"Ingested {len(documents)} files from directory: {directory}")
        return documents


# =============================================================================
# Convenience Functions
# =============================================================================

def ingest_from_url(url: str, doc_type: DocumentType = DocumentType.NEWS) -> Optional[SupplyChainDocument]:
    """Convenience function to ingest a single URL."""
    ingester = DocumentIngester()
    return ingester.ingest_url(url, doc_type)


def ingest_from_pdf(path: str, doc_type: DocumentType = DocumentType.REPORT) -> Optional[SupplyChainDocument]:
    """Convenience function to ingest a single PDF."""
    ingester = DocumentIngester()
    return ingester.ingest_pdf(path, doc_type)


def ingest_from_text(
    content: str,
    doc_type: DocumentType = DocumentType.INTERNAL,
    title: Optional[str] = None,
) -> SupplyChainDocument:
    """Convenience function to ingest raw text."""
    ingester = DocumentIngester()
    return ingester.ingest_text(content, doc_type, title=title)
