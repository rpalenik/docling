#!/usr/bin/env python3
"""
Annex Processor Module

Handles detection, downloading, conversion and integration of external PDF annexes
for Slovak law documents from Slov-Lex.

Usage:
    from annex_processor import AnnexProcessor, AnnexInfo
    
    processor = AnnexProcessor(cache_dir="cache")
    annexes = processor.detect_annexes_from_html("input/zakon.html")
    for annex in annexes:
        if annex.type == "external_pdf":
            doc = processor.load_or_convert_annex(annex)
"""

import json
import re
import time
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument, TextItem, ListGroup, InlineGroup
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingSerializerProvider


@dataclass
class AnnexInfo:
    """
    Information about a law annex (príloha).
    
    Attributes:
        number: Annex number (e.g., "1", "2", "3")
        title: Full title (e.g., "Príloha č. 1 k zákonu č. 595/2003 Z. z.")
        type: "external_pdf" for PDF annexes, "inline" for annexes embedded in HTML
        url: URL for external PDF (relative or absolute)
        local_path: Local path to downloaded PDF file
        docling_path: Path to cached Docling JSON conversion
        content_type: "table", "text", or "mixed"
        html_id: HTML element ID for this annex
        parser: Parser to use ("docling", "hybrid", "camelot", "pdfplumber")
        parser_options: Parser-specific options (e.g., {"reconstruct_tables": true})
    """
    number: str
    title: str
    type: str  # "external_pdf" | "inline"
    url: Optional[str] = None
    local_path: Optional[Path] = None
    docling_path: Optional[Path] = None
    content_type: str = "mixed"  # "table" | "text" | "mixed"
    html_id: Optional[str] = None
    parser: str = "docling"  # "docling" | "hybrid" | "camelot" | "pdfplumber"
    parser_options: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, base_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Convert to dictionary for YAML/JSON serialization.
        
        Args:
            base_dir: Base directory for relative paths. If provided, paths are made relative.
        """
        result = asdict(self)
        # Convert Path objects to strings (relative if base_dir provided)
        if result.get('local_path'):
            path = Path(result['local_path'])
            if base_dir and path.is_absolute():
                try:
                    result['filename'] = str(path.relative_to(base_dir))
                except ValueError:
                    result['filename'] = str(path)
            else:
                result['filename'] = str(path)
            del result['local_path']  # Use 'filename' in YAML
        
        if result.get('docling_path'):
            path = Path(result['docling_path'])
            if base_dir and path.is_absolute():
                try:
                    result['docling_cache'] = str(path.relative_to(base_dir))
                except ValueError:
                    result['docling_cache'] = str(path)
            else:
                result['docling_cache'] = str(path)
            del result['docling_path']  # Use 'docling_cache' in YAML
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Optional[Path] = None) -> 'AnnexInfo':
        """
        Create AnnexInfo from dictionary.
        
        Args:
            data: Dictionary with annex data
            base_dir: Base directory for resolving relative paths
        """
        data = data.copy()
        
        # Handle filename (YAML) or local_path (legacy) - resolve relative to base_dir
        if data.get('filename'):
            path_str = data['filename']
            if base_dir and not Path(path_str).is_absolute():
                data['local_path'] = base_dir / path_str
            else:
                data['local_path'] = Path(path_str)
            del data['filename']
        elif data.get('local_path'):
            path_str = data['local_path']
            if base_dir and not Path(path_str).is_absolute():
                data['local_path'] = base_dir / path_str
            else:
                data['local_path'] = Path(path_str)
        
        # Handle docling_cache (YAML) or docling_path (legacy)
        if data.get('docling_cache'):
            path_str = data['docling_cache']
            if base_dir and not Path(path_str).is_absolute():
                data['docling_path'] = base_dir / path_str
            else:
                data['docling_path'] = Path(path_str)
            del data['docling_cache']
        elif data.get('docling_path'):
            path_str = data['docling_path']
            if base_dir and not Path(path_str).is_absolute():
                data['docling_path'] = base_dir / path_str
            else:
                data['docling_path'] = Path(path_str)
        
        return cls(**data)


@dataclass
class LawManifest:
    """
    Manifest file for a law document with its annexes.
    
    Attributes:
        law_id: Law identifier (e.g., "595/2003")
        source: Source system (e.g., "slov-lex")
        main_document: Information about main document
        annexes: List of annex information
    """
    law_id: str
    source: str = "slov-lex"
    main_document: Dict[str, str] = field(default_factory=dict)
    annexes: List[AnnexInfo] = field(default_factory=list)
    
    def to_dict(self, base_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Convert to dictionary for YAML serialization.
        
        Args:
            base_dir: Base directory for relative paths. If provided, paths are made relative.
        """
        # Convert main_document paths to relative if base_dir provided
        main_doc = self.main_document.copy()
        if base_dir and 'path' in main_doc:
            try:
                main_doc['filename'] = str(Path(main_doc['path']).relative_to(base_dir))
                del main_doc['path']
            except (ValueError, KeyError):
                pass
        
        return {
            "law_id": self.law_id,
            "source": self.source,
            "main_document": main_doc,
            "annexes": [a.to_dict(base_dir) for a in self.annexes]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Optional[Path] = None) -> 'LawManifest':
        """
        Create LawManifest from dictionary.
        
        Args:
            data: Dictionary with manifest data
            base_dir: Base directory for resolving relative paths
        """
        # Handle main_document - convert filename to path if needed
        main_doc = data.get('main_document', {}).copy()
        if 'filename' in main_doc and base_dir:
            main_doc['path'] = str(base_dir / main_doc['filename'])
        
        annexes = [AnnexInfo.from_dict(a, base_dir) for a in data.get('annexes', [])]
        return cls(
            law_id=data['law_id'],
            source=data.get('source', 'slov-lex'),
            main_document=main_doc,
            annexes=annexes
        )
    
    def save(self, path: Path, base_dir: Optional[Path] = None) -> None:
        """
        Save manifest to YAML file.
        
        Args:
            path: Path to save manifest.yaml
            base_dir: Base directory for relative paths (usually law_dir)
        """
        if base_dir is None:
            base_dir = path.parent
        
        data = self.to_dict(base_dir)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, path: Path) -> 'LawManifest':
        """
        Load manifest from YAML file.
        
        Args:
            path: Path to manifest.yaml file
            
        Returns:
            Loaded LawManifest
        """
        base_dir = path.parent
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Convert relative paths to absolute
        manifest = cls.from_dict(data, base_dir)
        return manifest


class AnnexProcessor:
    """
    Processor for detecting, converting and integrating law annexes.
    
    Supports:
    - Detection of external PDF annexes from HTML
    - Detection of inline annexes from Docling documents
    - Caching of Docling conversions
    - Integration of annex content into output structure
    """
    
    # Regex patterns for annex detection
    ANNEX_NUMBER_PATTERN = re.compile(r'Príloha\s+č\.\s*(\d+)', re.IGNORECASE)
    SLOV_LEX_BASE_URL = "https://www.slov-lex.sk"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize AnnexProcessor.
        
        Args:
            cache_dir: Directory for caching Docling JSON conversions
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache")
        self._converter = None  # Lazy initialization
    
    @property
    def converter(self) -> DocumentConverter:
        """Lazy initialization of DocumentConverter."""
        if self._converter is None:
            self._converter = DocumentConverter()
        return self._converter
    
    def detect_annexes_from_html(self, html_path: Path) -> List[AnnexInfo]:
        """
        Detect annexes from HTML file.
        
        Parses HTML looking for:
        - External PDF annexes: <a class="predpis-pdf-priloha">
        - Inline annexes: <div class="priloha"> without PDF link
        
        Args:
            html_path: Path to HTML file
            
        Returns:
            List of detected AnnexInfo objects
        """
        html_path = Path(html_path)
        
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        annexes = []
        
        # Find all annex divs
        annex_divs = soup.find_all('div', class_='priloha')
        
        for div in annex_divs:
            # Get annex title from prilohaOznacenie
            title_div = div.find('div', class_='prilohaOznacenie')
            if not title_div:
                continue
            
            title = title_div.get_text(strip=True)
            html_id = div.get('id', '')
            
            # Extract annex number
            match = self.ANNEX_NUMBER_PATTERN.search(title)
            if not match:
                continue
            
            number = match.group(1)
            
            # Check for external PDF link
            pdf_link = div.find('a', class_='predpis-pdf-priloha')
            
            if pdf_link:
                # External PDF annex
                url = pdf_link.get('href', '')
                annex = AnnexInfo(
                    number=number,
                    title=title,
                    type="external_pdf",
                    url=url,
                    html_id=html_id,
                    content_type="table"  # Most PDF annexes are tables
                )
            else:
                # Inline annex
                annex = AnnexInfo(
                    number=number,
                    title=title,
                    type="inline",
                    html_id=html_id,
                    content_type="text"
                )
            
            annexes.append(annex)
        
        return sorted(annexes, key=lambda a: int(a.number))
    
    def detect_annexes_from_docling(self, doc: DoclingDocument) -> List[AnnexInfo]:
        """
        Detect annexes from Docling document.
        
        Looks for text elements starting with "Príloha č."
        
        Args:
            doc: DoclingDocument to analyze
            
        Returns:
            List of detected AnnexInfo objects
        """
        annexes = []
        seen_numbers = set()
        
        for text_elem in doc.texts:
            text = getattr(text_elem, 'text', '') or ''
            
            # Look for annex markers
            match = self.ANNEX_NUMBER_PATTERN.match(text.strip())
            if match:
                number = match.group(1)
                
                # Avoid duplicates
                if number in seen_numbers:
                    continue
                seen_numbers.add(number)
                
                # Check if this is a PDF link reference
                is_pdf_link = 'Prevziať prílohu' in text or '.pdf' in text.lower()
                
                annex = AnnexInfo(
                    number=number,
                    title=text.strip(),
                    type="external_pdf" if is_pdf_link else "inline",
                    content_type="mixed"
                )
                annexes.append(annex)
        
        return sorted(annexes, key=lambda a: int(a.number))
    
    def get_cache_path(self, law_dir: Path, annex_number: str) -> Path:
        """
        Get cache path for an annex's Docling JSON (relative to law_dir).
        
        Args:
            law_dir: Law directory (where cache/ subdirectory will be)
            annex_number: Annex number (e.g., "1")
            
        Returns:
            Path to cache file (relative to law_dir: cache/priloha_{number}_docling.json)
        """
        cache_dir = law_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"priloha_{annex_number}_docling.json"
    
    def is_cached(self, annex: AnnexInfo, law_dir: Path) -> bool:
        """
        Check if annex conversion is cached.
        
        Args:
            annex: AnnexInfo to check
            law_dir: Law directory
            
        Returns:
            True if cached conversion exists
        """
        if annex.docling_path and annex.docling_path.exists():
            return True
        
        cache_path = self.get_cache_path(law_dir, annex.number)
        return cache_path.exists()
    
    def load_from_cache(self, annex: AnnexInfo, law_dir: Path) -> Optional[DoclingDocument]:
        """
        Load annex conversion from cache.
        
        Args:
            annex: AnnexInfo to load
            law_dir: Law directory
            
        Returns:
            DoclingDocument if cached, None otherwise
        """
        cache_path = annex.docling_path or self.get_cache_path(law_dir, annex.number)
        
        if not cache_path.exists():
            return None
        
        try:
            return DoclingDocument.load_from_json(str(cache_path))
        except Exception as e:
            print(f"Warning: Failed to load cache for annex {annex.number}: {e}")
            return None
    
    def save_to_cache(self, doc: DoclingDocument, annex: AnnexInfo, law_dir: Path) -> Path:
        """
        Save annex conversion to cache.
        
        Args:
            doc: DoclingDocument to cache
            annex: AnnexInfo being cached
            law_dir: Law directory
            
        Returns:
            Path to saved cache file
        """
        cache_path = self.get_cache_path(law_dir, annex.number)
        doc.save_as_json(str(cache_path))
        annex.docling_path = cache_path
        return cache_path
    
    def convert_annex(self, pdf_path: Path) -> DoclingDocument:
        """
        Convert PDF annex to DoclingDocument.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Converted DoclingDocument
        """
        print(f"  Converting {pdf_path.name}...")
        start_time = time.time()
        
        result = self.converter.convert(str(pdf_path))
        doc = result.document
        
        elapsed = time.time() - start_time
        print(f"  ✓ Converted in {elapsed:.2f}s ({len(doc.texts)} texts, {len(doc.tables)} tables)")
        
        return doc
    
    def load_or_convert_annex(
        self, 
        annex: AnnexInfo, 
        law_dir: Path,
        annexes_dir: Optional[Path] = None
    ) -> Optional[DoclingDocument]:
        """
        Load annex from cache or convert from PDF.
        
        Args:
            annex: AnnexInfo to process
            law_dir: Law directory (where cache/ is located)
            annexes_dir: Directory containing PDF annexes (defaults to law_dir/annexes)
            
        Returns:
            DoclingDocument or None if not available
        """
        if annexes_dir is None:
            annexes_dir = law_dir / "annexes"
        
        # Try cache first
        cached_doc = self.load_from_cache(annex, law_dir)
        if cached_doc:
            print(f"  Loaded annex {annex.number} from cache")
            return cached_doc
        
        # Need to convert - find PDF
        pdf_path = None
        
        if annex.local_path and annex.local_path.exists():
            pdf_path = annex.local_path
        elif annexes_dir:
            # Try to find PDF in annexes directory with various naming patterns
            patterns = [
                f"*priloha*{annex.number}*.pdf",
                f"*príloha*{annex.number}*.pdf",
                f"*priloha_{annex.number}*.pdf",
                f"*príloha_{annex.number}*.pdf",
                f"*annex*{annex.number}*.pdf",
                f"*Priloha*{annex.number}*.pdf",
                f"*Príloha*{annex.number}*.pdf",
            ]
            
            for pattern in patterns:
                matches = list(annexes_dir.glob(pattern))
                if matches:
                    pdf_path = matches[0]
                    break
            
            # If still not found, try flexible search with Unicode normalization
            if not pdf_path:
                import unicodedata
                
                for file in annexes_dir.iterdir():
                    if file.suffix.lower() == '.pdf':
                        name = file.name
                        # Normalize Unicode to NFC form (important for macOS which uses NFD)
                        name_normalized = unicodedata.normalize('NFC', name).lower()
                        
                        # Check for annex number in various formats
                        # Pattern: priloha/príloha followed by separator and number
                        patterns_to_check = [
                            rf'pr[ií]loha[_\s-]*{annex.number}[^0-9]',
                            rf'pr[ií]loha[_\s-]*{annex.number}$',
                            rf'pr[ií]loha[_\s-]*{annex.number}\.',
                        ]
                        
                        for pattern in patterns_to_check:
                            if re.search(pattern, name_normalized):
                                pdf_path = file
                                break
                        
                        if pdf_path:
                            break
        
        if not pdf_path or not pdf_path.exists():
            print(f"  Warning: PDF not found for annex {annex.number}")
            return None
        
        # Convert and cache
        doc = self.convert_annex(pdf_path)
        self.save_to_cache(doc, annex, law_dir)
        annex.local_path = pdf_path
        
        return doc
    
    def extract_annex_content(
        self, 
        doc: DoclingDocument, 
        annex: AnnexInfo
    ) -> Dict[str, Any]:
        """
        Extract structured content from annex DoclingDocument.
        
        Args:
            doc: DoclingDocument of the annex
            annex: AnnexInfo for metadata
            
        Returns:
            Dictionary with extracted content in new structure format
        """
        from sequential_parser import format_table_for_json
        
        def decode_character_codes(text: str) -> str:
            """
            Decode character codes like /c90/c65... to actual text.
            Pattern: /c followed by decimal number represents Unicode code point.
            Includes Windows-1250 mapping for Slovak characters.
            """
            if not text or '/c' not in text:
                return text
            
            # Windows-1250 to Unicode mapping for Slovak characters
            win1250 = {
                138: 'Š', 154: 'š', 142: 'Ž', 158: 'ž', 141: 'Ť', 157: 'ť',
                188: 'Ľ', 190: 'ľ', 200: 'Č', 232: 'č', 207: 'Ď', 239: 'ď',
                210: 'Ň', 242: 'ň', 192: 'Ŕ', 224: 'ŕ', 197: 'Ĺ', 229: 'ĺ',
                193: 'Á', 225: 'á', 201: 'É', 233: 'é', 205: 'Í', 237: 'í',
                211: 'Ó', 243: 'ó', 218: 'Ú', 250: 'ú', 221: 'Ý', 253: 'ý',
                196: 'Ä', 228: 'ä', 212: 'Ô', 244: 'ô',
            }
            
            def replace_code(match):
                code_str = match.group(1)
                try:
                    code_point = int(code_str)
                    if code_point in win1250:
                        return win1250[code_point]
                    return chr(code_point)
                except (ValueError, OverflowError):
                    return match.group(0)  # Return original if can't decode
            
            # Replace /c followed by numbers with decoded character
            decoded = re.sub(r'/c(\d+)', replace_code, text)
            return decoded
        
        # Extract tables FIRST and build exclusion set
        tables = []
        table_text_content = set()  # Track text that's in tables to exclude from text extraction
        
        for idx, table in enumerate(doc.tables):
            try:
                table_data = format_table_for_json(table, doc, idx)
                
                # Collect all text from this table to exclude from text extraction
                for row in table_data.get("data", {}).get("rows", []):
                    for cell in row:
                        if cell and isinstance(cell, str):
                            # Normalize and add to exclusion set
                            normalized = cell.strip().lower()
                            if normalized:
                                table_text_content.add(normalized)
                                # Also add partial matches (for multi-word entries)
                                words = normalized.split()
                                if len(words) > 1:
                                    for word in words:
                                        if len(word) > 3:  # Only meaningful words
                                            table_text_content.add(word)
                
                # Try to detect table title from nearby text
                table_title = self._detect_table_title(doc, idx)
                if table_title:
                    table_data["title"] = table_title
                
                # Remove position_in_text if present (not needed in new structure)
                if "position_in_text" in table_data:
                    del table_data["position_in_text"]
                
                tables.append(table_data)
            except Exception as e:
                print(f"  Warning: Failed to extract table {idx}: {e}")
        
        def is_table_content(text: str) -> bool:
            """Check if text appears to be table content."""
            if not text or len(text.strip()) < 3:
                return False
            
            # Remove markdown list markers and other prefixes
            cleaned = text.strip()
            if cleaned.startswith('- '):
                cleaned = cleaned[2:].strip()
            if cleaned.startswith('## '):
                cleaned = cleaned[3:].strip()
            
            normalized = cleaned.lower()
            
            # Check if it's in our table content set
            if normalized in table_text_content:
                return True
            
            # Check if any significant word from the text is in table content
            words = normalized.split()
            for word in words:
                if len(word) > 3 and word in table_text_content:
                    return True
            
            # Pattern: "X-Y CODE Description" (typical table row format)
            # e.g., "1-1 01.41.10 Dojnice živé" or "- 1-1 01.41.10 Dojnice živé"
            if re.match(r'^[- ]*\d+-\d+\s+\d+[.\d]*\s+', normalized):
                return True
            
            # Pattern: Just codes like "01.41.10" or "28.93"
            if re.match(r'^\d+\.\d+\.?\d*\s*$', normalized):
                return True
            
            # Pattern: Single item numbers like "1-4", "1-7" (standalone)
            if re.match(r'^\d+-\d+\s*$', normalized):
                return True
            
            # Pattern: Multiple codes on separate lines (table content fragments)
            # e.g., "23.44\n23.9\n25.73" - if text has multiple code patterns
            code_patterns = re.findall(r'\d+\.\d+\.?\d*', normalized)
            if len(code_patterns) >= 2:
                return True
            
            return False
        
        # Extract text content using serializer, but EXCLUDE table content
        serializer_provider = ChunkingSerializerProvider()
        doc_serializer = serializer_provider.get_serializer(doc=doc)
        visited: set[str] = set()
        text_parts = []
        
        # Iterate through document body.children (like sequential_parser)
        if hasattr(doc, 'body') and hasattr(doc.body, 'children'):
            for child_ref in doc.body.children:
                if not hasattr(child_ref, 'resolve'):
                    continue
                try:
                    item = child_ref.resolve(doc)
                    if item is None:
                        continue
                    
                    # Skip table items entirely
                    if hasattr(item, 'label') and 'table' in str(item.label).lower():
                        continue
                    
                    # Use serializer for text items
                    if isinstance(item, (TextItem, ListGroup, InlineGroup)):
                        if item.self_ref in visited:
                            continue
                        visited.add(item.self_ref)
                        
                        ser_res = doc_serializer.serialize(item=item, visited=visited)
                        if ser_res.text:
                            cleaned_text = ser_res.text.strip()
                            # Decode character codes if present
                            if '/c' in cleaned_text:
                                cleaned_text = decode_character_codes(cleaned_text)
                            
                            # Skip if it's table content
                            if is_table_content(cleaned_text):
                                continue
                            
                            # Skip if it's just a table header row pattern
                            if cleaned_text.startswith('Položka') and ('KP' in cleaned_text or 'Názov' in cleaned_text):
                                continue
                            
                            if cleaned_text:
                                text_parts.append(cleaned_text)
                except Exception as e:
                    # Skip items that can't be resolved or serialized
                    continue
        
        # Fallback: if no text extracted via serializer, try direct access with filtering
        if not text_parts:
            for text_elem in doc.texts:
                text = getattr(text_elem, 'text', '') or ''
                if isinstance(text, str) and text.strip():
                    # Decode character codes if present
                    if '/c' in text:
                        text = decode_character_codes(text)
                    
                    # Skip table content
                    if is_table_content(text):
                        continue
                    
                    if text.strip():
                        text_parts.append(text.strip())
        
        annex_text = "\n".join(text_parts)
        
        # Extract pictures
        pictures = []
        if hasattr(doc, 'pictures') and doc.pictures:
            for idx, picture in enumerate(doc.pictures):
                pictures.append({
                    "index": idx
                })
        
        # Hybrid mode: reconstruct missing tables from fragments
        if annex.parser == "hybrid" or annex.parser_options.get("reconstruct_tables", False):
            reconstructed = self._reconstruct_tables_from_fragments(doc, tables)
            if reconstructed:
                print(f"  ✓ Reconstructed {len(reconstructed)} additional tables")
                tables.extend(reconstructed)
        
        # Extract annex heading and section headers from section_header elements
        annex_heading = None
        section_headers = {}  # section_num -> header_text
        poznamky_items = []  # Notes items
        in_poznamky = False
        
        for text_item in doc.texts:
            label = getattr(text_item, 'label', '')
            raw = getattr(text_item, 'text', '') or ''
            decoded = decode_character_codes(raw)
            
            # Check for Poznámky section
            if 'Poznámk' in decoded or (raw and '/c80/c111/c122/c110' in raw):  # "Pozn" pattern
                in_poznamky = True
                continue
            
            if in_poznamky and label == 'list_item':
                poznamky_items.append(decoded)
                continue
            
            if label == 'section_header':
                # First section header is the main heading
                if annex_heading is None:
                    annex_heading = decoded
                    continue
                
                # Look for "Odpisová skupina X" pattern
                match = re.search(r'Odpisová\s+skupina\s+(\d+)', decoded, re.IGNORECASE)
                if match:
                    section_num = match.group(1)
                    section_headers[section_num] = decoded
        
        # Associate section headers with tables based on item prefixes
        for table in tables:
            rows = table.get("data", {}).get("rows", [])
            if rows:
                first_item = str(rows[0][0]).strip() if rows[0] else ""
                # Extract group number from item like "2-1" -> "2"
                match = re.match(r'^(\d+)-', first_item)
                if match:
                    group_num = match.group(1)
                    if group_num in section_headers:
                        table["title"] = section_headers[group_num]
        
        # Merge split tables from the same group
        merged_tables = []
        i = 0
        while i < len(tables):
            current = tables[i]
            current_rows = current.get("data", {}).get("rows", [])
            
            if current_rows:
                first_item = str(current_rows[0][0]).strip() if current_rows[0] else ""
                match = re.match(r'^(\d+)-', first_item)
                
                if match:
                    current_group = match.group(1)
                    
                    # Look for next table with same group
                    while i + 1 < len(tables):
                        next_table = tables[i + 1]
                        next_rows = next_table.get("data", {}).get("rows", [])
                        
                        if next_rows:
                            next_first = str(next_rows[0][0]).strip() if next_rows[0] else ""
                            next_match = re.match(r'^(\d+)-', next_first)
                            
                            if next_match and next_match.group(1) == current_group:
                                # Same group - merge rows
                                current_rows.extend(next_rows)
                                current["data"]["rows"] = current_rows
                                # Update markdown
                                cols = current["data"].get("columns", ["Položka", "KP", "Názov"])
                                md_rows = [f"| {' | '.join(str(c) for c in row)} |" for row in current_rows]
                                current["markdown"] = f"| {' | '.join(cols)} |\n| {' | '.join(['---'] * len(cols))} |\n" + "\n".join(md_rows)
                                print(f"  ✓ Merged table for group {current_group}")
                                i += 1
                            else:
                                break
                        else:
                            break
            
            merged_tables.append(current)
            i += 1
        
        tables = merged_tables
        
        # Post-process: Split merged rows (e.g., "4-15 4-16" -> two rows)
        for table in tables:
            rows = table.get("data", {}).get("rows", [])
            new_rows = []
            for row in rows:
                if row and len(row) >= 3:
                    item_col = str(row[0]).strip()
                    # Check for merged items like "4-15 4-16"
                    merged_match = re.match(r'^(\d+-\d+)\s+(\d+-\d+)$', item_col)
                    if merged_match:
                        item1 = merged_match.group(1)
                        item2 = merged_match.group(2)
                        kp_col = str(row[1]).strip()
                        desc_col = str(row[2]).strip()
                        
                        # Try to split KP codes if they're also merged (e.g., "30.12 30.2")
                        kp_parts = kp_col.split()
                        kp1 = kp_parts[0] if len(kp_parts) >= 1 else kp_col
                        kp2 = kp_parts[1] if len(kp_parts) >= 2 else kp_col
                        
                        # Try to split description by finding capital letter boundary
                        # Pattern: "First phrase Second phrase" -> split at capital after lowercase
                        desc_split = re.split(r'(?<=[a-záäčďéíľĺňóôŕšťúýž])\s+(?=[A-ZÁÄČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ])', desc_col)
                        if len(desc_split) >= 2:
                            desc1 = desc_split[0].strip()
                            desc2 = ' '.join(desc_split[1:]).strip()
                        else:
                            desc1 = desc_col
                            desc2 = desc_col
                        
                        new_rows.append([item1, kp1, desc1])
                        new_rows.append([item2, kp2, desc2])
                        print(f"  ✓ Split merged row: {item_col} -> {item1}, {item2}")
                    else:
                        new_rows.append(row)
                else:
                    new_rows.append(row)
            if new_rows != rows:
                table["data"]["rows"] = new_rows
                # Regenerate markdown
                cols = table["data"].get("columns", ["Položka", "KP", "Názov"])
                md_rows = [f"| {' | '.join(str(c) for c in r)} |" for r in new_rows]
                table["markdown"] = f"| {' | '.join(cols)} |\n| {' | '.join(['---'] * len(cols))} |\n" + "\n".join(md_rows)
        
        # Post-process: Add missing items from text elements
        # Collect all item numbers that exist in tables
        existing_items = set()
        for table in tables:
            for row in table.get("data", {}).get("rows", []):
                if row:
                    existing_items.add(str(row[0]).strip())
        
        # Look for items in text elements that should be in tables
        missing_items = {}  # group_num -> [(item, kp, desc)]
        current_item = None
        current_item_desc = []
        
        for text_item in doc.texts:
            raw = getattr(text_item, 'text', '') or ''
            decoded = decode_character_codes(raw)
            label = getattr(text_item, 'label', '')
            
            # Stop collecting when we hit poznámky section
            if 'Poznámk' in decoded or (raw and '/c80/c111/c122/c110' in raw):
                # Save any pending item before stopping
                if current_item and current_item not in existing_items:
                    group_num = current_item.split('-')[0]
                    if group_num not in missing_items:
                        missing_items[group_num] = []
                    desc = ', '.join(current_item_desc) if current_item_desc else ''
                    missing_items[group_num].append((current_item, '', desc))
                current_item = None
                current_item_desc = []
                break  # Stop processing once we hit poznámky
            
            # Check for item number pattern like "4-17"
            item_match = re.match(r'^(\d+)-(\d+)$', decoded.strip())
            if item_match:
                # Save previous item if exists
                if current_item and current_item not in existing_items:
                    group_num = current_item.split('-')[0]
                    if group_num not in missing_items:
                        missing_items[group_num] = []
                    desc = ', '.join(current_item_desc) if current_item_desc else ''
                    missing_items[group_num].append((current_item, '', desc))
                
                current_item = decoded.strip()
                current_item_desc = []
            elif current_item and label == 'list_item':
                # Collect description items following the item number
                current_item_desc.append(decoded)
        
        # Don't forget the last item (only if we didn't break out of loop)
        if current_item and current_item not in existing_items:
            group_num = current_item.split('-')[0]
            if group_num not in missing_items:
                missing_items[group_num] = []
            desc = ', '.join(current_item_desc) if current_item_desc else ''
            missing_items[group_num].append((current_item, '', desc))
        
        # Add missing items to their respective tables
        for group_num, items in missing_items.items():
            # Find the table for this group
            for table in tables:
                table_rows = table.get("data", {}).get("rows", [])
                if table_rows:
                    first_item = str(table_rows[0][0]).strip()
                    if first_item.startswith(f"{group_num}-"):
                        # Add missing items to this table
                        for item, kp, desc in items:
                            table_rows.append([item, kp, desc])
                            print(f"  ✓ Added missing item: {item}")
                        # Sort rows by item number
                        table_rows.sort(key=lambda r: int(str(r[0]).split('-')[1]) if r and '-' in str(r[0]) else 999)
                        table["data"]["rows"] = table_rows
                        # Regenerate markdown
                        cols = table["data"].get("columns", ["Položka", "KP", "Názov"])
                        md_rows = [f"| {' | '.join(str(c) for c in r)} |" for r in table_rows]
                        table["markdown"] = f"| {' | '.join(cols)} |\n| {' | '.join(['---'] * len(cols))} |\n" + "\n".join(md_rows)
                        break
        
        # Build clean annex text with ONLY heading and Poznámky
        # (section headers are associated with tables, not in main text)
        text_parts_final = []
        if annex_heading:
            text_parts_final.append(f"# {annex_heading}")
        if poznamky_items:
            text_parts_final.append("\n## Poznámky:")
            for item in poznamky_items:
                text_parts_final.append(f"- {item}")
        
        annex_text = "\n".join(text_parts_final)
        
        # Return new structure format
        return {
            "id": f"annex-{annex.number}",
            "number": annex.number,
            "title": annex.title,
            "source": annex.type,
            "source_file": str(annex.local_path.name) if annex.local_path else None,
            "content": {
                "text": annex_text,
                "tables": tables,
                "pictures": pictures
            },
            "metadata": {
                "total_texts": len(doc.texts),
                "total_tables": len(doc.tables),
                "total_pictures": len(pictures),
                "status": "processed"
            }
        }
    
    def _detect_table_title(self, doc: DoclingDocument, table_idx: int) -> Optional[str]:
        """
        Try to detect title for a table based on nearby text.
        
        Looks for patterns like "Odpisová skupina X" before the table.
        """
        # This is a simplified heuristic - could be improved
        table = doc.tables[table_idx]
        
        # Check table caption
        if hasattr(table, 'captions') and table.captions:
            for caption in table.captions:
                if hasattr(caption, 'text') and caption.text:
                    return caption.text
        
        # Check first cell for group indicator
        if hasattr(table, 'data') and table.data:
            grid = getattr(table.data, 'grid', None)
            if grid and len(grid) > 0 and len(grid[0]) > 0:
                first_cell = getattr(grid[0][0], 'text', '')
                if 'Odpisová skupina' in first_cell or 'Položka' in first_cell:
                    return f"Tabuľka {table_idx + 1}"
        
        return None
    
    def _reconstruct_tables_from_fragments(
        self, 
        doc: DoclingDocument, 
        existing_tables: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct tables from fragmented text elements in DoclingDocument.
        
        Some PDFs have tables that Docling doesn't detect as tables - the content
        is parsed as individual text elements. This function detects such cases
        by looking for section headers (like "Odpisová skupina X") and attempts
        to reconstruct the table from nearby text elements.
        
        Args:
            doc: DoclingDocument with text elements
            existing_tables: Tables already extracted by Docling
            
        Returns:
            List of reconstructed tables in the standard format
        """
        from collections import defaultdict
        
        def decode_c_codes(text: str) -> str:
            """Decode /cXX character codes to actual characters."""
            if not text or '/c' not in text:
                return text
            
            win1250 = {
                138: 'Š', 154: 'š', 142: 'Ž', 158: 'ž', 141: 'Ť', 157: 'ť',
                188: 'Ľ', 190: 'ľ', 200: 'Č', 232: 'č', 207: 'Ď', 239: 'ď',
                210: 'Ň', 242: 'ň', 192: 'Ŕ', 224: 'ŕ', 197: 'Ĺ', 229: 'ĺ',
                193: 'Á', 225: 'á', 201: 'É', 233: 'é', 205: 'Í', 237: 'í',
                211: 'Ó', 243: 'ó', 218: 'Ú', 250: 'ú', 221: 'Ý', 253: 'ý',
                196: 'Ä', 228: 'ä', 212: 'Ô', 244: 'ô',
            }
            
            def replace(match):
                code = int(match.group(1))
                if code in win1250:
                    return win1250[code]
                return chr(code)
            
            return re.sub(r'/c(\d+)', replace, text)
        
        reconstructed = []
        
        # Collect existing table item prefixes to avoid duplicates
        existing_item_prefixes = set()
        for table in existing_tables:
            rows = table.get("data", {}).get("rows", [])
            for row in rows:
                if row and len(row) > 0:
                    item = str(row[0]).strip()
                    # Get prefix like "2-" from "2-1"
                    match = re.match(r'^(\d+)-', item)
                    if match:
                        existing_item_prefixes.add(match.group(1))
        
        # Find section headers in text elements
        section_headers = {}  # section_num -> header_text
        for text_item in doc.texts:
            raw = getattr(text_item, 'text', '') or ''
            decoded = decode_c_codes(raw)
            label = getattr(text_item, 'label', '')
            
            # Look for "Odpisová skupina X" headers
            match = re.search(r'Odpisová\s+skupina\s+(\d+)', decoded, re.IGNORECASE)
            if match:
                section_num = match.group(1)
                section_headers[section_num] = decoded
        
        # For each section header, check if it has a corresponding table
        for section_num, header_text in section_headers.items():
            if section_num in existing_item_prefixes:
                continue  # Already have a table for this section
            
            print(f"  Reconstructing table for: {header_text}")
            
            # Collect all items that belong to this section
            item_pattern = f"^{section_num}-"
            items_data = {}  # item_num -> {kp, desc}
            
            # First pass: get items from list_items
            for text_item in doc.texts:
                raw = getattr(text_item, 'text', '') or ''
                decoded = decode_c_codes(raw)
                label = getattr(text_item, 'label', '')
                
                # Match items like "1-X CODE Description"
                match = re.match(rf'^({section_num}-\d+)\s+(\d+[\.\d]*)\s+(.+)$', decoded)
                if match:
                    item_num = match.group(1)
                    kp_code = match.group(2)
                    desc = match.group(3)
                    items_data[item_num] = {'kp': kp_code, 'desc': desc}
            
            # Second pass: reconstruct from fragmented text by position
            page_elements = []
            for text_item in doc.texts:
                prov = getattr(text_item, 'prov', [])
                if not prov:
                    continue
                
                bbox = prov[0] if isinstance(prov[0], dict) else getattr(prov[0], '__dict__', {})
                if isinstance(prov[0], dict):
                    bbox = prov[0].get('bbox', {})
                else:
                    bbox = getattr(prov[0], 'bbox', {})
                    if hasattr(bbox, '__dict__'):
                        bbox = bbox.__dict__
                
                raw = getattr(text_item, 'text', '') or ''
                decoded = decode_c_codes(raw)
                
                top = bbox.get('t', 0) if isinstance(bbox, dict) else getattr(bbox, 't', 0)
                left = bbox.get('l', 0) if isinstance(bbox, dict) else getattr(bbox, 'l', 0)
                
                page_elements.append({
                    'text': decoded,
                    'top': top,
                    'left': left
                })
            
            # Group by rows
            rows_by_pos = defaultdict(list)
            for elem in page_elements:
                row_key = round(elem['top'] / 8) * 8
                rows_by_pos[row_key].append(elem)
            
            for row_key in rows_by_pos:
                rows_by_pos[row_key].sort(key=lambda x: x['left'])
            
            # Extract items from grouped rows
            for row_key in sorted(rows_by_pos.keys()):
                row_texts = [e['text'] for e in rows_by_pos[row_key]]
                
                if row_texts and re.match(rf'^{section_num}-\d+$', row_texts[0]):
                    item_num = row_texts[0]
                    if item_num not in items_data:
                        kp_code = ''
                        desc_parts = []
                        for txt in row_texts[1:]:
                            if re.match(r'^\d+\.[\d\.]*$', txt) and not kp_code:
                                kp_code = txt
                            else:
                                desc_parts.append(txt)
                        
                        desc = ' '.join(desc_parts)
                        items_data[item_num] = {'kp': kp_code, 'desc': desc}
            
            # Build table from collected items
            if items_data:
                sorted_items = sorted(
                    items_data.items(), 
                    key=lambda x: int(x[0].split('-')[1])
                )
                
                rows = [[item_num, data['kp'], data['desc']] 
                        for item_num, data in sorted_items]
                
                # Generate markdown
                md_rows = [f"| {r[0]} | {r[1]} | {r[2]} |" for r in rows]
                markdown = "| Položka | KP | Názov |\n| --- | --- | --- |\n" + "\n".join(md_rows)
                
                reconstructed.append({
                    "index": len(existing_tables) + len(reconstructed),
                    "title": header_text,
                    "markdown": markdown,
                    "caption": None,
                    "data": {
                        "columns": ["Položka", "KP", "Názov"],
                        "rows": rows
                    },
                    "reconstructed": True  # Mark as reconstructed
                })
                
                print(f"    ✓ Reconstructed {len(rows)} items")
        
        return reconstructed
    
    def integrate_annexes(
        self,
        structure: Dict[str, Any],
        annexes: List[AnnexInfo],
        law_dir: Path,
        annexes_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Integrate annex content into output structure.
        
        Args:
            structure: Main document structure
            annexes: List of annexes to integrate
            law_dir: Law directory (where cache/ and annexes/ are located)
            annexes_dir: Directory containing PDF annexes (defaults to law_dir/annexes)
            
        Returns:
            Updated structure with integrated annexes in new format
        """
        print(f"\nIntegrating {len(annexes)} annexes...")
        
        # Get existing inline annexes from structure (processed by sequential parser)
        existing_annex_list = structure.get("annexes", {}).get("annex_list", [])
        existing_annex_map = {a.get("number"): a for a in existing_annex_list}
        
        annex_list = []
        external_count = 0
        inline_count = 0
        
        for annex in annexes:
            print(f"\nProcessing annex {annex.number}: {annex.title}")
            
            if annex.type == "external_pdf":
                # Load or convert PDF annex
                doc = self.load_or_convert_annex(annex, law_dir, annexes_dir)
                
                if doc:
                    content = self.extract_annex_content(doc, annex)
                    annex_list.append(content)
                    external_count += 1
                    print(f"  ✓ Extracted {len(content['content']['tables'])} tables")
                else:
                    # Add placeholder for missing annex
                    annex_list.append({
                        "id": f"annex-{annex.number}",
                        "number": annex.number,
                        "title": annex.title,
                        "source": "external_pdf",
                        "source_file": str(annex.local_path.name) if annex.local_path else None,
                        "content": {
                            "text": "",
                            "tables": [],
                            "pictures": []
                        },
                        "metadata": {
                            "status": "not_available"
                        }
                    })
                    external_count += 1
            else:
                # Inline annex - merge content from main document
                inline_annex = existing_annex_map.get(annex.number)
                if inline_annex:
                    # Use the content already extracted by sequential parser
                    annex_list.append(inline_annex)
                    inline_count += 1
                    print(f"  ✓ Found inline content with {len(inline_annex.get('content', {}).get('tables', []))} tables")
                else:
                    # Inline annex not found in structure - add placeholder
                    annex_list.append({
                        "id": f"annex-{annex.number}",
                        "number": annex.number,
                        "title": annex.title,
                        "source": "inline",
                        "content": {
                            "text": "",
                            "tables": [],
                            "pictures": []
                        },
                        "metadata": {
                            "status": "not_found"
                        }
                    })
                    inline_count += 1
                    print(f"  ⚠ Inline annex not found in document structure")
        
        # Update structure
        if "annexes" not in structure:
            structure["annexes"] = {}
        
        structure["annexes"]["annex_list"] = annex_list
        structure["annexes"]["summary"] = {
            "total_annexes": len(annex_list),
            "external_annexes": external_count,
            "inline_annexes": inline_count
        }
        
        # Update metadata
        structure["metadata"]["total_annexes"] = len(annex_list)
        structure["metadata"]["external_annexes"] = external_count
        
        return structure


def create_manifest_from_html(
    html_path: Path,
    law_id: str,
    output_path: Optional[Path] = None,
    law_dir: Optional[Path] = None
) -> LawManifest:
    """
    Create a manifest file from HTML document.
    
    Args:
        html_path: Path to HTML file
        law_id: Law identifier
        output_path: Optional path to save manifest.yaml
        law_dir: Law directory for relative paths (defaults to output_path.parent)
        
    Returns:
        Created LawManifest
    """
    processor = AnnexProcessor()
    annexes = processor.detect_annexes_from_html(html_path)
    
    # Determine base directory for relative paths
    if law_dir is None:
        if output_path:
            law_dir = output_path.parent
        else:
            law_dir = html_path.parent
    
    manifest = LawManifest(
        law_id=law_id,
        source="slov-lex",
        main_document={
            "type": "html",
            "filename": "main.html" if html_path.name == "main.html" else str(html_path.relative_to(law_dir)) if html_path.is_relative_to(law_dir) else str(html_path)
        },
        annexes=annexes
    )
    
    if output_path:
        manifest.save(output_path, base_dir=law_dir)
        print(f"Manifest saved to {output_path}")
    
    return manifest


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Detect and process law annexes"
    )
    parser.add_argument(
        "html_file",
        type=str,
        help="Path to HTML law document"
    )
    parser.add_argument(
        "--law-id",
        type=str,
        required=True,
        help="Law identifier (e.g., '595/2003')"
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        help="Path to save manifest.yaml"
    )
    parser.add_argument(
        "--annexes-dir",
        type=str,
        help="Directory containing PDF annexes"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Directory for caching conversions"
    )
    
    args = parser.parse_args()
    
    html_path = Path(args.html_file)
    
    if not html_path.exists():
        print(f"Error: HTML file not found: {html_path}")
        exit(1)
    
    # Create manifest
    manifest_path = Path(args.output_manifest) if args.output_manifest else None
    law_dir = manifest_path.parent if manifest_path else html_path.parent
    manifest = create_manifest_from_html(html_path, args.law_id, manifest_path, law_dir)
    
    print(f"\nDetected {len(manifest.annexes)} annexes:")
    for annex in manifest.annexes:
        print(f"  {annex.number}. {annex.title} [{annex.type}]")
        if annex.url:
            print(f"      URL: {annex.url}")
    
    # If annexes directory provided, try to convert
    if args.annexes_dir:
        processor = AnnexProcessor()
        annexes_dir = Path(args.annexes_dir)
        law_dir = annexes_dir.parent  # Assume law_dir is parent of annexes_dir
        
        print(f"\nLooking for PDF annexes in {annexes_dir}...")
        
        for annex in manifest.annexes:
            if annex.type == "external_pdf":
                doc = processor.load_or_convert_annex(
                    annex, 
                    law_dir, 
                    annexes_dir
                )
                if doc:
                    print(f"  ✓ Annex {annex.number}: {len(doc.tables)} tables")

