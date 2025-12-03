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
from docling_core.types.doc import DoclingDocument


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
    """
    number: str
    title: str
    type: str  # "external_pdf" | "inline"
    url: Optional[str] = None
    local_path: Optional[Path] = None
    docling_path: Optional[Path] = None
    content_type: str = "mixed"  # "table" | "text" | "mixed"
    html_id: Optional[str] = None
    
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
            Dictionary with extracted content
        """
        from sequential_parser import format_table_for_json
        
        content = {
            "id": f"annex-{annex.number}",
            "number": annex.number,
            "title": annex.title,
            "source": annex.type,
            "source_file": str(annex.local_path.name) if annex.local_path else None,
            "tables": [],
            "text_content": "",
            "metadata": {
                "total_texts": len(doc.texts),
                "total_tables": len(doc.tables)
            }
        }
        
        # Extract tables
        for idx, table in enumerate(doc.tables):
            try:
                table_data = format_table_for_json(table, doc, idx)
                
                # Try to detect table title from nearby text
                table_title = self._detect_table_title(doc, idx)
                if table_title:
                    table_data["title"] = table_title
                
                content["tables"].append(table_data)
            except Exception as e:
                print(f"  Warning: Failed to extract table {idx}: {e}")
        
        # Extract text content
        text_parts = []
        for text_elem in doc.texts:
            text = getattr(text_elem, 'text', '') or ''
            if text.strip():
                text_parts.append(text.strip())
        
        content["text_content"] = "\n".join(text_parts)
        
        return content
    
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
            Updated structure with integrated annexes
        """
        print(f"\nIntegrating {len(annexes)} annexes...")
        
        annex_list = []
        
        for annex in annexes:
            print(f"\nProcessing annex {annex.number}: {annex.title}")
            
            if annex.type == "external_pdf":
                # Load or convert PDF annex
                doc = self.load_or_convert_annex(annex, law_dir, annexes_dir)
                
                if doc:
                    content = self.extract_annex_content(doc, annex)
                    annex_list.append(content)
                    print(f"  ✓ Extracted {len(content['tables'])} tables")
                else:
                    # Add placeholder for missing annex
                    annex_list.append({
                        "id": f"annex-{annex.number}",
                        "number": annex.number,
                        "title": annex.title,
                        "source": "external_pdf",
                        "status": "not_available",
                        "tables": [],
                        "text_content": ""
                    })
            else:
                # Inline annex - content should already be in structure
                # Just add metadata
                annex_list.append({
                    "id": f"annex-{annex.number}",
                    "number": annex.number,
                    "title": annex.title,
                    "source": "inline",
                    "tables": [],
                    "text_content": ""  # Will be filled from main document
                })
        
        # Update structure
        if "annexes" not in structure:
            structure["annexes"] = {}
        
        structure["annexes"]["annex_list"] = annex_list
        structure["metadata"]["total_annexes"] = len(annex_list)
        structure["metadata"]["external_annexes"] = sum(
            1 for a in annex_list if a.get("source") == "external_pdf"
        )
        
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

