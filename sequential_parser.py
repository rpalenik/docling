#!/usr/bin/env python3
"""
Sekvenčný parser pre docling dokumenty - úplne nová implementácia.

Tento modul implementuje sekvenčnú rekonštrukciu celého docling dokumentu
do hierarchickej JSON štruktúry v jednej O(n) iterácii cez doc.texts.

Vlastnosti:
- Jedna iterácia cez doc.texts (O(n))
- Bez rekurzie a opakovaných operácií
- Kompletná hierarchická štruktúra (parts → paragraphs → odseks → pismenos → subitems)
- Zachytiť všetky metadáta na najnižšej úrovni
- Referenčné markery zachované v texte
- Detailné logovanie priebehu
"""

import json
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterator

from docling_core.types.doc import DoclingDocument
from docling_core.transforms.chunker import BaseChunker, BaseChunk
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingSerializerProvider
from docling_core.transforms.serializer.base import BaseDocSerializer
from docling_core.transforms.chunker.doc_chunk import DocChunk, DocMeta
from docling_core.types.doc.document import (
    SectionHeaderItem, TextItem, TableItem, ListGroup, InlineGroup, 
    DocItem, TitleItem, NodeItem, ContentLayer
)
from docling_core.types.doc.labels import DocItemLabel
from pydantic import Field, ConfigDict


# ============================================================================
# Logging Functions
# ============================================================================

# Global log file handle
_log_file = None

def log_progress(level: str, message: str, timing: Optional[float] = None, log_to_file: bool = True) -> None:
    """
    Log progress message with timestamp and optional timing information.
    Logs to both stdout and optionally to log file.
    
    Args:
        level: Log level (INFO, DEBUG, WARNING, ERROR)
        message: Message to log
        timing: Optional timing in seconds to display
        log_to_file: Whether to log to file (default: True)
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    timing_str = f" [{timing:.2f}s]" if timing is not None else ""
    log_line = f"[{timestamp}] [{level}] {message}{timing_str}\n"
    
    # Always print to stdout
    print(f"[{timestamp}] [{level}] {message}{timing_str}", end='')
    
    # Also write to log file if available
    global _log_file
    if log_to_file and _log_file is not None:
        _log_file.write(log_line)
        _log_file.flush()  # Ensure immediate write


def set_log_file(log_file_path: str) -> None:
    """
    Set log file path and open file for writing.
    
    Args:
        log_file_path: Path to log file
    """
    global _log_file
    if _log_file is not None:
        _log_file.close()
    _log_file = open(log_file_path, 'w', encoding='utf-8')


def close_log_file() -> None:
    """Close log file if open."""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None


# ============================================================================
# Document Loading
# ============================================================================

def load_docling_document(json_path: str) -> DoclingDocument:
    """
    Load a DoclingDocument from a JSON file.
    
    Args:
        json_path: Path to JSON file containing DoclingDocument
        
    Returns:
        DoclingDocument object
    """
    return DoclingDocument.load_from_json(json_path)


# ============================================================================
# Marker Detection Functions
# ============================================================================

def detect_part_marker(text: str) -> Optional[str]:
    """
    Check if text is a part marker (ČASŤ).
    
    Args:
        text: Normalized text to check
        
    Returns:
        Part identifier (e.g., "PRVÁ ČASŤ") or None
    """
    pattern = r'^(PRVÁ|DRUHÁ|TRETIA|ŠTVRTÁ|PIATA|ŠIESTA|SEDMÁ|ÔSMA|DEVÄTÁ|DESIATA)\s+ČASŤ'
    match = re.match(pattern, text, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return None


def detect_paragraph_from_hyperlink(hyperlink: Optional[str]) -> Optional[str]:
    """
    Detect paragraph number from docling hyperlink (primary method).
    
    Args:
        hyperlink: Hyperlink string (e.g., "#paragraf-47")
        
    Returns:
        Paragraph number (e.g., "47") or None
    """
    if not hyperlink:
        return None
    
    hyperlink_str = str(hyperlink)
    if hyperlink_str.startswith('#paragraf-'):
        # Extract: #paragraf-47 -> "47"
        # Also handle: #paragraf-47.odsek-1 -> "47"
        parts = hyperlink_str.replace('#paragraf-', '').split('.')
        return parts[0] if parts else None
    
    return None


def detect_paragraph_marker(text: str, hyperlink: Optional[str] = None) -> Optional[str]:
    """
    Check if text is a paragraph marker (§ N).
    
    Uses hyperlink detection first (docling native), then falls back to regex.
    
    Args:
        text: Normalized text to check
        hyperlink: Optional hyperlink string (preferred method)
        
    Returns:
        Paragraph number (e.g., "5", "10a") or None
    """
    # PRIMARY: Try hyperlink detection (docling native)
    if hyperlink:
        para_num = detect_paragraph_from_hyperlink(hyperlink)
        if para_num:
            return para_num
    
    # FALLBACK: Regex detection from text
    if not text.startswith('§ '):
        return None
    
    match = re.match(r'^§\s+(\d+[a-zA-Z]*)', text)
    if match:
        return match.group(1)
    return None


def detect_odsek_from_hyperlink(hyperlink: Optional[str]) -> Optional[str]:
    """
    Detect odsek number from docling hyperlink (primary method).
    
    Args:
        hyperlink: Hyperlink string (e.g., "#paragraf-47.odsek-1")
        
    Returns:
        Odsek number (e.g., "1") or None
    """
    if not hyperlink:
        return None
    
    hyperlink_str = str(hyperlink)
    if 'odsek-' in hyperlink_str:
        # Extract: #paragraf-47.odsek-1 -> "1"
        match = re.search(r'odsek-(\d+)', hyperlink_str)
        return match.group(1) if match else None
    
    return None


def detect_odsek_marker(text: str, hyperlink: Optional[str] = None) -> Optional[str]:
    """
    Check if text is an odsek marker - supports multiple formats:
    - Standalone: (1), (2)
    - With spaces: ( 1 ), ( 2 )
    - At start of text: (1) text content
    - With non-breaking spaces
    
    Uses hyperlink detection first (docling native), then falls back to regex.
    
    Args:
        text: Normalized text to check
        hyperlink: Optional hyperlink string (preferred method)
        
    Returns:
        Odsek number (e.g., "1", "2") or None
    """
    # PRIMARY: Try hyperlink detection (docling native)
    if hyperlink:
        odsek_num = detect_odsek_from_hyperlink(hyperlink)
        if odsek_num:
            return odsek_num
    
    # FALLBACK: Regex detection from text
    # Pattern 1: Standalone marker (1), (2)
    if text.startswith('(') and text.endswith(')'):
        inner = text[1:-1].strip()
        if inner.isdigit():
            return inner
    
    # Pattern 2: Marker at start of text: (1) content
    match = re.match(r'^\((\d+)\)\s*', text)
    if match:
        return match.group(1)
    
    # Pattern 3: With non-breaking spaces: (\xa0 1 \xa0)
    match = re.match(r'^\([\s\xa0]*(\d+)[\s\xa0]*\)', text)
    if match:
        return match.group(1)
    
    return None


def detect_pismeno_from_hyperlink(hyperlink: Optional[str]) -> Optional[str]:
    """
    Detect pismeno letter from docling hyperlink (primary method).
    
    Args:
        hyperlink: Hyperlink string (e.g., "#paragraf-47.odsek-1.pismeno-a")
        
    Returns:
        Pismeno letter (e.g., "a", "aa") or None
    """
    if not hyperlink:
        return None
    
    hyperlink_str = str(hyperlink)
    if 'pismeno-' in hyperlink_str:
        # Extract: #paragraf-47.odsek-1.pismeno-a -> "a"
        match = re.search(r'pismeno-([a-z]+)', hyperlink_str, re.IGNORECASE)
        return match.group(1).lower() if match else None
    
    return None


def detect_pismeno_marker(text: str, hyperlink: Optional[str] = None) -> Optional[str]:
    """
    Check if text is a pismeno marker - supports multiple formats:
    - Standalone: a), b), aa), ab)
    - At start of text: a) content
    - With spaces: a ) content
    
    Uses hyperlink detection first (docling native), then falls back to regex.
    
    Args:
        text: Normalized text to check
        hyperlink: Optional hyperlink string (preferred method)
        
    Returns:
        Pismeno marker (e.g., "a", "b", "aa", "ab") or None
    """
    # PRIMARY: Try hyperlink detection (docling native)
    if hyperlink:
        pismeno_letter = detect_pismeno_from_hyperlink(hyperlink)
        if pismeno_letter:
            return pismeno_letter
    
    # FALLBACK: Regex detection from text
    # Pattern 1: Standalone marker a), b), aa), ab)
    if text.endswith(')') and not text.startswith('('):
        inner = text[:-1].strip()
        # Must be letters only, at least 1 character
        if len(inner) >= 1 and inner.isalpha():
            return inner.lower()
    
    # Pattern 2: Marker at start of text: a) content, aa) content
    match = re.match(r'^([a-z]+)\)\s*', text, re.IGNORECASE)
    if match:
        inner = match.group(1)
        if inner.isalpha() and len(inner) >= 1:
            return inner.lower()
    
    # Pattern 3: With spaces: a ) content
    match = re.match(r'^([a-z]+)\s+\)\s*', text, re.IGNORECASE)
    if match:
        inner = match.group(1)
        if inner.isalpha() and len(inner) >= 1:
            return inner.lower()
    
    return None


def detect_subitem_marker(text: str) -> Optional[str]:
    """
    Check if text is a subitem marker (1., 2., 3., etc.).
    
    Args:
        text: Normalized text to check
        
    Returns:
        Subitem number (e.g., "1", "2") or None
    """
    # Pattern 1: Standalone marker 1., 2.
    if text.endswith('.') and not text.startswith('.'):
        inner = text[:-1].strip()
        if inner.isdigit():
            return inner
    
    # Pattern 2: Marker at start of text: 1. content
    match = re.match(r'^(\d+)\.\s+', text)
    if match:
        return match.group(1)
    
    return None


def detect_law_end_marker(text: str) -> bool:
    """
    Detect end of main law text markers.
    
    Patterns:
    - "Tento zákon nadobúda účinnosť" (law effectiveness date)
    - "v. r." (signature marker - "vlastnou rukou")
    
    Args:
        text: Normalized text to check
        
    Returns:
        True if end of law marker found, False otherwise
    """
    normalized = text.strip().lower()
    
    # Pattern 1: Law effectiveness date
    if 'tento zákon nadobúda účinnosť' in normalized:
        return True
    
    # Pattern 2: Signature marker
    if normalized.endswith('v. r.') or normalized == 'v. r.':
        return True
    
    return False


def detect_annex_marker(text: str) -> Optional[str]:
    """
    Detect annex marker in text.
    
    Patterns: 
    - "Príloha č. 1 k zákonu" (most specific, preferred)
    - "Príloha č. 1", "Príloha č. 2" (standalone)
    
    Excludes:
    - "Prevziať prílohu" (download link)
    - "Príloha č. X tabuľka" (table reference, not annex title)
    
    Args:
        text: Normalized text to check
        
    Returns:
        Annex number ("1", "2") or None
    """
    # Exclude patterns that are NOT annex titles
    normalized = text.lower().strip()
    
    # Exclude download links
    if 'prevziať prílohu' in normalized:
        return None
    
    # Exclude table references (e.g., "Príloha č. 1 tabuľka A")
    if 'tabuľka' in normalized or 'tabula' in normalized:
        return None
    
    # Pattern 1: "Príloha č. X k zákonu" (most specific, preferred)
    match = re.search(r'príloha\s+č\.\s*(\d+)\s+k\s+zákonu', normalized, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 2: "Príloha č. 1", "Príloha č. 2" (standalone, not in sentence)
    # Check if it's at the start or is a standalone phrase
    match = re.match(r'^príloha\s+č\.\s*(\d+)\s*$', normalized, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 3: "Príloha č. 1" at start of text (may have more text after)
    # Only if it's clearly a heading (short text or followed by "k zákonu")
    match = re.match(r'^príloha\s+č\.\s*(\d+)', normalized, re.IGNORECASE)
    if match:
        # Only match if it's clearly a heading (short text or followed by "k zákonu")
        if len(text) < 100 or 'k zákonu' in normalized:
            return match.group(1)
    
    # Pattern 4: "Annex 1", "Annex 2" (English)
    match = re.search(r'^annex\s+(\d+)\s*$', normalized, re.IGNORECASE)
    if match:
        return match.group(1)
    
    return None


def detect_footnotes_section(text: str, hyperlink: Optional[str]) -> bool:
    """
    Detect "Poznámky" section header.
    
    Args:
        text: Text to check
        hyperlink: Hyperlink value
        
    Returns:
        True if this is the footnotes section header
    """
    if not hyperlink:
        return False
    hyperlink_str = str(hyperlink)
    normalized_text = text.strip().lower()
    
    # Check for "Poznámky" text with #poznamky hyperlink
    if normalized_text == "poznámky" and hyperlink_str == "#poznamky":
        return True
    
    return False


def detect_footnote_from_hyperlink(hyperlink: Optional[str]) -> Optional[str]:
    """
    Detect footnote ID from docling hyperlink (primary method for references).
    
    Note: This detects footnote REFERENCES in text (with hyperlink).
    For footnote DEFINITIONS (markers like "1)"), we still need text-based detection
    because definitions don't have hyperlinks.
    
    Args:
        hyperlink: Hyperlink string (e.g., "#poznamky.poznamka-1")
        
    Returns:
        Footnote ID (e.g., "1", "1a") or None
    """
    if not hyperlink:
        return None
    
    hyperlink_str = str(hyperlink)
    if 'poznamky.poznamka' in hyperlink_str or 'poznamka-' in hyperlink_str:
        # Extract: #poznamky.poznamka-1 -> "1"
        # Or: #poznamka-1 -> "1"
        match = re.search(r'poznamka-(\d+[a-z]*)', hyperlink_str)
        return match.group(1) if match else None
    
    return None


def detect_footnote_marker(text: str, hyperlink: Optional[str]) -> Optional[str]:
    """
    Detect footnote definition marker (not a reference in text).
    
    Uses hyperlink detection for references, text-based detection for definitions.
    
    Patterns for definitions (no hyperlink):
    - "1)", "2)", "37)" - simple numbered footnotes
    - "1a)", "1b)", "2aa)" - numbered with letter suffix
    - "37ab)", "37aba)" - complex nested footnotes
    
    Key: Definitions have NO hyperlink (unlike references in text).
    
    Args:
        text: Text to check
        hyperlink: Hyperlink value (should be None/empty for definitions)
        
    Returns:
        Footnote ID ("1", "1a", "2aa") or None
    """
    # If hyperlink exists, this is a reference, not a definition
    if hyperlink:
        hyperlink_str = str(hyperlink)
        if hyperlink_str and ('poznamky.poznamka' in hyperlink_str or 'poznamka-' in hyperlink_str):
            # This is a reference in text, not a definition
            return None
    
    # For definitions, use text-based detection (definitions don't have hyperlinks)
    normalized = text.strip()
    
    # Pattern 1: Simple numbered footnotes "1)", "2)", "37)"
    match = re.match(r'^(\d+)\)\s*$', normalized)
    if match:
        return match.group(1)
    
    # Pattern 2: Numbered with letter suffix "1a)", "1b)", "2aa)"
    match = re.match(r'^(\d+)([a-z]+)\)\s*$', normalized)
    if match:
        return match.group(1) + match.group(2)
    
    return None


def extract_marker_from_text(text: str, marker_type: str) -> Optional[Tuple[str, str]]:
    """
    Extract marker from text if it's part of larger content.
    
    Args:
        text: Text to search in
        marker_type: 'odsek', 'pismeno', or 'subitem'
        
    Returns:
        Tuple of (marker_value, remaining_text) or None
    """
    if marker_type == 'odsek':
        # Try to find (N) at start
        match = re.match(r'^\((\d+)\)\s*(.*)$', text)
        if match:
            return (match.group(1), match.group(2))
    
    elif marker_type == 'pismeno':
        # Try to find a) at start
        match = re.match(r'^([a-z]+)\)\s*(.*)$', text, re.IGNORECASE)
        if match:
            inner = match.group(1)
            if inner.isalpha() and len(inner) >= 1:
                return (inner.lower(), match.group(2))
    
    elif marker_type == 'subitem':
        # Try to find 1. at start
        match = re.match(r'^(\d+)\.\s+(.*)$', text)
        if match:
            return (match.group(1), match.group(2))
    
    return None


def is_pismeno_reference_in_context(
    previous_text: str, 
    marker: str, 
    current_element: Any = None,
    previous_elements: List[Any] = None
) -> bool:
    """
    Check if pismeno marker appears to be a reference in context.
    
    This function uses a hybrid approach:
    1. First checks explicit references from text elements (if available)
    2. Falls back to regex pattern matching on text content
    
    Args:
        previous_text: Previous text element content (from current pismeno or odsek)
        marker: Detected marker (e.g., "f")
        current_element: Current text element (optional, for checking explicit references)
        previous_elements: List of previous text elements (optional, for checking explicit references)
        
    Returns:
        True if marker is likely a reference, False if it's a structural marker
    """
    # ========================================================================
    # STEP 1: Check explicit references from text elements (most accurate)
    # ========================================================================
    
    # Check current element for references/hyperlinks
    if current_element is not None:
        # Check if current element has hyperlink (often indicates a reference)
        hyperlink = getattr(current_element, 'hyperlink', '')
        if hyperlink:
            # If hyperlink points to a pismeno (e.g., "#paragraf-3.odsek-1.pismeno-f")
            hyperlink_str = str(hyperlink)
            if 'pismeno' in hyperlink_str.lower():
                # Extract pismeno letter from hyperlink
                pismeno_match = re.search(r'pismeno[_-]?([a-z]+)', hyperlink_str, re.IGNORECASE)
                if pismeno_match and pismeno_match.group(1).lower() == marker.lower():
                    return True
        
        # Check if current element has references attribute
        element_references = getattr(current_element, 'references', [])
        if element_references:
            # Check if any reference points to the marker
            for ref in element_references:
                if isinstance(ref, dict):
                    ref_text = ref.get('text', '')
                    ref_target = ref.get('target', '')
                    # Check if reference contains the marker
                    if marker.lower() in ref_text.lower() or marker.lower() in str(ref_target).lower():
                        return True
                elif isinstance(ref, str):
                    if marker.lower() in ref.lower():
                        return True
    
    # Check previous elements for references/hyperlinks
    if previous_elements:
        for prev_elem in previous_elements[-3:]:  # Check last 3 previous elements
            if prev_elem is None:
                continue
            
            # Check hyperlink
            hyperlink = getattr(prev_elem, 'hyperlink', '')
            if hyperlink:
                hyperlink_str = str(hyperlink)
                # If previous element has hyperlink and current text is just the marker,
                # it's likely a continuation of the reference
                if 'pismeno' in hyperlink_str.lower() or 'paragraf' in hyperlink_str.lower():
                    # Check if previous text ends with reference words
                    prev_text = getattr(prev_elem, 'text', '')
                    if prev_text:
                        normalized_prev = prev_text.replace('\n', ' ').strip()
                        reference_endings = [
                            r'písmene\s*$',
                            r'písm\.\s*$',
                            r'pismeno\s*$',
                            r'písmena\s*$',
                            r'písm\s*$',
                        ]
                        for pattern in reference_endings:
                            if re.search(pattern, normalized_prev, re.IGNORECASE):
                                return True
            
            # Check references attribute
            prev_references = getattr(prev_elem, 'references', [])
            if prev_references:
                # If previous element has references, current marker might be part of it
                for ref in prev_references:
                    if isinstance(ref, dict):
                        ref_type = ref.get('type', '')
                        if ref_type == 'pismeno':
                            return True
                    elif isinstance(ref, str) and 'pismeno' in ref.lower():
                        return True
    
    # ========================================================================
    # STEP 2: Fallback to regex pattern matching (if explicit refs not available)
    # ========================================================================
    
    if not previous_text:
        return False
    
    # Normalize previous text (remove newlines for better matching)
    normalized_prev = previous_text.replace('\n', ' ').strip()
    
    # Check for reference patterns - marker should appear after reference words
    # Patterns like: "v písmene f)", "písm. f)", "pismeno f)", "písmena f)"
    reference_patterns = [
        r'písmene\s+' + re.escape(marker) + r'\)',
        r'písm\.\s+' + re.escape(marker) + r'\)',
        r'pismeno\s+' + re.escape(marker) + r'\)',
        r'písmena\s+' + re.escape(marker) + r'\)',
        r'písm\s+' + re.escape(marker) + r'\)',
    ]
    
    for pattern in reference_patterns:
        if re.search(pattern, normalized_prev, re.IGNORECASE):
            return True
    
    # Also check if previous text ends with reference words (common case when text is split)
    # If text ends with "v písmene" or "písm." and next element is just the marker, it's a reference
    reference_endings = [
        r'v\s+písmene\s*$',
        r'písm\.\s*$',
        r'pismeno\s*$',
        r'písmena\s*$',
        r'písm\s*$',
    ]
    
    for pattern in reference_endings:
        if re.search(pattern, normalized_prev, re.IGNORECASE):
            return True
    
    return False


# ============================================================================
# Metadata Extraction Functions
# ============================================================================

def extract_references_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract reference patterns from text (preserve markers in text).
    
    Patterns:
    - § N (paragraph reference)
    - odsek N (odsek reference)
    - odsek N.M (odsek with paragraph)
    - pismeno N (pismeno reference)
    
    Args:
        text: Text to search for references
        
    Returns:
        List of reference metadata dictionaries
    """
    references = []
    
    # Paragraph references: § 5, § 10a, etc.
    para_pattern = r'§\s+(\d+[a-zA-Z]*)'
    for match in re.finditer(para_pattern, text):
        references.append({
            'type': 'paragraph',
            'target': match.group(1),
            'text': match.group(0),
            'position': match.start()
        })
    
    # Odsek references: odsek 5.1, odsek (1), etc.
    odsek_pattern = r'odsek\s+(\d+\.\d+|\d+|\(\d+\))'
    for match in re.finditer(odsek_pattern, text, re.IGNORECASE):
        target = match.group(1).strip('()')
        references.append({
            'type': 'odsek',
            'target': target,
            'text': match.group(0),
            'position': match.start()
        })
    
    # Pismeno references: pismeno a), pismeno 5.1.a, etc.
    pismeno_pattern = r'pismeno\s+([a-z]\)|\d+\.\d+\.[a-z])'
    for match in re.finditer(pismeno_pattern, text, re.IGNORECASE):
        references.append({
            'type': 'pismeno',
            'target': match.group(1).strip(')'),
            'text': match.group(0),
            'position': match.start()
        })
    
    return references


def extract_footnotes_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract footnote markers from text (preserve markers in text).
    
    Patterns:
    - Superscript numbers: text¹, text²
    - Bracket numbers: text[1], text[2]
    - Parentheses numbers: text(1), text(2)
    
    Args:
        text: Text to search for footnotes
        
    Returns:
        List of footnote metadata dictionaries
    """
    footnotes = []
    
    # Superscript numbers (Unicode superscripts)
    superscript_pattern = r'([\u00B9\u00B2\u00B3\u2074\u2075\u2076\u2077\u2078\u2079\u2070]+)'
    for match in re.finditer(superscript_pattern, text):
        footnotes.append({
            'marker': match.group(1),
            'position': match.start(),
            'type': 'superscript'
        })
    
    # Bracket numbers: [1], [2], etc.
    bracket_pattern = r'\[(\d+)\]'
    for match in re.finditer(bracket_pattern, text):
        footnotes.append({
            'marker': match.group(0),
            'number': match.group(1),
            'position': match.start(),
            'type': 'bracket'
        })
    
    # Parentheses numbers: (1), (2) - but only if not an odsek marker
    # We'll check this more carefully in context
    paren_pattern = r'\((\d+)\)'
    for match in re.finditer(paren_pattern, text):
        # Check if this looks like a footnote (not at start of line, has text before)
        if match.start() > 0 and text[match.start()-1].isspace():
            # Might be odsek marker, skip
            continue
        footnotes.append({
            'marker': match.group(0),
            'number': match.group(1),
            'position': match.start(),
            'type': 'parentheses'
        })
    
    return footnotes


def extract_table_reference(hyperlink: str) -> Optional[int]:
    """
    Extract table index from hyperlink.
    
    Args:
        hyperlink: Hyperlink string (e.g., "#/tables/5")
        
    Returns:
        Table index or None
    """
    if '#/tables/' in hyperlink:
        try:
            idx = int(hyperlink.split('/')[-1])
            return idx
        except (ValueError, IndexError):
            return None
    return None


def extract_picture_reference(hyperlink: str) -> Optional[int]:
    """
    Extract picture index from hyperlink.
    
    Args:
        hyperlink: Hyperlink string (e.g., "#/pictures/3")
        
    Returns:
        Picture index or None
    """
    if '#/pictures/' in hyperlink:
        try:
            idx = int(hyperlink.split('/')[-1])
            return idx
        except (ValueError, IndexError):
            return None
    return None


# ============================================================================
# Text Processing Functions (GPT-inspired improvements)
# ============================================================================

def join_tokens(tokens: List[str]) -> str:
    """
    Spojí zoznam textových kúskov z Doclingu do jednej vety.
    Heuristické spájanie s riešením medzier a interpunkcie.
    
    Args:
        tokens: List of text tokens to join
        
    Returns:
        Joined text string
    """
    s = ""
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue

        if not s:
            s = tok
            continue

        # Čistá interpunkcia ide bez medzery
        if tok in [".", ",", ";", ":", ")", "(", "?", "!", "»", "«"]:
            s += tok
        # Token začína interpunkciou - tiež bez medzery
        elif tok[0] in ".,;:)]!?":
            s += tok
        # Predchádzajúci končí "(" - tiež bez medzery
        elif s.endswith("("):
            s += tok
        else:
            s += " " + tok

    # Drobné opravy špecifické pre právne texty
    s = s.replace("120 )", "120)")
    s = s.replace("č. 1 .", "č. 1.")
    s = re.sub(r"\s+([,.])", r"\1", s)

    return s


def find_full_paragraph_block(texts: List[dict], paragraf_number: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Nájde blok "plného" paragrafu (nie len štruktúrne značky).
    
    Rozlišuje medzi "štruktúrnou" reprezentáciou (hyperlinky, značky) 
    a "plnou" lineárnou reprezentáciou (skutočný text zákona).
    
    Heuristika: "§ N" + názov + "(1)" = začiatok plného paragrafu.
    
    Args:
        texts: List of text elements from doc.texts
        paragraf_number: Paragraph number to find
        
    Returns:
        Tuple of (start_idx, end_idx) or (None, None) if not found
    """
    target = f"§ {paragraf_number}"
    candidate_indices = [i for i, t in enumerate(texts) if t.get("text") == target]

    for i in candidate_indices:
        # Ochrana pred IndexError
        if i + 2 >= len(texts):
            continue

        next1 = texts[i + 1].get("text", "")
        next2 = texts[i + 2].get("text", "")

        # Heuristika: hľadáme variant, kde za "§ N" nasleduje názov a potom "(1)"
        if next2.strip() == "(1)":
            # Našli sme začiatok "plného" paragrafu
            start = i

            # Koniec bloku = ďalší riadok, ktorý začína "§ " (nový paragraf)
            end = start + 1
            while end < len(texts):
                tx = texts[end].get("text", "")
                if tx.startswith("§ ") and end != start:
                    break
                end += 1

            return start, end

    return None, None


def extract_odsek_text(
    texts: List[dict],
    par_start: int,
    par_end: int,
    odsek_number: int
) -> Optional[str]:
    """
    Z bloku paragrafu [par_start, par_end) vytiahne text konkrétneho odseku.
    
    Odsek je definovaný tokenom "(n)" a končí pred ďalším odsekom alebo koncom bloku.
    
    Args:
        texts: List of text elements from doc.texts
        par_start: Start index of paragraph block
        par_end: End index of paragraph block
        odsek_number: Odsek number to extract
        
    Returns:
        Extracted odsek text or None if not found
    """
    # Nájdeme všetky pozície "(1)", "(2)", ...
    label_positions: List[Tuple[int, int]] = []
    for i in range(par_start, par_end):
        txt = texts[i].get("text", "").strip()
        m = re.fullmatch(r"\((\d+)\)", txt)
        if m:
            label_positions.append((int(m.group(1)), i))

    label_positions.sort()  # Zoradíme podľa čísla odseku

    # Find start index for requested odsek
    start_idx = None
    for num, idx in label_positions:
        if num == odsek_number:
            start_idx = idx
            break

    if start_idx is None:
        return None

    # End index = index ďalšieho odseku alebo koniec paragrafu
    end_idx = par_end
    for num, idx in label_positions:
        if idx > start_idx:
            end_idx = idx
            break

    tokens = [texts[i].get("text", "") for i in range(start_idx, end_idx)]
    return join_tokens(tokens)


# ============================================================================
# Table Processing Functions (GPT-inspired improvements)
# ============================================================================

def decode_character_codes(text: str) -> str:
    """
    Decode character codes like /c90/c65... to actual text.
    Pattern: /c followed by decimal number represents Unicode code point.
    
    This is needed because some PDF pages use custom font encoding where
    characters are stored as /cXX codes instead of proper Unicode.
    
    Some PDFs use Windows-1250 (Central European) encoding values for
    Slovak characters, so we map those specially.
    """
    if not text or '/c' not in text:
        return text
    
    # Windows-1250 to Unicode mapping for Slovak characters
    # These code points in PDF map to Slovak/Czech characters
    win1250_to_unicode = {
        138: 'Š',  # /c138 -> Š
        140: 'Ś',  # /c140 -> Ś  
        141: 'Ť',  # /c141 -> Ť
        142: 'Ž',  # /c142 -> Ž
        150: '–',  # /c150 -> en dash
        154: 'š',  # /c154 -> š
        156: 'ś',  # /c156 -> ś
        157: 'ť',  # /c157 -> ť
        158: 'ž',  # /c158 -> ž
        159: 'ź',  # /c159 -> ź
        165: 'Ą',  # /c165 -> Ą
        169: '©',  # /c169 -> ©
        175: 'Ż',  # /c175 -> Ż
        179: 'ł',  # /c179 -> ł
        185: 'ą',  # /c185 -> ą
        188: 'Ľ',  # /c188 -> Ľ
        189: '˝',  # /c189 -> double acute
        190: 'ľ',  # /c190 -> ľ
        191: 'ż',  # /c191 -> ż
        192: 'Ŕ',  # /c192 -> Ŕ
        193: 'Á',  # /c193 -> Á
        194: 'Â',  # /c194 -> Â
        195: 'Ă',  # /c195 -> Ă
        196: 'Ä',  # /c196 -> Ä
        197: 'Ĺ',  # /c197 -> Ĺ
        198: 'Ć',  # /c198 -> Ć
        199: 'Ç',  # /c199 -> Ç
        200: 'Č',  # /c200 -> Č
        201: 'É',  # /c201 -> É
        202: 'Ę',  # /c202 -> Ę
        203: 'Ë',  # /c203 -> Ë
        204: 'Ě',  # /c204 -> Ě
        205: 'Í',  # /c205 -> Í
        206: 'Î',  # /c206 -> Î
        207: 'Ď',  # /c207 -> Ď
        208: 'Đ',  # /c208 -> Đ
        209: 'Ń',  # /c209 -> Ń
        210: 'Ň',  # /c210 -> Ň
        211: 'Ó',  # /c211 -> Ó
        212: 'Ô',  # /c212 -> Ô
        213: 'Ő',  # /c213 -> Ő
        214: 'Ö',  # /c214 -> Ö
        215: '×',  # /c215 -> multiplication sign
        216: 'Ř',  # /c216 -> Ř
        217: 'Ů',  # /c217 -> Ů
        218: 'Ú',  # /c218 -> Ú
        219: 'Ű',  # /c219 -> Ű
        220: 'Ü',  # /c220 -> Ü
        221: 'Ý',  # /c221 -> Ý
        222: 'Ţ',  # /c222 -> Ţ
        223: 'ß',  # /c223 -> ß
        224: 'ŕ',  # /c224 -> ŕ
        225: 'á',  # /c225 -> á
        226: 'â',  # /c226 -> â
        227: 'ă',  # /c227 -> ă
        228: 'ä',  # /c228 -> ä
        229: 'ĺ',  # /c229 -> ĺ
        230: 'ć',  # /c230 -> ć
        231: 'ç',  # /c231 -> ç
        232: 'č',  # /c232 -> č
        233: 'é',  # /c233 -> é
        234: 'ę',  # /c234 -> ę
        235: 'ë',  # /c235 -> ë
        236: 'ě',  # /c236 -> ě
        237: 'í',  # /c237 -> í
        238: 'î',  # /c238 -> î
        239: 'ď',  # /c239 -> ď
        240: 'đ',  # /c240 -> đ
        241: 'ń',  # /c241 -> ń
        242: 'ň',  # /c242 -> ň
        243: 'ó',  # /c243 -> ó
        244: 'ô',  # /c244 -> ô
        245: 'ő',  # /c245 -> ő
        246: 'ö',  # /c246 -> ö
        247: '÷',  # /c247 -> division sign
        248: 'ř',  # /c248 -> ř
        249: 'ů',  # /c249 -> ů
        250: 'ú',  # /c250 -> ú
        251: 'ű',  # /c251 -> ű
        252: 'ü',  # /c252 -> ü
        253: 'ý',  # /c253 -> ý
        254: 'ţ',  # /c254 -> ţ
        255: '˙',  # /c255 -> dot above
    }
    
    def replace_code(match):
        code_str = match.group(1)
        try:
            code_point = int(code_str)
            # Check if it's a Windows-1250 character
            if code_point in win1250_to_unicode:
                return win1250_to_unicode[code_point]
            return chr(code_point)
        except (ValueError, OverflowError):
            return match.group(0)  # Return original if can't decode
    
    # Replace /c followed by numbers with decoded character
    decoded = re.sub(r'/c(\d+)', replace_code, text)
    return decoded


def table_to_rows_from_grid(table: Any) -> List[List[str]]:
    """
    Prevedie Docling tabuľku na zoznam riadkov so stringami priamo z data.grid.
    
    Rýchlejší a priamejší prístup ako export_to_dataframe.
    Decodes /cXX character codes from PDF fonts with custom encoding.
    
    Args:
        table: Table object from doc.tables
        
    Returns:
        List of rows, each row is a list of cell strings
    """
    rows: List[List[str]] = []
    
    if not hasattr(table, 'data') or not hasattr(table.data, 'grid'):
        return rows
    
    grid = table.data.grid
    for row in grid:
        # Pre istotu zoradíme bunky podľa start_col_offset_idx
        # Bunky môžu byť objekty alebo dict
        def get_col_idx(cell):
            if hasattr(cell, 'start_col_offset_idx'):
                return getattr(cell, 'start_col_offset_idx', 0)
            elif isinstance(cell, dict):
                return cell.get("start_col_offset_idx", 0)
            else:
                return 0
        
        def get_cell_text(cell):
            if hasattr(cell, 'text'):
                text = getattr(cell, 'text', '').strip()
            elif isinstance(cell, dict):
                text = cell.get("text", "").strip()
            else:
                text = str(cell).strip()
            
            # Decode /cXX character codes if present
            if '/c' in text:
                text = decode_character_codes(text)
            return text
        
        cells = sorted(row, key=get_col_idx)
        rows.append([get_cell_text(c) for c in cells])
    
    return rows


def detect_table_in_text(text: str) -> Optional[Tuple[str, int, int]]:
    """
    Detekuje textovú reprezentáciu tabuľky v texte.
    
    Hľadá vzory:
    - Hlavička: text, číslo = hodnota. (napr. "Odpisová skupina, 1 = Doba odpisovania.")
    - Dátové riadky: číslo, číslo = hodnota. (napr. "0, 1 = 2 roky.", "1, 1 = 1/2.")
    
    Args:
        text: Text, v ktorom sa hľadá tabuľka
        
    Returns:
        Tuple (table_text, start_pos, end_pos) alebo None ak sa tabuľka nenašla
    """
    if not text:
        return None
    
    # Vzor pre hlavičku stĺpcov: text, číslo = hodnota.
    # Podporuje slovenské znaky a rôzne formáty
    header_pattern = r'([A-ZÁÉÍÓÚÝŽŠČŤĎĽŇa-záéíóúýžščťďľň\s\*\*]+,?\s*\d+\s*=\s*[^.]+\.)'
    
    # Vzor pre dátové riadky: číslo, číslo = hodnota.
    # Hodnota môže byť: číslo, zlomok (1/2), percento (29 %), text
    data_row_pattern = r'(\d+,\s*\d+\s*=\s*[^.]+\.)'
    
    # Hľadanie hlavičky
    header_match = re.search(header_pattern, text)
    if not header_match:
        return None
    
    # Hľadanie dátových riadkov po hlavičke
    start_pos = header_match.start()
    text_after_header = text[start_pos:]
    
    # Nájdeme všetky dátové riadky
    data_matches = list(re.finditer(data_row_pattern, text_after_header))
    
    # Potrebujeme aspoň 2 dátové riadky na to, aby to bola tabuľka
    if len(data_matches) < 2:
        return None
    
    # Koniec tabuľky je po poslednom dátovom riadku
    end_pos = start_pos + data_matches[-1].end()
    
    # Extrahujeme text tabuľky
    table_text = text[start_pos:end_pos]
    
    return (table_text, start_pos, end_pos)


def extract_table_from_text(text: str) -> Tuple[str, Optional[str]]:
    """
    Extrahuje tabuľku z textu a vráti text bez tabuľky a text tabuľky.
    
    Args:
        text: Text, z ktorého sa má extrahovať tabuľka
        
    Returns:
        Tuple (text_without_table, table_text)
        - text_without_table: Text bez tabuľky
        - table_text: Extrahovaný text tabuľky alebo None
    """
    result = detect_table_in_text(text)
    if not result:
        return (text, None)
    
    table_text, start_pos, end_pos = result
    
    # Odstránime tabuľku z textu
    text_before = text[:start_pos].rstrip()
    text_after = text[end_pos:].lstrip()
    
    # Spojíme text pred a po tabuľke
    text_without_table = text_before
    if text_before and text_after:
        # Ak je text pred aj po, pridáme medzeru
        text_without_table += " " + text_after
    elif text_after:
        text_without_table = text_after
    
    return (text_without_table, table_text)


def normalize_text_for_matching(text: str) -> str:
    """
    Normalizuje text pre porovnanie: lowercase, odstránenie diakritiky, normalizácia whitespace.
    
    Args:
        text: Text na normalizáciu
        
    Returns:
        Normalizovaný text
    """
    # Lowercase
    text = text.lower()
    
    # Odstránenie diakritiky
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Normalizácia whitespace
    text = ' '.join(text.split())
    
    return text


def parse_table_text(table_text: str) -> Tuple[List[str], List[str]]:
    """
    Parsuje textovú reprezentáciu tabuľky na hlavičku a dátové riadky.
    
    Args:
        table_text: Textová reprezentácia tabuľky
        
    Returns:
        Tuple (headers, data_rows)
        - headers: Zoznam hlavičiek stĺpcov
        - data_rows: Zoznam dátových riadkov
    """
    headers = []
    data_rows = []
    
    # Vzor pre hlavičku
    header_pattern = r'([A-ZÁÉÍÓÚÝŽŠČŤĎĽŇa-záéíóúýžščťďľň\s\*\*]+,?\s*\d+\s*=\s*[^.]+\.)'
    # Vzor pre dátový riadok
    data_row_pattern = r'(\d+,\s*\d+\s*=\s*[^.]+\.)'
    
    # Nájdeme všetky hlavičky
    header_matches = re.finditer(header_pattern, table_text)
    for match in header_matches:
        header = match.group(1).strip()
        # Odstránime koncovú bodku
        if header.endswith('.'):
            header = header[:-1]
        headers.append(header)
    
    # Nájdeme všetky dátové riadky
    data_matches = re.finditer(data_row_pattern, table_text)
    for match in data_matches:
        row = match.group(1).strip()
        # Odstránime koncovú bodku
        if row.endswith('.'):
            row = row[:-1]
        data_rows.append(row)
    
    return (headers, data_rows)


def find_table_by_text_match(table_text: str, doc: DoclingDocument, exclude_indices: Optional[List[int]] = None) -> Optional[int]:
    """
    Nájde zodpovedajúcu tabuľku v doc.tables podľa textovej reprezentácie.
    
    Args:
        table_text: Textová reprezentácia tabuľky
        doc: DoclingDocument
        exclude_indices: Zoznam indexov tabuliek, ktoré sa majú vylúčiť z hľadania
        
    Returns:
        Index nájdenej tabuľky alebo None
    """
    if not table_text or not hasattr(doc, 'tables') or not doc.tables:
        return None
    
    if exclude_indices is None:
        exclude_indices = []
    
    # Parsujeme textovú reprezentáciu tabuľky
    headers, data_rows = parse_table_text(table_text)
    
    if not headers and not data_rows:
        return None
    
    # Normalizujeme hlavičky a dátové riadky
    normalized_headers = [normalize_text_for_matching(h) for h in headers]
    normalized_data_rows = [normalize_text_for_matching(r) for r in data_rows]
    
    best_match_idx = None
    best_match_score = 0
    
    # Prejdeme všetky tabuľky
    for table_idx, table in enumerate(doc.tables):
        if table_idx in exclude_indices:
            continue
        
        try:
            # Získame riadky tabuľky
            rows = table_to_rows_from_grid(table)
            if not rows or len(rows) == 0:
                continue
            
            # Prvý riadok je hlavička
            table_header = rows[0]
            table_data_rows = rows[1:] if len(rows) > 1 else []
            
            # Normalizujeme hlavičku tabuľky
            normalized_table_header = [normalize_text_for_matching(str(cell)) for cell in table_header]
            table_header_text = ' '.join(normalized_table_header)
            
            # Skóre pre zhodu
            score = 0
            
            # Porovnanie hlavičky
            if normalized_headers:
                # Porovnáme každú hlavičku z textu s hlavičkou tabuľky
                header_match_count = 0
                for norm_header in normalized_headers:
                    # Kontrolujeme, či hlavička obsahuje kľúčové slová
                    header_words = set(norm_header.split())
                    table_header_words = set(table_header_text.split())
                    if header_words and table_header_words:
                        # Aspoň 50% zhodných slov
                        common_words = header_words & table_header_words
                        if len(common_words) >= len(header_words) * 0.5:
                            header_match_count += 1
                
                if header_match_count > 0:
                    score += 10 * header_match_count  # Bonus za zhodu hlavičky
            
            # Porovnanie dátových riadkov
            if normalized_data_rows and table_data_rows:
                # Normalizujeme dátové riadky tabuľky
                normalized_table_data_rows = []
                for row in table_data_rows:
                    # Skonvertujeme riadok na formát "číslo, číslo = hodnota"
                    row_str_parts = []
                    for i, cell in enumerate(row):
                        cell_str = str(cell).strip()
                        if cell_str:
                            row_str_parts.append(cell_str)
                    if row_str_parts:
                        # Vytvoríme formát podobný textovej reprezentácii
                        # Pre jednoduchosť porovnáme len hodnoty
                        row_normalized = normalize_text_for_matching(' '.join(row_str_parts))
                        normalized_table_data_rows.append(row_normalized)
                
                # Porovnáme každý dátový riadok z textu s riadkami tabuľky
                matched_rows = 0
                for norm_data_row in normalized_data_rows:
                    # Extrahujeme hodnotu z formátu "číslo, číslo = hodnota"
                    # Hľadáme hodnotu po "="
                    if '=' in norm_data_row:
                        value_part = norm_data_row.split('=', 1)[1].strip()
                        # Porovnáme s hodnotami v tabuľke
                        for table_row in normalized_table_data_rows:
                            if value_part in table_row or table_row in value_part:
                                matched_rows += 1
                                break
                
                # Aspoň 2 zhodné riadky
                if matched_rows >= 2:
                    score += matched_rows * 5  # Bonus za každý zhodný riadok
            
            # Ak máme dobré skóre, uložíme to
            if score > best_match_score:
                best_match_score = score
                best_match_idx = table_idx
        
        except Exception as e:
            # Ak sa vyskytne chyba pri spracovaní tabuľky, pokračujeme
            log_progress("DEBUG", f"Error processing table {table_idx} for matching: {e}")
            continue
    
    # Vrátime najlepšiu zhodu, ak má dostatočné skóre
    if best_match_score >= 10:  # Minimálne skóre pre akceptovanie zhodu
        return best_match_idx
    
    return None


# ============================================================================
# Table Classification - Metadata vs Legal Tables
# ============================================================================

# Vzory pre metadata tabuľky (nepatria do právnej štruktúry)
METADATA_TABLE_PATTERNS = [
    re.compile(r"^História$", re.IGNORECASE),
    re.compile(r"^Číslo predpisu", re.IGNORECASE),
    re.compile(r"^Predpis ruší", re.IGNORECASE),
    re.compile(r"^\d+/\d{4}\s+Z\.\s*z\.", re.IGNORECASE),  # "366/1999 Z. z."
    re.compile(r"^Súvisiace predpisy", re.IGNORECASE),
    re.compile(r"^Vykonávacie predpisy", re.IGNORECASE),
    re.compile(r"^Novelizované znenie", re.IGNORECASE),
    re.compile(r"^Účinnosť od", re.IGNORECASE),
    re.compile(r"^Dátum účinnosti", re.IGNORECASE),
    re.compile(r"^Predpis mení", re.IGNORECASE),
    re.compile(r"^Predpis dopĺňa", re.IGNORECASE),
]

# Vzory pre právne tabuľky (pozitívna identifikácia)
LEGAL_TABLE_PATTERNS = [
    re.compile(r"Odpisová skupina", re.IGNORECASE),
    re.compile(r"Doba odpisovania", re.IGNORECASE),
    re.compile(r"Ročný odpis", re.IGNORECASE),
    re.compile(r"Koeficient", re.IGNORECASE),
    re.compile(r"Sadzba", re.IGNORECASE),
    re.compile(r"Príloha č\.", re.IGNORECASE),
    re.compile(r"Počet vyživovaných", re.IGNORECASE),
    re.compile(r"Daňový bonus", re.IGNORECASE),
    re.compile(r"Nezdaniteľná", re.IGNORECASE),
    re.compile(r"Základ dane", re.IGNORECASE),
]


def get_table_first_cells_text(table: Any, max_cells: int = 10) -> List[str]:
    """
    Získa text z prvých buniek tabuľky pre klasifikáciu.
    
    Args:
        table: Table object from doc.tables
        max_cells: Maximálny počet buniek na kontrolu
        
    Returns:
        Zoznam textov z prvých buniek
    """
    texts = []
    data = getattr(table, 'data', None)
    if not data:
        return texts
    
    table_cells = getattr(data, 'table_cells', None)
    if table_cells is None and hasattr(data, 'get'):
        table_cells = data.get('table_cells', [])
    
    if not table_cells:
        return texts
    
    for i, cell in enumerate(table_cells[:max_cells]):
        if isinstance(cell, dict):
            text = cell.get('text', '')
        else:
            text = getattr(cell, 'text', '')
        if text:
            texts.append(text.strip())
    
    return texts


def is_metadata_table(table: Any, doc: DoclingDocument, table_idx: int) -> bool:
    """
    Určí, či tabuľka je metadata (nepatrí do právnej štruktúry).
    
    Kritériá:
    1. Obsah prvých buniek obsahuje metadata vzory (História, Číslo predpisu, atď.)
    2. Obsah prvých buniek obsahuje čísla zákonov (napr. "366/1999 Z. z.")
    3. Tabuľka je pred prvým paragrafom v dokumente
    
    Args:
        table: Tabuľka z doc.tables
        doc: DoclingDocument
        table_idx: Index tabuľky
        
    Returns:
        True ak je to metadata tabuľka
    """
    # Získame text z prvých buniek tabuľky
    first_cells_text = get_table_first_cells_text(table)
    
    if not first_cells_text:
        return False
    
    # Kontrola či je to právna tabuľka (pozitívna identifikácia má prioritu)
    for pattern in LEGAL_TABLE_PATTERNS:
        for cell_text in first_cells_text:
            if pattern.search(cell_text):
                log_progress("DEBUG", f"Table {table_idx} identified as LEGAL (pattern: {pattern.pattern}, cell: {cell_text[:50]})")
                return False  # Je to právna tabuľka
    
    # Kontrola metadata vzorov
    for pattern in METADATA_TABLE_PATTERNS:
        for cell_text in first_cells_text:
            if pattern.search(cell_text):
                log_progress("INFO", f"Table {table_idx} identified as METADATA (pattern: {pattern.pattern}, cell: {cell_text[:50]})")
                return True
    
    # Kontrola pozície v dokumente - ak parent text index je veľmi nízky
    parent = getattr(table, 'parent', None)
    if parent:
        parent_ref = None
        if isinstance(parent, dict) and '$ref' in parent:
            parent_ref = str(parent['$ref'])
        elif hasattr(parent, 'cref'):
            parent_ref = str(parent.cref)
        
        if parent_ref and '/texts/' in parent_ref:
            try:
                text_idx = int(parent_ref.split('/texts/')[-1])
                # Ak je parent text index veľmi nízky (pred hlavným textom)
                # a nie je to právna tabuľka, je to pravdepodobne metadata
                if text_idx < 50:
                    log_progress("INFO", f"Table {table_idx} identified as METADATA (low parent text idx: {text_idx})")
                    return True
            except ValueError:
                pass
    
    return False


def classify_all_tables(doc: DoclingDocument) -> Tuple[List[int], List[int]]:
    """
    Klasifikuje všetky tabuľky v dokumente na právne a metadata.
    
    Args:
        doc: DoclingDocument
        
    Returns:
        Tuple (legal_table_indices, metadata_table_indices)
    """
    legal_tables = []
    metadata_tables = []
    
    if not hasattr(doc, 'tables') or not doc.tables:
        return legal_tables, metadata_tables
    
    for table_idx, table in enumerate(doc.tables):
        if is_metadata_table(table, doc, table_idx):
            metadata_tables.append(table_idx)
        else:
            legal_tables.append(table_idx)
    
    log_progress("INFO", f"Table classification: {len(legal_tables)} legal, {len(metadata_tables)} metadata")
    return legal_tables, metadata_tables


# ============================================================================
# Table Context Assignment - Claude-inspired filtering + GPT global state
# ============================================================================

# Regex pre marker odseku: (1), (2), atď.
SUBSECTION_MARKER_PATTERN = re.compile(r'^\((\d+)\)$')

# Regex pre čistý marker paragrafu: § 27, § 27a
SECTION_PURE_PATTERN = re.compile(r'^§\s+(\d+[a-z]?)$')

# Regex pre paragraf s názvom: § 27 Názov (slovenské veľké písmená)
SECTION_WITH_TITLE_PATTERN = re.compile(
    r'^§\s+(\d+[a-z]?)\s+([A-ZÁÄČĎÉÍĽŇÓÔŔŠŤÚÝŽ].*)$'
)


def is_pure_section_marker(text: str) -> bool:
    """
    Kontroluje či text je čistý marker paragrafu (nie odkaz).
    
    KRITICKÉ: Táto funkcia je kľúčová pre správne fungovanie!
    Musí odlíšiť skutočné začiatky paragrafov od odkazov.
    
    Akceptuje:
        - "§ 27" - samostatné číslo
        - "§ 27a" - s písmenom
        - "§ 27 Rovnomerné odpisovanie" - s názvom
        
    Odmietne (odkazy):
        - "§ 26 ods. 1" - obsahuje "ods."
        - "podľa § 27" - nezačína s §
        - "ustanovenia § 22 až 29" - nezačína s §
        - "§ 26 ods. 1 takto:" - obsahuje "ods."
    
    Args:
        text: Text na kontrolu
        
    Returns:
        True ak je to skutočný marker paragrafu
    """
    text = text.strip()
    
    # Musí začínať priamo s §
    if not text.startswith('§'):
        return False
    
    # KRITICKÉ: Nesmie obsahovať "ods." - to je odkaz!
    if 'ods.' in text.lower():
        return False
    
    # Čistý "§ N" alebo "§ Na"
    if SECTION_PURE_PATTERN.match(text):
        return True
    
    # "§ N Názov" s veľkým písmenom na začiatku názvu
    if SECTION_WITH_TITLE_PATTERN.match(text):
        return True
    
    return False


def extract_section_number_from_marker(text: str) -> Optional[str]:
    """Extrahuje číslo paragrafu z textu markera."""
    text = text.strip()
    match = SECTION_PURE_PATTERN.match(text)
    if match:
        return match.group(1)
    match = SECTION_WITH_TITLE_PATTERN.match(text)
    if match:
        return match.group(1)
    return None


def extract_subsection_number_from_marker(text: str) -> Optional[str]:
    """Extrahuje číslo odseku z textu markera."""
    match = SUBSECTION_MARKER_PATTERN.match(text.strip())
    if match:
        return match.group(1)
    return None


def build_table_context_map(doc: DoclingDocument) -> Dict[int, Tuple[Optional[str], Optional[str]]]:
    """
    Vytvorí mapu kontextu pre všetky tabuľky rekurzívnym prechádzaním body.children.
    
    Kombinuje:
    - GPT prístup: Lineárny prechod s globálnym stavom
    - Claude prístup: Filtrovanie odkazov (is_pure_section_marker)
    - Rekurzívne prechádzanie: Správne spracovanie hierarchie
    
    Args:
        doc: DoclingDocument
        
    Returns:
        Dict mapujúci table_idx na (section_number, subsection_number)
    """
    current_section: Optional[str] = None
    current_subsection: Optional[str] = None
    table_context: Dict[int, Tuple[Optional[str], Optional[str]]] = {}
    
    def get_ref_from_child(child) -> Optional[str]:
        """Získa referenciu z child elementu."""
        if hasattr(child, 'cref'):
            return str(child.cref)
        elif isinstance(child, dict) and '$ref' in child:
            return str(child['$ref'])
        elif hasattr(child, 'get_ref'):
            return str(child.get_ref())
        return None
    
    def process_element(ref: str, parent_is_group: bool = False):
        """Rekurzívne spracuje element a jeho children.
        
        Args:
            ref: Referencia na element
            parent_is_group: True ak parent je group (čo znamená, že sme vnútri inline referencie)
        """
        nonlocal current_section, current_subsection
        
        if ref.startswith('#/texts/'):
            try:
                idx = int(ref.split('/')[-1])
            except ValueError:
                return
            
            if idx >= len(doc.texts):
                return
            
            text_elem = doc.texts[idx]
            text = getattr(text_elem, 'text', '') or ''
            
            # KRITICKÉ: Kontrola či tento text element je skutočný marker alebo inline referencia
            # Ak parent je group, toto je inline referencia v texte a NIE je to skutočný marker paragrafu
            # Skutočné markery majú parent ako texts/... (hlavný dokument), nie groups/...
            is_inside_inline_ref = parent_is_group
            
            # Kontrola ČISTÉHO markera paragrafu (Claude filter)
            # Ale len ak nie sme vnútri inline referencie!
            if not is_inside_inline_ref and is_pure_section_marker(text):
                new_section = extract_section_number_from_marker(text)
                if new_section:
                    current_section = new_section
                    current_subsection = None  # Reset odseku pri novom paragrafe
                    log_progress("DEBUG", f"Found section marker: § {current_section} at text idx {idx}")
            
            # Kontrola odseku - len ak nie sme vnútri inline referencie
            if not is_inside_inline_ref:
                subsection = extract_subsection_number_from_marker(text)
                if subsection:
                    current_subsection = subsection
                    log_progress("DEBUG", f"Found subsection marker: ({current_subsection}) at text idx {idx}")
            
            # Rekurzívne spracuj children
            # Children text elementu sú stále v kontexte hlavného dokumentu (nie inline ref)
            children = getattr(text_elem, 'children', None) or []
            for child in children:
                child_ref = get_ref_from_child(child)
                if child_ref:
                    # Ak child je group, markery vnútri neho sú inline referencie
                    child_is_group = child_ref.startswith('#/groups/')
                    process_element(child_ref, parent_is_group=child_is_group)
        
        elif ref.startswith('#/tables/'):
            try:
                idx = int(ref.split('/')[-1])
            except ValueError:
                return
            
            table_context[idx] = (current_section, current_subsection)
            log_progress("DEBUG", f"Assigned table {idx} to § {current_section} ods. {current_subsection}")
        
        elif ref.startswith('#/groups/'):
            try:
                idx = int(ref.split('/')[-1])
            except ValueError:
                return
            
            if idx >= len(doc.groups):
                return
            
            group = doc.groups[idx]
            children = getattr(group, 'children', None) or []
            for child in children:
                child_ref = get_ref_from_child(child)
                if child_ref:
                    # Všetky children vnútri group sú inline referencie
                    process_element(child_ref, parent_is_group=True)
        
        elif ref.startswith('#/pictures/'):
            # Obrázky ignorujeme
            pass
    
    # Začni od body.children
    if hasattr(doc, 'body') and hasattr(doc.body, 'children'):
        log_progress("INFO", f"Building table context map from {len(doc.body.children)} body children...")
        for child in doc.body.children:
            child_ref = get_ref_from_child(child)
            if child_ref:
                # Top-level children nie sú inline referencie
                process_element(child_ref, parent_is_group=False)
    
    log_progress("INFO", f"Table context map built: {len(table_context)} tables assigned")
    return table_context


def get_tables_for_section_subsection(
    table_context_map: Dict[int, Tuple[Optional[str], Optional[str]]],
    section_number: str,
    subsection_number: Optional[str] = None,
    legal_table_indices: Optional[List[int]] = None
) -> List[int]:
    """
    Nájde všetky tabuľky pre daný paragraf a odsek.
    
    Args:
        table_context_map: Mapa z build_table_context_map
        section_number: Číslo paragrafu (napr. "26")
        subsection_number: Číslo odseku (napr. "1") alebo None pre všetky odseky
        legal_table_indices: Zoznam indexov právnych tabuliek (vylúči metadata)
        
    Returns:
        Zoznam indexov tabuliek
    """
    result = []
    
    for table_idx, (sec, subsec) in table_context_map.items():
        # Ak máme filter na právne tabuľky, použijeme ho
        if legal_table_indices is not None and table_idx not in legal_table_indices:
            continue
        
        if sec == section_number:
            if subsection_number is None or subsec == subsection_number:
                result.append(table_idx)
    
    return sorted(result)


def get_table_position_in_body_children(doc: DoclingDocument, table_idx: int) -> Optional[int]:
    """
    Nájde pozíciu tabuľky v body.children.
    
    Args:
        doc: DoclingDocument
        table_idx: Index tabuľky v doc.tables
        
    Returns:
        Pozícia tabuľky v body.children alebo None ak sa nenašla
    """
    if not hasattr(doc, 'body') or not hasattr(doc.body, 'children'):
        return None
    
    table_ref = f"#/tables/{table_idx}"
    
    for pos, child in enumerate(doc.body.children):
        if hasattr(child, 'cref'):
            if child.cref == table_ref:
                return pos
        elif isinstance(child, dict) and '$ref' in child:
            if child['$ref'] == table_ref:
                return pos
        elif hasattr(child, 'get_ref'):
            if str(child.get_ref()) == table_ref:
                return pos
    
    return None


def get_unit_text_positions_in_body_children(
    doc: DoclingDocument,
    unit_start_idx: int,
    unit_end_idx: int
) -> Tuple[Optional[int], Optional[int]]:
    """
    Nájde pozície text elementov jednotky v body.children.
    
    Args:
        doc: DoclingDocument
        unit_start_idx: Start index of unit in doc.texts
        unit_end_idx: End index of unit in doc.texts
        
    Returns:
        Tuple (start_pos, end_pos) v body.children alebo (None, None) ak sa nenašli
    """
    if not hasattr(doc, 'body') or not hasattr(doc.body, 'children'):
        log_progress("DEBUG", f"get_unit_text_positions_in_body_children: doc.body.children not available")
        return (None, None)
    
    start_pos = None
    end_pos = None
    
    # Nájdeme prvý text element jednotky v body.children
    for pos, child in enumerate(doc.body.children):
        child_ref = None
        if hasattr(child, 'cref'):
            child_ref = child.cref
        elif isinstance(child, dict) and '$ref' in child:
            child_ref = child['$ref']
        elif hasattr(child, 'get_ref'):
            child_ref = str(child.get_ref())
        
        if child_ref and '/texts/' in child_ref:
            try:
                text_idx_str = child_ref.split('/texts/')[-1]
                text_idx = int(text_idx_str)
                
                if unit_start_idx <= text_idx < unit_end_idx:
                    if start_pos is None:
                        start_pos = pos
                    end_pos = pos  # Aktualizujeme end_pos pre každý text element v rozsahu
            except (ValueError, IndexError):
                pass
    
    if start_pos is None:
        log_progress("DEBUG", f"get_unit_text_positions_in_body_children: No text elements found in body.children for range [{unit_start_idx}, {unit_end_idx}), checked {len(doc.body.children)} children")
    
    return (start_pos, end_pos)


def find_tables_for_annex(doc: DoclingDocument, annex_start_idx: int, annex_end_idx: int, annex_marker_text_elem=None) -> List[int]:
    """
    Find tables that belong to an annex.
    
    This is a convenience wrapper around find_tables_for_unit specifically for annexes.
    
    Args:
        doc: DoclingDocument
        annex_start_idx: Index where annex marker was found in doc.texts
        annex_end_idx: Index where next annex starts (or end of document)
        annex_marker_text_elem: The text element of the annex marker (to get its parent)
        
    Returns:
        List of table indices that belong to this annex
    """
    log_progress("DEBUG", f"find_tables_for_annex: Searching for tables between idx {annex_start_idx} and {annex_end_idx}, total tables: {len(doc.tables) if hasattr(doc, 'tables') else 0}")
    
    # Extract annex text for heuristic matching
    annex_text = " ".join([
        t.get("text", "") if isinstance(t, dict) else str(t)
        for t in doc.texts[annex_start_idx:annex_end_idx]
    ])
    
    # Use general find_tables_for_unit function
    table_indices = find_tables_for_unit(
        doc, annex_start_idx, annex_end_idx,
        annex_marker_text_elem, annex_text
    )
    
    log_progress("DEBUG", f"find_tables_for_annex: Found {len(table_indices)} tables: {table_indices}")
    return table_indices


def find_tables_for_unit(
    doc: DoclingDocument,
    unit_start_idx: int,
    unit_end_idx: int,
    unit_marker_text_elem=None,
    unit_text: Optional[str] = None
) -> List[int]:
    """
    Všeobecná funkcia na nájdenie tabuliek pre akúkoľvek jednotku
    (paragraf, odsek, písmeno, príloha).
    
    Používa kombinovanú stratégiu:
    1a. Pozícia v body.children - ak tabuľka je medzi text elementmi jednotky
    1b. Spoločný parent + pozícia - ak tabuľka a jednotka majú rovnaký parent a tabuľka je v sekvencii jednotky
    1c. Parent text index v rozsahu - ak tabuľka má parent text index v rozsahu jednotky
    
    Args:
        doc: DoclingDocument
        unit_start_idx: Start index of unit in doc.texts
        unit_end_idx: End index of unit in doc.texts
        unit_marker_text_elem: Text element marking the unit (for parent reference)
        unit_text: Optional text content of the unit (for heuristic matching)
        
    Returns:
        List of table indices that belong to this unit
    """
    table_indices = []
    
    if not hasattr(doc, 'tables') or not doc.tables:
        return table_indices
    
    # Získame pozície text elementov jednotky v body.children
    unit_start_pos, unit_end_pos = get_unit_text_positions_in_body_children(
        doc, unit_start_idx, unit_end_idx
    )
    
    log_progress("DEBUG", f"find_tables_for_unit: unit_start_pos={unit_start_pos}, unit_end_pos={unit_end_pos} for range [{unit_start_idx}, {unit_end_idx})")
    
    # Získame parent referenciu jednotky (ak existuje)
    unit_parent_ref = None
    if unit_marker_text_elem:
        parent = getattr(unit_marker_text_elem, 'parent', None)
        if parent:
            if isinstance(parent, dict) and '$ref' in parent:
                unit_parent_ref = str(parent['$ref'])
            elif hasattr(parent, 'cref'):
                unit_parent_ref = str(parent.cref)
            elif hasattr(parent, 'get_ref'):
                unit_parent_ref = str(parent.get_ref())
            else:
                unit_parent_ref = str(parent)
    
    log_progress("DEBUG", f"find_tables_for_unit: unit_parent_ref={unit_parent_ref}")
    
    # Prejdeme všetky tabuľky
    for table_idx, table in enumerate(doc.tables):
        parent = getattr(table, 'parent', None)
        if not parent:
            continue
        
        # Získame parent referenciu tabuľky
        table_parent_ref = None
        if isinstance(parent, dict) and '$ref' in parent:
            table_parent_ref = str(parent['$ref'])
        elif hasattr(parent, 'cref'):
            table_parent_ref = str(parent.cref)
        elif hasattr(parent, 'get_ref'):
            table_parent_ref = str(parent.get_ref())
        else:
            table_parent_ref = str(parent)
        
        # Strategy 1c: Check if table's parent text element is in unit region
        # This works for tables with parent text index in unit range
        if table_parent_ref and '/texts/' in table_parent_ref:
            try:
                text_idx_str = table_parent_ref.split('/texts/')[-1]
                text_idx = int(text_idx_str)
                # Check if this text index is in unit region
                if unit_start_idx <= text_idx < unit_end_idx:
                    table_indices.append(table_idx)
                    log_progress("DEBUG", f"Found table {table_idx} for unit (Strategy 1c: parent text idx: {text_idx} in range [{unit_start_idx}, {unit_end_idx}))")
                    continue  # Skip other strategies for this table
            except (ValueError, IndexError):
                pass
        
        # Strategy 1a: Check position in body.children
        # If table is between unit's text elements in body.children, it belongs to the unit
        if unit_start_pos is not None and unit_end_pos is not None:
            table_pos = get_table_position_in_body_children(doc, table_idx)
            if table_pos is not None:
                # Check if table is between unit's text elements
                # We allow some flexibility: table can be slightly before start or after end
                # (within 5 positions) to account for groups and other elements
                if unit_start_pos - 5 <= table_pos <= unit_end_pos + 5:
                    table_indices.append(table_idx)
                    log_progress("DEBUG", f"Found table {table_idx} for unit (Strategy 1a: position {table_pos} in body.children between [{unit_start_pos}, {unit_end_pos}])")
                    continue  # Skip other strategies for this table
        
        # Strategy 1b: Shared parent + position check
        # If table and unit have same parent, check if table is in unit's sequence
        if unit_parent_ref and table_parent_ref and table_parent_ref == unit_parent_ref:
            # Both have same parent, verify position in body.children
            if unit_start_pos is not None and unit_end_pos is not None:
                table_pos = get_table_position_in_body_children(doc, table_idx)
                if table_pos is not None:
                    # Check if table is in unit's sequence (with some flexibility)
                    if unit_start_pos - 5 <= table_pos <= unit_end_pos + 5:
                        table_indices.append(table_idx)
                        log_progress("DEBUG", f"Found table {table_idx} for unit (Strategy 1b: shared parent {unit_parent_ref}, position {table_pos} in range [{unit_start_pos}, {unit_end_pos}])")
                        continue
    
    # Strategy 2: Get parent of unit marker and find tables with same parent
    # BUT only if they haven't been found yet AND verify they're in unit range
    # DISABLED - using only Strategy 1 (PRIMARY)
    # if not table_indices:  # Only use this if Strategy 1 found nothing
    #     unit_parent_ref = None
    #     if unit_marker_text_elem:
    #         parent = getattr(unit_marker_text_elem, 'parent', None)
    #         if parent:
    #             if isinstance(parent, dict) and '$ref' in parent:
    #                 unit_parent_ref = str(parent['$ref'])
    #             elif hasattr(parent, 'cref'):
    #                 unit_parent_ref = str(parent.cref)
    #             elif hasattr(parent, 'get_ref'):
    #                 unit_parent_ref = str(parent.get_ref())
    #             else:
    #                 unit_parent_ref = str(parent)
    #     
    #     if unit_parent_ref:
    #         # Find all tables with the same parent, BUT verify they're in unit range
    #         for table_idx, table in enumerate(doc.tables):
    #             parent = getattr(table, 'parent', None)
    #             if not parent:
    #                 continue
    #             
    #             # Get parent reference
    #             table_parent_ref = None
    #             if isinstance(parent, dict) and '$ref' in parent:
    #                 table_parent_ref = str(parent['$ref'])
    #             elif hasattr(parent, 'cref'):
    #                 table_parent_ref = str(parent.cref)
    #             elif hasattr(parent, 'get_ref'):
    #                 table_parent_ref = str(parent.get_ref())
    #             else:
    #                 table_parent_ref = str(parent)
    #             
    #             if table_parent_ref and table_parent_ref == unit_parent_ref:
    #                 # VERIFY: Check if table's parent text is in unit range
    #                 if '/texts/' in table_parent_ref:
    #                     try:
    #                         text_idx_str = table_parent_ref.split('/texts/')[-1]
    #                         text_idx = int(text_idx_str)
    #                         if unit_start_idx <= text_idx < unit_end_idx:
    #                             table_indices.append(table_idx)
    #                             log_progress("DEBUG", f"Found table {table_idx} for unit (shared parent: {unit_parent_ref}, verified in range)")
    #                     except (ValueError, IndexError):
    #                         # If we can't verify, skip it
    #                         pass
    
    # Strategy 3: Heuristic matching by content (if unit_text provided)
    # Only use if no tables found yet
    # DISABLED - using only Strategy 1 (PRIMARY)
    # if not table_indices and unit_text:
    #     unit_text_lower = unit_text.lower()
    #     # Common keywords that might indicate table references
    #     table_keywords = [
    #         "odpisová skupina", "odpisovanie", "odpis", "sadzba", 
    #         "tabuľka", "príloha", "uvedené v", "podľa tabuľky"
    #     ]
    #     
    #     has_table_keyword = any(keyword in unit_text_lower for keyword in table_keywords)
    #     
    #     if has_table_keyword:
    #         # Try to find tables by header content
    #         for table_idx, table in enumerate(doc.tables):
    #             # Check table header
    #             rows = table_to_rows_from_grid(table)
    #             if rows and len(rows) > 0:
    #                 first_row = rows[0]
    #                 header_text = " ".join(first_row).lower()
    #                 
    #                 # Match keywords from unit text with table header
    #                 if any(keyword in header_text for keyword in table_keywords):
    #                     # Additional check: if unit mentions specific table content
    #                     # and table header matches, it's likely related
    #                     table_indices.append(table_idx)
    #                     log_progress("DEBUG", f"Found table {table_idx} for unit (heuristic match: {header_text[:50]})")
    
    return sorted(table_indices)


def find_tables_for_paragraph_by_content(
    doc: DoclingDocument,
    para_start_idx: int,
    para_end_idx: int,
    para_marker_text_elem=None
) -> List[int]:
    """
    Nájde tabuľky pre paragraf pomocou kombinácie stratégií:
    1. Parent referencie (ako pre prílohy)
    2. Heuristiky podľa hlavičky tabuľky
    3. Pozície v texte
    
    Args:
        doc: DoclingDocument
        para_start_idx: Start index of paragraph in doc.texts
        para_end_idx: End index of paragraph in doc.texts
        para_marker_text_elem: Text element marking the paragraph
        
    Returns:
        List of table indices that belong to this paragraph
    """
    # Extract paragraph text for heuristic matching
    para_text = " ".join([
        getattr(t, 'text', '') if hasattr(t, 'text') else (t.get("text", "") if isinstance(t, dict) else str(t))
        for t in doc.texts[para_start_idx:para_end_idx]
    ])
    
    # Use general find_tables_for_unit function
    return find_tables_for_unit(
        doc, para_start_idx, para_end_idx, 
        para_marker_text_elem, para_text
    )


# ============================================================================
# Sequential Document Reconstruction - Docling Native Chunker
# ============================================================================

class SequentialLawChunker(BaseChunker):
    """
    Chunker for Slovak law documents using docling native patterns.
    
    This chunker extends BaseChunker to leverage docling's structured document
    iteration and serialization while preserving law-specific marker detection
    and hierarchical structure (parts → paragraphs → odseks → pismenos).
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    serializer_provider: ChunkingSerializerProvider = Field(
        default_factory=ChunkingSerializerProvider,
        exclude=True  # Exclude from serialization
    )
    
    def chunk(
        self,
        dl_doc: DoclingDocument,
        **kwargs: Any,
    ) -> Iterator[BaseChunk]:
        """
        Chunk the document using docling patterns with law-specific structure.
        
        This method uses doc.iterate_items() instead of doc.texts to leverage
        docling's structured document model, then applies law-specific marker
        detection to build the hierarchical structure.
        
        Args:
            dl_doc: DoclingDocument to chunk
            
        Yields:
            BaseChunk: For compatibility, but we build custom structure internally
        """
        # Build the law-specific structure using docling patterns
        structure = self._reconstruct_document_with_docling(dl_doc)
        
        # Convert structure to chunks for compatibility (though we primarily use structure)
        # For now, yield empty chunks - the main output is the structure dict
        # This maintains compatibility with BaseChunker interface
        yield from []
    
    def _reconstruct_document_with_docling(self, doc: DoclingDocument) -> Dict[str, Any]:
        """
        Reconstruct document using docling's iterate_items() and serializers.
        
        This is the core refactored method that replaces reconstruct_document().
        It uses docling patterns but preserves all law-specific logic.
        """
        start_time = time.time()
        log_progress("INFO", "Starting sequential document reconstruction with docling patterns...")
        
        # Initialize serializer
        doc_serializer = self.serializer_provider.get_serializer(doc=doc)
        visited: set[str] = set()
        heading_by_level: dict[int, str] = {}
        
        # Build mapping from items to text indices for compatibility
        # This allows us to maintain table detection logic that uses indices
        item_to_text_idx: dict[str, int] = {}
        if hasattr(doc, 'texts') and doc.texts:
            for idx, text_elem in enumerate(doc.texts):
                if hasattr(text_elem, 'self_ref'):
                    item_to_text_idx[text_elem.self_ref] = idx
        
        # Initialize structure
        structure = {
            "document_name": getattr(doc, 'name', 'Unknown'),
            "metadata": {
                "source_file": getattr(doc, 'name', 'Unknown'),
                "reconstruction_time": 0.0,
                "reconstruction_method": "sequential_docling",
                "total_text_elements": len(doc.texts) if hasattr(doc, 'texts') else 0,
                "total_parts": 0,
                "total_paragraphs": 0,
                "total_odseks": 0,
                "total_pismenos": 0,
                "total_subitems": 0,
                "total_annexes": 0,
                "total_footnotes": 0
            },
            "parts": [],
            "annexes": {
                "annex_list": [],
                "summary": {}
            },
            "footnotes": [],
            "references_index": {}
        }
        
        if not hasattr(doc, 'texts') or not doc.texts:
            log_progress("WARNING", "Document has no text elements")
            return structure
        
        # Build table context map using recursive traversal (GPT + Claude approach)
        log_progress("INFO", "Building table context map...")
        table_context_map = build_table_context_map(doc)
        log_progress("INFO", f"Table context map built with {len(table_context_map)} tables")
        
        # Classify tables into legal and metadata
        log_progress("INFO", "Classifying tables...")
        legal_table_indices, metadata_table_indices = classify_all_tables(doc)
        log_progress("INFO", f"Legal tables: {len(legal_table_indices)}, Metadata tables: {len(metadata_table_indices)}")
        
        # Store in structure for later use
        structure["_table_context_map"] = table_context_map
        structure["_legal_table_indices"] = set(legal_table_indices)
        structure["_metadata_table_indices"] = set(metadata_table_indices)
        
        # State tracking (same as original)
        current_part = None
        current_paragraph = None
        current_odsek = None
        current_pismeno = None
        
        # State tracking for annexes
        current_annex = None
        annex_texts = []
        in_annex_section = False
        annex_start_idx = None
        annex_marker_text_elem = None
        all_annex_texts = []
        seen_annex_tables = set()
        
        # State tracking for footnotes
        current_footnote = None
        footnote_texts = []
        in_footnotes_section = False
        
        # Accumulators for current level
        part_texts = []
        para_intro_texts = []
        odsek_texts = []
        pismeno_texts = []
        
        # Counters
        parts_count = 0
        paragraphs_count = 0
        odseks_count = 0
        pismenos_count = 0
        subitems_count = 0
        annexes_count = 0
        footnotes_count = 0
        
        # Track item index for progress logging
        item_count = 0
        total_items = sum(1 for _ in doc.iterate_items(with_groups=True))
        last_log_time = time.time()
        log_interval = 2.0
        
        # Process items using docling's iterate_items()
        for item, level in doc.iterate_items(with_groups=True):
            item_count += 1
            
            # Progress logging
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                progress = (item_count / total_items * 100) if total_items > 0 else 0
                log_progress("INFO", f"Processing item {item_count:,}/{total_items:,} ({progress:.1f}%)")
                last_log_time = current_time
            
            # Handle SectionHeaderItem for heading tracking
            if isinstance(item, SectionHeaderItem):
                heading_by_level[item.level] = item.text
                # Remove headings of higher level as they just went out of scope
                keys_to_del = [k for k in heading_by_level if k > item.level]
                for k in keys_to_del:
                    heading_by_level.pop(k, None)
                
                # Check if this is the footnotes section before skipping
                item_text = getattr(item, 'text', '')
                item_hyperlink = getattr(item, 'hyperlink', '')
                hyperlink_str_for_check = str(item_hyperlink) if item_hyperlink else ''
                normalized_item_text = item_text.strip().replace('\xa0', ' ') if item_text else ''
                
                if detect_footnotes_section(normalized_item_text, hyperlink_str_for_check if hyperlink_str_for_check else None):
                    in_footnotes_section = True
                    log_progress("INFO", f"Found footnotes section (SectionHeaderItem) at item_idx={item_count}")
                    # Continue - we've set the flag, but don't process as content
                    continue
                
                # Continue - headings are tracked but not processed as content
                continue
            
            # Skip TitleItem (handled separately if needed)
            if isinstance(item, TitleItem):
                # Check if this is the footnotes section before skipping
                item_text = getattr(item, 'text', '')
                item_hyperlink = getattr(item, 'hyperlink', '')
                hyperlink_str_for_check = str(item_hyperlink) if item_hyperlink else ''
                normalized_item_text = item_text.strip().replace('\xa0', ' ') if item_text else ''
                
                if detect_footnotes_section(normalized_item_text, hyperlink_str_for_check if hyperlink_str_for_check else None):
                    in_footnotes_section = True
                    log_progress("INFO", f"Found footnotes section (TitleItem) at item_idx={item_count}")
                    # Continue - we've set the flag, but don't process as content
                    continue
                
                continue
            
            # Get text from item using serializer
            # For TextItem, ListGroup, InlineGroup, we serialize to get text
            if isinstance(item, (TextItem, ListGroup, InlineGroup, DocItem)):
                if item.self_ref in visited:
                    continue
                
                # Serialize the item to get text
                ser_res = doc_serializer.serialize(item=item, visited=visited)
                if not ser_res.text:
                    continue
                
                # Get serialized text and normalize
                raw_text = ser_res.text
                text = raw_text.strip()
                normalized_text = text.replace('\xa0', ' ')
                
                # Get hyperlink from item attributes (docling pattern)
                hyperlink = getattr(item, 'hyperlink', '')
                hyperlink_str = str(hyperlink) if hyperlink else ''
                has_hyperlink = bool(hyperlink_str)
                
                # Get item index for compatibility with existing logic
                item_idx = item_to_text_idx.get(item.self_ref, item_count - 1)
                
                # Now process this text with all the existing marker detection logic
                # This preserves all law-specific logic while using docling's structured items
                structure, current_part, current_paragraph, current_odsek, current_pismeno, \
                current_annex, annex_texts, in_annex_section, annex_start_idx, annex_marker_text_elem, \
                all_annex_texts, seen_annex_tables, current_footnote, footnote_texts, in_footnotes_section, \
                part_texts, para_intro_texts, odsek_texts, pismeno_texts, \
                parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count, \
                annexes_count, footnotes_count = self._process_text_element(
                    doc, structure, normalized_text, raw_text, hyperlink_str, has_hyperlink, item_idx, item,
                    current_part, current_paragraph, current_odsek, current_pismeno,
                    current_annex, annex_texts, in_annex_section, annex_start_idx, annex_marker_text_elem,
                    all_annex_texts, seen_annex_tables, current_footnote, footnote_texts, in_footnotes_section,
                    part_texts, para_intro_texts, odsek_texts, pismeno_texts,
                    parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count,
                    annexes_count, footnotes_count
                )
            
            # Handle TableItem separately if needed
            elif isinstance(item, TableItem):
                # Tables are handled via hyperlinks in text items, but we could process them here too
                pass
        
        # Close any remaining open structures (same as original)
        total_elements = len(doc.texts) if hasattr(doc, 'texts') else item_count
        if current_paragraph:
            current_paragraph["_end_idx"] = total_elements
        
        _close_pismeno(current_pismeno, pismeno_texts, current_odsek, doc, structure)
        _close_odsek(current_odsek, odsek_texts, para_intro_texts, current_paragraph, doc, structure)
        _close_paragraph(current_paragraph, para_intro_texts, current_part, doc, structure)
        
        # Finalize part title_text
        if current_part and part_texts:
            combined_title = " ".join([t.get("text", t) if isinstance(t, dict) else t for t in part_texts])
            if current_part["title_text"]:
                current_part["title_text"] = current_part["title_text"] + " " + combined_title
            else:
                current_part["title_text"] = combined_title
        
        if current_part and current_part["paragraphs"]:
            structure["parts"].append(current_part)
        
        # Close any open annex
        if current_annex:
            _close_annex(current_annex, annex_texts, doc, structure, annex_start_idx, total_elements, 
                        annex_marker_text_elem, all_annex_texts, seen_annex_tables)
        
        # Close any open footnote
        if current_footnote:
            _close_footnote(current_footnote, footnote_texts, structure)
            footnotes_count += 1
        
        # Build references_index
        _build_references_index(structure)
        
        # Check table assignments
        _check_table_assignments(structure, doc)
        
        # Update metadata
        elapsed = time.time() - start_time
        actual_parts_count = len(structure["parts"])
        structure["metadata"]["reconstruction_time"] = elapsed
        structure["metadata"]["total_parts"] = actual_parts_count
        structure["metadata"]["total_paragraphs"] = paragraphs_count
        structure["metadata"]["total_odseks"] = odseks_count
        structure["metadata"]["total_pismenos"] = pismenos_count
        structure["metadata"]["total_subitems"] = subitems_count
        structure["metadata"]["total_annexes"] = len(structure.get("annexes", {}).get("annex_list", []))
        structure["metadata"]["total_footnotes"] = footnotes_count
        
        # Initialize annexes summary if not already set
        if "annexes" in structure and "summary" not in structure["annexes"]:
            annex_list = structure["annexes"].get("annex_list", [])
            inline_count = sum(1 for a in annex_list if a.get("source") == "inline")
            external_count = sum(1 for a in annex_list if a.get("source") == "external_pdf")
            structure["annexes"]["summary"] = {
                "total_annexes": len(annex_list),
                "external_annexes": external_count,
                "inline_annexes": inline_count
            }
        
        log_progress("INFO", f"Reconstruction complete: {actual_parts_count} parts, {paragraphs_count} paragraphs, "
                   f"{odseks_count} odseks, {pismenos_count} pismenos, {subitems_count} subitems, "
                   f"{structure['metadata']['total_annexes']} annexes, {footnotes_count} footnotes", elapsed)
        
        return structure
    
    def _process_text_element(
        self, doc, structure, normalized_text, raw_text, hyperlink_str, has_hyperlink, item_idx, item,
        current_part, current_paragraph, current_odsek, current_pismeno,
        current_annex, annex_texts, in_annex_section, annex_start_idx, annex_marker_text_elem,
        all_annex_texts, seen_annex_tables, current_footnote, footnote_texts, in_footnotes_section,
        part_texts, para_intro_texts, odsek_texts, pismeno_texts,
        parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count,
        annexes_count, footnotes_count
    ):
        """
        Process a single text element with all marker detection logic.
        
        This method contains the core logic from the original reconstruct_document()
        loop, adapted to work with docling items.
        """
        # Check for footnotes section start (before other checks)
        if detect_footnotes_section(normalized_text, hyperlink_str if hyperlink_str else None):
            in_footnotes_section = True
            log_progress("INFO", f"Found footnotes section at item_idx={item_idx}")
            return (structure, current_part, current_paragraph, current_odsek, current_pismeno,
                   current_annex, annex_texts, in_annex_section, annex_start_idx, annex_marker_text_elem,
                   all_annex_texts, seen_annex_tables, current_footnote, footnote_texts, in_footnotes_section,
                   part_texts, para_intro_texts, odsek_texts, pismeno_texts,
                   parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count,
                   annexes_count, footnotes_count)
        
        # If in footnotes section, check for footnote markers
        if in_footnotes_section:
            # FIRST: Check if this is a page footer (structural identification)
            # This is the most reliable way to detect document footer
            label = getattr(item, 'label', None)
            if label == DocItemLabel.PAGE_FOOTER:
                # Close last footnote if any
                if current_footnote:
                    _close_footnote(current_footnote, footnote_texts, structure)
                    footnotes_count += 1
                    current_footnote = None
                # End footnotes section - page footer starts here
                in_footnotes_section = False
                log_progress("INFO", f"Detected page footer (label={label}) at item_idx={item_idx}, ending footnotes section")
                return (structure, current_part, current_paragraph, current_odsek, current_pismeno,
                       current_annex, annex_texts, in_annex_section, annex_start_idx, annex_marker_text_elem,
                       all_annex_texts, seen_annex_tables, current_footnote, footnote_texts, in_footnotes_section,
                       part_texts, para_intro_texts, odsek_texts, pismeno_texts,
                       parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count,
                       annexes_count, footnotes_count)
            
            # Also check content_layer as fallback
            content_layer = getattr(item, 'content_layer', None)
            if content_layer == ContentLayer.FURNITURE:
                # Close last footnote if any
                if current_footnote:
                    _close_footnote(current_footnote, footnote_texts, structure)
                    footnotes_count += 1
                    current_footnote = None
                # End footnotes section - furniture layer (footer) starts here
                in_footnotes_section = False
                log_progress("INFO", f"Detected furniture layer (footer) at item_idx={item_idx}, ending footnotes section")
                return (structure, current_part, current_paragraph, current_odsek, current_pismeno,
                       current_annex, annex_texts, in_annex_section, annex_start_idx, annex_marker_text_elem,
                       all_annex_texts, seen_annex_tables, current_footnote, footnote_texts, in_footnotes_section,
                       part_texts, para_intro_texts, odsek_texts, pismeno_texts,
                       parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count,
                       annexes_count, footnotes_count)
            
            footnote_id = detect_footnote_marker(normalized_text, hyperlink_str if hyperlink_str else None)
            if footnote_id:
                # Close previous footnote if any
                if current_footnote:
                    _close_footnote(current_footnote, footnote_texts, structure)
                    footnotes_count += 1
                
                # Start new footnote
                current_footnote = {
                    "id": f"footnote-{footnote_id}",
                    "number": footnote_id,
                    "content": ""
                }
                footnote_texts = []
                log_progress("INFO", f"Found footnote: {footnote_id}")
                return (structure, current_part, current_paragraph, current_odsek, current_pismeno,
                       current_annex, annex_texts, in_annex_section, annex_start_idx, annex_marker_text_elem,
                       all_annex_texts, seen_annex_tables, current_footnote, footnote_texts, in_footnotes_section,
                       part_texts, para_intro_texts, odsek_texts, pismeno_texts,
                       parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count,
                       annexes_count, footnotes_count)
            
            # Collect content for current footnote
            if current_footnote:
                # Fallback: Filter out UI elements and footer text (pattern matching)
                # This is less reliable but catches cases where label/content_layer might not be set correctly
                ui_elements = ['icon-warning', 'button-close', 'button-search', 'button-download', 
                              'button-print', 'button-history', 'button-content', 'plus',
                              'Ministerstvo spravodlivosti', 'helpdesk@slov-lex', 
                              'Infolinka', 'Sekcia edičných činností', 'Račianska',
                              'Vytvorené v súlade', 'Jednotným dizajn manuálom',
                              'Prevádzkovateľom služby', 'Email']
                
                is_ui_element = any(ui_elem in normalized_text for ui_elem in ui_elements)
                
                # Also filter out phone numbers (text containing only digits, spaces, and hyphens)
                # Pattern: digits with spaces/hyphens (e.g., "02 888 91 862")
                phone_pattern = r'^[\d\s\-]+$'
                is_phone_number = bool(re.match(phone_pattern, normalized_text.strip())) and len(normalized_text.strip()) > 5
                
                # Filter out email addresses
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                is_email = bool(re.match(email_pattern, normalized_text.strip()))
                
                # Skip UI elements, phone numbers, and emails in footnotes (fallback)
                if is_ui_element or is_phone_number or is_email:
                    return (structure, current_part, current_paragraph, current_odsek, current_pismeno,
                           current_annex, annex_texts, in_annex_section, annex_start_idx, annex_marker_text_elem,
                           all_annex_texts, seen_annex_tables, current_footnote, footnote_texts, in_footnotes_section,
                           part_texts, para_intro_texts, odsek_texts, pismeno_texts,
                           parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count,
                           annexes_count, footnotes_count)
                
                footnote_texts.append({
                    "text": normalized_text,
                    "raw_text": raw_text,
                    "text_element_idx": item_idx
                })
                return (structure, current_part, current_paragraph, current_odsek, current_pismeno,
                       current_annex, annex_texts, in_annex_section, annex_start_idx, annex_marker_text_elem,
                       all_annex_texts, seen_annex_tables, current_footnote, footnote_texts, in_footnotes_section,
                       part_texts, para_intro_texts, odsek_texts, pismeno_texts,
                       parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count,
                       annexes_count, footnotes_count)
        
        # Check if we've reached the end of main law
        if detect_law_end_marker(normalized_text):
            in_annex_section = True
            log_progress("DEBUG", f"Found end of law marker: '{normalized_text[:50]}'")
        
        # If we're in annex section, check for annex markers FIRST
        if in_annex_section:
            annex_num = detect_annex_marker(normalized_text)
            if annex_num:
                # Close previous annex if exists
                if current_annex:
                    _close_annex(current_annex, annex_texts, doc, structure, annex_start_idx, item_idx, 
                               annex_marker_text_elem, all_annex_texts, seen_annex_tables)
                
                # Start new annex
                annexes_count += 1
                annex_start_idx = item_idx
                annex_marker_text_elem = item
                current_annex = {
                    "id": f"annex-{annex_num}",
                    "number": annex_num,
                    "title": normalized_text,
                    "content": "",
                    "tables": [],
                    "pictures": []
                }
                annex_texts = []
                log_progress("INFO", f"Found annex: {annex_num} at item_idx={item_idx}")
                return (structure, current_part, current_paragraph, current_odsek, current_pismeno,
                       current_annex, annex_texts, in_annex_section, annex_start_idx, annex_marker_text_elem,
                       all_annex_texts, seen_annex_tables, current_footnote, footnote_texts, in_footnotes_section,
                       part_texts, para_intro_texts, odsek_texts, pismeno_texts,
                       parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count,
                       annexes_count, footnotes_count)
            
            # If we're in an annex, collect content
            if current_annex:
                # Check for table/picture references
                table_idx = extract_table_reference(hyperlink_str) if has_hyperlink else None
                picture_idx = extract_picture_reference(hyperlink_str) if has_hyperlink else None
                
                # Also check for table_idx and picture_idx attributes directly
                if not table_idx:
                    table_idx = getattr(item, 'table_idx', None)
                if not picture_idx:
                    picture_idx = getattr(item, 'picture_idx', None)
                
                # Collect annex content
                annex_texts.append({
                    "text": normalized_text,
                    "raw_text": raw_text,
                    "table_idx": table_idx,
                    "picture_idx": picture_idx
                })
                return (structure, current_part, current_paragraph, current_odsek, current_pismeno,
                       current_annex, annex_texts, in_annex_section, annex_start_idx, annex_marker_text_elem,
                       all_annex_texts, seen_annex_tables, current_footnote, footnote_texts, in_footnotes_section,
                       part_texts, para_intro_texts, odsek_texts, pismeno_texts,
                       parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count,
                       annexes_count, footnotes_count)
        
        # Continue with the rest of the marker detection logic from original reconstruct_document
        # This is a large block, so we'll call a helper method that contains the original logic
        structure, current_part, current_paragraph, current_odsek, current_pismeno, \
        part_texts, para_intro_texts, odsek_texts, pismeno_texts, \
        parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count = self._process_main_content(
            doc, structure, normalized_text, raw_text, hyperlink_str, has_hyperlink, item_idx, item,
            current_part, current_paragraph, current_odsek, current_pismeno,
            part_texts, para_intro_texts, odsek_texts, pismeno_texts,
            parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count
        )
        
        # Return all state variables (including annex and footnote state that wasn't modified)
        return (structure, current_part, current_paragraph, current_odsek, current_pismeno,
               current_annex, annex_texts, in_annex_section, annex_start_idx, annex_marker_text_elem,
               all_annex_texts, seen_annex_tables, current_footnote, footnote_texts, in_footnotes_section,
               part_texts, para_intro_texts, odsek_texts, pismeno_texts,
               parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count,
               annexes_count, footnotes_count)
    
    def _process_main_content(
        self, doc, structure, normalized_text, raw_text, hyperlink_str, has_hyperlink, item_idx, item,
        current_part, current_paragraph, current_odsek, current_pismeno,
        part_texts, para_intro_texts, odsek_texts, pismeno_texts,
        parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count
    ):
        """
        Process main content with marker detection (parts, paragraphs, odseks, pismenos).
        
        This contains the core marker detection logic from the original loop.
        """
        # Debug logging for first 100 elements
        if item_idx < 100:
            log_progress("DEBUG", f"Element {item_idx}: text='{normalized_text[:80]}' "
                         f"has_hyperlink={has_hyperlink} "
                         f"current_para={current_paragraph['id'] if current_paragraph else None} "
                         f"current_odsek={current_odsek['id'] if current_odsek else None}")
        
        # Check for markers (in order of hierarchy: part → paragraph → odsek → pismeno → subitem)
        
        # 1. Check for part marker
        part_marker = detect_part_marker(normalized_text)
        if part_marker:
            # Close previous part if exists
            if current_part:
                # Close current paragraph, odsek, pismeno
                _close_pismeno(current_pismeno, pismeno_texts, current_odsek, doc, structure)
                _close_odsek(current_odsek, odsek_texts, para_intro_texts, current_paragraph, doc, structure)
                _close_paragraph(current_paragraph, para_intro_texts, current_part, doc, structure)
                
                # Only add part if it has content (paragraphs)
                if current_part["paragraphs"]:
                    structure["parts"].append(current_part)
                else:
                    log_progress("DEBUG", f"Skipping empty part: {current_part['title']}")
            
            # Start new part
            parts_count += 1
            current_part = {
                "id": f"part-{parts_count}",
                "title": part_marker,
                "title_text": normalized_text,
                "paragraphs": []
            }
            current_paragraph = None
            current_odsek = None
            current_pismeno = None
            part_texts = [normalized_text]
            para_intro_texts = []
            odsek_texts = []
            pismeno_texts = []
            log_progress("DEBUG", f"Found part: {part_marker}")
            return (structure, current_part, current_paragraph, current_odsek, current_pismeno,
                   part_texts, para_intro_texts, odsek_texts, pismeno_texts,
                   parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count)
        
        # If no part yet, create a default one
        if not current_part:
            parts_count += 1
            current_part = {
                "id": f"part-{parts_count}",
                "title": "UNNAMED PART",
                "title_text": "",
                "paragraphs": []
            }
        
        # 2. Check for paragraph marker
        para_num = detect_paragraph_marker(normalized_text, hyperlink_str if hyperlink_str else None)
        if para_num and not has_hyperlink:
            # Set end index for previous paragraph before closing
            if current_paragraph:
                current_paragraph["_end_idx"] = item_idx
            
            # Close previous paragraph, odsek, pismeno
            _close_pismeno(current_pismeno, pismeno_texts, current_odsek, doc, structure)
            _close_odsek(current_odsek, odsek_texts, para_intro_texts, current_paragraph, doc, structure)
            _close_paragraph(current_paragraph, para_intro_texts, current_part, doc, structure)
            
            # Before starting new paragraph, finalize part title_text if we collected additional text
            if current_part and part_texts:
                combined_title = " ".join(part_texts)
                current_part["title_text"] = combined_title
                part_texts = []
            
            # Start new paragraph
            paragraphs_count += 1
            current_paragraph = {
                "id": f"paragraf-{para_num}",
                "marker": f"§ {para_num}",
                "title": normalized_text,
                "intro_text": "",
                "odseks": [],
                "_start_idx": item_idx,
                "_marker_text_elem": item
            }
            current_odsek = None
            current_pismeno = None
            para_intro_texts = []
            odsek_texts = []
            pismeno_texts = []
            
            log_progress("DEBUG", f"Found paragraph: § {para_num} at idx {item_idx}")
            return (structure, current_part, current_paragraph, current_odsek, current_pismeno,
                   part_texts, para_intro_texts, odsek_texts, pismeno_texts,
                   parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count)
        
        # If no paragraph yet, collect as part of part title
        if not current_paragraph:
            if normalized_text and not detect_odsek_marker(normalized_text, hyperlink_str if hyperlink_str else None) and not detect_pismeno_marker(normalized_text, hyperlink_str if hyperlink_str else None):
                part_texts.append(normalized_text)
            return (structure, current_part, current_paragraph, current_odsek, current_pismeno,
                   part_texts, para_intro_texts, odsek_texts, pismeno_texts,
                   parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count)
        
        # 3. Check for pismeno marker FIRST (before odsek, since multi-letter pismenos like aa) could be confused)
        # Try standalone marker first
        pismeno_letter = detect_pismeno_marker(normalized_text, hyperlink_str if hyperlink_str else None)
        pismeno_extracted = None
        pismeno_remaining_text = None
        
        # If not found, try extracting from text (e.g., "a) content")
        if not pismeno_letter:
            pismeno_extracted = extract_marker_from_text(normalized_text, 'pismeno')
            if pismeno_extracted:
                pismeno_letter, pismeno_remaining_text = pismeno_extracted
        
        if pismeno_letter:
            # Check if this marker is likely a reference in context (not a structural marker)
            # Get previous text from current pismeno or odsek to check context
            previous_text = ""
            previous_elements = []
            
            if current_pismeno and pismeno_texts:
                previous_text = " ".join([t.get("text", "") for t in pismeno_texts[-3:]])
                if len(pismeno_texts) >= 3:
                    for t in pismeno_texts[-3:]:
                        if '_element' in t:
                            previous_elements.append(t['_element'])
            elif current_odsek and odsek_texts:
                previous_text = " ".join([t.get("text", "") for t in odsek_texts[-3:]])
                if len(odsek_texts) >= 3:
                    for t in odsek_texts[-3:]:
                        if '_element' in t:
                            previous_elements.append(t['_element'])
            
            # Check if marker is a reference in context (hybrid approach)
            is_reference = is_pismeno_reference_in_context(
                previous_text, 
                pismeno_letter,
                current_element=item,
                previous_elements=previous_elements
            )
            
            if is_reference:
                # This is a reference, not a structural marker - add to current content
                references = extract_references_from_text(normalized_text)
                footnotes = extract_footnotes_from_text(normalized_text)
                table_idx = extract_table_reference(hyperlink_str) if has_hyperlink else None
                picture_idx = extract_picture_reference(hyperlink_str) if has_hyperlink else None
                
                if not table_idx:
                    table_idx = getattr(item, 'table_idx', None)
                if not picture_idx:
                    picture_idx = getattr(item, 'picture_idx', None)
                
                element_references = getattr(item, 'references', [])
                element_footnotes = getattr(item, 'footnotes', [])
                all_references = references + (element_references if element_references else [])
                all_footnotes = footnotes + (element_footnotes if element_footnotes else [])
                
                # Add to appropriate level
                if current_pismeno:
                    if current_pismeno.get('_current_subitem'):
                        current_pismeno['_subitem_texts'].append({
                            "text": normalized_text,
                            "raw_text": raw_text,
                            "table_idx": table_idx,
                            "picture_idx": picture_idx,
                            "references": all_references,
                            "footnotes": all_footnotes
                        })
                    else:
                        pismeno_texts.append({
                            "text": normalized_text,
                            "raw_text": raw_text,
                            "table_idx": table_idx,
                            "picture_idx": picture_idx,
                            "references": all_references,
                            "footnotes": all_footnotes,
                            "_element": item
                        })
                elif current_odsek:
                    odsek_texts.append({
                        "text": normalized_text,
                        "raw_text": raw_text,
                        "table_idx": table_idx,
                        "picture_idx": picture_idx,
                        "references": all_references,
                        "footnotes": all_footnotes,
                        "_element": item
                    })
                elif current_paragraph:
                    para_intro_texts.append({
                        "text": normalized_text,
                        "raw_text": raw_text,
                        "table_idx": table_idx,
                        "picture_idx": picture_idx,
                        "references": all_references,
                        "footnotes": all_footnotes,
                        "_element": item
                    })
                return (structure, current_part, current_paragraph, current_odsek, current_pismeno,
                       part_texts, para_intro_texts, odsek_texts, pismeno_texts,
                       parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count)
            
            # This is a real structural marker - proceed with creating new pismeno
            _close_pismeno(current_pismeno, pismeno_texts, current_odsek, doc, structure)
            
            # Start new pismeno
            if current_odsek:
                pismenos_count += 1
                para_num_for_pismeno = current_paragraph["id"].replace("paragraf-", "")
                odsek_num_for_pismeno = current_odsek["id"].split(".")[-1]
                pismeno_id = f"pismeno-{para_num_for_pismeno}.{odsek_num_for_pismeno}.{pismeno_letter}"
                current_pismeno = {
                    "id": pismeno_id,
                    "marker": f"{pismeno_letter})",
                    "text": "",
                    "subitems": [],
                    "tables": [],
                    "pictures": [],
                    "references_metadata": [],
                    "footnotes_metadata": []
                }
                pismeno_texts = []
                
                # If we extracted marker from text, add remaining text to pismeno
                if pismeno_extracted and pismeno_remaining_text:
                    references = extract_references_from_text(pismeno_remaining_text)
                    footnotes = extract_footnotes_from_text(pismeno_remaining_text)
                    table_idx = extract_table_reference(hyperlink_str) if has_hyperlink else None
                    picture_idx = extract_picture_reference(hyperlink_str) if has_hyperlink else None
                    pismeno_texts.append({
                        "text": pismeno_remaining_text,
                        "raw_text": raw_text,
                        "table_idx": table_idx,
                        "picture_idx": picture_idx,
                        "references": references,
                        "footnotes": footnotes,
                        "_element": item
                    })
                
                log_progress("DEBUG", f"Found pismeno: {pismeno_letter}) in odsek {odsek_num_for_pismeno}")
            return (structure, current_part, current_paragraph, current_odsek, current_pismeno,
                   part_texts, para_intro_texts, odsek_texts, pismeno_texts,
                   parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count)
        
        # 4. Check for odsek marker (after pismeno check)
        odsek_marker = detect_odsek_marker(normalized_text, hyperlink_str if hyperlink_str else None)
        odsek_extracted = None
        odsek_remaining_text = None
        
        if not odsek_marker:
            odsek_extracted = extract_marker_from_text(normalized_text, 'odsek')
            if odsek_extracted:
                odsek_marker, odsek_remaining_text = odsek_extracted
        
        if odsek_marker:
            _close_pismeno(current_pismeno, pismeno_texts, current_odsek, doc, structure)
            _close_odsek(current_odsek, odsek_texts, para_intro_texts, current_paragraph, doc, structure)
            
            # Start new odsek
            odseks_count += 1
            para_num_for_odsek = current_paragraph["id"].replace("paragraf-", "")
            odsek_id = f"odsek-{para_num_for_odsek}.{odsek_marker}"
            current_odsek = {
                "id": odsek_id,
                "marker": f"({odsek_marker})",
                "text": "",
                "tables": [],
                "pictures": [],
                "references_metadata": [],
                "footnotes_metadata": [],
                "pismenos": [],
                "_start_idx": item_idx,  # Track start index for table detection
                "_marker_text_elem": item  # Track marker element for parent reference
            }
            current_pismeno = None
            odsek_texts = []
            pismeno_texts = []
            
            # If we extracted marker from text, add remaining text to odsek
            if odsek_extracted and odsek_remaining_text:
                references = extract_references_from_text(odsek_remaining_text)
                footnotes = extract_footnotes_from_text(odsek_remaining_text)
                table_idx = extract_table_reference(hyperlink_str) if has_hyperlink else None
                picture_idx = extract_picture_reference(hyperlink_str) if has_hyperlink else None
                odsek_texts.append({
                    "text": odsek_remaining_text,
                    "raw_text": raw_text,
                    "table_idx": table_idx,
                    "picture_idx": picture_idx,
                    "references": references,
                    "footnotes": footnotes,
                    "_element": item
                })
            
            log_progress("DEBUG", f"Found odsek: ({odsek_marker}) in paragraph {para_num_for_odsek}")
            return (structure, current_part, current_paragraph, current_odsek, current_pismeno,
                   part_texts, para_intro_texts, odsek_texts, pismeno_texts,
                   parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count)
        
        # 5. Check for subitem marker (within pismeno)
        if current_pismeno:
            subitem_marker = detect_subitem_marker(normalized_text)
            subitem_extracted = None
            subitem_remaining_text = None
            
            if not subitem_marker:
                subitem_extracted = extract_marker_from_text(normalized_text, 'subitem')
                if subitem_extracted:
                    subitem_marker, subitem_remaining_text = subitem_extracted
            
            if subitem_marker:
                # Start new subitem or add to existing
                if not current_pismeno.get('_current_subitem') or current_pismeno.get('_current_subitem') != subitem_marker:
                    # Close previous subitem if exists
                    if current_pismeno.get('_current_subitem'):
                        prev_subitem = {
                            "marker": f"{current_pismeno['_current_subitem']}.",
                            "text": "\n".join([t["text"] for t in current_pismeno.get('_subitem_texts', []) if t.get("text")]),
                            "tables": [],
                            "pictures": [],
                            "references_metadata": [],
                            "footnotes_metadata": []
                        }
                        text_pos = 0
                        for t in current_pismeno.get('_subitem_texts', []):
                            if t.get("table_idx") is not None:
                                table_idx = t["table_idx"]
                                if table_idx < len(doc.tables):
                                    prev_subitem["tables"].append({
                                        "index": table_idx,
                                        "position_in_text": text_pos
                                    })
                            if t.get("picture_idx") is not None:
                                picture_idx = t["picture_idx"]
                                if picture_idx < len(doc.pictures):
                                    prev_subitem["pictures"].append({
                                        "index": picture_idx,
                                        "position_in_text": text_pos
                                    })
                            prev_subitem["references_metadata"].extend(t.get("references", []))
                            prev_subitem["footnotes_metadata"].extend(t.get("footnotes", []))
                            text_pos += len(t.get("text", "")) + 1
                        
                        current_pismeno["subitems"].append(prev_subitem)
                    
                    # Start new subitem
                    current_pismeno['_current_subitem'] = subitem_marker
                    current_pismeno['_subitem_texts'] = []
                    subitems_count += 1
                
                # Add remaining text to subitem if extracted
                if subitem_extracted and subitem_remaining_text:
                    references = extract_references_from_text(subitem_remaining_text)
                    footnotes = extract_footnotes_from_text(subitem_remaining_text)
                    table_idx = extract_table_reference(hyperlink_str) if has_hyperlink else None
                    picture_idx = extract_picture_reference(hyperlink_str) if has_hyperlink else None
                    current_pismeno['_subitem_texts'].append({
                        "text": subitem_remaining_text,
                        "raw_text": raw_text,
                        "table_idx": table_idx,
                        "picture_idx": picture_idx,
                        "references": references,
                        "footnotes": footnotes
                    })
                    log_progress("DEBUG", f"Found subitem: {subitem_marker}. in pismeno {current_pismeno.get('marker', '')}")
                return (structure, current_part, current_paragraph, current_odsek, current_pismeno,
                       part_texts, para_intro_texts, odsek_texts, pismeno_texts,
                       parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count)
        
        # 6. Process content (not a marker)
        content_text = normalized_text
        
        # Extract references and footnotes
        references = extract_references_from_text(content_text)
        footnotes = extract_footnotes_from_text(content_text)
        
        # Check for table/picture references from hyperlink
        table_idx = extract_table_reference(hyperlink_str) if has_hyperlink else None
        picture_idx = extract_picture_reference(hyperlink_str) if has_hyperlink else None
        
        if not table_idx:
            table_idx = getattr(item, 'table_idx', None)
        if not picture_idx:
            picture_idx = getattr(item, 'picture_idx', None)
        
        # Get references and footnotes from element attributes
        element_references = getattr(item, 'references', [])
        element_footnotes = getattr(item, 'footnotes', [])
        
        # Combine extracted and element references/footnotes
        all_references = references + (element_references if element_references else [])
        all_footnotes = footnotes + (element_footnotes if element_footnotes else [])
        
        # Add to appropriate level
        if current_pismeno:
            if current_pismeno.get('_current_subitem'):
                current_pismeno['_subitem_texts'].append({
                    "text": content_text,
                    "raw_text": raw_text,
                    "table_idx": table_idx,
                    "picture_idx": picture_idx,
                    "references": all_references,
                    "footnotes": all_footnotes,
                    "_element": item
                })
            else:
                pismeno_texts.append({
                    "text": content_text,
                    "raw_text": raw_text,
                    "table_idx": table_idx,
                    "picture_idx": picture_idx,
                    "references": all_references,
                    "footnotes": all_footnotes,
                    "_element": item
                })
        elif current_odsek:
            odsek_texts.append({
                "text": content_text,
                "raw_text": raw_text,
                "table_idx": table_idx,
                "picture_idx": picture_idx,
                "references": all_references,
                "footnotes": all_footnotes,
                "_element": item
            })
        elif current_paragraph:
            para_intro_texts.append({
                "text": content_text,
                "raw_text": raw_text,
                "table_idx": table_idx,
                "picture_idx": picture_idx,
                "references": all_references,
                "footnotes": all_footnotes,
                "_element": item
            })
        else:
            part_texts.append({
                "text": content_text,
                "raw_text": raw_text
            })
        
        return (structure, current_part, current_paragraph, current_odsek, current_pismeno,
               part_texts, para_intro_texts, odsek_texts, pismeno_texts,
               parts_count, paragraphs_count, odseks_count, pismenos_count, subitems_count)


def reconstruct_document(doc: DoclingDocument) -> Dict[str, Any]:
    """
    Reconstruct entire document in a single sequential pass.
    
    This function processes doc.texts sequentially, building the complete
    hierarchical structure as it encounters markers. No recursion, no repeated
    operations - just one pass through all text elements.
    
    Args:
        doc: DoclingDocument to reconstruct
        
    Returns:
        Complete hierarchical structure dictionary
    """
    start_time = time.time()
    log_progress("INFO", "Starting sequential document reconstruction...")
    
    # Initialize structure
    structure = {
        "document_name": getattr(doc, 'name', 'Unknown'),
        "metadata": {
            "source_file": getattr(doc, 'name', 'Unknown'),
            "reconstruction_time": 0.0,
            "reconstruction_method": "sequential",
            "total_text_elements": len(doc.texts) if hasattr(doc, 'texts') else 0,
            "total_parts": 0,
            "total_paragraphs": 0,
            "total_odseks": 0,
            "total_pismenos": 0,
            "total_subitems": 0,
            "total_annexes": 0,
            "total_footnotes": 0
        },
        "parts": [],
        "annexes": {
            "annex_list": [],
            "summary": {}
        },
        "footnotes": [],
        "references_index": {}
    }
    
    if not hasattr(doc, 'texts') or not doc.texts:
        log_progress("WARNING", "Document has no text elements")
        return structure
    
    total_elements = len(doc.texts)
    log_progress("INFO", f"Processing {total_elements:,} text elements...")
    
    # Build table context map using recursive traversal (GPT + Claude approach)
    log_progress("INFO", "Building table context map...")
    table_context_map = build_table_context_map(doc)
    
    # Classify tables into legal and metadata
    log_progress("INFO", "Classifying tables...")
    legal_table_indices, metadata_table_indices = classify_all_tables(doc)
    
    # Store in structure for later use
    structure["metadata"]["legal_tables"] = legal_table_indices
    structure["metadata"]["metadata_tables"] = metadata_table_indices
    structure["_table_context_map"] = table_context_map
    structure["_legal_table_indices"] = set(legal_table_indices)
    
    # State tracking
    current_part = None
    current_paragraph = None
    current_odsek = None
    current_pismeno = None
    
    # State tracking for annexes
    current_annex = None
    annex_texts = []
    in_annex_section = False
    annex_start_idx = None  # Track where annex started for table detection
    annex_marker_text_elem = None  # Track the annex marker text element (for parent reference)
    all_annex_texts = []  # Collect all texts from all annexes
    seen_annex_tables = set()  # Track tables to avoid duplicates
    
    # State tracking for footnotes
    current_footnote = None
    footnote_texts = []
    in_footnotes_section = False
    
    # Accumulators for current level
    part_texts = []
    para_intro_texts = []
    odsek_texts = []
    pismeno_texts = []
    
    # Counters
    parts_count = 0
    paragraphs_count = 0
    odseks_count = 0
    pismenos_count = 0
    subitems_count = 0
    annexes_count = 0
    footnotes_count = 0
    
    # Process each text element sequentially
    last_log_time = time.time()
    log_interval = 2.0  # Log every 2 seconds
    
    for idx, text_element in enumerate(doc.texts):
        # Progress logging
        current_time = time.time()
        if current_time - last_log_time >= log_interval:
            progress = (idx + 1) / total_elements * 100
            log_progress("INFO", f"Processing text element {idx+1:,}/{total_elements:,} ({progress:.1f}%)")
            last_log_time = current_time
        
        # Get text content
        raw_text = getattr(text_element, 'text', '')
        text = raw_text.strip()
        normalized_text = text.replace('\xa0', ' ')
        
        # Get hyperlink
        hyperlink = getattr(text_element, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        has_hyperlink = bool(hyperlink_str)
        
        # Check for footnotes section start (before other checks)
        if detect_footnotes_section(normalized_text, hyperlink):
            in_footnotes_section = True
            log_progress("INFO", f"Found footnotes section at text_element_idx={idx}")
            continue
        
        # If in footnotes section, check for footnote markers
        if in_footnotes_section:
            # FIRST: Check if this is a page footer (structural identification)
            # This is the most reliable way to detect document footer
            label = getattr(text_element, 'label', None)
            if label == DocItemLabel.PAGE_FOOTER:
                # Close last footnote if any
                if current_footnote:
                    _close_footnote(current_footnote, footnote_texts, structure)
                    footnotes_count += 1
                    current_footnote = None
                # End footnotes section - page footer starts here
                in_footnotes_section = False
                log_progress("INFO", f"Detected page footer (label={label}) at text_element_idx={idx}, ending footnotes section")
                continue  # Skip footer content
            
            # Also check content_layer as fallback
            content_layer = getattr(text_element, 'content_layer', None)
            if content_layer == ContentLayer.FURNITURE:
                # Close last footnote if any
                if current_footnote:
                    _close_footnote(current_footnote, footnote_texts, structure)
                    footnotes_count += 1
                    current_footnote = None
                # End footnotes section - furniture layer (footer) starts here
                in_footnotes_section = False
                log_progress("INFO", f"Detected furniture layer (footer) at text_element_idx={idx}, ending footnotes section")
                continue  # Skip footer content
            
            footnote_id = detect_footnote_marker(normalized_text, hyperlink)
            if footnote_id:
                # Close previous footnote if any
                if current_footnote:
                    _close_footnote(current_footnote, footnote_texts, structure)
                    footnotes_count += 1
                
                # Start new footnote
                current_footnote = {
                    "id": f"footnote-{footnote_id}",
                    "number": footnote_id,
                    "content": ""
                }
                footnote_texts = []
                log_progress("INFO", f"Found footnote: {footnote_id}")
                continue
            
            # Collect content for current footnote
            if current_footnote:
                # Fallback: Filter out UI elements and footer text (pattern matching)
                # This is less reliable but catches cases where label/content_layer might not be set correctly
                ui_elements = ['icon-warning', 'button-close', 'button-search', 'button-download', 
                              'button-print', 'button-history', 'button-content', 'plus',
                              'Ministerstvo spravodlivosti', 'helpdesk@slov-lex', 
                              'Infolinka', 'Sekcia edičných činností', 'Račianska',
                              'Vytvorené v súlade', 'Jednotným dizajn manuálom',
                              'Prevádzkovateľom služby', 'Email']
                
                is_ui_element = any(ui_elem in normalized_text for ui_elem in ui_elements)
                
                # Also filter out phone numbers (text containing only digits, spaces, and hyphens)
                # Pattern: digits with spaces/hyphens (e.g., "02 888 91 862")
                phone_pattern = r'^[\d\s\-]+$'
                is_phone_number = bool(re.match(phone_pattern, normalized_text.strip())) and len(normalized_text.strip()) > 5
                
                # Filter out email addresses
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                is_email = bool(re.match(email_pattern, normalized_text.strip()))
                
                # Skip UI elements, phone numbers, and emails in footnotes (fallback)
                if is_ui_element or is_phone_number or is_email:
                    continue
                
                footnote_texts.append({
                    "text": normalized_text,
                    "raw_text": raw_text,
                    "text_element_idx": idx
                })
                continue
        
        # Check if we've reached the end of main law
        if detect_law_end_marker(normalized_text):
            # Mark that we're past main law content
            in_annex_section = True
            log_progress("DEBUG", f"Found end of law marker: '{normalized_text[:50]}'")
            # Continue processing - this text might be part of last paragraph or standalone
        
        # If we're in annex section, check for annex markers FIRST (before other markers)
        if in_annex_section:
            annex_num = detect_annex_marker(normalized_text)
            if annex_num:
                # Close previous annex if exists
                if current_annex:
                    # annex_end_idx is current idx (where new annex starts)
                    _close_annex(current_annex, annex_texts, doc, structure, annex_start_idx, idx, annex_marker_text_elem, all_annex_texts, seen_annex_tables)
                
                # Start new annex
                annexes_count += 1
                annex_start_idx = idx  # Store where this annex starts
                annex_marker_text_elem = text_element  # Store the marker element for parent reference
                current_annex = {
                    "id": f"annex-{annex_num}",
                    "number": annex_num,
                    "title": normalized_text,
                    "content": "",
                    "tables": [],
                    "pictures": []
                }
                annex_texts = []
                log_progress("INFO", f"Found annex: {annex_num} at text_element_idx={idx}")
                continue
            
            # If we're in an annex, collect content
            if current_annex:
                # Check for table/picture references
                table_idx = extract_table_reference(hyperlink_str) if has_hyperlink else None
                picture_idx = extract_picture_reference(hyperlink_str) if has_hyperlink else None
                
                # Also check for table_idx and picture_idx attributes directly
                if not table_idx:
                    table_idx = getattr(text_element, 'table_idx', None)
                if not picture_idx:
                    picture_idx = getattr(text_element, 'picture_idx', None)
                
                # Collect annex content
                annex_texts.append({
                    "text": normalized_text,
                    "raw_text": raw_text,
                    "table_idx": table_idx,
                    "picture_idx": picture_idx
                })
                continue
        
        # Debug logging for first 100 elements
        if idx < 100:
            log_progress("DEBUG", f"Element {idx}: text='{normalized_text[:80]}' "
                         f"has_hyperlink={has_hyperlink} "
                         f"current_para={current_paragraph['id'] if current_paragraph else None} "
                         f"current_odsek={current_odsek['id'] if current_odsek else None}")
        
        # Check for markers (in order of hierarchy: part → paragraph → odsek → pismeno → subitem)
        
        # 1. Check for part marker
        part_marker = detect_part_marker(normalized_text)
        if part_marker:
            # Close previous part if exists
            if current_part:
                # Close current paragraph, odsek, pismeno
                _close_pismeno(current_pismeno, pismeno_texts, current_odsek, doc, structure)
                _close_odsek(current_odsek, odsek_texts, para_intro_texts, current_paragraph, doc, structure)
                _close_paragraph(current_paragraph, para_intro_texts, current_part, doc, structure)
                
                # Only add part if it has content (paragraphs)
                if current_part["paragraphs"]:
                    structure["parts"].append(current_part)
                else:
                    log_progress("DEBUG", f"Skipping empty part: {current_part['title']}")
            
            # Start new part
            parts_count += 1
            current_part = {
                "id": f"part-{parts_count}",
                "title": part_marker,
                "title_text": normalized_text,
                "paragraphs": []
            }
            current_paragraph = None
            current_odsek = None
            current_pismeno = None
            part_texts = [normalized_text]  # Start collecting part title text
            para_intro_texts = []
            odsek_texts = []
            pismeno_texts = []
            log_progress("DEBUG", f"Found part: {part_marker}")
            continue
        
        # If no part yet, create a default one
        if not current_part:
            parts_count += 1
            current_part = {
                "id": f"part-{parts_count}",
                "title": "UNNAMED PART",
                "title_text": "",
                "paragraphs": []
            }
        
        # 2. Check for paragraph marker
        para_num = detect_paragraph_marker(normalized_text, hyperlink_str if hyperlink_str else None)
        if para_num and not has_hyperlink:
            # Set end index for previous paragraph before closing
            if current_paragraph:
                current_paragraph["_end_idx"] = idx
            
            # Close previous paragraph, odsek, pismeno
            _close_pismeno(current_pismeno, pismeno_texts, current_odsek, doc, structure)
            _close_odsek(current_odsek, odsek_texts, para_intro_texts, current_paragraph, doc, structure)
            _close_paragraph(current_paragraph, para_intro_texts, current_part, doc, structure)
            
            # Before starting new paragraph, finalize part title_text if we collected additional text
            if current_part and part_texts:
                # Combine all part title texts
                combined_title = " ".join(part_texts)
                current_part["title_text"] = combined_title
                part_texts = []  # Reset for next part
            
            # Start new paragraph
            paragraphs_count += 1
            current_paragraph = {
                "id": f"paragraf-{para_num}",
                "marker": f"§ {para_num}",
                "title": normalized_text,
                "intro_text": "",
                "odseks": [],
                "_start_idx": idx,  # Track start index for table detection
                "_marker_text_elem": text_element  # Track marker element for parent reference
            }
            current_odsek = None
            current_pismeno = None
            para_intro_texts = []
            odsek_texts = []
            pismeno_texts = []
            
            log_progress("DEBUG", f"Found paragraph: § {para_num} at idx {idx}")
            continue
        
        # If no paragraph yet, collect as part of part title
        if not current_paragraph:
            # This might be part of the part title (e.g., "ZÁKLADNÉ USTANOVENIA")
            # Collect it into part_texts
            if normalized_text and not detect_odsek_marker(normalized_text, hyperlink_str if hyperlink_str else None) and not detect_pismeno_marker(normalized_text, hyperlink_str if hyperlink_str else None):
                part_texts.append(normalized_text)
            continue
        
        # 3. Check for pismeno marker FIRST (before odsek, since multi-letter pismenos like aa) could be confused)
        # Try standalone marker first
        pismeno_letter = detect_pismeno_marker(normalized_text, hyperlink_str if hyperlink_str else None)
        pismeno_extracted = None
        pismeno_remaining_text = None
        
        # If not found, try extracting from text (e.g., "a) content")
        if not pismeno_letter:
            pismeno_extracted = extract_marker_from_text(normalized_text, 'pismeno')
            if pismeno_extracted:
                pismeno_letter, pismeno_remaining_text = pismeno_extracted
        
        if pismeno_letter:
            # Check if this marker is likely a reference in context (not a structural marker)
            # Get previous text from current pismeno or odsek to check context
            previous_text = ""
            previous_elements = []
            
            if current_pismeno and pismeno_texts:
                # Get last text from current pismeno
                previous_text = " ".join([t.get("text", "") for t in pismeno_texts[-3:]])  # Check last 3 text elements
                # Get previous text elements (if stored)
                if len(pismeno_texts) >= 3:
                    # Try to get previous elements from stored data
                    for t in pismeno_texts[-3:]:
                        if '_element' in t:
                            previous_elements.append(t['_element'])
            elif current_odsek and odsek_texts:
                # Get last text from current odsek
                previous_text = " ".join([t.get("text", "") for t in odsek_texts[-3:]])  # Check last 3 text elements
                # Get previous text elements (if stored)
                if len(odsek_texts) >= 3:
                    for t in odsek_texts[-3:]:
                        if '_element' in t:
                            previous_elements.append(t['_element'])
            
            # Also check immediate previous element in doc.texts
            if idx > 0:
                prev_elem = doc.texts[idx - 1]
                if prev_elem not in previous_elements:
                    previous_elements.append(prev_elem)
            
            # Check if marker is a reference in context (hybrid approach)
            is_reference = is_pismeno_reference_in_context(
                previous_text, 
                pismeno_letter,
                current_element=text_element,
                previous_elements=previous_elements
            )
            
            if is_reference:
                # This is a reference, not a structural marker - add to current content
                
                # Build full identification for logging
                location_parts = []
                if current_paragraph:
                    location_parts.append(f"paragraph={current_paragraph['id']}")
                if current_odsek:
                    location_parts.append(f"odsek={current_odsek['id']}")
                if current_pismeno:
                    location_parts.append(f"pismeno={current_pismeno['id']}")
                location_str = " | ".join(location_parts) if location_parts else "unknown location"
                
                # Get context snippet (last 150 chars of previous text)
                context_snippet = previous_text[-150:] if previous_text else "no previous text"
                context_snippet = context_snippet.replace('\n', ' ').strip()
                
                # Get current text element index for reference
                element_info = f"text_element_idx={idx}"
                
                log_progress("INFO", 
                    f"Ignoring pismeno marker '{pismeno_letter})' as reference | "
                    f"{location_str} | {element_info} | "
                    f"context: '...{context_snippet}'")
                
                # Add as regular text content to current pismeno or odsek
                references = extract_references_from_text(normalized_text)
                footnotes = extract_footnotes_from_text(normalized_text)
                table_idx = extract_table_reference(hyperlink_str) if has_hyperlink else None
                picture_idx = extract_picture_reference(hyperlink_str) if has_hyperlink else None
                
                # Also check for table_idx and picture_idx attributes directly
                if not table_idx:
                    table_idx = getattr(text_element, 'table_idx', None)
                if not picture_idx:
                    picture_idx = getattr(text_element, 'picture_idx', None)
                
                # Get references and footnotes from element attributes
                element_references = getattr(text_element, 'references', [])
                element_footnotes = getattr(text_element, 'footnotes', [])
                
                # Combine extracted and element references/footnotes
                all_references = references + (element_references if element_references else [])
                all_footnotes = footnotes + (element_footnotes if element_footnotes else [])
                
                # Add to appropriate level
                if current_pismeno:
                    if current_pismeno.get('_current_subitem'):
                        current_pismeno['_subitem_texts'].append({
                            "text": normalized_text,
                            "raw_text": raw_text,
                            "table_idx": table_idx,
                            "picture_idx": picture_idx,
                            "references": all_references,
                            "footnotes": all_footnotes
                        })
                    else:
                        pismeno_texts.append({
                            "text": normalized_text,
                            "raw_text": raw_text,
                            "table_idx": table_idx,
                            "picture_idx": picture_idx,
                            "references": all_references,
                            "footnotes": all_footnotes,
                            "_element": text_element  # Store element for reference checking
                        })
                elif current_odsek:
                    odsek_texts.append({
                        "text": normalized_text,
                        "raw_text": raw_text,
                        "table_idx": table_idx,
                        "picture_idx": picture_idx,
                        "references": all_references,
                        "footnotes": all_footnotes,
                        "_element": text_element  # Store element for reference checking
                    })
                elif current_paragraph:
                    para_intro_texts.append({
                        "text": normalized_text,
                        "raw_text": raw_text,
                        "table_idx": table_idx,
                        "picture_idx": picture_idx,
                        "references": all_references,
                        "footnotes": all_footnotes,
                        "_element": text_element  # Store element for reference checking
                    })
                continue
            
            # This is a real structural marker - proceed with creating new pismeno
            # Close previous pismeno
            _close_pismeno(current_pismeno, pismeno_texts, current_odsek, doc, structure)
            
            # Start new pismeno
            if current_odsek:
                pismenos_count += 1
                para_num_for_pismeno = current_paragraph["id"].replace("paragraf-", "")
                odsek_num_for_pismeno = current_odsek["id"].split(".")[-1]
                pismeno_id = f"pismeno-{para_num_for_pismeno}.{odsek_num_for_pismeno}.{pismeno_letter}"
                current_pismeno = {
                    "id": pismeno_id,
                    "marker": f"{pismeno_letter})",
                    "text": "",
                    "subitems": [],
                    "tables": [],
                    "pictures": [],
                    "references_metadata": [],
                    "footnotes_metadata": []
                }
                pismeno_texts = []
                
                # If we extracted marker from text, add remaining text to pismeno
                if pismeno_extracted and pismeno_remaining_text:
                    references = extract_references_from_text(pismeno_remaining_text)
                    footnotes = extract_footnotes_from_text(pismeno_remaining_text)
                    table_idx = extract_table_reference(hyperlink_str) if has_hyperlink else None
                    picture_idx = extract_picture_reference(hyperlink_str) if has_hyperlink else None
                    pismeno_texts.append({
                        "text": pismeno_remaining_text,
                        "raw_text": raw_text,
                        "table_idx": table_idx,
                        "picture_idx": picture_idx,
                        "references": references,
                        "footnotes": footnotes,
                        "_element": text_element  # Store element for reference checking
                    })
                
                log_progress("DEBUG", f"Found pismeno: {pismeno_letter}) in odsek {odsek_num_for_pismeno}")
            continue
        
        # 4. Check for odsek marker (after pismeno check)
        # Try standalone marker first
        odsek_marker = detect_odsek_marker(normalized_text, hyperlink_str if hyperlink_str else None)
        odsek_extracted = None
        odsek_remaining_text = None
        
        # If not found, try extracting from text (e.g., "(1) content")
        if not odsek_marker:
            odsek_extracted = extract_marker_from_text(normalized_text, 'odsek')
            if odsek_extracted:
                odsek_marker, odsek_remaining_text = odsek_extracted
        
        if odsek_marker:
            # Set end index for previous odsek before closing
            if current_odsek:
                current_odsek["_end_idx"] = item_idx
            # Close previous odsek and pismeno
            _close_pismeno(current_pismeno, pismeno_texts, current_odsek, doc, structure)
            _close_odsek(current_odsek, odsek_texts, para_intro_texts, current_paragraph, doc, structure)
            
            # Start new odsek
            odseks_count += 1
            para_num_for_odsek = current_paragraph["id"].replace("paragraf-", "")
            odsek_id = f"odsek-{para_num_for_odsek}.{odsek_marker}"
            current_odsek = {
                "id": odsek_id,
                "marker": f"({odsek_marker})",
                "text": "",
                "tables": [],
                "pictures": [],
                "references_metadata": [],
                "footnotes_metadata": [],
                "pismenos": [],
                "_start_idx": item_idx,  # Track start index for table detection
                "_marker_text_elem": item  # Track marker element for parent reference
            }
            current_pismeno = None
            odsek_texts = []
            pismeno_texts = []
            
            # If we extracted marker from text, add remaining text to odsek
            if odsek_extracted and odsek_remaining_text:
                references = extract_references_from_text(odsek_remaining_text)
                footnotes = extract_footnotes_from_text(odsek_remaining_text)
                table_idx = extract_table_reference(hyperlink_str) if has_hyperlink else None
                picture_idx = extract_picture_reference(hyperlink_str) if has_hyperlink else None
                odsek_texts.append({
                    "text": odsek_remaining_text,
                    "raw_text": raw_text,
                    "table_idx": table_idx,
                    "picture_idx": picture_idx,
                    "references": references,
                    "footnotes": footnotes,
                    "_element": text_element  # Store element for reference checking
                })
            
            log_progress("DEBUG", f"Found odsek: ({odsek_marker}) in paragraph {para_num_for_odsek}")
            continue
        
        # 5. Check for subitem marker (within pismeno)
        if current_pismeno:
            # Try standalone subitem marker first
            subitem_marker = detect_subitem_marker(normalized_text)
            subitem_extracted = None
            subitem_remaining_text = None
            
            # If not found, try extracting from text (e.g., "1. content")
            if not subitem_marker:
                subitem_extracted = extract_marker_from_text(normalized_text, 'subitem')
                if subitem_extracted:
                    subitem_marker, subitem_remaining_text = subitem_extracted
            
            if subitem_marker:
                # Start new subitem or add to existing
                # Find or create current subitem
                if not current_pismeno.get('_current_subitem') or current_pismeno.get('_current_subitem') != subitem_marker:
                    # Close previous subitem if exists
                    if current_pismeno.get('_current_subitem'):
                        prev_subitem = {
                            "marker": f"{current_pismeno['_current_subitem']}.",
                            "text": "\n".join([t["text"] for t in current_pismeno.get('_subitem_texts', []) if t.get("text")]),
                            "tables": [],
                            "pictures": [],
                            "references_metadata": [],
                            "footnotes_metadata": []
                        }
                        # Extract metadata from subitem texts
                        text_pos = 0
                        for t in current_pismeno.get('_subitem_texts', []):
                            if t.get("table_idx") is not None:
                                table_idx = t["table_idx"]
                                if table_idx < len(doc.tables):
                                    prev_subitem["tables"].append({
                                        "index": table_idx,
                                        "position_in_text": text_pos
                                    })
                            if t.get("picture_idx") is not None:
                                picture_idx = t["picture_idx"]
                                if picture_idx < len(doc.pictures):
                                    prev_subitem["pictures"].append({
                                        "index": picture_idx,
                                        "position_in_text": text_pos
                                    })
                            prev_subitem["references_metadata"].extend(t.get("references", []))
                            prev_subitem["footnotes_metadata"].extend(t.get("footnotes", []))
                            text_pos += len(t.get("text", "")) + 1  # +1 for newline
                        
                        current_pismeno["subitems"].append(prev_subitem)
                    
                    # Start new subitem
                    current_pismeno['_current_subitem'] = subitem_marker
                    current_pismeno['_subitem_texts'] = []
                    subitems_count += 1
                
                # Add remaining text to subitem if extracted
                if subitem_extracted and subitem_remaining_text:
                    references = extract_references_from_text(subitem_remaining_text)
                    footnotes = extract_footnotes_from_text(subitem_remaining_text)
                    table_idx = extract_table_reference(hyperlink_str) if has_hyperlink else None
                    picture_idx = extract_picture_reference(hyperlink_str) if has_hyperlink else None
                    current_pismeno['_subitem_texts'].append({
                        "text": subitem_remaining_text,
                        "raw_text": raw_text,
                        "table_idx": table_idx,
                        "picture_idx": picture_idx,
                        "references": references,
                        "footnotes": footnotes
                    })
                    log_progress("DEBUG", f"Found subitem: {subitem_marker}. in pismeno {current_pismeno.get('marker', '')}")
                    continue
                else:
                    # Standalone subitem marker - just log and continue
                    log_progress("DEBUG", f"Found subitem marker: {subitem_marker}. in pismeno {current_pismeno.get('marker', '')}")
                    continue
        
        # 6. Process content (not a marker)
        # Determine where to add this text
        content_text = normalized_text
        
        # Extract references and footnotes (preserve markers in text)
        references = extract_references_from_text(content_text)
        footnotes = extract_footnotes_from_text(content_text)
        
        # Check for table/picture references from hyperlink
        table_idx = extract_table_reference(hyperlink_str) if has_hyperlink else None
        picture_idx = extract_picture_reference(hyperlink_str) if has_hyperlink else None
        
        # Also check for table_idx and picture_idx attributes directly
        if not table_idx:
            table_idx = getattr(text_element, 'table_idx', None)
        if not picture_idx:
            picture_idx = getattr(text_element, 'picture_idx', None)
        
        # Get references and footnotes from element attributes
        element_references = getattr(text_element, 'references', [])
        element_footnotes = getattr(text_element, 'footnotes', [])
        
        # Combine extracted and element references/footnotes
        all_references = references + (element_references if element_references else [])
        all_footnotes = footnotes + (element_footnotes if element_footnotes else [])
        
        # Add to appropriate level
        if current_pismeno:
            # Check if we're in a subitem
            if current_pismeno.get('_current_subitem'):
                # Add to subitem
                current_pismeno['_subitem_texts'].append({
                    "text": content_text,
                    "raw_text": raw_text,
                    "table_idx": table_idx,
                    "picture_idx": picture_idx,
                    "references": all_references,
                    "footnotes": all_footnotes,
                    "_element": text_element  # Store element for reference checking
                })
            else:
                # Add to pismeno
                pismeno_texts.append({
                    "text": content_text,  # Keep reference markers in text
                    "raw_text": raw_text,
                    "table_idx": table_idx,
                    "picture_idx": picture_idx,
                    "references": all_references,
                    "footnotes": all_footnotes,
                    "_element": text_element  # Store element for reference checking
                })
        elif current_odsek:
            # Add to odsek
            odsek_texts.append({
                "text": content_text,  # Keep reference markers in text
                "raw_text": raw_text,
                "table_idx": table_idx,
                "picture_idx": picture_idx,
                "references": all_references,
                "footnotes": all_footnotes,
                "_element": text_element  # Store element for reference checking
            })
        elif current_paragraph:
            # Add to paragraph intro
            para_intro_texts.append({
                "text": content_text,  # Keep reference markers in text
                "raw_text": raw_text,
                "table_idx": table_idx,
                "picture_idx": picture_idx,
                "references": all_references,
                "footnotes": all_footnotes,
                "_element": text_element  # Store element for reference checking
            })
        else:
            # Part-level content (rare)
            part_texts.append({
                "text": content_text,
                "raw_text": raw_text
            })
    
    # Close any remaining open structures
    # Set end index for last paragraph and odsek before closing
    if current_paragraph:
        current_paragraph["_end_idx"] = total_elements
    if current_odsek:
        current_odsek["_end_idx"] = total_elements
    
    _close_pismeno(current_pismeno, pismeno_texts, current_odsek, doc, structure)
    _close_odsek(current_odsek, odsek_texts, para_intro_texts, current_paragraph, doc, structure)
    _close_paragraph(current_paragraph, para_intro_texts, current_part, doc, structure)
    
    # Finalize part title_text if we collected additional text
    if current_part and part_texts:
        combined_title = " ".join([t.get("text", t) if isinstance(t, dict) else t for t in part_texts])
        if current_part["title_text"]:
            current_part["title_text"] = current_part["title_text"] + " " + combined_title
        else:
            current_part["title_text"] = combined_title
    
    if current_part and current_part["paragraphs"]:  # Only add if has paragraphs
        structure["parts"].append(current_part)
    
    # Close any open annex
    if current_annex:
        # annex_end_idx is end of document (total_elements)
        _close_annex(current_annex, annex_texts, doc, structure, annex_start_idx, total_elements, annex_marker_text_elem, all_annex_texts, seen_annex_tables)
    
        # Close any open footnote
    if current_footnote:
        _close_footnote(current_footnote, footnote_texts, structure)
        footnotes_count += 1
    
    # Build references_index with correct paths after all items are added
    _build_references_index(structure)
    
    # Update metadata (use actual counts from structure, not counters)
    elapsed = time.time() - start_time
    actual_parts_count = len(structure["parts"])  # Only parts with content
    structure["metadata"]["reconstruction_time"] = elapsed
    structure["metadata"]["total_parts"] = actual_parts_count
    structure["metadata"]["total_paragraphs"] = paragraphs_count
    structure["metadata"]["total_odseks"] = odseks_count
    structure["metadata"]["total_pismenos"] = pismenos_count
    structure["metadata"]["total_subitems"] = subitems_count
    structure["metadata"]["total_annexes"] = len(structure.get("annexes", {}).get("annex_list", []))
    structure["metadata"]["total_footnotes"] = footnotes_count
    
    # Initialize annexes summary if not already set
    if "annexes" in structure and "summary" not in structure["annexes"]:
        annex_list = structure["annexes"].get("annex_list", [])
        inline_count = sum(1 for a in annex_list if a.get("source") == "inline")
        external_count = sum(1 for a in annex_list if a.get("source") == "external_pdf")
        structure["annexes"]["summary"] = {
            "total_annexes": len(annex_list),
            "external_annexes": external_count,
            "inline_annexes": inline_count
        }
    
    log_progress("INFO", f"Reconstruction complete: {actual_parts_count} parts, {paragraphs_count} paragraphs, {odseks_count} odseks, {pismenos_count} pismenos, {subitems_count} subitems, {structure['metadata']['total_annexes']} annexes, {footnotes_count} footnotes", elapsed)
    
    return structure


# ============================================================================
# Helper Functions for Closing Levels
# ============================================================================

def _close_annex(annex: Optional[Dict], texts: List[Dict], doc: DoclingDocument, structure: Dict, annex_start_idx: Optional[int] = None, annex_end_idx: Optional[int] = None, annex_marker_text_elem=None, all_annex_texts: List = None, seen_annex_tables: set = None) -> None:
    """
    Close current annex and add to annex_list with structured content.
    Stores content per annex (text, tables, pictures) instead of mixing at top-level.
    
    Args:
        annex: Current annex dictionary
        texts: List of text dictionaries for the annex
        doc: DoclingDocument (for tables/pictures)
        structure: Document structure to update
        annex_start_idx: Index where annex marker was found (for table detection)
        annex_end_idx: Index where next annex starts (for table detection)
        annex_marker_text_elem: The text element of the annex marker (for parent reference)
        all_annex_texts: List to collect all annex texts (deprecated, kept for compatibility)
        seen_annex_tables: Set to track tables and avoid duplicates
    """
    if not annex:
        return
    
    if seen_annex_tables is None:
        seen_annex_tables = set()
    
    # Combine text content for this annex (clean text, no encoding issues)
    content_lines = [t["text"] for t in texts if t.get("text")]
    annex_text = "\n".join(content_lines)
    
    # Collect tables and pictures for this annex
    annex_tables = []
    annex_pictures = []
    
    # Extract tables and pictures from text references
    for t in texts:
        if t.get("table_idx") is not None:
            table_idx = t["table_idx"]
            if hasattr(doc, 'tables') and table_idx < len(doc.tables):
                if table_idx not in seen_annex_tables:
                    # Format table with full data
                    table_data = format_table_for_json(doc.tables[table_idx], doc, table_idx)
                    # Remove position_in_text as it's not needed in per-annex structure
                    if "position_in_text" in table_data:
                        del table_data["position_in_text"]
                    annex_tables.append(table_data)
                    seen_annex_tables.add(table_idx)
        if t.get("picture_idx") is not None:
            picture_idx = t["picture_idx"]
            if hasattr(doc, 'pictures') and picture_idx < len(doc.pictures):
                annex_pictures.append({
                    "index": picture_idx
                })
    
    # Find additional tables that belong to annexes using document structure
    if annex_start_idx is not None and annex_end_idx is not None:
        additional_tables = find_tables_for_annex(doc, annex_start_idx, annex_end_idx, annex_marker_text_elem)
        for table_idx in additional_tables:
            if table_idx not in seen_annex_tables:
                # Format table with full data
                table_data = format_table_for_json(doc.tables[table_idx], doc, table_idx)
                if "position_in_text" in table_data:
                    del table_data["position_in_text"]
                annex_tables.append(table_data)
                seen_annex_tables.add(table_idx)
                log_progress("INFO", f"Found table {table_idx} for annexes section via structure analysis")
    
    # Add annex to annex_list with structured content
    structure["annexes"]["annex_list"].append({
        "id": annex["id"],
        "number": annex["number"],
        "title": annex["title"],
        "source": "inline",
        "content": {
            "text": annex_text,
            "tables": annex_tables,
            "pictures": annex_pictures
        }
    })


def _check_table_assignments(structure: Dict, doc: DoclingDocument) -> None:
    """
    Skontroluje, či sú všetky tabuľky z doc.tables priradené do štruktúry a či nie sú duplicity.
    
    Args:
        structure: Document structure
        doc: DoclingDocument
    """
    if not hasattr(doc, 'tables') or not doc.tables:
        log_progress("INFO", "No tables in document to check")
        return
    
    total_tables = len(doc.tables)
    log_progress("INFO", f"Checking table assignments: {total_tables} tables in document")
    
    # Zbierame všetky indexy tabuliek z štruktúry
    assigned_indices = []
    duplicate_indices = []
    
    # Prejdeme cez všetky parts, paragraphs, odseks, pismenos, subitems
    for part in structure.get("parts", []):
        # Paragraphs
        for para in part.get("paragraphs", []):
            for table in para.get("tables", []):
                idx = table.get("index")
                if idx is not None:
                    if idx in assigned_indices:
                        duplicate_indices.append(idx)
                    assigned_indices.append(idx)
        
        # Odseks
        for para in part.get("paragraphs", []):
            for odsek in para.get("odseks", []):
                for table in odsek.get("tables", []):
                    idx = table.get("index")
                    if idx is not None:
                        if idx in assigned_indices:
                            duplicate_indices.append(idx)
                        assigned_indices.append(idx)
                
                # Pismenos
                for pismeno in odsek.get("pismenos", []):
                    for table in pismeno.get("tables", []):
                        idx = table.get("index")
                        if idx is not None:
                            if idx in assigned_indices:
                                duplicate_indices.append(idx)
                            assigned_indices.append(idx)
                    
                    # Subitems
                    for subitem in pismeno.get("subitems", []):
                        for table in subitem.get("tables", []):
                            idx = table.get("index")
                            if idx is not None:
                                if idx in assigned_indices:
                                    duplicate_indices.append(idx)
                                assigned_indices.append(idx)
    
    # Annexes
    for table in structure.get("annexes", {}).get("tables", []):
        idx = table.get("index")
        if idx is not None:
            if idx in assigned_indices:
                duplicate_indices.append(idx)
            assigned_indices.append(idx)
    
    # Nájdeme nepriradené tabuľky
    all_indices = set(range(total_tables))
    assigned_set = set(assigned_indices)
    unassigned_indices = sorted(all_indices - assigned_set)
    
    # Vypíšeme výsledky
    log_progress("INFO", f"Table assignment summary:")
    log_progress("INFO", f"  Total tables in document: {total_tables}")
    log_progress("INFO", f"  Assigned tables: {len(assigned_set)}")
    log_progress("INFO", f"  Unassigned tables: {len(unassigned_indices)}")
    log_progress("INFO", f"  Duplicate assignments: {len(set(duplicate_indices))}")
    
    if unassigned_indices:
        log_progress("WARNING", f"Unassigned table indices: {unassigned_indices[:20]}{'...' if len(unassigned_indices) > 20 else ''}")
    
    if duplicate_indices:
        unique_duplicates = sorted(set(duplicate_indices))
        log_progress("WARNING", f"Duplicate table assignments: {unique_duplicates[:20]}{'...' if len(unique_duplicates) > 20 else ''}")


def _close_footnote(footnote: Optional[Dict], texts: List[Dict], structure: Dict) -> None:
    """
    Close current footnote and add to structure.
    
    Args:
        footnote: Current footnote dictionary
        texts: List of text dictionaries for the footnote
        structure: Document structure to update
    """
    if not footnote:
        return
    
    # Combine text content
    content_lines = [t["text"] for t in texts if t.get("text")]
    footnote["content"] = "\n".join(content_lines)
    
    # Add to structure
    structure["footnotes"].append(footnote)


def _close_pismeno(pismeno: Optional[Dict], texts: List[Dict], odsek: Optional[Dict], doc: DoclingDocument, structure: Dict) -> None:
    """Close current pismeno and add to odsek, including subitems."""
    if not pismeno or not odsek:
        return
    
    # Close any open subitem first
    if pismeno.get('_current_subitem'):
        subitem_text = "\n".join([t["text"] for t in pismeno.get('_subitem_texts', []) if t.get("text")])
        prev_subitem = {
            "marker": f"{pismeno['_current_subitem']}.",
            "text": subitem_text,
            "tables": [],
            "pictures": [],
            "references_metadata": [],
            "footnotes_metadata": []
        }
        
        # Extract table from subitem text if present
        text_without_table, table_text = extract_table_from_text(prev_subitem["text"])
        subitem_table_idx = None
        if table_text:
            # Update text without table
            prev_subitem["text"] = text_without_table
            log_progress("DEBUG", f"Detected table in text for subitem {prev_subitem.get('marker', 'unknown')} in {pismeno.get('id', 'unknown pismeno')}")
            
            # Find corresponding table in doc.tables
            existing_table_indices = [t.get("table_idx") for t in pismeno.get('_subitem_texts', []) if t.get("table_idx") is not None]
            subitem_table_idx = find_table_by_text_match(table_text, doc, exclude_indices=existing_table_indices)
            
            if subitem_table_idx is not None:
                log_progress("INFO", f"Extracted table {subitem_table_idx} from text in subitem {prev_subitem.get('marker', 'unknown')}")
            else:
                log_progress("WARNING", f"Table detected in subitem text but not found in doc.tables for {prev_subitem.get('marker', 'unknown')}")
        
        # Extract metadata from subitem texts
        text_pos = 0
        for t in pismeno.get('_subitem_texts', []):
            if t.get("table_idx") is not None:
                table_idx = t["table_idx"]
                if table_idx < len(doc.tables):
                    table_data = format_table_for_json(doc.tables[table_idx], doc, table_idx)
                    table_data["position_in_text"] = text_pos
                    prev_subitem["tables"].append(table_data)
            if t.get("picture_idx") is not None:
                picture_idx = t["picture_idx"]
                if picture_idx < len(doc.pictures):
                    prev_subitem["pictures"].append({
                        "index": picture_idx,
                        "position_in_text": text_pos
                    })
            prev_subitem["references_metadata"].extend(t.get("references", []))
            prev_subitem["footnotes_metadata"].extend(t.get("footnotes", []))
            text_pos += len(t.get("text", "")) + 1  # +1 for newline
        
        # Add table found from text extraction if any
        if subitem_table_idx is not None:
            # Check if table is not already in tables list
            if not any(t.get("index") == subitem_table_idx for t in prev_subitem["tables"]):
                table_data = format_table_for_json(doc.tables[subitem_table_idx], doc, subitem_table_idx)
                table_data["position_in_text"] = len(prev_subitem["text"])  # Position at end of text
                prev_subitem["tables"].append(table_data)
        
        pismeno["subitems"].append(prev_subitem)
        # Clean up temporary fields
        del pismeno['_current_subitem']
        del pismeno['_subitem_texts']
    
    # Combine text content (preserve reference markers)
    # Only include text that's not part of subitems
    content_lines = [t["text"] for t in texts if t.get("text")]
    pismeno["text"] = "\n".join(content_lines)
    
    # Extract table from pismeno text if present
    text_without_table, table_text = extract_table_from_text(pismeno["text"])
    pismeno_table_idx = None
    if table_text:
        # Update text without table
        pismeno["text"] = text_without_table
        log_progress("DEBUG", f"Detected table in text for {pismeno.get('id', 'unknown pismeno')}")
        
        # Find corresponding table in doc.tables
        existing_table_indices = [t.get("table_idx") for t in texts if t.get("table_idx") is not None]
        # Also exclude tables from subitems
        for subitem in pismeno.get("subitems", []):
            for table_ref in subitem.get("tables", []):
                if "index" in table_ref:
                    existing_table_indices.append(table_ref["index"])
        
        pismeno_table_idx = find_table_by_text_match(table_text, doc, exclude_indices=existing_table_indices)
        
        if pismeno_table_idx is not None:
            log_progress("INFO", f"Extracted table {pismeno_table_idx} from text in {pismeno.get('id', 'unknown pismeno')}")
        else:
            log_progress("WARNING", f"Table detected in pismeno text but not found in doc.tables for {pismeno.get('id', 'unknown pismeno')}")
    
    # Extract tables, pictures, references, footnotes
    tables = []
    pictures = []
    all_references = []
    all_footnotes = []
    
    text_pos = 0
    for t in texts:
        if t.get("table_idx") is not None:
            table_idx = t["table_idx"]
            if table_idx < len(doc.tables):
                table_data = format_table_for_json(doc.tables[table_idx], doc, table_idx)
                table_data["position_in_text"] = text_pos
                tables.append(table_data)
        if t.get("picture_idx") is not None:
            picture_idx = t["picture_idx"]
            if picture_idx < len(doc.pictures):
                pictures.append({
                    "index": picture_idx,
                    "position_in_text": text_pos
                })
        all_references.extend(t.get("references", []))
        all_footnotes.extend(t.get("footnotes", []))
        text_pos += len(t.get("text", "")) + 1  # +1 for newline
    
    # Add table found from text extraction if any
    if pismeno_table_idx is not None:
        # Check if table is not already in tables list
        if not any(t.get("index") == pismeno_table_idx for t in tables):
            table_data = format_table_for_json(doc.tables[pismeno_table_idx], doc, pismeno_table_idx)
            table_data["position_in_text"] = len(pismeno["text"])  # Position at end of text
            tables.append(table_data)
    
    # Add directly to pismeno
    pismeno["tables"] = tables
    pismeno["pictures"] = pictures
    pismeno["references_metadata"] = all_references
    pismeno["footnotes_metadata"] = all_footnotes
    
    # Add to odsek
    odsek["pismenos"].append(pismeno)


def _close_odsek(odsek: Optional[Dict], texts: List[Dict], para_intro_texts: List[Dict], paragraph: Optional[Dict], doc: DoclingDocument, structure: Dict) -> None:
    """Close current odsek and add to paragraph. Skips empty odseks."""
    if not odsek or not paragraph:
        return
    
    # Combine text content (preserve reference markers)
    content_lines = [t["text"] for t in texts if t.get("text")]
    odsek["text"] = "\n".join(content_lines)
    
    # Extract table from text if present
    text_without_table, table_text = extract_table_from_text(odsek["text"])
    table_idx = None
    if table_text:
        # Update text without table
        odsek["text"] = text_without_table
        log_progress("DEBUG", f"Detected table in text for {odsek.get('id', 'unknown odsek')}")
        
        # Find corresponding table in doc.tables
        # Get already found table indices to exclude them
        existing_table_indices = [t.get("table_idx") for t in texts if t.get("table_idx") is not None]
        table_idx = find_table_by_text_match(table_text, doc, exclude_indices=existing_table_indices)
        
        if table_idx is not None:
            log_progress("INFO", f"Extracted table {table_idx} from text in {odsek.get('id', 'unknown odsek')}")
        else:
            log_progress("WARNING", f"Table detected in text but not found in doc.tables for {odsek.get('id', 'unknown odsek')}")
    
    # Extract tables, pictures, references, footnotes
    tables = []
    pictures = []
    all_references = []
    all_footnotes = []
    
    text_pos = 0
    for t in texts:
        if t.get("table_idx") is not None:
            table_idx = t["table_idx"]
            if table_idx < len(doc.tables):
                table_data = format_table_for_json(doc.tables[table_idx], doc, table_idx)
                table_data["position_in_text"] = text_pos
                tables.append(table_data)
        if t.get("picture_idx") is not None:
            picture_idx = t["picture_idx"]
            if picture_idx < len(doc.pictures):
                pictures.append({
                    "index": picture_idx,
                    "position_in_text": text_pos
                })
        all_references.extend(t.get("references", []))
        all_footnotes.extend(t.get("footnotes", []))
        text_pos += len(t.get("text", "")) + 1  # +1 for newline
    
    # Add table found from text extraction if any
    if table_text and table_idx is not None:
        # Check if table is not already in tables list
        if not any(t.get("index") == table_idx for t in tables):
            table_data = format_table_for_json(doc.tables[table_idx], doc, table_idx)
            table_data["position_in_text"] = len(odsek["text"])  # Position at end of text
            tables.append(table_data)
    
    # Find additional tables using table context map (new approach)
    # Get paragraph and odsek numbers from IDs
    odsek_id = odsek.get("id", "")
    para_id = paragraph.get("id", "")
    
    # Extract section and subsection numbers
    para_num_match = re.match(r'paragraf-(\d+[a-z]?)', para_id)
    # Odsek ID format is "odsek-30e.1" where 30e is paragraph number and 1 is odsek number
    odsek_num_match = re.match(r'odsek-\d+[a-z]?\.(\d+)', odsek_id)
    
    para_num = para_num_match.group(1) if para_num_match else None
    odsek_num = odsek_num_match.group(1) if odsek_num_match else None
    
    # Try to use table context map first (new approach)
    additional_table_indices = []
    if structure and "_table_context_map" in structure and para_num and odsek_num:
        table_context_map = structure["_table_context_map"]
        legal_table_indices = structure.get("_legal_table_indices", set())
        
        # Find all tables assigned to this specific odsek
        for table_idx, (section, subsection) in table_context_map.items():
            if section == para_num and subsection == odsek_num:
                # Only include legal tables (not metadata)
                if table_idx in legal_table_indices:
                    additional_table_indices.append(table_idx)
        
        if additional_table_indices:
            log_progress("INFO", f"Found {len(additional_table_indices)} tables for {odsek_id} via context map: {additional_table_indices}")
    
    # Fallback to old position-based approach if no tables found via context map
    if not additional_table_indices:
        odsek_start_idx = odsek.get("_start_idx")
        odsek_end_idx = odsek.get("_end_idx")
        odsek_marker_elem = odsek.get("_marker_text_elem")
        
        if odsek_start_idx is not None:
            # If end_idx not set, find it (next odsek start or end of paragraph)
            if odsek_end_idx is None:
                # Find next odsek marker or end of paragraph
                odsek_end_idx = odsek_start_idx + 1
                while odsek_end_idx < len(doc.texts):
                    text_elem = doc.texts[odsek_end_idx]
                    tx = getattr(text_elem, 'text', '') if hasattr(text_elem, 'text') else (text_elem.get("text", "") if isinstance(text_elem, dict) else str(text_elem))
                    normalized_tx = tx.strip().replace('\xa0', ' ')
                    # Check if it's next odsek marker or paragraph marker
                    if (normalized_tx.startswith('(') and normalized_tx[1:2].isdigit()) or normalized_tx.startswith('§ '):
                        break
                    odsek_end_idx += 1
            
            log_progress("DEBUG", f"Searching for tables for {odsek.get('id', 'unknown odsek')} using find_tables_for_unit (start_idx={odsek_start_idx}, end_idx={odsek_end_idx})")
            
            # Find tables for this odsek using position-based strategy
            additional_table_indices = find_tables_for_unit(
                doc, odsek_start_idx, odsek_end_idx, odsek_marker_elem
            )
            
            log_progress("DEBUG", f"find_tables_for_unit returned {len(additional_table_indices)} tables: {additional_table_indices} for {odsek.get('id', 'unknown odsek')}")
            
            # Filter out metadata tables if we have the classification
            if structure and "_legal_table_indices" in structure:
                legal_table_indices = structure["_legal_table_indices"]
                additional_table_indices = [idx for idx in additional_table_indices if idx in legal_table_indices]
    
    # Filter out tables that are already found
    existing_table_indices = {t.get("index") for t in tables}
    additional_table_indices = [idx for idx in additional_table_indices if idx not in existing_table_indices]
    
    # Add additional tables
    for idx in additional_table_indices:
        if idx < len(doc.tables):
            table_data = format_table_for_json(doc.tables[idx], doc, idx)
            table_data["position_in_text"] = len(odsek["text"])  # Position at end of text
            tables.append(table_data)
            log_progress("DEBUG", f"Found table {idx} for {odsek.get('id', 'unknown odsek')}")
    
    # Add directly to odsek
    odsek["tables"] = tables
    odsek["pictures"] = pictures
    odsek["references_metadata"] = all_references
    odsek["footnotes_metadata"] = all_footnotes
    
    # Check if odsek has any content
    has_text = bool(odsek["text"].strip())
    has_pismenos = bool(odsek.get("pismenos", []))
    has_tables = bool(tables)
    has_pictures = bool(pictures)
    
    # Only add odsek if it has content
    if not (has_text or has_pismenos or has_tables or has_pictures):
        log_progress("DEBUG", f"Skipping empty odsek: {odsek['id']}")
        return
    
    # Add to paragraph
    paragraph["odseks"].append(odsek)


def _close_paragraph(paragraph: Optional[Dict], intro_texts: List[Dict], part: Optional[Dict], doc: DoclingDocument, structure: Optional[Dict] = None) -> None:
    """Close current paragraph and add to part.
    
    Uses table context map for accurate table assignment when available.
    """
    if not paragraph or not part:
        return
    
    # Combine intro text (preserve reference markers)
    intro_lines = [t["text"] for t in intro_texts if t.get("text")]
    paragraph["intro_text"] = "\n".join(intro_lines)
    
    # Extract table from intro text if present
    text_without_table, table_text = extract_table_from_text(paragraph["intro_text"])
    table_idx_from_text = None
    if table_text:
        # Update intro text without table
        paragraph["intro_text"] = text_without_table
        log_progress("DEBUG", f"Detected table in intro text for {paragraph.get('id', 'unknown paragraph')}")
        
        # Find corresponding table in doc.tables
        # Get already found table indices from odseks to exclude them
        existing_table_indices = []
        for odsek in paragraph.get("odseks", []):
            for table_ref in odsek.get("tables", []):
                if "index" in table_ref:
                    existing_table_indices.append(table_ref["index"])
        
        table_idx_from_text = find_table_by_text_match(table_text, doc, exclude_indices=existing_table_indices)
        
        if table_idx_from_text is not None:
            log_progress("INFO", f"Extracted table {table_idx_from_text} from intro text in {paragraph.get('id', 'unknown paragraph')}")
        else:
            log_progress("WARNING", f"Table detected in intro text but not found in doc.tables for {paragraph.get('id', 'unknown paragraph')}")
    
    # Collect table indices already assigned to odseks in this paragraph
    odsek_table_indices = set()
    for odsek in paragraph.get("odseks", []):
        for table_ref in odsek.get("tables", []):
            if "index" in table_ref:
                odsek_table_indices.add(table_ref["index"])
    
    # Get paragraph number from marker (e.g., "§ 26" -> "26")
    para_marker = paragraph.get("marker", "")
    para_num_match = re.match(r'§\s*(\d+[a-z]?)', para_marker)
    para_num = para_num_match.group(1) if para_num_match else None
    
    # Try to use table context map first (new approach)
    table_indices = []
    if structure and "_table_context_map" in structure and para_num:
        table_context_map = structure["_table_context_map"]
        legal_table_indices = structure.get("_legal_table_indices", set())
        
        # Find all tables assigned to this paragraph (any odsek)
        for table_idx, (section, subsection) in table_context_map.items():
            if section == para_num:
                # Only include legal tables (not metadata)
                if table_idx in legal_table_indices:
                    # Skip if already assigned to an odsek
                    if table_idx not in odsek_table_indices:
                        table_indices.append(table_idx)
        
        if table_indices:
            log_progress("INFO", f"Found {len(table_indices)} tables for paragraph § {para_num} via context map: {table_indices}")
    
    # Fallback to old approach if no tables found via context map
    if not table_indices:
        para_start_idx = paragraph.get("_start_idx")
        para_end_idx = paragraph.get("_end_idx")
        para_marker_elem = paragraph.get("_marker_text_elem")
        
        if para_start_idx is not None:
            # If end_idx not set, find it (next paragraph start or end of document)
            if para_end_idx is None:
                para_end_idx = para_start_idx + 1
                while para_end_idx < len(doc.texts):
                    text_elem = doc.texts[para_end_idx]
                    tx = getattr(text_elem, 'text', '') if hasattr(text_elem, 'text') else (text_elem.get("text", "") if isinstance(text_elem, dict) else str(text_elem))
                    if tx.startswith("§ ") and para_end_idx != para_start_idx:
                        break
                    para_end_idx += 1
            
            # Find tables for this paragraph using old approach
            table_indices = find_tables_for_paragraph_by_content(
                doc, para_start_idx, para_end_idx, para_marker_elem
            )
            
            # Filter out tables that are already assigned to odseks
            table_indices = [idx for idx in table_indices if idx not in odsek_table_indices]
            
            # Filter out metadata tables if we have the classification
            if structure and "_legal_table_indices" in structure:
                legal_table_indices = structure["_legal_table_indices"]
                table_indices = [idx for idx in table_indices if idx in legal_table_indices]
    
    # Initialize paragraph tables list with full table data
    paragraph["tables"] = []
    for idx in table_indices:
        if idx < len(doc.tables):
            table_data = format_table_for_json(doc.tables[idx], doc, idx)
            paragraph["tables"].append(table_data)
    
    # Add table found from text extraction if any
    if table_idx_from_text is not None:
        # Check if table is not already in tables list
        if not any(t.get("index") == table_idx_from_text for t in paragraph["tables"]):
            if table_idx_from_text < len(doc.tables):
                table_data = format_table_for_json(doc.tables[table_idx_from_text], doc, table_idx_from_text)
                paragraph["tables"].append(table_data)
    
    if paragraph["tables"]:
        log_progress("DEBUG", f"Found {len(paragraph['tables'])} tables for paragraph {paragraph.get('id')}: {[t.get('index') for t in paragraph['tables']]}")
    
    # Clean up internal tracking fields before adding to structure
    paragraph.pop("_start_idx", None)
    paragraph.pop("_end_idx", None)
    paragraph.pop("_marker_text_elem", None)
    
    # Add to part
    part["paragraphs"].append(paragraph)


# ============================================================================
# HTML Fallback for Missing Pismenos
# ============================================================================

def extract_pismenos_from_html(html_path: Path, paragraph_id: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extract intro text and pismenos from HTML source for a specific paragraph.
    
    The HTML source has proper structure with <div class="pismeno"> elements
    that Docling may have lost during conversion.
    
    Args:
        html_path: Path to HTML source file
        paragraph_id: Paragraph ID (e.g., "paragraf-2")
        
    Returns:
        Tuple of (intro_text, list of pismeno dictionaries)
    """
    from bs4 import BeautifulSoup
    
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
    except Exception as e:
        log_progress("WARNING", f"Failed to load HTML for pismeno extraction: {e}")
        return "", []
    
    # Find the paragraph div
    para_div = soup.find('div', id=paragraph_id)
    if not para_div:
        return "", []
    
    # Extract intro text - the direct child <div class="text"> that has id like "paragraf-2.text"
    intro_text = ""
    intro_div = para_div.find('div', id=f"{paragraph_id}.text", class_='text')
    if intro_div:
        intro_text = intro_div.get_text(strip=True)
    
    pismenos = []
    
    # Find all pismeno divs within this paragraph
    pismeno_divs = para_div.find_all('div', class_='pismeno')
    
    for pismeno_div in pismeno_divs:
        div_id = pismeno_div.get('id', '')
        
        # Check if this pismeno belongs to our paragraph
        if not div_id.startswith(f"{paragraph_id}.pismeno-"):
            continue
        
        # Extract pismeno letter from ID
        pismeno_letter = div_id.split('.pismeno-')[-1] if '.pismeno-' in div_id else ''
        
        # Get marker from pismenoOznacenie div
        marker_div = pismeno_div.find('div', class_='pismenoOznacenie')
        marker = marker_div.get_text(strip=True) if marker_div else f"{pismeno_letter})"
        
        # Get main text from text div (direct child, not from nested bods)
        text_div = pismeno_div.find('div', class_='text', recursive=False)
        # Also check for direct text div child
        if not text_div:
            for child in pismeno_div.children:
                if hasattr(child, 'get') and child.get('class') and 'text' in child.get('class', []):
                    text_div = child
                    break
        
        main_text = text_div.get_text(strip=True) if text_div else ''
        
        # Extract subitems (bod elements)
        subitems = []
        bod_divs = pismeno_div.find_all('div', class_='bod')
        
        for bod_div in bod_divs:
            bod_marker_div = bod_div.find('div', class_='bodOznacenie')
            bod_text_div = bod_div.find('div', class_='text')
            
            bod_marker = bod_marker_div.get_text(strip=True) if bod_marker_div else ''
            bod_text = bod_text_div.get_text(strip=True) if bod_text_div else ''
            
            if bod_marker or bod_text:
                subitems.append({
                    "marker": bod_marker,
                    "text": bod_text,
                    "tables": [],
                    "pictures": [],
                    "references_metadata": [],
                    "footnotes_metadata": []
                })
        
        # Create pismeno entry
        pismeno_entry = {
            "id": f"pismeno-{paragraph_id.replace('paragraf-', '')}.{pismeno_letter}",
            "marker": marker,
            "text": main_text,
            "subitems": subitems,
            "tables": [],
            "pictures": [],
            "references_metadata": [],
            "footnotes_metadata": []
        }
        
        pismenos.append(pismeno_entry)
    
    return intro_text, pismenos


def enrich_paragraphs_from_html(structure: Dict[str, Any], html_path: Path) -> int:
    """
    Post-process document structure to fill in missing pismenos from HTML source.
    
    Docling sometimes loses the pismeno structure for certain paragraphs (like § 2).
    This function detects such paragraphs (no odseks, content in intro_text) and
    extracts the proper pismeno structure from HTML.
    
    Args:
        structure: Document structure from reconstruct_document()
        html_path: Path to HTML source file
        
    Returns:
        Number of paragraphs enriched
    """
    if not html_path or not html_path.exists():
        log_progress("DEBUG", "No HTML source available for pismeno enrichment")
        return 0
    
    enriched_count = 0
    
    for part in structure.get("parts", []):
        for paragraph in part.get("paragraphs", []):
            # Check if this paragraph needs enrichment:
            # - Has no odseks
            # - Has significant intro_text (indicating content was dumped there)
            odseks = paragraph.get("odseks", [])
            intro_text = paragraph.get("intro_text", "")
            
            if not odseks and len(intro_text) > 200:
                paragraph_id = paragraph.get("id", "")
                
                # Try to extract intro text and pismenos from HTML
                html_intro_text, pismenos = extract_pismenos_from_html(html_path, paragraph_id)
                
                if pismenos:
                    # Create a synthetic odsek to hold the pismenos
                    # (since the original structure expects pismenos inside odseks)
                    synthetic_odsek = {
                        "id": f"odsek-{paragraph_id.replace('paragraf-', '')}.1",
                        "marker": "",  # No marker for synthetic odsek
                        "text": "",  # No odsek-level intro text
                        "tables": [],
                        "pictures": [],
                        "references_metadata": [],
                        "footnotes_metadata": [],
                        "pismenos": pismenos
                    }
                    
                    # Set paragraph intro to the extracted intro from HTML
                    # e.g., "Na účely tohto zákona sa rozumie"
                    paragraph["intro_text"] = html_intro_text
                    paragraph["odseks"] = [synthetic_odsek]
                    enriched_count += 1
                    
                    log_progress("INFO", f"Enriched {paragraph_id} with {len(pismenos)} pismenos from HTML (intro: '{html_intro_text[:50]}...')")
    
    return enriched_count


def _build_references_index(structure: Dict[str, Any]) -> None:
    """
    Build references_index with correct paths after all items are added.
    
    Args:
        structure: Document structure to update
    """
    index = {}
    
    for part_idx, part in enumerate(structure["parts"]):
        for para_idx, para in enumerate(part["paragraphs"]):
            para_id = para["id"]
            index[para_id] = {
                "type": "paragraph",
                "path": ["parts", part_idx, "paragraphs", para_idx]
            }
            
            for odsek_idx, odsek in enumerate(para["odseks"]):
                odsek_id = odsek["id"]
                index[odsek_id] = {
                    "type": "odsek",
                    "path": ["parts", part_idx, "paragraphs", para_idx, "odseks", odsek_idx]
                }
                
                for pismeno_idx, pismeno in enumerate(odsek["pismenos"]):
                    pismeno_id = pismeno["id"]
                    index[pismeno_id] = {
                        "type": "pismeno",
                        "path": ["parts", part_idx, "paragraphs", para_idx, "odseks", odsek_idx, "pismenos", pismeno_idx]
                    }
    
    structure["references_index"] = index


# ============================================================================
# Output Functions
# ============================================================================

def _clean_structure_for_json(obj: Any) -> Any:
    """
    Remove non-serializable fields from structure before JSON export.
    
    Removes:
    - _start_idx, _end_idx (internal tracking fields)
    - _marker_text_elem (TextItem objects that are not JSON serializable)
    - _table_context_map (internal table mapping)
    - _legal_table_indices (internal set)
    
    Args:
        obj: Structure object (dict, list, or primitive)
        
    Returns:
        Cleaned object ready for JSON serialization
    """
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            # Skip internal tracking fields
            if key in ['_start_idx', '_end_idx', '_marker_text_elem', '_table_context_map', '_legal_table_indices']:
                continue
            cleaned[key] = _clean_structure_for_json(value)
        return cleaned
    elif isinstance(obj, list):
        return [_clean_structure_for_json(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)  # Convert sets to lists for JSON
    elif hasattr(obj, '__dict__'):
        # Try to convert objects with __dict__ to dict
        try:
            # Check if it's a Docling object (TextItem, etc.)
            if hasattr(obj, 'text'):
                return str(getattr(obj, 'text', ''))
            elif hasattr(obj, '__class__'):
                # Try to get string representation
                return str(obj)
            else:
                return obj
        except:
            return str(obj)
    else:
        return obj


def save_json(structure: Dict[str, Any], output_path: str) -> None:
    """
    Save reconstructed structure to JSON file.
    
    Args:
        structure: Reconstructed document structure
        output_path: Path to output JSON file
    """
    start_time = time.time()
    log_progress("INFO", f"Saving reconstructed structure to JSON...")
    
    # Clean structure before serialization (remove non-serializable fields)
    cleaned_structure = _clean_structure_for_json(structure)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_structure, f, ensure_ascii=False, indent=2)
    
    elapsed = time.time() - start_time
    log_progress("INFO", f"JSON saved: {output_path}", elapsed)


def format_table_for_json(table: Any, doc: DoclingDocument, table_idx: int) -> Dict[str, Any]:
    """
    Format a table for JSON serialization with both markdown and structured data.
    
    Uses data.grid directly (faster) with DataFrame as fallback.
    
    Args:
        table: Table object from doc.tables
        doc: DoclingDocument
        table_idx: Index of the table
        
    Returns:
        Dictionary with table data in multiple formats
    """
    result = {
        "index": table_idx,
        "markdown": "",
        "caption": None,
        "data": None
    }
    
    try:
        # Get caption - caption_text might be a method, so try calling it
        caption = None
        if hasattr(table, 'caption_text'):
            caption_attr = getattr(table, 'caption_text')
            if callable(caption_attr):
                try:
                    caption = caption_attr()
                except:
                    pass
            else:
                caption = caption_attr
        
        if not caption and hasattr(table, 'caption'):
            caption_attr = getattr(table, 'caption')
            if callable(caption_attr):
                try:
                    caption = caption_attr()
                except:
                    pass
            else:
                caption = caption_attr
        
        result["caption"] = str(caption) if caption else None
        
        # Try to use data.grid directly first (faster approach)
        rows = table_to_rows_from_grid(table)
        
        if rows and len(rows) > 0:
            # We have data from grid, use it
            columns = rows[0] if rows else []
            data_rows = rows[1:] if len(rows) > 1 else []
            
            # Markdown format
            markdown_lines = []
            if columns:
                header = "| " + " | ".join(str(col) for col in columns) + " |"
                markdown_lines.append(header)
                separator = "| " + " | ".join(["---"] * len(columns)) + " |"
                markdown_lines.append(separator)
                for row in data_rows:
                    row_str = "| " + " | ".join(str(val) for val in row) + " |"
                    markdown_lines.append(row_str)
                result["markdown"] = "\n".join(markdown_lines)
            
            # JSON data format
            result["data"] = {
                "columns": columns,
                "rows": data_rows
            }
        else:
            # Fallback to DataFrame export if grid is not available
            df = table.export_to_dataframe(doc=doc)
            if df is not None and not df.empty:
                # Markdown format
                markdown_lines = []
                header = "| " + " | ".join(str(col) for col in df.columns) + " |"
                markdown_lines.append(header)
                separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
                markdown_lines.append(separator)
                for _, row in df.iterrows():
                    row_str = "| " + " | ".join(str(val) for val in row.values) + " |"
                    markdown_lines.append(row_str)
                result["markdown"] = "\n".join(markdown_lines)
                
                # JSON data format - convert to native Python types
                import numpy as np
                
                columns = [str(col) for col in df.columns]
                rows = []
                for _, row in df.iterrows():
                    # Convert each value to native Python type
                    row_data = []
                    for val in row.values:
                        # Handle numpy types and convert to native Python types
                        if val is None:
                            row_data.append(None)
                        elif isinstance(val, (np.integer, np.floating)):
                            row_data.append(val.item())
                        elif isinstance(val, np.ndarray):
                            row_data.append(val.tolist())
                        elif isinstance(val, (str, int, float, bool)):
                            row_data.append(val)
                        elif callable(val):  # Skip methods/functions
                            row_data.append(None)
                        else:
                            # Convert everything else to string
                            try:
                                row_data.append(str(val))
                            except:
                                row_data.append(None)
                    rows.append(row_data)
                
                result["data"] = {
                    "columns": columns,
                    "rows": rows
                }
    except Exception as e:
        result["markdown"] = f"[Table conversion error: {e}]"
        log_progress("WARNING", f"Error formatting table {table_idx}: {e}")
    
    return result


def format_table_as_markdown(table: Any, doc: DoclingDocument) -> str:
    """
    Format a table as markdown.
    
    Uses data.grid directly (faster) with DataFrame as fallback.
    
    Args:
        table: Table object from doc.tables
        doc: DoclingDocument
        
    Returns:
        Markdown-formatted table string
    """
    try:
        # Try to use data.grid directly first (faster approach)
        rows = table_to_rows_from_grid(table)
        
        if rows and len(rows) > 0:
            # We have data from grid, use it
            markdown_lines = []
            
            # Header
            if len(rows) > 0:
                header = "| " + " | ".join(str(col) for col in rows[0]) + " |"
                markdown_lines.append(header)
                
                # Separator
                separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
                markdown_lines.append(separator)
                
                # Rows
                for row in rows[1:]:
                    row_str = "| " + " | ".join(str(val) for val in row) + " |"
                    markdown_lines.append(row_str)
            
            return "\n".join(markdown_lines)
        else:
            # Fallback to DataFrame export if grid is not available
            df = table.export_to_dataframe(doc=doc)
            if df.empty:
                return ""
            
            # Convert DataFrame to markdown
            markdown_lines = []
            
            # Header
            header = "| " + " | ".join(str(col) for col in df.columns) + " |"
            markdown_lines.append(header)
            
            # Separator
            separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
            markdown_lines.append(separator)
            
            # Rows
            for _, row in df.iterrows():
                row_str = "| " + " | ".join(str(val) for val in row.values) + " |"
                markdown_lines.append(row_str)
            
            return "\n".join(markdown_lines)
    except Exception as e:
        return f"[Table conversion error: {e}]"


def format_picture_as_markdown(picture: Any, doc: DoclingDocument) -> str:
    """
    Format a picture as markdown.
    
    Args:
        picture: Picture object from doc.pictures
        doc: DoclingDocument
        
    Returns:
        Markdown-formatted picture string
    """
    caption = getattr(picture, 'caption', None)
    if caption:
        return f"\n![{caption}]\n\n"
    return "\n![Image]\n\n"


def save_markdown(structure: Dict[str, Any], doc: DoclingDocument, output_path: str) -> None:
    """
    Save reconstructed structure to Markdown file (readable like PDF).
    
    Args:
        structure: Reconstructed document structure
        doc: DoclingDocument (for tables/pictures)
        output_path: Path to output Markdown file
    """
    start_time = time.time()
    log_progress("INFO", f"Saving reconstructed structure to Markdown...")
    
    lines = []
    
    # Document header
    lines.append("# " + structure["document_name"] + "\n\n")
    lines.append(f"*Reconstructed from: {structure['metadata']['source_file']}*\n")
    lines.append(f"*Reconstruction time: {structure['metadata']['reconstruction_time']:.2f}s*\n")
    lines.append(f"*Parts: {structure['metadata']['total_parts']}, Paragraphs: {structure['metadata']['total_paragraphs']}, Odseks: {structure['metadata']['total_odseks']}, Pismenos: {structure['metadata']['total_pismenos']}, Subitems: {structure['metadata']['total_subitems']}, Annexes: {structure['metadata']['total_annexes']}, Footnotes: {structure['metadata']['total_footnotes']}*\n\n")
    lines.append("---\n\n")
    
    # Process each part
    for part in structure["parts"]:
        # Part title
        lines.append(f"# {part['title']}\n\n")
        # Display full title_text if it contains more than just the title
        if part.get("title_text"):
            title_text = part["title_text"].strip()
            # If title_text is different from title, display it
            if title_text != part["title"]:
                # Extract the additional text (everything after the title)
                additional_text = title_text.replace(part["title"], "").strip()
                if additional_text:
                    lines.append(f"{additional_text}\n\n")
        
        # Process paragraphs
        for para in part["paragraphs"]:
            # Paragraph title
            lines.append(f"## {para['title']}\n\n")
            
            # Paragraph intro text
            if para.get("intro_text"):
                lines.append(f"{para['intro_text']}\n\n")
            
            # Tables at paragraph level (found via GPT-inspired approach)
            for table_ref in para.get("tables", []):
                table_idx = table_ref["index"]
                if hasattr(doc, 'tables') and table_idx < len(doc.tables):
                    lines.append(format_table_as_markdown(doc.tables[table_idx], doc))
                    lines.append("\n")
            
            # Process odseks
            for odsek in para["odseks"]:
                # Odsek marker
                lines.append(f"### {odsek['marker']}\n\n")
                
                # Odsek content
                if odsek.get("text"):
                    lines.append(f"{odsek['text']}\n\n")
                
                # Tables in odsek
                for table_ref in odsek.get("tables", []):
                    table_idx = table_ref["index"]
                    if table_idx < len(doc.tables):
                        lines.append(format_table_as_markdown(doc.tables[table_idx], doc))
                
                # Pictures in odsek
                for pic_ref in odsek.get("pictures", []):
                    pic_idx = pic_ref["index"]
                    if pic_idx < len(doc.pictures):
                        lines.append(format_picture_as_markdown(doc.pictures[pic_idx], doc))
                
                # Process pismenos
                for pismeno in odsek["pismenos"]:
                    # Pismeno marker with bold formatting
                    if pismeno.get("text"):
                        text = pismeno['text'].strip()
                        # Put marker and first line of text together
                        lines.append(f"    **{pismeno['marker']}** {text}\n\n")
                    else:
                        lines.append(f"    **{pismeno['marker']}**\n\n")
                    
                    # Subitems in pismeno (if any) - indented further
                    for subitem in pismeno.get("subitems", []):
                        if subitem.get("text"):
                            lines.append(f"      - {subitem['marker']} {subitem['text']}\n")
                        else:
                            lines.append(f"      - {subitem['marker']}\n")
                    if pismeno.get("subitems"):
                        lines.append("\n")
                    
                    # Tables in pismeno
                    for table_ref in pismeno.get("tables", []):
                        table_idx = table_ref["index"]
                        if table_idx < len(doc.tables):
                            lines.append(format_table_as_markdown(doc.tables[table_idx], doc))
                    
                    # Pictures in pismeno
                    for pic_ref in pismeno.get("pictures", []):
                        pic_idx = pic_ref["index"]
                        if pic_idx < len(doc.pictures):
                            lines.append(format_picture_as_markdown(doc.pictures[pic_idx], doc))
            
            lines.append("\n---\n\n")
    
    # Process annexes (iterate through annex_list)
    annexes_section = structure.get("annexes", {})
    annex_list = annexes_section.get("annex_list", [])
    if annex_list:
        lines.append("# Prílohy\n\n")
        
        # Process each annex separately
        for annex in annex_list:
            # Annex title
            annex_title = annex.get('title') or f"Príloha č. {annex.get('number', '?')}"
            lines.append(f"## {annex_title}\n\n")
            
            # Annex text content
            content = annex.get("content", {})
            if content.get("text"):
                lines.append(f"{content['text']}\n\n")
            
            # Tables for this annex
            for table_data in content.get("tables", []):
                table_idx = table_data.get("index")
                if table_idx is not None and hasattr(doc, 'tables') and table_idx < len(doc.tables):
                    lines.append(format_table_as_markdown(doc.tables[table_idx], doc))
            
            # Pictures for this annex
            for pic_ref in content.get("pictures", []):
                pic_idx = pic_ref.get("index")
                if pic_idx is not None and hasattr(doc, 'pictures') and pic_idx < len(doc.pictures):
                    lines.append(format_picture_as_markdown(doc.pictures[pic_idx], doc))
            
            lines.append("\n---\n\n")
    
    # Process footnotes
    if structure.get("footnotes"):
        lines.append("# Poznámky\n\n")
        for footnote in structure["footnotes"]:
            lines.append(f"**{footnote['number']})** {footnote['content']}\n\n")
        lines.append("\n---\n\n")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("".join(lines))
    
    elapsed = time.time() - start_time
    log_progress("INFO", f"Markdown saved: {output_path}", elapsed)


# ============================================================================
# Manifest Loading
# ============================================================================

def load_manifest_yaml(path: Path) -> dict:
    """
    Load YAML manifest file.
    
    Args:
        path: Path to manifest.yaml
        
    Returns:
        Dictionary with manifest data
    """
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Main function for sequential document reconstruction.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Sequential document reconstruction parser for Slovak law documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python sequential_parser.py output/document.json
  
  # With annexes from PDF files
  python sequential_parser.py output/document.json --annexes-dir input --law-id "595/2003"
  
  # With HTML source for annex detection
  python sequential_parser.py output/document.json --html-source input/document.html --annexes-dir input
        """
    )
    parser.add_argument('input_json', type=str, help='Path to input Docling JSON file')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory (default: output)')
    parser.add_argument('--manifest', type=str, help='Path to YAML manifest file (alternative to --annexes-dir/--html-source)')
    parser.add_argument('--annexes-dir', type=str, help='Directory containing PDF annexes')
    parser.add_argument('--html-source', type=str, help='Path to source HTML file for annex detection')
    parser.add_argument('--law-id', type=str, help='Law identifier (e.g., "595/2003")')
    parser.add_argument('--cache-dir', type=str, default='cache', help='Directory for caching annex conversions')
    parser.add_argument('--no-annexes', action='store_true', help='Skip annex processing')
    
    args = parser.parse_args()
    
    # Prepare output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    input_path = Path(args.input_json)
    base_name = input_path.stem
    
    # Set up log file
    log_file_path = output_dir / f"{base_name}_sequential.log"
    set_log_file(str(log_file_path))
    
    try:
        # Load document
        load_start = time.time()
        log_progress("INFO", "=" * 70)
        log_progress("INFO", "Sequential Document Reconstruction Parser")
        log_progress("INFO", "=" * 70)
        log_progress("INFO", f"Loading document from {args.input_json}...")
        doc = load_docling_document(args.input_json)
        load_time = time.time() - load_start
        log_progress("INFO", f"Document loaded: {getattr(doc, 'name', 'Unknown')}", load_time)
        log_progress("INFO", f"  Texts: {len(doc.texts):,}, Tables: {len(doc.tables):,}, Pictures: {len(doc.pictures):,}")
        log_progress("INFO", f"Log file: {log_file_path}")
        log_progress("INFO", "-" * 70)
        
        # Reconstruct document using SequentialLawChunker (docling-native approach)
        chunker = SequentialLawChunker()
        structure = chunker._reconstruct_document_with_docling(doc)
        
        # Enrich paragraphs with pismenos from HTML source if available
        html_path = None
        if args.html_source:
            html_path = Path(args.html_source)
        else:
            # Try to find HTML in parent directory (common location)
            input_parent = input_path.parent.parent  # Go up from cache/ to law directory
            html_files = list(input_parent.glob("*.html"))
            if html_files:
                html_path = html_files[0]
                log_progress("INFO", f"Auto-detected HTML source: {html_path.name}")
        
        if html_path and html_path.exists():
            enriched = enrich_paragraphs_from_html(structure, html_path)
            if enriched > 0:
                log_progress("INFO", f"Enriched {enriched} paragraphs with pismenos from HTML")
        
        # Process annexes if requested
        if not args.no_annexes and (args.manifest or args.annexes_dir or args.html_source):
            log_progress("INFO", "-" * 70)
            log_progress("INFO", "Processing annexes...")
            
            try:
                from annex_processor import AnnexProcessor, LawManifest
                
                processor = AnnexProcessor()
                annexes = []
                law_dir = None
                
                # Load from manifest if provided
                if args.manifest:
                    manifest_path = Path(args.manifest)
                    if manifest_path.exists():
                        manifest = LawManifest.load(manifest_path)
                        annexes = manifest.annexes
                        law_dir = manifest_path.parent
                        log_progress("INFO", f"Loaded {len(annexes)} annexes from manifest")
                    else:
                        log_progress("WARNING", f"Manifest not found: {manifest_path}")
                
                # Otherwise detect annexes
                if not annexes:
                    if args.html_source:
                        html_path = Path(args.html_source)
                        if html_path.exists():
                            annexes = processor.detect_annexes_from_html(html_path)
                            log_progress("INFO", f"Detected {len(annexes)} annexes from HTML")
                        else:
                            log_progress("WARNING", f"HTML source not found: {html_path}")
                    
                    if not annexes:
                        # Try to detect from Docling document
                        annexes = processor.detect_annexes_from_docling(doc)
                        log_progress("INFO", f"Detected {len(annexes)} annexes from Docling document")
                
                # Determine law_dir if not from manifest
                if law_dir is None:
                    # Try to infer from input path or use cache_dir
                    if args.cache_dir:
                        # Assume law_dir is parent of cache_dir or use cache_dir itself
                        cache_path = Path(args.cache_dir)
                        # Try to find law directory structure
                        if cache_path.name == 'cache' and cache_path.parent.exists():
                            law_dir = cache_path.parent
                        else:
                            law_dir = cache_path
                    else:
                        # Fallback: use input directory
                        law_dir = input_path.parent
                
                # Integrate annexes
                if annexes:
                    annexes_dir = Path(args.annexes_dir) if args.annexes_dir else None
                    structure = processor.integrate_annexes(
                        structure, 
                        annexes, 
                        law_dir,
                        annexes_dir
                    )
                    log_progress("INFO", f"Integrated {len(structure.get('annexes', {}).get('annex_list', []))} annexes")
                
            except ImportError:
                log_progress("WARNING", "annex_processor module not available, skipping annex processing")
            except Exception as e:
                log_progress("ERROR", f"Error processing annexes: {e}")
                import traceback
                traceback.print_exc()
        
        json_output = output_dir / f"{base_name}_sequential.json"
        md_output = output_dir / f"{base_name}_sequential.md"
        
        # Save outputs
        log_progress("INFO", "-" * 70)
        save_json(structure, str(json_output))
        save_markdown(structure, doc, str(md_output))
        
        # Summary
        log_progress("INFO", "=" * 70)
        log_progress("INFO", "Reconstruction complete!")
        log_progress("INFO", f"Output files:")
        log_progress("INFO", f"  JSON: {json_output}")
        log_progress("INFO", f"  Markdown: {md_output}")
        log_progress("INFO", f"  Log: {log_file_path}")
        
        # Annex summary
        annex_list = structure.get('annexes', {}).get('annex_list', [])
        if annex_list:
            log_progress("INFO", f"  Annexes: {len(annex_list)}")
            for annex in annex_list:
                tables_count = len(annex.get('tables', []))
                log_progress("INFO", f"    - {annex.get('title', 'Unknown')}: {tables_count} tables")
        
        log_progress("INFO", "=" * 70)
    
    finally:
        # Always close log file
        close_log_file()


if __name__ == '__main__':
    main()

