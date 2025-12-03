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
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


from docling_core.types.doc import DoclingDocument


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


def detect_paragraph_marker(text: str) -> Optional[str]:
    """
    Check if text is a paragraph marker (§ N).
    
    Args:
        text: Normalized text to check
        
    Returns:
        Paragraph number (e.g., "5", "10a") or None
    """
    if not text.startswith('§ '):
        return None
    
    match = re.match(r'^§\s+(\d+[a-zA-Z]*)', text)
    if match:
        return match.group(1)
    return None


def detect_odsek_marker(text: str) -> Optional[str]:
    """
    Check if text is an odsek marker - supports multiple formats:
    - Standalone: (1), (2)
    - With spaces: ( 1 ), ( 2 )
    - At start of text: (1) text content
    - With non-breaking spaces
    
    Args:
        text: Normalized text to check
        
    Returns:
        Odsek number (e.g., "1", "2") or None
    """
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


def detect_pismeno_marker(text: str) -> Optional[str]:
    """
    Check if text is a pismeno marker - supports multiple formats:
    - Standalone: a), b), aa), ab)
    - At start of text: a) content
    - With spaces: a ) content
    
    Args:
        text: Normalized text to check
        
    Returns:
        Pismeno marker (e.g., "a", "b", "aa", "ab") or None
    """
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


def detect_footnote_marker(text: str, hyperlink: Optional[str]) -> Optional[str]:
    """
    Detect footnote definition marker (not a reference in text).
    
    Patterns:
    - "1)", "2)", "37)" - simple numbered footnotes
    - "1a)", "1b)", "2aa)" - numbered with letter suffix
    - "37ab)", "37aba)" - complex nested footnotes
    
    Key: These markers have NO hyperlink (unlike references in text).
    
    Args:
        text: Text to check
        hyperlink: Hyperlink value (should be None/empty for definitions)
        
    Returns:
        Footnote ID ("1", "1a", "2aa") or None
    """
    # Must NOT have hyperlink (definitions don't have links, references do)
    if hyperlink:
        hyperlink_str = str(hyperlink)
        if hyperlink_str and 'poznamky.poznamka' in hyperlink_str:
            # This is a reference in text, not a definition
            return None
    
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

def table_to_rows_from_grid(table: Any) -> List[List[str]]:
    """
    Prevedie Docling tabuľku na zoznam riadkov so stringami priamo z data.grid.
    
    Rýchlejší a priamejší prístup ako export_to_dataframe.
    
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
                return getattr(cell, 'text', '').strip()
            elif isinstance(cell, dict):
                return cell.get("text", "").strip()
            else:
                return str(cell).strip()
        
        cells = sorted(row, key=get_col_idx)
        rows.append([get_cell_text(c) for c in cells])
    
    return rows


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
    
    Používa kombináciu stratégií:
    1. Parent referencia (ak unit_marker_text_elem má parent)
    2. Pozícia v texte (ak tabuľka má parent text v rozsahu jednotky)
    3. Heuristika podľa obsahu (ak unit_text obsahuje kľúčové slová)
    
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
    
    # Strategy 1 (PRIMARY): Get parent of unit marker and find all tables with same parent
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
    
    if unit_parent_ref:
        # Find all tables with the same parent
        for table_idx, table in enumerate(doc.tables):
            parent = getattr(table, 'parent', None)
            if not parent:
                continue
            
            # Get parent reference
            table_parent_ref = None
            if isinstance(parent, dict) and '$ref' in parent:
                table_parent_ref = str(parent['$ref'])
            elif hasattr(parent, 'cref'):
                table_parent_ref = str(parent.cref)
            elif hasattr(parent, 'get_ref'):
                table_parent_ref = str(parent.get_ref())
            else:
                table_parent_ref = str(parent)
            
            if table_parent_ref and table_parent_ref == unit_parent_ref:
                table_indices.append(table_idx)
                log_progress("DEBUG", f"Found table {table_idx} for unit (shared parent: {unit_parent_ref})")
    
    # Strategy 2: Check if table's parent text element is in unit region
    for table_idx, table in enumerate(doc.tables):
        if table_idx in table_indices:
            continue  # Already found
        
        parent = getattr(table, 'parent', None)
        if not parent:
            continue
        
        # Try to find parent text element in doc.texts
        parent_ref = None
        if isinstance(parent, dict) and '$ref' in parent:
            parent_ref = str(parent['$ref'])
        elif hasattr(parent, 'cref'):
            parent_ref = str(parent.cref)
        elif hasattr(parent, 'get_ref'):
            parent_ref = str(parent.get_ref())
        
        if parent_ref and '/texts/' in parent_ref:
            # Extract text index from reference like "#/texts/3659"
            try:
                text_idx_str = parent_ref.split('/texts/')[-1]
                text_idx = int(text_idx_str)
                # Check if this text index is in unit region
                if unit_start_idx <= text_idx < unit_end_idx:
                    table_indices.append(table_idx)
                    log_progress("DEBUG", f"Found table {table_idx} for unit (parent text idx: {text_idx})")
            except (ValueError, IndexError):
                pass
    
    # Strategy 3: Heuristic matching by content (if unit_text provided)
    if unit_text:
        unit_text_lower = unit_text.lower()
        # Common keywords that might indicate table references
        table_keywords = [
            "odpisová skupina", "odpisovanie", "odpis", "sadzba", 
            "tabuľka", "príloha", "uvedené v", "podľa tabuľky"
        ]
        
        has_table_keyword = any(keyword in unit_text_lower for keyword in table_keywords)
        
        if has_table_keyword:
            # Try to find tables by header content
            for table_idx, table in enumerate(doc.tables):
                if table_idx in table_indices:
                    continue
                
                # Check table header
                rows = table_to_rows_from_grid(table)
                if rows and len(rows) > 0:
                    first_row = rows[0]
                    header_text = " ".join(first_row).lower()
                    
                    # Match keywords from unit text with table header
                    if any(keyword in header_text for keyword in table_keywords):
                        # Additional check: if unit mentions specific table content
                        # and table header matches, it's likely related
                        table_indices.append(table_idx)
                        log_progress("DEBUG", f"Found table {table_idx} for unit (heuristic match: {header_text[:50]})")
    
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
# Sequential Document Reconstruction
# ============================================================================

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
            "content": "",
            "tables": [],
            "pictures": [],
            "annex_list": []
        },
        "footnotes": [],
        "references_index": {}
    }
    
    if not hasattr(doc, 'texts') or not doc.texts:
        log_progress("WARNING", "Document has no text elements")
        return structure
    
    total_elements = len(doc.texts)
    log_progress("INFO", f"Processing {total_elements:,} text elements...")
    
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
                _close_paragraph(current_paragraph, para_intro_texts, current_part, doc)
                
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
        para_num = detect_paragraph_marker(normalized_text)
        if para_num and not has_hyperlink:
            # Set end index for previous paragraph before closing
            if current_paragraph:
                current_paragraph["_end_idx"] = idx
            
            # Close previous paragraph, odsek, pismeno
            _close_pismeno(current_pismeno, pismeno_texts, current_odsek, doc, structure)
            _close_odsek(current_odsek, odsek_texts, para_intro_texts, current_paragraph, doc, structure)
            _close_paragraph(current_paragraph, para_intro_texts, current_part, doc)
            
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
            if normalized_text and not detect_odsek_marker(normalized_text) and not detect_pismeno_marker(normalized_text):
                part_texts.append(normalized_text)
            continue
        
        # 3. Check for pismeno marker FIRST (before odsek, since multi-letter pismenos like aa) could be confused)
        # Try standalone marker first
        pismeno_letter = detect_pismeno_marker(normalized_text)
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
        odsek_marker = detect_odsek_marker(normalized_text)
        odsek_extracted = None
        odsek_remaining_text = None
        
        # If not found, try extracting from text (e.g., "(1) content")
        if not odsek_marker:
            odsek_extracted = extract_marker_from_text(normalized_text, 'odsek')
            if odsek_extracted:
                odsek_marker, odsek_remaining_text = odsek_extracted
        
        if odsek_marker:
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
                "pismenos": []
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
    # Set end index for last paragraph before closing
    if current_paragraph:
        current_paragraph["_end_idx"] = total_elements
    
    _close_pismeno(current_pismeno, pismeno_texts, current_odsek, doc, structure)
    _close_odsek(current_odsek, odsek_texts, para_intro_texts, current_paragraph, doc, structure)
    _close_paragraph(current_paragraph, para_intro_texts, current_part, doc)
    
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
    
    # Sort tables by index for consistency
    if structure["annexes"]["tables"]:
        structure["annexes"]["tables"].sort(key=lambda x: x["index"])
    
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
    
    log_progress("INFO", f"Reconstruction complete: {actual_parts_count} parts, {paragraphs_count} paragraphs, {odseks_count} odseks, {pismenos_count} pismenos, {subitems_count} subitems, {structure['metadata']['total_annexes']} annexes, {footnotes_count} footnotes", elapsed)
    
    return structure


# ============================================================================
# Helper Functions for Closing Levels
# ============================================================================

def _close_annex(annex: Optional[Dict], texts: List[Dict], doc: DoclingDocument, structure: Dict, annex_start_idx: Optional[int] = None, annex_end_idx: Optional[int] = None, annex_marker_text_elem=None, all_annex_texts: List = None, seen_annex_tables: set = None) -> None:
    """
    Close current annex and add to unified annexes section.
    Collects all texts and tables without duplication.
    
    Args:
        annex: Current annex dictionary
        texts: List of text dictionaries for the annex
        doc: DoclingDocument (for tables/pictures)
        structure: Document structure to update
        annex_start_idx: Index where annex marker was found (for table detection)
        annex_end_idx: Index where next annex starts (for table detection)
        annex_marker_text_elem: The text element of the annex marker (for parent reference)
        all_annex_texts: List to collect all annex texts
        seen_annex_tables: Set to track tables and avoid duplicates
    """
    if not annex:
        return
    
    if all_annex_texts is None:
        all_annex_texts = []
    if seen_annex_tables is None:
        seen_annex_tables = set()
    
    # Combine text content for this annex
    content_lines = [t["text"] for t in texts if t.get("text")]
    annex["content"] = "\n".join(content_lines)
    
    # Add annex to annex_list for references
    structure["annexes"]["annex_list"].append({
        "id": annex["id"],
        "number": annex["number"],
        "title": annex["title"],
        "content": annex["content"]
    })
    
    # Add all texts to unified content
    all_annex_texts.extend(texts)
    
    # Extract tables and pictures from text references
    text_pos = len(structure["annexes"]["content"])  # Current position in unified content
    for t in texts:
        if t.get("table_idx") is not None:
            table_idx = t["table_idx"]
            if hasattr(doc, 'tables') and table_idx < len(doc.tables):
                if table_idx not in seen_annex_tables:
                    # Format table with full data
                    table_data = format_table_for_json(doc.tables[table_idx], doc, table_idx)
                    table_data["position_in_text"] = text_pos
                    structure["annexes"]["tables"].append(table_data)
                    seen_annex_tables.add(table_idx)
        if t.get("picture_idx") is not None:
            picture_idx = t["picture_idx"]
            if hasattr(doc, 'pictures') and picture_idx < len(doc.pictures):
                structure["annexes"]["pictures"].append({
                    "index": picture_idx,
                    "position_in_text": text_pos
                })
        text_pos += len(t.get("text", "")) + 1  # +1 for newline
    
    # Find additional tables that belong to annexes using document structure
    if annex_start_idx is not None and annex_end_idx is not None:
        additional_tables = find_tables_for_annex(doc, annex_start_idx, annex_end_idx, annex_marker_text_elem)
        for table_idx in additional_tables:
            if table_idx not in seen_annex_tables:
                # Format table with full data
                table_data = format_table_for_json(doc.tables[table_idx], doc, table_idx)
                table_data["position_in_text"] = len(structure["annexes"]["content"])  # Approximate position
                structure["annexes"]["tables"].append(table_data)
                seen_annex_tables.add(table_idx)
                log_progress("INFO", f"Found table {table_idx} for annexes section via structure analysis")
    
    # Update unified content
    if structure["annexes"]["content"]:
        structure["annexes"]["content"] += "\n\n"
    structure["annexes"]["content"] += annex["content"]


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
        prev_subitem = {
            "marker": f"{pismeno['_current_subitem']}.",
            "text": "\n".join([t["text"] for t in pismeno.get('_subitem_texts', []) if t.get("text")]),
            "tables": [],
            "pictures": [],
            "references_metadata": [],
            "footnotes_metadata": []
        }
        # Extract metadata from subitem texts
        text_pos = 0
        for t in pismeno.get('_subitem_texts', []):
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
        
        pismeno["subitems"].append(prev_subitem)
        # Clean up temporary fields
        del pismeno['_current_subitem']
        del pismeno['_subitem_texts']
    
    # Combine text content (preserve reference markers)
    # Only include text that's not part of subitems
    content_lines = [t["text"] for t in texts if t.get("text")]
    pismeno["text"] = "\n".join(content_lines)
    
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
                tables.append({
                    "index": table_idx,
                    "position_in_text": text_pos
                })
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
                tables.append({
                    "index": table_idx,
                    "position_in_text": text_pos
                })
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


def _close_paragraph(paragraph: Optional[Dict], intro_texts: List[Dict], part: Optional[Dict], doc: DoclingDocument) -> None:
    """Close current paragraph and add to part."""
    if not paragraph or not part:
        return
    
    # Combine intro text (preserve reference markers)
    intro_lines = [t["text"] for t in intro_texts if t.get("text")]
    paragraph["intro_text"] = "\n".join(intro_lines)
    
    # Find tables for paragraph using GPT-inspired approach
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
        else:
            # Use provided end_idx
            pass
        
        # Find tables for this paragraph
        table_indices = find_tables_for_paragraph_by_content(
            doc, para_start_idx, para_end_idx, para_marker_elem
        )
        
        if table_indices:
            # Store table indices in paragraph (will be processed in markdown generation)
            paragraph["tables"] = [{"index": idx} for idx in table_indices]
            log_progress("DEBUG", f"Found {len(table_indices)} tables for paragraph {paragraph.get('id')}: {table_indices}")
    
    # Clean up internal tracking fields before adding to structure
    paragraph.pop("_start_idx", None)
    paragraph.pop("_end_idx", None)
    paragraph.pop("_marker_text_elem", None)
    
    # Add to part
    part["paragraphs"].append(paragraph)


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

def save_json(structure: Dict[str, Any], output_path: str) -> None:
    """
    Save reconstructed structure to JSON file.
    
    Args:
        structure: Reconstructed document structure
        output_path: Path to output JSON file
    """
    start_time = time.time()
    log_progress("INFO", f"Saving reconstructed structure to JSON...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)
    
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
    
    # Process annexes (unified section)
    annexes_section = structure.get("annexes", {})
    if annexes_section and (annexes_section.get("content") or annexes_section.get("tables") or annexes_section.get("annex_list")):
        lines.append("# Prílohy\n\n")
        
        # Unified annex content
        if annexes_section.get("content"):
            lines.append(f"{annexes_section['content']}\n\n")
        
        # Tables in annexes section (only once, no duplicates)
        for table_ref in annexes_section.get("tables", []):
            table_idx = table_ref["index"]
            if hasattr(doc, 'tables') and table_idx < len(doc.tables):
                lines.append(format_table_as_markdown(doc.tables[table_idx], doc))
        
        # Pictures in annexes section
        for pic_ref in annexes_section.get("pictures", []):
            pic_idx = pic_ref["index"]
            if hasattr(doc, 'pictures') and pic_idx < len(doc.pictures):
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
# Main Function
# ============================================================================

def main():
    """
    Main function for sequential document reconstruction.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Sequential document reconstruction parser')
    parser.add_argument('input_json', type=str, help='Path to input Docling JSON file')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory (default: output)')
    
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
        
        # Reconstruct document
        structure = reconstruct_document(doc)
        
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
        log_progress("INFO", "=" * 70)
    
    finally:
        # Always close log file
        close_log_file()


if __name__ == '__main__':
    main()

