#!/usr/bin/env python3
"""
Sequential document reconstruction for Docling documents.

This module implements a single-pass sequential reconstruction of the entire
Docling document into a complete hierarchical JSON structure. This is Pass 1
of a 2-pass approach where Pass 2 will extract chunks from this pre-built structure.

Features:
- Single sequential pass through doc.texts (O(n))
- Complete hierarchical structure (parts → paragraphs → odseks → pismenos)
- Reference markers preserved in text for LLM visibility
- Detailed logging with progress indicators
- JSON and Markdown output
"""

import json
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from docling_core.types.doc import DoclingDocument


# ============================================================================
# Logging Functions
# ============================================================================

def log_progress(level: str, message: str, timing: Optional[float] = None) -> None:
    """
    Log progress message with timestamp and optional timing information.
    
    Args:
        level: Log level (INFO, DEBUG, WARNING, ERROR)
        message: Message to log
        timing: Optional timing in seconds to display
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    timing_str = f" [{timing:.2f}s]" if timing is not None else ""
    print(f"[{timestamp}] [{level}] {message}{timing_str}")


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
# Marker Detection
# ============================================================================

def is_part_marker(text: str) -> Optional[str]:
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


def is_paragraph_marker(text: str) -> Optional[str]:
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


def is_odsek_marker(text: str) -> Optional[str]:
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


def is_pismeno_marker(text: str) -> Optional[str]:
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


def is_subitem_marker(text: str) -> Optional[str]:
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


# ============================================================================
# Content Extraction Helpers
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
# Sequential Document Reconstruction
# ============================================================================

def reconstruct_document_sequential(doc: DoclingDocument) -> Dict[str, Any]:
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
            "total_text_elements": len(doc.texts) if hasattr(doc, 'texts') else 0,
            "total_parts": 0,
            "total_paragraphs": 0,
            "total_odseks": 0,
            "total_pismenos": 0
        },
        "parts": [],
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
        
        # Debug logging for first 100 elements
        if idx < 100:
            log_progress("DEBUG", f"Element {idx}: text='{normalized_text[:80]}' "
                         f"has_hyperlink={has_hyperlink} "
                         f"current_para={current_paragraph['id'] if current_paragraph else None} "
                         f"current_odsek={current_odsek['id'] if current_odsek else None}")
        
        # Get hyperlink
        hyperlink = getattr(text_element, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        has_hyperlink = bool(hyperlink_str)
        
        # Check for markers (in order of hierarchy: part → paragraph → odsek → pismeno)
        
        # 1. Check for part marker
        part_marker = is_part_marker(normalized_text)
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
        para_num = is_paragraph_marker(normalized_text)
        if para_num and not has_hyperlink:
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
                "odseks": []
            }
            current_odsek = None
            current_pismeno = None
            para_intro_texts = []
            odsek_texts = []
            pismeno_texts = []
            
            # Add to references index (will be updated when paragraph is added)
            # We'll update the path when we actually add it to the part
            
            log_progress("DEBUG", f"Found paragraph: § {para_num}")
            continue
        
        # If no paragraph yet, collect as part of part title
        if not current_paragraph:
            # This might be part of the part title (e.g., "ZÁKLADNÉ USTANOVENIA")
            # Collect it into part_texts
            if normalized_text and not is_odsek_marker(normalized_text) and not is_pismeno_marker(normalized_text):
                part_texts.append(normalized_text)
            continue
        
        # 3. Check for pismeno marker FIRST (before odsek, since multi-letter pismenos like aa) could be confused)
        # Try standalone marker first
        pismeno_letter = is_pismeno_marker(normalized_text)
        pismeno_extracted = None
        pismeno_remaining_text = None
        
        # If not found, try extracting from text (e.g., "a) content")
        if not pismeno_letter:
            pismeno_extracted = extract_marker_from_text(normalized_text, 'pismeno')
            if pismeno_extracted:
                pismeno_letter, pismeno_remaining_text = pismeno_extracted
        
        if pismeno_letter:
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
                    "text": "",  # Direct, no content wrapper
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
                        "footnotes": footnotes
                    })
                
                log_progress("DEBUG", f"Found pismeno: {pismeno_letter}) in odsek {odsek_num_for_pismeno}")
            continue
        
        # 4. Check for odsek marker (after pismeno check)
        # Try standalone marker first
        odsek_marker = is_odsek_marker(normalized_text)
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
                "marker": f"({odsek_marker})",  # Only numeric odseks now
                "text": "",  # Direct, no content wrapper
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
                    "footnotes": footnotes
                })
            
            # Add to references index (will be updated when odsek is added)
            # We'll update the path when we actually add it to the paragraph
            
            log_progress("DEBUG", f"Found odsek: ({odsek_marker}) in paragraph {para_num_for_odsek}")
            continue
        
        # 5. Check for subitem marker (within pismeno)
        if current_pismeno:
            # Try standalone subitem marker first
            subitem_marker = is_subitem_marker(normalized_text)
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
                if not hasattr(current_pismeno, '_current_subitem') or current_pismeno.get('_current_subitem') != subitem_marker:
                    # Close previous subitem if exists
                    if current_pismeno.get('_current_subitem'):
                        prev_subitem = {
                            "marker": f"{current_pismeno['_current_subitem']}.",
                            "text": "\n".join([t["text"] for t in current_pismeno.get('_subitem_texts', [])]),
                            "tables": [],
                            "pictures": [],
                            "references_metadata": [],
                            "footnotes_metadata": []
                        }
                        # Extract metadata from subitem texts
                        for t in current_pismeno.get('_subitem_texts', []):
                            if t.get("table_idx") is not None:
                                prev_subitem["tables"].append({"index": t["table_idx"], "position_in_text": 0})
                            if t.get("picture_idx") is not None:
                                prev_subitem["pictures"].append({"index": t["picture_idx"], "position_in_text": 0})
                            prev_subitem["references_metadata"].extend(t.get("references", []))
                            prev_subitem["footnotes_metadata"].extend(t.get("footnotes", []))
                        current_pismeno["subitems"].append(prev_subitem)
                    
                    # Start new subitem
                    current_pismeno['_current_subitem'] = subitem_marker
                    current_pismeno['_subitem_texts'] = []
                
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
        
        # Check for table/picture references
        table_idx = extract_table_reference(hyperlink_str) if has_hyperlink else None
        picture_idx = extract_picture_reference(hyperlink_str) if has_hyperlink else None
        
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
                    "references": references,
                    "footnotes": footnotes
                })
            else:
                # Add to pismeno
                pismeno_texts.append({
                    "text": content_text,  # Keep reference markers in text
                    "raw_text": raw_text,
                    "table_idx": table_idx,
                    "picture_idx": picture_idx,
                    "references": references,
                    "footnotes": footnotes
                })
        elif current_odsek:
            # Add to odsek
            odsek_texts.append({
                "text": content_text,  # Keep reference markers in text
                "raw_text": raw_text,
                "table_idx": table_idx,
                "picture_idx": picture_idx,
                "references": references,
                "footnotes": footnotes
            })
        elif current_paragraph:
            # Add to paragraph intro
            para_intro_texts.append({
                "text": content_text,  # Keep reference markers in text
                "raw_text": raw_text,
                "table_idx": table_idx,
                "picture_idx": picture_idx,
                "references": references,
                "footnotes": footnotes
            })
        else:
            # Part-level content (rare)
            part_texts.append({
                "text": content_text,
                "raw_text": raw_text
            })
    
    # Close any remaining open structures
    _close_pismeno(current_pismeno, pismeno_texts, current_odsek, doc, structure)
    _close_odsek(current_odsek, odsek_texts, para_intro_texts, current_paragraph, doc, structure)
    _close_paragraph(current_paragraph, para_intro_texts, current_part, doc)
    
    # Finalize part title_text if we collected additional text
    if current_part and part_texts:
        combined_title = " ".join(part_texts)
        current_part["title_text"] = combined_title
    
    if current_part and current_part["paragraphs"]:  # Only add if has paragraphs
        structure["parts"].append(current_part)
    
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
    
    log_progress("INFO", f"Reconstruction complete: {actual_parts_count} parts, {paragraphs_count} paragraphs, {odseks_count} odseks, {pismenos_count} pismenos", elapsed)
    
    return structure


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
    pismeno["text"] = "\n".join(content_lines)  # Direct, no wrapper
    
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
    
    # Add directly to pismeno (no content wrapper)
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
    content_lines = [t["text"] for t in texts if t["text"]]
    odsek["text"] = "\n".join(content_lines)  # Direct, no wrapper
    
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
    
    # Add directly to odsek (no content wrapper)
    odsek["tables"] = tables
    odsek["pictures"] = pictures
    odsek["references_metadata"] = all_references
    odsek["footnotes_metadata"] = all_footnotes
    
    # Check if odsek has any content
    has_text = bool(odsek["text"].strip())
    has_pismenos = bool(odsek.get("pismenos", []))
    has_tables = bool(tables)
    has_pictures = bool(pictures)
    
    # Only add odsek if it has content (similar to empty parts check)
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
    intro_lines = [t["text"] for t in intro_texts if t["text"]]
    paragraph["intro_text"] = "\n".join(intro_lines)
    
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

def save_reconstructed_json(structure: Dict[str, Any], output_path: str) -> None:
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


def format_table_as_markdown(table: Any, doc: DoclingDocument) -> str:
    """
    Format a table as markdown.
    
    Args:
        table: Table object from doc.tables
        doc: DoclingDocument
        
    Returns:
        Markdown-formatted table string
    """
    # Simple table formatting - can be enhanced later
    return f"\n[Table: {getattr(table, 'caption', 'No caption')}]\n\n"


def format_picture_as_markdown(picture: Any, doc: DoclingDocument) -> str:
    """
    Format a picture as markdown.
    
    Args:
        picture: Picture object from doc.pictures
        doc: DoclingDocument
        
    Returns:
        Markdown-formatted picture string
    """
    caption = getattr(picture, 'caption', 'No caption')
    return f"\n![{caption}]\n\n"


def save_reconstructed_markdown(structure: Dict[str, Any], output_path: str, doc: DoclingDocument) -> None:
    """
    Save reconstructed structure to Markdown file (readable like PDF).
    
    Args:
        structure: Reconstructed document structure
        output_path: Path to output Markdown file
        doc: DoclingDocument (for tables/pictures)
    """
    start_time = time.time()
    log_progress("INFO", f"Saving reconstructed structure to Markdown...")
    
    lines = []
    
    # Document header
    lines.append("# " + structure["document_name"] + "\n\n")
    lines.append(f"*Reconstructed from: {structure['metadata']['source_file']}*\n")
    lines.append(f"*Reconstruction time: {structure['metadata']['reconstruction_time']:.2f}s*\n")
    lines.append(f"*Parts: {structure['metadata']['total_parts']}, Paragraphs: {structure['metadata']['total_paragraphs']}, Odseks: {structure['metadata']['total_odseks']}, Pismenos: {structure['metadata']['total_pismenos']}*\n\n")
    lines.append("---\n\n")
    
    # Process each part
    for part in structure["parts"]:
        # Part title (no numbering)
        lines.append(f"# {part['title']}\n\n")
        # Display full title_text if it contains more than just the title
        if part.get("title_text"):
            title_text = part["title_text"].strip()
            # If title_text is different from title, display it (e.g., "PRVÁ ČASŤ ZÁKLADNÉ USTANOVENIA")
            if title_text != part["title"]:
                # Extract the additional text (everything after the title)
                additional_text = title_text.replace(part["title"], "").strip()
                if additional_text:
                    lines.append(f"{additional_text}\n\n")
        
        # Process paragraphs
        for para in part["paragraphs"]:
            # Paragraph title (no numbering)
            lines.append(f"## {para['title']}\n\n")
            
            # Paragraph intro text
            if para.get("intro_text"):
                lines.append(f"{para['intro_text']}\n\n")
            
            # Process odseks
            for odsek in para["odseks"]:
                # Odsek marker (no numbering)
                lines.append(f"### {odsek['marker']}\n\n")
                
                # Odsek content (no content wrapper)
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
                    # Pismeno marker with bold formatting (like screenshot)
                    # Format: **a)** text (on same line if text exists)
                    if pismeno.get("text"):
                        text = pismeno['text'].strip()
                        # Put marker and first line of text together
                        lines.append(f"    **{pismeno['marker']}** {text}\n\n")
                    else:
                        lines.append(f"    **{pismeno['marker']}**\n\n")
                    
                    # Subitems in pismeno (if any) - indented further
                    for subitem in pismeno.get("subitems", []):
                        lines.append(f"      - {subitem}\n")
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
    
    parser = argparse.ArgumentParser(description='Sequential document reconstruction - Pass 1')
    parser.add_argument('input_json', type=str, help='Path to input Docling JSON file')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory (default: output)')
    
    args = parser.parse_args()
    
    # Load document
    load_start = time.time()
    log_progress("INFO", "=" * 70)
    log_progress("INFO", "Sequential Document Reconstruction - Pass 1")
    log_progress("INFO", "=" * 70)
    log_progress("INFO", f"Loading document from {args.input_json}...")
    doc = load_docling_document(args.input_json)
    load_time = time.time() - load_start
    log_progress("INFO", f"Document loaded: {getattr(doc, 'name', 'Unknown')}", load_time)
    log_progress("INFO", f"  Texts: {len(doc.texts):,}, Tables: {len(doc.tables):,}, Pictures: {len(doc.pictures):,}")
    log_progress("INFO", "-" * 70)
    
    # Reconstruct document
    structure = reconstruct_document_sequential(doc)
    
    # Prepare output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    input_path = Path(args.input_json)
    base_name = input_path.stem
    json_output = output_dir / f"{base_name}_reconstructed.json"
    md_output = output_dir / f"{base_name}_reconstructed.md"
    
    # Save outputs
    log_progress("INFO", "-" * 70)
    save_reconstructed_json(structure, str(json_output))
    save_reconstructed_markdown(structure, str(md_output), doc)
    
    # Summary
    log_progress("INFO", "=" * 70)
    log_progress("INFO", "Reconstruction complete!")
    log_progress("INFO", f"Output files:")
    log_progress("INFO", f"  JSON: {json_output}")
    log_progress("INFO", f"  Markdown: {md_output}")
    log_progress("INFO", "=" * 70)


if __name__ == '__main__':
    main()

