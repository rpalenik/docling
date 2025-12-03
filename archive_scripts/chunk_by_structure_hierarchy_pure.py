#!/usr/bin/env python3
"""
Hierarchy-based document reconstruction for Docling documents - Pure Hierarchy Approach (Option 2).

This module implements a pure hierarchy-based reconstruction that uses ONLY the document's
hierarchical structure (parent/children relationships, depth, group labels) to classify elements.
NO marker detection is used - classification is purely based on hierarchy depth and structure.

Features:
- Full hierarchy traversal using parent/children references
- Depth-based classification (NO marker detection)
- Complete hierarchical structure (parts → paragraphs → odseks → pismenos → subitems)
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
from collections import Counter

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
    Load a DoclingDocument from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        DoclingDocument instance
    """
    return DoclingDocument.load_from_json(json_path)


# ============================================================================
# Reference Resolution and Hierarchy Traversal
# ============================================================================

def resolve_ref_item(doc: DoclingDocument, ref_item: Any) -> Optional[Any]:
    """
    Resolve a reference item to actual object.
    
    Handles cref paths like "#/texts/0", "#/groups/5", etc.
    
    Args:
        doc: DoclingDocument
        ref_item: Reference item with cref or resolve method
        
    Returns:
        Resolved object (text, group, table, picture) or None
    """
    # Try to get cref directly
    cref = getattr(ref_item, 'cref', None)
    if cref:
        # Parse cref path
        if cref.startswith('#/texts/'):
            try:
                idx = int(cref.split('/')[-1])
                if 0 <= idx < len(doc.texts):
                    return doc.texts[idx]
            except (ValueError, IndexError):
                pass
        elif cref.startswith('#/groups/'):
            try:
                idx = int(cref.split('/')[-1])
                if 0 <= idx < len(doc.groups):
                    return doc.groups[idx]
            except (ValueError, IndexError):
                pass
        elif cref.startswith('#/tables/'):
            try:
                idx = int(cref.split('/')[-1])
                if 0 <= idx < len(doc.tables):
                    return doc.tables[idx]
            except (ValueError, IndexError):
                pass
        elif cref.startswith('#/pictures/'):
            try:
                idx = int(cref.split('/')[-1])
                if 0 <= idx < len(doc.pictures):
                    return doc.pictures[idx]
            except (ValueError, IndexError):
                pass
        elif cref == '#/body':
            return doc.body
    
    # Fallback to resolve method
    if hasattr(ref_item, 'resolve'):
        try:
            return ref_item.resolve(doc=doc)
        except Exception:
            pass
    
    return None


def traverse_hierarchy_full(doc: DoclingDocument) -> List[Tuple[Any, Any, int, List[str]]]:
    """
    Traverse full document hierarchy using parent/children references.
    
    Uses iterative stack-based traversal starting from doc.body.
    Follows children recursively to capture all hierarchy levels.
    
    Args:
        doc: DoclingDocument
        
    Returns:
        List of (text_element, parent_element, depth, hierarchy_path) tuples
        in hierarchical order (not sequential doc.texts order)
    """
    results = []
    visited = set()
    stack = [(doc.body, None, 0, [])]  # (element, parent, depth, hierarchy_path)
    
    while stack:
        element, parent, depth, hierarchy_path = stack.pop()  # LIFO
        
        # Check if visited (avoid cycles)
        self_ref = getattr(element, 'self_ref', None)
        if self_ref:
            self_ref_str = str(self_ref)
            if self_ref_str in visited:
                continue
            visited.add(self_ref_str)
        
        # If text element, add to results
        if hasattr(element, 'text'):
            results.append((element, parent, depth, hierarchy_path.copy()))
        
        # Process children
        children = getattr(element, 'children', None)
        if children:
            # Process in reverse to maintain order with stack.pop()
            for ref_item in reversed(children):
                resolved = resolve_ref_item(doc, ref_item)
                if resolved:
                    # Get cref for path
                    cref = getattr(ref_item, 'cref', None)
                    if not cref and hasattr(ref_item, 'get_ref'):
                        cref = str(ref_item.get_ref())
                    new_path = hierarchy_path + [cref] if cref else hierarchy_path
                    stack.append((resolved, element, depth + 1, new_path))
    
    return results


# ============================================================================
# Depth Analysis and Classification
# ============================================================================

def analyze_hierarchy_depths(doc: DoclingDocument, hierarchy_texts: List[Tuple[Any, Any, int, List[str]]]) -> Dict[str, Any]:
    """
    Analyze depth distribution of all text elements in hierarchy.
    
    Args:
        doc: DoclingDocument
        hierarchy_texts: List of (text_element, parent, depth, hierarchy_path) tuples
        
    Returns:
        Dictionary with depth statistics and analysis
    """
    if not hierarchy_texts:
        return {
            "depths": [],
            "depth_counts": {},
            "min_depth": 0,
            "max_depth": 0,
            "avg_depth": 0,
            "depth_percentiles": {}
        }
    
    depths = [depth for _, _, depth, _ in hierarchy_texts]
    depth_counts = Counter(depths)
    
    min_depth = min(depths)
    max_depth = max(depths)
    avg_depth = sum(depths) / len(depths) if depths else 0
    
    # Calculate percentiles
    sorted_depths = sorted(depths)
    n = len(sorted_depths)
    percentiles = {
        25: sorted_depths[n // 4] if n > 0 else 0,
        50: sorted_depths[n // 2] if n > 0 else 0,
        75: sorted_depths[3 * n // 4] if n > 0 else 0,
        90: sorted_depths[9 * n // 10] if n > 0 else 0
    }
    
    return {
        "depths": depths,
        "depth_counts": dict(depth_counts),
        "min_depth": min_depth,
        "max_depth": max_depth,
        "avg_depth": avg_depth,
        "depth_percentiles": percentiles
    }


def analyze_group_structure(doc: DoclingDocument) -> Dict[str, Any]:
    """
    Analyze group labels and structure to understand hierarchy patterns.
    
    Args:
        doc: DoclingDocument
        
    Returns:
        Dictionary with group analysis (labels, nesting patterns, etc.)
    """
    group_labels = Counter()
    group_depths = []
    
    def traverse_groups(element: Any, depth: int = 0):
        """Recursively traverse groups to collect labels and depths."""
        if hasattr(element, 'label'):
            label = getattr(element, 'label', None)
            if label:
                group_labels[label] += 1
                group_depths.append((label, depth))
        
        children = getattr(element, 'children', None)
        if children:
            for ref_item in children:
                resolved = resolve_ref_item(doc, ref_item)
                if resolved:
                    traverse_groups(resolved, depth + 1)
    
    if hasattr(doc, 'body'):
        traverse_groups(doc.body, 0)
    
    return {
        "group_labels": dict(group_labels),
        "group_depths": group_depths,
        "most_common_labels": group_labels.most_common(10) if group_labels else []
    }


def determine_depth_ranges(depth_analysis: Dict[str, Any], group_analysis: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
    """
    Determine depth ranges using clustering and frequency analysis.
    
    Uses frequency-based clustering to identify natural hierarchy breaks:
    - Parts: very shallow depths (0-1)
    - Paragraphs: medium depths (2-4)
    - Odseks: deeper depths (4-6)
    - Pismenos: deepest depths (6+)
    
    Args:
        depth_analysis: Results from analyze_hierarchy_depths()
        group_analysis: Results from analyze_group_structure()
        
    Returns:
        Dictionary mapping level names to (min_depth, max_depth) tuples
    """
    min_depth = depth_analysis.get("min_depth", 0)
    max_depth = depth_analysis.get("max_depth", 0)
    depth_counts = depth_analysis.get("depth_counts", {})
    
    if max_depth == 0 or not depth_counts:
        # Fallback: assume standard structure
        return {
            "part": (0, 1),
            "paragraph": (2, 4),
            "odsek": (4, 6),
            "pismeno": (6, 8),
            "subitem": (8, 15)
        }
    
    # Find depth clusters (depths with high frequency)
    sorted_depths = sorted(depth_counts.items(), key=lambda x: x[1], reverse=True)
    top_depths = [d for d, _ in sorted_depths[:5]]  # Top 5 most common depths
    
    # Identify natural breaks using frequency-based clustering
    depth_span = max_depth - min_depth
    
    # Handle very shallow hierarchies (depth span <= 2)
    if depth_span <= 2:
        # For shallow hierarchies, use fixed ranges based on depth
        if max_depth <= 1:
            # Very flat - everything at same level
            return {
                "part": (min_depth, min_depth),
                "paragraph": (min_depth, min_depth),
                "odsek": (min_depth, min_depth),
                "pismeno": (min_depth, min_depth),
                "subitem": (min_depth + 1, max_depth + 2)
            }
        else:
            # Two levels - split between part/paragraph and odsek/pismeno
            return {
                "part": (min_depth, min_depth),
                "paragraph": (min_depth, min_depth),
                "odsek": (max_depth, max_depth),
                "pismeno": (max_depth, max_depth),
                "subitem": (max_depth + 1, max_depth + 2)
            }
    
    # For deeper hierarchies, use frequency-based clustering
    # Ensure we have valid ranges (min <= max)
    if len(top_depths) >= 3 and max_depth > min_depth:
        # Parts: typically at very shallow depths (0-1)
        part_max = min(top_depths[0] + 1, 2) if top_depths[0] <= 2 else 1
        
        # Paragraphs: typically at medium depths (2-4)
        para_min = max(top_depths[0] + 1, 2) if top_depths[0] < 3 else 2
        para_max = max(top_depths[1] + 1, para_min + 1) if len(top_depths) > 1 else max(para_min + 1, 4)
        
        # Odseks: typically deeper (4-6)
        odsek_min = para_max
        odsek_max = max(top_depths[2] + 1, odsek_min + 1) if len(top_depths) > 2 else max(odsek_min + 1, 6)
        
        # Pismenos: typically deepest (6+)
        pismeno_min = odsek_max
        pismeno_max = max_depth + 1
    else:
        # Fallback to percentile-based or fixed ranges
        percentiles = depth_analysis.get("depth_percentiles", {})
        p25 = percentiles.get(25, min_depth)
        p50 = percentiles.get(50, (min_depth + max_depth) // 2)
        p75 = percentiles.get(75, max_depth)
        
        # Ensure valid ranges
        part_max = min(max(p25, min_depth), 2)
        para_min = max(p25, 2)
        para_max = max(p50, para_min + 1)
        odsek_min = para_max
        odsek_max = max(p75, odsek_min + 1)
        pismeno_min = odsek_max
        pismeno_max = max(max_depth, pismeno_min + 1)
    
    # Ensure subitem range is valid
    subitem_min = pismeno_max
    subitem_max = max(max_depth + 1, subitem_min + 1)
    
    return {
        "part": (min_depth, part_max),
        "paragraph": (para_min, para_max),
        "odsek": (odsek_min, odsek_max),
        "pismeno": (pismeno_min, pismeno_max),
        "subitem": (subitem_min, subitem_max)
    }


def classify_by_depth(element: Any, parent: Any, depth: int, depth_ranges: Dict[str, Tuple[int, int]], 
                      group_analysis: Dict[str, Any], current_state: Optional[str] = None,
                      text: str = "") -> Optional[str]:
    """
    Classify element using depth + state + lightweight text hints.
    
    Uses state-based classification with lightweight text pattern hints (not full marker detection).
    
    Args:
        element: Text element to classify
        parent: Parent element
        depth: Current depth in hierarchy
        depth_ranges: Depth ranges for each level
        group_analysis: Group structure analysis
        current_state: Current hierarchy state ("part", "paragraph", "odsek", "pismeno", None)
        text: Normalized text content (for lightweight pattern hints)
        
    Returns:
        Classification type: "part", "paragraph", "odsek", "pismeno", "subitem", or None
    """
    # Lightweight text pattern hints (not full marker detection, just basic checks)
    text_lower = text.lower().strip()
    has_para_symbol = text.startswith('§')
    has_odsek_pattern = text.startswith('(') and text.endswith(')') and len(text) <= 5
    has_pismeno_pattern = text.endswith(')') and not text.startswith('(') and len(text) <= 4
    has_part_pattern = 'časť' in text_lower or 'ČASŤ' in text
    
    # State-based classification: prefer transitions to deeper levels
    if current_state == "part":
        # In a part, next level should be paragraph
        if depth_ranges["paragraph"][0] <= depth <= depth_ranges["paragraph"][1]:
            if has_para_symbol:
                return "paragraph"
            elif depth <= depth_ranges["paragraph"][1]:
                return "paragraph"
    elif current_state == "paragraph":
        # In a paragraph, next level should be odsek
        if depth_ranges["odsek"][0] <= depth <= depth_ranges["odsek"][1]:
            if has_odsek_pattern:
                return "odsek"
            elif depth <= depth_ranges["odsek"][1]:
                return "odsek"
    elif current_state == "odsek":
        # In an odsek, next level should be pismeno
        if depth_ranges["pismeno"][0] <= depth <= depth_ranges["pismeno"][1]:
            if has_pismeno_pattern:
                return "pismeno"
            elif depth <= depth_ranges["pismeno"][1]:
                return "pismeno"
    elif current_state == "pismeno":
        # In a pismeno, next level should be subitem
        if depth_ranges["subitem"][0] <= depth:
            return "subitem"
    
    # No current state or state transition - use depth ranges with text hints
    # Check from most specific to least specific
    if depth_ranges["pismeno"][0] <= depth <= depth_ranges["pismeno"][1]:
        if has_pismeno_pattern:
            return "pismeno"
        elif current_state in ["odsek", "paragraph"]:
            return "pismeno"
    
    if depth_ranges["odsek"][0] <= depth <= depth_ranges["odsek"][1]:
        if has_odsek_pattern:
            return "odsek"
        elif current_state == "paragraph":
            return "odsek"
    
    if depth_ranges["paragraph"][0] <= depth <= depth_ranges["paragraph"][1]:
        if has_para_symbol:
            return "paragraph"
        elif current_state == "part" or current_state is None:
            return "paragraph"
    
    if depth_ranges["part"][0] <= depth <= depth_ranges["part"][1]:
        # Only classify as part if:
        # 1. Very shallow depth (0-1)
        # 2. Has part pattern hint
        # 3. No current state (start of document)
        if depth <= 1 and (has_part_pattern or current_state is None):
            return "part"
    
    # Subitem is catch-all for very deep elements
    if depth_ranges["subitem"][0] <= depth:
        return "subitem"
    
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
# Pure Hierarchy-Based Document Reconstruction
# ============================================================================

def reconstruct_document_hierarchy_pure(doc: DoclingDocument) -> Dict[str, Any]:
    """
    Reconstruct entire document using pure hierarchy-based classification.
    
    NO marker detection - classification is purely based on hierarchy depth and structure.
    
    Args:
        doc: DoclingDocument to reconstruct
        
    Returns:
        Complete hierarchical structure dictionary
    """
    start_time = time.time()
    log_progress("INFO", "Starting pure hierarchy-based document reconstruction...")
    
    # Initialize structure
    structure = {
        "document_name": getattr(doc, 'name', 'Unknown'),
        "metadata": {
            "source_file": getattr(doc, 'name', 'Unknown'),
            "reconstruction_time": 0.0,
            "reconstruction_method": "hierarchy_pure",
            "total_text_elements": 0,
            "total_parts": 0,
            "total_paragraphs": 0,
            "total_odseks": 0,
            "total_pismenos": 0
        },
        "parts": [],
        "references_index": {}
    }
    
    if not hasattr(doc, 'body'):
        log_progress("WARNING", "Document has no body element")
        return structure
    
    # Step 1: Traverse full hierarchy
    log_progress("INFO", "Traversing document hierarchy (parent/children references)...")
    hierarchy_texts = traverse_hierarchy_full(doc)
    total_elements = len(hierarchy_texts)
    log_progress("INFO", f"Found {total_elements:,} text elements in hierarchy")
    structure["metadata"]["total_text_elements"] = total_elements
    
    if not hierarchy_texts:
        log_progress("WARNING", "No text elements found in hierarchy")
        return structure
    
    # Step 2: Analyze depth distribution
    log_progress("INFO", "Analyzing hierarchy depth distribution...")
    depth_analysis = analyze_hierarchy_depths(doc, hierarchy_texts)
    log_progress("INFO", f"Depth range: {depth_analysis['min_depth']} - {depth_analysis['max_depth']} (avg: {depth_analysis['avg_depth']:.1f})")
    
    # Step 3: Analyze group structure
    log_progress("INFO", "Analyzing group structure...")
    group_analysis = analyze_group_structure(doc)
    log_progress("INFO", f"Found {len(group_analysis['group_labels'])} unique group labels")
    
    # Step 4: Determine depth ranges for classification
    log_progress("INFO", "Determining depth ranges for classification...")
    depth_ranges = determine_depth_ranges(depth_analysis, group_analysis)
    log_progress("INFO", f"Depth ranges: {depth_ranges}")
    
    # Step 5: Process elements using depth-based classification
    log_progress("INFO", f"Processing {total_elements:,} text elements using depth-based classification...")
    
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
    
    # State tracking for classification
    current_state = None  # Track current hierarchy level
    
    # Process each text element in hierarchy order
    last_log_time = time.time()
    log_interval = 2.0  # Log every 2 seconds
    
    for idx, (text_element, parent, depth, hierarchy_path) in enumerate(hierarchy_texts):
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
        
        # Classify by depth with state tracking and lightweight text hints
        classification = classify_by_depth(
            text_element, parent, depth, depth_ranges, group_analysis,
            current_state=current_state, text=normalized_text
        )
        
        # Process based on classification
        if classification == "part":
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
                    log_progress("DEBUG", f"Skipping empty part: {current_part.get('title', 'UNNAMED')}")
            
            # Start new part
            parts_count += 1
            current_part = {
                "id": f"part-{parts_count}",
                "title": normalized_text[:100] if normalized_text else f"PART {parts_count}",  # Use first 100 chars as title
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
            current_state = "part"  # Update state
            log_progress("DEBUG", f"Found part (depth {depth}): {current_part['title']}")
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
        
        if classification == "paragraph":
            # Close previous paragraph, odsek, pismeno
            _close_pismeno(current_pismeno, pismeno_texts, current_odsek, doc, structure)
            _close_odsek(current_odsek, odsek_texts, para_intro_texts, current_paragraph, doc, structure)
            _close_paragraph(current_paragraph, para_intro_texts, current_part, doc)
            
            # Before starting new paragraph, finalize part title_text if we collected additional text
            if current_part and part_texts:
                combined_title = " ".join(part_texts)
                current_part["title_text"] = combined_title
                part_texts = []
            
            # Start new paragraph
            paragraphs_count += 1
            # Try to extract paragraph number from text, otherwise use count
            para_num = str(paragraphs_count)
            if normalized_text:
                para_match = re.search(r'§\s*(\d+[a-zA-Z]*)', normalized_text)
                if para_match:
                    para_num = para_match.group(1)
            
            current_paragraph = {
                "id": f"paragraf-{para_num}",
                "marker": f"§ {para_num}",
                "title": normalized_text if normalized_text else f"§ {para_num}",
                "intro_text": "",
                "odseks": []
            }
            current_odsek = None
            current_pismeno = None
            para_intro_texts = []
            odsek_texts = []
            pismeno_texts = []
            current_state = "paragraph"  # Update state
            log_progress("DEBUG", f"Found paragraph (depth {depth}): § {para_num}")
            continue
        
        # If no paragraph yet, collect as part of part title
        if not current_paragraph:
            if normalized_text and classification not in ["odsek", "pismeno", "subitem"]:
                part_texts.append(normalized_text)
            continue
        
        if classification == "odsek":
            # Close previous odsek and pismeno
            _close_pismeno(current_pismeno, pismeno_texts, current_odsek, doc, structure)
            _close_odsek(current_odsek, odsek_texts, para_intro_texts, current_paragraph, doc, structure)
            
            # Start new odsek
            odseks_count += 1
            para_num_for_odsek = current_paragraph["id"].replace("paragraf-", "")
            # Try to extract odsek number from text, otherwise use count
            odsek_num = str(odseks_count)
            if normalized_text:
                odsek_match = re.search(r'\((\d+)\)', normalized_text)
                if odsek_match:
                    odsek_num = odsek_match.group(1)
            
            odsek_id = f"odsek-{para_num_for_odsek}.{odsek_num}"
            current_odsek = {
                "id": odsek_id,
                "marker": f"({odsek_num})",
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
            current_state = "odsek"  # Update state
            log_progress("DEBUG", f"Found odsek (depth {depth}): ({odsek_num}) in paragraph {para_num_for_odsek}")
            continue
        
        if classification == "pismeno":
            # Close previous pismeno
            _close_pismeno(current_pismeno, pismeno_texts, current_odsek, doc, structure)
            
            # Start new pismeno
            if current_odsek:
                pismenos_count += 1
                para_num_for_pismeno = current_paragraph["id"].replace("paragraf-", "")
                odsek_num_for_pismeno = current_odsek["id"].split(".")[-1]
                # Try to extract pismeno letter from text, otherwise use count
                pismeno_letter = chr(ord('a') + pismenos_count - 1) if pismenos_count <= 26 else f"p{pismenos_count}"
                if normalized_text:
                    pismeno_match = re.search(r'([a-z]+)\)', normalized_text, re.IGNORECASE)
                    if pismeno_match:
                        pismeno_letter = pismeno_match.group(1).lower()
                
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
                current_state = "pismeno"  # Update state
                log_progress("DEBUG", f"Found pismeno (depth {depth}): {pismeno_letter}) in odsek {odsek_num_for_pismeno}")
            continue
        
        # Process content (not a classified marker)
        content_text = normalized_text
        
        # Extract references and footnotes (preserve markers in text)
        references = extract_references_from_text(content_text)
        footnotes = extract_footnotes_from_text(content_text)
        
        # Check for table/picture references
        table_idx = extract_table_reference(hyperlink_str) if has_hyperlink else None
        picture_idx = extract_picture_reference(hyperlink_str) if has_hyperlink else None
        
        # Add to appropriate level
        if current_pismeno:
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
        combined_title = " ".join([t.get("text", t) if isinstance(t, dict) else t for t in part_texts])
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


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_content_from_texts(doc: DoclingDocument, texts: List[Dict]) -> Dict[str, Any]:
    """Extract content (text, tables, pictures, references, footnotes) from text list."""
    content_lines = [t["text"] for t in texts if t.get("text")]
    text_content = "\n".join(content_lines)
    
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
    
    return {
        "text": text_content,
        "tables": tables,
        "pictures": pictures,
        "references": all_references,
        "footnotes": all_footnotes
    }


def _close_pismeno(pismeno: Optional[Dict], texts: List[Dict], odsek: Optional[Dict], doc: DoclingDocument, structure: Dict) -> None:
    """Close current pismeno and add to odsek, including subitem extraction."""
    if not pismeno or not odsek:
        return
    
    # Separate intro text from subitems
    intro_texts = []
    current_subitem = None
    subitem_texts = []
    subitems = []
    
    for t in texts:
        text = t.get("text", "").strip()
        normalized_text = text.replace('\xa0', ' ')
        
        # Check if this is a numbered sub-item marker (1., 2., 3., etc.)
        if re.match(r'^\d+\.$', normalized_text):
            # Save previous subitem if exists
            if current_subitem:
                subitem_content = _extract_content_from_texts(doc, subitem_texts)
                subitems.append({
                    "marker": current_subitem,
                    "text": subitem_content.get("text", ""),
                    "tables": subitem_content.get("tables", []),
                    "pictures": subitem_content.get("pictures", []),
                    "references_metadata": subitem_content.get("references", []),
                    "footnotes_metadata": subitem_content.get("footnotes", [])
                })
            
            # Start new subitem
            current_subitem = normalized_text
            subitem_texts = []
        else:
            # Add to appropriate collection
            if current_subitem:
                subitem_texts.append(t)
            else:
                intro_texts.append(t)
    
    # Save last subitem if exists
    if current_subitem:
        subitem_content = _extract_content_from_texts(doc, subitem_texts)
        subitems.append({
            "marker": current_subitem,
            "text": subitem_content.get("text", ""),
            "tables": subitem_content.get("tables", []),
            "pictures": subitem_content.get("pictures", []),
            "references_metadata": subitem_content.get("references", []),
            "footnotes_metadata": subitem_content.get("footnotes", [])
        })
    
    # Combine intro text content (preserve reference markers)
    intro_content = _extract_content_from_texts(doc, intro_texts)
    pismeno["text"] = intro_content.get("text", "")
    pismeno["subitems"] = subitems
    
    # Merge tables, pictures, references, footnotes from intro and subitems
    all_tables = intro_content.get("tables", [])
    all_pictures = intro_content.get("pictures", [])
    all_references = intro_content.get("references", [])
    all_footnotes = intro_content.get("footnotes", [])
    
    for subitem in subitems:
        all_tables.extend(subitem.get("tables", []))
        all_pictures.extend(subitem.get("pictures", []))
        all_references.extend(subitem.get("references_metadata", []))
        all_footnotes.extend(subitem.get("footnotes_metadata", []))
    
    pismeno["tables"] = all_tables
    pismeno["pictures"] = all_pictures
    pismeno["references_metadata"] = all_references
    pismeno["footnotes_metadata"] = all_footnotes
    
    # Add to odsek
    odsek["pismenos"].append(pismeno)


def _close_odsek(odsek: Optional[Dict], texts: List[Dict], para_intro_texts: List[Dict], paragraph: Optional[Dict], doc: DoclingDocument, structure: Dict) -> None:
    """Close current odsek and add to paragraph."""
    if not odsek or not paragraph:
        return
    
    # Combine text content (preserve reference markers)
    content = _extract_content_from_texts(doc, texts)
    odsek["text"] = content["text"]
    odsek["tables"] = content["tables"]
    odsek["pictures"] = content["pictures"]
    odsek["references_metadata"] = content["references"]
    odsek["footnotes_metadata"] = content["footnotes"]
    
    # Add to paragraph
    paragraph["odseks"].append(odsek)


def _close_paragraph(paragraph: Optional[Dict], intro_texts: List[Dict], part: Optional[Dict], doc: DoclingDocument) -> None:
    """Close current paragraph and add to part."""
    if not paragraph or not part:
        return
    
    # Combine intro text (preserve reference markers)
    intro_lines = [t["text"] for t in intro_texts if t.get("text")]
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
    lines.append(f"# {structure['document_name']}\n\n")
    lines.append(f"*Reconstructed from: {structure['metadata']['source_file']}*\n")
    lines.append(f"*Reconstruction time: {structure['metadata']['reconstruction_time']:.2f}s*\n")
    lines.append(f"*Reconstruction method: {structure['metadata']['reconstruction_method']}*\n")
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
                        if subitem.get("text"):
                            text = subitem['text'].strip()
                            lines.append(f"      {subitem['marker']} {text}\n\n")
                        else:
                            lines.append(f"      {subitem['marker']}\n\n")
                    
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
        f.writelines(lines)
    
    elapsed = time.time() - start_time
    log_progress("INFO", f"Markdown saved: {output_path}", elapsed)


# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Main function for pure hierarchy-based document reconstruction.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Hierarchy-based document reconstruction (Pure Hierarchy Approach) - Pass 1')
    parser.add_argument('input_json', type=str, help='Path to input Docling JSON file')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory (default: output)')
    
    args = parser.parse_args()
    
    # Load document
    load_start = time.time()
    log_progress("INFO", "=" * 70)
    log_progress("INFO", "Hierarchy-Based Document Reconstruction (Pure) - Pass 1")
    log_progress("INFO", "=" * 70)
    log_progress("INFO", f"Loading document from {args.input_json}...")
    doc = load_docling_document(args.input_json)
    load_time = time.time() - load_start
    log_progress("INFO", f"Document loaded: {getattr(doc, 'name', 'Unknown')}", load_time)
    log_progress("INFO", f"  Texts: {len(doc.texts):,}, Tables: {len(doc.tables):,}, Pictures: {len(doc.pictures):,}")
    log_progress("INFO", "-" * 70)
    
    # Reconstruct document
    structure = reconstruct_document_hierarchy_pure(doc)
    
    # Prepare output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    input_path = Path(args.input_json)
    base_name = input_path.stem
    json_output = output_dir / f"{base_name}_reconstructed_pure.json"
    md_output = output_dir / f"{base_name}_reconstructed_pure.md"
    
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
