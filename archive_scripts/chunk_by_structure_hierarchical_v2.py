#!/usr/bin/env python3
"""
Hierarchical structure-based chunking for Docling documents (Version 2).

This version implements a 2-pass approach:
- Pass 1: Build nested hierarchical structure with superior content and references as metadata
- Pass 2: Resolve references from Pass 1 results using chunk lookup

Features:
- Nested JSON structure preserving document hierarchy
- Superior content at all levels
- Tables, pictures, references placed within structure where they appear
- References resolved exactly as defined (paragraph vs odsek vs pismeno)

Performance Optimizations:
- Hybrid optimized traversal: Pre-built reference maps + iterative traversal (1.6x faster)
- Cached text elements: Pre-computed normalized text eliminates 1.8M+ string operations
- Single-pass post-processing: Combined operations reduce iteration overhead
"""

import json
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from docling_core.types.doc import DoclingDocument


# ============================================================================
# Progress Logging Functions
# ============================================================================

# Global cache for paragraph structures to avoid re-parsing
_paragraph_structure_cache: Dict[str, Dict[str, Any]] = {}

# Global cache for hierarchy traversal (most expensive operation)
_hierarchy_traversal_cache: Optional[List[Tuple[Any, Any, int, List[str]]]] = None
_hierarchy_traversal_doc_id: Optional[str] = None


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


def time_function(func_name: str = ""):
    """
    Decorator to measure function execution time.
    
    Args:
        func_name: Optional function name for logging
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            if func_name:
                log_progress("DEBUG", f"{func_name} completed", elapsed)
            return result
        return wrapper
    return decorator


def traverse_hierarchy_optimized(doc: DoclingDocument) -> List[Tuple[Any, Any, int, List[str]]]:
    """
    Optimized hierarchy traversal using hybrid approach:
    - Pre-built reference maps for O(1) lookups
    - Iterative traversal (no recursion overhead)
    - Minimized string operations
    
    Args:
        doc: DoclingDocument
        
    Returns:
        List of (text_element, parent_element, depth, hierarchy_path) tuples
    """
    # Build reference maps upfront (one-time cost, ~0.01s)
    ref_maps = build_reference_maps(doc)
    
    results = []
    visited = set()
    # Use list instead of deque for better performance with LIFO
    stack = [(doc.body, None, 0, [])]
    
    while stack:
        element, parent, depth, hierarchy_path = stack.pop()  # LIFO
        
        # Check if visited
        self_ref = getattr(element, 'self_ref', None)
        if self_ref:
            self_ref_str = str(self_ref) if isinstance(self_ref, str) else str(self_ref)
            if self_ref_str in visited:
                continue
            visited.add(self_ref_str)
        
        # If text element, add to results
        if hasattr(element, 'text'):
            results.append((element, parent, depth, hierarchy_path[:]))
        
        # Process children
        children = getattr(element, 'children', None)
        if children:
            # Process in reverse to maintain order with stack.pop()
            for ref_item in reversed(children):
                # Get cref directly
                cref = getattr(ref_item, 'cref', None)
                if not cref:
                    # Try get_ref as fallback
                    if hasattr(ref_item, 'get_ref'):
                        cref = str(ref_item.get_ref())
                    else:
                        continue
                
                # Direct map lookup (O(1) instead of string parsing)
                resolved = ref_maps['cref_to_obj'].get(cref)
                if not resolved:
                    # Fallback to resolve method
                    if hasattr(ref_item, 'resolve'):
                        try:
                            resolved = ref_item.resolve(doc=doc)
                        except Exception:
                            pass
                
                if resolved:
                    # Append to path
                    new_path = hierarchy_path + [cref]
                    stack.append((resolved, element, depth + 1, new_path))
    
    return results


def get_hierarchy_texts_cached(doc: DoclingDocument) -> List[Tuple[Any, Any, int, List[str]]]:
    """
    Get hierarchy texts with caching. This is the most expensive operation,
    so we cache it to avoid re-traversing the entire document multiple times.
    
    Now uses optimized hybrid traversal for better performance.
    
    Args:
        doc: DoclingDocument
        
    Returns:
        List of (text_element, parent_element, depth, hierarchy_path) tuples
    """
    global _hierarchy_traversal_cache, _hierarchy_traversal_doc_id
    
    # Use document name as cache key to detect document changes
    doc_id = getattr(doc, 'name', None) or str(id(doc))
    
    # If cache exists and is for the same document, return cached result
    if _hierarchy_traversal_cache is not None and _hierarchy_traversal_doc_id == doc_id:
        return _hierarchy_traversal_cache
    
    # Cache miss - perform optimized traversal
    start_time = time.time()
    log_progress("DEBUG", "Traversing document hierarchy with optimized method (this will be cached)...")
    _hierarchy_traversal_cache = traverse_hierarchy_optimized(doc)
    _hierarchy_traversal_doc_id = doc_id
    elapsed = time.time() - start_time
    log_progress("DEBUG", f"Hierarchy traversal completed (cached for future use)", elapsed)
    
    return _hierarchy_traversal_cache


def clear_hierarchy_cache():
    """Clear hierarchy traversal cache (useful when switching documents)."""
    global _hierarchy_traversal_cache, _hierarchy_traversal_doc_id
    _hierarchy_traversal_cache = None
    _hierarchy_traversal_doc_id = None
    # Also clear cached elements
    clear_cached_elements()


def load_docling_document(json_path: str) -> DoclingDocument:
    """
    Load a DoclingDocument from a JSON file.
    
    Args:
        json_path: Path to JSON file containing DoclingDocument
        
    Returns:
        DoclingDocument object
    """
    return DoclingDocument.load_from_json(json_path)


def resolve_ref_item(doc: DoclingDocument, ref_item: Any) -> Any:
    """
    Resolve a RefItem to its actual object.
    
    Args:
        doc: DoclingDocument
        ref_item: RefItem to resolve (has .cref attribute or .resolve() method)
        
    Returns:
        The actual object (text, group, table, picture, or body)
    """
    # Try using resolve() method first (preferred)
    if hasattr(ref_item, 'resolve'):
        try:
            return ref_item.resolve(doc=doc)
        except Exception:
            pass
    
    # Fallback: use cref attribute
    if not hasattr(ref_item, 'cref'):
        return None
    
    ref_path = ref_item.cref
    if not ref_path:
        return None
    
    # Handle different reference types
    if ref_path == "#/body":
        return doc.body
    elif ref_path.startswith("#/texts/"):
        idx = int(ref_path.split("/")[-1])
        if idx < len(doc.texts):
            return doc.texts[idx]
    elif ref_path.startswith("#/groups/"):
        idx = int(ref_path.split("/")[-1])
        if idx < len(doc.groups):
            return doc.groups[idx]
    elif ref_path.startswith("#/tables/"):
        idx = int(ref_path.split("/")[-1])
        if idx < len(doc.tables):
            return doc.tables[idx]
    elif ref_path.startswith("#/pictures/"):
        idx = int(ref_path.split("/")[-1])
        if idx < len(doc.pictures):
            return doc.pictures[idx]
    
    return None


def build_reference_maps(doc: DoclingDocument) -> Dict[str, Any]:
    """
    Build reference maps upfront for O(1) lookups.
    Returns dict mapping cref paths to actual objects.
    
    This optimization eliminates repeated string parsing during traversal.
    
    Args:
        doc: DoclingDocument
        
    Returns:
        Dictionary with maps:
        - cref_to_obj: Maps cref paths (e.g., "#/texts/0") to objects
        - self_ref_to_obj: Maps self_ref strings to objects
        - texts_by_idx, groups_by_idx, tables_by_idx, pictures_by_idx: Index-based maps
    """
    maps = {
        'cref_to_obj': {},
        'self_ref_to_obj': {},
        'texts_by_idx': {},
        'groups_by_idx': {},
        'tables_by_idx': {},
        'pictures_by_idx': {}
    }
    
    # Map body
    maps['cref_to_obj']["#/body"] = doc.body
    if hasattr(doc.body, 'self_ref'):
        maps['self_ref_to_obj'][str(doc.body.self_ref)] = doc.body
    
    # Map texts
    for idx, text in enumerate(doc.texts):
        cref = f"#/texts/{idx}"
        maps['cref_to_obj'][cref] = text
        maps['texts_by_idx'][idx] = text
        if hasattr(text, 'self_ref'):
            maps['self_ref_to_obj'][str(text.self_ref)] = text
    
    # Map groups
    for idx, group in enumerate(doc.groups):
        cref = f"#/groups/{idx}"
        maps['cref_to_obj'][cref] = group
        maps['groups_by_idx'][idx] = group
        if hasattr(group, 'self_ref'):
            maps['self_ref_to_obj'][str(group.self_ref)] = group
    
    # Map tables
    for idx, table in enumerate(doc.tables):
        cref = f"#/tables/{idx}"
        maps['cref_to_obj'][cref] = table
        maps['tables_by_idx'][idx] = table
        if hasattr(table, 'self_ref'):
            maps['self_ref_to_obj'][str(table.self_ref)] = table
    
    # Map pictures
    for idx, picture in enumerate(doc.pictures):
        cref = f"#/pictures/{idx}"
        maps['cref_to_obj'][cref] = picture
        maps['pictures_by_idx'][idx] = picture
        if hasattr(picture, 'self_ref'):
            maps['self_ref_to_obj'][str(picture.self_ref)] = picture
    
    return maps


def traverse_hierarchy_recursive(doc: DoclingDocument, element: Any, visited: Optional[set] = None, 
                                 depth: int = 0, parent: Any = None, hierarchy_path: List[str] = None) -> List[Tuple[Any, Any, int, List[str]]]:
    """
    Recursively traverse hierarchy starting from an element.
    
    Args:
        doc: DoclingDocument
        element: Starting element (usually doc.body)
        visited: Set of visited element references to avoid cycles
        depth: Current depth in hierarchy
        parent: Parent element
        hierarchy_path: Path of references to this element
        
    Returns:
        List of (text_element, parent_element, depth, hierarchy_path) tuples
    """
    if visited is None:
        visited = set()
    if hierarchy_path is None:
        hierarchy_path = []
    
    results = []
    
    # Get self_ref to track visited elements
    self_ref = getattr(element, 'self_ref', None)
    if self_ref:
        self_ref_str = str(self_ref)
        if self_ref_str in visited:
            return results  # Avoid cycles
        visited.add(self_ref_str)
    
    # If this is a text element, add it to results
    if hasattr(element, 'text'):
        results.append((element, parent, depth, hierarchy_path.copy()))
    
    # Traverse children
    if hasattr(element, 'children') and element.children:
        for ref_item in element.children:
            # Get ref path from cref or resolve
            ref_path = None
            if hasattr(ref_item, 'cref'):
                ref_path = ref_item.cref
            elif hasattr(ref_item, 'get_ref'):
                ref_path = str(ref_item.get_ref())
            
            if not ref_path:
                continue
            
            resolved = resolve_ref_item(doc, ref_item)
            
            if resolved:
                # Build new path
                new_path = hierarchy_path + [str(ref_path)]
                
                # Recursively traverse children
                child_results = traverse_hierarchy_recursive(
                    doc, resolved, visited, depth + 1, element, new_path
                )
                results.extend(child_results)
    
    return results


# ============================================================================
# Post-Processing Optimization: Cached Text Elements
# ============================================================================

class CachedTextElement:
    """
    Cached text element with pre-computed normalized data.
    Eliminates repeated string operations during post-processing.
    """
    def __init__(self, text_element, parent, depth, path):
        self.element = text_element
        self.parent = parent
        self.depth = depth
        self.path = path
        
        # Pre-compute once - these are the expensive operations
        raw_text = getattr(text_element, 'text', '')
        self.text = raw_text.strip()
        self.normalized = self.text.replace('\xa0', ' ')
        self.normalized_lower = self.normalized.lower()
        
        # Pre-compute hyperlink info
        hyperlink = getattr(text_element, 'hyperlink', '')
        self.has_hyperlink = bool(hyperlink)
        self.hyperlink_str = str(hyperlink) if hyperlink else ''
        
        # Pre-compute flags for common checks
        self.starts_with_para = self.normalized.startswith('§ ')
        self.is_para_marker = self.starts_with_para and not self.has_hyperlink


# Global cache for cached elements (post-processing optimization)
_cached_elements_cache: Optional[List[CachedTextElement]] = None
_cached_elements_doc_id: Optional[str] = None


def get_cached_elements(doc: DoclingDocument) -> List[CachedTextElement]:
    """
    Get cached text elements with pre-computed normalized data.
    This eliminates repeated string operations during post-processing.
    
    Args:
        doc: DoclingDocument
        
    Returns:
        List of CachedTextElement objects
    """
    global _cached_elements_cache, _cached_elements_doc_id
    
    doc_id = getattr(doc, 'name', None) or str(id(doc))
    
    # Return cached if available
    if _cached_elements_cache is not None and _cached_elements_doc_id == doc_id:
        return _cached_elements_cache
    
    # Build cached elements from hierarchy texts
    hierarchy_texts = get_hierarchy_texts_cached(doc)
    _cached_elements_cache = [CachedTextElement(te, parent, depth, path) 
                               for te, parent, depth, path in hierarchy_texts]
    _cached_elements_doc_id = doc_id
    
    return _cached_elements_cache


def clear_cached_elements():
    """Clear cached elements cache (useful when switching documents)."""
    global _cached_elements_cache, _cached_elements_doc_id
    _cached_elements_cache = None
    _cached_elements_doc_id = None


def find_text_in_hierarchy(doc: DoclingDocument, text_pattern: str, no_hyperlink: bool = False, exact_match: bool = True) -> Optional[Tuple[Any, Any, int, List[str]]]:
    """
    Find text element in hierarchy by content.
    
    Optimized to use cached elements when available for faster string operations.
    
    Args:
        doc: DoclingDocument
        text_pattern: Text pattern to match
        no_hyperlink: If True, only match elements without hyperlinks
        exact_match: If True, require exact match; if False, match if text starts with pattern
        
    Returns:
        Tuple of (text_element, parent_element, depth, hierarchy_path) or None
    """
    if not hasattr(doc, 'body') or not doc.body:
        return None
    
    # Use cached elements for faster processing
    cached_elements = get_cached_elements(doc)
    normalized_pattern = text_pattern.replace('\xa0', ' ')
    
    for cached in cached_elements:
        # Use pre-computed normalized text
        if exact_match:
            matches = cached.normalized == normalized_pattern
        else:
            matches = cached.normalized.startswith(normalized_pattern)
        
        if matches:
            if no_hyperlink and cached.has_hyperlink:
                continue
            if not no_hyperlink or not cached.has_hyperlink:
                return (cached.element, cached.parent, cached.depth, cached.path)
    
    return None


def build_text_hierarchy_map(doc: DoclingDocument) -> Tuple[List[Any], Dict[str, int]]:
    """
    Build ordered list of all text elements as they appear in hierarchy.
    
    Args:
        doc: DoclingDocument
        
    Returns:
        Tuple of (ordered_text_list, text_to_index_map)
        - ordered_text_list: List of text elements in hierarchy order
        - text_to_index_map: Dict mapping text element self_ref to its index in doc.texts
    """
    if not hasattr(doc, 'body') or not doc.body:
        return [], {}
    
    # Build mapping from text element self_ref to doc.texts index
    text_to_index = {}
    for idx, text_item in enumerate(doc.texts):
        self_ref = getattr(text_item, 'self_ref', None)
        if self_ref:
            self_ref_str = str(self_ref)
            text_to_index[self_ref_str] = idx
    
    # Traverse hierarchy starting from body
    hierarchy_texts = get_hierarchy_texts_cached(doc)
    
    # Extract only text elements and maintain order
    ordered_texts = []
    seen_refs = set()
    for text_element, parent, depth, path in hierarchy_texts:
        self_ref = getattr(text_element, 'self_ref', None)
        if self_ref:
            self_ref_str = str(self_ref)
            if self_ref_str in text_to_index and self_ref_str not in seen_refs:
                ordered_texts.append(text_element)
                seen_refs.add(self_ref_str)
    
    return ordered_texts, text_to_index


def find_chunk_boundaries_hierarchical(doc: DoclingDocument, level: str, identifier: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Find start and end indices for a chunk using pure hierarchical traversal.
    
    Args:
        doc: DoclingDocument
        level: Chunking level ('paragraph', 'odsek', 'pismeno')
        identifier: Section identifier (e.g., '50' for paragraph, '50.1' for odsek, '50.1.a' for pismeno)
        
    Returns:
        Tuple of (start_index, end_index) in doc.texts, or (None, None) if not found
    """
    if not hasattr(doc, 'texts') or not doc.texts:
        return None, None
    
    if not hasattr(doc, 'body') or not doc.body:
        return None, None
    
    # Build mapping from text element self_ref to doc.texts index
    text_to_index = {}
    for idx, text_item in enumerate(doc.texts):
        self_ref = getattr(text_item, 'self_ref', None)
        if self_ref:
            text_to_index[str(self_ref)] = idx
    
    # Parse identifier
    if level == 'paragraph':
        para_num = identifier
        odsek = None
        pismeno = None
    elif level == 'odsek':
        parts = identifier.split('.')
        para_num = parts[0]
        odsek = parts[1] if len(parts) > 1 else None
        pismeno = None
    elif level == 'pismeno':
        parts = identifier.split('.')
        para_num = parts[0]
        odsek = parts[1] if len(parts) > 1 else None
        pismeno = parts[2] if len(parts) > 2 else None
    else:
        return None, None
    
    # Find paragraph marker in hierarchy
    # Paragraph markers may include titles (e.g., "§ 1 Predmet úpravy")
    para_marker = f'§ {para_num}'
    para_result = find_text_in_hierarchy(doc, para_marker, no_hyperlink=True, exact_match=False)
    
    if para_result is None:
        return None, None
    
    para_text, para_parent, para_depth, para_path = para_result
    
    # Find start text based on level
    start_text = None
    odsek_found_sequentially = False
    
    if pismeno:
        # Find pismeno marker in hierarchy - must be descendant of paragraph
        pismeno_marker = f'{pismeno})'
        # Get all texts in hierarchy order from paragraph container
        para_container = find_section_container_in_hierarchy(doc, para_text, para_path)
        if para_container:
            descendants = get_text_descendants(doc, para_container)
            for text in descendants:
                text_content = getattr(text, 'text', '').strip()
                if text_content == pismeno_marker:
                    start_text = text
                    break
        if start_text is None:
            return None, None
    elif odsek:
        # OPTIMIZED: For odsek, use simple sequential scan instead of expensive hierarchy traversal
        # This is much faster - O(n) scan vs O(n*m) hierarchy traversal
        odsek_marker = f'({odsek})'
        
        # First get paragraph boundaries (fast, already cached)
        para_start_idx, para_end_idx = find_chunk_boundaries_hierarchical(doc, 'paragraph', para_num)
        if para_start_idx is None or para_end_idx is None:
            return None, None
        
        # Scan for odsek marker sequentially (much faster than hierarchy traversal)
        start_idx = None
        end_idx = None
        
        # Define pismeno markers to avoid false positives
        pismeno_markers = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 
                          'k)', 'l)', 'm)', 'n)', 'o)', 'p)', 'q)', 'r)', 's)', 't)', 
                          'u)', 'v)', 'w)', 'x)', 'y)', 'z)']
        
        for i in range(para_start_idx, min(para_end_idx, para_start_idx + 2000)):  # Safety limit
            if i >= len(doc.texts):
                break
                
            text = getattr(doc.texts[i], 'text', '').strip()
            normalized = text.replace('\xa0', ' ')
            hyperlink = getattr(doc.texts[i], 'hyperlink', '')
            hyperlink_str = str(hyperlink) if hyperlink else ''
            
            # Found odsek start
            if normalized == odsek_marker:
                start_idx = i
                # Continue to find end (next odsek or paragraph)
                for j in range(i + 1, min(para_end_idx, i + 1000)):  # Safety limit
                    if j >= len(doc.texts):
                        break
                    next_text = getattr(doc.texts[j], 'text', '').strip()
                    next_normalized = next_text.replace('\xa0', ' ')
                    next_hyperlink = getattr(doc.texts[j], 'hyperlink', '')
                    next_hyperlink_str = str(next_hyperlink) if next_hyperlink else ''
                    
                    # Check for next numeric odsek: (1), (2), etc.
                    if next_normalized.startswith('(') and next_normalized[1:].rstrip(')').isdigit():
                        next_num = next_normalized[1:].rstrip(')')
                        if next_num != odsek:
                            end_idx = j
                            break
                    
                    # Check for next letter-based odsek: aa), ab), etc. (but not single letter pismenos)
                    if (next_normalized.endswith(')') and not next_normalized.startswith('(') and 
                        len(next_normalized) > 2 and next_normalized[:-1].isalpha() and 
                        next_normalized not in pismeno_markers):
                        next_odsek = next_normalized[:-1]
                        if next_odsek != odsek:
                            end_idx = j
                            break
                    
                    # Check for next paragraph
                    if next_normalized.startswith('§ ') and not next_hyperlink_str:
                        match = re.match(r'§\s+(\d+[a-zA-Z]*)', next_normalized)
                        if match and match.group(1) != para_num:
                            end_idx = j
                            break
                
                # If no end found, use paragraph end
                if end_idx is None:
                    end_idx = para_end_idx
                break
        
        # Return directly if found (skip expensive hierarchy traversal)
        if start_idx is not None and end_idx is not None:
            odsek_found_sequentially = True
            return start_idx, end_idx
        
        # Fallback to old method if sequential scan failed
        para_container = find_section_container_in_hierarchy(doc, para_text, para_path)
        if para_container:
            descendants = get_text_descendants(doc, para_container)
            for text in descendants:
                text_content = getattr(text, 'text', '').strip()
                if text_content == odsek_marker:
                    start_text = text
                    break
        if start_text is None:
            return None, None
    else:
        # Use paragraph marker as start
        start_text = para_text
    
    # Get start index
    start_self_ref = getattr(start_text, 'self_ref', None)
    if not start_self_ref:
        return None, None
    start_idx = text_to_index.get(str(start_self_ref))
    if start_idx is None:
        return None, None
    
    # Find container for the section
    hierarchy_texts = get_hierarchy_texts_cached(doc)
    start_path_for_container = None
    for te, parent, depth, path in hierarchy_texts:
        if te == start_text:
            start_path_for_container = path
            break
    
    if not start_path_for_container:
        return None, None
    
    section_container = find_section_container_in_hierarchy(doc, start_text, start_path_for_container)
    if not section_container:
        # Fallback: use paragraph container
        section_container = find_section_container_in_hierarchy(doc, para_text, para_path)
    
    # Define end condition based on level
    def end_condition(text_element: Any, parent: Any, depth: int, path: List[str]) -> bool:
        """Return True when we should stop collecting texts."""
        text_content = getattr(text_element, 'text', '').strip()
        # Normalize spaces (handle both regular and non-breaking spaces)
        normalized_content = text_content.replace('\xa0', ' ')
        hyperlink = getattr(text_element, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        
        if pismeno:
            # End at next pismeno, odsek, or paragraph
            pismeno_markers_list = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 
                                   'k)', 'l)', 'm)', 'n)', 'o)', 'p)', 'q)', 'r)', 's)', 't)', 
                                   'u)', 'v)', 'w)', 'x)', 'y)', 'z)']
            if normalized_content in pismeno_markers_list and normalized_content != f'{pismeno})':
                return True
            # Odsek: (číslo) alebo písmeno/písmená)
            if (normalized_content.startswith('(') and normalized_content[1:].rstrip(')').isdigit()) or \
               (normalized_content.endswith(')') and not normalized_content.startswith('(') and 
                len(normalized_content) > 2 and normalized_content[:-1].isalpha() and 
                normalized_content not in pismeno_markers_list):
                return True
            if normalized_content.startswith('§ ') and not hyperlink_str:
                match = re.match(r'§\s+(\d+[a-zA-Z]*)', normalized_content)
                if match and match.group(1) != para_num:
                    return True
        elif odsek:
            # End at next odsek or paragraph
            # Odsek môže byť: (číslo) alebo písmeno/písmená)
            pismeno_markers_list = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 
                                   'k)', 'l)', 'm)', 'n)', 'o)', 'p)', 'q)', 'r)', 's)', 't)', 
                                   'u)', 'v)', 'w)', 'x)', 'y)', 'z)']
            if normalized_content.startswith('(') and normalized_content[1:].rstrip(')').isdigit():
                next_num = normalized_content[1:].rstrip(')')
                if next_num != odsek:
                    return True
            elif (normalized_content.endswith(')') and not normalized_content.startswith('(') and 
                  len(normalized_content) > 2 and normalized_content[:-1].isalpha() and 
                  normalized_content not in pismeno_markers_list):
                # Písmenový odsek
                next_odsek = normalized_content[:-1]
                if next_odsek != odsek:
                    return True
            if normalized_content.startswith('§ ') and not hyperlink_str:
                match = re.match(r'§\s+(\d+[a-zA-Z]*)', normalized_content)
                if match and match.group(1) != para_num:
                    return True
        else:
            # End at next paragraph
            if normalized_content.startswith('§ ') and not hyperlink_str:
                match = re.match(r'§\s+(\d+[a-zA-Z]*)', normalized_content)
                if match and match.group(1) != para_num:
                    return True
        
        return False
    
    # Collect all texts from the container
    collected_texts = collect_texts_from_hierarchy_container(
        doc, section_container, start_text, end_condition
    )
    
    if not collected_texts:
        # Fallback: just use start_text
        collected_texts = [start_text]
    
    # Find end text (last text in collected list, or next boundary)
    end_text = None
    if len(collected_texts) > 0:
        # The last collected text is our end boundary
        # But we want the text AFTER the last one as the end index
        # So we need to find the next text in hierarchy
        last_text = collected_texts[-1]
        
        # Find next text in hierarchy order
        ordered_texts, _ = build_text_hierarchy_map(doc)
        last_self_ref = getattr(last_text, 'self_ref', None)
        if last_self_ref:
            last_self_ref_str = str(last_self_ref)
            last_pos = -1
            for i, text in enumerate(ordered_texts):
                text_self_ref = getattr(text, 'self_ref', None)
                if text_self_ref and str(text_self_ref) == last_self_ref_str:
                    last_pos = i
                    break
            
            # Find next text that meets end condition
            if last_pos >= 0:
                for i in range(last_pos + 1, len(ordered_texts)):
                    text = ordered_texts[i]
                    # Get hierarchy info
                    text_parent = None
                    text_depth = None
                    text_path = None
                    for te, parent, depth, path in hierarchy_texts:
                        if te == text:
                            text_parent = parent
                            text_depth = depth
                            text_path = path
                            break
                    
                    if end_condition(text, text_parent, text_depth or 0, text_path or []):
                        end_text = text
                        break
    
    # Map to doc.texts indices
    if end_text:
        end_self_ref = getattr(end_text, 'self_ref', None)
        if end_self_ref:
            end_idx = text_to_index.get(str(end_self_ref))
        else:
            end_idx = None
        if end_idx is None:
            # Use last collected text as fallback
            if collected_texts:
                last_text = collected_texts[-1]
                last_self_ref = getattr(last_text, 'self_ref', None)
                if last_self_ref:
                    end_idx = text_to_index.get(str(last_self_ref))
                    if end_idx is not None:
                        end_idx += 1  # Include the last text
            if end_idx is None:
                # Safety limit
                if pismeno:
                    end_idx = min(start_idx + 200, len(doc.texts))
                elif odsek:
                    end_idx = min(start_idx + 300, len(doc.texts))
                else:
                    end_idx = min(start_idx + 500, len(doc.texts))
    else:
        # Use last collected text
        if collected_texts:
            last_text = collected_texts[-1]
            last_self_ref = getattr(last_text, 'self_ref', None)
            if last_self_ref:
                end_idx = text_to_index.get(str(last_self_ref))
                if end_idx is not None:
                    end_idx += 1  # Include the last text
        if end_idx is None:
            # Safety limit
            if pismeno:
                end_idx = min(start_idx + 200, len(doc.texts))
            elif odsek:
                end_idx = min(start_idx + 300, len(doc.texts))
            else:
                end_idx = min(start_idx + 500, len(doc.texts))
    
    return start_idx, end_idx


def get_text_descendants(doc: DoclingDocument, text_element: Any) -> List[Any]:
    """
    Get all descendant text elements of a given text element by traversing its children.
    
    Args:
        doc: DoclingDocument
        text_element: Parent text element
        
    Returns:
        Ordered list of descendant text elements (including the element itself)
    """
    descendants = [text_element]  # Include the element itself
    
    # Recursively collect all children that are text elements
    if hasattr(text_element, 'children') and text_element.children:
        for ref_item in text_element.children:
            # Get ref path
            ref_path = None
            if hasattr(ref_item, 'cref'):
                ref_path = ref_item.cref
            elif hasattr(ref_item, 'get_ref'):
                ref_path = str(ref_item.get_ref())
            
            if not ref_path or not ref_path.startswith("#/texts/"):
                continue
            
            # This is a text reference
            resolved = resolve_ref_item(doc, ref_item)
            if resolved and hasattr(resolved, 'text'):
                # Add this text and recursively get its descendants
                descendants.append(resolved)
                child_descendants = get_text_descendants(doc, resolved)
                # Add descendants (excluding the element itself to avoid duplicates)
                for child_desc in child_descendants[1:]:  # Skip first (the element itself)
                    if child_desc not in descendants:
                        descendants.append(child_desc)
    
    return descendants


def find_section_container_in_hierarchy(doc: DoclingDocument, start_text: Any, start_path: List[str]) -> Optional[Any]:
    """
    Find the container element (group or parent) that contains the section in hierarchy.
    
    Args:
        doc: DoclingDocument
        start_text: Starting text element
        start_path: Hierarchy path to the starting text
        
    Returns:
        Container element (group or parent) that contains the section, or None
    """
    # Walk up the hierarchy path to find a group container
    # Start from body and follow the path
    current = doc.body
    if not current:
        return None
    
    # Follow path up to but not including the start_text itself
    for i, path_segment in enumerate(start_path[:-1]):  # Exclude last (the text itself)
        if not hasattr(current, 'children') or not current.children:
            break
        
        # Find child matching this path segment
        for ref_item in current.children:
            ref_path = None
            if hasattr(ref_item, 'cref'):
                ref_path = ref_item.cref
            elif hasattr(ref_item, 'get_ref'):
                ref_path = str(ref_item.get_ref())
            
            if ref_path == path_segment:
                resolved = resolve_ref_item(doc, ref_item)
                if resolved:
                    current = resolved
                    # If this is a group, it might be our container
                    if hasattr(resolved, 'label') or (hasattr(resolved, 'children') and resolved.children):
                        # Check if start_text is a descendant
                        descendants = get_text_descendants(doc, resolved)
                        if start_text in descendants:
                            return resolved
                break
    
    # If no group found, return the parent of start_text
    hierarchy_texts = get_hierarchy_texts_cached(doc)
    for te, parent, depth, path in hierarchy_texts:
        if te == start_text:
            return parent
    
    return None


def collect_texts_from_hierarchy_container(doc: DoclingDocument, container: Any, 
                                            start_text: Any, end_condition_func) -> List[Any]:
    """
    Collect all text elements from a hierarchy container until end condition is met.
    
    Args:
        doc: DoclingDocument
        container: Container element (group or parent)
        start_text: Starting text element
        end_condition_func: Function(text, parent, depth, path) -> bool that returns True when we should stop
        
    Returns:
        List of text elements in hierarchy order
    """
    if not container:
        return [start_text] if start_text else []
    
    collected_texts = []
    start_found = False
    
    # Traverse hierarchy starting from container
    hierarchy_texts = traverse_hierarchy_recursive(doc, container)
    
    for text_element, parent, depth, path in hierarchy_texts:
        # Check if we've reached the start
        if not start_found:
            if text_element == start_text:
                start_found = True
                collected_texts.append(text_element)
            continue
        
        # We're past the start, check end condition
        if end_condition_func(text_element, parent, depth, path):
            break
        
        collected_texts.append(text_element)
    
    return collected_texts


# ============================================================================
# PASS 1: Structure Building Functions
# ============================================================================

# ============================================================================
# Functions for finding all identifiers at different levels
# ============================================================================

def find_all_paragraphs(doc: DoclingDocument) -> List[str]:
    """
    Find all paragraphs in the document.
    
    Args:
        doc: DoclingDocument
        
    Returns:
        List of paragraph identifiers (e.g., ['1', '2', '3', '50', ...])
    """
    paragraph_numbers = []
    cached_elements = get_cached_elements(doc)
    paragraph_pattern = re.compile(r'§\s+(\d+[a-zA-Z]*)')
    
    # Use cached elements (no string operations needed)
    for cached in cached_elements:
        if cached.is_para_marker:
            match = paragraph_pattern.match(cached.normalized)
            if match:
                para_num_full = match.group(1)
                # Check if text starts with "§ {para_num_full}" (may include title)
                if cached.normalized.startswith(f'§ {para_num_full}') and para_num_full not in paragraph_numbers:
                    paragraph_numbers.append(para_num_full)
    
    return paragraph_numbers


def find_all_odseks_in_paragraph(doc: DoclingDocument, para_num: str) -> List[str]:
    """
    Find all odseks within a specific paragraph by scanning for markers.
    This is much faster than parsing the full paragraph structure.
    
    Args:
        doc: DoclingDocument
        para_num: Paragraph number (e.g., '5')
        
    Returns:
        List of odsek identifiers (e.g., ['5.1', '5.2', '5.3', ...])
    """
    start_time = time.time()
    
    # Find paragraph boundaries
    start_idx, end_idx = find_chunk_boundaries_hierarchical(doc, 'paragraph', para_num)
    if start_idx is None or end_idx is None:
        return []
    
    # Just scan for odsek markers, don't parse full structure
    # This is much faster - O(n) scan vs O(n*m) parsing
    odsek_identifiers = []
    odsek_numbers = set()
    
    # Define pismeno markers to avoid false positives
    pismeno_markers = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 
                      'k)', 'l)', 'm)', 'n)', 'o)', 'p)', 'q)', 'r)', 's)', 't)', 
                      'u)', 'v)', 'w)', 'x)', 'y)', 'z)']
    
    # Scan through paragraph text range
    for i in range(start_idx, min(end_idx, start_idx + 2000)):  # Safety limit
        if i >= len(doc.texts):
            break
            
        text = getattr(doc.texts[i], 'text', '').strip()
        normalized = text.replace('\xa0', ' ')
        hyperlink = getattr(doc.texts[i], 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        
        # Check for numeric odsek marker: (1), (2), etc.
        if normalized.startswith('(') and normalized[1:].rstrip(')').isdigit():
            odsek_num = normalized[1:].rstrip(')')
            odsek_numbers.add(odsek_num)
        # Check for letter-based odsek marker: aa), ab), etc. (but not single letter pismenos)
        elif (normalized.endswith(')') and not normalized.startswith('(') and 
              len(normalized) > 2 and normalized[:-1].isalpha() and 
              normalized not in pismeno_markers):
            odsek_num = normalized[:-1]  # Remove right parenthesis
            odsek_numbers.add(odsek_num)
        
        # Stop if we hit next paragraph
        if normalized.startswith('§ ') and not hyperlink_str:
            match = re.match(r'§\s+(\d+[a-zA-Z]*)', normalized)
            if match and match.group(1) != para_num:
                break
    
    # Convert to sorted list of identifiers
    sorted_numbers = sorted(odsek_numbers, key=lambda x: (len(x), x) if x.isalpha() else (0, int(x)))
    odsek_identifiers = [f'{para_num}.{num}' for num in sorted_numbers]
    
    elapsed = time.time() - start_time
    log_progress("DEBUG", f"Found {len(odsek_identifiers)} odseks in paragraph {para_num}", elapsed)
    
    return odsek_identifiers


def find_all_odseks_in_document(doc: DoclingDocument) -> List[str]:
    """
    Find all odseks in the entire document.
    
    Args:
        doc: DoclingDocument
        
    Returns:
        List of odsek identifiers (e.g., ['1.1', '1.2', '5.1', '5.2', ...])
    """
    all_odseks = []
    paragraphs = find_all_paragraphs(doc)
    
    for para_num in paragraphs:
        odseks = find_all_odseks_in_paragraph(doc, para_num)
        all_odseks.extend(odseks)
    
    return all_odseks


def find_all_pismenos_in_odsek(doc: DoclingDocument, para_num: str, odsek_num: str) -> List[str]:
    """
    Find all pismenos within a specific odsek by scanning for markers.
    This is much faster than parsing the full structure.
    
    Args:
        doc: DoclingDocument
        para_num: Paragraph number (e.g., '5')
        odsek_num: Odsek number (e.g., '1')
        
    Returns:
        List of pismeno identifiers (e.g., ['5.1.a', '5.1.b', ...])
    """
    start_time = time.time()
    
    # Find odsek boundaries
    identifier = f'{para_num}.{odsek_num}'
    start_idx, end_idx = find_chunk_boundaries_hierarchical(doc, 'odsek', identifier)
    if start_idx is None or end_idx is None:
        return []
    
    # Just scan for pismeno markers, don't parse full structure
    pismeno_identifiers = []
    pismeno_letters = set()
    
    # Define all possible pismeno markers (single letter only)
    pismeno_markers = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 
                      'k)', 'l)', 'm)', 'n)', 'o)', 'p)', 'q)', 'r)', 's)', 't)', 
                      'u)', 'v)', 'w)', 'x)', 'y)', 'z)']
    
    # Scan through odsek text range
    for i in range(start_idx, min(end_idx, start_idx + 500)):  # Safety limit
        if i >= len(doc.texts):
            break
            
        text = getattr(doc.texts[i], 'text', '').strip()
        normalized = text.replace('\xa0', ' ')
        
        # Check for pismeno marker: a), b), c), etc.
        if normalized in pismeno_markers:
            pismeno_letter = normalized.rstrip(')')
            pismeno_letters.add(pismeno_letter)
        
        # Stop if we hit next odsek or paragraph
        if normalized.startswith('(') and normalized[1:].rstrip(')').isdigit():
            next_num = normalized[1:].rstrip(')')
            if next_num != odsek_num:
                break
        elif normalized.startswith('§ '):
            match = re.match(r'§\s+(\d+[a-zA-Z]*)', normalized)
            if match and match.group(1) != para_num:
                break
    
    # Convert to sorted list of identifiers
    sorted_letters = sorted(pismeno_letters)
    pismeno_identifiers = [f'{para_num}.{odsek_num}.{letter}' for letter in sorted_letters]
    
    elapsed = time.time() - start_time
    log_progress("DEBUG", f"Found {len(pismeno_identifiers)} pismenos in odsek {para_num}.{odsek_num}", elapsed)
    
    return pismeno_identifiers


def find_all_pismenos_in_paragraph(doc: DoclingDocument, para_num: str) -> List[str]:
    """
    Find all pismenos within a specific paragraph (across all odseks).
    
    Args:
        doc: DoclingDocument
        para_num: Paragraph number (e.g., '5')
        
    Returns:
        List of pismeno identifiers (e.g., ['5.1.a', '5.1.b', '5.2.a', ...])
    """
    all_pismenos = []
    odseks = find_all_odseks_in_paragraph(doc, para_num)
    
    for odsek_id in odseks:
        parts = odsek_id.split('.')
        if len(parts) == 2:
            para, odsek = parts
            pismenos = find_all_pismenos_in_odsek(doc, para, odsek)
            all_pismenos.extend(pismenos)
    
    return all_pismenos


def find_all_pismenos_in_document(doc: DoclingDocument) -> List[str]:
    """
    Find all pismenos in the entire document.
    
    Args:
        doc: DoclingDocument
        
    Returns:
        List of pismeno identifiers (e.g., ['1.1.a', '1.1.b', '5.1.a', ...])
    """
    all_pismenos = []
    paragraphs = find_all_paragraphs(doc)
    
    for para_num in paragraphs:
        pismenos = find_all_pismenos_in_paragraph(doc, para_num)
        all_pismenos.extend(pismenos)
    
    return all_pismenos


def detect_parts(doc: DoclingDocument) -> List[Dict[str, Any]]:
    """
    Detect all parts (ČASŤ) in the document.
    
    Optimized to use cached elements for faster string operations.
    
    Args:
        doc: DoclingDocument
        
    Returns:
        List of part dictionaries with title and position
    """
    parts = []
    cached_elements = get_cached_elements(doc)
    
    part_patterns = [
        re.compile(r'PRVÁ\s+ČASŤ', re.IGNORECASE),
        re.compile(r'DRUHÁ\s+ČASŤ', re.IGNORECASE),
        re.compile(r'TRETIA\s+ČASŤ', re.IGNORECASE),
        re.compile(r'ŠTVRTÁ\s+ČASŤ', re.IGNORECASE),
        re.compile(r'PIATA\s+ČASŤ', re.IGNORECASE),
        re.compile(r'ŠIESTA\s+ČASŤ', re.IGNORECASE),
        re.compile(r'SEDMÁ\s+ČASŤ', re.IGNORECASE),
        re.compile(r'ÔSMA\s+ČASŤ', re.IGNORECASE),
        re.compile(r'DEVÄTÁ\s+ČASŤ', re.IGNORECASE),
        re.compile(r'DESIATA\s+ČASŤ', re.IGNORECASE)
    ]
    
    # Use cached normalized text (no string operations needed)
    for cached in cached_elements:
        for pattern in part_patterns:
            if pattern.search(cached.normalized):
                parts.append({
                    'title': cached.text,
                    'text_element': cached.element,
                    'parent': cached.parent,
                    'depth': cached.depth,
                    'path': cached.path
                })
                break  # Only one part per element
    
    return parts


def find_part_for_element(doc: DoclingDocument, element: Any, element_path: List[str]) -> Optional[Dict[str, Any]]:
    """
    Find the part (ČASŤ) that contains a given element.
    
    Args:
        doc: DoclingDocument
        element: Text element to find part for
        element_path: Hierarchy path to the element
        
    Returns:
        Part dictionary or None
    """
    parts = detect_parts(doc)
    if not parts:
        return None
    
    # Find which part contains this element
    # We'll check by comparing hierarchy positions
    cached_elements = get_cached_elements(doc)
    
    # Find element position in hierarchy
    element_pos = -1
    for i, cached in enumerate(cached_elements):
        if cached.element == element:
            element_pos = i
            break
    
    if element_pos < 0:
        return None
    
    # Find the last part that appears before this element
    last_part = None
    for part in parts:
        part_pos = -1
        for i, cached in enumerate(cached_elements):
            if cached.element == part['text_element']:
                part_pos = i
                break
        
        if part_pos >= 0 and part_pos <= element_pos:
            if last_part is None or part_pos > last_part.get('position', -1):
                last_part = part
                last_part['position'] = part_pos
    
    return last_part


def extract_paragraph_intro_text(doc: DoclingDocument, para_num: str, 
                                 para_structure: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract introductory text of a paragraph (text between paragraph title and first odsek).
    
    Uses already parsed paragraph structure for efficiency - avoids re-traversing the document.
    If para_structure is provided, uses it directly (fast!). Otherwise parses it (slower).
    
    Args:
        doc: DoclingDocument
        para_num: Paragraph number (e.g., '5')
        para_structure: Optional pre-parsed paragraph structure (for performance)
        
    Returns:
        Dictionary with text content (text, tables, pictures, references, footnotes)
    """
    start_time = time.time()
    
    # If structure provided, use it directly (fast path)
    if para_structure:
        intro_content = {
            'text': para_structure.get('content', {}).get('text', ''),
            'tables': para_structure.get('content', {}).get('tables', []),
            'pictures': para_structure.get('content', {}).get('pictures', []),
            'references': para_structure.get('content', {}).get('references', []),
            'footnotes': para_structure.get('content', {}).get('footnotes', [])
        }
        elapsed = time.time() - start_time
        log_progress("DEBUG", f"Extracted paragraph {para_num} intro text (from cache)", elapsed)
        return intro_content
    
    # Fallback: parse if structure not provided (slower path)
    para_start_idx, para_end_idx = find_chunk_boundaries_hierarchical(doc, 'paragraph', para_num)
    if para_start_idx is None or para_end_idx is None:
        return {
            'text': '',
            'tables': [],
            'pictures': [],
            'references': [],
            'footnotes': []
        }
    
    # Check cache first
    cache_key = f"{para_num}_{para_start_idx}_{para_end_idx}"
    if cache_key not in _paragraph_structure_cache:
        _paragraph_structure_cache[cache_key] = parse_paragraph_structure(doc, para_start_idx, para_end_idx)
    
    para_structure = _paragraph_structure_cache[cache_key]
    
    # Extract intro content directly from structure
    intro_content = {
        'text': para_structure.get('content', {}).get('text', ''),
        'tables': para_structure.get('content', {}).get('tables', []),
        'pictures': para_structure.get('content', {}).get('pictures', []),
        'references': para_structure.get('content', {}).get('references', []),
        'footnotes': para_structure.get('content', {}).get('footnotes', [])
    }
    
    elapsed = time.time() - start_time
    log_progress("DEBUG", f"Extracted paragraph {para_num} intro text", elapsed)
    
    return intro_content


def parse_odsek_structure(doc: DoclingDocument, start_idx: int, end_idx: int, 
                         para_num: str, odsek_num: str) -> Dict[str, Any]:
    """
    Parse odsek structure (only the odsek, not the whole paragraph).
    
    Args:
        doc: DoclingDocument
        start_idx: Start index in doc.texts (odsek start)
        end_idx: End index in doc.texts (odsek end)
        para_num: Paragraph number
        odsek_num: Odsek number
        
    Returns:
        Structure dictionary for the odsek only
    """
    # Get text items for this odsek
    text_items = doc.texts[start_idx:end_idx]
    
    structure = {
        'type': 'odsek',
        'marker': f'({odsek_num})',
        'content': {
            'text': '',
            'subsubsections': [],  # pismenos
            'tables': [],
            'pictures': [],
            'references': [],
            'footnotes': []
        }
    }
    
    current_pismeno = None
    odsek_texts = []
    pismeno_texts = []
    
    # Process texts
    for text_element in text_items:
        text = getattr(text_element, 'text', '').strip()
        normalized_text = text.replace('\xa0', ' ')
        hyperlink = getattr(text_element, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        
        # Skip odsek marker itself
        if normalized_text == f'({odsek_num})':
            continue
        
        # Define all possible pismeno markers (single letter only)
        pismeno_markers = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 
                          'k)', 'l)', 'm)', 'n)', 'o)', 'p)', 'q)', 'r)', 's)', 't)', 
                          'u)', 'v)', 'w)', 'x)', 'y)', 'z)']
        
        # Check for pismeno marker FIRST
        if normalized_text in pismeno_markers:
            # Save previous pismeno if exists
            if current_pismeno:
                pismeno_content = parse_pismeno_with_subitems(doc, pismeno_texts, current_pismeno)
                structure['content']['subsubsections'].append({
                    'type': 'pismeno',
                    'marker': current_pismeno,
                    'content': pismeno_content
                })
            
            # Start new pismeno
            current_pismeno = normalized_text
            pismeno_texts = []
            continue
        
        # Check for next odsek or paragraph (end condition)
        is_odsek = False
        if normalized_text.startswith('(') and normalized_text[1:].rstrip(')').isdigit():
            next_num = normalized_text[1:].rstrip(')')
            if next_num != odsek_num:
                # Next odsek, stop processing
                break
            is_odsek = True
        elif (normalized_text.endswith(')') and not normalized_text.startswith('(') and 
              len(normalized_text) > 2 and normalized_text[:-1].isalpha() and 
              normalized_text not in pismeno_markers):
            # Písmenový odsek
            is_odsek = True
        
        if normalized_text.startswith('§ ') and not hyperlink_str:
            match = re.match(r'§\s+(\d+[a-zA-Z]*)', normalized_text)
            if match and match.group(1) != para_num:
                # Next paragraph, stop processing
                break
        
        # Add text to appropriate level
        if current_pismeno:
            pismeno_texts.append(text_element)
        else:
            odsek_texts.append(text_element)
    
    # Save last pismeno if exists
    if current_pismeno:
        pismeno_content = parse_pismeno_with_subitems(doc, pismeno_texts, current_pismeno)
        structure['content']['subsubsections'].append({
            'type': 'pismeno',
            'marker': current_pismeno,
            'content': pismeno_content
        })
    
    # Extract odsek-level content
    odsek_content = extract_content_from_texts(doc, odsek_texts)
    structure['content'].update(odsek_content)
    
    return structure


def extract_superior_content(doc: DoclingDocument, level: str, identifier: str, 
                            start_text: Any, start_path: List[str],
                            para_structure: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract superior content (parts, sections) for a chunk.
    
    Args:
        doc: DoclingDocument
        level: Chunking level ('paragraph', 'odsek', 'pismeno')
        identifier: Section identifier
        start_text: Starting text element of the chunk
        start_path: Hierarchy path to starting text
        
    Returns:
        Dictionary with superior content (part, section if any, paragraph_intro for odsek/pismeno)
    """
    superior = {}
    
    # For odsek and pismeno levels, we need to find the paragraph first
    para_text = None
    para_path = None
    para_num = None
    
    if level == 'odsek' or level == 'pismeno':
        # Parse identifier to get paragraph number
        parts = identifier.split('.')
        para_num = parts[0]
        
        # Find paragraph marker in hierarchy
        para_marker = f'§ {para_num}'
        para_result = find_text_in_hierarchy(doc, para_marker, no_hyperlink=True, exact_match=False)
        
        if para_result:
            para_text, para_parent, para_depth, para_path = para_result
            # Use paragraph text for finding part
            start_text = para_text
            start_path = para_path
    
    # Find part (using paragraph text for odsek/pismeno, original start_text for paragraph)
    part = find_part_for_element(doc, start_text, start_path)
    if part:
        # Extract part content - only the title, stop at first paragraph or next part
        part_texts = []
        hierarchy_texts = get_hierarchy_texts_cached(doc)
        
        part_pos = -1
        for i, (te, parent, depth, path) in enumerate(hierarchy_texts):
            if te == part['text_element']:
                part_pos = i
                break
        
        if part_pos >= 0:
            # Collect texts until next part, first paragraph marker, or end
            next_part_pos = len(hierarchy_texts)
            for p in detect_parts(doc):
                p_pos = -1
                for i, (te, parent, depth, path) in enumerate(hierarchy_texts):
                    if te == p['text_element']:
                        p_pos = i
                        break
                if p_pos > part_pos:
                    next_part_pos = min(next_part_pos, p_pos)
                    break
            
            # Collect part title texts - stop at first paragraph marker (§) or next part
            for i in range(part_pos, min(next_part_pos, len(hierarchy_texts))):
                te, parent, depth, path = hierarchy_texts[i]
                text = getattr(te, 'text', '').strip()
                normalized_text = text.replace('\xa0', ' ')
                hyperlink = getattr(te, 'hyperlink', '')
                hyperlink_str = str(hyperlink) if hyperlink else ''
                
                # Stop if we encounter a paragraph marker (start of actual content)
                # This ensures superior content ends at the end of the part title
                if normalized_text.startswith('§ ') and not hyperlink_str:
                    break
                
                part_texts.append(te)
        
        # Extract text content from text elements (serializable format)
        part_content = extract_content_from_texts(doc, part_texts)
        
        superior['part'] = {
            'title': part['title'],
            'content': part_content
        }
    else:
        superior['part'] = None
    
    # For odsek and pismeno levels, add paragraph introductory text
    if (level == 'odsek' or level == 'pismeno') and para_num:
        # Pass para_structure if available to avoid re-parsing
        paragraph_intro = extract_paragraph_intro_text(doc, para_num, para_structure)
        superior['paragraph_intro'] = paragraph_intro
    else:
        superior['paragraph_intro'] = None
    
    # Sections - placeholder for future implementation
    superior['section'] = None
    
    return superior


def parse_paragraph_structure(doc: DoclingDocument, start_idx: int, end_idx: int) -> Dict[str, Any]:
    """
    Parse paragraph structure using hierarchy traversal to properly reconstruct nested structure.
    
    Args:
        doc: DoclingDocument
        start_idx: Start index in doc.texts
        end_idx: End index in doc.texts
        
    Returns:
        Nested structure dictionary with odseks and pismenos
    """
    # Get the paragraph start text element
    start_text = doc.texts[start_idx]
    
    # Find the paragraph container in hierarchy
    hierarchy_texts = get_hierarchy_texts_cached(doc)
    start_path = None
    for te, parent, depth, path in hierarchy_texts:
        if te == start_text:
            start_path = path
            break
    
    if not start_path:
        # Fallback: use sequential processing
        text_items = doc.texts[start_idx:end_idx]
        return parse_paragraph_structure_sequential(doc, text_items)
    
    # Find paragraph container
    para_container = find_section_container_in_hierarchy(doc, start_text, start_path)
    if not para_container:
        # Fallback: use sequential processing
        text_items = doc.texts[start_idx:end_idx]
        return parse_paragraph_structure_sequential(doc, text_items)
    
    # Get paragraph number from start_text for end condition
    start_text_content = getattr(start_text, 'text', '').strip()
    start_normalized = start_text_content.replace('\xa0', ' ')
    start_match = re.match(r'§\s+(\d+[a-zA-Z]*)', start_normalized)
    para_num = start_match.group(1) if start_match else None
    
    # Define end condition for paragraph
    def end_condition(text_element: Any, parent: Any, depth: int, path: List[str]) -> bool:
        """Return True when we should stop collecting texts (next paragraph)."""
        text_content = getattr(text_element, 'text', '').strip()
        normalized_content = text_content.replace('\xa0', ' ')
        hyperlink = getattr(text_element, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        
        # End at next paragraph
        if normalized_content.startswith('§ ') and not hyperlink_str:
            match = re.match(r'§\s+(\d+[a-zA-Z]*)', normalized_content)
            if match and para_num:
                if match.group(1) != para_num:
                    return True
        
        return False
    
    # Collect all texts from paragraph container in hierarchy order
    collected_texts = collect_texts_from_hierarchy_container(
        doc, para_container, start_text, end_condition
    )
    
    if not collected_texts:
        collected_texts = [start_text]
    
    # Now process collected texts in hierarchy order to build nested structure
    structure = {
        'type': 'paragraph',
        'title': '',
        'content': {
            'text': '',
            'subsections': [],
            'tables': [],
            'pictures': [],
            'references': [],
            'footnotes': []
        }
    }
    
    current_odsek = None
    current_pismeno = None
    paragraph_title = ''
    paragraph_text = []
    odsek_texts = []
    pismeno_texts = []
    
    # Process texts in hierarchy order (not sequential from doc.texts)
    for text_element in collected_texts:
        text = getattr(text_element, 'text', '').strip()
        normalized_text = text.replace('\xa0', ' ')
        hyperlink = getattr(text_element, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        
        # Check for paragraph title
        if normalized_text.startswith('§ ') and not hyperlink_str:
            match = re.match(r'§\s+(\d+[a-zA-Z]*)', normalized_text)
            if match:
                paragraph_title = normalized_text
                continue
        
        # Define all possible pismeno markers (single letter only)
        pismeno_markers = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 
                          'k)', 'l)', 'm)', 'n)', 'o)', 'p)', 'q)', 'r)', 's)', 't)', 
                          'u)', 'v)', 'w)', 'x)', 'y)', 'z)']
        
        # Check for pismeno marker FIRST (single letter: a), b), c), etc.)
        # This must be checked before odsek markers to avoid conflicts
        if normalized_text in pismeno_markers:
            # Save previous pismeno if exists
            if current_pismeno:
                if current_odsek:
                    pismeno_content = parse_pismeno_with_subitems(doc, pismeno_texts, current_pismeno)
                    current_odsek['content']['subsubsections'].append({
                        'type': 'pismeno',
                        'marker': current_pismeno,
                        'content': pismeno_content
                    })
            
            # Start new pismeno
            current_pismeno = normalized_text
            pismeno_texts = []
            continue
        
        # Check for odsek marker: (1), (2), etc. OR a), aa), ab), etc. (letters with right parenthesis)
        # Odseky môžu byť:
        # - Číselné v úplných zátvorkách: (1), (2)
        # - Písmenové len s pravou zátvorkou: a), aa), ab) (ale len ak to nie je pismeno - t.j. má viac ako jedno písmeno)
        is_odsek = False
        odsek_marker = None
        
        # Číselný odsek v úplných zátvorkách: (1), (2), atď.
        if normalized_text.startswith('(') and normalized_text[1:].rstrip(')').isdigit():
            is_odsek = True
            odsek_marker = normalized_text
        # Písmenový odsek len s pravou zátvorkou: a), aa), ab), atď.
        # (ale len ak to nie je pismeno - t.j. má viac ako jedno písmeno)
        elif (normalized_text.endswith(')') and not normalized_text.startswith('(') and 
              len(normalized_text) > 2 and normalized_text[:-1].isalpha() and 
              normalized_text not in pismeno_markers):
            is_odsek = True
            odsek_marker = normalized_text
        
        if is_odsek:
            # Save previous pismeno if exists
            if current_pismeno:
                if current_odsek:
                    pismeno_content = parse_pismeno_with_subitems(doc, pismeno_texts, current_pismeno)
                    current_odsek['content']['subsubsections'].append({
                        'type': 'pismeno',
                        'marker': current_pismeno,
                        'content': pismeno_content
                    })
                current_pismeno = None
                pismeno_texts = []
            
            # Save previous odsek if exists
            if current_odsek:
                odsek_content = extract_content_from_texts(doc, odsek_texts)
                current_odsek['content'].update(odsek_content)
                structure['content']['subsections'].append(current_odsek)
            
            # Start new odsek
            if odsek_marker.startswith('('):
                odsek_num = odsek_marker[1:].rstrip(')')
            else:
                odsek_num = odsek_marker[:-1]  # Remove right parenthesis
            current_odsek = {
                'type': 'odsek',
                'marker': odsek_marker,
                'content': {
                    'text': '',
                    'subsubsections': [],
                    'tables': [],
                    'pictures': [],
                    'references': [],
                    'footnotes': []
                }
            }
            odsek_texts = []
            continue
        
        # Add text to appropriate level based on current state
        # Since we're using hierarchy order, we don't need embedded marker detection
        # The hierarchy already groups texts correctly
        if current_pismeno:
            pismeno_texts.append(text_element)
        elif current_odsek:
            odsek_texts.append(text_element)
        else:
            paragraph_text.append(text_element)
    
    # Save last pismeno if exists
    if current_pismeno:
        pismeno_content = parse_pismeno_with_subitems(doc, pismeno_texts, current_pismeno)
        if current_odsek:
            current_odsek['content']['subsubsections'].append({
                'type': 'pismeno',
                'marker': current_pismeno,
                'content': pismeno_content
            })
    
    # Save last odsek if exists
    if current_odsek:
        odsek_content = extract_content_from_texts(doc, odsek_texts)
        current_odsek['content'].update(odsek_content)
        structure['content']['subsections'].append(current_odsek)
    
    # Extract paragraph-level content
    paragraph_content = extract_content_from_texts(doc, paragraph_text)
    structure['title'] = paragraph_title
    structure['content'].update(paragraph_content)
    
    return structure


def parse_paragraph_structure_sequential(doc: DoclingDocument, text_items: List[Any]) -> Dict[str, Any]:
    """
    Fallback: Parse paragraph structure sequentially (for when hierarchy traversal fails).
    
    Args:
        doc: DoclingDocument
        text_items: List of text items to process
        
    Returns:
        Nested structure dictionary with odseks and pismenos
    """
    structure = {
        'type': 'paragraph',
        'title': '',
        'content': {
            'text': '',
            'subsections': [],
            'tables': [],
            'pictures': [],
            'references': [],
            'footnotes': []
        }
    }
    
    current_odsek = None
    current_pismeno = None
    paragraph_title = ''
    paragraph_text = []
    odsek_texts = []
    pismeno_texts = []
    
    for item in text_items:
        text = getattr(item, 'text', '').strip()
        normalized_text = text.replace('\xa0', ' ')
        hyperlink = getattr(item, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        
        # Check for paragraph title
        if normalized_text.startswith('§ ') and not hyperlink_str:
            match = re.match(r'§\s+(\d+[a-zA-Z]*)', normalized_text)
            if match:
                paragraph_title = normalized_text
                continue
        
        # Define all possible pismeno markers (single letter only)
        pismeno_markers = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 
                          'k)', 'l)', 'm)', 'n)', 'o)', 'p)', 'q)', 'r)', 's)', 't)', 
                          'u)', 'v)', 'w)', 'x)', 'y)', 'z)']
        
        # Check for pismeno marker FIRST (single letter: a), b), c), etc.)
        # This must be checked before odsek markers to avoid conflicts
        if normalized_text in pismeno_markers:
            # Save previous pismeno if exists
            if current_pismeno:
                if current_odsek:
                    pismeno_content = parse_pismeno_with_subitems(doc, pismeno_texts, current_pismeno)
                    current_odsek['content']['subsubsections'].append({
                        'type': 'pismeno',
                        'marker': current_pismeno,
                        'content': pismeno_content
                    })
            
            # Start new pismeno
            current_pismeno = normalized_text
            pismeno_texts = []
            continue
        
        # Check for odsek marker: (1), (2), etc. OR a), aa), ab), etc. (letters with right parenthesis)
        # Odseky môžu byť:
        # - Číselné v úplných zátvorkách: (1), (2)
        # - Písmenové len s pravou zátvorkou: a), aa), ab) (ale len ak to nie je pismeno - t.j. má viac ako jedno písmeno)
        is_odsek = False
        odsek_marker = None
        
        # Číselný odsek v úplných zátvorkách: (1), (2), atď.
        if normalized_text.startswith('(') and normalized_text[1:].rstrip(')').isdigit():
            is_odsek = True
            odsek_marker = normalized_text
        # Písmenový odsek len s pravou zátvorkou: a), aa), ab), atď.
        # (ale len ak to nie je pismeno - t.j. má viac ako jedno písmeno)
        elif (normalized_text.endswith(')') and not normalized_text.startswith('(') and 
              len(normalized_text) > 2 and normalized_text[:-1].isalpha() and 
              normalized_text not in pismeno_markers):
            is_odsek = True
            odsek_marker = normalized_text
        
        if is_odsek:
            # Save previous pismeno if exists
            if current_pismeno:
                if current_odsek:
                    pismeno_content = parse_pismeno_with_subitems(doc, pismeno_texts, current_pismeno)
                    current_odsek['content']['subsubsections'].append({
                        'type': 'pismeno',
                        'marker': current_pismeno,
                        'content': pismeno_content
                    })
                current_pismeno = None
                pismeno_texts = []
            
            # Save previous odsek if exists
            if current_odsek:
                odsek_content = extract_content_from_texts(doc, odsek_texts)
                current_odsek['content'].update(odsek_content)
                structure['content']['subsections'].append(current_odsek)
            
            # Start new odsek
            if odsek_marker.startswith('('):
                odsek_num = odsek_marker[1:].rstrip(')')
            else:
                odsek_num = odsek_marker[:-1]  # Remove right parenthesis
            current_odsek = {
                'type': 'odsek',
                'marker': odsek_marker,
                'content': {
                    'text': '',
                    'subsubsections': [],
                    'tables': [],
                    'pictures': [],
                    'references': [],
                    'footnotes': []
                }
            }
            odsek_texts = []
            continue
        
        # Add text to appropriate level
        if current_pismeno:
            pismeno_texts.append(item)
        elif current_odsek:
            odsek_texts.append(item)
        else:
            paragraph_text.append(item)
    
    # Save last pismeno if exists
    if current_pismeno:
        pismeno_content = parse_pismeno_with_subitems(doc, pismeno_texts, current_pismeno)
        if current_odsek:
            current_odsek['content']['subsubsections'].append({
                'type': 'pismeno',
                'marker': current_pismeno,
                'content': pismeno_content
            })
    
    # Save last odsek if exists
    if current_odsek:
        odsek_content = extract_content_from_texts(doc, odsek_texts)
        current_odsek['content'].update(odsek_content)
        structure['content']['subsections'].append(current_odsek)
    
    # Extract paragraph-level content
    paragraph_content = extract_content_from_texts(doc, paragraph_text)
    structure['title'] = paragraph_title
    structure['content'].update(paragraph_content)
    
    return structure


def parse_pismeno_with_subitems(doc: DoclingDocument, text_items: List[Any], pismeno_marker: str) -> Dict[str, Any]:
    """
    Parse pismeno content and extract numbered sub-items (1., 2., 3., etc.).
    
    Args:
        doc: DoclingDocument
        text_items: List of text elements for the pismeno
        pismeno_marker: The pismeno marker (e.g., "q)")
        
    Returns:
        Dictionary with text, subitems, tables, pictures, references, footnotes
    """
    content = {
        'text': '',
        'subitems': [],
        'tables': [],
        'pictures': [],
        'references': [],
        'footnotes': []
    }
    
    if not text_items:
        return content
    
    # Separate intro text from subitems
    intro_texts = []
    current_subitem = None
    subitem_texts = []
    subitems = []
    
    for text_element in text_items:
        text = getattr(text_element, 'text', '').strip()
        normalized_text = text.replace('\xa0', ' ')
        
        # Check if this is a numbered sub-item marker (1., 2., 3., etc.)
        # Pattern: number followed by period (e.g., "1.", "2.", "10.")
        if re.match(r'^\d+\.$', normalized_text):
            # Save previous subitem if exists
            if current_subitem:
                subitem_content = extract_content_from_texts(doc, subitem_texts)
                subitems.append({
                    'type': 'subitem',
                    'marker': current_subitem,
                    'content': subitem_content
                })
            
            # Start new subitem
            current_subitem = normalized_text
            subitem_texts = []
        else:
            # Add to appropriate collection
            if current_subitem:
                subitem_texts.append(text_element)
            else:
                intro_texts.append(text_element)
    
    # Save last subitem if exists
    if current_subitem:
        subitem_content = extract_content_from_texts(doc, subitem_texts)
        subitems.append({
            'type': 'subitem',
            'marker': current_subitem,
            'content': subitem_content
        })
    
    # Extract intro text content
    if intro_texts:
        intro_content = extract_content_from_texts(doc, intro_texts)
        content['text'] = intro_content.get('text', '')
        content['tables'] = intro_content.get('tables', [])
        content['pictures'] = intro_content.get('pictures', [])
        content['references'] = intro_content.get('references', [])
        content['footnotes'] = intro_content.get('footnotes', [])
    
    # Add subitems
    content['subitems'] = subitems
    
    # Merge tables, pictures, references, footnotes from subitems into main content
    for subitem in subitems:
        subitem_content = subitem.get('content', {})
        content['tables'].extend(subitem_content.get('tables', []))
        content['pictures'].extend(subitem_content.get('pictures', []))
        content['references'].extend(subitem_content.get('references', []))
        content['footnotes'].extend(subitem_content.get('footnotes', []))
    
    return content


def extract_content_from_texts(doc: DoclingDocument, text_items: List[Any]) -> Dict[str, Any]:
    """
    Extract content (text, tables, pictures, references, footnotes) from text items.
    Places elements within the structure where they appear.
    
    Args:
        doc: DoclingDocument
        text_items: List of text elements
        
    Returns:
        Dictionary with text, tables, pictures, references, footnotes
    """
    content = {
        'text': '',
        'tables': [],
        'pictures': [],
        'references': [],
        'footnotes': []
    }
    
    # Extract text content
    text_parts = []
    seen_tables = set()
    seen_pictures = set()
    
    for item in text_items:
        text = getattr(item, 'text', '').strip()
        hyperlink = getattr(item, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        label = getattr(item, 'label', '')
        
        # Skip navigation elements
        if (text and 
            text not in ['plus', 'button-close', 'button-search', 'button-download', 
                        'button-print', 'button-history', 'button-content'] and
            label != 'caption'):
            
            # Handle structural markers
            if text.strip() in ['(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)', '(8)', '(9)', '(10)']:
                text_parts.append(f'\n\n### {text}\n\n')
            elif text.strip() in ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)']:
                text_parts.append(f'\n**{text}** ')
            else:
                # Regular text
                if text and len(text) > 1:
                    if text_parts and not text_parts[-1].endswith(' ') and not text_parts[-1].endswith('\n'):
                        if text[0] not in ['.', ',', ';', ':', '!', '?', ')', ']', '}', '(', '[', '{']:
                            text_parts.append(' ')
                    text_parts.append(text)
        
        # Extract tables
        if hyperlink_str and '#/tables/' in hyperlink_str:
            table_idx = int(hyperlink_str.split('/')[-1])
            if table_idx not in seen_tables and table_idx < len(doc.tables):
                table = doc.tables[table_idx]
                table_md = format_table_as_markdown(table, doc)
                if table_md:
                    content['tables'].append({
                        'index': table_idx,
                        'markdown': table_md,
                        'caption': getattr(table, 'caption_text', None)
                    })
                    seen_tables.add(table_idx)
        
        # Extract pictures
        if hyperlink_str and '#/pictures/' in hyperlink_str:
            pic_idx = int(hyperlink_str.split('/')[-1])
            if pic_idx not in seen_pictures and pic_idx < len(doc.pictures):
                picture = doc.pictures[pic_idx]
                pic_info = format_picture_reference(picture)
                content['pictures'].append({
                    'index': pic_idx,
                    **pic_info
                })
                seen_pictures.add(pic_idx)
        
        # Extract internal references
        if hyperlink_str and 'paragraf-' in hyperlink_str and 'poznamky' not in hyperlink_str:
            content['references'].append({
                'type': 'internal',
                'hyperlink': hyperlink_str,
                'reference_text': text,
                'position': 'inline'
            })
        
        # Extract external references (placeholder)
        if hyperlink_str and ('../../' in hyperlink_str or hyperlink_str.startswith('http')):
            content['references'].append({
                'type': 'external',
                'hyperlink': hyperlink_str,
                'reference_text': text,
                'position': 'inline'
            })
        
        # Extract footnotes
        if hyperlink_str and 'poznamky.poznamka' in hyperlink_str:
            if 'poznamka-' in hyperlink_str:
                fn_id = hyperlink_str.split('poznamka-')[-1]
                if text != ')':  # Don't count closing paren
                    content['footnotes'].append({
                        'type': 'footnote',
                        'footnote_id': fn_id,
                        'hyperlink': hyperlink_str,
                        'position': 'inline'
                    })
    
    # Format text
    text_content = ''.join(text_parts)
    text_content = re.sub(r' +', ' ', text_content)
    text_content = re.sub(r' \n', '\n', text_content)
    text_content = re.sub(r'\n{3,}', '\n\n', text_content)
    content['text'] = text_content.strip()
    
    return content


def format_table_as_markdown(table: Any, doc: DoclingDocument) -> str:
    """
    Convert a table object to markdown format.
    
    Args:
        table: Table object from doc.tables
        doc: DoclingDocument
        
    Returns:
        Markdown representation of the table
    """
    try:
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


def format_picture_reference(picture: Any) -> Dict[str, Any]:
    """
    Format picture as reference information.
    
    Args:
        picture: Picture object from doc.pictures
        
    Returns:
        Dictionary with picture metadata
    """
    prov = getattr(picture, 'prov', [])
    caption = getattr(picture, 'caption_text', None)
    
    info = {
        'caption': caption if caption and not callable(caption) else None,
        'page': prov[0].page_no if prov else None,
    }
    
    if prov and prov[0].bbox:
        info['bounding_box'] = {
            'left': prov[0].bbox.l,
            'top': prov[0].bbox.t,
            'right': prov[0].bbox.r,
            'bottom': prov[0].bbox.b
        }
    
    return info


def collect_references_metadata(structure: Dict[str, Any], path_prefix: str = '') -> List[Dict[str, Any]]:
    """
    Collect all references from nested structure with their positions.
    
    Args:
        structure: Nested structure dictionary
        path_prefix: Path prefix for position tracking (e.g., 'odsek-1.pismeno-a')
        
    Returns:
        List of reference metadata dictionaries
    """
    refs_metadata = {}
    
    def collect_from_content(content: Dict[str, Any], position: str):
        """Recursively collect references from content."""
        for ref in content.get('references', []):
            hyperlink = ref.get('hyperlink', '')
            if hyperlink:
                if hyperlink not in refs_metadata:
                    refs_metadata[hyperlink] = {
                        'hyperlink': hyperlink,
                        'reference_text': ref.get('reference_text', ''),
                        'occurrences': 1,
                        'positions': [position] if position else []
                    }
                else:
                    refs_metadata[hyperlink]['occurrences'] += 1
                    if position and position not in refs_metadata[hyperlink]['positions']:
                        refs_metadata[hyperlink]['positions'].append(position)
    
    # Collect from main content
    collect_from_content(structure.get('content', {}), path_prefix)
    
    # Collect from subsections (odseks)
    for i, odsek in enumerate(structure.get('content', {}).get('subsections', []), 1):
        odsek_path = f"{path_prefix}.odsek-{i}" if path_prefix else f"odsek-{i}"
        collect_from_content(odsek.get('content', {}), odsek_path)
        
        # Collect from subsubsections (pismenos)
        for j, pismeno in enumerate(odsek.get('content', {}).get('subsubsections', []), 1):
            pismeno_marker = pismeno.get('marker', '').rstrip(')')
            pismeno_path = f"{odsek_path}.pismeno-{pismeno_marker}"
            collect_from_content(pismeno.get('content', {}), pismeno_path)
            
            # Collect from subitems (numbered sub-items like 1., 2., etc.)
            subitems = pismeno.get('content', {}).get('subitems', [])
            for k, subitem in enumerate(subitems, 1):
                subitem_marker = subitem.get('marker', '').rstrip('.')
                subitem_path = f"{pismeno_path}.subitem-{subitem_marker}"
                collect_from_content(subitem.get('content', {}), subitem_path)
    
    return list(refs_metadata.values())


def collect_footnotes_metadata(structure: Dict[str, Any], path_prefix: str = '') -> List[Dict[str, Any]]:
    """
    Collect all footnotes from nested structure with their positions.
    
    Args:
        structure: Nested structure dictionary
        path_prefix: Path prefix for position tracking
        
    Returns:
        List of footnote metadata dictionaries
    """
    footnotes_metadata = {}
    
    def collect_from_content(content: Dict[str, Any], position: str):
        """Recursively collect footnotes from content."""
        for fn in content.get('footnotes', []):
            fn_id = fn.get('footnote_id', '')
            if fn_id:
                if fn_id not in footnotes_metadata:
                    footnotes_metadata[fn_id] = {
                        'footnote_id': fn_id,
                        'occurrences': 1,
                        'positions': [position] if position else []
                    }
                else:
                    footnotes_metadata[fn_id]['occurrences'] += 1
                    if position and position not in footnotes_metadata[fn_id]['positions']:
                        footnotes_metadata[fn_id]['positions'].append(position)
    
    # Collect from main content
    collect_from_content(structure.get('content', {}), path_prefix)
    
    # Collect from subsections (odseks)
    for i, odsek in enumerate(structure.get('content', {}).get('subsections', []), 1):
        odsek_path = f"{path_prefix}.odsek-{i}" if path_prefix else f"odsek-{i}"
        collect_from_content(odsek.get('content', {}), odsek_path)
        
        # Collect from subsubsections (pismenos)
        for j, pismeno in enumerate(odsek.get('content', {}).get('subsubsections', []), 1):
            pismeno_marker = pismeno.get('marker', '').rstrip(')')
            pismeno_path = f"{odsek_path}.pismeno-{pismeno_marker}"
            collect_from_content(pismeno.get('content', {}), pismeno_path)
            
            # Collect from subitems (numbered sub-items like 1., 2., etc.)
            subitems = pismeno.get('content', {}).get('subitems', [])
            for k, subitem in enumerate(subitems, 1):
                subitem_marker = subitem.get('marker', '').rstrip('.')
                subitem_path = f"{pismeno_path}.subitem-{subitem_marker}"
                collect_from_content(subitem.get('content', {}), subitem_path)
    
    return list(footnotes_metadata.values())


def build_chunk_structure_pass1(doc: DoclingDocument, level: str, identifier: str,
                                process_internal_refs: bool = False,
                                process_footnotes: bool = False,
                                process_external_refs: bool = False) -> Optional[Dict[str, Any]]:
    """
    Pass 1: Build chunk with nested hierarchical structure, superior content, and references as metadata.
    
    Args:
        doc: DoclingDocument
        level: Chunking level ('paragraph', 'odsek', 'pismeno')
        identifier: Section identifier
        process_internal_refs: If True, extract internal references (as metadata)
        process_footnotes: If True, extract footnotes (as metadata)
        process_external_refs: If True, extract external references (as metadata)
        
    Returns:
        Chunk dictionary with structure (references not resolved)
    """
    chunk_start = time.time()
    
    # Find chunk boundaries
    boundary_start = time.time()
    start_idx, end_idx = find_chunk_boundaries_hierarchical(doc, level, identifier)
    boundary_time = time.time() - boundary_start
    
    if start_idx is None or end_idx is None:
        return None
    
    # Get starting text element for superior content extraction
    start_text = doc.texts[start_idx]
    hierarchy_texts = get_hierarchy_texts_cached(doc)
    start_path = None
    for te, parent, depth, path in hierarchy_texts:
        if te == start_text:
            start_path = path
            break
    
    if start_path is None:
        start_path = []
    
    # For odsek/pismeno levels, get paragraph structure first (with caching)
    # This allows us to reuse it for superior content extraction
    para_structure = None
    if level == 'odsek' or level == 'pismeno':
        parts = identifier.split('.')
        para_num = parts[0]
        
        # Get paragraph boundaries
        para_start_idx, para_end_idx = find_chunk_boundaries_hierarchical(doc, 'paragraph', para_num)
        if para_start_idx is not None and para_end_idx is not None:
            # Check cache first
            cache_key = f"{para_num}_{para_start_idx}_{para_end_idx}"
            if cache_key not in _paragraph_structure_cache:
                para_structure_start = time.time()
                _paragraph_structure_cache[cache_key] = parse_paragraph_structure(doc, para_start_idx, para_end_idx)
                para_structure_time = time.time() - para_structure_start
                log_progress("DEBUG", f"Parsed paragraph {para_num} structure (cached)", para_structure_time)
            else:
                log_progress("DEBUG", f"Using cached paragraph {para_num} structure")
            para_structure = _paragraph_structure_cache[cache_key]
    
    # Extract superior content (pass para_structure for odsek/pismeno)
    superior_start = time.time()
    superior_content = extract_superior_content(doc, level, identifier, start_text, start_path, para_structure)
    superior_time = time.time() - superior_start
    
    # Build nested structure based on level
    structure_start = time.time()
    if level == 'paragraph':
        # Check cache for paragraph structure
        cache_key = f"{identifier}_{start_idx}_{end_idx}"
        if cache_key not in _paragraph_structure_cache:
            _paragraph_structure_cache[cache_key] = parse_paragraph_structure(doc, start_idx, end_idx)
        structure = _paragraph_structure_cache[cache_key]
    elif level == 'odsek':
        # Parse only the odsek, not the whole paragraph
        parts = identifier.split('.')
        para_num = parts[0]
        odsek_num = parts[1] if len(parts) > 1 else None
        if odsek_num:
            structure = parse_odsek_structure(doc, start_idx, end_idx, para_num, odsek_num)
        else:
            # Fallback to paragraph structure if odsek_num not found
            structure = parse_paragraph_structure(doc, start_idx, end_idx)
    elif level == 'pismeno':
        # For pismeno, we also parse only the pismeno structure
        # This is more complex - for now, use odsek structure as base
        parts = identifier.split('.')
        para_num = parts[0]
        odsek_num = parts[1] if len(parts) > 1 else None
        pismeno_letter = parts[2] if len(parts) > 2 else None
        
        if odsek_num and pismeno_letter:
            # Parse the odsek first, then extract only the pismeno
            odsek_structure = parse_odsek_structure(doc, start_idx, end_idx, para_num, odsek_num)
            # Find the specific pismeno in the odsek structure
            pismeno_structure = None
            for subsubsection in odsek_structure.get('content', {}).get('subsubsections', []):
                if subsubsection.get('marker') == f'{pismeno_letter})':
                    pismeno_structure = subsubsection
                    break
            
            if pismeno_structure:
                structure = pismeno_structure
            else:
                # Fallback: create minimal structure
                structure = {
                    'type': 'pismeno',
                    'marker': f'{pismeno_letter})',
                    'content': extract_content_from_texts(doc, doc.texts[start_idx:end_idx])
                }
        else:
            structure = parse_paragraph_structure(doc, start_idx, end_idx)
    else:
        structure = parse_paragraph_structure(doc, start_idx, end_idx)
    
    structure_time = time.time() - structure_start
    
    # Collect references metadata (if enabled)
    refs_start = time.time()
    references_metadata = []
    footnotes_metadata = []
    
    if process_internal_refs or process_external_refs:
        references_metadata = collect_references_metadata(structure)
    
    if process_footnotes:
        footnotes_metadata = collect_footnotes_metadata(structure)
    refs_time = time.time() - refs_start
    
    # Build chunk ID
    if level == 'paragraph':
        chunk_id = f'paragraf-{identifier}'
    elif level == 'odsek':
        chunk_id = f'paragraf-{identifier.replace(".", ".odsek-")}'
    else:  # pismeno
        parts = identifier.split('.')
        chunk_id = f'paragraf-{parts[0]}.odsek-{parts[1]}.pismeno-{parts[2]}'
    
    # Build chunk
    chunk = {
        'chunk_id': chunk_id,
        'level': level,
        'identifier': identifier,
        'superior_content': superior_content,
        'main_content': {
            'structure': structure,
            'references_metadata': references_metadata,
            'footnotes_metadata': footnotes_metadata
        },
        'internal_references': [],  # Will be populated in Pass 2
        'footnote_references': [],  # Will be populated in Pass 2
        'external_references': []   # Will be populated in Pass 2
    }
    
    return chunk


# ============================================================================
# Main Function for Pass 1 Testing
# ============================================================================

def main():
    """
    Main function to test Pass 1 chunking.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Hierarchical chunking v2 - Pass 1: Structure Building')
    parser.add_argument('input_json', type=str, help='Path to input Docling JSON file')
    parser.add_argument('--level', type=str, default='paragraph', 
                       choices=['paragraph', 'odsek', 'pismeno'],
                       help='Chunking level (default: paragraph)')
    parser.add_argument('--identifier', type=str, default=None,
                       help='Specific identifier to chunk (e.g., "5" for paragraph, "5.1" for odsek)')
    parser.add_argument('--parent-identifier', type=str, default=None,
                       help='Parent identifier for chunking all items at level within parent (e.g., "5" for all odseks in paragraph 5)')
    parser.add_argument('--process-internal-refs', action='store_true',
                       help='Extract internal references as metadata')
    parser.add_argument('--process-footnotes', action='store_true',
                       help='Extract footnotes as metadata')
    parser.add_argument('--process-external-refs', action='store_true',
                       help='Extract external references as metadata')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory (default: output)')
    
    args = parser.parse_args()
    
    total_start_time = time.time()
    log_progress("INFO", "=" * 60)
    log_progress("INFO", f"Starting chunking process")
    log_progress("INFO", f"  Level: {args.level}")
    log_progress("INFO", f"  Input: {args.input_json}")
    if args.identifier:
        log_progress("INFO", f"  Identifier: {args.identifier}")
    elif args.parent_identifier:
        log_progress("INFO", f"  Parent identifier: {args.parent_identifier}")
    log_progress("INFO", "=" * 60)
    
    # Validate parameters
    if args.identifier and args.parent_identifier:
        log_progress("ERROR", "--identifier and --parent-identifier cannot be used together")
        log_progress("ERROR", "  Use --identifier for a single chunk, or --parent-identifier for all chunks within a parent")
        return
    
    if args.parent_identifier:
        if args.level == 'paragraph':
            log_progress("ERROR", "--parent-identifier cannot be used with --level paragraph")
            log_progress("ERROR", "  Paragraphs are top-level and do not have a parent")
            return
        elif args.level == 'odsek':
            # Parent must be a paragraph (single number)
            if '.' in args.parent_identifier:
                log_progress("ERROR", f"--parent-identifier '{args.parent_identifier}' is invalid for --level odsek")
                log_progress("ERROR", "  For odsek level, parent must be a paragraph number (e.g., '5')")
                return
        elif args.level == 'pismeno':
            # Parent must be an odsek (two parts) or paragraph (single number)
            parts = args.parent_identifier.split('.')
            if len(parts) > 2:
                log_progress("ERROR", f"--parent-identifier '{args.parent_identifier}' is invalid for --level pismeno")
                log_progress("ERROR", "  For pismeno level, parent must be a paragraph (e.g., '5') or odsek (e.g., '5.1')")
                return
    
    # Load document
    load_start = time.time()
    log_progress("INFO", f"Loading document from {args.input_json}...")
    # Clear caches when loading new document
    clear_hierarchy_cache()
    _paragraph_structure_cache.clear()
    doc = load_docling_document(args.input_json)
    load_time = time.time() - load_start
    log_progress("INFO", f"Document loaded: {doc.name}", load_time)
    
    # Build chunks using Pass 1
    log_progress("INFO", f"Building chunks at {args.level} level...")
    
    if args.identifier:
        # Single chunk - specific identifier
        chunk_start = time.time()
        log_progress("INFO", f"Processing {args.level}: {args.identifier}")
        chunk = build_chunk_structure_pass1(
            doc, args.level, args.identifier,
            process_internal_refs=args.process_internal_refs,
            process_footnotes=args.process_footnotes,
            process_external_refs=args.process_external_refs
        )
        chunk_time = time.time() - chunk_start
        
        if chunk:
            chunks = [chunk]
            log_progress("INFO", f"Successfully created chunk for {args.level} {args.identifier}", chunk_time)
        else:
            log_progress("ERROR", f"Could not build chunk for {args.level} {args.identifier}", chunk_time)
            chunks = []
    elif args.parent_identifier:
        # All chunks at level within parent
        find_start = time.time()
        log_progress("INFO", f"Finding all {args.level}s within parent: {args.parent_identifier}")
        
        if args.level == 'odsek':
            # Find all odseks in the specified paragraph
            identifiers = find_all_odseks_in_paragraph(doc, args.parent_identifier)
        elif args.level == 'pismeno':
            # Check if parent is paragraph or odsek
            parts = args.parent_identifier.split('.')
            if len(parts) == 1:
                # Parent is paragraph - find all pismenos in paragraph
                identifiers = find_all_pismenos_in_paragraph(doc, args.parent_identifier)
            else:
                # Parent is odsek - find all pismenos in odsek
                para_num, odsek_num = parts[0], parts[1]
                identifiers = find_all_pismenos_in_odsek(doc, para_num, odsek_num)
        else:
            log_progress("ERROR", f"Invalid level '{args.level}' with --parent-identifier")
            return
        
        find_time = time.time() - find_start
        
        if not identifiers:
            log_progress("WARNING", f"No {args.level}s found within parent {args.parent_identifier}", find_time)
            chunks = []
        else:
            log_progress("INFO", f"Found {len(identifiers)} {args.level}(s) to process", find_time)
            chunks = []
            chunk_start_time = time.time()
            for idx, identifier in enumerate(identifiers, 1):
                item_start = time.time()
                log_progress("INFO", f"Processing {args.level} {identifier} ({idx}/{len(identifiers)})")
                chunk = build_chunk_structure_pass1(
                    doc, args.level, identifier,
                    process_internal_refs=args.process_internal_refs,
                    process_footnotes=args.process_footnotes,
                    process_external_refs=args.process_external_refs
                )
                item_time = time.time() - item_start
                if chunk:
                    chunks.append(chunk)
                    log_progress("INFO", f"  ✓ Created chunk for {args.level} {identifier}", item_time)
                else:
                    log_progress("WARNING", f"  ✗ Could not build chunk for {args.level} {identifier}", item_time)
            
            total_chunk_time = time.time() - chunk_start_time
            log_progress("INFO", f"Completed processing {len(chunks)}/{len(identifiers)} chunks", total_chunk_time)
    else:
        # All chunks at level in entire document
        find_start = time.time()
        log_progress("INFO", f"Finding all {args.level}s in document...")
        
        if args.level == 'paragraph':
            identifiers = find_all_paragraphs(doc)
        elif args.level == 'odsek':
            identifiers = find_all_odseks_in_document(doc)
        elif args.level == 'pismeno':
            identifiers = find_all_pismenos_in_document(doc)
        else:
            log_progress("ERROR", f"Invalid level '{args.level}'")
            return
        
        find_time = time.time() - find_start
        
        if not identifiers:
            log_progress("WARNING", f"No {args.level}s found in document", find_time)
            chunks = []
        else:
            log_progress("INFO", f"Found {len(identifiers)} {args.level}(s) to process", find_time)
            chunks = []
            chunk_start_time = time.time()
            for idx, identifier in enumerate(identifiers, 1):
                item_start = time.time()
                log_progress("INFO", f"Processing {args.level} {identifier} ({idx}/{len(identifiers)})")
                chunk = build_chunk_structure_pass1(
                    doc, args.level, identifier,
                    process_internal_refs=args.process_internal_refs,
                    process_footnotes=args.process_footnotes,
                    process_external_refs=args.process_external_refs
                )
                item_time = time.time() - item_start
                if chunk:
                    chunks.append(chunk)
                    log_progress("INFO", f"  ✓ Created chunk for {args.level} {identifier}", item_time)
                else:
                    log_progress("WARNING", f"  ✗ Could not build chunk for {args.level} {identifier}", item_time)
            
            total_chunk_time = time.time() - chunk_start_time
            log_progress("INFO", f"Completed processing {len(chunks)}/{len(identifiers)} chunks", total_chunk_time)
    
    if not chunks:
        log_progress("WARNING", "No chunks generated.")
        return
    
    # Save output
    save_start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    base_name = Path(args.input_json).stem.replace('_basic', '').replace('_basic_v0', '')
    if args.identifier:
        output_suffix = f"{args.level}_{args.identifier}"
    elif args.parent_identifier:
        output_suffix = f"{args.level}_parent_{args.parent_identifier}"
    else:
        output_suffix = f"{args.level}_all"
    
    output_json = output_dir / f"{base_name}_chunked_v2_pass1_{output_suffix}.json"
    
    log_progress("INFO", f"Saving {len(chunks)} chunk(s) to JSON...")
    json_start = time.time()
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    json_time = time.time() - json_start
    log_progress("INFO", f"  ✓ JSON saved: {output_json}", json_time)
    
    # Also save to Markdown
    output_md = output_json.with_suffix('.md')
    md_start = time.time()
    log_progress("INFO", f"Saving {len(chunks)} chunk(s) to Markdown...")
    export_chunks_to_markdown_v2(
        chunks,
        str(output_md),
        args.input_json,
        doc.name,
        args.level
    )
    md_time = time.time() - md_start
    log_progress("INFO", f"  ✓ Markdown saved: {output_md}", md_time)
    
    save_time = time.time() - save_start
    
    # Summary
    total_time = time.time() - total_start_time
    log_progress("INFO", "=" * 60)
    log_progress("INFO", "Pass 1 complete!")
    log_progress("INFO", f"  Chunks generated: {len(chunks)}")
    if args.process_internal_refs:
        total_refs = sum(len(chunk['main_content']['references_metadata']) for chunk in chunks)
        log_progress("INFO", f"  Internal references found: {total_refs}")
    if args.process_footnotes:
        total_fns = sum(len(chunk['main_content']['footnotes_metadata']) for chunk in chunks)
        log_progress("INFO", f"  Footnotes found: {total_fns}")
    log_progress("INFO", f"  Total time: {total_time:.2f}s")
    log_progress("INFO", f"  Time breakdown:")
    log_progress("INFO", f"    - Document loading: {load_time:.2f}s")
    log_progress("INFO", f"    - Chunk processing: {total_time - load_time - save_time:.2f}s")
    log_progress("INFO", f"    - File saving: {save_time:.2f}s")
    log_progress("INFO", f"  Output files:")
    log_progress("INFO", f"    JSON:     {output_json}")
    log_progress("INFO", f"    Markdown: {output_md}")
    log_progress("INFO", "=" * 60)


# ============================================================================
# Markdown Generation Functions
# ============================================================================

def format_structure_to_markdown(structure: Dict[str, Any], indent_level: int = 0) -> str:
    """
    Format nested structure to markdown with proper hierarchy display.
    
    Args:
        structure: Structure dictionary (paragraph, odsek, or pismeno)
        indent_level: Current indentation level
        
    Returns:
        Markdown string
    """
    markdown_parts = []
    indent = "  " * indent_level
    
    structure_type = structure.get('type', '')
    marker = structure.get('marker', '')
    title = structure.get('title', '')
    content = structure.get('content', {})
    
    # Format heading based on level
    if structure_type == 'paragraph':
        if title:
            markdown_parts.append(f"{indent}## {title}\n\n")
    elif structure_type == 'odsek':
        if marker:
            markdown_parts.append(f"{indent}### {marker}\n\n")
    elif structure_type == 'pismeno':
        if marker:
            markdown_parts.append(f"{indent}**{marker}** ")
    
    # Add text content
    text = content.get('text', '')
    if text:
        markdown_parts.append(f"{text}\n\n")
    
    # Add subitems (for pismenos with numbered sub-items like 1., 2., etc.)
    subitems = content.get('subitems', [])
    if subitems:
        for subitem in subitems:
            subitem_marker = subitem.get('marker', '')
            subitem_content = subitem.get('content', {})
            subitem_text = subitem_content.get('text', '')
            if subitem_marker and subitem_text:
                markdown_parts.append(f"{indent}  {subitem_marker} {subitem_text}\n\n")
    
    # Add tables
    tables = content.get('tables', [])
    if tables:
        for i, table in enumerate(tables, 1):
            caption = table.get('caption', '')
            if caption:
                markdown_parts.append(f"{indent}**Table {i}: {caption}**\n\n")
            table_md = table.get('markdown', '')
            if table_md:
                markdown_parts.append(f"{indent}{table_md.replace(chr(10), chr(10) + indent)}\n\n")
    
    # Add pictures
    pictures = content.get('pictures', [])
    if pictures:
        for i, pic in enumerate(pictures, 1):
            caption = pic.get('caption', '')
            page = pic.get('page', '')
            if caption:
                markdown_parts.append(f"{indent}**Picture {i}: {caption}**")
            else:
                markdown_parts.append(f"{indent}**Picture {i}**")
            if page:
                markdown_parts.append(f" (Page {page})")
            markdown_parts.append("\n\n")
    
    # Recursively process subsections (odseks)
    subsections = content.get('subsections', [])
    for subsection in subsections:
        subsection_md = format_structure_to_markdown(subsection, indent_level + 1)
        markdown_parts.append(subsection_md)
    
    # Recursively process subsubsections (pismenos)
    subsubsections = content.get('subsubsections', [])
    for subsubsection in subsubsections:
        subsubsection_md = format_structure_to_markdown(subsubsection, indent_level + 1)
        markdown_parts.append(subsubsection_md)
    
    return "".join(markdown_parts)


def format_chunk_to_markdown_v2(chunk: Dict[str, Any]) -> str:
    """
    Convert a chunk (v2 format) to markdown format with nested hierarchy.
    
    Args:
        chunk: Chunk dictionary with nested structure
        
    Returns:
        Complete markdown string
    """
    markdown_parts = []
    
    # Main content heading
    chunk_id = chunk.get('chunk_id', '')
    identifier = chunk.get('identifier', '')
    level = chunk.get('level', 'paragraph')
    
    if level == 'paragraph':
        title = f"§ {identifier}"
    elif level == 'odsek':
        parts = identifier.split('.')
        title = f"§ {parts[0]}, odsek {parts[1]}"
    else:  # pismeno
        parts = identifier.split('.')
        title = f"§ {parts[0]}, odsek {parts[1]}, písmeno {parts[2]})"
    
    markdown_parts.append(f"## {title}\n\n")
    
    # Superior content
    superior_content = chunk.get('superior_content', {})
    if superior_content.get('part'):
        part = superior_content['part']
        part_title = part.get('title', '')
        if part_title:
            markdown_parts.append(f"**Part:** {part_title}\n\n")
            part_content = part.get('content', {})
            part_text = part_content.get('text', '')
            if part_text:
                # Show full part content (usually just the title)
                markdown_parts.append(f"*{part_text}*\n\n")
    
    # Paragraph introductory text (for odsek and pismeno levels)
    if superior_content.get('paragraph_intro'):
        para_intro = superior_content['paragraph_intro']
        para_intro_text = para_intro.get('text', '')
        if para_intro_text:
            markdown_parts.append(f"**Paragraph:** {para_intro_text}\n\n")
    
    # Main content structure
    main_content = chunk.get('main_content', {})
    structure = main_content.get('structure', {})
    
    if structure:
        structure_md = format_structure_to_markdown(structure, indent_level=0)
        markdown_parts.append(structure_md)
    
    # Internal references (if resolved in Pass 2)
    internal_refs = chunk.get('internal_references', [])
    if internal_refs:
        markdown_parts.append("\n---\n")
        markdown_parts.append("\n## Referenced Sections\n\n")
        markdown_parts.append("*The following sections are referenced in the text above. Their full text is provided for context.*\n\n")
        
        for ref in internal_refs:
            ref_text = ref.get('reference_text', '')
            hyperlink = ref.get('hyperlink', '')
            occurrences = ref.get('occurrences', 1)
            content = ref.get('content', {})
            
            # Format reference heading
            if ref_text and '§' in ref_text:
                heading = ref_text
            else:
                heading = hyperlink.replace('#paragraf-', '§ ').replace('.odsek-', ', odsek ').replace('.pismeno-', ', písmeno ')
            
            markdown_parts.append(f"### {heading}\n\n")
            
            if isinstance(content, dict):
                ref_content_text = content.get('text', '')
                if ref_content_text:
                    markdown_parts.append(ref_content_text)
                    markdown_parts.append("\n\n")
                else:
                    markdown_parts.append("*[Referenced text not available]*\n\n")
            else:
                markdown_parts.append("*[Referenced text not available]*\n\n")
            
            if occurrences > 1:
                markdown_parts.append(f"*Referenced {occurrences} times in the text.*\n\n")
    
    # Footnote references (if resolved in Pass 2)
    footnote_refs = chunk.get('footnote_references', [])
    if footnote_refs:
        markdown_parts.append("\n---\n")
        markdown_parts.append("\n## Footnotes\n\n")
        
        for ref in footnote_refs:
            fn_id = ref.get('footnote_id', '')
            fn_content = ref.get('content', '')
            occurrences = ref.get('occurrences', 1)
            
            markdown_parts.append(f"### Footnote {fn_id}\n\n")
            
            if fn_content:
                markdown_parts.append(f"{fn_content}\n\n")
            else:
                markdown_parts.append("*[Footnote text not available]*\n\n")
            
            if occurrences > 1:
                markdown_parts.append(f"*Referenced {occurrences} times in the text.*\n\n")
    
    # External references (placeholder)
    external_refs = chunk.get('external_references', [])
    if external_refs:
        markdown_parts.append("\n---\n")
        markdown_parts.append("\n## External References\n\n")
        markdown_parts.append("*The following external references are referenced in the text above.*\n\n")
        
        for ref in external_refs:
            ref_text = ref.get('reference_text', '')
            markdown_parts.append(f"### {ref_text}\n\n")
            markdown_parts.append("*[External reference not available]*\n\n")
    
    return "".join(markdown_parts)


def export_chunks_to_markdown_v2(chunks: List[Dict[str, Any]], output_path: str, source_file: str, 
                                 document_name: str, chunking_level: str) -> None:
    """
    Export chunks (v2 format) to markdown format.
    
    Args:
        chunks: List of chunk dictionaries
        output_path: Path to output markdown file
        source_file: Source file path
        document_name: Document name
        chunking_level: Chunking level used
    """
    markdown_lines = []
    
    # Document header
    markdown_lines.append("# Legal Document Chunks (Hierarchical Structure)\n\n")
    markdown_lines.append(f"*Generated from: {source_file}*\n")
    markdown_lines.append(f"*Document: {document_name}*\n")
    markdown_lines.append(f"*Chunking method: hierarchy_based_v2 (2-pass approach)*\n")
    markdown_lines.append(f"*Chunking level: {chunking_level}*\n")
    markdown_lines.append(f"*Total chunks: {len(chunks)}*\n\n")
    markdown_lines.append("---\n\n")
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        if i > 0:
            markdown_lines.append("\n\n---\n\n")
        
        chunk_id = chunk.get('chunk_id', '?')
        chunk_md = format_chunk_to_markdown_v2(chunk)
        markdown_lines.append(chunk_md)
    
    full_markdown = "".join(markdown_lines)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_markdown)


if __name__ == '__main__':
    main()

