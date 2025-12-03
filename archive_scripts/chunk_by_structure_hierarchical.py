#!/usr/bin/env python3
"""
Hierarchical structure-based chunking for Docling documents.

Chunks documents at configurable levels (paragraph, odsek, pismeno) using
hierarchical traversal (body → children → recursive text parent-child relationships).
Extracts complete content including tables and pictures, and resolves all internal
and external references.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from docling_core.types.doc import DoclingDocument


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


def get_all_children(doc: DoclingDocument, element: Any) -> List[Any]:
    """
    Recursively get all children of an element.
    
    Args:
        doc: DoclingDocument
        element: Element to get children from
        
    Returns:
        List of all child elements (resolved)
    """
    children = []
    
    if not hasattr(element, 'children') or not element.children:
        return children
    
    for ref_item in element.children:
        resolved = resolve_ref_item(doc, ref_item)
        if resolved:
            children.append(resolved)
            # Recursively get children of children
            children.extend(get_all_children(doc, resolved))
    
    return children


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
    hierarchy_texts = traverse_hierarchy_recursive(doc, doc.body)
    
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


def find_text_in_hierarchy(doc: DoclingDocument, text_pattern: str, no_hyperlink: bool = False, exact_match: bool = True) -> Optional[Tuple[Any, Any, int, List[str]]]:
    """
    Find text element in hierarchy by content.
    
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
    
    hierarchy_texts = traverse_hierarchy_recursive(doc, doc.body)
    
    for text_element, parent, depth, path in hierarchy_texts:
        text = getattr(text_element, 'text', '').strip()
        hyperlink = getattr(text_element, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        
        # Check if text matches
        # Normalize spaces (handle both regular and non-breaking spaces)
        normalized_text = text.replace('\xa0', ' ')
        normalized_pattern = text_pattern.replace('\xa0', ' ')
        
        if exact_match:
            matches = normalized_text == normalized_pattern
        else:
            matches = normalized_text.startswith(normalized_pattern)
        
        if matches:
            if no_hyperlink and hyperlink_str:
                continue
            if not no_hyperlink or not hyperlink_str:
                return (text_element, parent, depth, path)
    
    return None


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
    hierarchy_texts = traverse_hierarchy_recursive(doc, doc.body)
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
        # Find odsek marker in hierarchy - must be descendant of paragraph
        odsek_marker = f'({odsek})'
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
    hierarchy_texts = traverse_hierarchy_recursive(doc, doc.body)
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
            if normalized_content in ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)'] and normalized_content != f'{pismeno})':
                return True
            if normalized_content.startswith('(') and normalized_content[1:].rstrip(')').isdigit():
                return True
            if normalized_content.startswith('§ ') and not hyperlink_str:
                match = re.match(r'§\s+(\d+[a-zA-Z]*)', normalized_content)
                if match and match.group(1) != para_num:
                    return True
        elif odsek:
            # End at next odsek or paragraph
            if normalized_content.startswith('(') and normalized_content[1:].rstrip(')').isdigit():
                next_num = normalized_content[1:].rstrip(')')
                if next_num != odsek:
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


def extract_text_content(text_items: List[Any]) -> str:
    """
    Extract and format text content from text items.
    
    Args:
        text_items: List of text elements
        
    Returns:
        Formatted text content as markdown
    """
    full_text = []
    
    for item in text_items:
        text = getattr(item, 'text', '').strip()
        hyperlink = getattr(item, 'hyperlink', '')
        label = getattr(item, 'label', '')
        
        # Convert hyperlink to string if needed
        hyperlink_str = str(hyperlink) if hyperlink else ''
        
        # Skip navigation elements
        if (text and 
            text not in ['plus', 'button-close', 'button-search', 'button-download', 
                        'button-print', 'button-history', 'button-content'] and
            label != 'caption'):
            
            # Handle structural markers
            if text.strip() in ['(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)', '(8)', '(9)', '(10)', 
                               '(11)', '(12)', '(13)', '(14)', '(15)', '(16)']:
                full_text.append(f'\n\n### {text}\n\n')
            elif text.strip() in ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)']:
                full_text.append(f'\n**{text}** ')
            elif text.strip() in ['1.', '2.', '3.', '4.', '5.']:
                full_text.append(f'    {text} ')
            # Handle references
            elif hyperlink_str and 'paragraf-' in hyperlink_str and 'poznamky' not in hyperlink_str:
                full_text.append(f' [{text}]({hyperlink_str})')
            elif hyperlink_str and 'poznamky.poznamka' in hyperlink_str:
                full_text.append(f'<sup>{text}</sup>)')
            else:
                # Regular text
                if text and len(text) > 1:
                    if full_text and not full_text[-1].endswith(' ') and not full_text[-1].endswith('\n'):
                        if text[0] not in ['.', ',', ';', ':', '!', '?', ')', ']', '}', '(', '[', '{']:
                            full_text.append(' ')
                    full_text.append(text)
    
    text_content = ''.join(full_text)
    text_content = re.sub(r' +', ' ', text_content)
    text_content = re.sub(r' \n', '\n', text_content)
    text_content = re.sub(r'\n{3,}', '\n\n', text_content)
    
    return text_content.strip()


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


def extract_chunk_content(doc: DoclingDocument, start_idx: int, end_idx: int) -> Dict[str, Any]:
    """
    Extract all content (text, tables, pictures) from a chunk region.
    
    Args:
        doc: DoclingDocument
        start_idx: Start index in doc.texts
        end_idx: End index in doc.texts
        
    Returns:
        Dictionary with text, tables, and pictures
    """
    # Extract text items in range
    text_items = doc.texts[start_idx:end_idx]
    text_content = extract_text_content(text_items)
    
    # Find tables and pictures that belong to this chunk
    tables = []
    pictures = []
    
    # Collect hyperlinks from text items to find table/picture references
    for item in text_items:
        hyperlink = getattr(item, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        if hyperlink_str:
            if '#/tables/' in hyperlink_str:
                table_idx = int(hyperlink_str.split('/')[-1])
                if table_idx < len(doc.tables):
                    table = doc.tables[table_idx]
                    table_md = format_table_as_markdown(table, doc)
                    if table_md:
                        tables.append({
                            'index': table_idx,
                            'markdown': table_md,
                            'caption': getattr(table, 'caption_text', None)
                        })
            elif '#/pictures/' in hyperlink_str:
                pic_idx = int(hyperlink_str.split('/')[-1])
                if pic_idx < len(doc.pictures):
                    picture = doc.pictures[pic_idx]
                    pic_info = format_picture_reference(picture)
                    pictures.append({
                        'index': pic_idx,
                        **pic_info
                    })
    
    return {
        'text': text_content,
        'tables': tables,
        'pictures': pictures
    }


def parse_reference_hyperlink(hyperlink: str) -> Dict[str, Optional[str]]:
    """
    Parse a hyperlink to extract paragraph number, odsek, and pismeno.
    
    Args:
        hyperlink: Hyperlink string (e.g., "#paragraf-47.odsek-1.pismeno-a")
        
    Returns:
        Dictionary with paragraph_number, odsek, pismeno
    """
    if not hyperlink or not hyperlink.startswith('#paragraf-'):
        return {'paragraph_number': None, 'odsek': None, 'pismeno': None}
    
    # Remove #paragraf- prefix
    path = hyperlink.replace('#paragraf-', '')
    
    # Split by dots
    parts = path.split('.')
    
    paragraph_number = parts[0] if parts else None
    odsek = None
    pismeno = None
    
    for part in parts[1:]:
        if part.startswith('odsek-'):
            odsek = part.replace('odsek-', '')
        elif part.startswith('pismeno-'):
            pismeno = part.replace('pismeno-', '')
    
    return {
        'paragraph_number': paragraph_number,
        'odsek': odsek,
        'pismeno': pismeno
    }


def resolve_internal_reference(doc: DoclingDocument, hyperlink: str) -> Dict[str, Any]:
    """
    Extract full content of a referenced section using hierarchical approach.
    
    Args:
        doc: DoclingDocument
        hyperlink: Hyperlink to the referenced section
        
    Returns:
        Dictionary with content (text, tables, pictures)
    """
    ref_path = parse_reference_hyperlink(hyperlink)
    para_num = ref_path['paragraph_number']
    odsek = ref_path['odsek']
    pismeno = ref_path['pismeno']
    
    if not para_num:
        return {'text': '', 'tables': [], 'pictures': []}
    
    # Determine level and identifier
    if pismeno:
        level = 'pismeno'
        identifier = f"{para_num}.{odsek}.{pismeno}"
    elif odsek:
        level = 'odsek'
        identifier = f"{para_num}.{odsek}"
    else:
        level = 'paragraph'
        identifier = para_num
    
    # Find boundaries using hierarchical approach
    start_idx, end_idx = find_chunk_boundaries_hierarchical(doc, level, identifier)
    
    if start_idx is None or end_idx is None:
        return {'text': '', 'tables': [], 'pictures': []}
    
    # Extract content
    return extract_chunk_content(doc, start_idx, end_idx)


def find_footnote_section(doc: DoclingDocument) -> Optional[int]:
    """
    Locate the Poznámky (footnotes) section.
    
    Args:
        doc: DoclingDocument
        
    Returns:
        Start index of footnotes section or None
    """
    if not hasattr(doc, 'texts') or not doc.texts:
        return None
    
    for i, text_item in enumerate(doc.texts):
        text = getattr(text_item, 'text', '')
        hyperlink = getattr(text_item, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        
        if 'Poznámky' in text or hyperlink_str == '#poznamky':
            return i
    
    return None


def resolve_footnote_by_id(doc: DoclingDocument, footnote_id: str) -> str:
    """
    Extract footnote content by footnote ID using pure hierarchical traversal.
    
    Strategy:
    1. Find Poznámky section using hierarchy
    2. Find footnote definition "footnote_id)" WITHOUT hyperlink in Poznámky section
    3. Collect all descendant texts until next footnote marker
    
    Args:
        doc: DoclingDocument
        footnote_id: Footnote identifier (e.g., "1", "136f", "7a")
        
    Returns:
        Footnote text content
    """
    if not hasattr(doc, 'texts') or not doc.texts:
        return ""
    
    if not hasattr(doc, 'body') or not doc.body:
        return ""
    
    # Step 1: Find Poznámky section using hierarchy
    poznamky_text_result = find_text_in_hierarchy(doc, 'Poznámky', no_hyperlink=False)
    if poznamky_text_result is None:
        # Fallback: try finding by hyperlink in flat structure
        for i, text_item in enumerate(doc.texts):
            hyperlink = getattr(text_item, 'hyperlink', '')
            hyperlink_str = str(hyperlink) if hyperlink else ''
            if hyperlink_str == '#poznamky':
                poznamky_start = i
                break
        else:
            return ""
    else:
        # Get index of Poznámky text in doc.texts
        poznamky_text, poznamky_parent, poznamky_depth, poznamky_path = poznamky_text_result
        poznamky_start = None
        for i, text_item in enumerate(doc.texts):
            if text_item == poznamky_text:
                poznamky_start = i
                break
        if poznamky_start is None:
            return ""
    
    # Step 2: Find footnote definition "footnote_id)" WITHOUT hyperlink
    # This is the definition in Poznámky section, not a reference in the text
    footnote_marker_idx = None
    
    # Search from Poznámky to end of document (footnotes can be far from Poznámky header)
    for i in range(poznamky_start, len(doc.texts)):
        text_item = doc.texts[i]
        hyperlink = getattr(text_item, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        text = getattr(text_item, 'text', '').strip()
        
        # Look for "footnote_id)" WITHOUT hyperlink (this is the definition in Poznámky)
        if text == f'{footnote_id})' and not hyperlink_str:
            footnote_marker_idx = i
            break
    
    if footnote_marker_idx is None:
        return ""
    
    # Step 3: Collect footnote content from text AFTER "footnote_id)"
    content_start_idx = footnote_marker_idx + 1
    
    # Collect until next footnote marker (pattern "N)" without hyperlink) or section break
    footnote_texts = []
    for i in range(content_start_idx, min(content_start_idx + 500, len(doc.texts))):
        text_item = doc.texts[i]
        text = getattr(text_item, 'text', '').strip()
        hyperlink = getattr(text_item, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        label = getattr(text_item, 'label', '')
        
        # Stop at next footnote marker (pattern "N)" without hyperlink where N != footnote_id)
        if text and text.endswith(')') and not hyperlink_str:
            # Check if it's a footnote marker (single digit/letter followed by paren)
            match = re.match(r'^(\d+[a-zA-Z]*)\)$', text)
            if match:
                next_footnote_id = match.group(1)
                if next_footnote_id != footnote_id:
                    # This is a different footnote, stop
                    break
        
        # Stop at major section break
        normalized_text = text.replace('\xa0', ' ') if text else ''
        if normalized_text and normalized_text.startswith('§ ') and i > content_start_idx + 5:
            break
        
        # Collect text (skip navigation elements)
        if (text and 
            text not in ['plus', 'button-close', 'button-search', 'button-download', 
                        'button-print', 'button-history', 'button-content'] and
            label != 'caption'):
            footnote_texts.append(text)
    
    footnote_content = ' '.join(footnote_texts)
    footnote_content = re.sub(r' +', ' ', footnote_content)
    return footnote_content.strip()


def extract_internal_references(text_items: List[Any]) -> List[Dict[str, Any]]:
    """
    Find all internal references in a chunk.
    
    Args:
        text_items: List of text elements in the chunk
        
    Returns:
        List of unique internal references
    """
    internal_refs = {}
    
    for item in text_items:
        hyperlink = getattr(item, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        text = getattr(item, 'text', '').strip()
        
        if hyperlink_str and 'paragraf-' in hyperlink_str and 'poznamky' not in hyperlink_str:
            if hyperlink_str not in internal_refs:
                internal_refs[hyperlink_str] = {
                    'hyperlink': hyperlink_str,
                    'text': text,
                    'occurrences': 1
                }
            else:
                internal_refs[hyperlink_str]['occurrences'] += 1
    
    return list(internal_refs.values())


def extract_footnote_references(text_items: List[Any]) -> List[Dict[str, Any]]:
    """
    Find all footnote references in a chunk.
    
    Args:
        text_items: List of text elements in the chunk
        
    Returns:
        List of unique footnote references
    """
    footnote_refs = {}
    
    for item in text_items:
        hyperlink = getattr(item, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        text = getattr(item, 'text', '').strip()
        
        if hyperlink_str and 'poznamky.poznamka' in hyperlink_str:
            # Extract footnote ID
            if 'poznamka-' in hyperlink_str:
                fn_id = hyperlink_str.split('poznamka-')[-1]
            else:
                continue
            
            # Only count the footnote number, not the closing paren
            if text != ')':
                if hyperlink_str not in footnote_refs:
                    footnote_refs[hyperlink_str] = {
                        'hyperlink': hyperlink_str,
                        'footnote_id': fn_id,
                        'text': text,
                        'occurrences': 1
                    }
                else:
                    footnote_refs[hyperlink_str]['occurrences'] += 1
    
    return list(footnote_refs.values())


def extract_external_references(text_items: List[Any]) -> List[Dict[str, Any]]:
    """
    Find all external references in a chunk.
    
    PLACEHOLDER: This function is a placeholder for future implementation.
    External references are references to other documents or external resources.
    
    Args:
        text_items: List of text elements in the chunk
        
    Returns:
        Empty list (placeholder implementation)
    """
    # TODO: Implement external reference extraction
    return []


def resolve_external_reference(doc: DoclingDocument, reference: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve an external reference to its content.
    
    PLACEHOLDER: This function is a placeholder for future implementation.
    External references are references to other documents or external resources.
    
    Args:
        doc: DoclingDocument
        reference: External reference dictionary
        
    Returns:
        Empty dictionary (placeholder implementation)
    """
    # TODO: Implement external reference resolution
    return {}


def build_chunk(doc: DoclingDocument, level: str, identifier: str, main_content: Dict[str, Any], 
                internal_refs: List[Dict[str, Any]], footnote_refs: List[Dict[str, Any]],
                external_refs: List[Dict[str, Any]] = None,
                process_internal_refs: bool = False, process_footnotes: bool = False, 
                process_external_refs: bool = False) -> Dict[str, Any]:
    """
    Assemble final chunk structure with main content and resolved references.
    
    Args:
        doc: DoclingDocument
        level: Chunking level
        identifier: Section identifier
        main_content: Main chunk content
        internal_refs: List of internal references
        footnote_refs: List of footnote references
        external_refs: List of external references (optional)
        process_internal_refs: If True, resolve internal references
        process_footnotes: If True, resolve footnote references
        process_external_refs: If True, resolve external references (placeholder)
        
    Returns:
        Complete chunk dictionary
    """
    # Build chunk ID
    if level == 'paragraph':
        chunk_id = f'paragraf-{identifier}'
    elif level == 'odsek':
        chunk_id = f'paragraf-{identifier.replace(".", ".odsek-")}'
    else:  # pismeno
        parts = identifier.split('.')
        chunk_id = f'paragraf-{parts[0]}.odsek-{parts[1]}.pismeno-{parts[2]}'
    
    # Resolve internal references (only if enabled)
    resolved_internal_refs = []
    if process_internal_refs:
        for ref in internal_refs:
            hyperlink = ref['hyperlink']
            print(f"  Resolving internal reference: {hyperlink}")
            resolved_content = resolve_internal_reference(doc, hyperlink)
            resolved_internal_refs.append({
                'hyperlink': hyperlink,
                'reference_text': ref['text'],
                'occurrences': ref['occurrences'],
                'content': resolved_content
            })
    
    # Resolve footnote references (only if enabled)
    resolved_footnote_refs = []
    if process_footnotes:
        for ref in footnote_refs:
            footnote_id = ref['footnote_id']
            print(f"  Resolving footnote: {footnote_id}")
            footnote_content = resolve_footnote_by_id(doc, footnote_id)
            resolved_footnote_refs.append({
                'footnote_id': footnote_id,
                'hyperlink': ref['hyperlink'],
                'reference_text': ref['text'],
                'occurrences': ref['occurrences'],
                'content': footnote_content
            })
    
    # Resolve external references (only if enabled - placeholder)
    resolved_external_refs = []
    if process_external_refs:
        if external_refs is None:
            external_refs = []
        for ref in external_refs:
            print(f"  Resolving external reference: {ref}")
            resolved_content = resolve_external_reference(doc, ref)
            resolved_external_refs.append({
                **ref,
                'content': resolved_content
            })
    
    return {
        'chunk_id': chunk_id,
        'level': level,
        'identifier': identifier,
        'main_content': main_content,
        'internal_references': resolved_internal_refs,
        'footnote_references': resolved_footnote_refs,
        'external_references': resolved_external_refs
    }


def chunk_document_hierarchical(doc: DoclingDocument, level: str = 'paragraph', specific_identifier: Optional[str] = None,
                                process_internal_refs: bool = False, process_footnotes: bool = False, 
                                process_external_refs: bool = False) -> List[Dict[str, Any]]:
    """
    Main chunking function using hierarchical traversal.
    
    Args:
        doc: DoclingDocument
        level: Chunking level ('paragraph', 'odsek', 'pismeno')
        specific_identifier: Optional specific section to chunk (e.g., '50' for paragraph, '50.1' for odsek)
        process_internal_refs: If True, extract and resolve internal references
        process_footnotes: If True, extract and resolve footnote references
        process_external_refs: If True, extract and resolve external references (placeholder)
        
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    
    if not hasattr(doc, 'texts') or not doc.texts:
        return chunks
    
    if not hasattr(doc, 'body') or not doc.body:
        return chunks
    
    if specific_identifier:
        # Chunk only the specified section
        start_idx, end_idx = find_chunk_boundaries_hierarchical(doc, level, specific_identifier)
        if start_idx is not None and end_idx is not None:
            text_items = doc.texts[start_idx:end_idx]
            main_content = extract_chunk_content(doc, start_idx, end_idx)
            # Conditionally extract references
            internal_refs = extract_internal_references(text_items) if process_internal_refs else []
            footnote_refs = extract_footnote_references(text_items) if process_footnotes else []
            external_refs = extract_external_references(text_items) if process_external_refs else []
            chunk = build_chunk(doc, level, specific_identifier, main_content, internal_refs, footnote_refs, 
                              external_refs, process_internal_refs, process_footnotes, process_external_refs)
            chunks.append(chunk)
    else:
        # Chunk all sections at the specified level using hierarchy
        ordered_texts, text_to_index = build_text_hierarchy_map(doc)
        
        if level == 'paragraph':
            # Find all paragraphs in hierarchy
            paragraph_numbers = []
            hierarchy_texts = traverse_hierarchy_recursive(doc, doc.body)
            
            for text_element, parent, depth, path in hierarchy_texts:
                text = getattr(text_element, 'text', '').strip()
                hyperlink = getattr(text_element, 'hyperlink', '')
                hyperlink_str = str(hyperlink) if hyperlink else ''
                
                # Normalize spaces (handle both regular and non-breaking spaces)
                normalized_text = text.replace('\xa0', ' ')
                if normalized_text.startswith('§ ') and not hyperlink_str:
                    match = re.match(r'§\s+(\d+[a-zA-Z]*)', normalized_text)
                    if match:
                        para_num_full = match.group(1)
                        # Check if text starts with "§ {para_num_full}" (may include title)
                        if normalized_text.startswith(f'§ {para_num_full}') and para_num_full not in paragraph_numbers:
                            paragraph_numbers.append(para_num_full)
            
            print(f'Found {len(paragraph_numbers)} paragraphs to chunk')
            for para_num in paragraph_numbers:
                print(f'Processing § {para_num}...')
                start_idx, end_idx = find_chunk_boundaries_hierarchical(doc, level, para_num)
                if start_idx is not None and end_idx is not None:
                    text_items = doc.texts[start_idx:end_idx]
                    main_content = extract_chunk_content(doc, start_idx, end_idx)
                    # Conditionally extract references
                    internal_refs = extract_internal_references(text_items) if process_internal_refs else []
                    footnote_refs = extract_footnote_references(text_items) if process_footnotes else []
                    external_refs = extract_external_references(text_items) if process_external_refs else []
                    chunk = build_chunk(doc, level, para_num, main_content, internal_refs, footnote_refs, 
                                      external_refs, process_internal_refs, process_footnotes, process_external_refs)
                    chunks.append(chunk)
        
        elif level == 'odsek':
            # Find all paragraphs first, then their odseks
            paragraph_numbers = []
            hierarchy_texts = traverse_hierarchy_recursive(doc, doc.body)
            
            for text_element, parent, depth, path in hierarchy_texts:
                text = getattr(text_element, 'text', '').strip()
                hyperlink = getattr(text_element, 'hyperlink', '')
                hyperlink_str = str(hyperlink) if hyperlink else ''
                
                # Normalize spaces (handle both regular and non-breaking spaces)
                normalized_text = text.replace('\xa0', ' ')
                if normalized_text.startswith('§ ') and not hyperlink_str:
                    match = re.match(r'§\s+(\d+[a-zA-Z]*)', normalized_text)
                    if match:
                        para_num_full = match.group(1)
                        # Check if text starts with "§ {para_num_full}" (may include title)
                        if normalized_text.startswith(f'§ {para_num_full}') and para_num_full not in paragraph_numbers:
                            paragraph_numbers.append(para_num_full)
            
            for para_num in paragraph_numbers:
                # Find all odseks in this paragraph using hierarchy
                para_start, _ = find_chunk_boundaries_hierarchical(doc, 'paragraph', para_num)
                if para_start is not None:
                    odsek_numbers = set()
                    for i in range(para_start, min(para_start + 500, len(doc.texts))):
                        text = getattr(doc.texts[i], 'text', '').strip()
                        normalized_text = text.replace('\xa0', ' ')
                        if normalized_text.startswith('(') and normalized_text[1:].rstrip(')').isdigit():
                            odsek_num = normalized_text[1:].rstrip(')')
                            odsek_numbers.add(odsek_num)
                        elif normalized_text.startswith('§ '):
                            match = re.match(r'§\s+(\d+[a-zA-Z]*)', normalized_text)
                            if match:
                                next_para_full = match.group(1)
                                if next_para_full != para_num:
                                    break
                    
                    odsek_numbers = sorted(odsek_numbers, key=lambda x: int(x))
                    for odsek_num in odsek_numbers:
                        identifier = f"{para_num}.{odsek_num}"
                        print(f'Processing § {para_num}, odsek {odsek_num}...')
                        start_idx, end_idx = find_chunk_boundaries_hierarchical(doc, level, identifier)
                        if start_idx is not None and end_idx is not None:
                            text_items = doc.texts[start_idx:end_idx]
                            main_content = extract_chunk_content(doc, start_idx, end_idx)
                            # Conditionally extract references
                            internal_refs = extract_internal_references(text_items) if process_internal_refs else []
                            footnote_refs = extract_footnote_references(text_items) if process_footnotes else []
                            external_refs = extract_external_references(text_items) if process_external_refs else []
                            chunk = build_chunk(doc, level, identifier, main_content, internal_refs, footnote_refs, 
                                              external_refs, process_internal_refs, process_footnotes, process_external_refs)
                            chunks.append(chunk)
        
        elif level == 'pismeno':
            # Find all paragraphs, then odseks, then pismenos
            paragraph_numbers = []
            hierarchy_texts = traverse_hierarchy_recursive(doc, doc.body)
            
            for text_element, parent, depth, path in hierarchy_texts:
                text = getattr(text_element, 'text', '').strip()
                hyperlink = getattr(text_element, 'hyperlink', '')
                hyperlink_str = str(hyperlink) if hyperlink else ''
                
                # Normalize spaces (handle both regular and non-breaking spaces)
                normalized_text = text.replace('\xa0', ' ')
                if normalized_text.startswith('§ ') and not hyperlink_str:
                    match = re.match(r'§\s+(\d+[a-zA-Z]*)', normalized_text)
                    if match:
                        para_num_full = match.group(1)
                        # Check if text starts with "§ {para_num_full}" (may include title)
                        if normalized_text.startswith(f'§ {para_num_full}') and para_num_full not in paragraph_numbers:
                            paragraph_numbers.append(para_num_full)
            
            for para_num in paragraph_numbers:
                para_start, _ = find_chunk_boundaries_hierarchical(doc, 'paragraph', para_num)
                if para_start is not None:
                    # Find all odseks
                    odsek_numbers = set()
                    for i in range(para_start, min(para_start + 500, len(doc.texts))):
                        text = getattr(doc.texts[i], 'text', '').strip()
                        normalized_text = text.replace('\xa0', ' ')
                        if normalized_text.startswith('(') and normalized_text[1:].rstrip(')').isdigit():
                            odsek_num = normalized_text[1:].rstrip(')')
                            odsek_numbers.add(odsek_num)
                        elif normalized_text.startswith('§ '):
                            match = re.match(r'§\s+(\d+[a-zA-Z]*)', normalized_text)
                            if match:
                                next_para_full = match.group(1)
                                if next_para_full != para_num:
                                    break
                    
                    for odsek_num in sorted(odsek_numbers, key=lambda x: int(x)):
                        odsek_start, _ = find_chunk_boundaries_hierarchical(doc, 'odsek', f"{para_num}.{odsek_num}")
                        if odsek_start is not None:
                            # Find all pismenos in this odsek
                            pismeno_letters = set()
                            for i in range(odsek_start, min(odsek_start + 300, len(doc.texts))):
                                text = getattr(doc.texts[i], 'text', '').strip()
                                normalized_text = text.replace('\xa0', ' ')
                                if normalized_text in ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)']:
                                    pismeno_letter = normalized_text[0]
                                    pismeno_letters.add(pismeno_letter)
                                elif normalized_text.startswith('(') and normalized_text[1:].rstrip(')').isdigit():
                                    break
                                elif normalized_text.startswith('§ '):
                                    match = re.match(r'§\s+(\d+[a-zA-Z]*)', normalized_text)
                                    if match:
                                        next_para_full = match.group(1)
                                        if next_para_full != para_num:
                                            break
                            
                            for pismeno_letter in sorted(pismeno_letters):
                                identifier = f"{para_num}.{odsek_num}.{pismeno_letter}"
                                print(f'Processing § {para_num}, odsek {odsek_num}, pismeno {pismeno_letter}...')
                                start_idx, end_idx = find_chunk_boundaries_hierarchical(doc, level, identifier)
                                if start_idx is not None and end_idx is not None:
                                    text_items = doc.texts[start_idx:end_idx]
                                    main_content = extract_chunk_content(doc, start_idx, end_idx)
                                    # Conditionally extract references
                                    internal_refs = extract_internal_references(text_items) if process_internal_refs else []
                                    footnote_refs = extract_footnote_references(text_items) if process_footnotes else []
                                    external_refs = extract_external_references(text_items) if process_external_refs else []
                                    chunk = build_chunk(doc, level, identifier, main_content, internal_refs, footnote_refs, 
                                                      external_refs, process_internal_refs, process_footnotes, process_external_refs)
                                    chunks.append(chunk)
    
    return chunks


def format_chunk_to_markdown(chunk: Dict[str, Any]) -> str:
    """
    Convert a chunk to markdown format.
    
    Args:
        chunk: Chunk dictionary
        
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
    
    # Main content text
    main_content = chunk.get('main_content', {})
    text = main_content.get('text', '')
    if text:
        markdown_parts.append(text)
        if not text.endswith('\n'):
            markdown_parts.append("\n")
    
    # Tables
    tables = main_content.get('tables', [])
    if tables:
        markdown_parts.append("\n### Tables\n\n")
        for i, table in enumerate(tables, 1):
            caption = table.get('caption', '')
            if caption:
                markdown_parts.append(f"**Table {i}: {caption}**\n\n")
            table_md = table.get('markdown', '')
            if table_md:
                markdown_parts.append(table_md)
                markdown_parts.append("\n\n")
    
    # Pictures
    pictures = main_content.get('pictures', [])
    if pictures:
        markdown_parts.append("\n### Pictures\n\n")
        for i, pic in enumerate(pictures, 1):
            caption = pic.get('caption', '')
            page = pic.get('page', '')
            if caption:
                markdown_parts.append(f"**Picture {i}: {caption}**")
            else:
                markdown_parts.append(f"**Picture {i}**")
            if page:
                markdown_parts.append(f" (Page {page})")
            markdown_parts.append("\n\n")
    
    # Internal references
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
            ref_content_text = content.get('text', '')
            
            # Format reference heading
            if ref_text and '§' in ref_text:
                heading = ref_text
            else:
                # Parse from hyperlink
                heading = hyperlink.replace('#paragraf-', '§ ').replace('.odsek-', ', odsek ').replace('.pismeno-', ', písmeno ')
            
            markdown_parts.append(f"### {heading}\n\n")
            
            if ref_content_text:
                markdown_parts.append(ref_content_text)
                markdown_parts.append("\n\n")
            else:
                markdown_parts.append("*[Referenced text not available]*\n\n")
            
            # Include tables and pictures from referenced content
            ref_tables = content.get('tables', [])
            ref_pictures = content.get('pictures', [])
            
            if ref_tables:
                markdown_parts.append("**Tables in referenced section:**\n\n")
                for table in ref_tables:
                    table_md = table.get('markdown', '')
                    if table_md:
                        markdown_parts.append(table_md)
                        markdown_parts.append("\n\n")
            
            if ref_pictures:
                markdown_parts.append("**Pictures in referenced section:**\n\n")
                for pic in ref_pictures:
                    caption = pic.get('caption', '')
                    if caption:
                        markdown_parts.append(f"- {caption}\n")
                markdown_parts.append("\n")
            
            if occurrences > 1:
                markdown_parts.append(f"*Referenced {occurrences} times in the text.*\n\n")
    
    # Footnote references
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
            content = ref.get('content', {})
            if content:
                markdown_parts.append("*[External reference content]*\n\n")
            else:
                markdown_parts.append("*[External reference not available]*\n\n")
    
    return "\n".join(markdown_parts)


def export_chunks_to_markdown(chunks: List[Dict[str, Any]], output_path: str, source_file: str, 
                              document_name: str, chunking_level: str) -> None:
    """
    Export chunks to markdown format.
    
    Args:
        chunks: List of chunk dictionaries
        output_path: Path to output markdown file
        source_file: Source file path
        document_name: Document name
        chunking_level: Chunking level used
    """
    print(f'\nSaving markdown to: {output_path}')
    
    markdown_lines = []
    
    # Document header
    markdown_lines.append("# Legal Document Chunks\n\n")
    markdown_lines.append(f"*Generated from: {source_file}*\n")
    markdown_lines.append(f"*Document: {document_name}*\n")
    markdown_lines.append(f"*Chunking method: hierarchy_based*\n")
    markdown_lines.append(f"*Chunking level: {chunking_level}*\n")
    markdown_lines.append(f"*Total chunks: {len(chunks)}*\n\n")
    markdown_lines.append("---\n\n")
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        if i > 0:
            markdown_lines.append("\n\n---\n\n")
        
        chunk_id = chunk.get('chunk_id', '?')
        print(f"Processing chunk {i+1}/{len(chunks)}: {chunk_id}")
        chunk_md = format_chunk_to_markdown(chunk)
        markdown_lines.append(chunk_md)
    
    full_markdown = "\n".join(markdown_lines)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_markdown)
    
    print(f"✓ Successfully saved {len(chunks)} chunk(s) to markdown")


def main():
    """CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Hierarchical structure-based chunking for Docling documents'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input JSON file path'
    )
    parser.add_argument(
        '--level',
        choices=['paragraph', 'odsek', 'pismeno'],
        default='paragraph',
        help='Chunking level (default: paragraph)'
    )
    parser.add_argument(
        '--identifier',
        help='Optional specific section to chunk (e.g., "50" for paragraph, "50.1" for odsek, "50.1.a" for pismeno)'
    )
    parser.add_argument(
        '--output',
        help='Output JSON file path (default: auto-generated)'
    )
    parser.add_argument(
        '--process-internal-refs',
        action='store_true',
        default=False,
        help='Process and resolve internal references (opt-in)'
    )
    parser.add_argument(
        '--process-footnotes',
        action='store_true',
        default=False,
        help='Process and resolve footnote references (opt-in)'
    )
    parser.add_argument(
        '--process-external-refs',
        action='store_true',
        default=False,
        help='Process and resolve external references (opt-in, placeholder)'
    )
    
    args = parser.parse_args()
    
    # Load document
    print(f'Loading DoclingDocument from: {args.input}')
    doc = load_docling_document(args.input)
    print(f'  ✓ Loaded document: {doc.name}')
    print(f'  ✓ Texts: {len(doc.texts)}, Tables: {len(doc.tables)}, Pictures: {len(doc.pictures)}\n')
    
    # Generate chunks using hierarchical approach
    print(f'Chunking at level: {args.level} (using hierarchical traversal)')
    if args.identifier:
        print(f'  Specific identifier: {args.identifier}')
    print()
    
    # Print reference processing options
    if args.process_internal_refs or args.process_footnotes or args.process_external_refs:
        print('Reference processing options:')
        if args.process_internal_refs:
            print('  ✓ Internal references: enabled')
        if args.process_footnotes:
            print('  ✓ Footnotes: enabled')
        if args.process_external_refs:
            print('  ✓ External references: enabled (placeholder)')
        print()
    
    chunks = chunk_document_hierarchical(
        doc, 
        level=args.level, 
        specific_identifier=args.identifier,
        process_internal_refs=args.process_internal_refs,
        process_footnotes=args.process_footnotes,
        process_external_refs=args.process_external_refs
    )
    
    # Prepare output
    if not args.output:
        input_path = Path(args.input)
        level_suffix = args.level
        if args.identifier:
            identifier_suffix = args.identifier.replace('.', '_')
            output_file = input_path.parent / f"{input_path.stem}_chunked_hierarchical_{level_suffix}_{identifier_suffix}.json"
        else:
            output_file = input_path.parent / f"{input_path.stem}_chunked_hierarchical_{level_suffix}_all.json"
    else:
        output_file = Path(args.output)
    
    # Create output structure
    output = {
        'source_file': str(args.input),
        'document_name': doc.name,
        'chunking_method': 'hierarchy_based',
        'chunking_level': args.level,
        'total_chunks': len(chunks),
        'chunks': chunks
    }
    
    # Save to JSON
    print(f'\nSaving {len(chunks)} chunk(s) to: {output_file}')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # Also save to Markdown
    md_output_file = output_file.with_suffix('.md')
    export_chunks_to_markdown(
        chunks,
        str(md_output_file),
        str(args.input),
        doc.name,
        args.level
    )
    
    print(f'\n✓ Successfully created {len(chunks)} chunk(s)')
    print(f'\nSummary:')
    for chunk in chunks:
        print(f"  {chunk['chunk_id']}:")
        print(f"    - Text length: {len(chunk['main_content']['text'])} characters")
        print(f"    - Tables: {len(chunk['main_content']['tables'])}")
        print(f"    - Pictures: {len(chunk['main_content']['pictures'])}")
        if args.process_internal_refs:
            print(f"    - Internal references: {len(chunk.get('internal_references', []))}")
        if args.process_footnotes:
            print(f"    - Footnote references: {len(chunk.get('footnote_references', []))}")
        if args.process_external_refs:
            print(f"    - External references: {len(chunk.get('external_references', []))}")
    print(f'\nOutput files:')
    print(f"  JSON:     {output_file}")
    print(f"  Markdown: {md_output_file}")


if __name__ == '__main__':
    main()

