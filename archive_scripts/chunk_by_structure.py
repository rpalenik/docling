#!/usr/bin/env python3
"""
Structure-based chunking for Docling documents.

Chunks documents at configurable levels (paragraph, odsek, pismeno) using
DoclingDocument tree navigation. Extracts complete content including tables
and pictures, and resolves all internal and external references.
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
        ref_item: RefItem to resolve (has .ref attribute)
        
    Returns:
        The actual object (text, group, table, picture, or body)
    """
    if not hasattr(ref_item, 'ref') or not ref_item.ref:
        return None
    
    ref_path = ref_item.ref
    
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


def find_element_by_text(doc: DoclingDocument, text_pattern: str, no_hyperlink: bool = False) -> Optional[Any]:
    """
    Find a text element by its text content.
    
    Args:
        doc: DoclingDocument
        text_pattern: Text pattern to match (exact match or regex)
        no_hyperlink: If True, only match elements without hyperlinks
        
    Returns:
        First matching text element or None
    """
    if not hasattr(doc, 'texts') or not doc.texts:
        return None
    
    for text_item in doc.texts:
        text = getattr(text_item, 'text', '').strip()
        hyperlink = getattr(text_item, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        
        # Check if text matches
        if text == text_pattern or (text_pattern.startswith('^') and re.match(text_pattern, text)):
            if no_hyperlink and hyperlink_str:
                continue
            if not no_hyperlink or not hyperlink_str:
                return text_item
    
    return None


def find_chunk_boundaries(doc: DoclingDocument, level: str, identifier: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Find start and end indices for a chunk at the specified level.
    
    Args:
        doc: DoclingDocument
        level: Chunking level ('paragraph', 'odsek', 'pismeno')
        identifier: Section identifier (e.g., '50' for paragraph, '50.1' for odsek, '50.1.a' for pismeno)
        
    Returns:
        Tuple of (start_index, end_index) in doc.texts, or (None, None) if not found
    """
    if not hasattr(doc, 'texts') or not doc.texts:
        return None, None
    
    texts = doc.texts
    
    # Parse identifier based on level
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
    
    # Find paragraph start: "§ N" with no hyperlink (exact match, not variants like "§ 50a")
    start_idx = None
    for i, text_item in enumerate(texts):
        text = getattr(text_item, 'text', '').strip()
        hyperlink = getattr(text_item, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        
        # Match exactly "§ {para_num}" (not "§ 50a" or "§ 50aa")
        if text == f'§ {para_num}' and not hyperlink_str:
            start_idx = i
            break
    
    if start_idx is None:
        return None, None
    
    # If odsek specified, find subsection marker "(N)"
    if odsek:
        subsection_marker = f'({odsek})'
        found = False
        for i in range(start_idx, min(start_idx + 500, len(texts))):
            text = getattr(texts[i], 'text', '').strip()
            if text == subsection_marker:
                start_idx = i
                found = True
                break
            # Stop if we hit next paragraph (including variants like "§ 50a", "§ 50aa")
            if text.startswith('§ '):
                match = re.match(r'§ (\d+[a-zA-Z]*)', text)
                if match:
                    next_para_full = match.group(1)
                    if next_para_full != para_num:
                        break
        if not found:
            return None, None
    
    # If pismeno specified, find písmeno marker "a)"
    if pismeno:
        pismeno_marker = f'{pismeno})'
        found = False
        for i in range(start_idx, min(start_idx + 300, len(texts))):
            text = getattr(texts[i], 'text', '').strip()
            if text == pismeno_marker:
                start_idx = i
                found = True
                break
            # Stop if we hit next subsection or paragraph
            if text.startswith('(') and text[1:].rstrip(')').isdigit():
                break
            # Stop if we hit next paragraph (including variants like "§ 50a", "§ 50aa")
            if text.startswith('§ '):
                match = re.match(r'§ (\d+[a-zA-Z]*)', text)
                if match:
                    next_para_full = match.group(1)
                    if next_para_full != para_num:
                        break
        if not found:
            return None, None
    
    # Find end index based on level
    end_idx = None
    
    if pismeno:
        # End at next písmeno, subsection, or paragraph
        for i in range(start_idx + 1, min(start_idx + 200, len(texts))):
            text = getattr(texts[i], 'text', '').strip()
            if text in ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)'] and text != f'{pismeno})':
                end_idx = i
                break
            if text.startswith('(') and text[1:].rstrip(')').isdigit():
                end_idx = i
                break
            # Check if this is a different paragraph (including variants like "§ 50a", "§ 50aa")
            if text.startswith('§ '):
                match = re.match(r'§ (\d+[a-zA-Z]*)', text)
                if match:
                    next_para_full = match.group(1)
                    if next_para_full != para_num:
                        end_idx = i
                        break
    elif odsek:
        # End at next subsection or paragraph
        for i in range(start_idx + 1, min(start_idx + 300, len(texts))):
            text = getattr(texts[i], 'text', '').strip()
            if text.startswith('(') and text[1:].rstrip(')').isdigit():
                next_num = text[1:].rstrip(')')
                if next_num != odsek:
                    end_idx = i
                    break
            # Check if this is a different paragraph (including variants like "§ 50a", "§ 50aa")
            if text.startswith('§ '):
                match = re.match(r'§ (\d+[a-zA-Z]*)', text)
                if match:
                    next_para_full = match.group(1)
                    if next_para_full != para_num:
                        end_idx = i
                        break
    else:
        # End at next paragraph (including variants like "§ 50a", "§ 50aa")
        for i in range(start_idx + 1, len(texts)):
            text = getattr(texts[i], 'text', '').strip()
            hyperlink = getattr(texts[i], 'hyperlink', '')
            hyperlink_str = str(hyperlink) if hyperlink else ''
            if text.startswith('§ ') and not hyperlink_str:
                # Extract the full paragraph identifier (number + optional letters)
                match = re.match(r'§ (\d+[a-zA-Z]*)', text)
                if match:
                    next_para_full = match.group(1)
                    # Check if it's exactly our paragraph number (not a variant)
                    if next_para_full == para_num:
                        # Still within the same paragraph, continue
                        continue
                    else:
                        # Different paragraph (could be 50a, 50aa, 51, etc.)
                        end_idx = i
                        break
                else:
                    # Some other format, treat as different paragraph
                    end_idx = i
                    break
    
    if end_idx is None:
        # Safety limit
        if pismeno:
            end_idx = min(start_idx + 200, len(texts))
        elif odsek:
            end_idx = min(start_idx + 300, len(texts))
        else:
            end_idx = min(start_idx + 500, len(texts))
    
    return start_idx, end_idx


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
    # For now, we'll identify them by checking if they appear in the text range
    # This is a simplified approach - in a full implementation, we'd use tree navigation
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
    Extract full content of a referenced section.
    
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
    
    # Find boundaries
    start_idx, end_idx = find_chunk_boundaries(doc, level, identifier)
    
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
    Extract footnote content by footnote ID.
    
    Args:
        doc: DoclingDocument
        footnote_id: Footnote identifier (e.g., "1", "136f")
        
    Returns:
        Footnote text content
    """
    if not hasattr(doc, 'texts') or not doc.texts:
        return ""
    
    # Find footnotes section
    poznamky_start = find_footnote_section(doc)
    if poznamky_start is None:
        return ""
    
    target_hyperlink = f'#poznamky.poznamka-{footnote_id}'
    
    # Find the footnote
    footnote_start = None
    for i in range(poznamky_start, min(poznamky_start + 10000, len(doc.texts))):
        text_item = doc.texts[i]
        hyperlink = getattr(text_item, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        self_ref = getattr(text_item, 'self_ref', '')
        self_ref_str = str(self_ref) if self_ref else ''
        text = getattr(text_item, 'text', '')
        
        if (hyperlink_str == target_hyperlink or 
            f'poznamka-{footnote_id}' in self_ref_str or
            (f'poznamka-{footnote_id}' in hyperlink_str and i > poznamky_start)):
            if text and text != footnote_id and text != ')':
                footnote_start = i
                break
    
    if footnote_start is None:
        return ""
    
    # Extract footnote text
    footnote_texts = []
    for i in range(footnote_start, min(footnote_start + 200, len(doc.texts))):
        text_item = doc.texts[i]
        text = getattr(text_item, 'text', '').strip()
        hyperlink = getattr(text_item, 'hyperlink', '')
        hyperlink_str = str(hyperlink) if hyperlink else ''
        label = getattr(text_item, 'label', '')
        
        # Stop at next footnote
        if hyperlink_str and 'poznamka-' in hyperlink_str and f'poznamka-{footnote_id}' not in hyperlink_str:
            break
        
        # Stop at major section break
        if text and '§' in text and i > footnote_start + 5:
            break
        
        if (text and 
            text not in ['plus', 'button-close'] and
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


def build_chunk(doc: DoclingDocument, level: str, identifier: str, main_content: Dict[str, Any], 
                internal_refs: List[Dict[str, Any]], footnote_refs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Assemble final chunk structure with main content and resolved references.
    
    Args:
        doc: DoclingDocument
        level: Chunking level
        identifier: Section identifier
        main_content: Main chunk content
        internal_refs: List of internal references
        footnote_refs: List of footnote references
        
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
    
    # Resolve internal references
    resolved_internal_refs = []
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
    
    # Resolve footnote references
    resolved_footnote_refs = []
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
    
    return {
        'chunk_id': chunk_id,
        'level': level,
        'identifier': identifier,
        'main_content': main_content,
        'internal_references': resolved_internal_refs,
        'footnote_references': resolved_footnote_refs
    }


def chunk_document(doc: DoclingDocument, level: str = 'paragraph', specific_identifier: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Main chunking function.
    
    Args:
        doc: DoclingDocument
        level: Chunking level ('paragraph', 'odsek', 'pismeno')
        specific_identifier: Optional specific section to chunk (e.g., '50' for paragraph, '50.1' for odsek)
        
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    
    if not hasattr(doc, 'texts') or not doc.texts:
        return chunks
    
    if specific_identifier:
        # Chunk only the specified section
        start_idx, end_idx = find_chunk_boundaries(doc, level, specific_identifier)
        if start_idx is not None and end_idx is not None:
            text_items = doc.texts[start_idx:end_idx]
            main_content = extract_chunk_content(doc, start_idx, end_idx)
            internal_refs = extract_internal_references(text_items)
            footnote_refs = extract_footnote_references(text_items)
            chunk = build_chunk(doc, level, specific_identifier, main_content, internal_refs, footnote_refs)
            chunks.append(chunk)
    else:
        # Chunk all sections at the specified level
        if level == 'paragraph':
            # Find all paragraphs (including variants like "50", "50a", "50aa")
            paragraph_numbers = []
            for text_item in doc.texts:
                text = getattr(text_item, 'text', '').strip()
                hyperlink = getattr(text_item, 'hyperlink', '')
                hyperlink_str = str(hyperlink) if hyperlink else ''
                if text.startswith('§ ') and not hyperlink_str:
                    # Extract full paragraph identifier (number + optional letters)
                    match = re.match(r'§ (\d+[a-zA-Z]*)', text)
                    if match:
                        para_num_full = match.group(1)
                        # Only add if it's exactly "§ {para_num_full}" (not a longer variant)
                        # This ensures we get "50", "50a", "50aa" as separate paragraphs
                        if text == f'§ {para_num_full}' and para_num_full not in paragraph_numbers:
                            paragraph_numbers.append(para_num_full)
            
            print(f'Found {len(paragraph_numbers)} paragraphs to chunk')
            for para_num in paragraph_numbers:
                print(f'Processing § {para_num}...')
                start_idx, end_idx = find_chunk_boundaries(doc, level, para_num)
                if start_idx is not None and end_idx is not None:
                    text_items = doc.texts[start_idx:end_idx]
                    main_content = extract_chunk_content(doc, start_idx, end_idx)
                    internal_refs = extract_internal_references(text_items)
                    footnote_refs = extract_footnote_references(text_items)
                    chunk = build_chunk(doc, level, para_num, main_content, internal_refs, footnote_refs)
                    chunks.append(chunk)
        
        elif level == 'odsek':
            # Find all odseks (need to find all paragraphs first, then their odseks)
            paragraph_numbers = []
            for text_item in doc.texts:
                text = getattr(text_item, 'text', '').strip()
                hyperlink = getattr(text_item, 'hyperlink', '')
                hyperlink_str = str(hyperlink) if hyperlink else ''
                if text.startswith('§ ') and not hyperlink_str:
                    # Extract full paragraph identifier (number + optional letters)
                    match = re.match(r'§ (\d+[a-zA-Z]*)', text)
                    if match:
                        para_num_full = match.group(1)
                        # Only add if it's exactly "§ {para_num_full}" (not a longer variant)
                        if text == f'§ {para_num_full}' and para_num_full not in paragraph_numbers:
                            paragraph_numbers.append(para_num_full)
            
            for para_num in paragraph_numbers:
                # Find all odseks in this paragraph
                odsek_numbers = set()
                para_start, _ = find_chunk_boundaries(doc, 'paragraph', para_num)
                if para_start is not None:
                    for i in range(para_start, min(para_start + 500, len(doc.texts))):
                        text = getattr(doc.texts[i], 'text', '').strip()
                        if text.startswith('(') and text[1:].rstrip(')').isdigit():
                            odsek_num = text[1:].rstrip(')')
                            odsek_numbers.add(odsek_num)
                        # Stop if we hit next paragraph (including variants like "§ 50a", "§ 50aa")
                        elif text.startswith('§ '):
                            match = re.match(r'§ (\d+[a-zA-Z]*)', text)
                            if match:
                                next_para_full = match.group(1)
                                if next_para_full != para_num:
                                    break
                
                odsek_numbers = sorted(odsek_numbers, key=lambda x: int(x))
                for odsek_num in odsek_numbers:
                    identifier = f"{para_num}.{odsek_num}"
                    print(f'Processing § {para_num}, odsek {odsek_num}...')
                    start_idx, end_idx = find_chunk_boundaries(doc, level, identifier)
                    if start_idx is not None and end_idx is not None:
                        text_items = doc.texts[start_idx:end_idx]
                        main_content = extract_chunk_content(doc, start_idx, end_idx)
                        internal_refs = extract_internal_references(text_items)
                        footnote_refs = extract_footnote_references(text_items)
                        chunk = build_chunk(doc, level, identifier, main_content, internal_refs, footnote_refs)
                        chunks.append(chunk)
        
        elif level == 'pismeno':
            # Similar logic for pismeno - find all paragraphs, then odseks, then pismenos
            paragraph_numbers = []
            for text_item in doc.texts:
                text = getattr(text_item, 'text', '').strip()
                hyperlink = getattr(text_item, 'hyperlink', '')
                hyperlink_str = str(hyperlink) if hyperlink else ''
                if text.startswith('§ ') and not hyperlink_str:
                    # Extract full paragraph identifier (number + optional letters)
                    match = re.match(r'§ (\d+[a-zA-Z]*)', text)
                    if match:
                        para_num_full = match.group(1)
                        # Only add if it's exactly "§ {para_num_full}" (not a longer variant)
                        if text == f'§ {para_num_full}' and para_num_full not in paragraph_numbers:
                            paragraph_numbers.append(para_num_full)
            
            for para_num in paragraph_numbers:
                para_start, _ = find_chunk_boundaries(doc, 'paragraph', para_num)
                if para_start is not None:
                    # Find all odseks
                    odsek_numbers = set()
                    for i in range(para_start, min(para_start + 500, len(doc.texts))):
                        text = getattr(doc.texts[i], 'text', '').strip()
                        if text.startswith('(') and text[1:].rstrip(')').isdigit():
                            odsek_num = text[1:].rstrip(')')
                            odsek_numbers.add(odsek_num)
                        # Stop if we hit next paragraph (including variants like "§ 50a", "§ 50aa")
                        elif text.startswith('§ '):
                            match = re.match(r'§ (\d+[a-zA-Z]*)', text)
                            if match:
                                next_para_full = match.group(1)
                                if next_para_full != para_num:
                                    break
                    
                    for odsek_num in sorted(odsek_numbers, key=lambda x: int(x)):
                        odsek_start, _ = find_chunk_boundaries(doc, 'odsek', f"{para_num}.{odsek_num}")
                        if odsek_start is not None:
                            # Find all pismenos in this odsek
                            pismeno_letters = set()
                            for i in range(odsek_start, min(odsek_start + 300, len(doc.texts))):
                                text = getattr(doc.texts[i], 'text', '').strip()
                                if text in ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)']:
                                    pismeno_letter = text[0]
                                    pismeno_letters.add(pismeno_letter)
                                elif text.startswith('(') and text[1:].rstrip(')').isdigit():
                                    break
                                # Stop if we hit next paragraph (including variants like "§ 50a", "§ 50aa")
                                elif text.startswith('§ '):
                                    match = re.match(r'§ (\d+[a-zA-Z]*)', text)
                                    if match:
                                        next_para_full = match.group(1)
                                        if next_para_full != para_num:
                                            break
                            
                            for pismeno_letter in sorted(pismeno_letters):
                                identifier = f"{para_num}.{odsek_num}.{pismeno_letter}"
                                print(f'Processing § {para_num}, odsek {odsek_num}, pismeno {pismeno_letter}...')
                                start_idx, end_idx = find_chunk_boundaries(doc, level, identifier)
                                if start_idx is not None and end_idx is not None:
                                    text_items = doc.texts[start_idx:end_idx]
                                    main_content = extract_chunk_content(doc, start_idx, end_idx)
                                    internal_refs = extract_internal_references(text_items)
                                    footnote_refs = extract_footnote_references(text_items)
                                    chunk = build_chunk(doc, level, identifier, main_content, internal_refs, footnote_refs)
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
    markdown_lines.append(f"*Chunking method: structure_based*\n")
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
        description='Structure-based chunking for Docling documents'
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
    
    args = parser.parse_args()
    
    # Load document
    print(f'Loading DoclingDocument from: {args.input}')
    doc = load_docling_document(args.input)
    print(f'  ✓ Loaded document: {doc.name}')
    print(f'  ✓ Texts: {len(doc.texts)}, Tables: {len(doc.tables)}, Pictures: {len(doc.pictures)}\n')
    
    # Generate chunks
    print(f'Chunking at level: {args.level}')
    if args.identifier:
        print(f'  Specific identifier: {args.identifier}')
    print()
    
    chunks = chunk_document(doc, level=args.level, specific_identifier=args.identifier)
    
    # Prepare output
    if not args.output:
        input_path = Path(args.input)
        level_suffix = args.level
        if args.identifier:
            identifier_suffix = args.identifier.replace('.', '_')
            output_file = input_path.parent / f"{input_path.stem}_chunked_{level_suffix}_{identifier_suffix}.json"
        else:
            output_file = input_path.parent / f"{input_path.stem}_chunked_{level_suffix}_all.json"
    else:
        output_file = Path(args.output)
    
    # Create output structure
    output = {
        'source_file': str(args.input),
        'document_name': doc.name,
        'chunking_method': 'structure_based',
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
        print(f"    - Internal references: {len(chunk['internal_references'])}")
        print(f"    - Footnote references: {len(chunk['footnote_references'])}")
    print(f'\nOutput files:')
    print(f"  JSON:     {output_file}")
    print(f"  Markdown: {md_output_file}")


if __name__ == '__main__':
    main()

