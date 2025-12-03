#!/usr/bin/env python3
"""
Custom chunking by paragraph (e.g., § 50) with metadata for internal and footnote references.
"""

import json
import re
from typing import List, Dict, Any, Optional

def extract_reference_text(texts: List[Dict], paragraph_number: str, odsek: Optional[str] = None, pismeno: Optional[str] = None) -> str:
    """
    Universal function to extract text for any reference specificity level.
    
    Uses deterministic rule: "§ N" with NO hyperlink = content section.
    Then navigates hierarchically to subsections/písmená if specified.
    
    Args:
        texts: List of text elements from Docling JSON
        paragraph_number: Paragraph number (e.g., "47", "33a")
        odsek: Optional subsection number (e.g., "1", "2")
        pismeno: Optional sub-subsection letter (e.g., "a", "b")
    
    Returns:
        Full text content of the reference, or empty string if not found
    """
    # Step 1: Find paragraph content section using deterministic rule
    # Rule: "§ N" (exact match) with NO hyperlink = content section
    para_content_start = None
    for i, text_item in enumerate(texts):
        text = text_item.get('text', '').strip()
        hyperlink = text_item.get('hyperlink', '')
        
        # Exact match for paragraph number with no hyperlink
        if text == f'§ {paragraph_number}' and not hyperlink:
            para_content_start = i
            break
    
    if para_content_start is None:
        return ""
    
    # Step 2: If odsek specified, navigate to subsection marker "(N)"
    start_idx = para_content_start
    if odsek:
        subsection_marker = f'({odsek})'
        found_subsection = False
        
        # Search within paragraph content (reasonable limit)
        for i in range(para_content_start, min(para_content_start + 500, len(texts))):
            text = texts[i].get('text', '').strip()
            
            # Check if we've moved to next paragraph
            if text.startswith('§ ') and not text.startswith(f'§ {paragraph_number}'):
                break
            
            # Found subsection marker
            if text == subsection_marker:
                start_idx = i
                found_subsection = True
                break
        
        if not found_subsection:
            return ""  # Subsection not found
    
    # Step 3: If pismeno specified, navigate to písmeno marker "a)"
    if pismeno:
        pismeno_marker = f'{pismeno})'
        found_pismeno = False
        
        # Search within subsection (reasonable limit)
        for i in range(start_idx, min(start_idx + 300, len(texts))):
            text = texts[i].get('text', '').strip()
            
            # Check if we've moved to next subsection or paragraph
            if text.startswith('(') and text[1:].rstrip(')').isdigit():
                # Next subsection - stop here
                break
            if text.startswith('§ ') and not text.startswith(f'§ {paragraph_number}'):
                # Next paragraph - stop here
                break
            
            # Found písmeno marker
            if text == pismeno_marker:
                start_idx = i
                found_pismeno = True
                break
        
        if not found_pismeno:
            return ""  # Písmeno not found
    
    # Step 4: Find end index based on specificity level
    end_idx = None
    target_para_num = paragraph_number
    
    if pismeno:
        # Extract until next písmeno, next subsection, or next paragraph
        for i in range(start_idx + 1, min(start_idx + 200, len(texts))):
            text = texts[i].get('text', '').strip()
            
            # Next písmeno (same subsection)
            if text in ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)']:
                # Check if it's a different písmeno
                if text != f'{pismeno})':
                    end_idx = i
                    break
            
            # Next subsection
            if text.startswith('(') and text[1:].rstrip(')').isdigit():
                end_idx = i
                break
            
            # Next paragraph
            if text.startswith('§ ') and not text.startswith(f'§ {target_para_num}'):
                end_idx = i
                break
    elif odsek:
        # Extract until next subsection or next paragraph
        for i in range(start_idx + 1, min(start_idx + 300, len(texts))):
            text = texts[i].get('text', '').strip()
            
            # Next subsection
            if text.startswith('(') and text[1:].rstrip(')').isdigit():
                next_num = text[1:].rstrip(')')
                if next_num != odsek:
                    end_idx = i
                    break
            
            # Next paragraph
            if text.startswith('§ ') and not text.startswith(f'§ {target_para_num}'):
                end_idx = i
                break
    else:
        # Extract until next paragraph
        # The actual next paragraph has "§ N" with NO hyperlink (deterministic rule)
        for i in range(start_idx + 1, len(texts)):
            text = texts[i].get('text', '').strip()
            hyperlink = texts[i].get('hyperlink', '')
            
            # Next paragraph: "§ N" with NO hyperlink (not a reference)
            if text.startswith('§ ') and not hyperlink:
                # Check if it's a different paragraph
                match = re.match(r'§ (\d+)', text)
                if match:
                    next_para = match.group(1)
                    if next_para != target_para_num:
                        end_idx = i
                        break
                elif not text.startswith(f'§ {target_para_num}'):
                    # Different paragraph (e.g., § 50aa, § 50a)
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
    
    # Step 5: Collect and reconstruct text
    reference_texts = []
    for i in range(start_idx, end_idx):
        text_item = texts[i]
        text = text_item.get('text', '').strip()
        hyperlink = text_item.get('hyperlink', '')
        label = text_item.get('label', '')
        
        if (text and 
            text not in ['plus', 'button-close', 'button-search', 'button-download', 
                        'button-print', 'button-history', 'button-content'] and
            label != 'caption'):
            reference_texts.append({
                'text': text,
                'hyperlink': hyperlink,
                'label': label
            })
    
    # Reconstruct text (same logic as other extraction functions)
    full_text = []
    for item in reference_texts:
        text = item['text']
        hyperlink = item.get('hyperlink', '')
        
        if text.strip() in ['(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)', '(8)', '(9)', '(10)', 
                           '(11)', '(12)', '(13)', '(14)', '(15)', '(16)']:
            full_text.append(f'\n\n### {text}\n\n')
        elif text.strip() in ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)']:
            full_text.append(f'\n**{text}** ')
        elif text.strip() in ['1.', '2.', '3.', '4.', '5.']:
            full_text.append(f'    {text} ')
        elif hyperlink and 'paragraf-' in hyperlink and 'poznamky' not in hyperlink:
            full_text.append(f' [{text}]({hyperlink})')
        elif hyperlink and 'poznamky.poznamka' in hyperlink:
            next_idx = reference_texts.index(item) + 1
            if next_idx < len(reference_texts):
                next_item = reference_texts[next_idx]
                if (next_item.get('hyperlink') == hyperlink and 
                    next_item.get('text') == ')'):
                    full_text.append(f'<sup>{text}</sup>)')
                    reference_texts[next_idx]['processed'] = True
                else:
                    full_text.append(f'<sup>{text}</sup>')
            else:
                full_text.append(f'<sup>{text}</sup>')
        else:
            if not item.get('processed', False):
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


def extract_paragraph_text(texts: List[Dict], paragraph_number: str) -> str:
    """
    Extract the full text content of a paragraph by its number.
    
    Uses the universal extract_reference_text() function with deterministic rule.
    
    Args:
        texts: List of text elements from Docling JSON
        paragraph_number: Paragraph number to extract (e.g., "50", "33a", "50aa")
    
    Returns:
        Full text content of the paragraph, or empty string if not found
    """
    return extract_reference_text(texts, paragraph_number)


def extract_footnote_text(texts: List[Dict], footnote_number: str) -> str:
    """
    Extract the full text content of a footnote from the Poznámky section.
    
    Args:
        texts: List of text elements from Docling JSON
        footnote_number: Footnote number to extract (e.g., "136f")
    
    Returns:
        Full text content of the footnote, or empty string if not found
    """
    # Find the Poznámky section (around index 3500)
    poznamky_start = None
    for i, text_item in enumerate(texts):
        text = text_item.get('text', '')
        hyperlink = text_item.get('hyperlink', '')
        if 'Poznámky' in text or hyperlink == '#poznamky':
            poznamky_start = i
            break
    
    if poznamky_start is None:
        return ""
    
    # Find the specific footnote
    footnote_start = None
    target_hyperlink = f'#poznamky.poznamka-{footnote_number}'
    
    for i in range(poznamky_start, min(poznamky_start + 10000, len(texts))):
        text_item = texts[i]
        hyperlink = text_item.get('hyperlink', '')
        self_ref = text_item.get('self_ref', '')
        
        # Look for the footnote heading or first text element with this hyperlink
        if (hyperlink == target_hyperlink or 
            f'poznamka-{footnote_number}' in self_ref or
            (f'poznamka-{footnote_number}' in hyperlink and i > poznamky_start)):
            # Check if this is the actual footnote content (not just a reference)
            text = text_item.get('text', '')
            if text and text != footnote_number and text != ')':
                footnote_start = i
                break
    
    if footnote_start is None:
        return ""
    
    # Extract all text elements belonging to this footnote
    footnote_texts = []
    i = footnote_start
    
    # Look backwards to find the footnote heading if we started in the middle
    while i > poznamky_start:
        text_item = texts[i]
        text = text_item.get('text', '')
        hyperlink = text_item.get('hyperlink', '')
        self_ref = text_item.get('self_ref', '')
        
        # Check if this is the footnote heading
        if (f'poznamka-{footnote_number}' in self_ref or
            (hyperlink == target_hyperlink and text.strip() == footnote_number)):
            footnote_start = i
            break
        i -= 1
    
    # Now collect all text until next footnote or section end
    for i in range(footnote_start, min(footnote_start + 200, len(texts))):
        text_item = texts[i]
        text = text_item.get('text', '').strip()
        hyperlink = text_item.get('hyperlink', '')
        label = text_item.get('label', '')
        
        # Stop at next footnote
        if hyperlink and 'poznamka-' in hyperlink and f'poznamka-{footnote_number}' not in hyperlink:
            break
        
        # Stop at major section break
        if text and '§' in text and i > footnote_start + 5:
            break
        
        if (text and 
            text not in ['plus', 'button-close'] and
            label != 'caption'):
            footnote_texts.append(text)
    
    # Reconstruct footnote text
    footnote_content = ' '.join(footnote_texts)
    footnote_content = re.sub(r' +', ' ', footnote_content)
    return footnote_content.strip()


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


def resolve_internal_reference(texts: List[Dict], target_paragraph: str = None, hyperlink: str = None) -> Dict[str, Any]:
    """
    Resolve an internal reference (single level, no recursion).
    Includes the referenced paragraph's text and its internal references (but doesn't resolve those).
    
    Uses universal extract_reference_text() to handle all specificity levels.
    
    Args:
        texts: List of text elements from Docling JSON
        target_paragraph: Paragraph number to resolve (e.g., "33") - for backward compatibility
        hyperlink: Hyperlink string (e.g., "#paragraf-47.odsek-1.pismeno-a") - preferred
    
    Returns:
        Dictionary with referenced_text and referenced_internal_refs (basic structure only)
    """
    # Parse hyperlink if provided, otherwise use target_paragraph
    if hyperlink:
        ref_path = parse_reference_hyperlink(hyperlink)
        para_num = ref_path['paragraph_number']
        odsek = ref_path['odsek']
        pismeno = ref_path['pismeno']
    else:
        para_num = target_paragraph
        odsek = None
        pismeno = None
    
    if not para_num:
        return {
            'referenced_text': '',
            'referenced_internal_refs': []
        }
    
    # Extract reference text using universal function
    referenced_text = extract_reference_text(texts, para_num, odsek, pismeno)
    
    if not referenced_text:
        return {
            'referenced_text': '',
            'referenced_internal_refs': []
        }
    
    # Extract internal references from the resolved text
    # Find the content section to get boundaries
    para_content_start = None
    for i, text_item in enumerate(texts):
        text = text_item.get('text', '').strip()
        hyperlink_item = text_item.get('hyperlink', '')
        if text == f'§ {para_num}' and not hyperlink_item:
            para_content_start = i
            break
    
    if para_content_start is None:
        return {
            'referenced_text': referenced_text,
            'referenced_internal_refs': []
        }
    
    # Determine end index based on specificity
    if pismeno:
        search_limit = 200
    elif odsek:
        search_limit = 300
    else:
        search_limit = 500
    
    # Find end of the reference section
    end_idx = None
    for i in range(para_content_start + 1, min(para_content_start + search_limit, len(texts))):
        text = texts[i].get('text', '').strip()
        
        # Check boundaries based on specificity level
        if pismeno:
            if text in ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)'] and text != f'{pismeno})':
                end_idx = i
                break
            if text.startswith('(') and text[1:].rstrip(')').isdigit():
                end_idx = i
                break
        elif odsek:
            if text.startswith('(') and text[1:].rstrip(')').isdigit():
                next_num = text[1:].rstrip(')')
                if next_num != odsek:
                    end_idx = i
                    break
        
        if text.startswith('§ ') and not text.startswith(f'§ {para_num}'):
            end_idx = i
            break
    
    if end_idx is None:
        end_idx = min(para_content_start + search_limit, len(texts))
    
    # Collect internal references from this section (basic structure only)
    paragraph_internal_refs = []
    for i in range(para_content_start, end_idx):
        text_item = texts[i]
        text = text_item.get('text', '').strip()
        hyperlink_item = text_item.get('hyperlink', '')
        
        if hyperlink_item and 'paragraf-' in hyperlink_item and 'poznamky' not in hyperlink_item:
            # Extract paragraph number from reference
            ref_path = parse_reference_hyperlink(hyperlink_item)
            ref_para = ref_path['paragraph_number']
            
            # Exclude self-references
            if ref_para and ref_para != para_num:
                paragraph_internal_refs.append({
                    'text': text,
                    'hyperlink': hyperlink_item,
                    'target_paragraph': ref_para,
                    'odsek': ref_path['odsek'],
                    'pismeno': ref_path['pismeno']
                })
    
    # Get unique references (basic structure, no nested resolution)
    unique_refs = {}
    for ref in paragraph_internal_refs:
        key = ref['hyperlink']
        if key not in unique_refs:
            unique_refs[key] = ref
    
    # Return basic structure (no nested resolution)
    referenced_internal_refs = list(unique_refs.values())
    
    return {
        'referenced_text': referenced_text,
        'referenced_internal_refs': referenced_internal_refs
    }


def extract_subsection_chunk(texts: List[Dict], paragraph_number: str, subsection_number: str) -> Dict[str, Any]:
    """
    Extract a subsection chunk (e.g., § 50, odsek 1).
    
    Args:
        texts: List of text elements from Docling JSON
        paragraph_number: Paragraph number (e.g., "50")
        subsection_number: Subsection number (e.g., "1")
    
    Returns:
        Dictionary with chunk data including text, metadata, and references
    """
    # Find the subsection heading in the content section
    # First, find where paragraph 50 content starts
    para_start_idx = None
    for i, text_item in enumerate(texts):
        text = text_item.get('text', '')
        if text.strip() == f'§ {paragraph_number}':
            if i + 1 < len(texts):
                next_text = texts[i + 1].get('text', '')
                if i + 2 < len(texts):
                    next_next_text = texts[i + 2].get('text', '')
                    if next_next_text.strip() == '(1)':
                        if i + 3 < len(texts):
                            content_check = texts[i + 3].get('text', '')
                            if 'Daňovník' in content_check or len(content_check) > 10:
                                para_start_idx = i
                                break
    
    if para_start_idx is None:
        return None
    
    # Now find the specific subsection within the paragraph content
    content_start_idx = None
    target_marker = f'({subsection_number})'
    
    # Search from paragraph start
    for i in range(para_start_idx, min(para_start_idx + 400, len(texts))):
        text_item = texts[i]
        text = text_item.get('text', '').strip()
        
        # Look for subsection marker
        if text == target_marker:
            # Verify this is the right subsection by checking context
            # Should be after paragraph heading
            if i >= para_start_idx + 2:  # After heading (>= instead of >)
                # Check if next text is actual content (not just another marker)
                if i + 1 < len(texts):
                    next_text = texts[i + 1].get('text', '')
                    # Make sure it's not another subsection marker
                    if not (next_text.strip().startswith('(') and next_text.strip().endswith(')')):
                        if len(next_text) > 3:  # Has content
                            content_start_idx = i
                            break
    
    if content_start_idx is None:
        return None
    
    # Find where this subsection ends (next subsection or next paragraph)
    end_idx = None
    next_subsection = int(subsection_number) + 1
    next_para = int(paragraph_number) + 1
    
    for i in range(content_start_idx + 1, min(content_start_idx + 200, len(texts))):
        text = texts[i].get('text', '').strip()
        
        # Stop at next subsection marker (just check text, not hyperlink)
        if text == f'({next_subsection})':
            end_idx = i
            break
        
        # Stop at next paragraph
        if f'§ {next_para}' in text or f'§ {paragraph_number}aa' in text or f'§ {paragraph_number}a' in text:
            end_idx = i
            break
    
    if end_idx is None:
        end_idx = min(content_start_idx + 200, len(texts))  # Safety limit
    
    # Collect all text elements for this subsection
    subsection_texts = []
    internal_refs = []
    footnote_refs = []
    
    for i in range(content_start_idx, end_idx):
        text_item = texts[i]
        text = text_item.get('text', '').strip()
        hyperlink = text_item.get('hyperlink', '')
        label = text_item.get('label', '')
        
        # Skip navigation elements
        if (text and 
            text not in ['plus', 'button-close', 'button-search', 'button-download', 
                        'button-print', 'button-history', 'button-content'] and
            label != 'caption'):
            
            subsection_texts.append({
                'text': text,
                'hyperlink': hyperlink,
                'label': label,
                'idx': i
            })
            
            # Track references
            if hyperlink:
                if 'paragraf-' in hyperlink and 'poznamky' not in hyperlink:
                    # Internal reference - only if it's to a DIFFERENT paragraph/subsection
                    ref_num = hyperlink.split('paragraf-')[-1].split('.')[0]
                    # Filter out references to the same paragraph (internal structure)
                    if ref_num != paragraph_number:
                        internal_refs.append({
                            'text': text,
                            'hyperlink': hyperlink,
                            'target_paragraph': ref_num,
                            'type': 'internal'
                        })
                elif 'poznamky.poznamka' in hyperlink:
                    # Footnote reference
                    fn_num = hyperlink.split('poznamka-')[-1] if 'poznamka-' in hyperlink else None
                    # Only count the footnote number, not the closing paren
                    if text != ')':
                        footnote_refs.append({
                            'text': text,
                            'hyperlink': hyperlink,
                            'footnote_number': fn_num,
                            'type': 'footnote'
                        })
    
    # Reconstruct the full text
    full_text = []
    prev_hyperlink = None
    
    for item in subsection_texts:
        text = item['text']
        hyperlink = item.get('hyperlink', '')
        
        # Handle structural markers
        if text.strip() in ['(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)', '(8)', '(9)', '(10)', 
                           '(11)', '(12)', '(13)', '(14)', '(15)', '(16)']:
            full_text.append(f'\n\n### {text}\n\n')
        elif text.strip() in ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)']:
            full_text.append(f'\n**{text}** ')
        elif text.strip() in ['1.', '2.', '3.', '4.', '5.']:
            full_text.append(f'    {text} ')
        # Handle references
        elif hyperlink and 'paragraf-' in hyperlink and 'poznamky' not in hyperlink:
            full_text.append(f' [{text}]({hyperlink})')
        elif hyperlink and 'poznamky.poznamka' in hyperlink:
            # Check if next item is closing paren
            next_idx = subsection_texts.index(item) + 1
            if next_idx < len(subsection_texts):
                next_item = subsection_texts[next_idx]
                if (next_item.get('hyperlink') == hyperlink and 
                    next_item.get('text') == ')'):
                    full_text.append(f'<sup>{text}</sup>)')
                    # Mark next item as processed
                    subsection_texts[next_idx]['processed'] = True
                else:
                    full_text.append(f'<sup>{text}</sup>')
            else:
                full_text.append(f'<sup>{text}</sup>')
        # Regular text
        else:
            if not item.get('processed', False):
                if text and len(text) > 1:
                    if full_text and not full_text[-1].endswith(' ') and not full_text[-1].endswith('\n'):
                        if text[0] not in ['.', ',', ';', ':', '!', '?', ')', ']', '}', '(', '[', '{']:
                            full_text.append(' ')
                    full_text.append(text)
    
    # Clean up the text
    text_content = ''.join(full_text)
    text_content = re.sub(r' +', ' ', text_content)
    text_content = re.sub(r' \n', '\n', text_content)
    text_content = re.sub(r'\n{3,}', '\n\n', text_content)
    
    # Get unique references
    unique_internal_refs = {}
    for ref in internal_refs:
        key = ref['hyperlink']
        if key not in unique_internal_refs:
            unique_internal_refs[key] = {
                'text': ref['text'],
                'hyperlink': ref['hyperlink'],
                'target_paragraph': ref['target_paragraph'],
                'occurrences': 1
            }
        else:
            unique_internal_refs[key]['occurrences'] += 1
    
    unique_footnote_refs = {}
    for ref in footnote_refs:
        key = ref['hyperlink']
        if key not in unique_footnote_refs:
            unique_footnote_refs[key] = {
                'text': ref['text'],
                'hyperlink': ref['hyperlink'],
                'footnote_number': ref['footnote_number'],
                'occurrences': 1
            }
        else:
            unique_footnote_refs[key]['occurrences'] += 1
    
    # Resolve internal references (single level, no recursion)
    for ref_key, ref_data in unique_internal_refs.items():
        hyperlink = ref_data.get('hyperlink', '')
        target_para = ref_data.get('target_paragraph', '')
        print(f"  Resolving internal reference to § {target_para}...")
        resolved = resolve_internal_reference(texts, target_paragraph=target_para, hyperlink=hyperlink)
        ref_data['referenced_text'] = resolved['referenced_text']
        ref_data['referenced_internal_refs'] = resolved['referenced_internal_refs']
    
    # Resolve footnote references
    for ref_key, ref_data in unique_footnote_refs.items():
        print(f"  Resolving footnote {ref_data['footnote_number']}...")
        footnote_text = extract_footnote_text(texts, ref_data['footnote_number'])
        ref_data['footnote_text'] = footnote_text
    
    # Create the chunk
    chunk = {
        'chunk_id': f'paragraf-{paragraph_number}.odsek-{subsection_number}',
        'paragraph_number': paragraph_number,
        'subsection_number': subsection_number,
        'text': text_content.strip(),
        'metadata': {
            'internal_references': list(unique_internal_refs.values()),
            'footnote_references': list(unique_footnote_refs.values()),
            'total_internal_refs': len(internal_refs),
            'total_footnote_refs': len(footnote_refs),
            'unique_internal_refs': len(unique_internal_refs),
            'unique_footnote_refs': len(unique_footnote_refs),
            'text_elements_count': len(subsection_texts),
            'start_index': content_start_idx,
            'end_index': end_idx
        }
    }
    
    return chunk


def extract_paragraph_chunk(texts: List[Dict], paragraph_number: str) -> Dict[str, Any]:
    """
    Extract a complete paragraph chunk with all text content and references.
    
    Args:
        texts: List of text elements from Docling JSON
        paragraph_number: Paragraph number to extract (e.g., "50")
    
    Returns:
        Dictionary with chunk data including text, metadata, and references
    """
    # Find the paragraph heading - there are two locations:
    # 1. Structural markers (around index 2681)
    # 2. Actual content (around index 17618)
    
    # First, find the actual content section
    # The heading may be split across multiple text elements
    content_start_idx = None
    
    # Look for "§ 50" followed by the title and then content
    # We need to find the CONTENT section, not the structural markers section
    for i, text_item in enumerate(texts):
        text = text_item.get('text', '')
        # Look for "§ 50" (standalone, not part of 50aa or 50a)
        if text.strip() == f'§ {paragraph_number}':
            # Check if next element is the title
            if i + 1 < len(texts):
                next_text = texts[i + 1].get('text', '')
                # Check if next after that is content (not just structural)
                if i + 2 < len(texts):
                    next_next_text = texts[i + 2].get('text', '')
                    # This is the content section if we see "(1)" followed by actual text
                    if next_next_text.strip() == '(1)':
                        # Check one more ahead for actual content
                        if i + 3 < len(texts):
                            content_check = texts[i + 3].get('text', '')
                            if 'Daňovník' in content_check or len(content_check) > 10:
                                content_start_idx = i
                                break
                    elif 'Daňovník' in next_next_text:
                        content_start_idx = i
                        break
    
    if content_start_idx is None:
        return None
    
    # Find where this paragraph ends (next paragraph or section)
    end_idx = None
    next_para = int(paragraph_number) + 1
    
    for i in range(content_start_idx + 1, len(texts)):
        text = texts[i].get('text', '')
        # Stop at next paragraph or section
        if (f'§ {next_para}' in text or 
            f'§ {paragraph_number}aa' in text or
            f'§ {paragraph_number}a' in text):
            end_idx = i
            break
    
    if end_idx is None:
        end_idx = min(content_start_idx + 300, len(texts))  # Safety limit
    
    # Collect all text elements for this paragraph
    paragraph_texts = []
    internal_refs = []
    footnote_refs = []
    
    for i in range(content_start_idx, end_idx):
        text_item = texts[i]
        text = text_item.get('text', '').strip()
        hyperlink = text_item.get('hyperlink', '')
        label = text_item.get('label', '')
        
        # Skip navigation elements
        if (text and 
            text not in ['plus', 'button-close', 'button-search', 'button-download', 
                        'button-print', 'button-history', 'button-content'] and
            label != 'caption'):
            
            paragraph_texts.append({
                'text': text,
                'hyperlink': hyperlink,
                'label': label,
                'idx': i
            })
            
            # Track references
            if hyperlink:
                if 'paragraf-' in hyperlink and 'poznamky' not in hyperlink:
                    # Internal reference - only if it's to a DIFFERENT paragraph
                    ref_num = hyperlink.split('paragraf-')[-1].split('.')[0]
                    # Filter out references to the same paragraph (internal structure)
                    if ref_num != paragraph_number:
                        internal_refs.append({
                            'text': text,
                            'hyperlink': hyperlink,
                            'target_paragraph': ref_num,
                            'type': 'internal'
                        })
                elif 'poznamky.poznamka' in hyperlink:
                    # Footnote reference
                    fn_num = hyperlink.split('poznamka-')[-1] if 'poznamka-' in hyperlink else None
                    # Only count the footnote number, not the closing paren
                    if text != ')':
                        footnote_refs.append({
                            'text': text,
                            'hyperlink': hyperlink,
                            'footnote_number': fn_num,
                            'type': 'footnote'
                        })
    
    # Reconstruct the full text
    full_text = []
    prev_hyperlink = None
    
    for item in paragraph_texts:
        text = item['text']
        hyperlink = item.get('hyperlink', '')
        
        # Handle structural markers
        if text.strip() in ['(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)', '(8)', '(9)', '(10)', 
                           '(11)', '(12)', '(13)', '(14)', '(15)', '(16)']:
            full_text.append(f'\n\n### {text}\n\n')
        elif text.strip() in ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)']:
            full_text.append(f'\n**{text}** ')
        elif text.strip() in ['1.', '2.', '3.', '4.', '5.']:
            full_text.append(f'    {text} ')
        # Handle references
        elif hyperlink and 'paragraf-' in hyperlink and 'poznamky' not in hyperlink:
            full_text.append(f' [{text}]({hyperlink})')
        elif hyperlink and 'poznamky.poznamka' in hyperlink:
            # Check if next item is closing paren
            next_idx = paragraph_texts.index(item) + 1
            if next_idx < len(paragraph_texts):
                next_item = paragraph_texts[next_idx]
                if (next_item.get('hyperlink') == hyperlink and 
                    next_item.get('text') == ')'):
                    full_text.append(f'<sup>{text}</sup>)')
                    # Mark next item as processed
                    paragraph_texts[next_idx]['processed'] = True
                else:
                    full_text.append(f'<sup>{text}</sup>')
            else:
                full_text.append(f'<sup>{text}</sup>')
        # Regular text
        else:
            if not item.get('processed', False):
                if text and len(text) > 1:
                    if full_text and not full_text[-1].endswith(' ') and not full_text[-1].endswith('\n'):
                        if text[0] not in ['.', ',', ';', ':', '!', '?', ')', ']', '}', '(', '[', '{']:
                            full_text.append(' ')
                    full_text.append(text)
    
    # Clean up the text
    text_content = ''.join(full_text)
    text_content = re.sub(r' +', ' ', text_content)
    text_content = re.sub(r' \n', '\n', text_content)
    text_content = re.sub(r'\n{3,}', '\n\n', text_content)
    
    # Get unique references
    unique_internal_refs = {}
    for ref in internal_refs:
        key = ref['hyperlink']
        if key not in unique_internal_refs:
            unique_internal_refs[key] = {
                'text': ref['text'],
                'hyperlink': ref['hyperlink'],
                'target_paragraph': ref['target_paragraph'],
                'occurrences': 1
            }
        else:
            unique_internal_refs[key]['occurrences'] += 1
    
    unique_footnote_refs = {}
    for ref in footnote_refs:
        key = ref['hyperlink']
        if key not in unique_footnote_refs:
            unique_footnote_refs[key] = {
                'text': ref['text'],
                'hyperlink': ref['hyperlink'],
                'footnote_number': ref['footnote_number'],
                'occurrences': 1
            }
        else:
            unique_footnote_refs[key]['occurrences'] += 1
    
    # Resolve internal references (single level, no recursion)
    for ref_key, ref_data in unique_internal_refs.items():
        hyperlink = ref_data.get('hyperlink', '')
        target_para = ref_data.get('target_paragraph', '')
        print(f"  Resolving internal reference to § {target_para}...")
        resolved = resolve_internal_reference(texts, target_paragraph=target_para, hyperlink=hyperlink)
        ref_data['referenced_text'] = resolved['referenced_text']
        ref_data['referenced_internal_refs'] = resolved['referenced_internal_refs']
    
    # Resolve footnote references
    for ref_key, ref_data in unique_footnote_refs.items():
        print(f"  Resolving footnote {ref_data['footnote_number']}...")
        footnote_text = extract_footnote_text(texts, ref_data['footnote_number'])
        ref_data['footnote_text'] = footnote_text
    
    # Create the chunk
    chunk = {
        'chunk_id': f'paragraf-{paragraph_number}',
        'paragraph_number': paragraph_number,
        'text': text_content.strip(),
        'metadata': {
            'internal_references': list(unique_internal_refs.values()),
            'footnote_references': list(unique_footnote_refs.values()),
            'total_internal_refs': len(internal_refs),
            'total_footnote_refs': len(footnote_refs),
            'unique_internal_refs': len(unique_internal_refs),
            'unique_footnote_refs': len(unique_footnote_refs),
            'text_elements_count': len(paragraph_texts),
            'start_index': content_start_idx,
            'end_index': end_idx
        }
    }
    
    return chunk


def chunk_paragraph_subsections(texts: List[Dict], paragraph_number: str) -> List[Dict[str, Any]]:
    """
    Extract chunks for all subsections of a paragraph.
    
    Args:
        texts: List of text elements from Docling JSON
        paragraph_number: Paragraph number (e.g., "50")
    
    Returns:
        List of subsection chunks
    """
    chunks = []
    
    # Find all subsections for this paragraph
    subsection_numbers = set()
    for text_item in texts:
        hyperlink = text_item.get('hyperlink', '')
        if hyperlink and f'paragraf-{paragraph_number}.odsek-' in hyperlink:
            # Extract subsection number
            if '.odsek-' in hyperlink:
                odsek_part = hyperlink.split('.odsek-')[1].split('.')[0]
                if odsek_part.isdigit():
                    subsection_numbers.add(odsek_part)
    
    subsection_numbers = sorted(subsection_numbers, key=lambda x: int(x))
    print(f'Found {len(subsection_numbers)} subsections for § {paragraph_number}')
    
    # Extract each subsection
    for odsek_num in subsection_numbers:
        print(f'Processing § {paragraph_number}, odsek {odsek_num}...')
        chunk = extract_subsection_chunk(texts, paragraph_number, odsek_num)
        if chunk:
            chunks.append(chunk)
    
    return chunks


def chunk_all_paragraphs(texts: List[Dict]) -> List[Dict[str, Any]]:
    """
    Extract chunks for all paragraphs in the document.
    
    Args:
        texts: List of text elements from Docling JSON
    
    Returns:
        List of paragraph chunks
    """
    chunks = []
    
    # Find all paragraph headings
    paragraph_numbers = []
    for i, text_item in enumerate(texts):
        text = text_item.get('text', '')
        # Match pattern like "§ 50" but not "§ 50a" or "§ 50aa"
        match = re.search(r'§ (\d+)(?:\s|$)', text)
        if match:
            para_num = match.group(1)
            # Check if it's a main paragraph (not a sub-paragraph)
            if f'§ {para_num} ' in text or f'§ {para_num}\n' in text:
                if para_num not in paragraph_numbers:
                    paragraph_numbers.append(para_num)
    
    print(f'Found {len(paragraph_numbers)} paragraphs to chunk')
    
    # Extract each paragraph
    for para_num in paragraph_numbers:
        print(f'Processing § {para_num}...')
        chunk = extract_paragraph_chunk(texts, para_num)
        if chunk:
            chunks.append(chunk)
    
    return chunks


def main():
    import sys
    
    # Load the Docling JSON
    input_file = 'output/595:2003 Z. z. - Zákon o dani z príjmov_basic.json'
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    print(f'Loading Docling JSON from: {input_file}')
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = data.get('texts', [])
    print(f'Loaded {len(texts)} text elements\n')
    
    # Check what to process
    if len(sys.argv) > 2 and sys.argv[2] == '--all':
        print('Processing ALL paragraphs...\n')
        chunks = chunk_all_paragraphs(texts)
        output_file = 'output/paragraph_chunks_all.json'
    elif len(sys.argv) > 2 and sys.argv[2] == '--paragraph':
        # Process a single full paragraph
        para_num = sys.argv[3] if len(sys.argv) > 3 else '50'
        print(f'Processing full paragraph § {para_num}...\n')
        chunk = extract_paragraph_chunk(texts, para_num)
        if chunk:
            chunks = [chunk]
        else:
            print(f'Error: Could not find paragraph § {para_num}')
            return
        output_file = f'output/paragraph_chunk_{para_num}.json'
    elif len(sys.argv) > 2 and sys.argv[2] == '--subsections':
        # Process subsections of a paragraph
        para_num = sys.argv[3] if len(sys.argv) > 3 else '50'
        print(f'Processing subsections of § {para_num}...\n')
        chunks = chunk_paragraph_subsections(texts, para_num)
        output_file = f'output/subsection_chunks_{para_num}.json'
    else:
        # Default: process subsections of § 50
        print('Processing subsections of § 50 (test mode)...\n')
        chunks = chunk_paragraph_subsections(texts, '50')
        output_file = 'output/subsection_chunks_50.json'
        
        if not chunks:
            print('Error: Could not find subsections for § 50')
            return
    
    # Create output structure
    output = {
        'source_file': input_file,
        'chunking_method': 'paragraph_based',
        'total_chunks': len(chunks),
        'chunks': chunks
    }
    
    # Check for format parameter
    output_format = 'json'
    if len(sys.argv) > 2:
        # Check if format is specified (e.g., --format md or --format markdown)
        for i, arg in enumerate(sys.argv):
            if arg in ['--format', '-f'] and i + 1 < len(sys.argv):
                format_arg = sys.argv[i + 1].lower()
                if format_arg in ['md', 'markdown']:
                    output_format = 'markdown'
                break
    
    # Save to JSON
    print(f'\nSaving {len(chunks)} chunk(s) to: {output_file}')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # Also save to markdown if requested
    if output_format == 'markdown':
        # Import markdown conversion functions
        try:
            from json_to_markdown import process_json_file
            md_output_file = output_file.replace('.json', '.md')
            print(f'\nConverting to markdown: {md_output_file}')
            process_json_file(output_file, md_output_file)
        except ImportError:
            print('Warning: Could not import json_to_markdown module. Markdown output skipped.')
    
    print(f'\n✓ Successfully created {len(chunks)} chunk(s)')
    print(f'\nSummary:')
    for chunk in chunks:
        meta = chunk['metadata']
        print(f"  § {chunk['paragraph_number']}:")
        print(f"    - Text length: {len(chunk['text'])} characters")
        print(f"    - Internal references: {meta['unique_internal_refs']} unique ({meta['total_internal_refs']} total)")
        print(f"    - Footnote references: {meta['unique_footnote_refs']} unique ({meta['total_footnote_refs']} total)")


if __name__ == '__main__':
    main()

