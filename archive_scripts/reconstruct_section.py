#!/usr/bin/env python3
"""
Reconstruct a specific section (e.g., § 50) from Docling JSON output.

This script addresses the issue where hierarchical chunking creates very granular
chunks that only contain structural markers, not the actual text content.
"""

import json
import sys
from pathlib import Path


def reconstruct_section(json_file, section_number):
    """Reconstruct a complete section from Docling JSON."""
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = data.get('texts', [])
    
    # Find the section heading
    section_idx = None
    for i, text_item in enumerate(texts):
        text = text_item.get('text', '')
        if f'§ {section_number}' in text and 'Použitie' in text:
            section_idx = i
            break
    
    if section_idx is None:
        print(f"Section § {section_number} not found")
        return None
    
    print(f"Found § {section_number} heading at index {section_idx}")
    print("="*70)
    
    # Collect all text elements that belong to this section
    # We'll collect sequential text elements until we hit the next major section
    section_texts = []
    i = section_idx
    
    while i < len(texts) and i < section_idx + 500:  # Look at next 500 items
        text_item = texts[i]
        text = text_item.get('text', '').strip()
        hyperlink = text_item.get('hyperlink', '')
        label = text_item.get('label', '')
        
        # Stop if we hit the next major section
        try:
            next_section = int(section_number) + 1
            if (f'§ {section_number}' not in text and 
                f'paragraf-{section_number}' not in hyperlink and 
                f'§ {next_section}' in text):
                break
        except ValueError:
            # If section_number is not a number, just check for next section pattern
            if (f'§ {section_number}' not in text and 
                f'paragraf-{section_number}' not in hyperlink and 
                '§ 51' in text):
                break
        
        # Skip navigation/caption elements
        if (text and 
            text not in ['plus', 'button-close', 'button-search', 'button-download', 
                        'button-print', 'button-history', 'button-content'] and
            label != 'caption'):
            section_texts.append({
                'text': text,
                'hyperlink': hyperlink,
                'label': label,
                'idx': i
            })
        
        i += 1
    
    # Now reconstruct with proper formatting
    print(f"\n=== RECONSTRUCTED § {section_number} ===\n")
    
    reconstructed = []
    current_subsection = None
    current_letter = None
    current_number = None
    
    for item in section_texts:
        text = item['text']
        hyperlink = item.get('hyperlink', '')
        
        # Main heading
        if hyperlink == f'#paragraf-{section_number}':
            if 'Použitie' in text or len(reconstructed) == 0:
                reconstructed.append(f"\n## {text}\n")
        
        # Subsection (1), (2), etc.
        elif f'paragraf-{section_number}.odsek' in hyperlink:
            if 'pismeno' not in hyperlink and 'bod' not in hyperlink:
                # This is a subsection number
                reconstructed.append(f"\n### {text}\n")
                current_subsection = hyperlink
                current_letter = None
                current_number = None
            elif 'bod' in hyperlink:
                # This is a numbered item (1., 2., etc.)
                if current_number != hyperlink:
                    reconstructed.append(f"\n    {text} ")
                    current_number = hyperlink
                else:
                    reconstructed.append(f"{text} ")
            elif 'pismeno' in hyperlink:
                # This is a letter (a), b), etc.)
                if current_letter != hyperlink:
                    reconstructed.append(f"\n**{text}** ")
                    current_letter = hyperlink
                else:
                    reconstructed.append(f"{text} ")
            else:
                reconstructed.append(f"{text} ")
        
        # Regular text content
        else:
            # Add space before if needed
            if reconstructed and not reconstructed[-1].endswith(' ') and not reconstructed[-1].endswith('\n'):
                reconstructed.append(' ')
            reconstructed.append(text)
    
    result = ''.join(reconstructed)
    print(result[:5000])  # Print first 5000 chars
    if len(result) > 5000:
        print("\n... (truncated)")
    
    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python reconstruct_section.py <section_number> [json_file]")
        print("Example: python reconstruct_section.py 50")
        sys.exit(1)
    
    section_num = sys.argv[1]
    json_file = sys.argv[2] if len(sys.argv) > 2 else 'output/595:2003 Z. z. - Zákon o dani z príjmov_basic.json'
    
    reconstruct_section(json_file, section_num)

