#!/usr/bin/env python3
"""
Reconstruct § 50 from Docling JSON with all text content included.
"""

import json

# Load the original Docling JSON
with open('output/595:2003 Z. z. - Zákon o dani z príjmov_basic.json', 'r') as f:
    data = json.load(f)

texts = data.get('texts', [])

# Find § 50 heading
paragraf_50_idx = None
for i, text_item in enumerate(texts):
    if '§ 50' in text_item.get('text', '') and 'Použitie' in text_item.get('text', ''):
        paragraf_50_idx = i
        break

print(f'Found § 50 heading at index {paragraf_50_idx}')
print('='*80)

# Collect all text elements that belong to § 50
# We need to collect BOTH structural markers AND the actual text content
section_50_items = []

i = paragraf_50_idx
while i < len(texts) and i < paragraf_50_idx + 500:
    text_item = texts[i]
    text = text_item.get('text', '').strip()
    hyperlink = text_item.get('hyperlink', '')
    label = text_item.get('label', '')
    
    # Stop if we hit the next major section
    if (f'§ 50' not in text and 
        f'paragraf-50' not in hyperlink and 
        '§ 51' in text):
        break
    
    # Collect all text elements (both structural and content)
    if (text and 
        text not in ['plus', 'button-close', 'button-search', 'button-download', 
                    'button-print', 'button-history', 'button-content'] and
        label != 'caption'):
        section_50_items.append({
            'text': text,
            'hyperlink': hyperlink,
            'label': label,
            'idx': i
        })
    
    i += 1

print(f'Collected {len(section_50_items)} text elements')
print('\n=== RECONSTRUCTED § 50 (COMPLETE WITH TEXT CONTENT) ===\n')

# Reconstruct with proper formatting
reconstructed = []
current_subsection = None
current_letter = None
current_number = None

for item in section_50_items:
    text = item['text']
    hyperlink = item.get('hyperlink', '')
    
    # Main heading
    if hyperlink == '#paragraf-50' and 'Použitie' in text:
        reconstructed.append(f'\n## {text}\n\n')
    
    # Subsection markers (1), (2), etc.
    elif hyperlink and 'paragraf-50.odsek' in hyperlink:
        if 'pismeno' not in hyperlink and 'bod' not in hyperlink:
            # Subsection number
            reconstructed.append(f'\n### {text}\n\n')
            current_subsection = hyperlink
            current_letter = None
            current_number = None
        elif 'bod' in hyperlink:
            # Numbered item (1., 2., etc.)
            if current_number != hyperlink:
                reconstructed.append(f'\n    {text} ')
                current_number = hyperlink
            else:
                reconstructed.append(f'{text} ')
        elif 'pismeno' in hyperlink:
            # Letter (a), b), etc.)
            if current_letter != hyperlink:
                reconstructed.append(f'\n**{text}** ')
                current_letter = hyperlink
            else:
                reconstructed.append(f'{text} ')
        else:
            reconstructed.append(f'{text} ')
    
    # Regular text content (no hyperlink or different hyperlink)
    else:
        # This is the actual paragraph text
        if text and len(text) > 1:
            # Add space if needed (but not after punctuation)
            if reconstructed:
                last_char = reconstructed[-1][-1] if reconstructed[-1] else ''
                if (last_char not in [' ', '\n', '(', '.', ',', ';', ':', '!', '?'] and 
                    text[0] not in ['.', ',', ';', ':', '!', '?', ')']):
                    reconstructed.append(' ')
            reconstructed.append(text)

result = ''.join(reconstructed)
print(result)
print('\n' + '='*80)
print(f'Total length: {len(result)} characters')



