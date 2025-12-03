#!/usr/bin/env python3
"""
Reconstruct § 50 from Docling JSON with all text content, internal references, and footnotes.
Shows how references are captured in the Docling output.
"""

import json
from collections import defaultdict

# Load the original Docling JSON
with open('output/595:2003 Z. z. - Zákon o dani z príjmov_basic.json', 'r') as f:
    data = json.load(f)

texts = data.get('texts', [])

# Find § 50 content section (starting around index 17618)
start_idx = 17618
end_idx = None

# Find where § 50 ends (look for § 50aa or § 51)
for i in range(start_idx, min(len(texts), start_idx + 300)):
    text = texts[i].get('text', '')
    if '§ 50aa' in text or ('§ 51' in text and i > start_idx + 50):
        end_idx = i
        break

if not end_idx:
    end_idx = start_idx + 250

print('Reconstructing § 50 with all references:')
print('='*80)

# Collect all text elements
section_50_content = []
for i in range(start_idx, end_idx):
    text_item = texts[i]
    text = text_item.get('text', '').strip()
    hyperlink = text_item.get('hyperlink', '')
    label = text_item.get('label', '')
    
    if (text and 
        text not in ['plus', 'button-close'] and
        label != 'caption'):
        section_50_content.append({
            'text': text,
            'hyperlink': hyperlink,
            'label': label,
            'idx': i
        })

print(f'Collected {len(section_50_content)} text elements\n')

# Track references for analysis
internal_refs = []
footnote_refs = []
footnote_pairs = defaultdict(list)  # Group footnote number and closing paren

# Reconstruct with proper formatting
reconstructed = []
prev_item = None

for item in section_50_content:
    text = item['text']
    hyperlink = item.get('hyperlink', '')
    
    # Track references
    if hyperlink:
        if 'paragraf-' in hyperlink and 'poznamky' not in hyperlink:
            internal_refs.append({'text': text, 'hyperlink': hyperlink, 'idx': item['idx']})
        elif 'poznamky.poznamka' in hyperlink:
            footnote_refs.append({'text': text, 'hyperlink': hyperlink, 'idx': item['idx']})
            # Extract footnote number
            fn_num = hyperlink.split('poznamka-')[-1] if 'poznamka-' in hyperlink else None
            if fn_num:
                footnote_pairs[fn_num].append(item)
    
    # Main heading
    if '§ 50' in text and 'Použitie' in text and '50aa' not in text and '50a' not in text:
        reconstructed.append(f'\n## {text}\n\n')
    # Subsection markers
    elif text.strip() in ['(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)', '(8)', '(9)', '(10)', 
                     '(11)', '(12)', '(13)', '(14)', '(15)', '(16)']:
        reconstructed.append(f'\n### {text}\n\n')
    # Letter markers
    elif text.strip() in ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)']:
        reconstructed.append(f'\n**{text}** ')
    # Numbered items
    elif text.strip() in ['1.', '2.', '3.', '4.', '5.']:
        reconstructed.append(f'    {text} ')
    # Internal references (sections) - format as markdown link
    elif hyperlink and 'paragraf-' in hyperlink and 'poznamky' not in hyperlink:
        ref_text = text if text else hyperlink.split('#')[-1]
        reconstructed.append(f'[{text}]({hyperlink})')
    # Footnote references - handle number and closing paren together
    elif hyperlink and 'poznamky.poznamka' in hyperlink:
        # Check if next item is the closing paren for this footnote
        fn_num = hyperlink.split('poznamka-')[-1] if 'poznamka-' in hyperlink else None
        if text.isdigit() or text.replace(')', '').isdigit():
            # This is the footnote number
            reconstructed.append(f'<sup>{text.replace(")", "")}</sup>')
        elif text == ')':
            # This is the closing paren - add it
            reconstructed.append(')')
        else:
            reconstructed.append(f'[{text}]({hyperlink})')
    # All other text (the actual content)
    else:
        if text and len(text) > 1:
            # Smart spacing
            if reconstructed:
                last = reconstructed[-1]
                needs_space = (not last.endswith(' ') and 
                             not last.endswith('\n') and 
                             not last.endswith(')') and
                             not last.endswith(']') and
                             text[0] not in ['.', ',', ';', ':', '!', '?', ')', ']', '}', '(', '[', '{'])
                if needs_space:
                    reconstructed.append(' ')
            reconstructed.append(text)
    
    prev_item = item

result = ''.join(reconstructed)

# Clean up double spaces and formatting issues
import re
result = re.sub(r'\s+', ' ', result)  # Multiple spaces to single
result = re.sub(r' \n', '\n', result)  # Space before newline
result = re.sub(r'\n{3,}', '\n\n', result)  # Multiple newlines to double

print('=== RECONSTRUCTED § 50 (COMPLETE WITH ALL REFERENCES) ===\n')
print(result)
print('\n' + '='*80)
print(f'Total length: {len(result)} characters')

# Reference analysis
print(f'\n=== REFERENCE ANALYSIS ===\n')

# Internal references
print(f'**Internal References Found: {len(internal_refs)}**')
internal_refs_unique = {}
for ref in internal_refs:
    key = ref['hyperlink']
    if key not in internal_refs_unique:
        internal_refs_unique[key] = ref['text']
    
print('\nUnique internal references:')
for hyperlink, ref_text in sorted(internal_refs_unique.items()):
    print(f'  - [{ref_text}]({hyperlink})')
    # Count occurrences
    count = sum(1 for r in internal_refs if r['hyperlink'] == hyperlink)
    if count > 1:
        print(f'    (appears {count} times)')

# Footnote references
print(f'\n**Footnote References Found: {len(footnote_refs)}**')
footnote_refs_unique = {}
for ref in footnote_refs:
    key = ref['hyperlink']
    if key not in footnote_refs_unique:
        footnote_refs_unique[key] = ref['text']
    
print('\nUnique footnote references:')
for hyperlink, ref_text in sorted(footnote_refs_unique.items()):
    fn_num = hyperlink.split('poznamka-')[-1] if 'poznamka-' in hyperlink else '?'
    print(f'  - Footnote {fn_num}: [{ref_text}]({hyperlink})')
    # Count occurrences
    count = sum(1 for r in footnote_refs if r['hyperlink'] == hyperlink)
    if count > 1:
        print(f'    (appears {count} times)')

print(f'\n=== HOW REFERENCES ARE CAPTURED ===\n')
print('1. **Internal References**:')
print('   - Captured as separate text elements with hyperlink property')
print('   - Format: `{"text": "§ 33", "hyperlink": "#paragraf-33"}`')
print('   - Can reference sections (§ 33), subsections (§ 39 ods. 7), etc.')
print('   - All internal references are preserved with their hyperlinks\n')

print('2. **Footnote References**:')
print('   - Captured as TWO separate text elements:')
print('     a) The footnote number (e.g., "136f")')
print('     b) The closing parenthesis ")"')
print('   - Both have the same hyperlink: `#poznamky.poznamka-136f`')
print('   - Format: `{"text": "136f", "hyperlink": "#poznamky.poznamka-136f"}`')
print('   - Format: `{"text": ")", "hyperlink": "#poznamky.poznamka-136f"}`')
print('   - This allows precise reconstruction of footnote citations\n')

print('3. **Reference Preservation**:')
print('   - All hyperlinks are preserved in the Docling JSON output')
print('   - References can be extracted and used for navigation or analysis')
print('   - The hierarchical chunking preserves hyperlinks in chunk metadata')



