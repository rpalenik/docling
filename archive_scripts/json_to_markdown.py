#!/usr/bin/env python3
"""
Convert paragraph chunk JSON to markdown representation for LLM consumption.
"""

import json
import sys
from typing import Dict, List, Any


def format_paragraph_text(chunk: Dict[str, Any]) -> str:
    """
    Format the main paragraph text with heading.
    
    Args:
        chunk: Chunk dictionary from JSON
    
    Returns:
        Formatted markdown string
    """
    para_num = chunk.get('paragraph_number', '')
    subsection_num = chunk.get('subsection_number', '')
    text = chunk.get('text', '')
    
    # Format heading based on whether it's a subsection or full paragraph
    if subsection_num:
        # Subsection chunk
        title = f"§ {para_num}, odsek {subsection_num}"
    else:
        # Full paragraph chunk
        # Extract title from text if available
        lines = text.split('\n')
        title = None
        if lines and '§' in lines[0] and 'Použitie' in lines[0]:
            title = lines[0].strip()
            # Remove markdown link formatting if present
            if '](' in title:
                import re
                title = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', title)
        
        if not title:
            title = f"§ {para_num}"
    
    markdown = f"## {title}\n\n"
    markdown += text
    if not text.endswith('\n'):
        markdown += "\n"
    
    return markdown


def format_internal_references(internal_refs: List[Dict[str, Any]], indent_level: int = 0) -> str:
    """
    Format internal references section with resolved text.
    
    Args:
        internal_refs: List of internal reference dictionaries
        indent_level: Current indentation level for nested references
    
    Returns:
        Formatted markdown string
    """
    if not internal_refs:
        return ""
    
    markdown = ""
    indent = "  " * indent_level
    heading_level = "###" if indent_level == 0 else "####"
    
    for ref in internal_refs:
        ref_text = ref.get('text', '')
        target_para = ref.get('target_paragraph', '')
        referenced_text = ref.get('referenced_text', '')
        nested_refs = ref.get('referenced_internal_refs', [])
        
        # Format reference heading
        if ref_text and '§' in ref_text:
            heading = ref_text
        else:
            heading = f"§ {target_para}"
        
        markdown += f"\n{indent}{heading_level} {heading}\n\n"
        
        # Include referenced text
        if referenced_text:
            # Add indentation to referenced text
            ref_lines = referenced_text.split('\n')
            indented_ref_text = '\n'.join([f"{indent}  {line}" if line.strip() else line 
                                          for line in ref_lines])
            markdown += f"{indent}{indented_ref_text}\n\n"
        else:
            markdown += f"{indent}*[Referenced text not available]*\n\n"
        
        # Include nested references if present
        if nested_refs:
            markdown += f"{indent}**Referenced sections within this section:**\n\n"
            nested_md = format_internal_references(nested_refs, indent_level + 1)
            markdown += nested_md
    
    return markdown


def format_footnote_references(footnote_refs: List[Dict[str, Any]]) -> str:
    """
    Format footnote references section.
    
    Args:
        footnote_refs: List of footnote reference dictionaries
    
    Returns:
        Formatted markdown string
    """
    if not footnote_refs:
        return ""
    
    markdown = "\n## Footnotes\n\n"
    
    for ref in footnote_refs:
        fn_num = ref.get('footnote_number', '')
        fn_text = ref.get('footnote_text', '')
        occurrences = ref.get('occurrences', 1)
        
        markdown += f"### Footnote {fn_num}\n\n"
        
        if fn_text:
            markdown += f"{fn_text}\n\n"
        else:
            markdown += "*[Footnote text not available in Docling JSON output]*\n\n"
        
        if occurrences > 1:
            markdown += f"*Referenced {occurrences} times in the text.*\n\n"
    
    return markdown


def chunk_to_markdown(chunk: Dict[str, Any]) -> str:
    """
    Convert a single chunk to markdown format.
    
    Args:
        chunk: Chunk dictionary from JSON
    
    Returns:
        Complete markdown string
    """
    markdown_parts = []
    
    # Main paragraph text
    markdown_parts.append(format_paragraph_text(chunk))
    
    # Internal references section
    metadata = chunk.get('metadata', {})
    internal_refs = metadata.get('internal_references', [])
    
    if internal_refs:
        markdown_parts.append("\n---\n")
        markdown_parts.append("\n## Referenced Sections\n\n")
        markdown_parts.append("*The following sections are referenced in the text above. Their full text is provided for context.*\n\n")
        markdown_parts.append(format_internal_references(internal_refs))
    
    # Footnote references section
    footnote_refs = metadata.get('footnote_references', [])
    
    if footnote_refs:
        markdown_parts.append("\n---\n")
        markdown_parts.append(format_footnote_references(footnote_refs))
    
    return "\n".join(markdown_parts)


def process_json_file(input_file: str, output_file: str = None) -> str:
    """
    Process JSON file and convert chunks to markdown.
    
    Args:
        input_file: Path to input JSON file
        output_file: Optional path to output markdown file (if None, print to stdout)
    
    Returns:
        Complete markdown string
    """
    print(f"Loading JSON from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data.get('chunks', [])
    print(f"Found {len(chunks)} chunk(s) to process\n")
    
    all_markdown = []
    
    # Add header
    all_markdown.append("# Legal Document Chunks\n\n")
    all_markdown.append(f"*Generated from: {data.get('source_file', 'unknown')}*\n")
    all_markdown.append(f"*Chunking method: {data.get('chunking_method', 'unknown')}*\n")
    all_markdown.append(f"*Total chunks: {data.get('total_chunks', 0)}*\n\n")
    all_markdown.append("---\n\n")
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        if i > 0:
            all_markdown.append("\n\n---\n\n")
        
        para_num = chunk.get('paragraph_number', '?')
        subsection_num = chunk.get('subsection_number', '')
        if subsection_num:
            print(f"Processing chunk {i+1}/{len(chunks)}: § {para_num}, odsek {subsection_num}")
        else:
            print(f"Processing chunk {i+1}/{len(chunks)}: § {para_num}")
        chunk_md = chunk_to_markdown(chunk)
        all_markdown.append(chunk_md)
    
    full_markdown = "\n".join(all_markdown)
    
    # Output
    if output_file:
        print(f"\nSaving markdown to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_markdown)
        print(f"✓ Successfully saved {len(chunks)} chunk(s) to markdown")
    else:
        print(full_markdown)
    
    return full_markdown


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert paragraph chunk JSON to markdown for LLM consumption'
    )
    parser.add_argument(
        'input_file',
        help='Input JSON file path'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output markdown file path (default: print to stdout)'
    )
    
    args = parser.parse_args()
    
    process_json_file(args.input_file, args.output)


if __name__ == '__main__':
    main()

