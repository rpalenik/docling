#!/usr/bin/env python3
"""
Chunk to Markdown Converter

Converts parsed JSON law documents into markdown chunks with proper visual indentation.
Chunks at the "pismeno" level (or lowest available level) with TAB-based hierarchy.

Usage:
    from chunk_to_markdown import chunk_document, format_chunks_to_file
    
    with open("document.json") as f:
        data = json.load(f)
    
    chunks = chunk_document(data)
    format_chunks_to_file(chunks, "output.md")
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional


# Indentation constants
TAB = "\t"  # Use actual TAB character
INDENT_ODSEK = 1      # 1 TAB for odsek
INDENT_PISMENO = 2    # 2 TABs for pismeno
INDENT_SUBITEM = 3    # 3 TABs for subitems


def indent_lines(text: str, level: int) -> str:
    """
    Indent all lines of text by the specified number of TABs.
    
    Args:
        text: Text to indent
        level: Number of TABs to add
        
    Returns:
        Indented text
    """
    if not text:
        return text
    
    indent = TAB * level
    lines = text.split('\n')
    return '\n'.join(f"{indent}{line}" if line.strip() else line for line in lines)


def format_marker_text(marker: str, text: str, indent_level: int) -> str:
    """
    Format a marker and text with proper indentation.
    Multi-line text is aligned with the text start (not the marker).
    
    Args:
        marker: The marker (e.g., "(1)", "a)", "1.")
        text: The text content
        indent_level: Number of TABs for indentation
        
    Returns:
        Formatted string with marker and indented text
    """
    indent = TAB * indent_level
    
    if not text:
        return f"{indent}{marker}"
    
    lines = text.split('\n')
    if not lines:
        return f"{indent}{marker}"
    
    # First line: indent + marker + space + text
    result = [f"{indent}{marker} {lines[0].strip()}"]
    
    # Subsequent lines: align with text start
    # Calculate continuation indent (indent + marker width + 1 space)
    continuation_indent = indent + " " * (len(marker) + 1)
    
    for line in lines[1:]:
        stripped = line.strip()
        if stripped:
            result.append(f"{continuation_indent}{stripped}")
        else:
            result.append("")  # Preserve empty lines
    
    return '\n'.join(result)


def parse_embedded_pismenos(intro_text: str) -> str:
    """
    Parse intro_text that contains embedded pismenos and format with proper indentation.
    
    Handles two patterns:
    1. Explicit markers: a), b), c) etc.
    2. Definition terms (like § 2 "Základné pojmy"): term followed by definition, 
       with numbered subitems 1., 2., 1a., 1b. etc.
    
    Args:
        intro_text: The paragraph intro text that may contain embedded pismenos
        
    Returns:
        Formatted text with proper indentation for embedded pismenos
    """
    import re
    
    if not intro_text:
        return intro_text
    
    lines = intro_text.split('\n')
    result = []
    current_indent = 0  # 0 = paragraph intro, 1 = pismeno/term level, 2 = numbered subitem
    in_definition_section = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            result.append("")
            continue
        
        # Check for explicit pismeno marker at start (a), b), etc.)
        pismeno_match = re.match(r'^([a-z])\)\s*(.*)$', stripped)
        if pismeno_match:
            letter = pismeno_match.group(1)
            text = pismeno_match.group(2)
            result.append(f"{TAB}{letter}) {text}")
            current_indent = 1
            in_definition_section = True
            continue
        
        # Check for numbered subitem (1., 2., 1a., 1b., etc.) - must be standalone or at start
        numbered_match = re.match(r'^(\d+[a-z]?)\.\s*$', stripped)
        if numbered_match:
            # Standalone number like "1." on its own line
            number = numbered_match.group(1)
            result.append(f"{TAB}{TAB}{number}.")
            current_indent = 2
            continue
        
        # Check for numbered item with text
        numbered_text_match = re.match(r'^(\d+[a-z]?)\.\s+(.+)$', stripped)
        if numbered_text_match:
            number = numbered_text_match.group(1)
            text = numbered_text_match.group(2)
            result.append(f"{TAB}{TAB}{number}. {text}")
            current_indent = 2
            continue
        
        # Check if this looks like a definition term (short line ending with comma or starting a definition)
        # Definition terms are typically: single word or short phrase, often followed by definition text
        is_definition_term = (
            in_definition_section and 
            current_indent <= 1 and
            len(stripped) < 80 and
            (stripped.endswith(',') or 
             re.match(r'^[a-záäčďéíľĺňóôŕšťúýž\s]+$', stripped.split()[0] if stripped.split() else '', re.IGNORECASE))
        )
        
        # Detect start of definitions section (after intro like "Na účely tohto zákona sa rozumie")
        if 'sa rozumie' in stripped or 'rozumie sa' in stripped:
            result.append(stripped)
            in_definition_section = True
            current_indent = 0
            continue
        
        # Handle based on context
        if current_indent == 2:
            # Continuation of numbered subitem
            result.append(f"{TAB}{TAB}   {stripped}")
        elif current_indent == 1 or (in_definition_section and not stripped[0].isupper()):
            # Continuation of pismeno/term or lowercase continuation
            result.append(f"{TAB}   {stripped}")
        elif in_definition_section and stripped[0].islower():
            # New definition term (starts with lowercase in Slovak)
            result.append(f"{TAB}{stripped}")
            current_indent = 1
        else:
            # Regular paragraph text
            result.append(stripped)
            if in_definition_section:
                current_indent = 1
    
    return '\n'.join(result)


def format_table(table: Dict[str, Any], indent_level: int = 0) -> str:
    """
    Format a table with proper indentation.
    
    Args:
        table: Table dictionary with 'markdown' key
        indent_level: Number of TABs for indentation
        
    Returns:
        Formatted table markdown
    """
    markdown = table.get('markdown', '')
    if not markdown:
        return ""
    
    indent = TAB * indent_level
    lines = markdown.split('\n')
    indented_lines = [f"{indent}{line}" if line.strip() else line for line in lines]
    
    return '\n'.join(indented_lines)


def format_subitem(subitem: Dict[str, Any]) -> str:
    """
    Format a subitem with proper indentation.
    
    Args:
        subitem: Subitem dictionary
        
    Returns:
        Formatted subitem string
    """
    marker = subitem.get('marker', '')
    text = subitem.get('text', '')
    
    return format_marker_text(marker, text, INDENT_SUBITEM)


def format_pismeno(pismeno: Dict[str, Any], include_subitems: bool = True) -> str:
    """
    Format a pismeno with proper indentation.
    
    Args:
        pismeno: Pismeno dictionary
        include_subitems: Whether to include subitems
        
    Returns:
        Formatted pismeno string
    """
    parts = []
    
    marker = pismeno.get('marker', '')
    text = pismeno.get('text', '')
    
    # Main pismeno text
    parts.append(format_marker_text(marker, text, INDENT_PISMENO))
    
    # Subitems
    if include_subitems:
        subitems = pismeno.get('subitems', [])
        for subitem in subitems:
            parts.append(format_subitem(subitem))
    
    # Tables at pismeno level
    tables = pismeno.get('tables', [])
    for table in tables:
        if table:  # Could be just index reference
            table_md = format_table(table, INDENT_PISMENO)
            if table_md:
                parts.append(table_md)
    
    return '\n'.join(parts)


def format_odsek(odsek: Dict[str, Any], include_pismenos: bool = True) -> str:
    """
    Format an odsek with proper indentation.
    
    Args:
        odsek: Odsek dictionary
        include_pismenos: Whether to include pismenos
        
    Returns:
        Formatted odsek string
    """
    parts = []
    
    marker = odsek.get('marker', '')
    text = odsek.get('text', '')
    
    # Main odsek text
    parts.append(format_marker_text(marker, text, INDENT_ODSEK))
    
    # Tables at odsek level (before pismenos)
    tables = odsek.get('tables', [])
    for table in tables:
        if table:
            table_md = format_table(table, INDENT_ODSEK)
            if table_md:
                parts.append(table_md)
    
    # Pismenos
    if include_pismenos:
        pismenos = odsek.get('pismenos', [])
        for pismeno in pismenos:
            parts.append(format_pismeno(pismeno))
    
    return '\n'.join(parts)


def format_paragraph(paragraph: Dict[str, Any], include_odseks: bool = True) -> str:
    """
    Format a paragraph with proper structure.
    
    Args:
        paragraph: Paragraph dictionary
        include_odseks: Whether to include odseks
        
    Returns:
        Formatted paragraph string
    """
    parts = []
    
    # Paragraph header
    marker = paragraph.get('marker', '')
    parts.append(f"\n## {marker}")
    
    # Intro text (title/description)
    intro_text = paragraph.get('intro_text', '')
    if intro_text:
        # If intro is short (like a title), use H3
        if len(intro_text) < 100 and '\n' not in intro_text:
            parts.append(f"### {intro_text}")
        else:
            parts.append(f"\n{intro_text}")
    
    # Tables at paragraph level
    tables = paragraph.get('tables', [])
    for table in tables:
        if table:
            table_md = format_table(table, 0)
            if table_md:
                parts.append(f"\n{table_md}")
    
    # Odseks
    if include_odseks:
        odseks = paragraph.get('odseks', [])
        for odsek in odseks:
            parts.append("")  # Empty line before odsek
            parts.append(format_odsek(odsek))
    
    return '\n'.join(parts)


def format_part(part: Dict[str, Any], include_paragraphs: bool = True) -> str:
    """
    Format a part with proper structure.
    
    Args:
        part: Part dictionary
        include_paragraphs: Whether to include paragraphs
        
    Returns:
        Formatted part string
    """
    parts = []
    
    # Part header
    title = part.get('title', '')
    title_text = part.get('title_text', '')
    
    if title:
        parts.append(f"\n# {title}")
    if title_text and title_text != title:
        parts.append(f"## {title_text}")
    
    parts.append("\n---")
    
    # Paragraphs
    if include_paragraphs:
        paragraphs = part.get('paragraphs', [])
        for paragraph in paragraphs:
            parts.append(format_paragraph(paragraph))
            parts.append("\n---")
    
    return '\n'.join(parts)


def create_chunk_metadata(
    part: Optional[Dict] = None,
    paragraph: Optional[Dict] = None,
    odsek: Optional[Dict] = None,
    pismeno: Optional[Dict] = None,
    subitem: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Create metadata for a chunk based on its location in the hierarchy.
    
    Returns:
        Metadata dictionary with path information
    """
    # Determine chunk level
    if subitem:
        level = "subitem"
    elif pismeno:
        level = "pismeno"
    elif odsek:
        level = "odsek"
    elif paragraph:
        level = "paragraph"
    elif part:
        level = "part"
    else:
        level = "unknown"
    
    # Build chunk ID
    id_parts = []
    if part:
        id_parts.append(part.get('id', 'part'))
    if paragraph:
        id_parts.append(paragraph.get('id', 'paragraph'))
    if odsek:
        id_parts.append(odsek.get('id', 'odsek'))
    if pismeno:
        id_parts.append(pismeno.get('id', 'pismeno'))
    if subitem:
        id_parts.append(f"subitem-{subitem.get('marker', '')}")
    
    chunk_id = '.'.join(id_parts) if id_parts else "chunk"
    
    # Build path
    path = {}
    if part:
        path['part'] = {
            'id': part.get('id'),
            'title': part.get('title'),
            'title_text': part.get('title_text')
        }
    if paragraph:
        path['paragraph'] = {
            'id': paragraph.get('id'),
            'marker': paragraph.get('marker'),
            'intro_text': paragraph.get('intro_text', '')[:100]  # Truncate for metadata
        }
    if odsek:
        path['odsek'] = {
            'id': odsek.get('id'),
            'marker': odsek.get('marker')
        }
    if pismeno:
        path['pismeno'] = {
            'id': pismeno.get('id'),
            'marker': pismeno.get('marker')
        }
    if subitem:
        path['subitem'] = {
            'marker': subitem.get('marker')
        }
    
    return {
        'chunk_id': chunk_id,
        'level': level,
        'path': path
    }


def create_chunk(
    content: str,
    metadata: Dict[str, Any],
    tables: List[Dict] = None,
    has_subitems: bool = False
) -> Dict[str, Any]:
    """
    Create a chunk dictionary.
    
    Args:
        content: Markdown formatted content
        metadata: Chunk metadata
        tables: List of tables in this chunk
        has_subitems: Whether chunk contains subitems
        
    Returns:
        Chunk dictionary
    """
    return {
        **metadata,
        'content': content,
        'tables': tables or [],
        'has_subitems': has_subitems,
        'content_length': len(content)
    }


def chunk_at_pismeno_level(
    data: Dict[str, Any],
    include_context: bool = True
) -> List[Dict[str, Any]]:
    """
    Chunk document at pismeno level (or lowest available).
    
    Args:
        data: Parsed JSON document
        include_context: Whether to include parent context in each chunk
        
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    
    parts = data.get('parts', [])
    
    for part in parts:
        part_context = ""
        if include_context:
            title = part.get('title', '')
            title_text = part.get('title_text', '')
            if title:
                part_context = f"# {title}\n"
            if title_text and title_text != title:
                part_context += f"## {title_text}\n"
            part_context += "\n---\n"
        
        paragraphs = part.get('paragraphs', [])
        
        for paragraph in paragraphs:
            para_context = ""
            intro = paragraph.get('intro_text', '')
            marker = paragraph.get('marker', '')
            title = paragraph.get('title', marker)  # Use title if available, fallback to marker
            odseks = paragraph.get('odseks', [])
            
            # Check if this paragraph has embedded pismenos (no odseks but pismeno patterns in intro)
            # Patterns: explicit a), b) markers OR numbered items directly in text
            has_embedded_pismenos = (
                not odseks and 
                intro and 
                len(intro) > 100 and
                (
                    any(c in intro for c in ['\na)', '\nb)', '\nc)', '\nd)']) or
                    '\n1.' in intro  # Numbered items directly in intro
                )
            )
            
            if include_context:
                # Use title if it contains more info than just marker (e.g., "§ 2 Základné pojmy")
                para_header = title if title and title != marker else marker
                para_context = f"\n## {para_header}\n"
                if intro:
                    if has_embedded_pismenos:
                        # Parse embedded pismenos with proper indentation
                        # First line might be a title, rest has embedded structure
                        lines = intro.split('\n')
                        if lines:
                            # First line is typically the title
                            first_line = lines[0].strip()
                            if first_line and len(first_line) < 100:
                                para_context += f"### {first_line}\n"
                            rest = '\n'.join(lines[1:]) if len(lines) > 1 else ''
                            if rest:
                                formatted_rest = parse_embedded_pismenos(rest)
                                para_context += f"\n{formatted_rest}\n"
                    elif len(intro) < 100 and '\n' not in intro:
                        # Short intro - use as subtitle only if title doesn't already contain nadpis
                        # e.g., "Na účely tohto zákona sa rozumie" for § 2
                        para_context += f"### {intro}\n"
                    else:
                        para_context += f"\n{intro}\n"
            
            # If paragraph has no odseks, chunk at paragraph level
            if not odseks:
                content = part_context + para_context
                
                # Add paragraph-level tables
                tables = paragraph.get('tables', [])
                for table in tables:
                    if table:
                        content += f"\n{format_table(table, 0)}\n"
                
                metadata = create_chunk_metadata(part=part, paragraph=paragraph)
                chunks.append(create_chunk(content, metadata, tables))
                continue
            
            for odsek in odseks:
                odsek_context = ""
                if include_context:
                    marker = odsek.get('marker', '')
                    text = odsek.get('text', '')
                    odsek_context = f"\n{format_marker_text(marker, text, INDENT_ODSEK)}\n"
                
                pismenos = odsek.get('pismenos', [])
                
                # If odsek has no pismenos, chunk at odsek level
                if not pismenos:
                    content = part_context + para_context + odsek_context
                    
                    # Add odsek-level tables
                    tables = odsek.get('tables', [])
                    for table in tables:
                        if table:
                            content += f"\n{format_table(table, INDENT_ODSEK)}\n"
                    
                    metadata = create_chunk_metadata(part=part, paragraph=paragraph, odsek=odsek)
                    chunks.append(create_chunk(content, metadata, tables))
                    continue
                
                # Chunk at pismeno level
                for pismeno in pismenos:
                    content = part_context + para_context + odsek_context
                    
                    # Add pismeno content
                    pismeno_content = format_pismeno(pismeno)
                    content += pismeno_content
                    
                    # Collect tables
                    tables = pismeno.get('tables', [])
                    has_subitems = bool(pismeno.get('subitems', []))
                    
                    metadata = create_chunk_metadata(
                        part=part, 
                        paragraph=paragraph, 
                        odsek=odsek, 
                        pismeno=pismeno
                    )
                    chunks.append(create_chunk(content, metadata, tables, has_subitems))
    
    return chunks


def chunk_annexes(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunk annexes separately.
    
    Args:
        data: Parsed JSON document
        
    Returns:
        List of annex chunk dictionaries
    """
    chunks = []
    
    annexes = data.get('annexes', {}).get('annex_list', [])
    
    for annex in annexes:
        content_parts = []
        
        # Annex header
        title = annex.get('title', f"Príloha č. {annex.get('number', '?')}")
        content_parts.append(f"# {title}")
        
        # Annex text content
        annex_content = annex.get('content', {})
        text = annex_content.get('text', '')
        if text:
            content_parts.append(f"\n{text}")
        
        # Annex tables
        tables = annex_content.get('tables', [])
        for table in tables:
            table_title = table.get('title', '')
            if table_title:
                content_parts.append(f"\n### {table_title}")
            
            table_md = table.get('markdown', '')
            if table_md:
                content_parts.append(f"\n{table_md}")
        
        content = '\n'.join(content_parts)
        
        metadata = {
            'chunk_id': f"annex-{annex.get('number', '?')}",
            'level': 'annex',
            'path': {
                'annex': {
                    'id': annex.get('id'),
                    'number': annex.get('number'),
                    'title': title
                }
            }
        }
        
        chunks.append(create_chunk(content, metadata, tables))
    
    return chunks


def chunk_document(
    data: Dict[str, Any],
    chunk_level: str = "pismeno",
    include_context: bool = True,
    include_annexes: bool = True
) -> List[Dict[str, Any]]:
    """
    Chunk a parsed JSON document into markdown chunks.
    
    Args:
        data: Parsed JSON document
        chunk_level: Target chunking level ("pismeno", "odsek", "paragraph", "part")
        include_context: Whether to include parent context in each chunk
        include_annexes: Whether to include annexes as separate chunks
        
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    
    # Main document chunks
    if chunk_level == "pismeno":
        chunks.extend(chunk_at_pismeno_level(data, include_context))
    else:
        # TODO: Implement other chunking levels
        chunks.extend(chunk_at_pismeno_level(data, include_context))
    
    # Annex chunks
    if include_annexes:
        chunks.extend(chunk_annexes(data))
    
    return chunks


def format_chunks_to_markdown(chunks: List[Dict[str, Any]], include_separators: bool = True) -> str:
    """
    Combine all chunks into a single markdown string.
    
    Args:
        chunks: List of chunk dictionaries
        include_separators: Whether to add separators between chunks
        
    Returns:
        Combined markdown string
    """
    parts = []
    
    for i, chunk in enumerate(chunks):
        if include_separators and i > 0:
            parts.append("\n\n---\n")
        parts.append(chunk['content'])
    
    return '\n'.join(parts)


def save_chunks(
    chunks: List[Dict[str, Any]], 
    output_path: Path,
    format: str = "json"
) -> None:
    """
    Save chunks to file.
    
    Args:
        chunks: List of chunk dictionaries
        output_path: Output file path
        format: Output format ("json", "md", "both")
    """
    output_path = Path(output_path)
    
    if format in ("json", "both"):
        json_path = output_path.with_suffix('.chunks.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(chunks)} chunks to {json_path}")
    
    if format in ("md", "both"):
        md_path = output_path.with_suffix('.chunks.md')
        md_content = format_chunks_to_markdown(chunks)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"Saved markdown to {md_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert parsed JSON law documents to markdown chunks"
    )
    parser.add_argument(
        "input_json",
        type=str,
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (default: input with .chunks suffix)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "md", "both"],
        default="both",
        help="Output format (default: both)"
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Don't include parent context in chunks"
    )
    parser.add_argument(
        "--no-annexes",
        action="store_true",
        help="Don't include annexes"
    )
    parser.add_argument(
        "--chunk-level",
        choices=["pismeno", "odsek", "paragraph", "part"],
        default="pismeno",
        help="Chunking level (default: pismeno)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_json)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    # Load JSON
    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Chunk document
    print(f"Chunking at {args.chunk_level} level...")
    chunks = chunk_document(
        data,
        chunk_level=args.chunk_level,
        include_context=not args.no_context,
        include_annexes=not args.no_annexes
    )
    
    print(f"Created {len(chunks)} chunks")
    
    # Save output
    output_path = Path(args.output) if args.output else input_path
    save_chunks(chunks, output_path, args.format)
    
    # Print summary
    levels = {}
    for chunk in chunks:
        level = chunk.get('level', 'unknown')
        levels[level] = levels.get(level, 0) + 1
    
    print("\nChunk summary:")
    for level, count in sorted(levels.items()):
        print(f"  {level}: {count}")
    
    return 0


if __name__ == "__main__":
    exit(main())

