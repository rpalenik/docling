"""
Convert Docling JSON document to chunked output.

This script loads a DoclingDocument JSON file, applies chunking methods to generate chunks,
and exports the chunks in both JSON and Markdown formats.

Supports different chunking methods:
- Hierarchical chunking: Based on document structure (sections, paragraphs, tables, lists)
- Hybrid chunking: Combines hierarchical and semantic chunking with token limits

Configurable options:
- max_tokens: Maximum tokens per chunk
- merge_peers: Merge adjacent compatible chunks
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from docling_core.transforms.chunker.hierarchical_chunker import (
    DocChunk,
    DocMeta,
    HierarchicalChunker,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument


def load_docling_document(input_file_path):
    """Load a DoclingDocument from a JSON file or convert from PDF/HTML.
    
    Args:
        input_file_path: Path to input file (JSON, PDF, or HTML)
        
    Returns:
        DoclingDocument: The loaded document
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be loaded as DoclingDocument
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    
    input_path = Path(input_file_path)
    file_ext = input_path.suffix.lower()
    
    # If it's a JSON file, load directly
    if file_ext == '.json':
        print(f"Loading DoclingDocument from JSON: {input_file_path}")
        try:
            document = DoclingDocument.load_from_json(input_file_path)
            print(f"  ✓ Successfully loaded document: {document.name}")
            return document
        except Exception as e:
            raise ValueError(f"Failed to load DoclingDocument from JSON: {str(e)}") from e
    
    # Otherwise, use DocumentConverter to convert from PDF/HTML
    else:
        print(f"Converting document using DocumentConverter: {input_file_path}")
        try:
            converter = DocumentConverter()
            result = converter.convert(input_file_path)
            document = result.document
            print(f"  ✓ Successfully converted document: {document.name}")
            return document
        except Exception as e:
            raise ValueError(f"Failed to convert document: {str(e)}") from e


def generate_chunks(document, max_tokens=None, merge_peers=None, chunking_method="hybrid", merge_list_items=True):
    """Generate chunks from a DoclingDocument using specified chunking method.
    
    Args:
        document: DoclingDocument to chunk
        max_tokens: Optional max tokens per chunk (only for hybrid chunking)
        merge_peers: Optional flag to merge adjacent compatible chunks (only for hybrid chunking)
        chunking_method: Chunking method to use ("hierarchical" or "hybrid")
        merge_list_items: Whether to merge successive list items (for hierarchical chunking)
        
    Returns:
        tuple: (list of chunk objects, chunker instance)
    """
    print(f"\nInitializing {chunking_method.upper()} chunking...")
    
    try:
        method_lower = chunking_method.lower()
        
        # Initialize chunker based on method
        if method_lower == "hierarchical":
            # Hierarchical chunking with merge_list_items parameter
            chunker = HierarchicalChunker(merge_list_items=merge_list_items)
            print("  Method: Structure-based chunking (respects document hierarchy)")
            print(f"  Merge list items: {merge_list_items}")
            
        elif method_lower == "hybrid":
            # Build chunker configuration for hybrid chunking
            chunker_kwargs = {}
            
            if max_tokens is not None:
                chunker_kwargs['max_tokens'] = max_tokens
                print(f"  Max tokens per chunk: {max_tokens}")
            else:
                print("  Max tokens per chunk: (using default)")
            
            if merge_peers is not None:
                chunker_kwargs['merge_peers'] = merge_peers
                print(f"  Merge peers: {merge_peers}")
            else:
                print("  Merge peers: (using default)")
            
            chunker = HybridChunker(**chunker_kwargs)
            print("  Method: Hybrid chunking (hierarchical + semantic + token-aware)")
            print("  Using tokenizer: sentence-transformers/all-MiniLM-L6-v2")
            
            # Display actual settings (may differ from requested if using defaults)
            if not chunker_kwargs:
                print(f"  Actual max tokens per chunk: {chunker.max_tokens}")
                print(f"  Actual merge peers: {chunker.merge_peers}")
        else:
            raise ValueError(f"Unknown chunking method: {chunking_method}. Supported: 'hierarchical', 'hybrid'")
        
        # Check document content before chunking
        has_body_content = (
            hasattr(document, 'body') and document.body and 
            hasattr(document.body, 'children') and len(document.body.children) > 0
        )
        has_texts = hasattr(document, 'texts') and len(document.texts) > 0
        has_groups = hasattr(document, 'groups') and len(document.groups) > 0
        
        if not (has_body_content or has_texts or has_groups):
            print("  ⚠ WARNING: Document appears to have no chunkable content!")
            print("     This may be a JavaScript-rendered page or empty document.")
            return [], chunker
        
        # Generate chunks - use chunk(doc) instead of chunk(dl_doc=document)
        print("\nGenerating chunks...")
        chunks = list(chunker.chunk(document))
        
        # Filter out chunks with empty or "<unknown>" text
        valid_chunks = [c for c in chunks if c.text and c.text.strip() and c.text != "<unknown>"]
        
        print(f"  ✓ Generated {len(chunks)} total chunks")
        print(f"  ✓ {len(valid_chunks)} chunks with valid content")
        if len(chunks) - len(valid_chunks) > 0:
            print(f"  ⚠ {len(chunks) - len(valid_chunks)} chunks filtered out (empty or unknown)")
        
        if len(valid_chunks) == 0:
            print("\n  ⚠ WARNING: No valid chunks generated!")
            print("     Possible reasons:")
            print("     - Document is a JavaScript-rendered page (content loads via JS)")
            print("     - Document has no extractable text content")
            print("     - All content is in 'furniture' layer (headers/footers)")
        
        return valid_chunks, chunker
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate chunks: {str(e)}") from e


def export_chunks_to_json(chunks, output_path, source_file, chunker=None, chunking_method="hybrid"):
    """Export chunks to JSON format.
    
    Args:
        chunks: List of chunk objects
        output_path: Path to output JSON file
        source_file: Source file path for metadata
        chunker: Chunker instance for metadata
        chunking_method: Method used for chunking
    """
    print(f"\nExporting chunks to JSON: {output_path}")
    
    # Convert chunks to JSON-serializable format
    chunks_data = []
    for i, chunk in enumerate(chunks):
        chunk_dict = chunk.export_json_dict()
        chunks_data.append(chunk_dict)
    
    # Build metadata
    metadata = {
        "source_file": str(source_file),
        "chunk_count": len(chunks),
        "generated_at": datetime.now().isoformat(),
        "chunking_method": chunking_method
    }
    
    # Add chunker-specific metadata if available
    if chunker:
        if hasattr(chunker, 'max_tokens'):
            metadata["max_tokens"] = chunker.max_tokens
            metadata["tokenizer"] = "sentence-transformers/all-MiniLM-L6-v2"
        if hasattr(chunker, 'merge_peers'):
            metadata["merge_peers"] = chunker.merge_peers
    
    # Create output structure with metadata
    output_data = {
        "metadata": metadata,
        "chunks": chunks_data
    }
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Exported {len(chunks)} chunks to JSON")


def format_headings_as_markdown(headings):
    """Format headings list as markdown heading hierarchy.
    
    Args:
        headings: List of heading strings (from most general to most specific)
        
    Returns:
        str: Markdown formatted headings
    """
    if not headings:
        return ""
    
    markdown_lines = []
    for i, heading in enumerate(headings, start=1):
        # Use heading level based on position in hierarchy
        # First heading is #, second is ##, etc.
        markdown_lines.append(f"{'#' * i} {heading}")
    
    return "\n".join(markdown_lines)


def export_chunks_to_markdown(chunks, output_path, source_file, document_name, chunker=None, chunking_method="hybrid"):
    """Export chunks to Markdown format.
    
    Args:
        chunks: List of chunk objects
        output_path: Path to output Markdown file
        source_file: Source file path for metadata
        document_name: Name of the document
        chunker: Chunker instance for metadata
        chunking_method: Method used for chunking
    """
    print(f"\nExporting chunks to Markdown: {output_path}")
    
    markdown_lines = []
    
    # Document header
    markdown_lines.append(f"# {chunking_method.upper()} Chunking Output")
    markdown_lines.append("")
    markdown_lines.append("## Document Information")
    markdown_lines.append("")
    markdown_lines.append(f"- **Document Name:** {document_name}")
    markdown_lines.append(f"- **Source File:** `{source_file}`")
    markdown_lines.append(f"- **Total Chunks:** {len(chunks)}")
    markdown_lines.append(f"- **Generated At:** {datetime.now().isoformat()}")
    markdown_lines.append(f"- **Chunking Method:** {chunking_method.upper()}")
    
    # Add chunker-specific settings
    if chunker:
        if hasattr(chunker, 'max_tokens'):
            markdown_lines.append(f"- **Max Tokens per Chunk:** {chunker.max_tokens}")
            markdown_lines.append(f"- **Tokenizer:** sentence-transformers/all-MiniLM-L6-v2")
        if hasattr(chunker, 'merge_peers'):
            markdown_lines.append(f"- **Merge Peers:** {chunker.merge_peers}")
    
    markdown_lines.append("")
    markdown_lines.append("---")
    markdown_lines.append("")
    
    # Export each chunk
    for i, chunk in enumerate(chunks, start=1):
        markdown_lines.append(f"## Chunk {i}")
        markdown_lines.append("")
        
        # Add headings if available
        if chunk.meta.headings:
            headings_md = format_headings_as_markdown(chunk.meta.headings)
            if headings_md:
                markdown_lines.append(headings_md)
                markdown_lines.append("")
        
        # Add chunk text
        if chunk.text and chunk.text.strip():
            markdown_lines.append(chunk.text)
            markdown_lines.append("")
        
        # Add metadata section (collapsible)
        markdown_lines.append("<details>")
        markdown_lines.append("<summary>Chunk Metadata</summary>")
        markdown_lines.append("")
        markdown_lines.append(f"- **Schema:** `{chunk.meta.schema_name}`")
        markdown_lines.append(f"- **Version:** `{chunk.meta.version}`")
        markdown_lines.append(f"- **Document Items:** {len(chunk.meta.doc_items)}")
        if chunk.meta.headings:
            markdown_lines.append(f"- **Headings:** {len(chunk.meta.headings)}")
        if chunk.meta.origin:
            origin_info = chunk.meta.origin
            if hasattr(origin_info, 'filename'):
                markdown_lines.append(f"- **Origin:** {origin_info.filename}")
        markdown_lines.append("</details>")
        markdown_lines.append("")
        
        # Separator between chunks
        if i < len(chunks):
            markdown_lines.append("---")
            markdown_lines.append("")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_lines))
    
    print(f"  ✓ Exported {len(chunks)} chunks to Markdown")


def main():
    """Main function to run the conversion."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Convert Docling JSON document to chunked output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use hierarchical chunking (structure-based)
  python chunk_document.py output/document.json --chunking-method hierarchical
  
  # Use hybrid chunking with default settings
  python chunk_document.py output/document.json --chunking-method hybrid
  
  # Hybrid chunking with custom max tokens
  python chunk_document.py output/document.json --chunking-method hybrid --max-tokens 512
  
  # Hybrid chunking with disabled peer merging
  python chunk_document.py output/document.json --chunking-method hybrid --no-merge-peers
  
  # Full custom configuration
  python chunk_document.py output/document.json --chunking-method hybrid --max-tokens 1024 --merge-peers
        """
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        default="input/595:2003 Z. z. - Zákon o dani z príjmov.html",
        help='Path to input file (JSON, PDF, or HTML). Default: input/595:2003 Z. z. - Zákon o dani z príjmov.html'
    )
    
    parser.add_argument(
        '--chunking-method',
        choices=['hierarchical', 'hybrid'],
        default='hierarchical',
        help='Chunking method to use: "hierarchical" (structure-based) or "hybrid" (hierarchical + semantic + token-aware). Default: hierarchical'
    )
    
    parser.add_argument(
        '--merge-list-items',
        action='store_true',
        default=True,
        help='Enable merging of successive list items (for hierarchical chunking, default: True)'
    )
    
    parser.add_argument(
        '--no-merge-list-items',
        action='store_false',
        dest='merge_list_items',
        help='Disable merging of successive list items (for hierarchical chunking)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=None,
        help='Maximum tokens per chunk (only for hybrid chunking, default: uses chunker default, typically 512)'
    )
    
    parser.add_argument(
        '--merge-peers',
        action='store_true',
        default=None,
        help='Enable merging of adjacent compatible chunks (only for hybrid chunking, default: True)'
    )
    
    parser.add_argument(
        '--no-merge-peers',
        action='store_false',
        dest='merge_peers',
        help='Disable merging of adjacent compatible chunks (only for hybrid chunking)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for chunked files (default: output)'
    )
    
    parser.add_argument(
        '--output-format',
        choices=['md', 'json', 'both'],
        default='md',
        help='Output format: "md" (Markdown only), "json" (JSON only), or "both". Default: md'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}")
        parser.print_help()
        sys.exit(1)
    
    # Validate that hierarchical chunking doesn't use hybrid-specific options
    if args.chunking_method == 'hierarchical':
        if args.max_tokens is not None:
            print("⚠ Warning: --max-tokens is only applicable to hybrid chunking. Ignoring.")
        if args.merge_peers is not None:
            print("⚠ Warning: --merge-peers is only applicable to hybrid chunking. Ignoring.")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("DOCLING CHUNKING CONVERSION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Chunking method: {args.chunking_method}")
    if args.chunking_method == 'hybrid':
        if args.max_tokens:
            print(f"  Max tokens: {args.max_tokens}")
        if args.merge_peers is not None:
            print(f"  Merge peers: {args.merge_peers}")
    elif args.chunking_method == 'hierarchical':
        print(f"  Merge list items: {args.merge_list_items}")
    print(f"  Input file: {input_path}")
    print(f"  Output format: {args.output_format}")
    print(f"  Output directory: {output_dir.absolute()}")
    
    try:
        # Load document (from JSON or convert from PDF/HTML)
        document = load_docling_document(str(input_path))
        
        # Generate chunks
        chunks, chunker = generate_chunks(
            document,
            max_tokens=args.max_tokens if args.chunking_method == 'hybrid' else None,
            merge_peers=args.merge_peers if args.chunking_method == 'hybrid' else None,
            chunking_method=args.chunking_method,
            merge_list_items=args.merge_list_items if args.chunking_method == 'hierarchical' else True
        )
        
        # Prepare output filenames
        input_stem = input_path.stem
        method_suffix = args.chunking_method.lower()
        json_output = output_dir / f"{input_stem}_{method_suffix}_chunks.json"
        md_output = output_dir / f"{input_stem}_{method_suffix}_chunks.md"
        
        # Check if we have any chunks to export
        if len(chunks) == 0:
            print("\n" + "=" * 70)
            print("WARNING: NO CHUNKS GENERATED")
            print("=" * 70)
            print(f"\n⚠ No chunks were generated from: {input_path.name}")
            print("\nThis typically happens when:")
            print("  - The document is a JavaScript-rendered page (SPA)")
            print("  - The document has no extractable text content")
            print("  - All content is in metadata/furniture layers")
            print("\nThe output files will be created but will be empty.")
            print("Consider using the PDF version of the document instead.")
        else:
            # Export based on output format
            output_files = []
            
            if args.output_format in ['json', 'both']:
                export_chunks_to_json(
                    chunks, 
                    str(json_output), 
                    str(input_path),
                    chunker=chunker,
                    chunking_method=args.chunking_method
                )
                output_files.append(f"  JSON:     {json_output.absolute()}")
            
            if args.output_format in ['md', 'both']:
                export_chunks_to_markdown(
                    chunks, 
                    str(md_output), 
                    str(input_path), 
                    document.name, 
                    chunker=chunker,
                    chunking_method=args.chunking_method
                )
                output_files.append(f"  Markdown: {md_output.absolute()}")
        
        # Summary
        print("\n" + "=" * 70)
        print("CONVERSION COMPLETE")
        print("=" * 70)
        print(f"\n✓ Processed {len(chunks)} chunks from: {input_path.name}")
        if output_files:
            print(f"\nOutput files:")
            for file_info in output_files:
                print(file_info)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



