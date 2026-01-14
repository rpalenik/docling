#!/usr/bin/env python3
"""
Universal PDF to Docling JSON converter.

Usage:
    python convert_pdf_to_docling.py <input_pdf> [--output-dir <dir>]
    
Examples:
    python convert_pdf_to_docling.py input/priloha_1.pdf
    python convert_pdf_to_docling.py input/priloha_1.pdf --output-dir output
    python convert_pdf_to_docling.py input/*.pdf  # Multiple files
"""

import argparse
import sys
import time
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument


def convert_pdf_to_docling(pdf_path: Path, output_dir: Path) -> Path:
    """
    Convert a PDF file to Docling JSON format.
    
    Args:
        pdf_path: Path to input PDF file
        output_dir: Directory for output files
        
    Returns:
        Path to output JSON file
    """
    print(f"\n{'='*60}")
    print(f"Converting: {pdf_path.name}")
    print(f"{'='*60}")
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert PDF
    start_time = time.time()
    print(f"  Loading PDF...")
    
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document
    
    conversion_time = time.time() - start_time
    print(f"  ✓ Converted in {conversion_time:.2f}s")
    
    # Generate output filename
    output_name = pdf_path.stem.replace(' ', '_') + '_docling.json'
    output_path = output_dir / output_name
    
    # Save to JSON
    doc.save_as_json(str(output_path))
    
    # Print summary
    file_size = output_path.stat().st_size / 1024
    print(f"\n  Document summary:")
    print(f"    Name: {doc.name}")
    print(f"    Text elements: {len(doc.texts):,}")
    print(f"    Tables: {len(doc.tables)}")
    print(f"    Pictures: {len(doc.pictures)}")
    print(f"    Groups: {len(doc.groups)}")
    print(f"\n  Output: {output_path}")
    print(f"  Size: {file_size:.2f} KB")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert PDF files to Docling JSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_pdf_to_docling.py input/document.pdf
  python convert_pdf_to_docling.py input/doc1.pdf input/doc2.pdf
  python convert_pdf_to_docling.py input/*.pdf --output-dir output/converted
        """
    )
    parser.add_argument(
        'pdf_files',
        nargs='+',
        type=str,
        help='PDF file(s) to convert'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Process each PDF file
    results = []
    errors = []
    
    total_start = time.time()
    
    for pdf_file in args.pdf_files:
        pdf_path = Path(pdf_file)
        
        try:
            output_path = convert_pdf_to_docling(pdf_path, output_dir)
            results.append((pdf_path, output_path))
        except Exception as e:
            print(f"\n  ✗ Error converting {pdf_path}: {e}")
            errors.append((pdf_path, str(e)))
    
    total_time = time.time() - total_start
    
    # Print final summary
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {len(errors)}")
    
    if results:
        print(f"\n  Output files:")
        for pdf_path, output_path in results:
            print(f"    {pdf_path.name} -> {output_path.name}")
    
    if errors:
        print(f"\n  Errors:")
        for pdf_path, error in errors:
            print(f"    {pdf_path.name}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()




