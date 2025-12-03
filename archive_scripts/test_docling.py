"""
Test script for IBM Docling framework.

This script demonstrates processing PDF and HTML input files using both:
1. Basic pipeline (default DocumentConverter)
2. VLM pipeline (Vision-Language Model with Granite-Docling)

All outputs are exported to JSON format for comparison.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


def export_to_json(document, output_path):
    """Export docling document to JSON format."""
    # Try export_to_dict first (preferred for formatting control)
    # If not available, try export_to_json and parse it
    try:
        if hasattr(document, 'export_to_dict'):
            json_output = document.export_to_dict()
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, indent=2, ensure_ascii=False)
        elif hasattr(document, 'export_to_json'):
            json_str = document.export_to_json()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
        else:
            # Fallback: convert document to dict manually
            json_output = document.model_dump() if hasattr(document, 'model_dump') else dict(document)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  ⚠ Warning: Could not export using standard methods: {str(e)}")
        # Last resort: export to markdown and save as text
        try:
            md_output = document.export_to_markdown()
            with open(output_path.replace('.json', '.md'), 'w', encoding='utf-8') as f:
                f.write(md_output)
            print(f"  ✓ Exported to Markdown instead: {output_path.replace('.json', '.md')}")
        except:
            print(f"  ✗ Failed to export document")
            raise
    
    print(f"  ✓ Exported to: {output_path}")


def process_with_basic_pipeline(input_file, output_file, input_type="PDF"):
    """
    Process a document using the basic/default pipeline.
    
    The basic pipeline uses traditional document parsing techniques:
    - Text extraction from PDF/HTML
    - Layout analysis using rule-based methods
    - Structure detection without AI models
    - Faster processing, lower resource requirements
    - Good for well-structured documents
    
    Args:
        input_file: Path to input file (PDF or HTML)
        output_file: Path to output JSON file
        input_type: Type of input file ("PDF" or "HTML")
    """
    print(f"\n{'='*60}")
    print(f"Processing {input_type} with BASIC PIPELINE")
    print(f"{'='*60}")
    print(f"Input: {input_file}")
    
    try:
        # Basic pipeline: Use default DocumentConverter (no VLM)
        # This uses traditional parsing methods without AI models
        converter = DocumentConverter()
        
        # Convert the document
        result = converter.convert(input_file)
        
        # Export to JSON
        export_to_json(result.document, output_file)
        
        print(f"  ✓ Successfully processed with basic pipeline")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing with basic pipeline: {str(e)}")
        return False


def process_with_vlm_pipeline(input_file, output_file, input_type="PDF"):
    """
    Process a document using the VLM (Vision-Language Model) pipeline.
    
    The VLM pipeline uses AI models (Granite-Docling) for:
    - Advanced document understanding with vision-language models
    - Better handling of complex layouts and visual elements
    - Improved table extraction and structure recognition
    - Better OCR and text recognition from images
    - More accurate document structure understanding
    - Slower processing, requires more resources (GPU recommended)
    - Best for complex documents, scanned PDFs, or documents with images
    
    Args:
        input_file: Path to input file (PDF or HTML)
        output_file: Path to output JSON file
        input_type: Type of input file ("PDF" or "HTML")
    """
    print(f"\n{'='*60}")
    print(f"Processing {input_type} with VLM PIPELINE")
    print(f"{'='*60}")
    print(f"Input: {input_file}")
    
    try:
        # VLM pipeline: Configure for PDF processing with Granite-Docling model
        # Note: For HTML, VLM pipeline may not be as necessary, but we'll configure it
        if input_type == "PDF":
            # Configure VLM pipeline options for PDF
            pipeline_options = VlmPipelineOptions(
                vlm_options=vlm_model_specs.GRANITEDOCLING_MLX,  # Use MLX for Apple Silicon
            )
            
            # Create converter with VLM pipeline for PDF
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                        pipeline_options=pipeline_options,
                    ),
                }
            )
        else:
            # For HTML, we can still use VLM but it's less common
            # Using basic converter for HTML with VLM is not standard
            # We'll use the basic converter for HTML
            print("  Note: VLM pipeline is primarily designed for PDF/image documents.")
            print("  Using basic converter for HTML input.")
            converter = DocumentConverter()
        
        # Convert the document
        result = converter.convert(input_file)
        
        # Export to JSON
        export_to_json(result.document, output_file)
        
        print(f"  ✓ Successfully processed with VLM pipeline")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing with VLM pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run all processing tests."""
    
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="Test IBM Docling framework with PDF and HTML files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process PDF with both pipelines (default)
  python test_docling.py input/document.pdf
  
  # Process PDF with basic pipeline only
  python test_docling.py input/document.pdf --pipeline basic
  
  # Process PDF with VLM pipeline only
  python test_docling.py input/document.pdf --pipeline vlm
  
  # Process both PDF and HTML with basic pipeline only
  python test_docling.py input/document.pdf input/page.html --pipeline basic
        """
    )
    
    parser.add_argument(
        'pdf_file',
        help='Path to the PDF file to process'
    )
    
    parser.add_argument(
        'html_file',
        nargs='?',
        default=None,
        help='Optional: Path to the HTML file to process'
    )
    
    parser.add_argument(
        '--pipeline',
        choices=['basic', 'vlm', 'both'],
        default='both',
        help='Which pipeline(s) to run: basic, vlm, or both (default: both)'
    )
    
    args = parser.parse_args()
    
    pdf_file = args.pdf_file
    html_file = args.html_file
    pipeline_choice = args.pipeline.lower()
    
    # Validate input files
    if not os.path.exists(pdf_file):
        print(f"Error: PDF file not found: {pdf_file}")
        sys.exit(1)
    
    if html_file and not os.path.exists(html_file):
        print(f"Error: HTML file not found: {html_file}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("IBM DOCLING FRAMEWORK TEST")
    print("="*60)
    print(f"\nPipeline selection: {pipeline_choice.upper()}")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Determine which pipelines to run
    run_basic = pipeline_choice in ['basic', 'both']
    run_vlm = pipeline_choice in ['vlm', 'both']
    
    # Process PDF with selected pipeline(s)
    pdf_basename = Path(pdf_file).stem
    
    if run_basic:
        pdf_basic_output = output_dir / f"{pdf_basename}_basic.json"
        process_with_basic_pipeline(pdf_file, str(pdf_basic_output), "PDF")
    
    if run_vlm:
        pdf_vlm_output = output_dir / f"{pdf_basename}_vlm.json"
        process_with_vlm_pipeline(pdf_file, str(pdf_vlm_output), "PDF")
    
    # Process HTML with selected pipeline(s) (if provided)
    if html_file:
        html_basename = Path(html_file).stem
        
        if run_basic:
            html_basic_output = output_dir / f"{html_basename}_basic.json"
            process_with_basic_pipeline(html_file, str(html_basic_output), "HTML")
        
        if run_vlm:
            html_vlm_output = output_dir / f"{html_basename}_vlm.json"
            process_with_vlm_pipeline(html_file, str(html_vlm_output), "HTML")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    
    if run_basic and run_vlm:
        print("\nNext steps:")
        print("  1. Review the JSON output files to compare results")
        print("  2. Run compare_pipelines.py for detailed comparison")
        print("  3. Check README.md for pipeline explanations")
    else:
        print(f"\nNote: Only {pipeline_choice.upper()} pipeline was executed.")
        if pipeline_choice == 'basic':
            print("   Run with --pipeline vlm to test VLM pipeline, or --pipeline both for comparison.")
        elif pipeline_choice == 'vlm':
            print("   Run with --pipeline basic to test basic pipeline, or --pipeline both for comparison.")


if __name__ == "__main__":
    main()

