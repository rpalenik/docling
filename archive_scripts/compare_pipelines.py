"""
Comparison script for Basic vs VLM pipeline results.

This script helps you understand the differences between the two pipelines
by comparing their JSON outputs and explaining when to use each approach.
"""

import json
import sys
from pathlib import Path


def load_json(file_path):
    """Load JSON file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {str(e)}")
        return None


def compare_documents(basic_json, vlm_json, input_type="PDF"):
    """
    Compare two document JSON outputs and highlight differences.
    
    Args:
        basic_json: JSON output from basic pipeline
        vlm_json: JSON output from VLM pipeline
        input_type: Type of input ("PDF" or "HTML")
    """
    print(f"\n{'='*70}")
    print(f"COMPARISON: Basic Pipeline vs VLM Pipeline ({input_type})")
    print(f"{'='*70}\n")
    
    if not basic_json or not vlm_json:
        print("Cannot compare: One or both JSON files are missing or invalid.")
        return
    
    # Compare top-level structure
    print("1. DOCUMENT STRUCTURE COMPARISON")
    print("-" * 70)
    
    basic_keys = set(basic_json.keys()) if isinstance(basic_json, dict) else set()
    vlm_keys = set(vlm_json.keys()) if isinstance(vlm_json, dict) else set()
    
    common_keys = basic_keys & vlm_keys
    only_basic = basic_keys - vlm_keys
    only_vlm = vlm_keys - basic_keys
    
    print(f"   Common fields: {len(common_keys)}")
    if common_keys:
        print(f"   Fields: {', '.join(sorted(list(common_keys)[:10]))}")
        if len(common_keys) > 10:
            print(f"   ... and {len(common_keys) - 10} more")
    
    if only_basic:
        print(f"   Only in Basic: {', '.join(sorted(only_basic))}")
    if only_vlm:
        print(f"   Only in VLM: {', '.join(sorted(only_vlm))}")
    
    # Try to extract text content for comparison
    print("\n2. CONTENT ANALYSIS")
    print("-" * 70)
    
    def extract_text_recursive(obj, max_depth=3, current_depth=0):
        """Recursively extract text content from JSON structure."""
        if current_depth >= max_depth:
            return []
        
        texts = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.lower() in ['text', 'content', 'value'] and isinstance(value, str):
                    texts.append(value)
                else:
                    texts.extend(extract_text_recursive(value, max_depth, current_depth + 1))
        elif isinstance(obj, list):
            for item in obj:
                texts.extend(extract_text_recursive(item, max_depth, current_depth + 1))
        elif isinstance(obj, str) and len(obj) > 20:
            texts.append(obj)
        
        return texts
    
    basic_texts = extract_text_recursive(basic_json)
    vlm_texts = extract_text_recursive(vlm_json)
    
    basic_text_length = sum(len(t) for t in basic_texts)
    vlm_text_length = sum(len(t) for t in vlm_texts)
    
    print(f"   Basic Pipeline - Text segments found: {len(basic_texts)}")
    print(f"   Basic Pipeline - Total text length: {basic_text_length:,} characters")
    print(f"   VLM Pipeline - Text segments found: {len(vlm_texts)}")
    print(f"   VLM Pipeline - Total text length: {vlm_text_length:,} characters")
    
    if basic_text_length > 0 and vlm_text_length > 0:
        diff_percent = ((vlm_text_length - basic_text_length) / basic_text_length) * 100
        print(f"   Difference: {diff_percent:+.1f}%")
    
    # Compare structure depth
    def get_max_depth(obj, current_depth=0, max_seen=0):
        """Calculate maximum nesting depth of JSON structure."""
        if not isinstance(obj, (dict, list)):
            return current_depth
        
        max_seen = max(max_seen, current_depth)
        if isinstance(obj, dict):
            for value in obj.values():
                max_seen = max(max_seen, get_max_depth(value, current_depth + 1, max_seen))
        elif isinstance(obj, list):
            for item in obj:
                max_seen = max(max_seen, get_max_depth(item, current_depth + 1, max_seen))
        
        return max_seen
    
    basic_depth = get_max_depth(basic_json)
    vlm_depth = get_max_depth(vlm_json)
    
    print(f"\n   Basic Pipeline - Max structure depth: {basic_depth}")
    print(f"   VLM Pipeline - Max structure depth: {vlm_depth}")
    
    # Summary
    print("\n3. SUMMARY & RECOMMENDATIONS")
    print("-" * 70)
    
    print("\n   BASIC PIPELINE:")
    print("   • Uses traditional parsing methods (rule-based)")
    print("   • Faster processing time")
    print("   • Lower resource requirements (CPU only)")
    print("   • Good for well-structured documents")
    print("   • Best for: Standard PDFs, HTML pages, simple layouts")
    
    print("\n   VLM PIPELINE:")
    print("   • Uses AI vision-language models (Granite-Docling)")
    print("   • Slower processing time")
    print("   • Higher resource requirements (GPU recommended)")
    print("   • Better for complex layouts and visual elements")
    print("   • Best for: Scanned PDFs, complex layouts, documents with images,")
    print("              tables, or non-standard formatting")
    
    print("\n   WHEN TO USE EACH:")
    print("   → Use BASIC if: Document is well-structured, speed is important,")
    print("                  or you're processing many simple documents")
    print("   → Use VLM if: Document has complex layouts, contains images,")
    print("                is scanned, or requires advanced understanding")


def main():
    """Main comparison function."""
    
    if len(sys.argv) < 3:
        print("Usage: python compare_pipelines.py <basic_output.json> <vlm_output.json>")
        print("\nExample:")
        print("  python compare_pipelines.py outputs/document_basic.json outputs/document_vlm.json")
        sys.exit(1)
    
    basic_file = sys.argv[1]
    vlm_file = sys.argv[2]
    
    print("\n" + "="*70)
    print("PIPELINE COMPARISON TOOL")
    print("="*70)
    
    # Load JSON files
    print(f"\nLoading files...")
    print(f"  Basic pipeline output: {basic_file}")
    print(f"  VLM pipeline output: {vlm_file}")
    
    basic_json = load_json(basic_file)
    vlm_json = load_json(vlm_file)
    
    if not basic_json or not vlm_json:
        print("\nError: Could not load one or both JSON files.")
        sys.exit(1)
    
    # Determine input type from filename
    input_type = "PDF"
    if "html" in basic_file.lower() or "html" in vlm_file.lower():
        input_type = "HTML"
    
    # Perform comparison
    compare_documents(basic_json, vlm_json, input_type)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print("\nTip: Open the JSON files in a text editor or JSON viewer")
    print("     to see the detailed structure differences.")


if __name__ == "__main__":
    main()

