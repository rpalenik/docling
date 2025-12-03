#!/usr/bin/env python3
"""
Migrate existing law documents to collections structure.

This script helps migrate existing HTML files and their associated data
into the new collections-based directory structure with YAML manifests.

Usage:
    python migrate_to_collections.py \
      --html-file "input/595:2003 Z. z. - Zákon o dani z príjmov.html" \
      --collection collections/dane \
      --law-id "595/2003" \
      --annexes-dir input
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import Optional

from annex_processor import AnnexProcessor, create_manifest_from_html


def sanitize_law_id(law_id: str) -> str:
    """
    Sanitize law ID for use in directory names.
    
    Args:
        law_id: Law identifier (e.g., "595/2003")
        
    Returns:
        Sanitized ID (e.g., "595_2003")
    """
    return law_id.replace('/', '_').replace(' ', '_')


def migrate_law_to_collection(
    html_file: Path,
    collection_dir: Path,
    law_id: str,
    annexes_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Migrate a law document to collection structure.
    
    Args:
        html_file: Path to source HTML file
        collection_dir: Target collection directory
        law_id: Law identifier (e.g., "595/2003")
        annexes_dir: Directory containing PDF annexes (optional)
        cache_dir: Source cache directory (optional)
        output_dir: Source output directory (optional)
        
    Returns:
        Path to created law directory
    """
    html_file = Path(html_file).resolve()
    collection_dir = Path(collection_dir).resolve()
    
    if not html_file.exists():
        raise FileNotFoundError(f"HTML file not found: {html_file}")
    
    # Create collection directory
    collection_dir.mkdir(parents=True, exist_ok=True)
    
    # Create law directory
    law_dir_name = sanitize_law_id(law_id)
    law_dir = collection_dir / law_dir_name
    law_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Migrating law: {law_id}")
    print(f"Target directory: {law_dir}")
    print(f"{'='*60}")
    
    # Copy HTML file to main.html
    main_html = law_dir / "main.html"
    print(f"\nCopying HTML file...")
    print(f"  {html_file.name} -> {main_html.name}")
    shutil.copy2(html_file, main_html)
    
    # Create annexes directory and copy PDFs
    annexes_target_dir = law_dir / "annexes"
    annexes_target_dir.mkdir(exist_ok=True)
    
    if annexes_dir:
        annexes_dir = Path(annexes_dir).resolve()
        if annexes_dir.exists():
            print(f"\nCopying annexes from {annexes_dir}...")
            
            # Find PDF files matching annex patterns
            patterns = [
                "*priloha*.pdf",
                "*príloha*.pdf",
                "*Priloha*.pdf",
                "*Príloha*.pdf",
                "*annex*.pdf"
            ]
            
            copied_count = 0
            for pattern in patterns:
                for pdf_file in annexes_dir.glob(pattern):
                    target = annexes_target_dir / pdf_file.name
                    if not target.exists():
                        shutil.copy2(pdf_file, target)
                        print(f"  {pdf_file.name} -> annexes/{target.name}")
                        copied_count += 1
            
            if copied_count == 0:
                print("  No annex PDFs found")
            else:
                print(f"  Copied {copied_count} annex file(s)")
    
    # Detect annexes and create manifest
    print(f"\nDetecting annexes and creating manifest...")
    manifest_path = law_dir / "manifest.yaml"
    
    manifest = create_manifest_from_html(
        main_html,
        law_id,
        manifest_path,
        law_dir
    )
    
    print(f"  Detected {len(manifest.annexes)} annexes")
    print(f"  Manifest saved to {manifest_path.name}")
    
    # Migrate cache if exists
    if cache_dir:
        cache_dir = Path(cache_dir).resolve()
        if cache_dir.exists():
            print(f"\nMigrating cache from {cache_dir}...")
            
            # Find cache subdirectory for this law
            safe_law_id = sanitize_law_id(law_id)
            cache_subdir = cache_dir / safe_law_id
            
            if cache_subdir.exists():
                cache_target = law_dir / "cache"
                cache_target.mkdir(exist_ok=True)
                
                # Copy cache files
                copied_count = 0
                for cache_file in cache_subdir.glob("*.json"):
                    target = cache_target / cache_file.name
                    if not target.exists():
                        shutil.copy2(cache_file, target)
                        copied_count += 1
                
                if copied_count > 0:
                    print(f"  Copied {copied_count} cache file(s)")
                else:
                    print("  No cache files found")
            else:
                print(f"  Cache subdirectory not found: {cache_subdir}")
    
    # Migrate outputs if exists
    if output_dir:
        output_dir = Path(output_dir).resolve()
        if output_dir.exists():
            print(f"\nMigrating outputs from {output_dir}...")
            
            output_target = law_dir / "output"
            output_target.mkdir(exist_ok=True)
            
            # Find output files matching law name
            law_name_pattern = re.escape(law_id.replace('/', ':').replace(' ', '_'))
            patterns = [
                f"*{law_name_pattern}*.json",
                f"*{sanitize_law_id(law_id)}*.json",
                f"*{law_id.replace('/', '_')}*.json"
            ]
            
            copied_count = 0
            for pattern in patterns:
                for output_file in output_dir.glob(pattern):
                    target = output_target / output_file.name
                    if not target.exists():
                        shutil.copy2(output_file, target)
                        print(f"  {output_file.name} -> output/{target.name}")
                        copied_count += 1
            
            if copied_count == 0:
                print("  No matching output files found")
            else:
                print(f"  Copied {copied_count} output file(s)")
    
    print(f"\n{'='*60}")
    print(f"✓ Migration complete: {law_dir}")
    print(f"{'='*60}")
    
    return law_dir


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate law documents to collections structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic migration
  python migrate_to_collections.py \\
    --html-file "input/595:2003 Z. z. - Zákon o dani z príjmov.html" \\
    --collection collections/dane \\
    --law-id "595/2003"
  
  # With annexes and cache
  python migrate_to_collections.py \\
    --html-file "input/595:2003 Z. z. - Zákon o dani z príjmov.html" \\
    --collection collections/dane \\
    --law-id "595/2003" \\
    --annexes-dir input \\
    --cache-dir cache \\
    --output-dir output
        """
    )
    
    parser.add_argument(
        '--html-file',
        type=str,
        required=True,
        help='Path to source HTML file'
    )
    
    parser.add_argument(
        '--collection',
        type=str,
        required=True,
        help='Target collection directory (e.g., collections/dane)'
    )
    
    parser.add_argument(
        '--law-id',
        type=str,
        required=True,
        help='Law identifier (e.g., "595/2003")'
    )
    
    parser.add_argument(
        '--annexes-dir',
        type=str,
        help='Directory containing PDF annexes'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        help='Source cache directory to migrate'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Source output directory to migrate'
    )
    
    args = parser.parse_args()
    
    try:
        law_dir = migrate_law_to_collection(
            html_file=Path(args.html_file),
            collection_dir=Path(args.collection),
            law_id=args.law_id,
            annexes_dir=Path(args.annexes_dir) if args.annexes_dir else None,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            output_dir=Path(args.output_dir) if args.output_dir else None
        )
        
        print(f"\nNext steps:")
        print(f"  1. Review manifest: {law_dir / 'manifest.yaml'}")
        print(f"  2. Process law: python process_law.py --manifest {law_dir / 'manifest.yaml'}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

