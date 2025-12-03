#!/usr/bin/env python3
"""
Process Law Documents with Collections Support

This script processes law documents using YAML manifests, supporting both
individual law processing and batch collection processing.

Usage:
    # Process a single law
    python process_law.py --collection collections/dane --law 595_2003
    
    # Process entire collection
    python process_law.py --collection collections/dane
    
    # Process using direct manifest path
    python process_law.py --manifest collections/dane/595_2003/manifest.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument

from annex_processor import AnnexProcessor, LawManifest
from sequential_parser import SequentialLawChunker, _clean_structure_for_json


def load_manifest_yaml(path: Path) -> Dict[str, Any]:
    """
    Load YAML manifest file.
    
    Args:
        path: Path to manifest.yaml
        
    Returns:
        Dictionary with manifest data
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def process_law(
    law_dir: Path,
    force: bool = False,
    skip_annexes: bool = False
) -> bool:
    """
    Process a single law document.
    
    Args:
        law_dir: Directory containing manifest.yaml and law files
        force: Force reprocessing even if output exists
        skip_annexes: Skip annex processing
        
    Returns:
        True if successful, False otherwise
    """
    law_dir = Path(law_dir).resolve()
    
    # Load manifest
    manifest_path = law_dir / "manifest.yaml"
    if not manifest_path.exists():
        print(f"Error: manifest.yaml not found in {law_dir}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Processing law: {law_dir.name}")
    print(f"{'='*60}")
    
    manifest = LawManifest.load(manifest_path)
    print(f"Law ID: {manifest.law_id}")
    print(f"Source: {manifest.source}")
    
    # Check if output already exists
    output_dir = law_dir / "output"
    output_file = output_dir / f"{law_dir.name}_sequential.json"
    
    if output_file.exists() and not force:
        print(f"Output already exists: {output_file}")
        print("Use --force to reprocess")
        return True
    
    # Load or convert main document
    main_doc_info = manifest.main_document
    main_doc_type = main_doc_info.get('type', 'html')
    main_doc_path = None
    
    if 'filename' in main_doc_info:
        main_doc_path = law_dir / main_doc_info['filename']
    elif 'path' in main_doc_info:
        main_doc_path = Path(main_doc_info['path'])
    
    if not main_doc_path or not main_doc_path.exists():
        print(f"Error: Main document not found: {main_doc_path}")
        return False
    
    print(f"\nLoading main document: {main_doc_path.name}")
    
    # Check for cached Docling conversion
    docling_cache_path = None
    if 'docling_cache' in main_doc_info:
        docling_cache_path = law_dir / main_doc_info['docling_cache']
    elif 'docling_path' in main_doc_info:
        docling_cache_path = Path(main_doc_info['docling_path'])
    
    doc = None
    if docling_cache_path and docling_cache_path.exists():
        print(f"  Loading from cache: {docling_cache_path.name}")
        try:
            doc = DoclingDocument.load_from_json(str(docling_cache_path))
        except Exception as e:
            print(f"  Warning: Failed to load cache: {e}")
            doc = None
    
    if doc is None:
        print(f"  Converting {main_doc_type} to Docling...")
        converter = DocumentConverter()
        result = converter.convert(str(main_doc_path))
        doc = result.document
        
        # Save to cache
        if docling_cache_path:
            cache_dir = docling_cache_path.parent
            cache_dir.mkdir(parents=True, exist_ok=True)
            doc.save_as_json(str(docling_cache_path))
            print(f"  Saved to cache: {docling_cache_path.name}")
    
    print(f"  ✓ Loaded: {len(doc.texts)} texts, {len(doc.tables)} tables")
    
    # Process document
    print("\nReconstructing document structure...")
    chunker = SequentialLawChunker()
    structure = chunker._reconstruct_document_with_docling(doc)
    
    # Process annexes
    if not skip_annexes and manifest.annexes:
        print(f"\nProcessing {len(manifest.annexes)} annexes...")
        processor = AnnexProcessor()
        structure = processor.integrate_annexes(
            structure,
            manifest.annexes,
            law_dir
        )
    else:
        if skip_annexes:
            print("\nSkipping annex processing (--skip-annexes)")
        else:
            print("\nNo annexes found in manifest")
    
    # Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving output to {output_file}...")
    # Clean structure for JSON serialization
    cleaned_structure = _clean_structure_for_json(structure)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_structure, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Successfully processed {law_dir.name}")
    return True


def process_collection(
    collection_dir: Path,
    force: bool = False,
    skip_annexes: bool = False,
    law_id: Optional[str] = None
) -> bool:
    """
    Process all laws in a collection or a specific law.
    
    Args:
        collection_dir: Directory containing law subdirectories
        force: Force reprocessing even if output exists
        skip_annexes: Skip annex processing
        law_id: Specific law ID to process (optional)
        
    Returns:
        True if all successful, False otherwise
    """
    collection_dir = Path(collection_dir).resolve()
    
    if not collection_dir.exists():
        print(f"Error: Collection directory not found: {collection_dir}")
        return False
    
    # Find all law directories (subdirectories with manifest.yaml)
    law_dirs = []
    
    if law_id:
        # Process specific law
        law_dir = collection_dir / law_id
        if (law_dir / "manifest.yaml").exists():
            law_dirs.append(law_dir)
        else:
            print(f"Error: Law {law_id} not found in collection {collection_dir}")
            return False
    else:
        # Process all laws
        for subdir in collection_dir.iterdir():
            if subdir.is_dir() and (subdir / "manifest.yaml").exists():
                law_dirs.append(subdir)
    
    if not law_dirs:
        print(f"No laws found in collection: {collection_dir}")
        return False
    
    print(f"\nFound {len(law_dirs)} law(s) to process")
    
    # Process each law
    success_count = 0
    for law_dir in sorted(law_dirs):
        try:
            if process_law(law_dir, force=force, skip_annexes=skip_annexes):
                success_count += 1
            else:
                print(f"✗ Failed to process {law_dir.name}")
        except Exception as e:
            print(f"✗ Error processing {law_dir.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Processed {success_count}/{len(law_dirs)} laws successfully")
    print(f"{'='*60}")
    
    return success_count == len(law_dirs)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Process law documents using YAML manifests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single law
  python process_law.py --collection collections/dane --law 595_2003
  
  # Process entire collection
  python process_law.py --collection collections/dane
  
  # Process using direct manifest path
  python process_law.py --manifest collections/dane/595_2003/manifest.yaml
        """
    )
    
    # Mutually exclusive: collection or manifest
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--collection',
        type=str,
        help='Path to collection directory (e.g., collections/dane)'
    )
    group.add_argument(
        '--manifest',
        type=str,
        help='Direct path to manifest.yaml file'
    )
    
    parser.add_argument(
        '--law',
        type=str,
        help='Law ID to process (only with --collection, e.g., 595_2003)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if output exists'
    )
    
    parser.add_argument(
        '--skip-annexes',
        action='store_true',
        help='Skip annex processing'
    )
    
    args = parser.parse_args()
    
    # Process based on arguments
    if args.manifest:
        # Direct manifest path
        manifest_path = Path(args.manifest).resolve()
        if not manifest_path.exists():
            print(f"Error: Manifest not found: {manifest_path}")
            sys.exit(1)
        
        law_dir = manifest_path.parent
        success = process_law(law_dir, force=args.force, skip_annexes=args.skip_annexes)
        
    elif args.collection:
        # Collection processing
        collection_dir = Path(args.collection).resolve()
        success = process_collection(
            collection_dir,
            force=args.force,
            skip_annexes=args.skip_annexes,
            law_id=args.law
        )
    else:
        parser.print_help()
        sys.exit(1)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

