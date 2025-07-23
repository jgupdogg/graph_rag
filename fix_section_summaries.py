#!/usr/bin/env python3
"""
Fix missing section summaries for existing documents.
This script regenerates section summaries for documents that don't have them.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app_logic import get_all_documents, get_document_by_id
from enhanced_document_processor import EnhancedDocumentProcessor
from structure_aware_chunking import StructureAwareChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_and_fix_section_summaries():
    """Check all documents and regenerate missing section summaries."""
    
    print("=== Section Summaries Fix Tool ===\n")
    
    # Get all documents
    all_docs = get_all_documents()
    print(f"Found {len(all_docs)} documents in database\n")
    
    documents_missing_summaries = []
    documents_with_summaries = []
    
    # Check each document
    for doc in all_docs:
        doc_details = get_document_by_id(doc['id'])
        
        if doc_details:
            has_summaries = bool(doc_details.get('section_summaries'))
            
            if has_summaries:
                documents_with_summaries.append(doc)
                print(f"✓ {doc['display_name']}: Has section summaries")
            else:
                documents_missing_summaries.append(doc)
                print(f"✗ {doc['display_name']}: Missing section summaries")
        else:
            print(f"? {doc['display_name']}: Could not load details")
    
    print(f"\nSummary:")
    print(f"- Documents with section summaries: {len(documents_with_summaries)}")
    print(f"- Documents missing section summaries: {len(documents_missing_summaries)}")
    
    if not documents_missing_summaries:
        print("\nAll documents have section summaries! Nothing to fix.")
        return
    
    # Ask user if they want to fix
    print(f"\nWould you like to generate section summaries for the {len(documents_missing_summaries)} documents missing them?")
    response = input("Type 'yes' to proceed: ").strip().lower()
    
    if response != 'yes':
        print("Aborted.")
        return
    
    # Fix missing summaries
    print("\nGenerating section summaries...")
    
    # Use dummy API key if not set (for structure extraction only)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GRAPHRAG_API_KEY") or "dummy"
    processor = EnhancedDocumentProcessor(api_key=api_key)
    chunker = StructureAwareChunker()
    
    fixed_count = 0
    
    for doc in documents_missing_summaries:
        print(f"\nProcessing: {doc['display_name']}")
        
        try:
            # Find workspace
            workspace_base = Path("workspaces")
            workspace_dir = None
            
            for workspace_path in workspace_base.glob("*"):
                if workspace_path.is_dir():
                    # Check if this workspace has the document
                    cache_info_file = workspace_path / "cache" / "processing_info.json"
                    if cache_info_file.exists():
                        with open(cache_info_file, 'r') as f:
                            info = json.load(f)
                            if info.get('document_id') == doc['id']:
                                workspace_dir = workspace_path
                                break
            
            if not workspace_dir:
                print(f"  ❌ Could not find workspace for document {doc['id']}")
                continue
            
            # Check for input file
            input_dir = workspace_dir / "input"
            input_files = list(input_dir.glob("*")) if input_dir.exists() else []
            
            if not input_files:
                print(f"  ❌ No input file found in {input_dir}")
                continue
            
            input_file = input_files[0]
            
            # Read document content
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"  📄 Read {len(content)} characters from {input_file.name}")
            
            # Extract structure and create chunks
            chunks = chunker.chunk_document(content, str(input_file))
            print(f"  📊 Created {len(chunks)} chunks")
            
            # Create section summaries
            section_summaries = chunker.create_section_summaries(chunks)
            print(f"  📝 Generated {len(section_summaries)} section summaries")
            
            if section_summaries:
                # Save to cache
                cache_dir = workspace_dir / "cache"
                cache_dir.mkdir(exist_ok=True)
                
                section_summaries_file = cache_dir / "section_summaries.json"
                
                summary_data = {
                    "document_id": doc['id'],
                    "document_name": doc['display_name'],
                    "generated_at": str(Path.ctime(Path())),
                    "sections": section_summaries
                }
                
                with open(section_summaries_file, 'w', encoding='utf-8') as f:
                    json.dump(summary_data, f, indent=2, ensure_ascii=False)
                
                print(f"  ✅ Saved section summaries to {section_summaries_file}")
                fixed_count += 1
            else:
                print(f"  ⚠️  No section summaries generated (document may lack structure)")
            
        except Exception as e:
            print(f"  ❌ Error processing document: {e}")
            logger.error(f"Failed to fix section summaries for {doc['display_name']}: {e}")
    
    print(f"\n✅ Fixed section summaries for {fixed_count}/{len(documents_missing_summaries)} documents")
    print("\nNote: You may need to restart the Streamlit app to see the changes.")


if __name__ == "__main__":
    check_and_fix_section_summaries()