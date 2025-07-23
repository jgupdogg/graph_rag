#!/usr/bin/env python3
"""
Create missing processing_info.json files for workspaces.
"""

import os
import json
from pathlib import Path
from datetime import datetime

def create_processing_info_files():
    """Create processing_info.json for workspaces that don't have them."""
    
    workspace_base = Path("workspaces")
    created_count = 0
    
    # Mapping of workspace IDs to document names (from the database output)
    workspace_mappings = {
        "8cc84a8b-2383-48f3-b6c6-98db7c608457": "chapter4.pdf",
        "ea71556b-a91c-4339-ad6e-fe61dfb89155": "chapter2.pdf",
        "8f4b1b51-56ae-4e58-accb-4a63e04fafc3": "chapter2.pdf"
    }
    
    for workspace_id, doc_name in workspace_mappings.items():
        workspace_path = workspace_base / workspace_id
        
        if workspace_path.exists():
            cache_dir = workspace_path / "cache"
            cache_dir.mkdir(exist_ok=True)
            
            processing_info_file = cache_dir / "processing_info.json"
            
            if not processing_info_file.exists():
                # Create processing info
                processing_info = {
                    "document_id": workspace_id,
                    "document_name": doc_name,
                    "workspace_path": str(workspace_path),
                    "created_at": datetime.now().isoformat(),
                    "status": "completed"
                }
                
                with open(processing_info_file, 'w') as f:
                    json.dump(processing_info, f, indent=2)
                
                print(f"✅ Created processing_info.json for {doc_name} ({workspace_id})")
                created_count += 1
            else:
                print(f"ℹ️  Processing info already exists for {doc_name}")
    
    print(f"\nCreated {created_count} processing_info.json files")


if __name__ == "__main__":
    create_processing_info_files()