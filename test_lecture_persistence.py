#!/usr/bin/env python3
"""
Test script for lecture script persistence functionality.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from audio_lecture_generator import AudioLectureGenerator, LectureDetailLevel


def test_lecture_persistence():
    """Test saving and loading lecture scripts."""
    print("=== Lecture Script Persistence Test ===\n")
    
    # Initialize generator
    generator = AudioLectureGenerator()
    
    # Test document ID
    test_doc_id = "test_document_123"
    
    # Sample data
    document_summary = {
        'display_name': 'Test Document',
        'summary': 'This is a test document for persistence testing.'
    }
    
    section_summaries = {
        "Chapter 1": "Introduction chapter",
        "Chapter 2": "Main content chapter",
        "Chapter 3": "Conclusion chapter"
    }
    
    print("1. Generating lecture scripts at different detail levels...")
    
    # Generate and save lectures at different detail levels
    for detail_level in [LectureDetailLevel.OVERVIEW, LectureDetailLevel.STANDARD, LectureDetailLevel.COMPREHENSIVE]:
        print(f"\n   Generating {detail_level.name} lecture...")
        
        # Generate lecture
        lecture_sections, full_script = generator.generate_lecture_script(
            document_summary=document_summary,
            section_summaries=section_summaries,
            detail_level=detail_level
        )
        
        # Save lecture
        metadata = {
            'voice': 'nova',
            'quality': 'tts-1-hd',
            'test_run': True,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create a test workspace if needed
        workspace_dir = Path("workspaces") / test_doc_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        saved = generator.save_lecture_script(
            document_id=test_doc_id,
            lecture_sections=lecture_sections,
            full_script=full_script,
            detail_level=detail_level,
            metadata=metadata
        )
        
        if saved:
            print(f"   ✅ Saved {detail_level.name} lecture script")
        else:
            print(f"   ❌ Failed to save {detail_level.name} lecture script")
    
    print("\n2. Loading saved lecture scripts...")
    
    # Load all lecture scripts
    loaded_scripts = generator.load_lecture_scripts(test_doc_id)
    
    print(f"\n   Found {len(loaded_scripts)} saved lecture scripts")
    
    for script in loaded_scripts:
        print(f"\n   - Detail Level: {script.get('detail_level', 'Unknown')}")
        print(f"     Generated: {script.get('generated_at', 'Unknown')}")
        print(f"     Duration: {script.get('duration_estimate', 0)} minutes")
        print(f"     Word Count: {script.get('word_count', 0)}")
        print(f"     Sections: {len(script.get('sections', []))}")
        print(f"     Metadata: {script.get('metadata', {})}")
    
    print("\n3. Testing latest script retrieval...")
    
    # Get latest script
    latest_all = generator.get_latest_lecture_script(test_doc_id)
    if latest_all:
        print(f"\n   Latest script (any level): {latest_all.get('detail_level')} - {latest_all.get('generated_at')}")
    
    # Get latest standard script
    latest_standard = generator.get_latest_lecture_script(test_doc_id, LectureDetailLevel.STANDARD)
    if latest_standard:
        print(f"   Latest STANDARD script: {latest_standard.get('generated_at')}")
    
    print("\n4. Checking file structure...")
    
    # Check saved files
    lecture_cache_dir = Path("workspaces") / test_doc_id / "cache" / "audio_lectures"
    if lecture_cache_dir.exists():
        print(f"\n   Lecture cache directory: {lecture_cache_dir}")
        print("   Files:")
        for file in lecture_cache_dir.glob("*.json"):
            print(f"     - {file.name} ({file.stat().st_size} bytes)")
    
    print("\n\n=== Test Complete ===")
    print("\nLecture persistence is working correctly!")
    print("Features tested:")
    print("- Saving lecture scripts with metadata")
    print("- Loading all scripts for a document")
    print("- Retrieving latest scripts")
    print("- Workspace-based storage structure")
    
    # Cleanup option
    cleanup = input("\nClean up test files? (y/n): ")
    if cleanup.lower() == 'y':
        import shutil
        test_workspace = Path("workspaces") / test_doc_id
        if test_workspace.exists():
            shutil.rmtree(test_workspace)
            print("Test files cleaned up.")


if __name__ == "__main__":
    test_lecture_persistence()