#!/usr/bin/env python3
"""
Create simple section summaries for documents based on their structure.
This is a lightweight alternative that doesn't require extensive AI processing.
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List

def extract_sections_from_text(text: str, filename: str) -> Dict[str, str]:
    """Extract sections from text based on headers and structure."""
    sections = {}
    
    # Common section patterns
    patterns = [
        r'^#{1,3}\s+(.+)$',  # Markdown headers
        r'^Chapter\s+\d+[:\s]+(.+)$',  # Chapter headings
        r'^Section\s+\d+[:\s]+(.+)$',  # Section headings
        r'^\d+\.\s+(.+)$',  # Numbered sections
        r'^[A-Z][A-Z\s]+$',  # ALL CAPS headers
    ]
    
    lines = text.split('\n')
    current_section = "Introduction"
    current_content = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Check if this line is a section header
        is_header = False
        for pattern in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                # Save previous section
                if current_content:
                    content = ' '.join(current_content)[:500]  # First 500 chars
                    sections[current_section] = f"This section covers: {content}..."
                
                # Start new section
                current_section = line
                current_content = []
                is_header = True
                break
        
        if not is_header and len(current_content) < 10:  # Limit content per section
            current_content.append(line)
    
    # Save last section
    if current_content:
        content = ' '.join(current_content)[:500]
        sections[current_section] = f"This section covers: {content}..."
    
    # If no sections found, create a basic structure
    if len(sections) <= 1:
        # Try to create sections based on content chunks
        chunk_size = len(text) // 4  # Divide into 4 parts
        sections = {
            "Overview": f"Document overview: {text[:500]}...",
            "Main Content Part 1": f"This section covers: {text[chunk_size:chunk_size+500]}...",
            "Main Content Part 2": f"This section covers: {text[chunk_size*2:chunk_size*2+500]}...",
            "Conclusion": f"Final section: {text[-500:]}..."
        }
    
    return sections


def create_simple_section_summaries():
    """Create section summaries for all documents missing them."""
    
    print("=== Simple Section Summaries Creator ===\n")
    
    workspace_base = Path("workspaces")
    created_count = 0
    
    # Process each workspace
    for workspace_path in workspace_base.glob("*"):
        if not workspace_path.is_dir():
            continue
            
        # Check if section summaries already exist
        section_summaries_file = workspace_path / "cache" / "section_summaries.json"
        if section_summaries_file.exists():
            print(f"✓ {workspace_path.name}: Already has section summaries")
            continue
        
        # Find input text file
        input_dir = workspace_path / "input"
        if not input_dir.exists():
            continue
            
        txt_files = list(input_dir.glob("*.txt"))
        if not txt_files:
            print(f"✗ {workspace_path.name}: No text file found")
            continue
            
        input_file = txt_files[0]
        
        try:
            # Read content
            with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            print(f"\n📄 Processing {input_file.name} in {workspace_path.name}")
            
            # Extract sections
            sections = extract_sections_from_text(content, input_file.name)
            print(f"   Found {len(sections)} sections")
            
            # Save section summaries
            cache_dir = workspace_path / "cache"
            cache_dir.mkdir(exist_ok=True)
            
            summary_data = {
                "document_id": workspace_path.name,
                "document_name": input_file.name,
                "generated_at": datetime.now().isoformat(),
                "sections": sections
            }
            
            with open(section_summaries_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ Created section summaries")
            created_count += 1
            
            # Show sample sections
            for section_name in list(sections.keys())[:3]:
                print(f"   - {section_name}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Created section summaries for {created_count} documents")
    print("\nNote: Restart the Streamlit app to see the changes.")


if __name__ == "__main__":
    create_simple_section_summaries()