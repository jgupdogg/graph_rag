#!/usr/bin/env python3
"""
Extract and clean Baltimore City specifications text for GraphRAG processing
Focus on procurement and contracting sections starting around line 829
"""

import re
import os

def extract_and_clean_baltimore_specs():
    """Extract and clean Baltimore specs text starting from line 829"""
    
    input_file = "baltimore_specs_complete.txt"
    output_file = "baltimore_specs_procurement_section.txt"
    
    print("🔍 Extracting Baltimore City Specs - Procurement Section")
    print("=" * 60)
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Read the complete file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the start line (around line 829 - "INSTRUCTIONS")
    start_line = None
    for i, line in enumerate(lines):
        if "INSTRUCTIONS" in line and "00 2113.01" in lines[i+1] if i+1 < len(lines) else False:
            start_line = i
            break
    
    if start_line is None:
        print("❌ Could not find starting point (INSTRUCTIONS section)")
        return
    
    # Extract approximately 1000 lines from the start point (covers multiple sections)
    end_line = min(start_line + 1000, len(lines))
    extracted_lines = lines[start_line:end_line]
    
    print(f"📄 Extracting lines {start_line+1} to {end_line}")
    print(f"📊 Total lines extracted: {len(extracted_lines)}")
    
    # Clean the text
    cleaned_text = []
    
    for line in extracted_lines:
        # Remove line numbers (format: "   829→")
        line = re.sub(r'^\s*\d+→', '', line)
        
        # Remove page markers (format: "[PAGE: 18]")
        line = re.sub(r'\[PAGE: \d+\]', '', line)
        
        # Remove extra whitespace but preserve structure
        line = line.strip()
        
        # Skip empty lines that were just line numbers
        if line:
            cleaned_text.append(line)
    
    # Join with newlines and clean up extra spaces
    final_text = '\n'.join(cleaned_text)
    
    # Remove excessive newlines (more than 2 consecutive)
    final_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', final_text)
    
    # Write the cleaned text
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_text)
    
    print(f"✅ Cleaned text saved to: {output_file}")
    print(f"📈 Character count: {len(final_text):,}")
    print(f"📝 Word count: {len(final_text.split()):,}")
    
    # Show a preview of the cleaned text
    print("\n📋 PREVIEW OF CLEANED TEXT:")
    print("-" * 40)
    preview_lines = final_text.split('\n')[:20]
    for line in preview_lines:
        print(line)
    
    if len(preview_lines) == 20:
        print("... (content continues)")
    
    return output_file

if __name__ == "__main__":
    extract_and_clean_baltimore_specs()