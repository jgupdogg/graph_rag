#!/usr/bin/env python3
"""
Quick quality check on extracted text
"""

import re
import sys
import os

def analyze_extraction_quality(text_file):
    """Analyze the quality of extracted text"""
    
    if not os.path.exists(text_file):
        print(f"Text file not found: {text_file}")
        return False
    
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Quality Analysis for: {text_file}")
    print("=" * 50)
    
    # Basic statistics
    total_chars = len(text)
    total_words = len(text.split())
    total_lines = len(text.split('\\n'))
    
    print(f"📊 Basic Statistics:")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Total words: {total_words:,}")
    print(f"  Total lines: {total_lines:,}")
    print(f"  Average words per line: {total_words/total_lines:.1f}")
    
    # Check for potential issues
    print(f"\\n🔍 Quality Checks:")
    
    # 1. Check for garbled text patterns
    garbled_patterns = [
        r'[^\w\s\-.,;:()\[\]{}"\'\\n\\t]{3,}',  # Random characters
        r'\w{50,}',  # Very long words
        r'[A-Z]{10,}',  # Too many consecutive capitals
    ]
    
    garbled_count = 0
    for pattern in garbled_patterns:
        matches = re.findall(pattern, text)
        garbled_count += len(matches)
    
    print(f"  Garbled text sequences: {garbled_count}")
    
    # 2. Check encoding issues
    encoding_issues = len(re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', text))
    print(f"  Encoding issues: {encoding_issues}")
    
    # 3. Check for technical content indicators
    technical_patterns = [
        r'specification',
        r'section\s+\d+',
        r'\d+\.\d+\.\d+',
        r'shall\s+be',
        r'in\s+accordance\s+with',
    ]
    
    technical_matches = 0
    for pattern in technical_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        technical_matches += len(matches)
    
    print(f"  Technical indicators found: {technical_matches}")
    
    # 4. Check for proper structure
    page_markers = len(re.findall(r'\[PAGE \d+\]', text))
    print(f"  Page markers: {page_markers}")
    
    # 5. Sample some content
    print(f"\\n📝 Content Samples:")
    
    # Find table of contents
    toc_match = re.search(r'TABLE OF CONTENTS.*?(?=\[PAGE|$)', text, re.DOTALL | re.IGNORECASE)
    if toc_match:
        print(f"  ✅ Table of Contents found")
        toc_sample = toc_match.group(0)[:200]
        print(f"     Sample: {toc_sample.strip()}...")
    
    # Find specification sections
    spec_matches = re.findall(r'\d{2}\s+\d{2}\s+\d{2}.*?\n', text)
    if spec_matches:
        print(f"  ✅ Specification sections found: {len(spec_matches)}")
        print(f"     Example: {spec_matches[0].strip()}")
    
    # Calculate quality score
    quality_factors = []
    
    # Text density (words per character)
    density = total_words / total_chars if total_chars > 0 else 0
    density_score = min(density / 0.2, 1.0)  # Target ~0.2 words per char
    quality_factors.append(density_score)
    
    # Low garbled text
    garbled_score = max(0, 1 - (garbled_count / 50))
    quality_factors.append(garbled_score)
    
    # Technical content presence
    tech_score = min(technical_matches / 10, 1.0)
    quality_factors.append(tech_score)
    
    # Structure preservation
    structure_score = min(page_markers / 3, 1.0)  # Expect page markers
    quality_factors.append(structure_score)
    
    overall_quality = sum(quality_factors) / len(quality_factors)
    
    print(f"\\n🎯 Quality Score: {overall_quality:.2f}/1.00")
    
    if overall_quality >= 0.8:
        print("  ✅ Excellent quality - ready for GraphRAG")
    elif overall_quality >= 0.6:
        print("  ⚠️  Good quality - minor improvements possible")
    else:
        print("  ❌ Poor quality - needs improvement")
    
    # Recommendations
    print(f"\\n💡 Recommendations:")
    if garbled_count > 10:
        print("  • Try pdfplumber extraction method for better character accuracy")
    if technical_matches < 5:
        print("  • Document may not be technical - adjust entity types accordingly")
    if page_markers < 3:
        print("  • Enable structure preservation in processing")
    
    print(f"\\n🚀 Ready for full processing: {'YES' if overall_quality >= 0.6 else 'NEEDS WORK'}")
    
    return overall_quality >= 0.6

if __name__ == "__main__":
    import sys
    filename = sys.argv[1] if len(sys.argv) > 1 else "test_extraction_sample.txt"
    success = analyze_extraction_quality(filename)
    print(f"\\nQuality check {'PASSED' if success else 'FAILED'}")