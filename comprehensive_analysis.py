#!/usr/bin/env python3
"""
Comprehensive analysis of the 100-page Baltimore specs extraction
"""

import re
import json
import os

def comprehensive_analysis():
    """Perform comprehensive analysis of the extraction"""
    
    text_file = "baltimore_specs_100pages.txt"
    json_file = "baltimore_specs_100pages.json"
    
    print("🔍 Comprehensive Analysis: Baltimore City Engineering Specifications")
    print("📄 Sample: First 100 pages")
    print("=" * 70)
    
    # Read the text file
    if not os.path.exists(text_file):
        print(f"❌ Text file not found: {text_file}")
        return
    
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Read metadata
    metadata = {}
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    print("📊 EXTRACTION STATISTICS")
    print("-" * 30)
    print(f"Pages processed: {metadata.get('pages_processed', 'Unknown')}")
    print(f"Processing time: {metadata.get('processing_time', 0):.2f} seconds")
    print(f"Text length: {len(text):,} characters")
    print(f"File size: {os.path.getsize(text_file):,} bytes")
    
    # Word and line statistics
    words = text.split()
    lines = text.split('\\n')
    
    print(f"Word count: {len(words):,}")
    print(f"Line count: {len(lines):,}")
    print(f"Average words per line: {len(words)/len(lines):.1f}")
    
    print()
    print("🎯 CONTENT QUALITY ANALYSIS")
    print("-" * 30)
    
    # 1. Page markers analysis
    page_markers = re.findall(r'\[PAGE: (\d+)\]', text)
    print(f"Page markers found: {len(page_markers)}")
    if page_markers:
        first_page = min(int(p) for p in page_markers)
        last_page = max(int(p) for p in page_markers)
        print(f"Page range: {first_page} to {last_page}")
    
    # 2. Document structure
    print(f"\\nDocument Structure:")
    
    # Table of contents
    toc_present = 'TABLE OF CONTENTS' in text.upper()
    print(f"  Table of Contents: {'✅ Found' if toc_present else '❌ Not found'}")
    
    # Division markers
    divisions = re.findall(r'DIVISION\s+(\d+)', text, re.IGNORECASE)
    print(f"  Divisions found: {len(set(divisions))} unique")
    if divisions:
        print(f"    Examples: {', '.join(sorted(set(divisions))[:5])}")
    
    # Section numbers
    sections = re.findall(r'\b\d{2}\s+\d{2}\s+\d{2}\b', text)
    print(f"  Section numbers: {len(sections)}")
    if sections:
        print(f"    Examples: {', '.join(sections[:5])}")
    
    print()
    print("🔧 TECHNICAL CONTENT ANALYSIS")
    print("-" * 30)
    
    # Technical terms
    technical_terms = {
        'specifications': len(re.findall(r'\bspecification', text, re.IGNORECASE)),
        'requirements': len(re.findall(r'\brequirement', text, re.IGNORECASE)),
        'shall': len(re.findall(r'\bshall\b', text, re.IGNORECASE)),
        'materials': len(re.findall(r'\bmaterial', text, re.IGNORECASE)),
        'construction': len(re.findall(r'\bconstruction', text, re.IGNORECASE)),
        'contractor': len(re.findall(r'\bcontractor', text, re.IGNORECASE)),
        'work': len(re.findall(r'\bwork\b', text, re.IGNORECASE)),
    }
    
    print("Technical term frequency:")
    for term, count in technical_terms.items():
        print(f"  {term.capitalize()}: {count}")
    
    # Standards and references
    standards = re.findall(r'\b(ASTM|AASHTO|ANSI|ISO)\s*[A-Z]?\d+', text, re.IGNORECASE)
    print(f"\nStandards referenced: {len(standards)}")
    if standards:
        unique_standards = list(set(standards))[:10]
        print(f"  Examples: {', '.join(unique_standards)}")
    
    # Measurements and units
    measurements = re.findall(r'\d+(?:\.\d+)?\s*(inch|foot|feet|yard|mile|mm|cm|meter|psi|psf|mph)', text, re.IGNORECASE)
    print(f"\nMeasurements found: {len(measurements)}")
    if measurements:
        units = {}
        for match in measurements:
            unit = match.lower() if isinstance(match, str) else match[1].lower()
            units[unit] = units.get(unit, 0) + 1
        print(f"  Common units: {dict(list(units.items())[:5])}")
    
    print()
    print("⚠️  QUALITY ISSUES")
    print("-" * 30)
    
    # Character encoding issues
    encoding_chars = re.findall(r'[•·~‚„†‡‰Š‹Œ''""–—™š›œŸ]', text)
    print(f"Encoding issues: {len(encoding_chars)} characters")
    if encoding_chars:
        unique_chars = list(set(encoding_chars))[:10]
        print(f"  Problem characters: {unique_chars}")
    
    # Very long words (likely extraction errors)
    long_words = re.findall(r'\b\w{30,}\b', text)
    print(f"Very long words: {len(long_words)}")
    if long_words:
        print(f"  Examples: {long_words[:3]}")
    
    # Repeated characters (extraction artifacts)
    repeated = re.findall(r'(.)\1{5,}', text)
    print(f"Repeated character sequences: {len(repeated)}")
    
    print()
    print("📈 GRAPHRAG READINESS ASSESSMENT")
    print("-" * 30)
    
    # Calculate readiness score
    factors = []
    
    # Content coverage (pages found vs expected)
    expected_pages = 100
    actual_pages = len(page_markers)
    coverage = actual_pages / expected_pages if expected_pages > 0 else 0
    factors.append(('Page coverage', coverage, 0.9 if coverage > 0.9 else 0.7 if coverage > 0.7 else 0.3))
    
    # Technical content density
    total_technical = sum(technical_terms.values())
    tech_density = total_technical / len(words) * 1000 if words else 0
    tech_score = min(tech_density / 50, 1.0)  # Target ~50 technical terms per 1000 words
    factors.append(('Technical density', tech_density, tech_score))
    
    # Structure preservation
    structure_elements = len(sections) + len(divisions) + (1 if toc_present else 0)
    structure_score = min(structure_elements / 50, 1.0)
    factors.append(('Structure elements', structure_elements, structure_score))
    
    # Content quality (low encoding issues)
    quality_score = max(0, 1 - (len(encoding_chars) / 100))
    factors.append(('Character quality', len(encoding_chars), quality_score))
    
    print("Readiness factors:")
    for name, value, score in factors:
        print(f"  {name}: {value} (score: {score:.2f})")
    
    overall_score = sum(score for _, _, score in factors) / len(factors)
    print(f"\\n🎯 Overall Readiness Score: {overall_score:.2f}/1.00")
    
    if overall_score >= 0.8:
        status = "✅ EXCELLENT - Ready for GraphRAG"
        recommendation = "Proceed with full document processing"
    elif overall_score >= 0.6:
        status = "⚠️  GOOD - Minor improvements needed"
        recommendation = "Consider using pdfplumber for better quality"
    else:
        status = "❌ POOR - Needs significant improvement"
        recommendation = "Try different extraction method or preprocessing"
    
    print(f"Status: {status}")
    print(f"Recommendation: {recommendation}")
    
    print()
    print("💡 PROCESSING RECOMMENDATIONS")
    print("-" * 30)
    
    if len(encoding_chars) > 50:
        print("• Use pdfplumber instead of PyMuPDF for better character handling")
    
    if actual_pages < expected_pages * 0.9:
        print("• Check page range settings and PDF structure")
    
    if tech_density < 30:
        print("• Verify this is a technical document")
        print("• Adjust GraphRAG entity types for procurement/contracting content")
    
    print("• For GraphRAG processing:")
    print(f"  - Recommended chunk size: 600-800 tokens")
    print(f"  - Overlap: 150-200 tokens")
    print(f"  - Entity types: ['specification', 'requirement', 'contractor', 'material', 'procedure', 'standard']")
    
    # Show sample content
    print()
    print("📖 CONTENT SAMPLE")
    print("-" * 30)
    
    # Find a good content section
    content_start = text.find('[DOCUMENT_CONTENT_START]')
    if content_start != -1:
        sample_start = content_start + len('[DOCUMENT_CONTENT_START]')
        sample = text[sample_start:sample_start + 1000].strip()
        print(sample[:800] + "..." if len(sample) > 800 else sample)
    
    return overall_score >= 0.6

if __name__ == "__main__":
    success = comprehensive_analysis()
    print(f"\\n{'✅ READY FOR FULL PROCESSING' if success else '❌ NEEDS IMPROVEMENT'}")