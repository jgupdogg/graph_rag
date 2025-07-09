#!/usr/bin/env python3
"""
Process a large sample (100 pages) of the Baltimore City Engineering Specifications
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add pdf_processing to Python path
sys.path.insert(0, 'pdf_processing')

from pdf_to_text_processor import PDFProcessor

def process_large_sample():
    """Process first 100 pages with optimized settings"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('large_sample_processing.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    pdf_path = "graphrag/input/CityofBaltimoreSpecifications(GreenBook)-2006.pdf"
    output_path = "baltimore_specs_100pages.txt"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    print("🚀 Processing Baltimore City Engineering Specifications")
    print("📄 Target: First 100 pages")
    print("=" * 60)
    
    # Create processor with optimized settings for large sample
    processor = PDFProcessor(
        chunk_size=50,          # 50 pages per chunk for better quality
        overlap_size=5,         # 5-page overlap for continuity
        max_workers=2,          # Conservative for stability
        cache_dir="pdf_cache"   # Enable caching
    )
    
    try:
        # Get PDF info
        logger.info("Getting PDF information...")
        pdf_info = processor.get_pdf_info(pdf_path)
        
        print(f"📊 PDF Information:")
        print(f"   File size: {pdf_info['file_size']:,} bytes ({pdf_info['file_size']/1024/1024:.1f} MB)")
        print(f"   Total pages: {pdf_info['total_pages']:,}")
        print(f"   Processing: First 100 pages ({100/pdf_info['total_pages']*100:.1f}% of document)")
        print()
        
        # Process first 100 pages
        start_time = time.time()
        
        # Temporarily modify the processor to only process first 100 pages
        # We'll do this by processing in a custom way
        
        logger.info("Starting extraction of first 100 pages...")
        print("🔄 Processing in progress...")
        
        # Use PyMuPDF directly for controlled page range
        import fitz
        
        all_text_parts = []
        tables = []
        structure = {'pages': [], 'sections': []}
        pages_processed = 0
        
        with fitz.open(pdf_path) as doc:
            target_pages = min(100, doc.page_count)
            
            for page_num in range(target_pages):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    
                    if page_text.strip():
                        # Add page marker and text
                        all_text_parts.append(f"[PAGE: {page_num + 1}]\\n")
                        all_text_parts.append(page_text)
                        all_text_parts.append("\\n\\n")
                        
                        pages_processed += 1
                        
                        # Progress indicator
                        if (page_num + 1) % 10 == 0:
                            progress = (page_num + 1) / target_pages * 100
                            print(f"   📄 Processed {page_num + 1:3d}/{target_pages} pages ({progress:5.1f}%)")
                    
                    # Basic structure detection
                    if page_text:
                        # Look for section headers
                        lines = page_text.split('\\n')
                        for line_num, line in enumerate(lines):
                            line = line.strip()
                            if line and len(line) < 100:  # Potential header
                                # Check for section patterns
                                import re
                                if re.match(r'^\\d{2}\\s+\\d{2}\\s+\\d{2}', line):
                                    structure['sections'].append({
                                        'page': page_num + 1,
                                        'line': line_num,
                                        'text': line,
                                        'type': 'specification_section'
                                    })
                                elif re.match(r'^[A-Z][A-Z\\s]{10,}$', line):
                                    structure['sections'].append({
                                        'page': page_num + 1,
                                        'line': line_num,
                                        'text': line,
                                        'type': 'major_header'
                                    })
                        
                        structure['pages'].append({
                            'page': page_num + 1,
                            'char_count': len(page_text),
                            'line_count': len(lines)
                        })
                
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {e}")
                    continue
        
        # Combine all text
        final_text = ''.join(all_text_parts)
        processing_time = time.time() - start_time
        
        # Add document metadata header
        metadata_header = [
            "[DOCUMENT_START]",
            f"[TITLE: Baltimore City Engineering Specifications (Green Book) 2006]",
            f"[AUTHOR: City of Baltimore Department of Public Works]",
            f"[TOTAL_PAGES: {pdf_info['total_pages']}]",
            f"[PROCESSED_PAGES: {pages_processed}]",
            f"[FILE_SIZE: {pdf_info['file_size']} bytes]",
            f"[PROCESSING_TIME: {processing_time:.2f} seconds]",
            f"[SECTIONS_FOUND: {len(structure['sections'])}]",
            "[DOCUMENT_CONTENT_START]",
            ""
        ]
        
        final_output = "\\n".join(metadata_header) + "\\n" + final_text
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_output)
        
        # Save metadata
        metadata_file = output_path.replace('.txt', '.json')
        import json
        metadata = {
            'source_file': pdf_path,
            'pages_processed': pages_processed,
            'total_pages': pdf_info['total_pages'],
            'processing_time': processing_time,
            'file_size': pdf_info['file_size'],
            'text_length': len(final_text),
            'sections_found': len(structure['sections']),
            'structure': structure,
            'extraction_method': 'pymupdf',
            'timestamp': time.time()
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Print results
        print()
        print("✅ Processing completed successfully!")
        print("=" * 60)
        print(f"📄 Pages processed: {pages_processed}/{pdf_info['total_pages']:,}")
        print(f"📝 Text extracted: {len(final_text):,} characters")
        print(f"📁 Sections found: {len(structure['sections'])}")
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"🚀 Speed: {pages_processed/processing_time:.1f} pages/second")
        print(f"💾 Output saved to: {output_path}")
        print(f"📊 Metadata saved to: {metadata_file}")
        
        # Quick quality preview
        print()
        print("🔍 Quick Quality Check:")
        
        # Check for technical content
        import re
        spec_sections = len(re.findall(r'\\d{2}\\s+\\d{2}\\s+\\d{2}', final_text))
        print(f"   Specification sections: {spec_sections}")
        
        technical_terms = len(re.findall(r'\\b(concrete|steel|asphalt|specification|shall|minimum|maximum)\\b', final_text, re.IGNORECASE))
        print(f"   Technical terms: {technical_terms}")
        
        page_markers = len(re.findall(r'\\[PAGE: \\d+\\]', final_text))
        print(f"   Page markers: {page_markers}")
        
        # Sample content
        print()
        print("📖 Sample Content (first 500 characters):")
        print("-" * 50)
        sample = final_text[final_text.find('[DOCUMENT_CONTENT_START]'):].replace('[DOCUMENT_CONTENT_START]\\n', '')[:500]
        print(sample)
        if len(final_text) > 500:
            print("...")
        print("-" * 50)
        
        logger.info(f"Successfully processed {pages_processed} pages in {processing_time:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = process_large_sample()
    if success:
        print("\\n🎉 Large sample processing completed!")
        print("Run quality validation next with:")
        print("python test_quality_check.py baltimore_specs_100pages.txt")
    else:
        print("\\n❌ Processing failed. Check the logs for details.")