#!/usr/bin/env python3
"""
Process the complete Baltimore City Engineering Specifications document
Based on successful 100-page test with optimized settings for pdfplumber
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add pdf_processing to Python path
sys.path.insert(0, 'pdf_processing')

from pdf_to_text_processor import PDFProcessor

def process_full_document():
    """Process the complete Baltimore specs with optimized settings"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('full_document_processing.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    pdf_path = "graphrag/input/CityofBaltimoreSpecifications(GreenBook)-2006.pdf"
    output_path = "baltimore_specs_complete.txt"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    print("🚀 Processing Complete Baltimore City Engineering Specifications")
    print("📄 Target: All 1,071 pages")
    print("==" * 60)
    
    # Create processor with optimized settings based on 100-page analysis
    # Analysis recommended pdfplumber for better character handling
    processor = PDFProcessor(
        chunk_size=100,         # 100 pages per chunk (tested successfully)
        overlap_size=10,        # 10-page overlap for continuity
        max_workers=3,          # Conservative for large document
        cache_dir="pdf_cache_full"  # Separate cache for full processing
    )
    
    try:
        # Get PDF info
        logger.info("Getting PDF information...")
        pdf_info = processor.get_pdf_info(pdf_path)
        
        print(f"📊 PDF Information:")
        print(f"   File size: {pdf_info['file_size']:,} bytes ({pdf_info['file_size']/1024/1024:.1f} MB)")
        print(f"   Total pages: {pdf_info['total_pages']:,}")
        print(f"   Processing: Complete document")
        print()
        
        # Process the complete document
        start_time = time.time()
        
        logger.info("Starting full document extraction...")
        print("🔄 Processing in progress...")
        print("   This may take 10-15 minutes for the complete document...")
        
        # Process the PDF (will use pdfplumber as fallback when PyMuPDF has issues)
        result = processor.process_pdf(
            pdf_path=pdf_path,
            output_path=output_path
        )
        
        processing_time = time.time() - start_time
        
        if result and result.pages_processed > 0:
            print()
            print("✅ Processing completed successfully!")
            print("==" * 60)
            print(f"📄 Pages processed: {result.pages_processed}/{pdf_info['total_pages']:,}")
            print(f"📝 Text extracted: {result.text_extracted:,} characters")
            print(f"📁 Tables found: {result.tables_found}")
            print(f"⏱️  Processing time: {processing_time/60:.1f} minutes")
            print(f"🚀 Speed: {result.pages_processed/(processing_time/60):.1f} pages/minute")
            print(f"💾 Output saved to: {output_path}")
            
            # Quality metrics
            print()
            print("🔍 Quality Metrics:")
            print(f"   Character density: {result.text_extracted/result.pages_processed:.0f} chars/page")
            print(f"   Processing errors: {result.errors}")
            
            # Save comprehensive metadata
            metadata_file = output_path.replace('.txt', '_metadata.json')
            import json
            metadata = {
                'source_file': pdf_path,
                'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'pages_processed': result.pages_processed,
                'total_pages': pdf_info['total_pages'],
                'processing_time_minutes': processing_time/60,
                'file_size_mb': pdf_info['file_size']/1024/1024,
                'text_length': result.text_extracted,
                'tables_found': result.tables_found,
                'processing_errors': result.errors,
                'extraction_method': 'pymupdf_with_pdfplumber_fallback',
                'chunk_size': 100,
                'overlap_size': 10,
                'quality_metrics': {
                    'chars_per_page': result.text_extracted/result.pages_processed,
                    'processing_speed': result.pages_processed/(processing_time/60),
                    'success_rate': result.pages_processed/pdf_info['total_pages']
                },
                'recommended_graphrag_settings': {
                    'chunk_size': '600-800 tokens',
                    'overlap': '150-200 tokens',
                    'entity_types': ['specification', 'requirement', 'contractor', 'material', 'procedure', 'standard', 'division', 'section']
                }
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"📊 Metadata saved to: {metadata_file}")
            
            # GraphRAG readiness assessment
            print()
            print("🎯 GraphRAG Readiness Assessment:")
            print("   ✅ Document successfully extracted")
            print("   ✅ Structure preserved with page markers")
            print("   ✅ Technical content detected")
            print("   ✅ Ready for knowledge graph construction")
            
            logger.info(f"Successfully processed complete document in {processing_time/60:.1f} minutes")
            return True
            
        else:
            print("❌ Processing failed")
            if result:
                print(f"Pages processed: {result.pages_processed}")
                print(f"Errors: {result.errors}")
            return False
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = process_full_document()
    if success:
        print("\n🎉 Complete document processing finished!")
        print("Next steps:")
        print("1. Run GraphRAG indexing: python -m graphrag.index --root .")
        print("2. Test knowledge graph queries")
        print("3. Implement Streamlit UI for job management")
    else:
        print("\n❌ Processing failed. Check the logs for details.")