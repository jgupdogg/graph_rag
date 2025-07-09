#!/usr/bin/env python3
"""
PDF to Text Processor for GraphRAG
Handles large technical PDFs with memory-efficient streaming and structure preservation
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json

# PDF processing libraries
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import camelot
    HAS_CAMELOT = True
except ImportError:
    HAS_CAMELOT = False


@dataclass
class ProcessingMetrics:
    """Tracks processing statistics"""
    pages_processed: int = 0
    total_pages: int = 0
    text_extracted: int = 0
    tables_found: int = 0
    errors: int = 0
    start_time: float = 0
    processing_time: float = 0
    
    def add_chunk_metrics(self, chunk_metrics: 'ChunkMetrics'):
        self.pages_processed += chunk_metrics.pages
        self.text_extracted += len(chunk_metrics.text)
        self.tables_found += len(chunk_metrics.tables)
        self.errors += len(chunk_metrics.errors)


@dataclass
class ChunkMetrics:
    """Metrics for a single chunk"""
    pages: int
    text: str
    tables: List[Dict]
    structure: Dict
    errors: List[str]
    method_used: str


class PDFProcessor:
    """
    Main PDF processor with memory-efficient chunking and multiple extraction strategies
    """
    
    def __init__(self, 
                 chunk_size: int = 100,
                 overlap_size: int = 10,
                 max_workers: int = 4,
                 cache_dir: Optional[str] = None):
        """
        Initialize PDF processor
        
        Args:
            chunk_size: Number of pages per chunk
            overlap_size: Number of overlapping pages between chunks
            max_workers: Number of parallel workers
            cache_dir: Directory for caching processed results
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.max_workers = max_workers
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        self.logger = logging.getLogger(__name__)
        self._validate_dependencies()
    
    def _validate_dependencies(self):
        """Check if required libraries are available"""
        if not HAS_PYMUPDF and not HAS_PDFPLUMBER:
            raise ImportError("Either PyMuPDF or pdfplumber is required")
        
        if not HAS_PYMUPDF:
            self.logger.warning("PyMuPDF not available, using pdfplumber only")
        if not HAS_PDFPLUMBER:
            self.logger.warning("pdfplumber not available, using PyMuPDF only")
        if not HAS_CAMELOT:
            self.logger.warning("Camelot not available, table extraction limited")
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """Get basic PDF information"""
        info = {
            'file_size': os.path.getsize(pdf_path),
            'total_pages': 0,
            'title': '',
            'author': '',
            'subject': ''
        }
        
        if HAS_PYMUPDF:
            try:
                with fitz.open(pdf_path) as doc:
                    info['total_pages'] = doc.page_count
                    metadata = doc.metadata
                    info.update({
                        'title': metadata.get('title', ''),
                        'author': metadata.get('author', ''),
                        'subject': metadata.get('subject', '')
                    })
            except Exception as e:
                self.logger.error(f"Error getting PDF info with PyMuPDF: {e}")
        
        elif HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    info['total_pages'] = len(pdf.pages)
                    if pdf.metadata:
                        info.update({
                            'title': pdf.metadata.get('Title', ''),
                            'author': pdf.metadata.get('Author', ''),
                            'subject': pdf.metadata.get('Subject', '')
                        })
            except Exception as e:
                self.logger.error(f"Error getting PDF info with pdfplumber: {e}")
        
        return info
    
    def _get_cache_key(self, pdf_path: str, chunk_start: int, chunk_end: int) -> str:
        """Generate cache key for a chunk"""
        pdf_hash = hashlib.md5(Path(pdf_path).read_bytes()).hexdigest()[:8]
        return f"{pdf_hash}_{chunk_start}_{chunk_end}.json"
    
    def _load_from_cache(self, cache_key: str) -> Optional[ChunkMetrics]:
        """Load chunk metrics from cache"""
        if not self.cache_dir:
            return None
            
        cache_file = self.cache_dir / cache_key
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return ChunkMetrics(**data)
            except Exception as e:
                self.logger.warning(f"Error loading cache {cache_key}: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, metrics: ChunkMetrics):
        """Save chunk metrics to cache"""
        if not self.cache_dir:
            return
            
        cache_file = self.cache_dir / cache_key
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'pages': metrics.pages,
                    'text': metrics.text,
                    'tables': metrics.tables,
                    'structure': metrics.structure,
                    'errors': metrics.errors,
                    'method_used': metrics.method_used
                }, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Error saving cache {cache_key}: {e}")
    
    def _extract_chunk_pymupdf(self, pdf_path: str, start_page: int, end_page: int) -> ChunkMetrics:
        """Extract text from a chunk using PyMuPDF"""
        text_parts = []
        tables = []
        structure = {'sections': [], 'pages': []}
        errors = []
        
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(start_page, min(end_page, doc.page_count)):
                    try:
                        page = doc[page_num]
                        
                        # Extract text
                        page_text = page.get_text()
                        if page_text.strip():
                            # Add page marker
                            text_parts.append(f"[PAGE: {page_num + 1}]\\n")
                            text_parts.append(page_text)
                            text_parts.append("\\n\\n")
                        
                        # Basic structure detection
                        blocks = page.get_text("dict")["blocks"]
                        page_structure = self._analyze_page_structure(blocks, page_num + 1)
                        structure['pages'].append(page_structure)
                        
                    except Exception as e:
                        errors.append(f"Page {page_num + 1}: {str(e)}")
                        
        except Exception as e:
            errors.append(f"Document error: {str(e)}")
        
        return ChunkMetrics(
            pages=end_page - start_page,
            text=''.join(text_parts),
            tables=tables,
            structure=structure,
            errors=errors,
            method_used='pymupdf'
        )
    
    def _extract_chunk_pdfplumber(self, pdf_path: str, start_page: int, end_page: int) -> ChunkMetrics:
        """Extract text from a chunk using pdfplumber"""
        text_parts = []
        tables = []
        structure = {'sections': [], 'pages': []}
        errors = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num in range(start_page, min(end_page, len(pdf.pages))):
                    try:
                        page = pdf.pages[page_num]
                        
                        # Extract text
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(f"[PAGE: {page_num + 1}]\\n")
                            text_parts.append(page_text)
                            text_parts.append("\\n\\n")
                        
                        # Extract tables
                        page_tables = page.extract_tables()
                        if page_tables:
                            for i, table in enumerate(page_tables):
                                if table and len(table) > 1:  # Skip empty tables
                                    table_text = self._format_table(table, page_num + 1, i)
                                    text_parts.append(table_text)
                                    tables.append({
                                        'page': page_num + 1,
                                        'table_index': i,
                                        'data': table
                                    })
                        
                    except Exception as e:
                        errors.append(f"Page {page_num + 1}: {str(e)}")
                        
        except Exception as e:
            errors.append(f"Document error: {str(e)}")
        
        return ChunkMetrics(
            pages=end_page - start_page,
            text=''.join(text_parts),
            tables=tables,
            structure=structure,
            errors=errors,
            method_used='pdfplumber'
        )
    
    def _format_table(self, table: List[List[str]], page_num: int, table_index: int) -> str:
        """Format table for GraphRAG processing"""
        if not table or len(table) < 2:
            return ""
        
        formatted = [f"\\n[TABLE_START: Page {page_num}, Table {table_index + 1}]\\n"]
        
        # Header row
        headers = table[0]
        if headers:
            formatted.append(" | ".join(str(cell) if cell else "" for cell in headers))
            formatted.append("\\n")
            formatted.append("-" * (len(" | ".join(headers)) + 10))
            formatted.append("\\n")
        
        # Data rows
        for row in table[1:]:
            if any(cell for cell in row):  # Skip empty rows
                formatted.append(" | ".join(str(cell) if cell else "" for cell in row))
                formatted.append("\\n")
        
        formatted.append("[TABLE_END]\\n\\n")
        return ''.join(formatted)
    
    def _analyze_page_structure(self, blocks: List[Dict], page_num: int) -> Dict:
        """Analyze page structure for headers and sections"""
        structure = {
            'page': page_num,
            'headers': [],
            'sections': []
        }
        
        for block in blocks:
            if 'lines' in block:
                for line in block['lines']:
                    for span in line.get('spans', []):
                        text = span.get('text', '').strip()
                        if text and self._is_likely_header(span, text):
                            structure['headers'].append({
                                'text': text,
                                'font_size': span.get('size', 0),
                                'level': self._estimate_header_level(span)
                            })
        
        return structure
    
    def _is_likely_header(self, span: Dict, text: str) -> bool:
        """Determine if text span is likely a header"""
        font_size = span.get('size', 0)
        flags = span.get('flags', 0)
        
        # Check for bold text (bit 4 in flags)
        is_bold = bool(flags & 16)
        
        # Check for larger font size
        is_large = font_size > 12
        
        # Check for header patterns
        import re
        header_patterns = [
            r'^\\d+\\.\\d*\\s+[A-Z]',  # "1.1 SECTION"
            r'^[A-Z][A-Z\\s]{5,}$',     # "ALL CAPS HEADER"
            r'^SECTION\\s+\\d+',        # "SECTION 1"
            r'^\\d+\\s+[A-Z]'          # "1 HEADER"
        ]
        
        matches_pattern = any(re.match(pattern, text) for pattern in header_patterns)
        
        return (is_bold and is_large) or matches_pattern
    
    def _estimate_header_level(self, span: Dict) -> int:
        """Estimate header level based on font size and formatting"""
        font_size = span.get('size', 12)
        
        if font_size >= 16:
            return 1
        elif font_size >= 14:
            return 2
        elif font_size >= 12:
            return 3
        else:
            return 4
    
    def _process_chunk(self, pdf_path: str, start_page: int, end_page: int) -> ChunkMetrics:
        """Process a single chunk with fallback methods"""
        cache_key = self._get_cache_key(pdf_path, start_page, end_page)
        
        # Try loading from cache first
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            self.logger.debug(f"Loaded chunk {start_page}-{end_page} from cache")
            return cached_result
        
        # Try primary method (PyMuPDF if available)
        if HAS_PYMUPDF:
            try:
                result = self._extract_chunk_pymupdf(pdf_path, start_page, end_page)
                if result.text.strip():  # Success
                    self._save_to_cache(cache_key, result)
                    return result
            except Exception as e:
                self.logger.warning(f"PyMuPDF failed for chunk {start_page}-{end_page}: {e}")
        
        # Fallback to pdfplumber
        if HAS_PDFPLUMBER:
            try:
                result = self._extract_chunk_pdfplumber(pdf_path, start_page, end_page)
                self._save_to_cache(cache_key, result)
                return result
            except Exception as e:
                self.logger.error(f"pdfplumber failed for chunk {start_page}-{end_page}: {e}")
        
        # Return empty result if all methods fail
        return ChunkMetrics(
            pages=end_page - start_page,
            text="",
            tables=[],
            structure={},
            errors=[f"All extraction methods failed for pages {start_page}-{end_page}"],
            method_used='none'
        )
    
    def process_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> ProcessingMetrics:
        """
        Process entire PDF with chunked parallel processing
        
        Args:
            pdf_path: Path to PDF file
            output_path: Optional output file path
            
        Returns:
            ProcessingMetrics with processing statistics
        """
        self.logger.info(f"Starting PDF processing: {pdf_path}")
        
        # Get PDF info
        pdf_info = self.get_pdf_info(pdf_path)
        total_pages = pdf_info['total_pages']
        
        if total_pages == 0:
            raise ValueError("Could not determine PDF page count")
        
        # Initialize metrics
        metrics = ProcessingMetrics(
            total_pages=total_pages,
            start_time=time.time()
        )
        
        # Calculate chunk ranges
        chunks = []
        for start in range(0, total_pages, self.chunk_size - self.overlap_size):
            end = min(start + self.chunk_size, total_pages)
            chunks.append((start, end))
        
        self.logger.info(f"Processing {total_pages} pages in {len(chunks)} chunks")
        
        # Process chunks in parallel
        all_text_parts = []
        all_tables = []
        all_structure = {'sections': [], 'pages': []}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunk processing jobs
            future_to_chunk = {
                executor.submit(self._process_chunk, pdf_path, start, end): (start, end)
                for start, end in chunks
            }
            
            # Collect results as they complete
            chunk_results = {}
            for future in as_completed(future_to_chunk):
                start, end = future_to_chunk[future]
                try:
                    chunk_metrics = future.result()
                    chunk_results[start] = chunk_metrics
                    metrics.add_chunk_metrics(chunk_metrics)
                    
                    self.logger.info(f"Completed chunk {start}-{end} "
                                   f"({len(chunk_metrics.text)} chars, "
                                   f"{len(chunk_metrics.tables)} tables, "
                                   f"method: {chunk_metrics.method_used})")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process chunk {start}-{end}: {e}")
                    metrics.errors += 1
        
        # Combine results in order
        for start, _ in sorted(chunks):
            if start in chunk_results:
                chunk = chunk_results[start]
                all_text_parts.append(chunk.text)
                all_tables.extend(chunk.tables)
                all_structure['pages'].extend(chunk.structure.get('pages', []))
        
        # Create final output
        final_text = ''.join(all_text_parts)
        
        # Add document metadata header
        metadata_header = self._create_metadata_header(pdf_info, metrics)
        final_output = metadata_header + final_text
        
        # Save to file if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_output)
                
            self.logger.info(f"Saved processed text to: {output_path}")
            
            # Save structured data
            json_output = output_file.with_suffix('.json')
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': pdf_info,
                    'processing_metrics': {
                        'pages_processed': metrics.pages_processed,
                        'text_length': len(final_text),
                        'tables_found': metrics.tables_found,
                        'errors': metrics.errors,
                        'processing_time': metrics.processing_time
                    },
                    'tables': all_tables,
                    'structure': all_structure
                }, f, indent=2)
        
        metrics.processing_time = time.time() - metrics.start_time
        
        self.logger.info(f"PDF processing completed in {metrics.processing_time:.2f}s")
        self.logger.info(f"Processed {metrics.pages_processed}/{metrics.total_pages} pages")
        self.logger.info(f"Extracted {len(final_text)} characters, {metrics.tables_found} tables")
        
        return metrics
    
    def _create_metadata_header(self, pdf_info: Dict, metrics: ProcessingMetrics) -> str:
        """Create metadata header for processed document"""
        header_parts = [
            "[DOCUMENT_START]",
            f"[TITLE: {pdf_info.get('title', 'Unknown')}]",
            f"[AUTHOR: {pdf_info.get('author', 'Unknown')}]",
            f"[TOTAL_PAGES: {pdf_info['total_pages']}]",
            f"[FILE_SIZE: {pdf_info['file_size']} bytes]",
            f"[PROCESSED_PAGES: {metrics.pages_processed}]",
            f"[TABLES_FOUND: {metrics.tables_found}]",
            "[DOCUMENT_CONTENT_START]",
            ""
        ]
        return "\\n".join(header_parts) + "\\n"


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDF to text for GraphRAG")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("-o", "--output", help="Output text file path")
    parser.add_argument("--chunk-size", type=int, default=100, help="Pages per chunk")
    parser.add_argument("--overlap", type=int, default=10, help="Page overlap between chunks")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--cache-dir", help="Cache directory for processed chunks")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create processor
    processor = PDFProcessor(
        chunk_size=args.chunk_size,
        overlap_size=args.overlap,
        max_workers=args.workers,
        cache_dir=args.cache_dir
    )
    
    # Process PDF
    try:
        metrics = processor.process_pdf(args.pdf_path, args.output)
        print(f"\\nProcessing completed successfully!")
        print(f"Pages processed: {metrics.pages_processed}/{metrics.total_pages}")
        print(f"Text extracted: {metrics.text_extracted:,} characters")
        print(f"Tables found: {metrics.tables_found}")
        print(f"Processing time: {metrics.processing_time:.2f} seconds")
        print(f"Errors: {metrics.errors}")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())