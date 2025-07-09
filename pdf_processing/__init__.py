"""
PDF Processing Pipeline for GraphRAG

A comprehensive PDF to text extraction system optimized for large technical documents
and GraphRAG knowledge extraction.

Main components:
- PDFProcessor: Core processing engine with chunked parallel processing
- ExtractionStrategies: Intelligent extraction methods for different document types
- TableExtractor: Advanced table detection and extraction
- QualityValidator: Comprehensive quality assessment and validation
"""

from .pdf_to_text_processor import PDFProcessor, ProcessingMetrics, ChunkMetrics
from .extraction_strategies import (
    ExtractionStrategy,
    TechnicalDocumentStrategy,
    GeneralDocumentStrategy,
    TableFocusedStrategy,
    StrategySelector,
    ExtractionResult,
    ExtractionQuality,
    PDFType,
    detect_pdf_type
)
from .table_extractor import (
    TableExtractor,
    ExtractedTable,
    TableMetadata,
    TableType
)
from .quality_validator import (
    QualityValidator,
    QualityMetrics,
    QualityIssue,
    ValidationReport,
    ValidationLevel,
    IssueType
)

__version__ = "1.0.0"
__author__ = "GraphRAG PDF Processing Team"
__email__ = "support@graphrag.com"

__all__ = [
    # Core processor
    "PDFProcessor",
    "ProcessingMetrics", 
    "ChunkMetrics",
    
    # Extraction strategies
    "ExtractionStrategy",
    "TechnicalDocumentStrategy",
    "GeneralDocumentStrategy", 
    "TableFocusedStrategy",
    "StrategySelector",
    "ExtractionResult",
    "ExtractionQuality",
    "PDFType",
    "detect_pdf_type",
    
    # Table extraction
    "TableExtractor",
    "ExtractedTable",
    "TableMetadata",
    "TableType",
    
    # Quality validation
    "QualityValidator",
    "QualityMetrics",
    "QualityIssue", 
    "ValidationReport",
    "ValidationLevel",
    "IssueType",
]

def get_version():
    """Get the current version of the PDF processing pipeline."""
    return __version__

def get_supported_formats():
    """Get list of supported input formats."""
    return [".pdf"]

def get_available_strategies():
    """Get list of available extraction strategies."""
    return [
        "TechnicalDocumentStrategy",
        "GeneralDocumentStrategy",
        "TableFocusedStrategy"
    ]

def get_available_extractors():
    """Get list of available PDF extraction libraries."""
    available = []
    
    try:
        import fitz
        available.append("PyMuPDF")
    except ImportError:
        pass
    
    try:
        import pdfplumber
        available.append("pdfplumber")
    except ImportError:
        pass
    
    try:
        import camelot
        available.append("Camelot")
    except ImportError:
        pass
    
    return available

def quick_extract(pdf_path, output_path=None, **kwargs):
    """
    Quick extraction function for simple use cases.
    
    Args:
        pdf_path: Path to PDF file
        output_path: Optional output path
        **kwargs: Additional arguments passed to PDFProcessor
    
    Returns:
        ProcessingMetrics: Processing statistics
    """
    processor = PDFProcessor(**kwargs)
    return processor.process_pdf(pdf_path, output_path)

def validate_extraction(text_file, pdf_file, level="normal"):
    """
    Quick validation function.
    
    Args:
        text_file: Path to extracted text file
        pdf_file: Path to original PDF file
        level: Validation level (strict, normal, lenient)
    
    Returns:
        ValidationReport: Quality assessment report
    """
    with open(text_file, 'r', encoding='utf-8') as f:
        extracted_text = f.read()
    
    validator = QualityValidator(ValidationLevel(level))
    return validator.validate_extraction(
        extracted_text=extracted_text,
        pdf_path=pdf_file,
        processing_metrics={}
    )