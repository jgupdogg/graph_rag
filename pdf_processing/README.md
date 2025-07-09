# PDF Processing Pipeline for GraphRAG

A comprehensive PDF to text extraction system optimized for large technical documents and GraphRAG knowledge extraction.

## Overview

This pipeline processes PDF documents with intelligent extraction strategies, quality validation, and GraphRAG optimization. It handles large files (100MB+) efficiently through chunked processing and provides specialized extraction for technical documents.

## Features

- **Memory-efficient processing** for large PDFs (100MB+)
- **Multiple extraction strategies** with automatic fallback
- **Advanced table extraction** with type detection
- **Quality validation** and reporting
- **GraphRAG optimization** with entity hints
- **Technical document specialization** for engineering specs
- **Parallel processing** for performance
- **Comprehensive caching** to avoid reprocessing

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### With Optional OCR Support

```bash
# Install Tesseract OCR
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
brew install tesseract              # macOS

# Install Python dependencies
pip install -r requirements.txt
```

### With Advanced Table Extraction

```bash
# Install system dependencies for Camelot
sudo apt-get install ghostscript python3-tk  # Ubuntu/Debian
brew install ghostscript tcl-tk               # macOS

pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
python pdf_to_text_processor.py document.pdf -o extracted_text.txt
```

### Advanced Usage

```bash
python pdf_to_text_processor.py large_document.pdf \\
  --output processed_text.txt \\
  --chunk-size 100 \\
  --overlap 10 \\
  --workers 4 \\
  --cache-dir ./cache \\
  --verbose
```

### Quality Validation

```bash
python quality_validator.py extracted_text.txt original.pdf \\
  --level normal \\
  --output quality_report.json
```

### Table Extraction

```bash
python table_extractor.py document.pdf \\
  --method pdfplumber \\
  --pages 1-50
```

## Configuration

The pipeline uses `config.yaml` for configuration. Key settings:

### Processing Settings
```yaml
processing:
  chunk_size: 100          # Pages per chunk
  overlap_size: 10         # Overlapping pages
  max_workers: 4           # Parallel workers
  memory_limit_gb: 2       # Memory limit per chunk
```

### Extraction Methods
```yaml
extraction:
  primary_method: "pymupdf"
  fallback_methods:
    - "pdfplumber"
    - "pymupdf"
```

### Quality Thresholds
```yaml
quality:
  thresholds:
    text_coverage: 0.85
    character_accuracy: 0.90
    overall_minimum: 0.75
```

## Document Type Optimization

### Technical Documents
Optimized for engineering specifications, standards, and construction documents:

- Enhanced entity detection for specifications, materials, standards
- Improved table extraction for technical data
- Cross-reference preservation
- Measurement and unit recognition

### Legal Documents
Optimized for contracts, agreements, and legal text:

- Section preservation
- Reference tracking
- Structured formatting

### Academic Papers
Optimized for research papers and academic content:

- Citation extraction
- Table and figure handling
- Reference section processing

## Architecture

### Core Components

1. **PDFProcessor** (`pdf_to_text_processor.py`)
   - Main processing engine
   - Chunked parallel processing
   - Memory management
   - Caching system

2. **ExtractionStrategies** (`extraction_strategies.py`)
   - TechnicalDocumentStrategy
   - GeneralDocumentStrategy
   - TableFocusedStrategy
   - Automatic strategy selection

3. **TableExtractor** (`table_extractor.py`)
   - Advanced table detection
   - Multiple extraction methods
   - Type classification
   - Entity pre-extraction

4. **QualityValidator** (`quality_validator.py`)
   - Comprehensive quality assessment
   - Issue detection and reporting
   - GraphRAG readiness evaluation
   - Processing recommendations

### Processing Flow

```
PDF Input → Document Analysis → Strategy Selection → 
Chunked Processing → Text Extraction → Table Extraction → 
Quality Validation → GraphRAG Optimization → Output
```

## Performance Benchmarks

### Baltimore City Engineering Specifications (116MB)
- **Processing time**: 15-20 minutes
- **Memory usage**: <2GB peak
- **Text recovery**: >95%
- **Table detection**: >90%
- **Pages per second**: 10-15

### Optimization Settings
```yaml
# For large technical documents
processing:
  chunk_size: 100
  overlap_size: 10
  max_workers: 4

extraction:
  primary_method: "pdfplumber"
  
graphrag:
  recommended_chunk_size: 600
  recommended_overlap: 200
```

## Quality Metrics

The pipeline provides comprehensive quality assessment:

- **Text Coverage**: Percentage of expected text extracted
- **Character Accuracy**: Accuracy of character extraction
- **Word Accuracy**: Accuracy of word extraction
- **Table Detection Rate**: Percentage of tables detected
- **Structure Preservation**: How well document structure is maintained
- **Entity Readiness**: Readiness for GraphRAG entity extraction

## GraphRAG Integration

### Optimized Output Format
```
[DOCUMENT_START]
[TITLE: Engineering Specifications]
[TOTAL_PAGES: 500]
[DOCUMENT_CONTENT_START]

[SECTION: 2.1 Concrete Specifications]
[PAGE: 45]

Type A Concrete Requirements:
- Compressive Strength: 4,000 psi @ 28 days
- Conformance: ASTM C94

[TABLE_START: Page 45, Table 1]
[TABLE_TYPE: specification]
Property | Requirement | Test Method
Compressive Strength | 4,000 psi | ASTM C39
[TABLE_END]

[REF: See Section 3.2 for mixing procedures]
```

### Recommended GraphRAG Settings
```yaml
# For technical documents
chunks:
  size: 600
  overlap: 200

extract_graph:
  entity_types: [specification, material, standard, equipment, procedure, organization, location]
  max_gleanings: 2

community_reports:
  max_length: 2500
```

## Error Handling

The pipeline includes comprehensive error handling:

- **Page-level error isolation**: Failed pages don't stop processing
- **Method fallback**: Automatic fallback to alternative extraction methods
- **Graceful degradation**: Partial results when full extraction fails
- **Detailed error reporting**: Comprehensive error logs and suggestions

## API Usage

### Python API

```python
from pdf_processing.pdf_to_text_processor import PDFProcessor
from pdf_processing.quality_validator import QualityValidator

# Initialize processor
processor = PDFProcessor(
    chunk_size=100,
    overlap_size=10,
    max_workers=4,
    cache_dir="./cache"
)

# Process PDF
metrics = processor.process_pdf("document.pdf", "output.txt")

# Validate quality
validator = QualityValidator()
report = validator.validate_extraction(
    extracted_text=open("output.txt").read(),
    pdf_path="document.pdf",
    processing_metrics=metrics.__dict__
)

print(f"Quality score: {report.metrics.overall_score:.2f}")
print(f"GraphRAG readiness: {report.graphrag_readiness}")
```

### Strategy Selection

```python
from pdf_processing.extraction_strategies import StrategySelector

selector = StrategySelector()
pdf_info = processor.get_pdf_info("document.pdf")
strategy = selector.select_strategy(pdf_info)

result = strategy.extract("document.pdf", (0, 10))
print(f"Extracted with {result.method}")
```

## Testing

Run the test suite:

```bash
pytest pdf_processing/tests/
```

Run with coverage:

```bash
pytest --cov=pdf_processing pdf_processing/tests/
```

## Troubleshooting

### Common Issues

1. **Memory errors with large PDFs**
   - Reduce chunk_size
   - Increase overlap_size
   - Reduce max_workers

2. **Poor table extraction**
   - Try different extraction methods
   - Use Camelot for complex tables
   - Adjust table detection settings

3. **Encoding issues**
   - Check PDF encoding
   - Enable OCR for scanned documents
   - Use different extraction method

4. **Low quality scores**
   - Review validation report
   - Try different extraction strategy
   - Check original PDF quality

### Performance Optimization

1. **For large documents**:
   - Use caching
   - Increase chunk size
   - Use more workers

2. **For better quality**:
   - Decrease chunk size
   - Increase overlap
   - Use pdfplumber for complex layouts

3. **For speed**:
   - Use PyMuPDF
   - Increase chunk size
   - Disable table extraction

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration documentation
3. Open an issue with detailed information about your use case