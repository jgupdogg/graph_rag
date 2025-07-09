# PDF to Text Processing Plan for GraphRAG

## Overview

This document outlines a comprehensive plan for processing PDF documents into text format optimized for GraphRAG knowledge extraction. The focus is on handling large technical documents like the Baltimore City Engineering Specifications (116MB) while preserving structure and relationships.

## Key Requirements

### Content Types to Extract
- **Plain Text**: Body content, headers, paragraphs
- **Tables**: Specifications, measurements, requirements  
- **Lists**: Numbered/bulleted items, procedures
- **Headers/Sections**: Document structure hierarchy
- **Cross-references**: "See Section X", "per Standard Y"
- **Metadata**: Page numbers, document info
- **Special Elements**: Formulas, measurements, codes

### Processing Architecture

```
PDF Input → Chunked Processing → Multi-Method Extraction → 
Structure Preservation → Quality Validation → GraphRAG-Ready Output
```

## Implementation Components

### A. Core PDF Processor (`pdf_to_text_processor.py`)

**Features:**
- Memory-efficient streaming for large files
- Multiple extraction methods with fallbacks
- Parallel processing for performance
- Structure and hierarchy preservation
- Table detection and formatting
- Cross-reference tracking

**Key Methods:**
1. **Text Extraction Pipeline**
   - Primary: PyMuPDF (fast, good for most text)
   - Fallback: pdfplumber (better for complex layouts)
   - OCR: OCRmyPDF (for scanned pages)

2. **Table Processing**
   - Camelot for structured tables
   - Custom parsing for specification tables
   - Preserve column headers and relationships

3. **Structure Preservation**
   - Section hierarchy tracking
   - Header level detection
   - Cross-reference mapping
   - Page number preservation

### B. Memory Management Strategy

For 116MB PDF:
- Process in 100-page chunks
- 10-page overlap for context
- Stream processing (never load full file)
- Parallel chunk processing (4 workers)
- Memory limit: 2GB per chunk

### C. Output Formats

#### 1. GraphRAG-Optimized Text
```
[SECTION: 2.1 Concrete Specifications]
[PAGE: 45]

Type A Concrete Requirements:
- Compressive Strength: 4,000 psi @ 28 days
- Conformance: ASTM C94
[TABLE_START]
Property | Requirement | Test Method
...
[TABLE_END]

[XREF: See Section 3.2 for mixing procedures]
```

#### 2. Structured JSON
```json
{
  "sections": [...],
  "tables": [...],
  "cross_references": [...],
  "entities_hint": [...]
}
```

### D. Quality Validation Suite

1. **Extraction Metrics**
   - Text coverage percentage
   - Table detection rate
   - Structure preservation score
   - Cross-reference accuracy

2. **Performance Metrics**
   - Processing speed (pages/second)
   - Memory usage
   - Error rate by page type

3. **GraphRAG Readiness**
   - Entity hint extraction
   - Relationship preservation
   - Chunk boundary validation

## Technical Implementation Details

### PDF Processing Libraries

1. **PyMuPDF (Primary)**
   - Fast text extraction
   - Good Unicode support
   - Handles most PDF structures

2. **pdfplumber (Complex Layouts)**
   - Better table detection
   - Precise positioning
   - Handles complex formatting

3. **Camelot (Tables)**
   - Specialized table extraction
   - Multiple parsing methods
   - CSV/DataFrame output

4. **OCRmyPDF (Scanned Content)**
   - Tesseract integration
   - Language detection
   - Image preprocessing

### Processing Pipeline

```python
class PDFProcessor:
    def process(self, pdf_path):
        # 1. Analyze PDF structure
        doc_info = self.analyze_structure(pdf_path)
        
        # 2. Determine processing strategy
        strategy = self.select_strategy(doc_info)
        
        # 3. Process in chunks
        for chunk in self.chunk_pages(pdf_path):
            # Extract text with primary method
            text = self.extract_text(chunk, strategy)
            
            # Extract tables
            tables = self.extract_tables(chunk)
            
            # Preserve structure
            structure = self.extract_structure(chunk)
            
            # Merge and validate
            yield self.merge_content(text, tables, structure)
```

### Optimization for Technical Documents

1. **Entity Pre-extraction**
   - Identify specification numbers
   - Extract measurement values
   - Detect standard references
   - Mark material names

2. **Relationship Hints**
   - "complies with" → compliance relationship
   - "per" → reference relationship
   - "requires" → dependency relationship
   - "see" → cross-reference relationship

3. **Section-Based Chunking**
   - Respect document boundaries
   - Keep related content together
   - Preserve context for specifications

## Error Handling & Recovery

1. **Graceful Degradation**
   - Multiple extraction methods
   - Page-level error isolation
   - Partial success handling

2. **Error Logging**
   - Page-specific errors
   - Method failure tracking
   - Recovery statistics

3. **Manual Review Flags**
   - Complex diagrams
   - Failed OCR pages
   - Corrupted sections

## Performance Benchmarks

For 116MB Baltimore Specs PDF:
- **Target Speed**: 10-15 pages/second
- **Memory Usage**: <2GB peak
- **Text Recovery**: >95%
- **Table Detection**: >90%
- **Processing Time**: 15-20 minutes total

## Integration with GraphRAG

1. **Pre-configured Output**
   - Optimal chunk boundaries
   - Entity hints embedded
   - Relationship markers preserved

2. **Metadata Enhancement**
   - Source page tracking
   - Section hierarchy
   - Confidence scores

3. **Configuration Recommendations**
   - Smaller chunks (600-800 tokens)
   - Higher overlap (150-200 tokens)
   - Custom entity types for technical content

## File Structure

```
pdf_processing/
├── pdf_to_text_processor.py    # Main processor
├── extraction_strategies.py     # Different extraction methods
├── table_extractor.py          # Specialized table handling
├── structure_analyzer.py       # Document structure detection
├── memory_manager.py           # Chunk processing
├── output_formatter.py         # GraphRAG formatting
├── quality_validator.py        # Validation tools
├── config.yaml                 # Processing configuration
└── tests/
    ├── test_extraction.py
    ├── test_performance.py
    └── sample_pdfs/
```

## Next Steps

1. **Install Dependencies**
   ```bash
   pip install PyMuPDF pdfplumber camelot-py[cv] ocrmypdf
   ```

2. **Implement Core Processor**
   - Create main processing class
   - Add chunking strategy
   - Implement extraction methods

3. **Add Table Extraction**
   - Integrate Camelot
   - Custom table parsing
   - Format preservation

4. **Create Validation Suite**
   - Quality metrics
   - Performance testing
   - GraphRAG compatibility check

5. **Test with Baltimore Specs**
   - Process sample sections
   - Validate output quality
   - Optimize for technical content

This comprehensive approach ensures maximum information extraction from complex technical PDFs while maintaining performance and preparing optimal input for GraphRAG processing.