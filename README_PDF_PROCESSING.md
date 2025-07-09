# GraphRAG PDF Processing Pipeline

This repository contains a complete pipeline for processing PDF documents with Microsoft's GraphRAG to build knowledge graphs.

## Overview

GraphRAG (Graph Retrieval-Augmented Generation) is a powerful system for extracting structured knowledge from unstructured text. However, it doesn't natively support PDF files. This pipeline bridges that gap by providing:

1. **PDF Processing**: Extract text from PDFs while preserving structure and metadata
2. **Custom Loader**: Integrate PDF processing directly into GraphRAG's pipeline
3. **Visualization**: Tools to visualize and analyze the resulting knowledge graphs
4. **Complete Pipeline**: End-to-end solution from PDFs to queryable knowledge graph

## Components

### 1. `pdf_processor.py`
Handles PDF text extraction with two methods:
- **PyPDF2**: Basic text extraction
- **pdfplumber**: Advanced extraction with table support

Features:
- Metadata preservation
- Multi-page handling
- Table extraction
- Text cleaning
- Batch processing

### 2. `graphrag_pdf_loader.py`
Custom GraphRAG input loader that:
- Processes PDFs alongside text files
- Integrates seamlessly with GraphRAG pipeline
- Preserves document structure
- Handles mixed document types

### 3. `run_graphrag_with_pdfs.py`
Complete pipeline script that:
- Preprocesses PDFs
- Configures GraphRAG
- Runs indexing
- Enables querying
- Provides analysis tools

### 4. `visualize_graph.py`
Visualization utilities:
- Static graphs with matplotlib
- Interactive graphs with plotly
- Subgraph extraction
- Community detection
- Automated reporting

## Installation

1. Install GraphRAG:
```bash
pip install graphrag
```

2. Install additional dependencies:
```bash
pip install PyPDF2 pdfplumber matplotlib plotly pandas networkx
```

3. Set up OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Quick Start

1. **Prepare your documents**:
```bash
mkdir documents
# Copy your PDF files to the documents directory
cp /path/to/your/*.pdf documents/
```

2. **Run the pipeline**:
```bash
python run_graphrag_with_pdfs.py
```

This will:
- Process all PDFs in the documents directory
- Build a knowledge graph using GraphRAG
- Generate analysis and statistics
- Enable querying

### Step-by-Step Usage

#### 1. Process PDFs Only
```python
from pdf_processor import PDFProcessor

processor = PDFProcessor(output_dir="processed_documents")

# Process single PDF
result = processor.process_pdf("example.pdf")

# Process directory
results = processor.process_directory("pdfs/")
```

#### 2. Use Custom PDF Loader with GraphRAG
```python
from graphrag_pdf_loader import patch_graphrag_loaders

# Apply PDF support to GraphRAG
patch_graphrag_loaders()

# Now GraphRAG will process PDFs automatically
```

#### 3. Visualize the Knowledge Graph
```python
from visualize_graph import GraphRAGVisualizer

visualizer = GraphRAGVisualizer(output_dir="./graphrag_output")
visualizer.load_graph_data()

# Create visualizations
visualizer.visualize_matplotlib(save_path="graph.png")
visualizer.visualize_plotly(save_path="interactive.html")

# Generate report
visualizer.create_report()
```

## How It Works

### 1. PDF Processing
- PDFs are read using PyPDF2 or pdfplumber
- Text is extracted page by page
- Tables are identified and formatted
- Metadata is preserved (title, author, creation date, etc.)
- Text is cleaned and structured

### 2. Chunking
- Processed text is split into chunks (default: 1200 tokens)
- Chunks overlap to preserve context (default: 100 tokens)
- Metadata can be prepended to chunks

### 3. Entity Extraction
GraphRAG uses LLMs to extract:
- **Entities**: People, organizations, locations, events, etc.
- **Relationships**: Connections between entities with descriptions
- **Attributes**: Properties and descriptions of entities

### 4. Graph Construction
- Entities become nodes in the graph
- Relationships become edges
- Multiple extractions are merged and deduplicated
- Descriptions are summarized using LLMs

### 5. Knowledge Graph Storage
The graph is stored as:
- `entities.parquet`: All extracted entities
- `relationships.parquet`: All relationships
- Additional indexes for efficient querying

## Configuration

### Entity Types
Default entity types extracted:
- organization
- person
- location
- event
- product
- technology
- concept
- document

Customize in the configuration:
```python
"entity_types": ["custom_type1", "custom_type2", ...]
```

### Chunking Strategy
Options:
- `tokens`: Split by token count (default)
- `sentence`: Split by sentences

### LLM Settings
Configure models for different tasks:
```python
"extract_graph": {
    "llm": {
        "model": "gpt-4",  # Entity extraction
        "temperature": 0.0
    }
},
"summarize_descriptions": {
    "llm": {
        "model": "gpt-3.5-turbo",  # Description summarization
        "temperature": 0.0
    }
}
```

## Examples

### Example 1: Process Research Papers
```python
# Process directory of research PDFs
processor = PDFProcessor()
results = processor.process_directory("research_papers/")

# Run GraphRAG
pipeline = GraphRAGPDFPipeline(input_dir="research_papers")
await pipeline.run_indexing()

# Query for insights
result = await pipeline.query_graph(
    "What are the main research topics and how are they connected?"
)
```

### Example 2: Analyze Company Documents
```python
# Process company reports
processor.process_directory("company_reports/", pattern="*Q[1-4]*.pdf")

# Build graph with focus on business entities
config = create_pdf_workflow_config()
config["extract_graph"]["entity_types"] = [
    "company", "product", "executive", "metric", "initiative"
]

# Run analysis
pipeline.run_indexing()
pipeline.analyze_graph()
```

### Example 3: Extract Specific Subgraph
```python
visualizer = GraphRAGVisualizer()
visualizer.load_graph_data()

# Get subgraph around specific entity
subgraph = visualizer.get_subgraph(
    center_node="Microsoft", 
    max_depth=2,
    max_nodes=50
)

# Visualize
visualizer.visualize_plotly(
    subgraph=subgraph,
    title="Microsoft's Network"
)
```

## Best Practices

1. **PDF Quality**: 
   - Use pdfplumber for complex layouts
   - Check extracted text quality before processing
   - Consider OCR for scanned PDFs

2. **Chunking**:
   - Larger chunks (1500-2000 tokens) for technical documents
   - Smaller chunks (500-800 tokens) for diverse content
   - Increase overlap for documents with complex references

3. **Entity Types**:
   - Define domain-specific entity types
   - Start with fewer types for better accuracy
   - Iterate based on results

4. **Performance**:
   - Process PDFs in batches
   - Use caching for LLM calls
   - Consider using GPT-3.5 for summarization

5. **Visualization**:
   - Start with subgraphs for large datasets
   - Use interactive visualizations for exploration
   - Generate reports for documentation

## Troubleshooting

### Common Issues

1. **PDF Extraction Fails**
   - Try alternative extraction method
   - Check if PDF is encrypted
   - Consider using OCR tools for scanned PDFs

2. **Memory Issues with Large Graphs**
   - Process documents in batches
   - Use subgraph extraction
   - Increase system memory

3. **Poor Entity Extraction**
   - Review and adjust prompts
   - Check chunk size and overlap
   - Ensure text quality from PDFs

4. **Slow Processing**
   - Reduce concurrent API calls
   - Use smaller models where appropriate
   - Enable caching

## Advanced Features

### Custom Entity Extraction Prompts
Modify extraction prompts for domain-specific needs:
```python
config["extract_graph"]["extraction_prompt"] = """
Extract medical entities including:
- Diseases and conditions
- Treatments and medications
- Medical procedures
- Healthcare providers
...
"""
```

### Multi-Language Support
Process PDFs in different languages:
```python
processor = PDFProcessor()
# Extract text (language-agnostic)
text, metadata = processor.extract_text_pdfplumber("document_fr.pdf")

# Configure GraphRAG for French
config["extract_graph"]["extraction_prompt"] = "Extraire les entités..."
```

### Integration with Existing Systems
```python
# Export to standard formats
import json
entities_df = pd.read_parquet("graphrag_output/entities.parquet")
entities_df.to_json("entities.json", orient="records")

# Convert to other graph formats
G = visualizer.graph
nx.write_gexf(G, "knowledge_graph.gexf")  # Gephi format
nx.write_graphml(G, "knowledge_graph.graphml")  # GraphML format
```

## Contributing

Feel free to extend this pipeline with:
- Additional PDF processing methods
- Custom entity extractors
- New visualization types
- Integration with other systems

## License

This PDF processing pipeline is provided as-is for use with GraphRAG. Ensure you comply with GraphRAG's license and OpenAI's usage policies.