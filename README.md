# Baltimore City GraphRAG Explorer

A production-ready interactive knowledge graph system for exploring multiple documents using Microsoft GraphRAG with PyVis visualization and enhanced document management.

## Overview

This project combines Microsoft GraphRAG's powerful knowledge extraction capabilities with an interactive PyVis-based web interface to explore relationships and entities within Baltimore City's engineering specifications. The system processes PDF documents, extracts structured knowledge graphs, and provides an intuitive web interface for exploration.

## Features

- **✅ Multi-Document Processing**: Upload and process multiple PDF/TXT files simultaneously
- **✅ Real PDF Text Extraction**: Robust PDF text extraction using PyPDF2 with fallback libraries
- **✅ Interactive PyVis Visualization**: Dynamic network graphs with document color-coding and filtering
- **✅ Enhanced Document Management**: Comprehensive upload, processing, and error recovery system
- **✅ Graph Merging & Source Tracking**: Combine multiple documents with source attribution
- **✅ Error Recovery & Retry**: One-click retry functionality for failed document processing
- **✅ Real-time Processing Status**: Live updates on document processing progress
- **✅ Advanced Filtering**: Filter by entity type, degree range, search terms, and source document
- **✅ Processing Logs & Debugging**: Detailed step-by-step processing logs with status indicators
- **✅ Background Processing**: Non-blocking document processing with status updates
- **✅ Global & Local Search**: High-level insights and specific entity queries across documents
- **✅ Performance Optimization**: Handles large multi-document graphs efficiently
- **✅ Document Chat Interface**: Interactive chat functionality with GraphRAG-powered responses
- **✅ Multi-Search Methods**: Support for local, global, drift, and basic search types
- **✅ Chat History**: Conversation history with source attribution and method tracking

## Project Structure

```
graph_rag/
├── streamlit_app.py         # Main Streamlit web interface
├── pyvis_graph.py          # PyVis network creation and rendering
├── query_logic.py          # GraphRAG query engine and chat functionality
├── app_logic.py            # Document processing and management logic
├── config_manager.py       # Configuration management
├── extract_baltimore_specs.py  # PDF processing utility
├── run_pipeline.py         # GraphRAG pipeline runner
├── run_streamlit.sh        # Streamlit startup script
├── graphrag/               # GraphRAG workspace
│   ├── input/             # Input documents (gitignored)
│   ├── output/            # Generated knowledge graphs
│   ├── prompts/           # System prompts for operations
│   ├── settings.yaml      # Configuration file
│   └── .env              # Environment variables (API keys)
├── venv/                  # Python virtual environment
├── README.md
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.10-3.12
- OpenAI API key or Azure OpenAI access

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jgupdogg/graph_rag.git
   cd graph_rag
   ```

2. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install graphrag pyvis streamlit PyPDF2
   ```

4. **Configure API key**:
   Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GRAPHRAG_API_KEY=your_openai_api_key_here
   ```

## Usage

### Running the Interactive Visualization

Start the Streamlit web interface:

```bash
./run_streamlit.sh
# or
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` to access the interactive graph explorer.

### Processing Documents

1. **Extract PDF content**:
   ```bash
   python extract_baltimore_specs.py
   ```

2. **Run GraphRAG pipeline**:
   ```bash
   python run_pipeline.py
   # or
   graphrag index --root ./graphrag
   ```

3. **Launch visualization**:
   ```bash
   streamlit run streamlit_app.py
   ```

### Interactive Features

The PyVis visualization provides:

- **Node Interaction**: Click nodes to view detailed popups with entity information
- **Filtering Controls**: 
  - Entity type selection (Person, Organization, Location, etc.)
  - Degree range slider for connection filtering
  - Search functionality for specific entities
- **Layout Options**: Choose from Barnes-Hut, Force Atlas 2, or Hierarchical layouts
- **Performance Mode**: Automatically optimizes display for large graphs (100+ nodes)
- **Selection Highlighting**: Click nodes to highlight connections and neighbors

### Document Chat Interface

The interactive chat interface provides GraphRAG-powered querying capabilities:

- **Multiple Search Methods**: 
  - **Local Search**: Find specific details and direct answers from documents
  - **Global Search**: Discover high-level themes and patterns across documents
  - **Drift Search**: Contextual search that follows semantic relationships
  - **Basic Search**: Simple keyword-based search functionality

- **Multi-Document Querying**: Ask questions across all selected documents simultaneously
- **Chat History**: View conversation history with timestamps and source attribution
- **Source Attribution**: See which documents contributed to each response
- **Error Handling**: Robust error handling with helpful error messages

#### Chat Usage Examples

**Local Search** (specific details):
- "What are the engineering requirements for water systems?"
- "Who are the key contractors mentioned?"
- "What are the compliance standards for this project?"

**Global Search** (themes and patterns):
- "What are the main themes across all documents?"
- "What are the common requirements mentioned?"
- "What are the key priorities in these documents?"

### Command Line Queries

#### Global Search
For high-level questions about themes and patterns:

```bash
graphrag query \
  --root ./graphrag \
  --method global \
  --query "What are the main procurement requirements in Baltimore specifications?"
```

#### Local Search
For specific questions about entities and relationships:

```bash
graphrag query \
  --root ./graphrag \
  --method local \
  --query "What are the engineering standards for water systems?"
```

## Configuration

The main configuration is in `graphrag/settings.yaml`. Key settings include:

- **Models**: Configure OpenAI or Azure OpenAI models
- **Chunking**: Text chunk size and overlap settings
- **Entity Extraction**: Types of entities to extract from city specifications
- **Community Detection**: Graph clustering parameters
- **Search Settings**: Prompts and parameters for different search types

## Adding Your Own Documents

1. Place your PDF or text files in `graphrag/input/`
2. Update the configuration in `graphrag/settings.yaml` if needed
3. Run the indexing pipeline: `graphrag index --root ./graphrag`
4. Launch the Streamlit interface to explore your data

## Example Queries

### Global Search Examples
- "What are the main themes in Baltimore city specifications?"
- "What are the key procurement requirements across all documents?"
- "Summarize the engineering standards and compliance requirements"

### Local Search Examples
- "What are the water system engineering requirements?"
- "Who are the key contractors mentioned in the specifications?"
- "What are the environmental compliance standards?"

## Visualization Controls

### Sidebar Controls
- **Entity Types**: Multi-select filter for different entity categories
- **Degree Range**: Slider to filter nodes by number of connections
- **Search Entities**: Text input to find specific entities
- **Layout Algorithm**: Choose visualization layout algorithm
- **Graph Optimization**: Toggle performance mode for large graphs

### Graph Statistics
- Total entities and relationships count
- Graph density and connectivity metrics
- Connected components analysis

## Troubleshooting

1. **API Key Issues**: Ensure your OpenAI API key is correctly set in `graphrag/.env`
2. **Rate Limits**: Adjust `tokens_per_minute` and `requests_per_minute` in `settings.yaml`
3. **Memory Issues**: Reduce `concurrent_requests` in the model configuration
4. **Large Files**: Adjust chunk size and overlap in the `chunks` section
5. **Visualization Loading**: Try enabling graph optimization for better performance

## Resources

- [Microsoft GraphRAG Documentation](https://microsoft.github.io/graphrag/)
- [PyVis Documentation](https://pyvis.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Configuration Reference](https://microsoft.github.io/graphrag/config/yaml/)

## License

This project uses the Microsoft GraphRAG library. Please refer to the original GraphRAG repository for licensing information.