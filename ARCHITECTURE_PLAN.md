# GraphRAG Multi-Document Architecture Evolution Plan

## Project Overview

Transform the current single-document GraphRAG system into a multi-document knowledge base explorer with dynamic document management, graph merging, and enhanced UI capabilities.

### Current State Analysis
- Single document processing (Baltimore specs)
- Static workspace (`graphrag/` directory)
- Manual pipeline execution
- Basic Streamlit UI with graph visualization
- PyVis-based interactive graphs

### Target Architecture
- Multi-document processing with isolated workspaces
- Dynamic document upload and management
- Graph merging across documents
- Enhanced UI with document selection
- Persistent metadata tracking

## Phase 1: Database & Architecture Setup (Foundation)

### 1.1 Database Infrastructure
**File**: `app_logic.py`

**Database Schema**:
```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    status TEXT NOT NULL,
    workspace_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_size INTEGER,
    processing_time INTEGER
);
```

**Status Enum**:
- `UPLOADED` - File uploaded, workspace created
- `PROCESSING` - GraphRAG pipeline running
- `COMPLETED` - Successfully processed
- `ERROR` - Processing failed

**Core Functions**:
- `init_db()` - Initialize database and workspaces directory
- `get_processed_documents()` - Fetch completed documents
- `update_document_status(doc_id, status)` - Update processing status
- `delete_document(doc_id)` - Remove document and workspace

### 1.2 Multi-Workspace Architecture

**Directory Structure**:
```
graph_rag/
├── app_logic.py              # NEW: Backend logic
├── config_manager.py         # NEW: Config management
├── query_engine.py          # NEW: Search engine
├── streamlit_app.py         # MODIFIED: Enhanced UI
├── pyvis_graph.py          # MODIFIED: Multi-doc viz
├── graphrag_config/        # MOVED: Base configs
│   ├── prompts/
│   └── settings.yaml
├── workspaces/             # NEW: Document workspaces
│   ├── {doc_id_1}/
│   │   ├── input/
│   │   ├── output/
│   │   ├── settings.yaml
│   │   └── prompts/
│   └── {doc_id_2}/
├── metadata.db             # NEW: Document database
├── ARCHITECTURE_PLAN.md    # NEW: This file
└── venv/
```

**Workspace Isolation**:
- Each document gets unique UUID-based workspace
- Independent GraphRAG processing environments
- Isolated caching and output storage
- Copy base configuration to each workspace

### 1.3 Configuration Management
**File**: `config_manager.py`

**Features**:
- Template-based workspace initialization
- Dynamic settings.yaml generation
- Environment variable management
- Workspace cleanup utilities

## Phase 2: Document Processing Pipeline

### 2.1 Document Ingestion Pipeline
**Function**: `process_new_document(uploaded_file)`

**Flow**:
1. Generate unique document ID
2. Create workspace directory structure
3. Save uploaded file to workspace/input/
4. Extract text from PDF (integrate existing logic)
5. Update database status to PROCESSING
6. Trigger GraphRAG indexing
7. Update status to COMPLETED or ERROR

### 2.2 Asynchronous Processing
**Features**:
- Background processing with threading
- Progress tracking and status updates
- Error handling and retry mechanisms
- Processing queue management

### 2.3 Text Extraction Enhancement
**Integrate**: `extract_baltimore_specs.py` logic

**Improvements**:
- Generic PDF text extraction
- Configurable section extraction
- Text cleaning and preprocessing
- Multiple file format support

## Phase 3: Enhanced UI & Document Selection

### 3.1 Streamlit Interface Redesign
**File**: `streamlit_app.py`

**Layout**:
```
Sidebar:
├── Upload Section
│   ├── File uploader
│   ├── Processing status
│   └── Upload button
├── Document Selection
│   ├── Multi-select widget
│   ├── Document metadata
│   └── Delete/manage options
└── Graph Controls
    ├── Filter options
    ├── Layout settings
    └── Export controls

Main Area:
├── Graph Visualization (70%)
│   ├── Interactive PyVis graph
│   ├── Statistics panel
│   └── Node/edge details
└── Query Interface (30%)
    ├── Search input
    ├── Results display
    └── Query history
```

### 3.2 Document Management Features
- List all documents with metadata
- Delete/archive functionality
- Rename documents
- View processing logs
- Download processed data

### 3.3 Real-time Updates
- Auto-refresh document list
- Live processing status
- Progress indicators
- Error notifications

## Phase 4: Graph Merging & Visualization

### 4.1 Graph Merging Logic
**Function**: `load_and_merge_graphs(selected_doc_ids)`

**Process**:
1. Load entities/relationships from selected workspaces
2. Deduplicate entities by name/type matching
3. Merge relationship data with source tracking
4. Add document source metadata to all elements
5. Generate combined NetworkX graph

**Entity Deduplication**:
- Match by name (case-insensitive)
- Consider entity type
- Merge descriptions and attributes
- Track all source documents

### 4.2 Enhanced Visualization
**File**: `pyvis_graph.py`

**Multi-document Features**:
- Color-code nodes by source document
- Document badges in node popups
- Cross-document relationship highlighting
- Interactive legend with document colors
- Filter by source document

**Visual Enhancements**:
- Improved node sizing algorithm
- Better edge bundling
- Hierarchical layouts for large graphs
- Zoom and pan controls

### 4.3 Cross-Document Analysis
**Features**:
- Bridge entity identification
- Cross-document pattern detection
- Document similarity metrics
- Relationship strength analysis

## Phase 5: Query & Search Enhancement

### 5.1 Multi-Document Query Engine
**File**: `query_engine.py`

**Features**:
- Route queries to appropriate workspaces
- Aggregate results from multiple documents
- Rank results by relevance and source
- Support for global and local search modes

**Search Types**:
- **Global Search**: Themes across all selected documents
- **Local Search**: Specific entities in selected documents
- **Cross-Document Search**: Find connections between documents
- **Entity Search**: Find specific entities across documents

### 5.2 Advanced Search UI
**Features**:
- Document-specific vs. global search toggle
- Filter results by source document
- Export query results
- Search history and bookmarks

## Phase 6: Additional Features

### 6.1 Document Comparison
**Features**:
- Side-by-side graph views
- Highlight differences/similarities
- Generate comparison reports
- Entity overlap analysis

### 6.2 Export & Sharing
**Features**:
- Export merged graphs (various formats)
- Save analysis sessions
- Generate shareable links
- PDF report generation

### 6.3 Performance Optimization
**Features**:
- Caching for processed graphs
- Lazy loading for large datasets
- Background graph pre-computation
- Memory usage optimization

## Implementation Timeline

### Week 1-2: Foundation
- [x] Create architecture plan
- [ ] Database setup (app_logic.py)
- [ ] Multi-workspace architecture
- [ ] Basic file upload UI

### Week 3-4: Processing Pipeline
- [ ] Document processing integration
- [ ] Status tracking system
- [ ] Error handling and recovery
- [ ] Background processing

### Week 5-6: Multi-Document Features
- [ ] Graph merging logic
- [ ] Enhanced visualization
- [ ] Document selection UI
- [ ] Cross-document analysis

### Week 7-8: Advanced Features
- [ ] Query enhancement
- [ ] Comparison tools
- [ ] Export functionality
- [ ] Performance optimization

## Technical Specifications

### Database Schema (SQLite)
```sql
-- Documents table
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('UPLOADED', 'PROCESSING', 'COMPLETED', 'ERROR')),
    workspace_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_size INTEGER,
    processing_time INTEGER,
    error_message TEXT
);

-- Processing logs table
CREATE TABLE processing_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    message TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level TEXT DEFAULT 'INFO',
    FOREIGN KEY (document_id) REFERENCES documents(id)
);
```

### API Design (app_logic.py)
```python
# Core functions
def init_db() -> None
def get_processed_documents() -> List[Dict]
def process_new_document(uploaded_file) -> str
def load_and_merge_graphs(selected_doc_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]
def delete_document(doc_id: str) -> bool
def get_document_status(doc_id: str) -> str
def update_document_status(doc_id: str, status: str, error_message: str = None) -> None
```

### Configuration Schema (settings.yaml)
```yaml
# Base configuration template
encoding: utf-8
skip_workflows: []
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  model: gpt-3.5-turbo
  model_supports_json: true
  max_tokens: 4000
  temperature: 0

parallelization:
  stagger: 0.3
  num_threads: 50

async_mode: threaded

embeddings:
  async_mode: threaded
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: text-embedding-3-small
    max_tokens: 8191

input:
  type: file
  file_type: text
  base_dir: "input"
  file_encoding: utf-8
  file_pattern: ".*\\.txt$"

cache:
  type: file
  base_dir: "cache"

storage:
  type: file
  base_dir: "output"

reporting:
  type: file
  base_dir: "output"

entity_extraction:
  prompt: "prompts/entity_extraction.txt"
  entity_types: [organization, person, geo, event]
  max_gleanings: 0

summarize_descriptions:
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

claim_extraction:
  prompt: "prompts/claim_extraction.txt"
  description: "Any claims or facts that could be relevant to information discovery."
  max_gleanings: 0

community_reports:
  prompt: "prompts/community_report.txt"
  max_length: 2000
  max_input_length: 8000

group_by_columns: [id, short_id]

embed_graph:
  enabled: false

umap:
  enabled: false

snapshots:
  graphml: false
  raw_entities: false
  top_level_nodes: false

local_search:
  text_unit_prop: 0.5
  community_prop: 0.1
  conversation_history_max_turns: 5
  top_k_mapped_entities: 10
  top_k_relationships: 10
  max_tokens: 12000

global_search:
  max_tokens: 12000
  data_max_tokens: 12000
  map_max_tokens: 1000
  reduce_max_tokens: 2000
  concurrency: 32
```

## Success Metrics

### Functional Metrics
- Support for 10+ simultaneous documents
- Graph merging within 30 seconds
- File upload processing under 5 minutes
- Cross-document query response under 10 seconds

### User Experience Metrics
- Intuitive document upload process
- Clear processing status feedback
- Responsive graph interactions
- Effective search and filtering

### Technical Metrics
- Database query performance
- Memory usage optimization
- Error recovery mechanisms
- Scalability to 100+ documents

## Risk Mitigation

### Technical Risks
- **Large graph performance**: Implement pagination and lazy loading
- **Memory usage**: Add garbage collection and caching strategies
- **Processing failures**: Robust error handling and retry mechanisms
- **Database corruption**: Regular backups and transaction safety

### User Experience Risks
- **Complex UI**: Progressive disclosure and user testing
- **Processing delays**: Clear progress indicators and background processing
- **Data loss**: Automatic saving and recovery mechanisms

## Future Enhancements

### Advanced Analytics
- Document similarity analysis
- Topic modeling across documents
- Timeline visualization
- Sentiment analysis integration

### Collaboration Features
- Multi-user support
- Shared workspaces
- Comment and annotation system
- Version control for documents

### Integration Capabilities
- API for external tools
- Export to other graph databases
- Integration with document management systems
- Real-time document monitoring

This plan provides a comprehensive roadmap for evolving your GraphRAG system into a powerful multi-document analysis platform while maintaining the strengths of your current implementation.