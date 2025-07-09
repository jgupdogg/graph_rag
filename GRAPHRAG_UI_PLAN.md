# GraphRAG UI Plan: Building a Job Execution Interface

## Overview

This document outlines the plan for building a basic UI that allows users to configure and execute GraphRAG jobs with custom settings, including document selection, prompt customization, model configuration, and output management.

## UI Framework Choice: Streamlit

### Why Streamlit?
- **Rapid Development**: Built specifically for data/ML applications
- **Built-in Features**: File upload, forms, progress bars, caching
- **Pure Python**: No need for separate frontend/backend
- **State Management**: Handles session state automatically
- **Deployment Ready**: Easy to deploy with Streamlit Cloud or Docker

## Core UI Components

### 1. Main Application Structure (`app.py`)

The application will have three main tabs:
- **Job Configuration**: Set up all parameters for the GraphRAG run
- **Job Execution & Monitoring**: Submit jobs and monitor progress
- **Results Viewer**: View completed jobs and access outputs

### 2. Job Configuration Form

#### File Upload Section
- Drag-and-drop file upload area
- Support for multiple file types (PDF, TXT, DOC)
- File preview with metadata (name, size, type)
- Option to remove uploaded files
- Total size calculation and warnings for large uploads

#### Model Configuration
```
Model Provider: [OpenAI | Azure OpenAI | Anthropic | Local]
Model Selection: [GPT-4 | GPT-3.5-Turbo | Claude-3 | Custom]
API Key: [****************] (masked input)
Temperature: [0.0 - 1.0 slider]
Max Tokens: [input field]
```

#### Processing Options
```
Chunking Strategy:
  - Chunk Size: [300-2000 tokens slider]
  - Overlap Size: [50-500 tokens slider]
  - Strategy: [Token-based | Sentence-based]

Entity Extraction:
  - Default Types: ☑ Organization ☑ Person ☑ Location ☑ Event
  - Custom Types: [+ Add custom entity type]
  - Max Gleanings: [1-3 slider]

Community Detection:
  - Max Cluster Size: [5-20 slider]
  - Algorithm: [Hierarchical Leiden | Louvain]
```

#### Prompt Selection
```
Prompt Template: [Default | Technical | Legal | Medical | Custom]
[Text area for viewing/editing selected prompt]
[Save as Template] button
```

#### Output Configuration
```
Knowledge Base Name: [input field]
Output Directory: [path selector]
Storage Type: [Local Files | Azure Blob | S3]
Generate Reports: ☑ Yes ☐ No
```

### 3. Job Execution Interface

#### Pre-submission
- Configuration summary card
- Cost estimation display
- Warning for large files or expensive operations
- [Validate Configuration] button

#### During Execution
- Real-time progress bar with stages:
  - File Processing (10%)
  - Text Chunking (20%)
  - Entity Extraction (60%)
  - Graph Construction (80%)
  - Report Generation (90%)
  - Finalization (100%)
- Live log viewer (collapsible)
- Current operation display
- Elapsed time and ETA
- [Pause] [Cancel] buttons

#### Post-execution
- Job summary with statistics
- Total cost breakdown
- Output file locations
- [View Results] [Download Outputs] buttons

### 4. Configuration Management (`config_manager.py`)

```python
class ConfigurationManager:
    """Handles saving, loading, and generating configurations"""
    
    def save_config(self, config_dict, name):
        """Save configuration as reusable template"""
        
    def load_config(self, name):
        """Load saved configuration"""
        
    def generate_yaml(self, ui_inputs):
        """Convert UI inputs to GraphRAG settings.yaml"""
        
    def validate_config(self, config):
        """Validate configuration before execution"""
        
    def get_presets(self):
        """Return available preset configurations"""
```

### 5. Job Runner (`job_runner.py`)

```python
class GraphRAGJobRunner:
    """Executes GraphRAG jobs with progress tracking"""
    
    def __init__(self, config, progress_callback):
        self.config = config
        self.progress_callback = progress_callback
        
    async def run(self):
        """Execute GraphRAG pipeline with progress updates"""
        
    def parse_progress(self, log_line):
        """Extract progress information from logs"""
        
    def estimate_cost(self):
        """Calculate estimated API costs"""
        
    def cancel(self):
        """Gracefully cancel running job"""
```

## File Structure

```
graph_rag/
├── ui/
│   ├── app.py                    # Main Streamlit application
│   ├── components/
│   │   ├── file_uploader.py     # File upload component
│   │   ├── config_form.py       # Configuration form
│   │   ├── job_monitor.py       # Job execution monitor
│   │   └── results_viewer.py    # Results display
│   ├── config_manager.py         # Configuration handling
│   ├── job_runner.py            # GraphRAG execution wrapper
│   ├── pdf_processor.py         # PDF to text conversion
│   ├── cost_estimator.py        # API cost calculations
│   └── utils.py                 # Helper functions
├── templates/                    # Preset configurations
│   ├── technical_docs.yaml
│   ├── legal_docs.yaml
│   ├── medical_records.yaml
│   └── general_docs.yaml
├── custom_prompts/              # User-defined prompts
│   └── .gitkeep
├── jobs/                        # Job history and outputs
│   └── .gitkeep
└── requirements_ui.txt          # UI-specific dependencies
```

## Key Features Implementation

### 1. Dynamic Configuration Generation

```python
def generate_config(form_data):
    """Generate GraphRAG settings from UI inputs"""
    config = {
        'models': {
            'default_chat_model': {
                'type': form_data['provider'],
                'model': form_data['model'],
                'api_key': form_data['api_key'],
                'temperature': form_data['temperature']
            }
        },
        'chunks': {
            'size': form_data['chunk_size'],
            'overlap': form_data['overlap']
        },
        'extract_graph': {
            'entity_types': form_data['entity_types'],
            'max_gleanings': form_data['max_gleanings']
        }
    }
    return yaml.dump(config)
```

### 2. PDF Processing Integration

```python
class PDFProcessor:
    """Handle PDF to text conversion"""
    
    def process_file(self, file_path):
        """Convert PDF to text with metadata preservation"""
        if file_path.endswith('.pdf'):
            return self.extract_pdf_text(file_path)
        elif file_path.endswith('.txt'):
            return self.read_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
```

### 3. Progress Tracking

```python
class ProgressTracker:
    """Track and display job progress"""
    
    def __init__(self, progress_bar, status_text):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.stages = {
            'initialization': 0.1,
            'chunking': 0.2,
            'extraction': 0.6,
            'graph_construction': 0.8,
            'reports': 0.9,
            'complete': 1.0
        }
    
    def update(self, stage, message):
        """Update progress bar and status"""
        self.progress_bar.progress(self.stages[stage])
        self.status_text.text(message)
```

### 4. Cost Estimation

```python
def estimate_cost(config, file_sizes):
    """Estimate API costs based on configuration and input size"""
    total_tokens = sum(file_sizes) * 1.5  # Rough estimate
    chunks = total_tokens / config['chunk_size']
    
    costs = {
        'gpt-4': 0.03,  # per 1K tokens
        'gpt-3.5-turbo': 0.001,
        'text-embedding-3-small': 0.00002
    }
    
    # Calculate based on model and operations
    return calculate_total_cost(chunks, config['model'], costs)
```

## UI Mockup Flow

### 1. Welcome Screen
```
Welcome to GraphRAG UI
━━━━━━━━━━━━━━━━━━━━
Build knowledge graphs from your documents

[🚀 New Job]  [📁 Load Previous]  [📚 Documentation]
```

### 2. Configuration Tab
```
Job Configuration
━━━━━━━━━━━━━━━━

📤 Upload Documents
[Drop files here or click to browse]
├── engineering_specs.pdf (116.8 MB)
└── requirements.txt (2.3 KB)

⚙️ Model Settings
Provider: [OpenAI ▼]
Model: [GPT-4 ▼]
API Key: [••••••••••••]

📊 Processing Options
[Show Advanced Settings ▼]

💾 Save Configuration As: [my_config]
[Save] [Load Preset ▼]
```

### 3. Execution Tab
```
Job Execution
━━━━━━━━━━━

📋 Configuration Summary
• Files: 2 documents (119.1 MB)
• Model: GPT-4
• Chunks: 800 tokens with 200 overlap
• Estimated Cost: $75-100
• Estimated Time: 2-3 hours

[▶️ Start Processing] [💾 Save Config]

Progress
━━━━━━━━━━━━━━━━━━━━━━━━ 45%
Currently: Extracting entities from chunk 234/520

📜 Logs (click to expand)
```

### 4. Results Tab
```
Job Results
━━━━━━━━━

✅ Job completed successfully!
Duration: 2h 34m
Total Cost: $82.50

📊 Statistics
• Entities Found: 1,247
• Relationships: 3,892
• Communities: 47
• Reports Generated: 47

📥 Outputs
[Download All] [View in Explorer]

🔍 Query Your Knowledge Graph
[Open Query Interface]
```

## Implementation Steps

### Phase 1: Basic Structure (Week 1)
1. Set up Streamlit app with navigation
2. Create basic file upload functionality
3. Implement configuration form UI
4. Set up project structure

### Phase 2: Core Functionality (Week 2)
1. Implement configuration management
2. Create job runner with mock execution
3. Add progress tracking
4. Integrate PDF processing

### Phase 3: Integration (Week 3)
1. Connect to actual GraphRAG pipeline
2. Implement real progress parsing
3. Add cost estimation
4. Create results viewer

### Phase 4: Polish & Features (Week 4)
1. Add preset templates
2. Implement job history
3. Add error handling and recovery
4. Create documentation

## Additional Considerations

### Security
- Store API keys in environment variables or secure storage
- Validate file uploads for security
- Implement user authentication if multi-user
- Sanitize user inputs

### Performance
- Implement file size limits
- Use async processing for large jobs
- Cache configuration templates
- Stream logs efficiently

### User Experience
- Clear error messages with solutions
- Tooltips for complex settings
- Confirmation dialogs for expensive operations
- Auto-save configuration drafts

### Extensibility
- Plugin system for custom processors
- API for external integrations
- Export configurations as code
- Webhook notifications for job completion

## Deployment Options

### Local Deployment
```bash
pip install -r requirements_ui.txt
streamlit run ui/app.py
```

### Docker Deployment
```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "ui/app.py"]
```

### Cloud Deployment
- Streamlit Cloud (easiest)
- AWS EC2 with Docker
- Azure Container Instances
- Google Cloud Run

## Future Enhancements

1. **Batch Processing**: Queue multiple jobs
2. **Scheduling**: Run jobs on schedule
3. **Comparison Tool**: Compare different configurations
4. **Query Builder**: Visual query interface
5. **Visualization**: Graph visualization in UI
6. **Collaboration**: Share configurations and results
7. **Monitoring Dashboard**: Track all running jobs
8. **Cost Optimization**: Suggest cheaper configurations

This UI plan provides a comprehensive solution for making GraphRAG accessible to non-technical users while maintaining flexibility for advanced use cases.