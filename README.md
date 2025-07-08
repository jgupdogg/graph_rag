# GraphRAG Project

This repository contains a Microsoft GraphRAG implementation for knowledge extraction and question answering from text documents.

## Overview

GraphRAG (Graph Retrieval-Augmented Generation) is a powerful system that creates a knowledge graph from text documents and enables both global and local search capabilities. This project is set up with Charles Dickens' "A Christmas Carol" as sample data.

## Features

- **Knowledge Graph Construction**: Automatically extracts entities, relationships, and communities from text
- **Global Search**: High-level questions about themes, patterns, and overall insights
- **Local Search**: Specific questions about entities and their relationships
- **Configurable Models**: Support for OpenAI and Azure OpenAI models
- **Vector Storage**: Built-in LanceDB integration for efficient similarity search

## Project Structure

```
graph_rag/
├── venv/                    # Python virtual environment
├── ragtest/                 # GraphRAG workspace
│   ├── input/              # Input documents
│   │   └── book.txt        # Sample data (A Christmas Carol)
│   ├── prompts/            # System prompts for different operations
│   ├── settings.yaml       # Configuration file
│   └── .env               # Environment variables (API keys)
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

3. **Install dependencies** (if needed):
   ```bash
   pip install graphrag
   ```

4. **Configure API key**:
   Edit `ragtest/.env` and replace `<API_KEY>` with your actual OpenAI API key:
   ```
   GRAPHRAG_API_KEY=your_openai_api_key_here
   ```

## Usage

### Running the Indexing Pipeline

Process the input documents to create the knowledge graph:

```bash
source venv/bin/activate
graphrag index --root ./ragtest
```

This will create output files in `ragtest/output/` containing the processed knowledge graph.

### Querying the System

#### Global Search
For high-level questions about themes, patterns, and overall insights:

```bash
graphrag query \
  --root ./ragtest \
  --method global \
  --query "What are the top themes in this story?"
```

#### Local Search
For specific questions about entities and relationships:

```bash
graphrag query \
  --root ./ragtest \
  --method local \
  --query "Who is Scrooge and what are his main relationships?"
```

## Configuration

The main configuration is in `ragtest/settings.yaml`. Key settings include:

- **Models**: Configure OpenAI or Azure OpenAI models
- **Chunking**: Text chunk size and overlap settings
- **Entity Extraction**: Types of entities to extract
- **Community Detection**: Graph clustering parameters
- **Search Settings**: Prompts and parameters for different search types

## Adding Your Own Data

1. Place your text files in `ragtest/input/`
2. Update the configuration in `ragtest/settings.yaml` if needed
3. Run the indexing pipeline: `graphrag index --root ./ragtest`
4. Query your data using the search commands

## Example Queries

### Global Search Examples
- "What are the main themes in the document?"
- "What are the key insights from this text?"
- "Summarize the overall narrative structure"

### Local Search Examples
- "Who is [character name] and what do they do?"
- "What is the relationship between [entity1] and [entity2]?"
- "Tell me about [specific topic] mentioned in the text"

## Troubleshooting

1. **API Key Issues**: Make sure your OpenAI API key is correctly set in `ragtest/.env`
2. **Rate Limits**: Adjust `tokens_per_minute` and `requests_per_minute` in `settings.yaml`
3. **Memory Issues**: Reduce `concurrent_requests` in the model configuration
4. **Large Files**: Adjust chunk size and overlap in the `chunks` section

## Resources

- [Microsoft GraphRAG Documentation](https://microsoft.github.io/graphrag/)
- [Configuration Reference](https://microsoft.github.io/graphrag/config/yaml/)
- [GraphRAG GitHub Repository](https://github.com/microsoft/graphrag)

## License

This project uses the Microsoft GraphRAG library. Please refer to the original GraphRAG repository for licensing information.