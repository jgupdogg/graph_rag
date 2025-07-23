# Enhanced GraphRAG Features Implementation

## Overview
This implementation adds three major enhancements to the GraphRAG system:
1. **Rate Limiting** for API calls to prevent rate limit errors
2. **Raw Text Embeddings** for more precise retrieval (instead of summary embeddings)
3. **Enhanced Two-Stage Search** that combines raw text embeddings with graph knowledge

## Key Components Added

### 1. Rate Limiter (`rate_limiter.py`)
- Thread-safe rate limiting with configurable limits
- Exponential backoff for rate limit errors
- Request tracking and statistics
- Default limits: 50 requests/min, 150k tokens/min
- Minimum 200ms delay between requests

### 2. Raw Text Embeddings (`raw_text_embeddings.py`)
- Generates embeddings from original text chunks (not summaries)
- Stores in LanceDB alongside summary embeddings
- Batch processing with rate limiting
- Integrated into document processing pipeline

### 3. Enhanced Query Handler (`enhanced_query_handler.py`)
- Two-stage retrieval process:
  1. Initial retrieval using raw text embeddings
  2. Context expansion using graph relationships
- Returns both text content and graph context
- Supports both GPT-4 and O1 models for answer generation

## Integration Points

### Document Processing Pipeline
- Added raw text embedding generation after GraphRAG indexing
- Runs in parallel with summary generation
- Stores embeddings in separate LanceDB table

### Query Interface
- New "enhanced" search method available in Streamlit UI
- Single document support (can be extended to multi-doc)
- Shows retrieved chunks with relevance scores
- Includes entity and relationship information

## Usage

### Processing Documents
When documents are processed, the system now:
1. Runs standard GraphRAG indexing
2. Generates raw text embeddings (with rate limiting)
3. Generates summary embeddings (existing functionality)

### Querying
Users can select "Enhanced Search" in the chat interface to:
- Get more precise retrieval based on exact text matches
- See graph context (entities, relationships) for retrieved chunks
- Use O1 models for complex reasoning tasks

## Configuration

### Rate Limiting
Default settings in `rate_limiter.py`:
```python
requests_per_minute=50
tokens_per_minute=150000
min_delay_between_requests=0.2
```

### Embedding Model
Uses OpenAI's `text-embedding-3-small` (1536 dimensions)

### Storage
- Raw embeddings: `{workspace}/output/lancedb/raw-text-embeddings`
- Summary embeddings: `{workspace}/output/lancedb/default-summary-text`

## Benefits

1. **Better Retrieval Accuracy**: Raw text embeddings preserve all details
2. **Rate Limit Protection**: Prevents API errors during heavy processing
3. **Graph Context**: Combines vector search with knowledge graph
4. **Flexible Architecture**: Can easily extend to multi-document search

## Testing
Run `test_enhanced_features.py` to verify:
- Rate limiter functionality
- Raw text embedding storage
- Enhanced query handler initialization

## Future Enhancements
- Multi-document enhanced search
- Configurable embedding models
- Hybrid scoring (combine embedding similarity with graph metrics)
- Caching for frequently accessed embeddings