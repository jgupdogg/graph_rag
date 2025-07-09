# GraphRAG Technical Analysis: Understanding the Inner Workings

## Overview

This document provides a comprehensive analysis of how Microsoft GraphRAG processes documents, builds knowledge graphs, and enables intelligent querying. This analysis is based on examining the source code and understanding the internal pipeline, particularly relevant for processing large technical documents like the Baltimore City Engineering Specifications PDF.

## How GraphRAG Works: The Complete Pipeline

### 1. Document Input and Processing

#### Supported File Types
- **Native support**: text (.txt), CSV (.csv), JSON (.json)
- **No native PDF support** - PDFs must be converted to text first
- Input processing located in `graphrag/index/input/`

#### Text Chunking Strategy
GraphRAG uses sophisticated chunking to prepare text for LLM processing:

**Token-based Chunking (Default)**:
- Uses OpenAI's tiktoken library for tokenization
- Default configuration:
  - Chunk size: 1200 tokens
  - Overlap: 100 tokens
  - Encoding: cl100k_base
- Maintains document references and chunk IDs
- Preserves context across chunk boundaries

**Sentence-based Chunking (Alternative)**:
- Uses NLTK sentence tokenizer
- Better for preserving complete thoughts
- Less efficient for technical documents

### 2. Entity and Relationship Extraction

#### LLM-Based Extraction Process
GraphRAG uses structured prompts to extract entities and relationships from each chunk:

**Entity Format**:
```
("entity"|<entity_name>|<entity_type>|<entity_description>)
```

**Relationship Format**:
```
("relationship"|<source_entity>|<target_entity>|<relationship_description>|<relationship_strength>)
```

**Default Entity Types**:
- organization
- person
- geo (geographic location)
- event

**Extraction Features**:
- **Gleaning mechanism**: Multiple extraction passes to catch missed entities
- **Structured output**: Uses JSON mode for consistent parsing
- **Context preservation**: Maintains chunk context for better extraction
- **Deduplication**: Merges entities with same name across chunks

### 3. Graph Construction and Analysis

#### Graph Building Process
1. **NetworkX Graph Creation**: Entities become nodes, relationships become edges
2. **Weight Aggregation**: Combines relationship strengths from multiple mentions
3. **Degree Computation**: Calculates node importance based on connections

#### Community Detection
Uses **Hierarchical Leiden Algorithm** from graspologic:
- Identifies clusters of related entities
- Supports configurable maximum cluster size
- Creates hierarchical community structure
- Enables multi-level analysis

#### Graph Embeddings
Uses **Node2Vec Algorithm** for vector representations:
- Default: 1536-dimensional embeddings
- Parameters:
  - num_walks: 10
  - walk_length: 40
  - window_size: 2
  - iterations: 3
- Enables similarity search and clustering

### 4. Storage Architecture

#### Multi-Layer Storage System

**Parquet Files** (Structured Data):
- `create_final_entities.parquet` - All extracted entities
- `create_final_relationships.parquet` - All relationships
- `create_final_communities.parquet` - Community clusters
- `create_final_community_reports.parquet` - AI-generated summaries
- `create_final_text_units.parquet` - Processed text chunks
- `create_final_documents.parquet` - Source document metadata

**LanceDB** (Vector Storage):
- High-performance vector database
- Stores embeddings for similarity search
- Optimized for fast retrieval
- Located in `output/lancedb/`

**Cache Layer**:
- LLM response caching (reduces API costs)
- Preprocessed text chunks
- Intermediate computation results

### 5. Community Report Generation

#### Hierarchical Summarization
- Processes communities at different hierarchy levels
- Aggregates entity and relationship information
- Generates structured reports with:
  - Title
  - Summary
  - Key findings (with explanations)
  - Importance ratings

#### Report Configuration
- Maximum report length: configurable (default 2000 tokens)
- Supports both text and graph-based prompts
- Async processing for parallel generation

### 6. Query Processing

#### Search Methods

**Global Search**:
- Searches across all community reports
- Best for high-level questions and themes
- Uses map-reduce pattern for comprehensive answers

**Local Search**:
- Focuses on specific entities and their neighborhoods
- Best for detailed questions about specific topics
- Uses vector similarity and graph traversal

**Drift Search**:
- Follows conversation context
- Adapts search based on previous queries

**Basic Search**:
- Simple keyword-based retrieval
- Fastest but least intelligent

## Processing Large Technical Documents (Baltimore City Specs)

### Challenges with Large PDFs

1. **File Size**: 116MB PDF = ~500,000+ tokens
2. **Technical Density**: Engineering specifications are information-dense
3. **Structure Preservation**: Tables, diagrams, references need special handling
4. **Domain Specificity**: Default entity types may not capture technical concepts

### Recommended Configuration for Technical Documents

```yaml
# Chunking Configuration
chunks:
  size: 800        # Smaller chunks for dense technical content
  overlap: 200     # More overlap to preserve context
  group_by_columns: [section, specification_id]

# Entity Extraction
extract_graph:
  entity_types: [
    specification,
    material,
    standard,
    equipment,
    procedure,
    requirement,
    organization,
    location
  ]
  max_gleanings: 2  # Multiple passes for complex text

# Community Detection
cluster_graph:
  max_cluster_size: 15  # Larger clusters for related specifications

# Report Generation
community_reports:
  max_length: 2500     # Longer reports for technical summaries
  max_input_length: 10000  # Handle larger contexts
```

### Cost and Performance Estimates

For a 116MB engineering specification PDF:

**Token Analysis**:
- Estimated text tokens: ~500,000
- Number of chunks (800 tokens): ~625 chunks
- Overlap tokens: ~125,000 additional

**API Calls Required**:
- Entity extraction: 625-1,250 calls (with gleaning)
- Embeddings: 625 calls
- Community reports: 50-100 calls

**Estimated Costs**:
- GPT-4 Turbo: $50-100
- GPT-3.5 Turbo: $5-15
- Embeddings: $1-2

**Processing Time**:
- With concurrency (25 requests): 2-4 hours
- Sequential processing: 8-12 hours

### Optimization Strategies

1. **Preprocessing**:
   ```python
   # Extract structured data first
   - Pull out tables as CSV
   - Extract section headers
   - Identify specification numbers
   ```

2. **Selective Processing**:
   - Process by chapter or section
   - Focus on specific specification types
   - Use regex to pre-filter relevant content

3. **Custom Entity Extraction**:
   - Add domain-specific examples to prompts
   - Define technical relationship types
   - Include measurement and tolerance extraction

4. **Caching Strategy**:
   - GraphRAG automatically caches LLM responses
   - Rerun with same content reuses cached results
   - Enables iterative refinement

## Query Examples for Technical Documents

### Global Queries
- "What are the main categories of specifications in this document?"
- "Summarize all concrete-related requirements"
- "What standards are most frequently referenced?"

### Local Queries
- "What are the specifications for Type A concrete?"
- "Which materials must comply with ASTM standards?"
- "What is the relationship between drainage and grading requirements?"

## Advanced Features and Customization

### Custom Prompt Engineering
Modify `prompts/extract_graph.txt` to include technical examples:

```
Example: Technical Specification
Entity_types: SPECIFICATION,MATERIAL,STANDARD
Text: Type A concrete shall have a minimum compressive strength of 4,000 psi at 28 days and shall conform to ASTM C94.
Output:
("entity"|TYPE A CONCRETE|MATERIAL|Concrete with 4,000 psi minimum compressive strength at 28 days)
("entity"|ASTM C94|STANDARD|Standard specification for ready-mixed concrete)
("relationship"|TYPE A CONCRETE|ASTM C94|Type A concrete must conform to this standard|9)
```

### Extending Entity Types
Add domain-specific entities in settings.yaml:
- measurement
- tolerance
- test_method
- approval_requirement
- submittal
- warranty

### Performance Tuning
1. **Reduce chunk size** for dense technical sections
2. **Increase overlap** for specification continuity
3. **Use GPT-3.5** for initial extraction, GPT-4 for reports
4. **Implement parallel processing** with higher concurrency
5. **Cache aggressively** for iterative development

## Conclusion

GraphRAG provides a powerful framework for extracting knowledge from documents, but processing large technical PDFs requires careful configuration and optimization. The system's modular architecture allows for extensive customization to handle domain-specific content effectively.

Key success factors:
1. Proper PDF-to-text conversion
2. Domain-specific entity configuration
3. Appropriate chunking strategy
4. Cost-aware processing approach
5. Iterative refinement using cached results

With these considerations, GraphRAG can effectively transform the Baltimore City Engineering Specifications into a queryable knowledge graph, enabling intelligent search and analysis of complex technical requirements.