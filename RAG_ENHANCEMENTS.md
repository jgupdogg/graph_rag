# RAG Enhancements: AI-Driven Document Classification & Context-Aware Chunking

## Problem Statement

Current GraphRAG chunking creates semantic gaps where related information gets separated, leading to incomplete retrieval results. For example:
- **Query**: "storm drains" 
- **Missing**: Concrete specifications that apply to storm drains
- **Cause**: Related content split across chunks without preserving contextual relationships

## Solution Overview

Implement a two-phase AI enhancement system:
1. **Document Classification**: Use OpenAI API to classify document types and determine optimal processing strategies
2. **Section Summarization**: Generate contextual summaries that get prepended to every chunk, preserving semantic relationships

## Current State Analysis

### Existing Strengths
- ✅ **Structure-aware chunking** (`structure_aware_chunking.py`)
- ✅ **Document structure parser** (`document_structure_parser.py`)
- ✅ **Section metadata preservation** (section paths, levels, titles)
- ✅ **GraphRAG integration** with configurable chunk sizes (500 tokens, 50 overlap)
- ✅ **Section markers** for hierarchical context

### Current Limitations
- ❌ **Semantic isolation**: Related sections lose contextual links
- ❌ **One-size-fits-all**: Same chunking strategy for all document types
- ❌ **Missing cross-references**: No automatic linking of related sections
- ❌ **Limited context**: Chunks lack broader section purpose/scope

## Enhanced Solution Architecture

### Phase 1: AI Document Classifier Service

#### 1.1 Document Classification API Integration
```python
class DocumentClassifier:
    def classify_document(self, text_preview: str, metadata: dict) -> DocumentClassification:
        """
        Analyze document preview to determine type and processing strategy
        
        Args:
            text_preview: First 1000 characters of document
            metadata: File info (name, size, structure complexity)
            
        Returns:
            DocumentClassification with type, confidence, and strategy
        """
```

#### 1.2 Classification Categories
- **Technical Specifications**: Engineering docs, standards, specs
- **Business Documents**: Reports, proposals, presentations
- **Legal/Contracts**: Agreements, policies, compliance docs
- **Manuals/Procedures**: Instructions, SOPs, guidelines
- **Research Papers**: Academic, technical research
- **Correspondence**: Letters, emails, memos

#### 1.3 Classification Prompt Template
```
Analyze this document preview and classify its type:

Document Preview:
{text_preview}

File Info:
- Name: {filename}
- Size: {file_size}
- Sections Found: {section_count}

Classify as one of: Technical Specification, Business Document, Legal/Contract, Manual/Procedure, Research Paper, Correspondence

Provide:
1. Document Type
2. Confidence (0-100%)
3. Key Characteristics Observed
4. Recommended Processing Strategy
```

### Phase 2: Document-Type-Specific Processing Strategies

#### 2.1 Technical Specifications Strategy
- **Chunk Size**: 750 tokens (larger for technical context)
- **Overlap**: 100 tokens (high overlap for technical relationships)
- **Special Processing**:
  - Extract technical terms and definitions
  - Map material/component relationships
  - Cross-reference related sections (materials → applications)
  - Include standards and reference numbers

#### 2.2 Business Documents Strategy
- **Chunk Size**: 500 tokens (standard)
- **Overlap**: 75 tokens (medium overlap)
- **Special Processing**:
  - Preserve executive summary relationships
  - Link action items to context
  - Maintain data/metrics relationships
  - Connect recommendations to supporting evidence

#### 2.3 Legal/Contracts Strategy
- **Chunk Size**: 400 tokens (smaller for precision)
- **Overlap**: 50 tokens (minimal to avoid confusion)
- **Special Processing**:
  - Maintain clause relationships
  - Preserve legal references
  - Keep definitions linked to usage
  - Maintain obligation/responsibility chains

### Phase 3: AI Section Summarization System

#### 3.1 Section Summary Generation
```python
class SectionSummarizer:
    def generate_section_summary(self, section: Section, doc_type: str) -> str:
        """
        Generate 1-2 sentence summary of section purpose and content
        
        Args:
            section: Section object with content and metadata
            doc_type: Document classification for context
            
        Returns:
            Concise summary for chunk prefixing
        """
```

#### 3.2 Summary Prompt Templates

**For Technical Specifications**:
```
Summarize this technical section's purpose and main topics in 1-2 sentences:

Section: {section_title}
Context: {parent_sections}
Content Preview: {first_500_chars}

Focus on: What this section specifies, what components/materials it covers, and how it relates to the overall system.
```

**For Business Documents**:
```
Summarize this business section's purpose and key points in 1-2 sentences:

Section: {section_title}
Content Preview: {first_500_chars}

Focus on: What business objective this section addresses and its main conclusions or recommendations.
```

#### 3.3 Context-Enhanced Chunk Format
```
[DOCUMENT: Technical Specification - Storm Water Management]
[SECTION CONTEXT: This section specifies concrete requirements and installation procedures for storm drainage systems, including material grades and structural specifications.]
[HIERARCHY: Specifications > Materials > Concrete > Storm Drain Applications]
[RELATED: Drainage Systems (4.2), Installation Procedures (5.1)]

{original_chunk_content}
```

### Phase 4: Enhanced Context Injection

#### 4.1 Cross-Reference Mapping
- **Automatic Detection**: Identify sections that reference each other
- **Semantic Linking**: Connect related concepts across sections
- **Bidirectional References**: Ensure mutual discoverability

#### 4.2 Context Enrichment Strategies

**Technical Documents**:
- Include parent specification context
- Add related material/component information
- Reference applicable standards and codes
- Connect to installation/maintenance procedures

**Business Documents**:
- Include executive summary context
- Link to supporting data/metrics
- Connect recommendations to evidence
- Reference related business objectives

#### 4.3 Smart Chunk Metadata Enhancement
```python
{
    "text": "enhanced_chunk_content",
    "section_metadata": {
        "section_title": "Concrete Requirements",
        "section_summary": "Specifies concrete grades and installation for storm drainage",
        "section_path": ["Specifications", "Materials", "Concrete", "Storm Drains"],
        "related_sections": ["Drainage Systems", "Installation Procedures"],
        "document_type": "Technical Specification",
        "semantic_tags": ["concrete", "storm drainage", "materials", "installation"],
        "cross_references": ["Section 4.2", "Section 5.1"]
    }
}
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
1. **Create Document Classifier**
   - Implement `DocumentClassifier` class
   - Add OpenAI API integration
   - Define classification categories and prompts
   - Test with existing documents

2. **Extend Configuration**
   - Add document type strategies to settings.yaml
   - Create processing strategy configs
   - Update workspace creation to include classification

### Phase 2: Section Summarization (Week 2)
1. **Implement Section Summarizer**
   - Create `SectionSummarizer` class
   - Add type-specific prompt templates
   - Integrate with existing structure parser
   - Cache summaries to avoid reprocessing

2. **Enhance Chunking Pipeline**
   - Modify `StructureAwareChunker` to include summaries
   - Add context prefixing to chunks
   - Update metadata structure

### Phase 3: Context Enhancement (Week 3)
1. **Cross-Reference Detection**
   - Implement automatic section relationship detection
   - Add semantic linking between related sections
   - Create bidirectional reference mapping

2. **Enhanced Metadata**
   - Add semantic tagging to chunks
   - Include relationship indicators
   - Expand context breadcrumbs

### Phase 4: Integration & Optimization (Week 4)
1. **GraphRAG Integration**
   - Update embedding pipeline to handle enhanced chunks
   - Optimize for semantic search performance
   - Test retrieval quality improvements

2. **Performance Optimization**
   - Cache classification and summarization results
   - Batch API calls for efficiency
   - Add fallback strategies for API failures

## Technical Implementation Details

### API Integration
```python
class EnhancedDocumentProcessor:
    def __init__(self):
        self.classifier = DocumentClassifier()
        self.summarizer = SectionSummarizer()
        self.chunker = StructureAwareChunker()
    
    def process_document(self, text: str, metadata: dict) -> List[EnhancedChunk]:
        # 1. Classify document
        doc_classification = self.classifier.classify_document(text[:1000], metadata)
        
        # 2. Parse structure
        sections, structured_text, section_metadata = extract_document_structure(text)
        
        # 3. Generate section summaries
        for section in sections:
            section.summary = self.summarizer.generate_section_summary(
                section, doc_classification.type
            )
        
        # 4. Create enhanced chunks with context
        strategy = self.get_processing_strategy(doc_classification.type)
        chunks = self.chunker.chunk_with_enhanced_context(
            structured_text, sections, strategy
        )
        
        return chunks
```

### Configuration Extensions
```yaml
# Add to settings.yaml
document_processing:
  classification:
    enabled: true
    cache_results: true
    api_model: "gpt-3.5-turbo"
    
  summarization:
    enabled: true
    cache_results: true
    api_model: "gpt-3.5-turbo"
    max_summary_length: 150
    
  strategies:
    technical_specification:
      chunk_size: 750
      overlap: 100
      include_cross_references: true
      extract_technical_terms: true
      
    business_document:
      chunk_size: 500
      overlap: 75
      preserve_metrics: true
      link_recommendations: true
      
    legal_contract:
      chunk_size: 400
      overlap: 50
      maintain_clause_refs: true
      preserve_definitions: true
```

## Expected Benefits

### Improved Retrieval Quality
- **25-40% improvement** in semantic search relevance
- **Cross-section discovery**: Related content across document sections
- **Context preservation**: Broader meaning maintained in chunks

### Enhanced User Experience
- **More complete answers**: RAG responses include related context
- **Better coverage**: Finds relevant information across document structure
- **Semantic understanding**: AI understands document intent and relationships

### System Efficiency
- **Intelligent processing**: Document-appropriate strategies
- **Reduced redundancy**: Better chunk boundaries reduce overlap needs
- **Cached results**: Classification and summarization cached for reuse

## Cost Considerations

### API Usage Estimates
- **Classification**: ~100 tokens per document (1-2 cents)
- **Summarization**: ~200 tokens per section (2-4 cents per section)
- **Total per document**: $0.10-0.50 depending on document complexity

### Cost Optimization Strategies
- Cache all AI-generated content
- Batch process documents when possible
- Use classification to skip summarization for simple documents
- Implement fallback strategies for cost-sensitive scenarios

## Success Metrics

### Quantitative Measures
- **Retrieval precision**: Percentage of relevant results in top-10
- **Cross-section discovery**: How often related sections are found
- **User satisfaction**: Query success rate improvements

### Qualitative Assessment
- **Storm drain scenario**: Can find concrete specs when querying drainage
- **Technical coherence**: Related technical sections discoverable
- **Business context**: Recommendations linked to supporting evidence

## Future Enhancements

### Advanced Features
- **Multi-document relationships**: Link related content across documents
- **Dynamic summarization**: Update summaries based on query context
- **Learning system**: Improve classification based on user feedback

### Integration Opportunities
- **Domain-specific models**: Train classifiers for specific industries
- **Custom embedding models**: Fine-tune embeddings for enhanced contexts
- **Real-time processing**: Stream-process documents as they're uploaded

---

## Implementation Notes

This enhancement builds on the existing solid foundation while maintaining backwards compatibility. The phased approach allows for incremental implementation and testing, ensuring the system remains stable while gaining these powerful new capabilities.

The solution directly addresses the semantic chunking problem by ensuring that every chunk carries sufficient context to be discoverable and meaningful, regardless of where it appears in the document structure.