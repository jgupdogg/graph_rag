"""
Enhanced Document Processor with AI-Driven Classification and Context-Aware Chunking
Implements Phase 1-3 of the RAG Enhancement Plan
"""

import json
import os
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import openai
from openai import OpenAI
import yaml
import hashlib

from document_structure_parser import Section, extract_document_structure
from structure_aware_chunking import StructureAwareChunk, StructureAwareChunker

logger = logging.getLogger(__name__)


@dataclass
class DocumentClassification:
    """Result of document classification"""
    document_type: str
    confidence: float
    characteristics: List[str]
    processing_strategy: str
    reasoning: str


@dataclass
class ProcessingStrategy:
    """Document type-specific processing configuration"""
    chunk_size: int
    overlap: int
    include_cross_references: bool = True
    extract_technical_terms: bool = False
    preserve_metrics: bool = False
    maintain_clause_refs: bool = False
    preserve_definitions: bool = False


@dataclass
class EnhancedChunk:
    """Enhanced chunk with AI-generated context and metadata"""
    text: str
    chunk_id: str
    section_metadata: Dict[str, Any]
    enhanced_context: str
    document_classification: DocumentClassification
    processing_strategy: ProcessingStrategy
    semantic_tags: List[str]
    cross_references: List[str]
    start_pos: int
    end_pos: int
    token_count: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame compatibility"""
        return {
            'text': self.enhanced_context + "\n\n" + self.text,
            'id': self.chunk_id,
            'n_tokens': self.token_count,
            'metadata': json.dumps({
                **self.section_metadata,
                'document_type': self.document_classification.document_type,
                'semantic_tags': self.semantic_tags,
                'cross_references': self.cross_references,
                'enhanced_context': self.enhanced_context
            }),
            'section_path': ' > '.join(self.section_metadata.get('section_path', [])),
            'section_title': self.section_metadata.get('section_title', ''),
            'section_level': self.section_metadata.get('section_level', 0),
            'document_type': self.document_classification.document_type,
            'semantic_tags': ', '.join(self.semantic_tags)
        }


class DocumentClassifier:
    """AI-powered document classifier using OpenAI API"""
    
    # Classification categories and their characteristics
    DOCUMENT_TYPES = {
        "Technical Specification": {
            "keywords": ["specification", "technical", "engineering", "standard", "requirement", "procedure"],
            "indicators": ["section numbers", "technical terms", "standards references", "measurements"]
        },
        "Business Document": {
            "keywords": ["report", "proposal", "analysis", "business", "revenue", "strategy", "executive"],
            "indicators": ["executive summary", "recommendations", "metrics", "objectives"]
        },
        "Legal/Contract": {
            "keywords": ["agreement", "contract", "terms", "conditions", "legal", "clause", "liability"],
            "indicators": ["legal language", "clauses", "definitions", "obligations"]
        },
        "Manual/Procedure": {
            "keywords": ["manual", "procedure", "instructions", "steps", "guide", "how-to"],
            "indicators": ["step-by-step", "numbered lists", "instructions", "procedures"]
        },
        "Research Paper": {
            "keywords": ["research", "study", "analysis", "methodology", "results", "conclusion"],
            "indicators": ["abstract", "methodology", "references", "citations"]
        },
        "Correspondence": {
            "keywords": ["letter", "memo", "email", "correspondence", "communication"],
            "indicators": ["date", "recipient", "sender", "informal tone"]
        }
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo", cache_enabled: bool = True):
        """
        Initialize document classifier
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: OpenAI model to use for classification
            cache_enabled: Whether to cache classification results
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.cache_enabled = cache_enabled
        self.cache_dir = Path("cache/classifications")
        if cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def classify_document(self, text_preview: str, metadata: Dict[str, Any]) -> DocumentClassification:
        """
        Classify document based on preview text and metadata
        
        Args:
            text_preview: First 1000 characters of document
            metadata: File metadata (name, size, etc.)
            
        Returns:
            DocumentClassification object
        """
        # Check cache first
        if self.cache_enabled:
            cache_key = self._get_cache_key(text_preview, metadata)
            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                return cached_result
        
        # Prepare classification prompt
        prompt = self._create_classification_prompt(text_preview, metadata)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert document classifier. Analyze documents and classify them accurately."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = self._parse_classification_response(response.choices[0].message.content)
            
            # Cache result
            if self.cache_enabled:
                self._save_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            # Return default classification
            return DocumentClassification(
                document_type="Business Document",
                confidence=0.5,
                characteristics=["Unknown structure"],
                processing_strategy="default",
                reasoning="Classification failed, using default"
            )
    
    def _create_classification_prompt(self, text_preview: str, metadata: Dict[str, Any]) -> str:
        """Create classification prompt"""
        document_types = list(self.DOCUMENT_TYPES.keys())
        
        return f"""Analyze this document preview and classify its type:

Document Preview:
{text_preview}

File Info:
- Name: {metadata.get('filename', 'Unknown')}
- Size: {metadata.get('file_size', 'Unknown')} characters
- Sections Found: {metadata.get('section_count', 0)}

Classify as one of: {', '.join(document_types)}

Provide your response in this exact JSON format:
{{
    "document_type": "chosen_type",
    "confidence": 0.85,
    "characteristics": ["characteristic1", "characteristic2", "characteristic3"],
    "processing_strategy": "strategy_recommendation",
    "reasoning": "Brief explanation of classification decision"
}}

Focus on identifying:
1. Document structure and formatting patterns
2. Language style (technical, legal, business, etc.)
3. Content purpose and target audience
4. Specific terminology and jargon used
"""
    
    def _parse_classification_response(self, response: str) -> DocumentClassification:
        """Parse AI response into DocumentClassification object"""
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end]
            
            data = json.loads(json_str)
            
            return DocumentClassification(
                document_type=data.get('document_type', 'Business Document'),
                confidence=float(data.get('confidence', 0.5)),
                characteristics=data.get('characteristics', []),
                processing_strategy=data.get('processing_strategy', 'default'),
                reasoning=data.get('reasoning', 'No reasoning provided')
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse classification response: {e}")
            # Fallback parsing
            return self._fallback_classification(response)
    
    def _fallback_classification(self, response: str) -> DocumentClassification:
        """Fallback classification if JSON parsing fails"""
        response_lower = response.lower()
        
        # Simple keyword matching
        for doc_type, info in self.DOCUMENT_TYPES.items():
            if any(keyword in response_lower for keyword in info["keywords"]):
                return DocumentClassification(
                    document_type=doc_type,
                    confidence=0.7,
                    characteristics=["Keyword-based classification"],
                    processing_strategy="default",
                    reasoning="Fallback classification based on keywords"
                )
        
        # Default classification
        return DocumentClassification(
            document_type="Business Document",
            confidence=0.5,
            characteristics=["Default classification"],
            processing_strategy="default",
            reasoning="Unable to classify, using default"
        )
    
    def _get_cache_key(self, text_preview: str, metadata: Dict[str, Any]) -> str:
        """Generate cache key for classification result"""
        content = text_preview + str(sorted(metadata.items()))
        return hashlib.md5(content.encode()).hexdigest()
    
    def _save_to_cache(self, cache_key: str, result: DocumentClassification) -> None:
        """Save classification result to cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save classification to cache: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[DocumentClassification]:
        """Load classification result from cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return DocumentClassification(**data)
        except Exception as e:
            logger.warning(f"Failed to load classification from cache: {e}")
        return None


class SectionSummarizer:
    """AI-powered section summarizer for enhanced context"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo", cache_enabled: bool = True):
        """
        Initialize section summarizer
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            cache_enabled: Whether to cache summaries
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.cache_enabled = cache_enabled
        self.cache_dir = Path("cache/summaries")
        if cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_section_summary(self, section: Section, doc_type: str) -> str:
        """
        Generate contextual summary for a section
        
        Args:
            section: Section object with content and metadata
            doc_type: Document classification type
            
        Returns:
            Concise summary for chunk prefixing
        """
        # Check cache first
        if self.cache_enabled:
            cache_key = self._get_summary_cache_key(section, doc_type)
            cached_summary = self._load_summary_from_cache(cache_key)
            if cached_summary:
                return cached_summary
        
        # Generate summary
        try:
            prompt = self._create_summary_prompt(section, doc_type)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating concise, contextual summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Cache result
            if self.cache_enabled:
                self._save_summary_to_cache(cache_key, summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            # Return fallback summary
            return f"This section covers {section.title} in the context of {' > '.join(section.parent_path)}."
    
    def _create_summary_prompt(self, section: Section, doc_type: str) -> str:
        """Create summary generation prompt based on document type"""
        
        content_preview = section.content[:500] if section.content else section.title
        
        if doc_type == "Technical Specification":
            focus = "What this section specifies, what components/materials it covers, and how it relates to the overall system."
        elif doc_type == "Business Document":
            focus = "What business objective this section addresses and its main conclusions or recommendations."
        elif doc_type == "Legal/Contract":
            focus = "What legal obligations or rights this section establishes and their scope."
        elif doc_type == "Manual/Procedure":
            focus = "What process or procedure this section describes and its purpose."
        else:
            focus = "The main purpose and key points of this section."
        
        return f"""Summarize this {doc_type.lower()} section's purpose in 1-2 concise sentences:

Section: {section.title}
Context: {' > '.join(section.parent_path) if section.parent_path else 'Root level'}
Content Preview: {content_preview}

Focus on: {focus}

Provide a summary that would help someone understand what this section contains without reading the full content."""
    
    def _get_summary_cache_key(self, section: Section, doc_type: str) -> str:
        """Generate cache key for section summary"""
        content = f"{section.title}_{doc_type}_{section.content[:200] if section.content else ''}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _save_summary_to_cache(self, cache_key: str, summary: str) -> None:
        """Save summary to cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.txt"
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(summary)
        except Exception as e:
            logger.warning(f"Failed to save summary to cache: {e}")
    
    def _load_summary_from_cache(self, cache_key: str) -> Optional[str]:
        """Load summary from cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.txt"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to load summary from cache: {e}")
        return None


class EnhancedDocumentProcessor:
    """Main processor that orchestrates classification, summarization, and enhanced chunking"""
    
    # Processing strategies by document type
    PROCESSING_STRATEGIES = {
        "Technical Specification": ProcessingStrategy(
            chunk_size=750,
            overlap=100,
            include_cross_references=True,
            extract_technical_terms=True
        ),
        "Business Document": ProcessingStrategy(
            chunk_size=500,
            overlap=75,
            preserve_metrics=True
        ),
        "Legal/Contract": ProcessingStrategy(
            chunk_size=400,
            overlap=50,
            maintain_clause_refs=True,
            preserve_definitions=True
        ),
        "Manual/Procedure": ProcessingStrategy(
            chunk_size=600,
            overlap=75,
            include_cross_references=True
        ),
        "Research Paper": ProcessingStrategy(
            chunk_size=600,
            overlap=75,
            include_cross_references=True
        ),
        "Correspondence": ProcessingStrategy(
            chunk_size=400,
            overlap=25
        )
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced document processor
        
        Args:
            config: Configuration dictionary with API settings
        """
        self.config = config or {}
        
        # Initialize components
        api_key = self.config.get('api_key') or os.getenv("OPENAI_API_KEY")
        model = self.config.get('model', 'gpt-3.5-turbo')
        cache_enabled = self.config.get('cache_enabled', True)
        
        self.classifier = DocumentClassifier(api_key, model, cache_enabled)
        self.summarizer = SectionSummarizer(api_key, model, cache_enabled)
        
        # Default chunker - will be updated based on document type
        self.chunker = StructureAwareChunker(chunk_size=500, overlap=50)
    
    def process_document(self, text: str, metadata: Dict[str, Any]) -> Tuple[List[EnhancedChunk], DocumentClassification]:
        """
        Process document with AI-enhanced classification and context-aware chunking
        
        Args:
            text: Full document text
            metadata: Document metadata (filename, size, etc.)
            
        Returns:
            Tuple of (enhanced chunks list, document classification)
        """
        logger.info(f"Processing document: {metadata.get('filename', 'Unknown')}")
        
        # Step 1: Classify document
        text_preview = text[:1000]
        doc_classification = self.classifier.classify_document(text_preview, metadata)
        logger.info(f"Classified as: {doc_classification.document_type} (confidence: {doc_classification.confidence:.2f})")
        
        # Step 2: Parse document structure
        sections, structured_text, section_metadata = extract_document_structure(text)
        logger.info(f"Extracted {len(sections)} sections")
        
        # Step 3: Generate section summaries
        for section in sections:
            section.summary = self.summarizer.generate_section_summary(section, doc_classification.document_type)
        
        # Step 4: Get processing strategy
        strategy = self.get_processing_strategy(doc_classification.document_type)
        
        # Step 5: Update chunker with strategy-specific settings
        self.chunker = StructureAwareChunker(
            chunk_size=strategy.chunk_size,
            overlap=strategy.overlap
        )
        
        # Step 6: Create enhanced chunks
        enhanced_chunks = self._create_enhanced_chunks(
            structured_text, 
            sections, 
            doc_classification, 
            strategy,
            metadata
        )
        
        logger.info(f"Created {len(enhanced_chunks)} enhanced chunks")
        return enhanced_chunks, doc_classification
    
    def get_processing_strategy(self, doc_type: str) -> ProcessingStrategy:
        """Get processing strategy for document type"""
        return self.PROCESSING_STRATEGIES.get(doc_type, self.PROCESSING_STRATEGIES["Business Document"])
    
    def _create_enhanced_chunks(
        self, 
        structured_text: str, 
        sections: List[Section], 
        doc_classification: DocumentClassification,
        strategy: ProcessingStrategy,
        metadata: Dict[str, Any]
    ) -> List[EnhancedChunk]:
        """Create enhanced chunks with AI-generated context"""
        
        # First create regular structure-aware chunks
        regular_chunks = self.chunker.chunk_with_structure(
            structured_text, 
            Path("dummy")  # We'll pass sections directly
        )
        
        enhanced_chunks = []
        
        for chunk in regular_chunks:
            # Find corresponding section
            section = self._find_section_for_chunk(sections, chunk)
            
            # Generate enhanced context
            enhanced_context = self._generate_enhanced_context(
                chunk, section, doc_classification, strategy, metadata
            )
            
            # Extract semantic tags
            semantic_tags = self._extract_semantic_tags(chunk.text, doc_classification.document_type)
            
            # Find cross-references
            cross_references = self._extract_cross_references(chunk.text, sections)
            
            # Create enhanced chunk
            enhanced_chunk = EnhancedChunk(
                text=chunk.text,
                chunk_id=chunk.chunk_id,
                section_metadata=chunk.section_metadata,
                enhanced_context=enhanced_context,
                document_classification=doc_classification,
                processing_strategy=strategy,
                semantic_tags=semantic_tags,
                cross_references=cross_references,
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
                token_count=chunk.token_count
            )
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def _find_section_for_chunk(self, sections: List[Section], chunk: StructureAwareChunk) -> Optional[Section]:
        """Find the section that contains this chunk"""
        for section in sections:
            if (hasattr(section, 'start_pos') and hasattr(section, 'end_pos') and
                section.start_pos <= chunk.start_pos < section.end_pos):
                return section
        return None
    
    def _generate_enhanced_context(
        self, 
        chunk: StructureAwareChunk, 
        section: Optional[Section],
        doc_classification: DocumentClassification,
        strategy: ProcessingStrategy,
        metadata: Dict[str, Any]
    ) -> str:
        """Generate enhanced context header for chunk"""
        
        context_parts = []
        
        # Document-level context
        context_parts.append(f"[DOCUMENT: {doc_classification.document_type} - {metadata.get('filename', 'Unknown')}]")
        
        # Section context
        if section and hasattr(section, 'summary'):
            context_parts.append(f"[SECTION CONTEXT: {section.summary}]")
        
        # Hierarchy context
        section_path = chunk.section_metadata.get('section_path', [])
        if section_path:
            context_parts.append(f"[HIERARCHY: {' > '.join(section_path)}]")
        
        # Cross-references (placeholder for now)
        if strategy.include_cross_references:
            context_parts.append("[RELATED: To be populated with cross-references]")
        
        return '\n'.join(context_parts)
    
    def _extract_semantic_tags(self, text: str, doc_type: str) -> List[str]:
        """Extract semantic tags from chunk text (simple keyword-based for now)"""
        # This is a simplified implementation
        # In a full implementation, this could use NLP or AI for better tagging
        
        text_lower = text.lower()
        tags = []
        
        # Document-type specific tags
        if doc_type == "Technical Specification":
            tech_keywords = ["specification", "requirement", "standard", "procedure", "material", "installation", "testing"]
            tags.extend([keyword for keyword in tech_keywords if keyword in text_lower])
        
        elif doc_type == "Business Document":
            business_keywords = ["strategy", "analysis", "revenue", "growth", "market", "customer", "objective"]
            tags.extend([keyword for keyword in business_keywords if keyword in text_lower])
        
        # General tags
        if "storm" in text_lower and "drain" in text_lower:
            tags.append("storm_drainage")
        if "concrete" in text_lower:
            tags.append("concrete")
        if "installation" in text_lower:
            tags.append("installation")
        
        return list(set(tags))  # Remove duplicates
    
    def _extract_cross_references(self, text: str, sections: List[Section]) -> List[str]:
        """Extract cross-references to other sections (simplified implementation)"""
        import re
        
        references = []
        
        # Look for section references like "Section 4.2", "see 3.1", etc.
        section_patterns = [
            r'[Ss]ection\s+(\d+(?:\.\d+)*)',
            r'[Ss]ee\s+(\d+(?:\.\d+)*)',
            r'[Rr]efer\s+to\s+(\d+(?:\.\d+)*)',
            r'\b(\d+(?:\.\d+)+)\b'  # Plain section numbers
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, text)
            references.extend(matches)
        
        # Validate references against actual sections
        valid_references = []
        section_numbers = set()
        for section in sections:
            # Extract section numbers from titles
            section_title = section.title
            section_num_match = re.search(r'^(\d+(?:\.\d+)*)', section_title)
            if section_num_match:
                section_numbers.add(section_num_match.group(1))
        
        for ref in references:
            if ref in section_numbers:
                valid_references.append(f"Section {ref}")
        
        return list(set(valid_references))  # Remove duplicates


def load_enhanced_processor_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration for enhanced document processor"""
    
    default_config = {
        'classification': {
            'enabled': True,
            'cache_results': True,
            'api_model': 'gpt-3.5-turbo'
        },
        'summarization': {
            'enabled': True,
            'cache_results': True,
            'api_model': 'gpt-3.5-turbo',
            'max_summary_length': 150
        },
        'processing_strategies': {
            'technical_specification': {
                'chunk_size': 750,
                'overlap': 100,
                'include_cross_references': True,
                'extract_technical_terms': True
            },
            'business_document': {
                'chunk_size': 500,
                'overlap': 75,
                'preserve_metrics': True,
                'link_recommendations': True
            },
            'legal_contract': {
                'chunk_size': 400,
                'overlap': 50,
                'maintain_clause_refs': True,
                'preserve_definitions': True
            }
        }
    }
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            # Merge with defaults
            if 'document_processing' in file_config:
                default_config.update(file_config['document_processing'])
        
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    return default_config