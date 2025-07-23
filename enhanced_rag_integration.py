"""
Integration module for enhanced RAG features into the existing GraphRAG pipeline.
Provides a unified interface to integrate structure-aware chunking and bullet point extraction.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import shutil
import yaml
from datetime import datetime

try:
    # First try the standard version (thread-safe)
    from enhanced_document_processor import EnhancedDocumentProcessor, DocumentType
except ImportError:
    # Fall back to timeout version if standard is not available
    # Note: This version uses signals and won't work in threaded environments like Streamlit
    from enhanced_document_processor_with_timeout import EnhancedDocumentProcessor, DocumentType
    import logging
    logging.getLogger(__name__).warning("Using timeout version of EnhancedDocumentProcessor - may not work in threaded environments")
from bullet_point_extractor import BulletPointExtractor
from structure_aware_chunking import StructuredChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedRAGIntegration:
    """Integrates enhanced RAG features into the existing GraphRAG pipeline"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.processor = EnhancedDocumentProcessor(api_key)
        self.bullet_extractor = BulletPointExtractor(api_key)
    
    def process_document_enhanced(
        self, 
        workspace_path: Path, 
        doc_id: str,
        display_name: str,
        use_bullet_points: bool = True
    ) -> Dict[str, Any]:
        """
        Process document with enhanced RAG features
        
        Returns:
            Dictionary containing processing results and metadata
        """
        
        logger.info(f"Starting enhanced processing for document: {display_name}")
        
        # Find the text file in workspace
        input_dir = workspace_path / "input"
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
            
        text_files = list(input_dir.glob("*.txt"))
        
        if not text_files:
            # Try to find any text-like files
            text_files = list(input_dir.glob("*"))
            text_files = [f for f in text_files if f.suffix.lower() in ['.txt', '.md', '.text']]
            
            if not text_files:
                raise ValueError(f"No text file found in {input_dir}. Expected at least one .txt file.")
        
        text_file = text_files[0]
        if len(text_files) > 1:
            logger.warning(f"Multiple text files found in {input_dir}, using: {text_file}")
        logger.info(f"Processing text file: {text_file}")
        
        # Read document text
        with open(text_file, 'r', encoding='utf-8') as f:
            document_text = f.read()
        
        # Step 1: Process document with classification and structure-aware chunking
        processing_result = self.processor.process_document(
            document_text, 
            display_name
        )
        
        # Step 2: Extract bullet points if enabled
        refined_sections = []
        if use_bullet_points:
            logger.info("Extracting bullet points from chunks...")
            
            # Extract bullet points from each chunk
            all_bullets = {}
            for chunk_data in processing_result['chunks']:
                chunk = StructuredChunk(
                    chunk_id=chunk_data['chunk_id'],
                    content=chunk_data['content'],
                    source_doc=chunk_data['source_doc'],
                    section_hierarchy=chunk_data['section_hierarchy'],
                    section_title=chunk_data['section_title'],
                    section_type=chunk_data['section_type'],
                    page_numbers=[],
                    metadata=chunk_data['metadata']
                )
                
                bullets = self.bullet_extractor.extract_bullet_points(chunk)
                section_path = chunk_data['metadata']['section_path']
                
                if section_path not in all_bullets:
                    all_bullets[section_path] = []
                all_bullets[section_path].extend(bullets)
            
            # Consolidate and refine bullet points
            logger.info("Consolidating bullet points...")
            consolidated = self.bullet_extractor.consolidate_section_bullets(all_bullets)
            
            # Create section metadata
            section_metadata = {}
            for chunk_data in processing_result['chunks']:
                section_path = chunk_data['metadata']['section_path']
                if section_path not in section_metadata:
                    section_metadata[section_path] = {
                        'title': chunk_data['section_title'],
                        'section_type': chunk_data['section_type']
                    }
            
            # Refine document structure
            refined_sections = self.bullet_extractor.refine_document_structure(
                consolidated, 
                section_metadata
            )
            
            # Create hierarchical summary
            hierarchical_summary = self.bullet_extractor.create_hierarchical_summary(refined_sections)
            
            # Store hierarchical summary
            summary_file = workspace_path / "enhanced_summary.md"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(hierarchical_summary)
            
            logger.info(f"Saved hierarchical summary to {summary_file}")
        
        # Step 3: Prepare enhanced chunks for GraphRAG
        enhanced_chunks = self._prepare_chunks_for_graphrag(
            processing_result, 
            refined_sections if use_bullet_points else None,
            workspace_path
        )
        
        # Save section summaries to workspace cache
        if processing_result.get('section_summaries'):
            section_cache_dir = workspace_path / "cache"
            section_cache_dir.mkdir(exist_ok=True)
            
            section_summary_file = section_cache_dir / "section_summaries.json"
            section_data = {
                "document": display_name,
                "created_at": str(datetime.now()),
                "sections": processing_result['section_summaries']
            }
            
            with open(section_summary_file, 'w', encoding='utf-8') as f:
                json.dump(section_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(processing_result['section_summaries'])} section summaries to workspace cache")
        
        # Step 4: Create metadata for integration
        integration_metadata = {
            'doc_id': doc_id,
            'display_name': display_name,
            'classification': {
                'type': processing_result['classification'].document_type.value,
                'confidence': processing_result['classification'].confidence,
                'characteristics': processing_result['classification'].key_characteristics
            },
            'statistics': processing_result['statistics'],
            'enhanced_features': {
                'structure_aware_chunking': True,
                'bullet_point_extraction': use_bullet_points,
                'section_summaries': len(processing_result['section_summaries']),
                'cross_references': len(processing_result['cross_references'])
            }
        }
        
        # Save metadata
        metadata_file = workspace_path / "enhanced_rag_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(integration_metadata, f, indent=2)
        
        logger.info(f"Enhanced RAG processing completed for {display_name}")
        
        return {
            'metadata': integration_metadata,
            'enhanced_chunks': enhanced_chunks,
            'refined_sections': refined_sections,
            'processing_result': processing_result
        }
    
    def _prepare_chunks_for_graphrag(
        self, 
        processing_result: Dict[str, Any],
        refined_sections: Optional[List[Any]],
        workspace_path: Path
    ) -> List[Dict[str, Any]]:
        """Prepare enhanced chunks for GraphRAG processing"""
        
        # Create enhanced text file for GraphRAG
        enhanced_text_lines = []
        
        if refined_sections:
            # Use refined bullet points
            for section in refined_sections:
                enhanced_text_lines.append(f"\n## {section.section_title}\n")
                enhanced_text_lines.append(f"{section.summary}\n")
                for bullet in section.bullet_points:
                    enhanced_text_lines.append(f"• {bullet}")
                enhanced_text_lines.append("\n")
        else:
            # Use structured chunks
            for chunk in processing_result['chunks']:
                section_path = chunk['metadata']['section_path']
                enhanced_text_lines.append(f"\n## {section_path}\n")
                enhanced_text_lines.append(chunk['content'])
                enhanced_text_lines.append("\n")
        
        # Write enhanced text file
        enhanced_file = workspace_path / "input" / "enhanced_document.txt"
        with open(enhanced_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(enhanced_text_lines))
        
        logger.info(f"Created enhanced document at {enhanced_file}")
        
        # Prepare chunk metadata for embeddings
        if refined_sections:
            return self.bullet_extractor.extract_for_embeddings(refined_sections)
        else:
            return self.processor.create_enhanced_embeddings(processing_result['chunks'])
    
    def update_graphrag_config_for_enhanced_rag(self, workspace_path: Path):
        """Update GraphRAG configuration to use enhanced features"""
        
        settings_file = workspace_path / "settings.yaml"
        
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                settings = yaml.safe_load(f)
            
            # Update chunk settings for enhanced processing
            if 'chunks' in settings:
                settings['chunks']['size'] = 600  # Larger chunks for structured content
                settings['chunks']['overlap'] = 100
            
            # Add enhanced metadata to embeddings
            if 'embeddings' in settings:
                settings['embeddings']['batch_size'] = 8  # Smaller batch for richer content
            
            # Write updated settings
            with open(settings_file, 'w') as f:
                yaml.dump(settings, f, default_flow_style=False)
            
            logger.info("Updated GraphRAG settings for enhanced processing")
    
    def extract_knowledge_graph_data(
        self,
        refined_sections: List[Any]
    ) -> Dict[str, Any]:
        """Extract entities and relationships for knowledge graph"""
        
        if not refined_sections:
            return {'entities': [], 'relationships': []}
        
        return self.bullet_extractor.extract_for_knowledge_graph(refined_sections)