"""
Enhanced GraphRAG Integration
Connects the enhanced document processor with GraphRAG workflow
"""

import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any
import yaml
import json
import os

from enhanced_document_processor import (
    EnhancedDocumentProcessor, 
    EnhancedChunk, 
    DocumentClassification,
    load_enhanced_processor_config
)
from document_structure_parser import extract_document_structure

logger = logging.getLogger(__name__)


class EnhancedGraphRAGWorkflow:
    """Enhanced GraphRAG workflow with AI-driven document processing"""
    
    def __init__(self, workspace_path: Path, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced GraphRAG workflow
        
        Args:
            workspace_path: Path to GraphRAG workspace
            config: Optional configuration override
        """
        self.workspace_path = Path(workspace_path)
        self.input_dir = self.workspace_path / "input"
        self.output_dir = self.workspace_path / "output"
        self.cache_dir = self.workspace_path / "cache"
        
        # Ensure directories exist
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config)
        
        # Initialize enhanced processor
        self.processor = EnhancedDocumentProcessor(self.config)
        
        # Track processing results
        self.processing_results = {}
    
    def _load_config(self, config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load configuration from settings.yaml and apply overrides"""
        settings_path = self.workspace_path / "settings.yaml"
        
        # Load base configuration
        config = load_enhanced_processor_config(settings_path)
        
        # Apply API key from environment or settings
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GRAPHRAG_API_KEY")
        if api_key:
            config['api_key'] = api_key
        
        # Apply overrides
        if config_override:
            config.update(config_override)
        
        return config
    
    def process_all_documents(self) -> pd.DataFrame:
        """
        Process all documents in the input directory with enhanced processing
        
        Returns:
            DataFrame with enhanced chunks ready for GraphRAG
        """
        logger.info(f"Starting enhanced document processing for workspace: {self.workspace_path}")
        
        all_enhanced_chunks = []
        document_classifications = {}
        
        # Process each text file in input directory
        for text_file in self.input_dir.glob("*.txt"):
            logger.info(f"Processing document: {text_file.name}")
            
            try:
                # Read document text
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Prepare metadata
                metadata = {
                    'filename': text_file.name,
                    'file_size': len(text),
                    'file_path': str(text_file)
                }
                
                # Estimate section count for metadata
                sections, _, _ = extract_document_structure(text)
                metadata['section_count'] = len(sections)
                
                # Process document with enhanced processor
                enhanced_chunks, doc_classification = self.processor.process_document(text, metadata)
                
                # Store results
                document_classifications[text_file.stem] = doc_classification
                
                # Convert chunks to DataFrame-compatible format
                for chunk in enhanced_chunks:
                    chunk_dict = chunk.to_dict()
                    chunk_dict['document'] = text_file.stem
                    chunk_dict['document_ids'] = [text_file.stem]
                    chunk_dict['source_file'] = str(text_file)
                    all_enhanced_chunks.append(chunk_dict)
                
                logger.info(f"Processed {text_file.name}: {len(enhanced_chunks)} enhanced chunks created")
                
            except Exception as e:
                logger.error(f"Failed to process {text_file.name}: {e}")
                continue
        
        # Create DataFrame
        if all_enhanced_chunks:
            df = pd.DataFrame(all_enhanced_chunks)
            
            # Save enhanced chunks
            enhanced_chunks_path = self.output_dir / "enhanced_text_units.parquet"
            df.to_parquet(enhanced_chunks_path, index=False)
            logger.info(f"Saved {len(df)} enhanced chunks to {enhanced_chunks_path}")
            
            # Save document classifications
            classifications_path = self.output_dir / "document_classifications.json"
            with open(classifications_path, 'w') as f:
                json.dump(
                    {doc: {
                        'document_type': cls.document_type,
                        'confidence': cls.confidence,
                        'characteristics': cls.characteristics,
                        'reasoning': cls.reasoning
                    } for doc, cls in document_classifications.items()},
                    f, indent=2
                )
            
            # Store processing results
            self.processing_results = {
                'total_chunks': len(df),
                'documents_processed': len(document_classifications),
                'document_classifications': document_classifications,
                'enhanced_chunks_path': enhanced_chunks_path,
                'classifications_path': classifications_path
            }
            
            return df
        
        else:
            logger.warning("No documents were successfully processed")
            return pd.DataFrame()
    
    def create_graphrag_compatible_chunks(self, enhanced_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create GraphRAG-compatible text_units.parquet from enhanced chunks
        
        Args:
            enhanced_df: DataFrame with enhanced chunks
            
        Returns:
            GraphRAG-compatible DataFrame
        """
        logger.info("Creating GraphRAG-compatible chunks")
        
        # Map enhanced chunks to GraphRAG text_units format
        graphrag_chunks = []
        
        for _, row in enhanced_df.iterrows():
            # GraphRAG text_units format
            text_unit = {
                'id': row['id'],
                'text': row['text'],  # This already includes enhanced context
                'n_tokens': row['n_tokens'],
                'document_ids': row['document_ids'],
                'entity_ids': [],  # Will be populated by GraphRAG
                'relationship_ids': [],  # Will be populated by GraphRAG
                
                # Enhanced metadata (preserved for compatibility)
                'section_title': row.get('section_title', ''),
                'section_path': row.get('section_path', ''),
                'section_level': row.get('section_level', 0),
                'document_type': row.get('document_type', ''),
                'semantic_tags': row.get('semantic_tags', ''),
            }
            
            graphrag_chunks.append(text_unit)
        
        # Create DataFrame
        graphrag_df = pd.DataFrame(graphrag_chunks)
        
        # Save as text_units.parquet for GraphRAG
        text_units_path = self.output_dir / "text_units.parquet"
        graphrag_df.to_parquet(text_units_path, index=False)
        logger.info(f"Saved GraphRAG-compatible text units to {text_units_path}")
        
        return graphrag_df
    
    def update_graphrag_documents(self, enhanced_df: pd.DataFrame) -> pd.DataFrame:
        """
        Update or create documents.parquet with enhanced metadata
        
        Args:
            enhanced_df: DataFrame with enhanced chunks
            
        Returns:
            Documents DataFrame
        """
        logger.info("Creating enhanced documents metadata")
        
        # Group by document to create document-level metadata
        documents = []
        
        for document_name in enhanced_df['document'].unique():
            doc_chunks = enhanced_df[enhanced_df['document'] == document_name]
            
            # Get document classification
            doc_classification = None
            if document_name in self.processing_results.get('document_classifications', {}):
                doc_classification = self.processing_results['document_classifications'][document_name]
            
            # Create document metadata
            document_meta = {
                'id': document_name,
                'title': document_name,
                'raw_content': '',  # Would need to be loaded from source
                'summary': f"Document classified as {doc_classification.document_type if doc_classification else 'Unknown'} with {len(doc_chunks)} enhanced chunks",
                
                # Enhanced metadata
                'document_type': doc_classification.document_type if doc_classification else 'Unknown',
                'classification_confidence': doc_classification.confidence if doc_classification else 0.0,
                'total_chunks': len(doc_chunks),
                'total_tokens': doc_chunks['n_tokens'].sum(),
                'section_count': len(doc_chunks['section_title'].unique()),
                'processing_strategy': doc_classification.processing_strategy if doc_classification else 'default'
            }
            
            documents.append(document_meta)
        
        # Create DataFrame
        documents_df = pd.DataFrame(documents)
        
        # Save documents metadata
        documents_path = self.output_dir / "documents.parquet"
        documents_df.to_parquet(documents_path, index=False)
        logger.info(f"Saved enhanced documents metadata to {documents_path}")
        
        return documents_df
    
    def run_enhanced_processing(self) -> Dict[str, Any]:
        """
        Run the complete enhanced processing workflow
        
        Returns:
            Dictionary with processing results and file paths
        """
        logger.info("Starting enhanced GraphRAG processing workflow")
        
        try:
            # Step 1: Process all documents with AI enhancement
            enhanced_df = self.process_all_documents()
            
            if enhanced_df.empty:
                raise ValueError("No documents were successfully processed")
            
            # Step 2: Create GraphRAG-compatible chunks
            graphrag_df = self.create_graphrag_compatible_chunks(enhanced_df)
            
            # Step 3: Update documents metadata
            documents_df = self.update_graphrag_documents(enhanced_df)
            
            # Step 4: Generate processing report
            report = self._generate_processing_report(enhanced_df)
            
            results = {
                'status': 'success',
                'enhanced_chunks': len(enhanced_df),
                'graphrag_chunks': len(graphrag_df),
                'documents': len(documents_df),
                'files_created': {
                    'enhanced_text_units': str(self.output_dir / "enhanced_text_units.parquet"),
                    'text_units': str(self.output_dir / "text_units.parquet"),
                    'documents': str(self.output_dir / "documents.parquet"),
                    'classifications': str(self.output_dir / "document_classifications.json"),
                    'processing_report': str(self.output_dir / "processing_report.json")
                },
                'report': report
            }
            
            # Save processing report
            with open(self.output_dir / "processing_report.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info("Enhanced processing workflow completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Enhanced processing workflow failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'enhanced_chunks': 0,
                'graphrag_chunks': 0,
                'documents': 0
            }
    
    def _generate_processing_report(self, enhanced_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate detailed processing report"""
        
        # Document type distribution
        doc_type_counts = enhanced_df['document_type'].value_counts().to_dict()
        
        # Chunk size statistics
        chunk_stats = {
            'total_chunks': len(enhanced_df),
            'avg_tokens_per_chunk': enhanced_df['n_tokens'].mean(),
            'min_tokens_per_chunk': enhanced_df['n_tokens'].min(),
            'max_tokens_per_chunk': enhanced_df['n_tokens'].max(),
            'total_tokens': enhanced_df['n_tokens'].sum()
        }
        
        # Section distribution
        section_stats = {
            'total_sections': enhanced_df['section_title'].nunique(),
            'avg_chunks_per_section': len(enhanced_df) / enhanced_df['section_title'].nunique() if enhanced_df['section_title'].nunique() > 0 else 0
        }
        
        # Semantic tags analysis
        all_tags = []
        for tags_str in enhanced_df['semantic_tags'].dropna():
            if tags_str:
                all_tags.extend([tag.strip() for tag in tags_str.split(',')])
        
        from collections import Counter
        top_tags = dict(Counter(all_tags).most_common(10))
        
        return {
            'document_type_distribution': doc_type_counts,
            'chunk_statistics': chunk_stats,
            'section_statistics': section_stats,
            'top_semantic_tags': top_tags,
            'documents_processed': enhanced_df['document'].nunique(),
            'processing_timestamp': pd.Timestamp.now().isoformat()
        }


def run_enhanced_graphrag_workflow(workspace_path: Path, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to run the enhanced GraphRAG workflow
    
    Args:
        workspace_path: Path to GraphRAG workspace
        config: Optional configuration override
        
    Returns:
        Processing results dictionary
    """
    workflow = EnhancedGraphRAGWorkflow(workspace_path, config)
    return workflow.run_enhanced_processing()


# Integration with existing GraphRAG pipeline
def integrate_with_existing_pipeline(workspace_path: Path):
    """
    Integrate enhanced processing with existing GraphRAG pipeline
    This function can be called before running the standard GraphRAG indexing
    """
    logger.info(f"Integrating enhanced processing with GraphRAG pipeline at {workspace_path}")
    
    # Run enhanced processing
    results = run_enhanced_graphrag_workflow(workspace_path)
    
    if results['status'] == 'success':
        logger.info(f"Enhanced processing completed. Ready for GraphRAG indexing.")
        logger.info(f"Created {results['enhanced_chunks']} enhanced chunks from {results['documents']} documents")
        
        # The text_units.parquet file is now ready for GraphRAG to use
        return True
    else:
        logger.error(f"Enhanced processing failed: {results.get('error', 'Unknown error')}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python enhanced_graphrag_integration.py <workspace_path>")
        sys.exit(1)
    
    workspace_path = Path(sys.argv[1])
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run enhanced workflow
    results = run_enhanced_graphrag_workflow(workspace_path)
    
    if results['status'] == 'success':
        print(f"✅ Enhanced processing completed successfully!")
        print(f"📊 Created {results['enhanced_chunks']} enhanced chunks from {results['documents']} documents")
        print(f"📁 Files created:")
        for name, path in results['files_created'].items():
            print(f"   - {name}: {path}")
    else:
        print(f"❌ Enhanced processing failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)