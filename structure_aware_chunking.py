"""
Structure-Aware Chunking Strategy for GraphRAG
Chunks text while respecting document structure and section boundaries
"""

import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StructureAwareChunk:
    """A chunk that preserves its structural context"""
    text: str
    chunk_id: str
    section_metadata: Dict[str, any]
    start_pos: int
    end_pos: int
    token_count: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame"""
        return {
            'text': self.text,
            'id': self.chunk_id,
            'n_tokens': self.token_count,
            'metadata': json.dumps(self.section_metadata),
            'section_path': ' > '.join(self.section_metadata.get('section_path', [])),
            'section_title': self.section_metadata.get('section_title', ''),
            'section_level': self.section_metadata.get('section_level', 0)
        }


class StructureAwareChunker:
    """Chunks documents while preserving structural boundaries"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_with_structure(self, text: str, structure_json_path: Path) -> List[StructureAwareChunk]:
        """
        Chunk text using structural information
        
        Args:
            text: The document text (with section markers)
            structure_json_path: Path to the JSON file containing section metadata
            
        Returns:
            List of StructureAwareChunk objects
        """
        # Load section metadata
        if structure_json_path.exists():
            with open(structure_json_path, 'r', encoding='utf-8') as f:
                structure_data = json.load(f)
                sections = structure_data.get('sections', [])
        else:
            logger.warning(f"Structure file not found: {structure_json_path}")
            sections = []
        
        # Parse section markers from text
        chunks = []
        current_section = None
        lines = text.split('\n')
        current_text = []
        current_pos = 0
        
        for line in lines:
            # Check for section marker
            if line.startswith('[SECTION:') and line.endswith(']'):
                # Process accumulated text
                if current_text and current_section:
                    section_chunks = self._chunk_section(
                        '\n'.join(current_text),
                        current_section,
                        current_pos
                    )
                    chunks.extend(section_chunks)
                
                # Extract section path from marker
                section_path_str = line[9:-1]  # Remove [SECTION: and ]
                section_parts = [p.strip() for p in section_path_str.split(' > ')]
                
                # Find matching section metadata
                current_section = self._find_section_metadata(sections, section_parts)
                current_text = []
            else:
                current_text.append(line)
            
            current_pos += len(line) + 1
        
        # Process any remaining text
        if current_text:
            if current_section:
                section_chunks = self._chunk_section(
                    '\n'.join(current_text),
                    current_section,
                    current_pos - sum(len(l) + 1 for l in current_text)
                )
                chunks.extend(section_chunks)
            else:
                # No section metadata, create generic chunks
                generic_chunks = self._chunk_text_basic(
                    '\n'.join(current_text),
                    {'section_title': 'Document', 'section_path': ['Document'], 'section_level': 0},
                    current_pos - sum(len(l) + 1 for l in current_text)
                )
                chunks.extend(generic_chunks)
        
        return chunks
    
    def _find_section_metadata(self, sections: List[Dict], section_path: List[str]) -> Dict:
        """Find section metadata matching the given path"""
        for section in sections:
            if section.get('section_path', []) == section_path:
                return section
        
        # Return default metadata if not found
        return {
            'section_title': section_path[-1] if section_path else 'Unknown',
            'section_path': section_path,
            'section_level': len(section_path)
        }
    
    def _chunk_section(self, text: str, section_metadata: Dict, start_pos: int) -> List[StructureAwareChunk]:
        """Chunk a single section, respecting boundaries"""
        # For small sections, keep them intact
        estimated_tokens = len(text.split()) * 1.3  # Rough token estimate
        
        if estimated_tokens <= self.chunk_size * 1.5:
            # Keep section intact
            return [StructureAwareChunk(
                text=text.strip(),
                chunk_id=f"chunk_{start_pos}",
                section_metadata=section_metadata,
                start_pos=start_pos,
                end_pos=start_pos + len(text),
                token_count=int(estimated_tokens)
            )]
        else:
            # Break into chunks with overlap
            return self._chunk_text_basic(text, section_metadata, start_pos)
    
    def _chunk_text_basic(self, text: str, section_metadata: Dict, start_pos: int) -> List[StructureAwareChunk]:
        """Basic chunking with overlap for longer sections"""
        chunks = []
        words = text.split()
        
        # Rough token-to-word ratio
        words_per_chunk = int(self.chunk_size / 1.3)
        words_overlap = int(self.overlap / 1.3)
        
        idx = 0
        while idx < len(words):
            # Get chunk words
            end_idx = min(idx + words_per_chunk, len(words))
            chunk_words = words[idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            # Create chunk with metadata
            chunk = StructureAwareChunk(
                text=chunk_text,
                chunk_id=f"chunk_{start_pos + idx}",
                section_metadata=section_metadata,
                start_pos=start_pos + len(' '.join(words[:idx])),
                end_pos=start_pos + len(' '.join(words[:end_idx])),
                token_count=len(chunk_words) * 1.3  # Rough estimate
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            idx += words_per_chunk - words_overlap
            
            # Avoid tiny last chunks
            if idx < len(words) and len(words) - idx < words_overlap:
                break
        
        return chunks


def create_structured_chunks_dataframe(
    input_dir: Path,
    output_path: Path,
    chunk_size: int = 500,
    overlap: int = 50
) -> pd.DataFrame:
    """
    Create a DataFrame of structured chunks from all documents in input directory
    
    Args:
        input_dir: Directory containing text files and structure JSON files
        output_path: Path to save the chunks DataFrame
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens
        
    Returns:
        DataFrame with structured chunks
    """
    chunker = StructureAwareChunker(chunk_size, overlap)
    all_chunks = []
    
    # Process each text file
    for text_file in input_dir.glob("*.txt"):
        logger.info(f"Processing {text_file.name}")
        
        # Look for corresponding structure file
        structure_file = text_file.parent / f"{text_file.stem}_structure.json"
        
        # Read text
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Chunk with structure
        chunks = chunker.chunk_with_structure(text, structure_file)
        
        # Add document reference
        for chunk in chunks:
            chunk_dict = chunk.to_dict()
            chunk_dict['document'] = text_file.stem
            chunk_dict['document_ids'] = [text_file.stem]
            all_chunks.append(chunk_dict)
    
    # Create DataFrame
    df = pd.DataFrame(all_chunks)
    
    # Save to parquet
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} chunks to {output_path}")
    
    return df


def integrate_with_graphrag(workspace_path: Path):
    """
    Integrate structure-aware chunking with GraphRAG workflow
    
    This function can be called after document processing to create
    structure-aware chunks that GraphRAG can use.
    """
    input_dir = workspace_path / "input"
    output_dir = workspace_path / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Load chunking settings from GraphRAG config
    settings_path = workspace_path / "settings.yaml"
    chunk_size = 500
    overlap = 50
    
    if settings_path.exists():
        import yaml
        with open(settings_path, 'r') as f:
            settings = yaml.safe_load(f)
            chunk_config = settings.get('chunks', {})
            chunk_size = chunk_config.get('size', 500)
            overlap = chunk_config.get('overlap', 50)
    
    # Create structured chunks
    chunks_path = output_dir / "structured_chunks.parquet"
    df = create_structured_chunks_dataframe(
        input_dir,
        chunks_path,
        chunk_size,
        overlap
    )
    
    return df