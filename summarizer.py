"""
Two-stage document summarization module for GraphRAG processed documents.
Implements a map-reduce approach with configurable AI models for chunk-level and consolidated summaries.
"""

import pandas as pd
import openai
from pathlib import Path
import os
import logging
from typing import List, Optional, Dict, Tuple
import json
import numpy as np
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkSummary:
    """Represents a chunk summary with embedding capabilities."""
    chunk_id: str
    raw_text: str
    summary_text: str
    summary_vector: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

class DocumentSummarizer:
    """Handles two-stage document summarization with configurable models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the summarizer with OpenAI API configuration."""
        # Use provided API key or fall back to environment variable
        self.api_key = api_key or os.getenv("GRAPHRAG_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set GRAPHRAG_API_KEY or OPENAI_API_KEY environment variable.")
        
        # Configure OpenAI client
        openai.api_key = self.api_key
        
        # Model configuration - using high-quality models as recommended
        self.chunk_model = "gpt-4o-mini"  # For chunk-level summaries (accuracy)
        self.consolidation_model = "o1-mini"  # For final consolidation (reasoning)
        self.embedding_model = "text-embedding-3-small"  # For vector embeddings
        
        # Enable embedding generation by default
        self.generate_embeddings = True
        
        logger.info(f"Initialized DocumentSummarizer with chunk_model: {self.chunk_model}, consolidation_model: {self.consolidation_model}, embedding_model: {self.embedding_model}")
    
    def _get_llm_response(self, prompt: str, model: str = None, temperature: float = 0.0) -> str:
        """Helper function to get a response from an LLM."""
        if model is None:
            model = self.chunk_model
        
        try:
            # Configure parameters based on model type
            if model.startswith("o1"):
                # o1 models don't support temperature or max_tokens parameters
                response = openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}  # o1 models don't use system messages
                    ]
                )
            else:
                # Standard models (gpt-4o, gpt-4o-mini, etc.)
                response = openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant designed to extract factual information and create structured summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=2000
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM API call failed with model {model}: {e}")
            raise
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for the given text."""
        try:
            response = openai.embeddings.create(
                model=self.embedding_model,
                input=text.replace("\n", " "),  # Clean text for embedding
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Embedding API call failed: {e}")
            raise
    
    def _batch_get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts in batch for efficiency."""
        try:
            # Clean texts for embedding
            cleaned_texts = [text.replace("\n", " ") for text in texts]
            
            response = openai.embeddings.create(
                model=self.embedding_model,
                input=cleaned_texts
            )
            
            embeddings = [np.array(item.embedding) for item in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings using {self.embedding_model}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding API call failed: {e}")
            # Fallback to individual embeddings
            logger.info("Falling back to individual embedding generation...")
            embeddings = []
            for text in texts:
                try:
                    embedding = self._get_embedding(text)
                    embeddings.append(embedding)
                except Exception as embed_error:
                    logger.error(f"Failed to generate embedding for text: {embed_error}")
                    # Create zero vector as placeholder
                    embeddings.append(np.zeros(1536))  # text-embedding-3-small dimension
            return embeddings
    
    def summarize_chunk(self, chunk_text: str, chunk_id: str = None) -> str:
        """Generates a factual, one-sentence summary for a text chunk."""
        if not chunk_text or not chunk_text.strip():
            return ""
        
        # Truncate very long chunks to avoid token limits
        max_chunk_length = 3000
        if len(chunk_text) > max_chunk_length:
            chunk_text = chunk_text[:max_chunk_length] + "..."
        
        prompt = f"""
Based *only* on the text provided below, generate a single, concise bullet point that summarizes the main fact or concept.
Do not add any information not present in the text. If the text is empty or meaningless, return an empty string.
Keep the summary under 100 words and focus on the most important factual content.

TEXT:
---
{chunk_text}
---

SUMMARY BULLET POINT (return only the bullet point, starting with '-'):
"""
        
        try:
            summary = self._get_llm_response(prompt, self.chunk_model)
            # Ensure it starts with a dash for consistency
            if summary and not summary.startswith('-'):
                summary = f"- {summary}"
            
            if chunk_id:
                logger.debug(f"Summarized chunk {chunk_id}: {len(chunk_text)} chars -> {len(summary)} chars")
            
            return summary
        except Exception as e:
            logger.error(f"Failed to summarize chunk {chunk_id}: {e}")
            return f"- [Error summarizing chunk: {str(e)}]"
    
    def summarize_chunk_with_embedding(self, chunk_text: str, chunk_id: str = None) -> ChunkSummary:
        """Generate both summary and embedding for a text chunk."""
        if not chunk_text or not chunk_text.strip():
            return ChunkSummary(
                chunk_id=chunk_id or "empty",
                raw_text="",
                summary_text="",
                summary_vector=np.zeros(1536) if self.generate_embeddings else None,
                metadata={"error": "Empty chunk"}
            )
        
        # Generate summary
        summary_text = self.summarize_chunk(chunk_text, chunk_id)
        
        # Generate embedding for the summary
        summary_vector = None
        if self.generate_embeddings and summary_text:
            try:
                summary_vector = self._get_embedding(summary_text)
            except Exception as e:
                logger.warning(f"Failed to generate embedding for chunk {chunk_id}: {e}")
                summary_vector = np.zeros(1536)
        
        return ChunkSummary(
            chunk_id=chunk_id or "unknown",
            raw_text=chunk_text,
            summary_text=summary_text,
            summary_vector=summary_vector,
            metadata={
                "raw_length": len(chunk_text),
                "summary_length": len(summary_text),
                "has_embedding": summary_vector is not None
            }
        )
    
    def consolidate_summaries(self, bullet_points: List[str], document_name: str = "document") -> str:
        """Consolidates a list of bullet points into a structured summary."""
        if not bullet_points:
            return "No content available for summarization."
        
        # Filter out empty bullet points
        valid_bullets = [bp for bp in bullet_points if bp and bp.strip() and bp.strip() != "-"]
        
        if not valid_bullets:
            return "No valid content found for summarization."
        
        bullets_str = "\n".join(valid_bullets)
        
        prompt = f"""
You are a technical writer creating a comprehensive summary. Consolidate the following bullet points into a clean, well-structured, and factual summary.

Follow these rules:
1. Remove redundant or duplicate points
2. Group related points together under logical headings (use ## for section titles)
3. Maintain a factual and neutral tone - do not add any new information
4. The final output should be in Markdown format
5. Create a brief introduction paragraph before the detailed sections
6. Ensure all important information from the bullet points is preserved
7. Organize content logically by topic or theme

Document: {document_name}

RAW BULLET POINTS:
---
{bullets_str}
---

CONSOLIDATED SUMMARY (in Markdown format):
"""
        
        try:
            consolidated = self._get_llm_response(prompt, self.consolidation_model, temperature=0.1)
            logger.info(f"Consolidated {len(valid_bullets)} bullet points into {len(consolidated)} character summary")
            return consolidated
        except Exception as e:
            logger.error(f"Failed to consolidate summaries: {e}")
            # Return a basic fallback summary
            return f"## Summary\n\nDocument contains {len(valid_bullets)} key points:\n\n" + "\n".join(valid_bullets)
    
    def generate_document_summary(self, workspace_path: Path, document_name: str = None) -> str:
        """
        Main function to generate a full document summary from GraphRAG text chunks.
        
        Args:
            workspace_path: Path to the document workspace
            document_name: Optional display name for the document
        
        Returns:
            Consolidated summary in Markdown format
        """
        workspace = Path(workspace_path)
        
        # Find the text units file - try multiple possible locations
        text_units_paths = [
            workspace / "output" / "text_units.parquet",
            workspace / "output" / "artifacts" / "text_units.parquet",
        ]
        
        text_units_path = None
        for path in text_units_paths:
            if path.exists():
                text_units_path = path
                break
        
        if not text_units_path:
            raise FileNotFoundError(f"Could not find text_units.parquet in {workspace}. Checked: {[str(p) for p in text_units_paths]}")

        # Load the parquet file
        try:
            df = pd.read_parquet(text_units_path)
            logger.info(f"Loaded text_units.parquet with {len(df)} rows and columns: {list(df.columns)}")
        except Exception as e:
            raise Exception(f"Failed to read text_units.parquet: {e}")
        
        # Check for required columns - GraphRAG might use different column names
        text_column = None
        for col in ['chunk', 'text', 'content', 'text_unit']:
            if col in df.columns:
                text_column = col
                break
        
        if not text_column:
            raise KeyError(f"No text column found in text_units.parquet. Available columns: {list(df.columns)}")
        
        logger.info(f"Using column '{text_column}' for text content")
        
        # Stage 1: Map - Summarize each chunk
        logger.info(f"Stage 1: Summarizing {len(df)} text chunks...")
        chunk_summaries = []
        
        for idx, row in df.iterrows():
            text_content = row[text_column]
            if isinstance(text_content, str) and text_content.strip():
                chunk_id = row.get('id', f"chunk_{idx}")
                summary = self.summarize_chunk(text_content, chunk_id)
                if summary:
                    chunk_summaries.append(summary)
        
        logger.info(f"Stage 1 completed: Generated {len(chunk_summaries)} chunk summaries")
        
        if not chunk_summaries:
            return "No content could be summarized from this document."
        
        # Stage 2: Reduce - Consolidate all chunk summaries
        logger.info(f"Stage 2: Consolidating {len(chunk_summaries)} chunk summaries...")
        doc_name = document_name or workspace.name
        final_summary = self.consolidate_summaries(chunk_summaries, doc_name)
        
        logger.info(f"Stage 2 completed: Generated final summary ({len(final_summary)} characters)")
        
        return final_summary
    
    def generate_document_summary_with_embeddings(self, workspace_path: Path, document_name: str = None) -> Tuple[str, List[ChunkSummary]]:
        """
        Generate document summary and chunk embeddings.
        
        Returns:
            Tuple of (consolidated_summary, list_of_chunk_summaries_with_embeddings)
        """
        workspace = Path(workspace_path)
        
        # Find the text units file - try multiple possible locations
        text_units_paths = [
            workspace / "output" / "text_units.parquet",
            workspace / "output" / "artifacts" / "text_units.parquet",
        ]
        
        text_units_path = None
        for path in text_units_paths:
            if path.exists():
                text_units_path = path
                break
        
        if not text_units_path:
            raise FileNotFoundError(f"Could not find text_units.parquet in {workspace}. Checked: {[str(p) for p in text_units_paths]}")

        # Load the parquet file
        try:
            df = pd.read_parquet(text_units_path)
            logger.info(f"Loaded text_units.parquet with {len(df)} rows and columns: {list(df.columns)}")
        except Exception as e:
            raise Exception(f"Failed to read text_units.parquet: {e}")
        
        # Check for required columns - GraphRAG might use different column names
        text_column = None
        id_column = None
        
        for col in ['text', 'chunk', 'content', 'text_unit']:
            if col in df.columns:
                text_column = col
                break
        
        for col in ['id', 'human_readable_id', 'chunk_id']:
            if col in df.columns:
                id_column = col
                break
        
        if not text_column:
            raise KeyError(f"No text column found in text_units.parquet. Available columns: {list(df.columns)}")
        
        logger.info(f"Using column '{text_column}' for text content and '{id_column}' for IDs")
        
        # Stage 1: Generate summaries and embeddings for each chunk
        logger.info(f"Stage 1: Generating summaries and embeddings for {len(df)} text chunks...")
        chunk_summaries = []
        summary_texts = []  # For consolidated summary
        
        for idx, row in df.iterrows():
            text_content = row[text_column]
            chunk_id = str(row.get(id_column, f"chunk_{idx}"))
            
            if isinstance(text_content, str) and text_content.strip():
                chunk_summary = self.summarize_chunk_with_embedding(text_content, chunk_id)
                
                if chunk_summary.summary_text:
                    chunk_summaries.append(chunk_summary)
                    summary_texts.append(chunk_summary.summary_text)
        
        logger.info(f"Stage 1 completed: Generated {len(chunk_summaries)} chunk summaries with embeddings")
        
        if not chunk_summaries:
            return "No content could be summarized from this document.", []
        
        # Stage 2: Consolidate all chunk summaries
        logger.info(f"Stage 2: Consolidating {len(summary_texts)} chunk summaries...")
        doc_name = document_name or workspace.name
        final_summary = self.consolidate_summaries(summary_texts, doc_name)
        
        logger.info(f"Stage 2 completed: Generated final summary ({len(final_summary)} characters)")
        
        return final_summary, chunk_summaries


def generate_document_summary(workspace_path: Path, document_name: str = None) -> str:
    """
    Convenience function to generate a document summary.
    This is the main entry point called from app_logic.py
    
    Args:
        workspace_path: Path to the document workspace
        document_name: Optional display name for the document
    
    Returns:
        Consolidated summary in Markdown format
    """
    try:
        summarizer = DocumentSummarizer()
        return summarizer.generate_document_summary(workspace_path, document_name)
    except Exception as e:
        logger.error(f"Failed to generate document summary: {e}")
        raise