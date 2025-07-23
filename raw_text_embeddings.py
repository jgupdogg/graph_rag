"""
Raw Text Embeddings Module for GraphRAG Enhanced Search.
Generates and stores embeddings from raw text chunks (not summaries) for better retrieval accuracy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dependencies
try:
    import lancedb
    import pyarrow as pa
    LANCEDB_AVAILABLE = True
    logger.info("LanceDB dependencies loaded successfully")
except ImportError as e:
    LANCEDB_AVAILABLE = False
    logger.warning(f"LanceDB not available: {e}")

from rate_limiter import RateLimitedOpenAIClient, create_rate_limited_client


@dataclass
class RawTextChunk:
    """Represents a raw text chunk with its embedding."""
    chunk_id: str
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None
    char_count: int = 0
    token_count: int = 0


class RawTextEmbeddingStore:
    """Manages raw text embeddings in LanceDB for precise retrieval."""
    
    def __init__(self, workspace_path: Path, api_key: str):
        """Initialize the raw text embedding store."""
        self.workspace_path = Path(workspace_path)
        self.db_path = self.workspace_path / "output" / "lancedb"
        self.table_name = "raw-text-embeddings"
        
        if not LANCEDB_AVAILABLE:
            raise ImportError("LanceDB dependencies not available. Install with: pip install lancedb pyarrow")
        
        # Initialize rate-limited OpenAI client
        self.client = create_rate_limited_client(
            api_key=api_key,
            requests_per_minute=100,  # Higher limit for embeddings
            tokens_per_minute=200000
        )
        
        # Ensure directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize LanceDB connection
        self.db = lancedb.connect(str(self.db_path))
        
        logger.info(f"Initialized RawTextEmbeddingStore for workspace: {workspace_path}")
    
    def _create_schema(self) -> pa.Schema:
        """Create PyArrow schema for raw text embeddings table."""
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), 1536)),  # text-embedding-3-small
            pa.field("char_count", pa.int32()),
            pa.field("token_count", pa.int32()),
            pa.field("metadata", pa.string()),  # JSON string
            pa.field("created_at", pa.timestamp("s")),
        ])
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4
    
    def _batch_generate_embeddings(self, texts: List[str], batch_size: int = 20) -> List[np.ndarray]:
        """Generate embeddings for multiple texts with batching and rate limiting."""
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        logger.info(f"Generating embeddings for {len(texts)} texts in {total_batches} batches")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
                
                # Clean texts for embedding
                cleaned_batch = [text.replace("\n", " ")[:8000] for text in batch]  # Limit text length
                
                # Generate embeddings with rate limiting
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=cleaned_batch
                )
                
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Log progress
                if batch_num % 5 == 0:
                    stats = self.client.get_stats()
                    logger.info(f"Rate limiter stats: {stats['requests_in_last_minute']} requests/min")
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {batch_num}: {e}")
                # Create zero vectors as fallback
                fallback_embeddings = [np.zeros(1536) for _ in batch]
                embeddings.extend(fallback_embeddings)
        
        return embeddings
    
    def process_and_store_raw_chunks(self, document_id: str) -> bool:
        """Process raw text chunks from GraphRAG output and store with embeddings."""
        try:
            # Find the text units file
            text_units_paths = [
                self.workspace_path / "output" / "text_units.parquet",
                self.workspace_path / "output" / "artifacts" / "text_units.parquet",
            ]
            
            text_units_path = None
            for path in text_units_paths:
                if path.exists():
                    text_units_path = path
                    break
            
            if not text_units_path:
                logger.error(f"Could not find text_units.parquet in {self.workspace_path}")
                return False
            
            # Load text chunks
            df = pd.read_parquet(text_units_path)
            logger.info(f"Loaded {len(df)} text chunks from {text_units_path}")
            
            # Find text column
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
                logger.error(f"No text column found. Available columns: {list(df.columns)}")
                return False
            
            # Extract texts and generate embeddings
            valid_chunks = []
            texts_to_embed = []
            
            for idx, row in df.iterrows():
                text = str(row[text_column])
                chunk_id = str(row.get(id_column, f"chunk_{idx}"))
                
                if text and text.strip():
                    valid_chunks.append({
                        'chunk_id': chunk_id,
                        'text': text,
                        'char_count': len(text),
                        'token_count': self._estimate_tokens(text)
                    })
                    texts_to_embed.append(text)
            
            if not valid_chunks:
                logger.warning("No valid text chunks found")
                return False
            
            logger.info(f"Processing {len(valid_chunks)} valid text chunks")
            
            # Generate embeddings with rate limiting
            embeddings = self._batch_generate_embeddings(texts_to_embed)
            
            # Prepare records for storage
            records = []
            for i, (chunk_data, embedding) in enumerate(zip(valid_chunks, embeddings)):
                record = {
                    "id": f"{document_id}_{chunk_data['chunk_id']}",
                    "chunk_id": chunk_data['chunk_id'],
                    "document_id": document_id,
                    "text": chunk_data['text'],
                    "embedding": embedding.tolist(),
                    "char_count": chunk_data['char_count'],
                    "token_count": chunk_data['token_count'],
                    "metadata": json.dumps({
                        "index": i,
                        "total_chunks": len(valid_chunks)
                    }),
                    "created_at": pd.Timestamp.now()
                }
                records.append(record)
            
            # Store in LanceDB
            df_store = pd.DataFrame(records)
            
            if self.table_name in self.db.table_names():
                # Append to existing table
                table = self.db.open_table(self.table_name)
                table.add(df_store)
                logger.info(f"Added {len(records)} raw text embeddings to existing table")
            else:
                # Create new table
                table = self.db.create_table(self.table_name, df_store, schema=self._create_schema())
                logger.info(f"Created new raw text embeddings table with {len(records)} records")
            
            # Log final stats
            stats = self.client.get_stats()
            logger.info(f"Embedding generation complete. Total tokens used: {stats['total_tokens_used']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process raw chunks: {e}")
            return False
    
    def search_similar_chunks(
        self, 
        query_vector: np.ndarray, 
        limit: int = 10, 
        document_id: Optional[str] = None
    ) -> List[Dict]:
        """Search for similar raw text chunks using vector similarity."""
        try:
            if self.table_name not in self.db.table_names():
                logger.warning(f"Table {self.table_name} does not exist")
                return []
            
            table = self.db.open_table(self.table_name)
            
            # Build query
            query = table.search(query_vector.tolist()).limit(limit)
            
            # Filter by document if specified
            if document_id:
                query = query.where(f"document_id = '{document_id}'")
            
            # Execute search
            results = query.to_pandas()
            
            # Convert to list of dictionaries
            search_results = []
            for _, row in results.iterrows():
                result = {
                    "id": row["id"],
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "text": row["text"],
                    "char_count": row["char_count"],
                    "token_count": row["token_count"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "score": row.get("_distance", 0.0),  # Lower is better
                }
                search_results.append(result)
            
            logger.info(f"Found {len(search_results)} similar raw text chunks")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search raw chunks: {e}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str, document_id: str) -> Optional[Dict]:
        """Retrieve a specific chunk by ID."""
        try:
            if self.table_name not in self.db.table_names():
                return None
            
            table = self.db.open_table(self.table_name)
            
            # Query for specific chunk
            full_id = f"{document_id}_{chunk_id}"
            results = table.search().where(f"id = '{full_id}'").limit(1).to_pandas()
            
            if len(results) > 0:
                row = results.iloc[0]
                return {
                    "id": row["id"],
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "text": row["text"],
                    "char_count": row["char_count"],
                    "token_count": row["token_count"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get chunk by ID: {e}")
            return None
    
    def delete_document_embeddings(self, document_id: str) -> bool:
        """Delete all embeddings for a specific document."""
        try:
            if self.table_name not in self.db.table_names():
                return True
            
            table = self.db.open_table(self.table_name)
            table.delete(f"document_id = '{document_id}'")
            
            logger.info(f"Deleted raw text embeddings for document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document embeddings: {e}")
            return False
    
    def get_table_info(self) -> Dict:
        """Get information about the raw text embeddings table."""
        try:
            if self.table_name not in self.db.table_names():
                return {"exists": False, "count": 0, "documents": []}
            
            table = self.db.open_table(self.table_name)
            df = table.to_pandas()
            
            info = {
                "exists": True,
                "count": len(df),
                "documents": df["document_id"].unique().tolist() if len(df) > 0 else [],
                "total_chars": df["char_count"].sum() if len(df) > 0 else 0,
                "total_tokens": df["token_count"].sum() if len(df) > 0 else 0,
                "avg_chunk_size": df["char_count"].mean() if len(df) > 0 else 0,
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get table info: {e}")
            return {"exists": False, "error": str(e)}