"""
Enhanced Query Handler for GraphRAG.
Implements two-stage retrieval: raw text embeddings for precision, graph for context expansion.
"""

import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dependencies
from raw_text_embeddings import RawTextEmbeddingStore
from rate_limiter import create_rate_limited_client


@dataclass
class QueryResult:
    """Represents a query result with both raw text and graph context."""
    chunk_id: str
    text: str
    relevance_score: float
    entities: List[Dict] = None
    relationships: List[Dict] = None
    community_context: Optional[str] = None
    metadata: Dict = None


class EnhancedQueryHandler:
    """Handles enhanced queries using both raw text embeddings and graph knowledge."""
    
    def __init__(self, workspace_path: Path, api_key: str):
        """Initialize the enhanced query handler."""
        self.workspace_path = Path(workspace_path)
        self.api_key = api_key
        
        # Initialize rate-limited client
        self.client = create_rate_limited_client(
            api_key=api_key,
            requests_per_minute=50,
            tokens_per_minute=150000
        )
        
        # Initialize raw text embedding store
        self.raw_text_store = RawTextEmbeddingStore(workspace_path, api_key)
        
        # Graph data paths
        self.entities_path = workspace_path / "output" / "entities.parquet"
        self.relationships_path = workspace_path / "output" / "relationships.parquet"
        self.communities_path = workspace_path / "output" / "communities.parquet"
        
        logger.info(f"Initialized EnhancedQueryHandler for workspace: {workspace_path}")
    
    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for the query text."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=query.replace("\n", " ")
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    def _load_graph_context(self, chunk_ids: List[str]) -> Dict[str, List[Dict]]:
        """Load graph context (entities, relationships) for given chunks."""
        import pandas as pd
        
        context = {
            "entities": [],
            "relationships": [],
            "communities": []
        }
        
        try:
            # Load entities if available
            if self.entities_path.exists():
                entities_df = pd.read_parquet(self.entities_path)
                
                # Filter entities related to our chunks
                # This assumes entities have a reference to text chunks
                if 'text_unit_ids' in entities_df.columns:
                    relevant_entities = []
                    for _, entity in entities_df.iterrows():
                        text_units = entity.get('text_unit_ids', [])
                        if isinstance(text_units, str):
                            import ast
                            try:
                                text_units = ast.literal_eval(text_units)
                            except:
                                text_units = []
                        
                        if any(chunk_id in str(text_units) for chunk_id in chunk_ids):
                            relevant_entities.append({
                                'id': entity.get('id', ''),
                                'name': entity.get('name', ''),
                                'type': entity.get('type', ''),
                                'description': entity.get('description', '')
                            })
                    
                    context['entities'] = relevant_entities[:20]  # Limit to top 20
            
            # Load relationships if available
            if self.relationships_path.exists():
                relationships_df = pd.read_parquet(self.relationships_path)
                
                # Get entity IDs from our relevant entities
                entity_ids = [e['id'] for e in context['entities']]
                
                # Filter relationships involving our entities
                relevant_rels = []
                for _, rel in relationships_df.iterrows():
                    source = rel.get('source', '')
                    target = rel.get('target', '')
                    
                    if source in entity_ids or target in entity_ids:
                        relevant_rels.append({
                            'source': source,
                            'target': target,
                            'description': rel.get('description', ''),
                            'weight': rel.get('weight', 1.0)
                        })
                
                context['relationships'] = relevant_rels[:30]  # Limit to top 30
            
            logger.info(f"Loaded graph context: {len(context['entities'])} entities, "
                       f"{len(context['relationships'])} relationships")
            
        except Exception as e:
            logger.error(f"Failed to load graph context: {e}")
        
        return context
    
    def _expand_query_with_entities(self, query: str, entities: List[Dict]) -> str:
        """Expand query with relevant entity information."""
        if not entities:
            return query
        
        # Add top entities to query context
        entity_names = [e['name'] for e in entities[:5]]
        expanded = f"{query}\n\nRelated entities: {', '.join(entity_names)}"
        
        return expanded
    
    def query(
        self,
        query_text: str,
        document_id: str,
        initial_limit: int = 10,
        final_limit: int = 5,
        use_graph_expansion: bool = True
    ) -> List[QueryResult]:
        """
        Perform enhanced two-stage query:
        1. Initial retrieval using raw text embeddings
        2. Context expansion using graph knowledge
        
        Args:
            query_text: The query string
            document_id: Document to search within
            initial_limit: Number of chunks to retrieve in first stage
            final_limit: Number of final results to return
            use_graph_expansion: Whether to use graph for context expansion
        
        Returns:
            List of QueryResult objects with text and graph context
        """
        try:
            logger.info(f"Processing query: '{query_text[:100]}...'")
            
            # Stage 1: Generate query embedding and search raw text
            query_embedding = self._generate_query_embedding(query_text)
            
            raw_results = self.raw_text_store.search_similar_chunks(
                query_vector=query_embedding,
                limit=initial_limit,
                document_id=document_id
            )
            
            if not raw_results:
                logger.warning("No results found in raw text search")
                return []
            
            logger.info(f"Stage 1: Found {len(raw_results)} raw text matches")
            
            # Stage 2: Load graph context if enabled
            if use_graph_expansion:
                chunk_ids = [r['chunk_id'] for r in raw_results]
                graph_context = self._load_graph_context(chunk_ids)
                
                # If we found entities, we could optionally re-rank or expand search
                if graph_context['entities']:
                    logger.info(f"Stage 2: Expanding with {len(graph_context['entities'])} entities")
                    
                    # Optionally: Create expanded query and search again
                    # expanded_query = self._expand_query_with_entities(query_text, graph_context['entities'])
                    # expanded_embedding = self._generate_query_embedding(expanded_query)
                    # additional_results = self.raw_text_store.search_similar_chunks(...)
            else:
                graph_context = {"entities": [], "relationships": []}
            
            # Combine results
            final_results = []
            for i, raw_result in enumerate(raw_results[:final_limit]):
                # Find entities and relationships for this specific chunk
                chunk_entities = []
                chunk_relationships = []
                
                if use_graph_expansion:
                    # Filter entities for this chunk
                    for entity in graph_context['entities']:
                        # This is a simplified check - in practice, you'd need proper chunk-entity mapping
                        if entity['name'].lower() in raw_result['text'].lower():
                            chunk_entities.append(entity)
                    
                    # Filter relationships for chunk entities
                    chunk_entity_ids = [e['id'] for e in chunk_entities]
                    chunk_relationships = [
                        r for r in graph_context['relationships']
                        if r['source'] in chunk_entity_ids or r['target'] in chunk_entity_ids
                    ]
                
                result = QueryResult(
                    chunk_id=raw_result['chunk_id'],
                    text=raw_result['text'],
                    relevance_score=1.0 - raw_result['score'],  # Convert distance to similarity
                    entities=chunk_entities,
                    relationships=chunk_relationships,
                    metadata={
                        'char_count': raw_result['char_count'],
                        'token_count': raw_result['token_count'],
                        'rank': i + 1
                    }
                )
                final_results.append(result)
            
            logger.info(f"Query complete: returning {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def format_results_for_context(self, results: List[QueryResult]) -> str:
        """Format query results into context string for LLM."""
        if not results:
            return "No relevant context found."
        
        context_parts = []
        
        for i, result in enumerate(results):
            context_part = f"[Chunk {i+1} - Score: {result.relevance_score:.3f}]\n"
            context_part += f"{result.text}\n"
            
            if result.entities:
                entity_list = ", ".join([f"{e['name']} ({e['type']})" for e in result.entities[:5]])
                context_part += f"Entities: {entity_list}\n"
            
            if result.relationships:
                context_part += f"Relationships: {len(result.relationships)} found\n"
            
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def generate_answer(
        self,
        query: str,
        results: List[QueryResult],
        use_o1_model: bool = False
    ) -> str:
        """Generate an answer using the query results as context."""
        if not results:
            return "I couldn't find relevant information to answer your question."
        
        # Format context
        context = self.format_results_for_context(results)
        
        # Select model
        model = "o1-preview" if use_o1_model else "gpt-4o-mini"
        
        # Create prompt
        if use_o1_model:
            # o1 models work better with direct prompts
            prompt = f"""Based on the following context, answer this question: {query}

Context:
{context}

Please provide a comprehensive and accurate answer based only on the information provided."""
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context. "
                               "Only use information from the context to answer questions."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ]
        
        try:
            if use_o1_model:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                    # o1 models don't support temperature parameter
                )
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1500
                )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"Error generating answer: {str(e)}"