"""
GraphRAG Query Logic for Multi-Document Chat Interface
Handles GraphRAG queries across multiple documents with different search types.
"""

import os
import sys
import subprocess
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import sqlite3
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Represents the result of a GraphRAG query."""
    query: str
    response: str
    search_type: str
    source_documents: List[str]
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = None
    context_info: Optional[Dict[str, Any]] = None

class GraphRAGQueryEngine:
    """Main class for handling GraphRAG queries across multiple documents."""
    
    def __init__(self, db_path: str = "metadata.db"):
        self.db_path = Path(db_path)
        self.workspaces_dir = Path("workspaces")
        self.venv_path = Path(__file__).parent / "venv"
        
    def _extract_context_info(self, raw_response: str) -> Dict[str, Any]:
        """Extract context information from GraphRAG output for debugging."""
        context_info = {
            "entities_found": [],
            "relationships_found": [],
            "text_units_found": [],
            "community_reports_found": [],
            "vector_search_results": [],
            "data_references": [],
            "retrieved_content": [],
            "search_context": {},
            "configuration": {}
        }
        
        try:
            lines = raw_response.split('\n')
            
            # Look for data references in the response
            import re
            data_pattern = r'\[Data: ([^\]]+)\]'
            data_matches = re.findall(data_pattern, raw_response)
            context_info["data_references"] = data_matches
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Look for entity information
                if any(term in line.lower() for term in ["entity", "entities"]) and any(term in line.lower() for term in ["found", "retrieved", "selected", "matching"]):
                    context_info["entities_found"].append(line)
                
                # Look for relationship information
                if any(term in line.lower() for term in ["relationship", "relationships"]) and any(term in line.lower() for term in ["found", "retrieved", "selected", "matching"]):
                    context_info["relationships_found"].append(line)
                
                # Look for text unit information
                if any(term in line.lower() for term in ["text_unit", "text unit", "text chunk", "chunk"]):
                    context_info["text_units_found"].append(line)
                
                # Look for community report information
                if any(term in line.lower() for term in ["community", "report"]):
                    context_info["community_reports_found"].append(line)
                
                # Look for vector search results and similarity scores
                if any(term in line.lower() for term in ["vector", "similarity", "embedding", "score", "distance"]):
                    context_info["vector_search_results"].append(line)
                
                # Look for retrieved content sections
                if any(term in line.lower() for term in ["retrieved", "context", "relevant", "matching"]) and len(line) > 50:
                    # Check if next few lines contain substantial content
                    content_block = []
                    for j in range(i, min(i + 5, len(lines))):
                        if lines[j].strip() and len(lines[j].strip()) > 20:
                            content_block.append(lines[j].strip())
                    if content_block:
                        context_info["retrieved_content"].extend(content_block)
                
                # Look for search context information
                if any(term in line.lower() for term in ["searching", "query", "looking for", "matching"]):
                    context_info["search_context"][f"step_{len(context_info['search_context'])}"] = line
                
                # Look for configuration information
                if any(config_term in line.lower() for config_term in ["vector store", "embedding", "model", "api", "database", "index"]):
                    if ":" in line:
                        key_value = line.split(":", 1)
                        if len(key_value) == 2:
                            key = key_value[0].strip()
                            value = key_value[1].strip()
                            context_info["configuration"][key] = value
            
            # Extract the actual response content for context
            response_start = raw_response.find("SUCCESS:")
            if response_start != -1:
                response_content = raw_response[response_start:]
                # Look for data citations in the response
                citations = re.findall(r'\[Data: ([^\]]+)\]', response_content)
                if citations:
                    context_info["data_references"] = list(set(context_info["data_references"] + citations))
            
            return context_info
            
        except Exception as e:
            logger.error(f"Error extracting context info: {e}")
            return context_info
    
    def _extract_actual_response(self, raw_response: str) -> str:
        """Extract the actual response from GraphRAG output, filtering out logs."""
        try:
            lines = raw_response.split('\n')
            
            # Look for the actual response after configuration logs
            response_lines = []
            found_response = False
            skip_patterns = [
                "INFO: Vector Store Args:",
                "SUCCESS: Local Search Response:",
                "INFO: Loading",
                "INFO: Initializing",
                "INFO: Creating",
                "INFO: Using",
                "INFO: Found",
                "INFO: Processing",
                "INFO: Query",
                "INFO: Embedding",
                "INFO: Searching",
                "INFO: Retrieved",
                "INFO: Generating",
                "INFO: Completed",
                "args:",
                "config:",
                "default_vector_store",
                "type",
                "db_uri",
                "container_name",
                "overwrite",
                "embedding_dimension",
                "collection_name",
                "vector_store_type",
                "completion_delimiter",
                "tuple_delimiter",
                "record_delimiter",
                "\"",
                "{",
                "}",
                "true",
                "false",
                "null"
            ]
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and log lines
                if not line or any(pattern in line for pattern in skip_patterns):
                    continue
                
                # Look for the actual response content
                if not found_response and len(line) > 50:  # Actual responses are typically longer
                    found_response = True
                
                if found_response:
                    response_lines.append(line)
            
            # Join the response lines
            actual_response = '\n'.join(response_lines)
            
            # If we couldn't extract a proper response, return the last substantial line
            if not actual_response or len(actual_response) < 20:
                substantial_lines = [line for line in lines if len(line.strip()) > 50]
                if substantial_lines:
                    actual_response = substantial_lines[-1]
                else:
                    actual_response = raw_response
            
            return actual_response
            
        except Exception as e:
            logger.error(f"Error extracting response: {e}")
            return raw_response
    
    def _ensure_env_files(self, workspaces: List[Tuple[str, Path, str]]):
        """Ensure all workspaces have .env files."""
        config_env_path = Path("graphrag_config") / ".env"
        
        if not config_env_path.exists():
            logger.warning("Main .env file not found at graphrag_config/.env")
            return
        
        for doc_id, workspace_path, display_name in workspaces:
            env_path = workspace_path / ".env"
            if not env_path.exists():
                try:
                    import shutil
                    shutil.copy2(config_env_path, env_path)
                    logger.info(f"Copied .env file to workspace: {workspace_path}")
                except Exception as e:
                    logger.error(f"Failed to copy .env file to {workspace_path}: {e}")
    
    def _get_document_workspaces(self, doc_ids: List[str]) -> List[Tuple[str, Path, str]]:
        """Get workspace paths for the given document IDs."""
        try:
            conn = sqlite3.connect(self.db_path)
            placeholders = ','.join('?' for _ in doc_ids)
            query = f"""
                SELECT id, workspace_path, display_name 
                FROM documents 
                WHERE id IN ({placeholders}) AND status = 'COMPLETED'
            """
            
            cursor = conn.cursor()
            cursor.execute(query, doc_ids)
            results = cursor.fetchall()
            conn.close()
            
            # Convert to Path objects and validate
            workspaces = []
            for doc_id, workspace_path, display_name in results:
                workspace = Path(workspace_path)
                if workspace.exists():
                    workspaces.append((doc_id, workspace, display_name))
                else:
                    logger.warning(f"Workspace not found for document {doc_id}: {workspace}")
            
            return workspaces
            
        except Exception as e:
            logger.error(f"Error getting document workspaces: {e}")
            return []
    
    def _run_graphrag_query(self, workspace_path: Path, query: str, method: str = "local") -> Dict[str, Any]:
        """Run a GraphRAG query on a single workspace."""
        try:
            # Prepare the command
            graphrag_cmd = self.venv_path / "bin" / "graphrag"
            if not graphrag_cmd.exists():
                # Fallback to system graphrag
                graphrag_cmd = "graphrag"
            
            command = [
                str(graphrag_cmd),
                "query",
                "--root", str(workspace_path),
                "--method", method,
                "--query", query
                # Note: --verbose flag may not be available in all GraphRAG versions
            ]
            
            # Run the command
            logger.info(f"Running GraphRAG query: {' '.join(command)}")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent),
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                # Keep the raw response for debugging
                raw_response = result.stdout.strip()
                actual_response = self._extract_actual_response(raw_response)
                
                # Extract context information for debugging
                context_info = self._extract_context_info(raw_response)
                
                return {
                    "success": True,
                    "response": actual_response,
                    "raw_response": raw_response,
                    "context_info": context_info,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "response": None,
                    "raw_response": result.stderr.strip(),
                    "context_info": {},
                    "error": result.stderr.strip() or "Unknown error"
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "response": None,
                "raw_response": "Query timed out",
                "context_info": {},
                "error": "Query timed out after 5 minutes"
            }
        except Exception as e:
            return {
                "success": False,
                "response": None,
                "raw_response": str(e),
                "context_info": {},
                "error": str(e)
            }
    
    def _merge_context_info(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge context information from multiple responses."""
        merged_context = {
            "entities_found": [],
            "relationships_found": [],
            "text_units_found": [],
            "community_reports_found": [],
            "vector_search_results": [],
            "configuration": {}
        }
        
        for response in responses:
            context_info = response.get("context_info", {})
            for key in merged_context.keys():
                if key in context_info:
                    if isinstance(merged_context[key], list):
                        merged_context[key].extend(context_info[key])
                    elif isinstance(merged_context[key], dict):
                        merged_context[key].update(context_info[key])
        
        return merged_context
    
    def _merge_query_responses(self, responses: List[Dict[str, Any]], query: str, method: str) -> str:
        """Merge multiple query responses into a single coherent response."""
        successful_responses = [r for r in responses if r.get("success") and r.get("response")]
        
        if not successful_responses:
            return "No relevant information found in the selected documents."
        
        if len(successful_responses) == 1:
            return successful_responses[0]["response"]
        
        # For multiple responses, create a merged response
        merged_response = f"Based on the analysis of {len(successful_responses)} documents:\n\n"
        
        for i, response in enumerate(successful_responses, 1):
            doc_name = response.get("document_name", f"Document {i}")
            merged_response += f"## From {doc_name}:\n{response['response']}\n\n"
        
        return merged_response.strip()
    
    def query_documents(self, doc_ids: List[str], query: str, method: str = "local") -> QueryResult:
        """Query multiple documents and return a merged response."""
        if not doc_ids:
            return QueryResult(
                query=query,
                response="No documents selected for querying.",
                search_type=method,
                source_documents=[],
                success=False,
                error_message="No documents selected"
            )
        
        # Get workspace paths
        workspaces = self._get_document_workspaces(doc_ids)
        if not workspaces:
            return QueryResult(
                query=query,
                response="No valid workspaces found for the selected documents.",
                search_type=method,
                source_documents=[],
                success=False,
                error_message="No valid workspaces found"
            )
        
        # Ensure all workspaces have .env file
        self._ensure_env_files(workspaces)
        
        # Run queries on each workspace
        responses = []
        source_documents = []
        
        for doc_id, workspace_path, display_name in workspaces:
            logger.info(f"Querying document: {display_name} ({doc_id})")
            
            response = self._run_graphrag_query(workspace_path, query, method)
            response["document_name"] = display_name
            response["document_id"] = doc_id
            responses.append(response)
            
            if response["success"]:
                source_documents.append(display_name)
        
        # Check if any queries succeeded
        successful_responses = [r for r in responses if r.get("success")]
        if not successful_responses:
            error_messages = [r.get("error", "Unknown error") for r in responses]
            return QueryResult(
                query=query,
                response="Failed to query documents.",
                search_type=method,
                source_documents=source_documents,
                success=False,
                error_message="; ".join(error_messages)
            )
        
        # Merge responses
        merged_response = self._merge_query_responses(successful_responses, query, method)
        
        # Collect context information from all responses
        merged_context_info = self._merge_context_info(responses)
        
        # Collect raw responses for debugging
        raw_responses = [r.get("raw_response", "") for r in responses]
        
        return QueryResult(
            query=query,
            response=merged_response,
            search_type=method,
            source_documents=source_documents,
            success=True,
            metadata={
                "total_documents": len(workspaces),
                "successful_queries": len(successful_responses),
                "failed_queries": len(responses) - len(successful_responses)
            },
            raw_response="\n\n--- DOCUMENT SEPARATOR ---\n\n".join(raw_responses),
            context_info=merged_context_info
        )
    
    def global_search(self, doc_ids: List[str], query: str) -> QueryResult:
        """Perform a global search across selected documents."""
        return self.query_documents(doc_ids, query, method="global")
    
    def local_search(self, doc_ids: List[str], query: str) -> QueryResult:
        """Perform a local search across selected documents."""
        return self.query_documents(doc_ids, query, method="local")
    
    def drift_search(self, doc_ids: List[str], query: str) -> QueryResult:
        """Perform a drift search across selected documents."""
        return self.query_documents(doc_ids, query, method="drift")
    
    def basic_search(self, doc_ids: List[str], query: str) -> QueryResult:
        """Perform a basic search across selected documents."""
        return self.query_documents(doc_ids, query, method="basic")
    
    def get_supported_methods(self) -> List[str]:
        """Get list of supported query methods."""
        return ["local", "global", "drift", "basic"]
    
    def inspect_document_knowledge_base(self, doc_id: str) -> Dict[str, Any]:
        """Inspect what's actually in a document's GraphRAG knowledge base."""
        try:
            workspaces = self._get_document_workspaces([doc_id])
            if not workspaces:
                return {"error": "Document workspace not found"}
            
            doc_id, workspace_path, display_name = workspaces[0]
            output_path = workspace_path / "output"
            
            inspection = {
                "document_name": display_name,
                "workspace_path": str(workspace_path),
                "entities": [],
                "relationships": [],
                "text_units": [],
                "community_reports": [],
                "vector_store_info": {},
                "file_sizes": {}
            }
            
            # Check what files exist
            for file_name in ["entities.parquet", "relationships.parquet", "text_units.parquet", "community_reports.parquet"]:
                file_path = output_path / file_name
                if file_path.exists():
                    inspection["file_sizes"][file_name] = file_path.stat().st_size
                    
                    # Load and inspect the data
                    try:
                        import pandas as pd
                        df = pd.read_parquet(file_path)
                        
                        if file_name == "entities.parquet":
                            inspection["entities"] = [
                                {"title": row.get("title", ""), "type": row.get("type", ""), "description": row.get("description", "")[:200]}
                                for _, row in df.head(10).iterrows()
                            ]
                        elif file_name == "relationships.parquet":
                            inspection["relationships"] = [
                                {"source": row.get("source", ""), "target": row.get("target", ""), "description": row.get("description", "")[:200]}
                                for _, row in df.head(10).iterrows()
                            ]
                        elif file_name == "text_units.parquet":
                            inspection["text_units"] = [
                                {"id": row.get("id", ""), "text": row.get("text", "")[:300]}
                                for _, row in df.head(5).iterrows()
                            ]
                        elif file_name == "community_reports.parquet":
                            inspection["community_reports"] = [
                                {"title": row.get("title", ""), "summary": row.get("summary", "")[:300]}
                                for _, row in df.head(5).iterrows()
                            ]
                    except Exception as e:
                        inspection[file_name.replace(".parquet", "_error")] = str(e)
            
            # Check vector store
            vector_store_path = output_path / "lancedb"
            if vector_store_path.exists():
                inspection["vector_store_info"]["path"] = str(vector_store_path)
                inspection["vector_store_info"]["tables"] = [d.name for d in vector_store_path.iterdir() if d.is_dir()]
            
            return inspection
            
        except Exception as e:
            return {"error": str(e)}
    
    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """Validate a query string."""
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        if len(query.strip()) < 3:
            return False, "Query must be at least 3 characters long"
        
        if len(query) > 1000:
            return False, "Query is too long (max 1000 characters)"
        
        return True, None

class ChatHistory:
    """Manages chat history for document conversations."""
    
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
    
    def add_query(self, query: str, result: QueryResult, doc_ids: List[str]):
        """Add a query and response to the history."""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": result.response,
            "search_type": result.search_type,
            "success": result.success,
            "source_documents": result.source_documents,
            "document_ids": doc_ids.copy(),
            "error_message": result.error_message,
            "metadata": result.metadata
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the complete chat history."""
        return self.history.copy()
    
    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent chat history."""
        return self.history[-limit:] if self.history else []
    
    def clear_history(self):
        """Clear the chat history."""
        self.history.clear()
    
    def get_context_for_query(self, limit: int = 3) -> str:
        """Get context from recent queries for better continuity."""
        if not self.history:
            return ""
        
        recent = self.get_recent_history(limit)
        context_parts = []
        
        for entry in recent:
            if entry["success"]:
                context_parts.append(f"Previous Q: {entry['query']}")
                context_parts.append(f"Previous A: {entry['response'][:200]}...")
        
        return "\n".join(context_parts) if context_parts else ""

# Global instances
query_engine = GraphRAGQueryEngine()
chat_history = ChatHistory()

# Convenience functions
def query_documents(doc_ids: List[str], query: str, method: str = "local") -> QueryResult:
    """Query multiple documents with the specified method."""
    return query_engine.query_documents(doc_ids, query, method)

def global_search(doc_ids: List[str], query: str) -> QueryResult:
    """Perform a global search across selected documents."""
    return query_engine.global_search(doc_ids, query)

def local_search(doc_ids: List[str], query: str) -> QueryResult:
    """Perform a local search across selected documents."""
    return query_engine.local_search(doc_ids, query)

def drift_search(doc_ids: List[str], query: str) -> QueryResult:
    """Perform a drift search across selected documents."""
    return query_engine.drift_search(doc_ids, query)

def basic_search(doc_ids: List[str], query: str) -> QueryResult:
    """Perform a basic search across selected documents."""
    return query_engine.basic_search(doc_ids, query)

def get_supported_methods() -> List[str]:
    """Get list of supported query methods."""
    return query_engine.get_supported_methods()

def validate_query(query: str) -> Tuple[bool, Optional[str]]:
    """Validate a query string."""
    return query_engine.validate_query(query)

def add_to_chat_history(query: str, result: QueryResult, doc_ids: List[str]):
    """Add a query and response to the chat history."""
    chat_history.add_query(query, result, doc_ids)

def get_chat_history() -> List[Dict[str, Any]]:
    """Get the complete chat history."""
    return chat_history.get_history()

def clear_chat_history():
    """Clear the chat history."""
    chat_history.clear_history()

def inspect_document_knowledge_base(doc_id: str) -> Dict[str, Any]:
    """Inspect what's in a document's knowledge base."""
    return query_engine.inspect_document_knowledge_base(doc_id)