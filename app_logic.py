"""
Backend logic for GraphRAG multi-document processing.
Handles database operations, document processing, and graph merging.
"""

import sqlite3
import subprocess
import pandas as pd
import shutil
import uuid
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
DB_PATH = Path("metadata.db")
WORKSPACES_DIR = Path("workspaces")
GRAPHRAG_CONFIG_DIR = Path("graphrag_config")
CURRENT_GRAPHRAG_DIR = Path("graphrag")
VENV_GRAPHRAG_PATH = Path(__file__).parent / "venv" / "bin" / "graphrag"
PROJECT_ROOT = Path(__file__).parent.resolve()

class DBManager:
    """Context manager for SQLite database operations."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
    
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        return self.cursor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.conn.rollback()
            logger.error(f"Database error: {exc_val}")
        else:
            self.conn.commit()
        self.conn.close()

class DocumentStatus:
    """Document processing status constants."""
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"

class GraphRAGProcessor:
    """Main class for handling GraphRAG multi-document processing."""
    
    def __init__(self):
        self.db_manager = DBManager(DB_PATH)
        self.init_db()
        self._ensure_directories()
        self.fix_workspace_paths()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        WORKSPACES_DIR.mkdir(exist_ok=True)
        if not GRAPHRAG_CONFIG_DIR.exists() and CURRENT_GRAPHRAG_DIR.exists():
            logger.info("Moving current graphrag/ to graphrag_config/")
            shutil.move(str(CURRENT_GRAPHRAG_DIR), str(GRAPHRAG_CONFIG_DIR))
    
    def init_db(self):
        """Initialize SQLite database and create tables if they don't exist."""
        with self.db_manager as cursor:
            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('UPLOADED', 'PROCESSING', 'COMPLETED', 'ERROR')),
                    workspace_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_size INTEGER,
                    processing_time INTEGER,
                    error_message TEXT
                )
            """)
            
            # Create processing logs table with CASCADE delete
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    level TEXT DEFAULT 'INFO',
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)
        logger.info("Database initialized successfully")
    
    def get_all_documents(self, status: Optional[str] = None) -> List[Dict]:
        """Fetch all documents or documents filtered by status from the database."""
        try:
            with self.db_manager as cursor:
                query = """
                    SELECT id, display_name, original_filename, status, created_at, 
                           updated_at, file_size, processing_time, error_message
                    FROM documents
                """
                params = ()
                if status:
                    query += " WHERE status = ?"
                    params = (status,)
                query += " ORDER BY created_at DESC"
                
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError as e:
            logger.error(f"Database error: {e}")
            return []
    
    def get_processed_documents(self) -> List[Dict]:
        """Fetch all successfully processed documents from the database."""
        return self.get_all_documents(status=DocumentStatus.COMPLETED)
    
    def get_document_status(self, doc_id: str) -> Optional[str]:
        """Get the current status of a document."""
        try:
            with self.db_manager as cursor:
                cursor.execute("SELECT status FROM documents WHERE id = ?", (doc_id,))
                result = cursor.fetchone()
                return result[0] if result else None
        except sqlite3.Error as e:
            logger.error(f"Error getting document status: {e}")
            return None
    
    def update_document_status(self, doc_id: str, status: str, error_message: str = None):
        """Update document status in the database."""
        try:
            with self.db_manager as cursor:
                cursor.execute("""
                    UPDATE documents 
                    SET status = ?, updated_at = CURRENT_TIMESTAMP, error_message = ?
                    WHERE id = ?
                """, (status, error_message, doc_id))
            logger.info(f"Updated document {doc_id} status to {status}")
        except sqlite3.Error as e:
            logger.error(f"Error updating document status: {e}")
    
    def log_processing_step(self, doc_id: str, stage: str, message: str, level: str = "INFO"):
        """Log a processing step for a document."""
        try:
            with self.db_manager as cursor:
                cursor.execute("""
                    INSERT INTO processing_logs (document_id, stage, message, level)
                    VALUES (?, ?, ?, ?)
                """, (doc_id, stage, message, level))
        except sqlite3.Error as e:
            logger.error(f"Error logging processing step: {e}")
    
    def create_workspace(self, doc_id: str) -> Path:
        """Create a new workspace for a document."""
        workspace_path = WORKSPACES_DIR / doc_id
        workspace_path.mkdir(parents=True, exist_ok=True)
        (workspace_path / "input").mkdir(exist_ok=True)
        (workspace_path / "output").mkdir(exist_ok=True)
        (workspace_path / "cache").mkdir(exist_ok=True)
        
        # Use config manager to set up workspace configuration
        from config_manager import config_manager
        success = config_manager.create_workspace_config(workspace_path)
        
        if not success:
            logger.warning(f"Failed to create workspace config for {workspace_path}, using fallback")
            # Fallback: manually copy files
            if GRAPHRAG_CONFIG_DIR.exists():
                config_files = ["settings.yaml", ".env"]
                for config_file in config_files:
                    src = GRAPHRAG_CONFIG_DIR / config_file
                    dst = workspace_path / config_file
                    if src.exists():
                        shutil.copy2(src, dst)
                
                # Copy prompts directory
                prompts_src = GRAPHRAG_CONFIG_DIR / "prompts"
                prompts_dst = workspace_path / "prompts"
                if prompts_src.exists():
                    shutil.copytree(prompts_src, prompts_dst, dirs_exist_ok=True)
        
        # Always ensure .env file exists in workspace
        env_src = GRAPHRAG_CONFIG_DIR / ".env"
        env_dst = workspace_path / ".env"
        if env_src.exists() and not env_dst.exists():
            shutil.copy2(env_src, env_dst)
        
        logger.info(f"Created workspace: {workspace_path}")
        return workspace_path
    
    def extract_text_from_pdf(self, pdf_path: Path, output_path: Path) -> bool:
        """Extract text from PDF file using PyPDF2 or other libraries."""
        try:
            # Import our document structure parser
            from document_structure_parser import extract_document_structure
            
            # If it's already a text file, process it for structure
            if pdf_path.suffix.lower() == '.txt':
                with open(pdf_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            # For PDF files, try multiple extraction methods
            elif pdf_path.suffix.lower() == '.pdf':
                text_content = self._extract_pdf_text_content(pdf_path)
                if not text_content:
                    # Fallback: create a placeholder text file
                    text_content = f"Text extracted from {pdf_path.name}\nThis is a placeholder. Implement actual PDF extraction.\n"
            else:
                # Unsupported file type
                text_content = f"Text extracted from {pdf_path.name}\nUnsupported file type.\n"
            
            # Extract document structure
            sections, structured_text, metadata_list = extract_document_structure(text_content)
            
            # Save structured text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(structured_text)
            
            # Save section metadata as JSON file
            import json
            metadata_path = output_path.parent / f"{output_path.stem}_structure.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'sections': metadata_list,
                    'total_sections': len(sections),
                    'document': pdf_path.name
                }, f, indent=2)
            
            logger.info(f"Extracted {len(sections)} sections from {pdf_path.name}")
            
            return True
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return False
    
    def _extract_pdf_text_content(self, pdf_path: Path) -> str:
        """Extract text content from PDF using available libraries."""
        # Try libraries in order of preference
        extraction_methods = [
            ("PyPDF2", self._extract_with_pypdf2),
            ("pdfplumber", self._extract_with_pdfplumber),
            ("PyMuPDF", self._extract_with_pymupdf)
        ]
        
        for lib_name, extract_func in extraction_methods:
            try:
                logger.info(f"Attempting PDF extraction with {lib_name}")
                text = extract_func(pdf_path)
                if text:
                    return text
            except ImportError:
                logger.warning(f"{lib_name} not available")
            except Exception as e:
                logger.error(f"Error with {lib_name}: {e}")
        
        # If no libraries work, return placeholder
        return f"PDF text extraction from {pdf_path.name}\n(No PDF libraries available - install PyPDF2, pdfplumber, or PyMuPDF)"
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2."""
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber."""
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> str:
        """Extract text using PyMuPDF."""
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text() + "\n"
        doc.close()
        return text.strip()
    
    def _validate_workspace(self, workspace_path: Path, text_path: Optional[Path] = None) -> None:
        """Validate that workspace has required structure and files."""
        absolute_workspace_path = workspace_path.resolve()
        
        if not absolute_workspace_path.exists():
            raise Exception(f"Workspace directory does not exist: {absolute_workspace_path}")
        
        input_dir = absolute_workspace_path / "input"
        if not input_dir.exists():
            raise Exception(f"Input directory does not exist: {input_dir}")
        
        # Check for text files
        if text_path and not text_path.exists():
            raise Exception(f"Text file does not exist: {text_path}")
        elif not text_path and not list(input_dir.glob("*.txt")):
            raise Exception(f"No text files found in input directory: {input_dir}")
        
        # Check settings file
        settings_path = absolute_workspace_path / "settings.yaml"
        if not settings_path.exists():
            raise Exception(f"Settings file does not exist: {settings_path}")
    
    def _run_graphrag_indexing(self, doc_id: str, workspace_path: Path) -> None:
        """Run GraphRAG indexing process."""
        self.log_processing_step(doc_id, "GRAPHRAG", f"Running GraphRAG indexing on {workspace_path.resolve()}")
        
        command = [str(VENV_GRAPHRAG_PATH), "index", "--root", str(workspace_path.resolve())]
        
        self.log_processing_step(doc_id, "GRAPHRAG", f"Command: {' '.join(command)}")
        self.log_processing_step(doc_id, "GRAPHRAG", f"Working directory: {PROJECT_ROOT}")
        
        result = subprocess.run(command, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        
        if result.returncode != 0:
            error_msg = f"GraphRAG failed: {result.stderr}"
            self.log_processing_step(doc_id, "GRAPHRAG", error_msg, "ERROR")
            if result.stdout:
                self.log_processing_step(doc_id, "GRAPHRAG", f"GraphRAG stdout: {result.stdout}", "INFO")
            raise Exception(error_msg)
        
        self.log_processing_step(doc_id, "GRAPHRAG", "GraphRAG indexing completed successfully")
        if result.stdout:
            self.log_processing_step(doc_id, "GRAPHRAG", f"GraphRAG output: {result.stdout}", "INFO")
    
    def _process_document_core(self, doc_id: str, workspace_path: Path, uploaded_file_buffer: bytes = None, original_filename: str = None) -> None:
        """Core document processing logic shared between new and reprocess workflows."""
        start_time = datetime.now()
        
        # Save file if this is a new upload
        if uploaded_file_buffer and original_filename:
            self.log_processing_step(doc_id, "UPLOAD", f"Saving uploaded file: {original_filename}")
            input_path = workspace_path / "input" / original_filename
            with open(input_path, "wb") as f:
                f.write(uploaded_file_buffer)
            
            # Extract text
            self.log_processing_step(doc_id, "EXTRACT", "Extracting text from document")
            text_path = workspace_path / "input" / f"{input_path.stem}.txt"
            if not self.extract_text_from_pdf(input_path, text_path):
                raise Exception("Failed to extract text from document")
            
            # Validate workspace
            self.log_processing_step(doc_id, "VALIDATE", "Validating workspace structure")
            self._validate_workspace(workspace_path, text_path)
        else:
            # For reprocessing, just validate existing workspace
            self.log_processing_step(doc_id, "VALIDATE", "Validating workspace structure")
            self._validate_workspace(workspace_path)
        
        # Run GraphRAG indexing
        self._run_graphrag_indexing(doc_id, workspace_path)
        
        # Update processing time and status
        processing_time = (datetime.now() - start_time).total_seconds()
        with self.db_manager as cursor:
            cursor.execute("UPDATE documents SET processing_time = ? WHERE id = ?", (int(processing_time), doc_id))
        
        self.update_document_status(doc_id, DocumentStatus.COMPLETED)
        self.log_processing_step(doc_id, "COMPLETE", f"Processing completed in {processing_time:.2f} seconds")
    
    def process_document_background(self, doc_id: str, uploaded_file, _display_name: str):
        """Process document in background thread."""
        try:
            # Get workspace path
            with self.db_manager as cursor:
                cursor.execute("SELECT workspace_path FROM documents WHERE id = ?", (doc_id,))
                result = cursor.fetchone()
                if not result:
                    raise Exception("Document not found in database")
                workspace_path = Path(result[0])
            
            # Process document with uploaded file
            self._process_document_core(doc_id, workspace_path, uploaded_file.getbuffer(), uploaded_file.name)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing document {doc_id}: {error_msg}")
            self.update_document_status(doc_id, DocumentStatus.ERROR, error_msg)
            self.log_processing_step(doc_id, "ERROR", error_msg, "ERROR")
    
    def fix_workspace_paths(self):
        """Fix relative workspace paths to absolute paths in existing documents."""
        try:
            with self.db_manager as cursor:
                cursor.execute("SELECT id, workspace_path FROM documents WHERE workspace_path NOT LIKE '/%'")
                docs = cursor.fetchall()
                
                for doc_id, workspace_path in docs:
                    # Convert relative to absolute path
                    abs_path = Path(workspace_path).resolve()
                    cursor.execute("UPDATE documents SET workspace_path = ? WHERE id = ?", (str(abs_path), doc_id))
                    logger.info(f"Fixed workspace path for {doc_id}: {workspace_path} -> {abs_path}")
            
            logger.info(f"Fixed {len(docs)} workspace paths")
            
        except Exception as e:
            logger.error(f"Error fixing workspace paths: {e}")

    def reprocess_failed_document(self, doc_id: str) -> str:
        """Reprocess a failed document."""
        try:
            with self.db_manager as cursor:
                cursor.execute("SELECT workspace_path, display_name FROM documents WHERE id = ? AND status = ?", (doc_id, DocumentStatus.ERROR))
                result = cursor.fetchone()
                
                if not result:
                    return f"❌ Document {doc_id} not found or not in ERROR status"
                
                workspace_path = Path(result[0])
                display_name = result[1]
            
            # Reset status to PROCESSING
            self.update_document_status(doc_id, DocumentStatus.PROCESSING, None)
            
            # Start background processing for reprocessing
            thread = threading.Thread(
                target=self.reprocess_document_background,
                args=(doc_id, workspace_path, display_name)
            )
            thread.daemon = True
            thread.start()
            
            return f"✅ Started reprocessing '{display_name}' (ID: {doc_id})"
            
        except Exception as e:
            error_msg = f"❌ Error reprocessing document: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def reprocess_document_background(self, doc_id: str, workspace_path: Path, _display_name: str):
        """Reprocess document in background thread without file upload."""
        try:
            # Process document without file upload (reprocessing existing)
            self._process_document_core(doc_id, workspace_path)
            
        except Exception as e:
            error_msg = f"Error reprocessing document {doc_id}: {str(e)}"
            logger.error(error_msg)
            self.update_document_status(doc_id, DocumentStatus.ERROR, error_msg)
            self.log_processing_step(doc_id, "ERROR", error_msg, "ERROR")

    def process_new_document(self, uploaded_file, display_name: str = None) -> str:
        """Process a new document upload."""
        try:
            # Generate unique document ID
            doc_id = str(uuid.uuid4())
            if display_name is None:
                display_name = uploaded_file.name
            
            # Create workspace
            workspace_path = self.create_workspace(doc_id)
            
            # Add to database
            with self.db_manager as cursor:
                cursor.execute("""
                    INSERT INTO documents (id, display_name, original_filename, status, workspace_path, file_size)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    doc_id, 
                    display_name, 
                    uploaded_file.name, 
                    DocumentStatus.PROCESSING,
                    str(workspace_path.resolve()),
                    len(uploaded_file.getbuffer())
                ))
            
            # Start background processing
            thread = threading.Thread(
                target=self.process_document_background,
                args=(doc_id, uploaded_file, display_name)
            )
            thread.daemon = True
            thread.start()
            
            return f"✅ Started processing '{display_name}' (ID: {doc_id})"
            
        except Exception as e:
            error_msg = f"❌ Error starting processing: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def load_and_merge_graphs(self, selected_doc_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and merge graph data from selected documents."""
        if not selected_doc_ids:
            return pd.DataFrame(), pd.DataFrame()
        
        try:
            with self.db_manager as cursor:
                placeholders = ','.join('?' for _ in selected_doc_ids)
                query = f"SELECT id, workspace_path, display_name FROM documents WHERE id IN ({placeholders}) AND status = ?"
                cursor.execute(query, selected_doc_ids + [DocumentStatus.COMPLETED])
                docs = cursor.fetchall()
            
            if not docs:
                return pd.DataFrame(), pd.DataFrame()
            
            all_entities = []
            all_relationships = []
            
            for doc_id, workspace_path, display_name in docs:
                workspace = Path(workspace_path)
                
                # Look for GraphRAG output files
                entity_file = workspace / "output" / "entities.parquet"
                relationship_file = workspace / "output" / "relationships.parquet"
                
                # Try alternative locations
                if not entity_file.exists():
                    entity_file = workspace / "output" / "artifacts" / "entities.parquet"
                if not relationship_file.exists():
                    relationship_file = workspace / "output" / "artifacts" / "relationships.parquet"
                
                if entity_file.exists():
                    entities_df = pd.read_parquet(entity_file)
                    # Add source document information
                    entities_df['source_document'] = display_name
                    entities_df['source_doc_id'] = doc_id
                    all_entities.append(entities_df)
                    logger.info(f"Loaded {len(entities_df)} entities from {display_name}")
                
                if relationship_file.exists():
                    relationships_df = pd.read_parquet(relationship_file)
                    # Add source document information
                    relationships_df['source_document'] = display_name
                    relationships_df['source_doc_id'] = doc_id
                    all_relationships.append(relationships_df)
                    logger.info(f"Loaded {len(relationships_df)} relationships from {display_name}")
            
            if not all_entities:
                logger.warning("No entity data found for selected documents")
                return pd.DataFrame(), pd.DataFrame()
            
            # Merge dataframes
            merged_entities_df = pd.concat(all_entities, ignore_index=True)
            merged_relationships_df = pd.concat(all_relationships, ignore_index=True) if all_relationships else pd.DataFrame()
            
            # Basic deduplication (can be enhanced later)
            if 'id' in merged_entities_df.columns:
                merged_entities_df = merged_entities_df.drop_duplicates(subset=['id']).reset_index(drop=True)
            
            if not merged_relationships_df.empty and 'id' in merged_relationships_df.columns:
                merged_relationships_df = merged_relationships_df.drop_duplicates(subset=['id']).reset_index(drop=True)
            
            logger.info(f"Merged graph: {len(merged_entities_df)} entities, {len(merged_relationships_df)} relationships")
            return merged_entities_df, merged_relationships_df
            
        except Exception as e:
            logger.error(f"Error loading and merging graphs: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its workspace."""
        try:
            # Get workspace path
            with self.db_manager as cursor:
                cursor.execute("SELECT workspace_path FROM documents WHERE id = ?", (doc_id,))
                result = cursor.fetchone()
                
                if not result:
                    logger.warning(f"Document not found: {doc_id}")
                    return False
                
                workspace_path = Path(result[0])
                
                # Delete workspace directory
                if workspace_path.exists():
                    shutil.rmtree(workspace_path)
                    logger.info(f"Deleted workspace: {workspace_path}")
                
                # Delete from database (CASCADE will handle processing_logs)
                cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                
                logger.info(f"Deleted document: {doc_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def get_processing_logs(self, doc_id: str) -> List[Dict]:
        """Get processing logs for a document."""
        try:
            with self.db_manager as cursor:
                cursor.execute("""
                    SELECT stage, message, timestamp, level
                    FROM processing_logs
                    WHERE document_id = ?
                    ORDER BY timestamp ASC
                """, (doc_id,))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Error getting processing logs: {e}")
            return []

# Global instance
processor = GraphRAGProcessor()

# Convenience functions for backward compatibility
def init_db():
    """Initialize the database."""
    return processor.init_db()

def get_processed_documents():
    """Get all processed documents."""
    return processor.get_processed_documents()

def process_new_document(uploaded_file, display_name: str = None):
    """Process a new document."""
    return processor.process_new_document(uploaded_file, display_name)

def load_and_merge_graphs(selected_doc_ids: List[str]):
    """Load and merge graphs from selected documents."""
    return processor.load_and_merge_graphs(selected_doc_ids)

def delete_document(doc_id: str):
    """Delete a document."""
    return processor.delete_document(doc_id)

def get_document_status(doc_id: str):
    """Get document status."""
    return processor.get_document_status(doc_id)

def update_document_status(doc_id: str, status: str, error_message: str = None):
    """Update document status."""
    return processor.update_document_status(doc_id, status, error_message)

def reprocess_failed_document(doc_id: str):
    """Reprocess a failed document."""
    return processor.reprocess_failed_document(doc_id)

def get_processing_logs(doc_id: str):
    """Get processing logs for a document."""
    return processor.get_processing_logs(doc_id)

def get_all_documents():
    """Get all documents."""
    return processor.get_all_documents()