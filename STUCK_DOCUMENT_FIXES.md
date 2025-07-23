# Stuck Document Processing Fixes

## Problem Summary
Documents were getting stuck in PROCESSING status for hours due to:
1. **GraphRAG taking too long** - Process was making continuous API calls but not progressing
2. **No timeout protection** - Processes could run indefinitely 
3. **Sequential processing** - GraphRAG config had `concurrent_requests: 1` making it extremely slow
4. **No monitoring** - No way to detect or handle stuck documents automatically

## Root Cause Analysis
- **Document ID**: `a194be59-808d-4636-863e-35b2b1529f14` (chapter2.pdf)
- **Enhanced RAG**: Took 1 hour (13:33 to 14:32) due to rate limiting
- **GraphRAG indexing**: Process ran for 20+ minutes making 600+ API calls but stuck in entity extraction
- **Configuration**: GraphRAG was configured with `concurrent_requests: 1` which is extremely slow

## Fixes Implemented

### 1. Immediate Resolution ✅
- **Killed stuck process**: Terminated PID 3644752 that was running for hours
- **Updated document status**: Marked as ERROR with descriptive message
- **Database cleanup**: Document now shows proper error state

### 2. Timeout Protection ✅
Added comprehensive timeout protection to `app_logic.py`:

```python
def _run_graphrag_indexing(self, doc_id: str, workspace_path: Path) -> None:
    # Set timeout to 30 minutes (1800 seconds) to prevent indefinite hangs
    timeout_seconds = 1800
    
    result = subprocess.run(
        command, 
        capture_output=True, 
        text=True, 
        cwd=str(PROJECT_ROOT),
        timeout=timeout_seconds  # NEW: Timeout protection
    )
```

### 3. Stuck Document Detection ✅
Added automatic stuck document detection:

```python
def check_and_handle_stuck_documents(self, timeout_minutes: int = 30) -> List[str]:
    """Check for documents stuck in PROCESSING status and mark them as ERROR."""
    # Finds documents processing > timeout_minutes
    # Automatically marks them as ERROR
    # Logs the timeout event
```

### 4. Monitoring Tools ✅
Created comprehensive monitoring system:

- **`check_stuck_documents.py`**: Standalone monitoring script
  - `--status`: Show processing status of all documents
  - `--auto-fix`: Automatically mark stuck documents as ERROR
  - `--monitor`: Continuously monitor for stuck documents
  - `--timeout`: Configure timeout threshold

### 5. Automatic Startup Check ✅
Added automatic stuck document checking on app startup:

```python
def get_processor():
    """Get or create the global GraphRAG processor instance."""
    global processor
    if processor is None:
        processor = GraphRAGProcessor()
        # Check for stuck documents on startup
        stuck_docs = processor.check_and_handle_stuck_documents(timeout_minutes=30)
        if stuck_docs:
            logger.info(f"Fixed {len(stuck_docs)} stuck documents on startup")
```

### 6. Optimized Configuration Template ✅
Created `optimized_settings_template.yaml` with better defaults:

```yaml
llm:
  concurrent_requests: 5  # Increased from 1
  requests_per_minute: 50  # Controlled limit
  max_retries: 5  # Reduced from 10 to fail faster
  request_timeout: 120.0  # Reduced timeout

embeddings:
  llm:
    concurrent_requests: 10  # Higher for embeddings
    requests_per_minute: 100  # Higher limit
    request_timeout: 60.0  # Shorter timeout
```

## Usage

### Check Current Status
```bash
source venv/bin/activate
python check_stuck_documents.py --status
```

### Monitor Continuously
```bash
python check_stuck_documents.py --monitor --interval 5 --timeout 30
```

### Manual Check and Fix
```bash
python check_stuck_documents.py --auto-fix --timeout 30
```

## Prevention Measures

### 1. Timeout Protection
- **GraphRAG indexing**: 30-minute timeout
- **Subprocess management**: Automatic termination of long-running processes
- **Error handling**: Proper error messages and logging

### 2. Configuration Optimization  
- **Increased concurrency**: `concurrent_requests: 5` instead of 1
- **Balanced rate limits**: Prevent both slowdowns and rate limit errors
- **Shorter timeouts**: Fail faster instead of hanging indefinitely

### 3. Monitoring Integration
- **Startup checks**: Automatic detection of stuck documents when app starts
- **Periodic monitoring**: Can run continuous monitoring in background
- **Status visibility**: Clear status reporting in UI and CLI

### 4. Enhanced Error Handling
- **Descriptive error messages**: Clear explanation of what went wrong
- **Proper logging**: Full processing history preserved
- **Graceful degradation**: App continues working even with some failed documents

## Testing

### Verified Fixes
✅ Stuck document properly marked as ERROR  
✅ Timeout protection working in GraphRAG indexing  
✅ Monitoring script detects and handles stuck documents  
✅ App startup automatically checks for stuck documents  
✅ All convenience functions updated to use new processor system  

### Current Status
- **Database**: Clean with 1 ERROR document (properly handled)
- **Processing Pipeline**: Ready with timeout protection
- **Monitoring**: Active with automatic detection
- **Configuration**: Optimized for better performance

## Future Improvements

1. **Progress Tracking**: Show processing progress in UI
2. **Estimated Time**: Calculate expected completion time based on document size
3. **Chunking Optimization**: Reduce chunk size for large documents to speed processing
4. **Caching**: Cache intermediate results to avoid reprocessing on restarts
5. **Resource Monitoring**: Track CPU/memory usage during processing

## Files Modified/Created

### Core Files
- `app_logic.py`: Added timeout protection and stuck document detection
- `optimized_settings_template.yaml`: Better GraphRAG configuration

### Monitoring Tools  
- `check_stuck_documents.py`: Comprehensive monitoring script
- `STUCK_DOCUMENT_FIXES.md`: This documentation

### Database
- Updated document status to ERROR with descriptive messages
- Added automatic stuck document detection on startup

The system is now robust against stuck document processing and includes comprehensive monitoring and prevention measures.