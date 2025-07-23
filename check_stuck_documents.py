#!/usr/bin/env python3
"""
Monitor and handle stuck documents in the GraphRAG processing pipeline.
"""

import logging
import argparse
from pathlib import Path
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_stuck_documents(timeout_minutes: int = 30, auto_fix: bool = False):
    """Check for and optionally fix stuck documents."""
    try:
        from app_logic import GraphRAGProcessor
        
        processor = GraphRAGProcessor()
        
        # Check for stuck documents
        stuck_doc_ids = processor.check_and_handle_stuck_documents(timeout_minutes=timeout_minutes)
        
        if stuck_doc_ids:
            logger.warning(f"Found {len(stuck_doc_ids)} stuck documents")
            for doc_id in stuck_doc_ids:
                logger.warning(f"  - {doc_id}")
            
            if auto_fix:
                logger.info("Auto-fix enabled: Documents have been marked as ERROR")
            else:
                logger.info("Use --auto-fix to automatically mark these as ERROR")
                
            return stuck_doc_ids
        else:
            logger.info("No stuck documents found")
            return []
            
    except Exception as e:
        logger.error(f"Error checking stuck documents: {e}")
        return []

def show_processing_status():
    """Show current processing status of all documents."""
    try:
        from app_logic import GraphRAGProcessor
        
        processor = GraphRAGProcessor()
        docs = processor.get_all_documents()
        
        if not docs:
            logger.info("No documents found in database")
            return
        
        logger.info(f"Found {len(docs)} total documents:")
        
        status_counts = {}
        for doc in docs:
            status = doc['status']
            status_counts[status] = status_counts.get(status, 0) + 1
            
            if status == 'PROCESSING':
                created_at = doc['created_at']
                logger.warning(f"  🔄 PROCESSING: {doc['display_name']} (since {created_at})")
            elif status == 'ERROR':
                error = doc.get('error_message', 'Unknown error')
                logger.error(f"  ❌ ERROR: {doc['display_name']} - {error[:100]}...")
            elif status == 'COMPLETED':
                logger.info(f"  ✅ COMPLETED: {doc['display_name']}")
        
        logger.info(f"Status summary: {status_counts}")
        
    except Exception as e:
        logger.error(f"Error showing status: {e}")

def monitor_continuous(check_interval_minutes: int = 5, timeout_minutes: int = 30):
    """Continuously monitor for stuck documents."""
    logger.info(f"Starting continuous monitoring (check every {check_interval_minutes} min, timeout {timeout_minutes} min)")
    
    try:
        while True:
            logger.info("Checking for stuck documents...")
            stuck_docs = check_stuck_documents(timeout_minutes=timeout_minutes, auto_fix=True)
            
            if stuck_docs:
                logger.warning(f"Fixed {len(stuck_docs)} stuck documents")
            
            logger.info(f"Sleeping for {check_interval_minutes} minutes...")
            time.sleep(check_interval_minutes * 60)
            
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Monitoring error: {e}")

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Monitor GraphRAG document processing")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Timeout in minutes for stuck document detection (default: 30)")
    parser.add_argument("--auto-fix", action="store_true",
                       help="Automatically mark stuck documents as ERROR")
    parser.add_argument("--status", action="store_true",
                       help="Show processing status of all documents")
    parser.add_argument("--monitor", action="store_true",
                       help="Continuously monitor for stuck documents")
    parser.add_argument("--interval", type=int, default=5,
                       help="Check interval in minutes for continuous monitoring (default: 5)")
    
    args = parser.parse_args()
    
    try:
        if args.status:
            show_processing_status()
        elif args.monitor:
            monitor_continuous(args.interval, args.timeout)
        else:
            check_stuck_documents(args.timeout, args.auto_fix)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()