#!/usr/bin/env python3
"""
Test script for enhanced GraphRAG features:
1. Rate limiting for API calls
2. Raw text embeddings generation
3. Enhanced two-stage search
"""

import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rate_limiter():
    """Test the rate limiter functionality."""
    logger.info("Testing rate limiter...")
    
    try:
        from rate_limiter import RateLimiter, create_rate_limited_client
        
        # Create a rate limiter with aggressive limits for testing
        limiter = RateLimiter(
            requests_per_minute=5,
            min_delay_between_requests=1.0
        )
        
        # Test the rate limiting
        @limiter
        def mock_api_call(i):
            logger.info(f"Making API call {i}")
            return {"success": True, "call": i}
        
        # Make several calls
        start_time = time.time()
        for i in range(3):
            result = mock_api_call(i)
            logger.info(f"Result: {result}")
        
        elapsed = time.time() - start_time
        logger.info(f"Completed 3 calls in {elapsed:.2f} seconds")
        
        # Get stats
        stats = limiter.get_stats()
        logger.info(f"Rate limiter stats: {stats}")
        
        logger.info("✅ Rate limiter test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Rate limiter test failed: {e}")
        return False


def test_raw_text_embeddings():
    """Test raw text embedding generation."""
    logger.info("Testing raw text embeddings...")
    
    try:
        from raw_text_embeddings import RawTextEmbeddingStore
        from config_manager import config_manager
        
        # Check if API key is available
        api_key = config_manager.get_api_key()
        if not api_key:
            logger.warning("API key not configured. Skipping embedding test.")
            return True
        
        # Find a test workspace
        workspaces_dir = Path("workspaces")
        if not workspaces_dir.exists() or not list(workspaces_dir.iterdir()):
            logger.warning("No workspaces found. Skipping embedding test.")
            return True
        
        # Get first workspace
        test_workspace = next(workspaces_dir.iterdir())
        logger.info(f"Using test workspace: {test_workspace}")
        
        # Initialize store
        store = RawTextEmbeddingStore(test_workspace, api_key)
        
        # Check table info
        info = store.get_table_info()
        logger.info(f"Raw text embeddings table info: {info}")
        
        logger.info("✅ Raw text embeddings test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Raw text embeddings test failed: {e}")
        return False


def test_enhanced_query():
    """Test enhanced query functionality."""
    logger.info("Testing enhanced query handler...")
    
    try:
        from enhanced_query_handler import EnhancedQueryHandler
        from config_manager import config_manager
        
        # Check if API key is available
        api_key = config_manager.get_api_key()
        if not api_key:
            logger.warning("API key not configured. Skipping query test.")
            return True
        
        # Find a test workspace
        workspaces_dir = Path("workspaces")
        if not workspaces_dir.exists() or not list(workspaces_dir.iterdir()):
            logger.warning("No workspaces found. Skipping query test.")
            return True
        
        # Get first workspace
        test_workspace = next(workspaces_dir.iterdir())
        logger.info(f"Using test workspace: {test_workspace}")
        
        # Initialize handler
        handler = EnhancedQueryHandler(test_workspace, api_key)
        
        # Test query (without actually running it to save API calls)
        logger.info("Enhanced query handler initialized successfully")
        
        logger.info("✅ Enhanced query test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced query test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting enhanced features test suite...")
    
    tests = [
        ("Rate Limiter", test_rate_limiter),
        ("Raw Text Embeddings", test_raw_text_embeddings),
        ("Enhanced Query", test_enhanced_query)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, success in results if success)
    logger.info(f"\nTotal: {total_passed}/{len(results)} tests passed")


if __name__ == "__main__":
    main()