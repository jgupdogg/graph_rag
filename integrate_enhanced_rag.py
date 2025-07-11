#!/usr/bin/env python3
"""
Enhanced RAG Integration Script
Simple script to integrate enhanced processing with existing GraphRAG workflows
"""

import sys
import os
import logging
from pathlib import Path
import argparse

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_requirements():
    """Check if all requirements are met"""
    logger = logging.getLogger(__name__)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY environment variable not set")
        logger.error("Please set your OpenAI API key:")
        logger.error("export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    # Check required files
    required_files = [
        "enhanced_document_processor.py",
        "enhanced_graphrag_integration.py",
        "document_structure_parser.py",
        "structure_aware_chunking.py"
    ]
    
    current_dir = Path(__file__).parent
    for file in required_files:
        if not (current_dir / file).exists():
            logger.error(f"❌ Required file not found: {file}")
            return False
    
    logger.info("✅ All requirements met")
    return True

def integrate_workspace(workspace_path: Path, dry_run: bool = False, config_override: dict = None):
    """Integrate enhanced processing with a GraphRAG workspace"""
    logger = logging.getLogger(__name__)
    
    if not workspace_path.exists():
        logger.error(f"❌ Workspace not found: {workspace_path}")
        return False
    
    if not (workspace_path / "input").exists():
        logger.error(f"❌ Input directory not found: {workspace_path}/input")
        return False
    
    # Count input documents
    input_files = list((workspace_path / "input").glob("*.txt"))
    if not input_files:
        logger.warning("⚠️  No .txt files found in input directory")
        return False
    
    logger.info(f"🔍 Found {len(input_files)} documents to process:")
    for file in input_files:
        logger.info(f"   • {file.name}")
    
    if dry_run:
        logger.info("🏃 Dry run mode - would process these files with enhanced RAG")
        return True
    
    # Import and run enhanced processing
    try:
        from enhanced_graphrag_integration import run_enhanced_graphrag_workflow
        
        logger.info("🚀 Starting enhanced processing...")
        results = run_enhanced_graphrag_workflow(workspace_path, config_override)
        
        if results["status"] == "success":
            logger.info("✅ Enhanced processing completed successfully!")
            logger.info(f"📊 Results:")
            logger.info(f"   • Documents processed: {results['documents']}")
            logger.info(f"   • Enhanced chunks created: {results['enhanced_chunks']}")
            logger.info(f"   • Output files:")
            for name, path in results.get('files_created', {}).items():
                logger.info(f"     - {name}: {path}")
            
            logger.info("\n🎯 Next Steps:")
            logger.info("1. Your enhanced chunks are ready in output/text_units.parquet")
            logger.info("2. Run your normal GraphRAG indexing process")
            logger.info("3. The enhanced context will improve your RAG results")
            
            return True
        else:
            logger.error(f"❌ Enhanced processing failed: {results.get('error', 'Unknown error')}")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        logger.error("Make sure all enhanced RAG files are in the current directory")
        return False
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Integrate Enhanced RAG processing with GraphRAG workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a workspace
  python integrate_enhanced_rag.py /path/to/workspace

  # Dry run to see what would be processed
  python integrate_enhanced_rag.py /path/to/workspace --dry-run

  # Verbose output
  python integrate_enhanced_rag.py /path/to/workspace --verbose

  # Use custom cache size
  python integrate_enhanced_rag.py /path/to/workspace --cache-size 500
        """
    )
    
    parser.add_argument(
        "workspace",
        type=Path,
        help="Path to GraphRAG workspace directory"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually running"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--cache-size",
        type=int,
        default=100,
        help="Cache size in MB (default: 100)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent workers (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Print header
    logger.info("🔧 Enhanced RAG Integration Script")
    logger.info("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Prepare configuration
    config_override = {
        'max_cache_size_mb': args.cache_size,
        'max_workers': args.max_workers,
        'cache_enabled': True
    }
    
    # Run integration
    success = integrate_workspace(args.workspace, args.dry_run, config_override)
    
    if success:
        if not args.dry_run:
            logger.info("\n🎉 Integration completed successfully!")
            logger.info("Your GraphRAG workspace now has enhanced processing capabilities.")
        return 0
    else:
        logger.error("\n❌ Integration failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())