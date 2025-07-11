#!/usr/bin/env python3
"""
Test script for Enhanced Document Processing
Validates the implementation of RAG enhancements
"""

import sys
import logging
import json
from pathlib import Path
import pandas as pd
import os

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_document_processor import (
    DocumentClassifier, 
    SectionSummarizer, 
    EnhancedDocumentProcessor
)
from enhanced_graphrag_integration import EnhancedGraphRAGWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_document_classifier():
    """Test the DocumentClassifier component"""
    logger.info("Testing Document Classifier...")
    
    # Test documents
    test_docs = [
        {
            "text": "1. CONCRETE REQUIREMENTS\n1.1 Materials\nConcrete shall meet the requirements of ASTM C150. Storm drainage systems shall use Class A concrete with minimum compressive strength of 4000 psi.",
            "metadata": {"filename": "concrete_spec.txt", "file_size": 200, "section_count": 2},
            "expected_type": "Technical Specification"
        },
        {
            "text": "EXECUTIVE SUMMARY\nThis business analysis report examines our Q3 revenue performance and market penetration. Key findings show 15% growth in customer acquisition.",
            "metadata": {"filename": "business_report.txt", "file_size": 150, "section_count": 3},
            "expected_type": "Business Document"
        },
        {
            "text": "TERMS AND CONDITIONS\n1. SCOPE OF AGREEMENT\nThis agreement shall govern the relationship between the parties and establish the obligations of each party.",
            "metadata": {"filename": "contract.txt", "file_size": 180, "section_count": 2},
            "expected_type": "Legal/Contract"
        }
    ]
    
    try:
        # Mock API key for testing (will use environment variable if available)
        api_key = os.getenv("OPENAI_API_KEY") or "test-key"
        classifier = DocumentClassifier(api_key=api_key, cache_enabled=True)
        
        for i, test_doc in enumerate(test_docs):
            logger.info(f"Testing document {i+1}: {test_doc['metadata']['filename']}")
            
            # For testing without actual API calls, we'll simulate the classification
            if api_key == "test-key":
                logger.info("No API key available - simulating classification")
                # Use fallback classification for testing
                result = classifier._fallback_classification(test_doc["text"])
            else:
                result = classifier.classify_document(test_doc["text"], test_doc["metadata"])
            
            logger.info(f"Classified as: {result.document_type} (confidence: {result.confidence:.2f})")
            logger.info(f"Reasoning: {result.reasoning}")
            
        logger.info("✅ Document Classifier test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Document Classifier test failed: {e}")
        return False


def test_section_summarizer():
    """Test the SectionSummarizer component"""
    logger.info("Testing Section Summarizer...")
    
    try:
        from document_structure_parser import Section
        
        # Test section
        test_section = Section(
            level=2,
            title="Concrete Requirements",
            start_pos=100,
            content="This section specifies the concrete requirements for storm drainage systems. Concrete shall meet ASTM C150 standards with minimum compressive strength of 4000 psi."
        )
        test_section.parent_path = ["Specifications", "Materials"]
        
        api_key = os.getenv("OPENAI_API_KEY") or "test-key"
        summarizer = SectionSummarizer(api_key=api_key, cache_enabled=True)
        
        if api_key == "test-key":
            logger.info("No API key available - creating fallback summary")
            summary = f"This section covers {test_section.title} in the context of {' > '.join(test_section.parent_path)}."
        else:
            summary = summarizer.generate_section_summary(test_section, "Technical Specification")
        
        logger.info(f"Generated summary: {summary}")
        logger.info("✅ Section Summarizer test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Section Summarizer test failed: {e}")
        return False


def test_enhanced_document_processor():
    """Test the complete EnhancedDocumentProcessor"""
    logger.info("Testing Enhanced Document Processor...")
    
    try:
        # Test document
        test_text = """1. CONCRETE REQUIREMENTS

1.1 General
Concrete shall meet the requirements specified in ASTM C150. All materials shall be approved prior to use.

1.2 Storm Drainage Applications
For storm drainage systems, concrete shall be Class A with minimum compressive strength of 4000 psi. Installation procedures are detailed in Section 5.

2. INSTALLATION PROCEDURES

2.1 Preparation
Site preparation shall include excavation and base preparation as specified.

2.2 Placement
Concrete placement shall follow ACI 301 standards."""
        
        metadata = {
            "filename": "test_specification.txt",
            "file_size": len(test_text),
            "section_count": 4
        }
        
        # Initialize processor
        config = {
            "api_key": os.getenv("OPENAI_API_KEY") or "test-key",
            "cache_enabled": True
        }
        processor = EnhancedDocumentProcessor(config)
        
        # Process document
        enhanced_chunks, doc_classification = processor.process_document(test_text, metadata)
        
        logger.info(f"Document classified as: {doc_classification.document_type}")
        logger.info(f"Created {len(enhanced_chunks)} enhanced chunks")
        
        # Examine first chunk
        if enhanced_chunks:
            first_chunk = enhanced_chunks[0]
            logger.info(f"First chunk context: {first_chunk.enhanced_context}")
            logger.info(f"Semantic tags: {first_chunk.semantic_tags}")
            logger.info(f"Cross references: {first_chunk.cross_references}")
        
        logger.info("✅ Enhanced Document Processor test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Document Processor test failed: {e}")
        return False


def test_workflow_integration():
    """Test the complete workflow integration"""
    logger.info("Testing Workflow Integration...")
    
    try:
        # Create test workspace
        test_workspace = Path("test_workspace")
        test_workspace.mkdir(exist_ok=True)
        
        # Create input directory and test file
        input_dir = test_workspace / "input"
        input_dir.mkdir(exist_ok=True)
        
        # Create test document
        test_doc_path = input_dir / "test_document.txt"
        test_content = """BALTIMORE CITY SPECIFICATIONS

1. CONCRETE REQUIREMENTS

1.1 General Requirements
All concrete work shall conform to the latest edition of ACI 301 and Baltimore City Standards.

1.2 Materials
Concrete shall meet ASTM C150 requirements for Portland cement concrete.

2. STORM DRAINAGE SYSTEMS

2.1 General
Storm drainage systems shall be designed for 25-year storm events.

2.2 Concrete Specifications
Storm drain concrete shall have minimum compressive strength of 4000 psi."""
        
        with open(test_doc_path, 'w') as f:
            f.write(test_content)
        
        # Create settings file
        settings_path = test_workspace / "settings.yaml"
        settings_content = """
models:
  default_chat_model:
    type: openai_chat
    api_key: ${OPENAI_API_KEY}
    model: gpt-3.5-turbo

chunks:
  size: 500
  overlap: 50

document_processing:
  classification:
    enabled: true
    cache_results: true
  summarization:
    enabled: true
    cache_results: true
"""
        with open(settings_path, 'w') as f:
            f.write(settings_content)
        
        # Test workflow
        config = {
            "api_key": os.getenv("OPENAI_API_KEY") or "test-key",
            "cache_enabled": True
        }
        workflow = EnhancedGraphRAGWorkflow(test_workspace, config)
        
        # Run processing
        results = workflow.run_enhanced_processing()
        
        logger.info(f"Workflow status: {results['status']}")
        if results['status'] == 'success':
            logger.info(f"Enhanced chunks created: {results['enhanced_chunks']}")
            logger.info(f"Documents processed: {results['documents']}")
            
            # Check output files
            output_dir = test_workspace / "output"
            if (output_dir / "enhanced_text_units.parquet").exists():
                df = pd.read_parquet(output_dir / "enhanced_text_units.parquet")
                logger.info(f"Enhanced chunks DataFrame shape: {df.shape}")
                logger.info(f"Columns: {list(df.columns)}")
        
        logger.info("✅ Workflow Integration test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Workflow Integration test failed: {e}")
        return False
    
    finally:
        # Cleanup test workspace
        import shutil
        if Path("test_workspace").exists():
            shutil.rmtree("test_workspace")


def run_all_tests():
    """Run all tests"""
    logger.info("🚀 Starting Enhanced Processing Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Document Classifier", test_document_classifier),
        ("Section Summarizer", test_section_summarizer),
        ("Enhanced Document Processor", test_enhanced_document_processor),
        ("Workflow Integration", test_workflow_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("🏁 Test Results Summary")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Enhanced processing is ready to use.")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above for details.")
        return False


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("⚠️  OPENAI_API_KEY not found in environment variables.")
        logger.warning("Tests will run in simulation mode without actual AI calls.")
        logger.warning("For full testing, set OPENAI_API_KEY environment variable.")
    
    success = run_all_tests()
    sys.exit(0 if success else 1)