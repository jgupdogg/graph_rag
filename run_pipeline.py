#!/usr/bin/env python3
# GraphRAG Processing Pipeline for Baltimore Specs
import os
import subprocess
import time
from datetime import datetime

def run_graphrag_pipeline():
    print(f"🚀 Starting GraphRAG processing: {datetime.now()}")
    
    # Set environment
    os.environ['GRAPHRAG_API_KEY'] = open('graphrag/.env').read().split('=')[1].strip()
    
    # Run indexing
    cmd = [
        'python', '-m', 'graphrag', 'index',
        '--root', 'graphrag',
        '--verbose'
    ]
    
    print("📊 Running GraphRAG indexing...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ GraphRAG indexing completed successfully!")
        print("📁 Output files created in graphrag/output/")
        return True
    else:
        print(f"❌ GraphRAG indexing failed: {result.stderr}")
        return False

if __name__ == "__main__":
    run_graphrag_pipeline()
