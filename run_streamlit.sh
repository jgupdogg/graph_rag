#!/bin/bash
# Simple script to run the Streamlit app

echo "Starting Baltimore City GraphRAG Explorer..."
echo "This will open a web interface to explore the knowledge graph."
echo ""
echo "You can access the app at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

# Activate virtual environment and run streamlit
source venv/bin/activate
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0