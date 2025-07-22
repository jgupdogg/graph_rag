#!/bin/bash

echo "🚀 Starting GraphRAG with TTS support in Docker..."

# Make sure API keys are set
if [ -z "$OPENAI_API_KEY" ] && [ -z "$GRAPHRAG_API_KEY" ]; then
    echo "⚠️ Warning: No API keys set. Please set OPENAI_API_KEY or GRAPHRAG_API_KEY"
fi

# Build and run the Docker container
docker-compose up --build

echo "✅ GraphRAG with TTS is now running at http://localhost:8502"