FROM python:3.12-slim

# Install system dependencies including TTS support
RUN apt-get update && apt-get install -y \
    espeak espeak-data libespeak1 libespeak-dev \
    speech-dispatcher \
    festival festvox-kallpc16k \
    alsa-utils \
    pulseaudio \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create directories for GraphRAG
RUN mkdir -p workspaces cache

# Expose port for Streamlit
EXPOSE 8501

# Configure Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Start speech dispatcher and Streamlit
CMD ["sh", "-c", "speech-dispatcher & streamlit run streamlit_app.py"]