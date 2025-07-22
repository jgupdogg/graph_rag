"""
Simple OpenAI Text-to-Speech integration for GraphRAG Explorer.
Clean, minimal implementation.
"""

import streamlit as st
import openai
import os
from datetime import datetime
import logging
from pathlib import Path

# Load environment variables - force load
def load_env_vars():
    """Force load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass
    
    # Manual loading as backup
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Load environment on import
load_env_vars()

logger = logging.getLogger(__name__)

def create_openai_tts_client():
    """Create OpenAI client for TTS."""
    try:
        # Check for API keys
        openai_key = os.getenv("OPENAI_API_KEY")
        graphrag_key = os.getenv("GRAPHRAG_API_KEY")
        
        api_key = openai_key or graphrag_key
        
        # Try Streamlit secrets as fallback
        if not api_key:
            try:
                api_key = st.secrets.get("OPENAI_API_KEY", "")
            except:
                pass
        
        if api_key:
            return openai.OpenAI(api_key=api_key)
        else:
            return None
    except Exception as e:
        logger.error(f"Failed to create OpenAI client: {e}")
        return None

def generate_tts_audio(text, voice="nova", model="tts-1"):
    """Generate TTS audio using OpenAI."""
    client = create_openai_tts_client()
    
    if not client:
        return None, "OpenAI API key not configured"
    
    if not text.strip():
        return None, "No text provided"
    
    try:
        # Limit text length for TTS
        text_for_tts = text[:4000] if len(text) > 4000 else text
        
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text_for_tts
        )
        
        return response.content, None
        
    except Exception as e:
        error_msg = f"TTS generation failed: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def render_simple_tts_section(text, unique_key=""):
    """Render a simple TTS section with OpenAI options."""
    
    st.markdown("### 🎤 Generate Audio (OpenAI TTS)")
    
    # Check if OpenAI is available
    client = create_openai_tts_client()
    if not client:
        st.warning("⚠️ OpenAI TTS not available. Please configure your OPENAI_API_KEY.")
        return
    
    # Voice selection
    voice_options = {
        "alloy": "Alloy (Neutral)",
        "echo": "Echo (Clear)",
        "fable": "Fable (Warm)",
        "onyx": "Onyx (Deep)",
        "nova": "Nova (Bright)",
        "shimmer": "Shimmer (Soft)"
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_voice = st.selectbox(
            "Choose Voice:",
            options=list(voice_options.keys()),
            format_func=lambda x: voice_options[x],
            index=4,  # Default to nova
            key=f"voice_select_{unique_key}"
        )
    
    with col2:
        selected_model = st.selectbox(
            "Quality:",
            options=["tts-1", "tts-1-hd"],
            format_func=lambda x: "Standard (Fast)" if x == "tts-1" else "HD (High Quality)",
            key=f"model_select_{unique_key}"
        )
    
    # Text preview
    text_preview = text[:300] + "..." if len(text) > 300 else text
    st.text_area(
        "Text to convert:",
        value=text_preview,
        height=100,
        disabled=True,
        key=f"text_preview_{unique_key}"
    )
    
    if len(text) > 4000:
        st.warning(f"⚠️ Text will be truncated to 4000 characters (current: {len(text)})")
    
    # Generate button
    if st.button("🎤 Generate Audio", type="primary", key=f"generate_{unique_key}"):
        with st.spinner("🔄 Generating audio..."):
            audio_bytes, error = generate_tts_audio(text, selected_voice, selected_model)
            
            if audio_bytes:
                # Create filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tts_audio_{timestamp}.mp3"
                
                st.success("✅ Audio generated successfully!")
                
                # Play audio
                st.audio(audio_bytes, format="audio/mp3")
                
                # Download button
                st.download_button(
                    label="📥 Download MP3",
                    data=audio_bytes,
                    file_name=filename,
                    mime="audio/mpeg",
                    key=f"download_{unique_key}"
                )
                
            else:
                st.error(f"❌ Failed to generate audio: {error}")

def test_openai_tts_simple():
    """Test function for the simple TTS."""
    st.title("🔧 Simple OpenAI TTS Test")
    
    test_text = "Hello! This is a test of the simple OpenAI TTS integration. It should generate high-quality speech from this text."
    
    render_simple_tts_section(test_text, "test")

if __name__ == "__main__":
    test_openai_tts_simple()