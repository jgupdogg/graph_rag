#!/usr/bin/env python3
"""
Test the audio lecture feature integration with Streamlit app.
"""

import streamlit as st
from audio_lecture_generator import AudioLectureGenerator, LectureDetailLevel
from openai_tts_simple import generate_tts_audio

st.set_page_config(page_title="Audio Lecture Test", layout="wide")

st.title("🎙️ Audio Lecture Generator Test")

# Test data
document_summary = {
    'display_name': 'Sample Document',
    'summary': 'This is a test document about artificial intelligence and machine learning.'
}

section_summaries = {
    "Chapter 1: Introduction": "An introduction to the key concepts.",
    "Chapter 2: Main Content": "The main content of the document.",
    "Chapter 3: Conclusion": "Summary and conclusions."
}

# UI Controls
col1, col2 = st.columns(2)

with col1:
    detail_level = st.select_slider(
        "Detail Level",
        options=[
            LectureDetailLevel.OVERVIEW,
            LectureDetailLevel.MAIN_POINTS,
            LectureDetailLevel.STANDARD,
            LectureDetailLevel.DETAILED,
            LectureDetailLevel.COMPREHENSIVE
        ],
        value=LectureDetailLevel.STANDARD,
        format_func=lambda x: x.name
    )

with col2:
    voice = st.selectbox(
        "Voice",
        options=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        index=4
    )

if st.button("Generate Audio Lecture"):
    generator = AudioLectureGenerator()
    
    with st.spinner("Generating lecture script..."):
        lecture_sections, full_script = generator.generate_lecture_script(
            document_summary=document_summary,
            section_summaries=section_summaries,
            detail_level=detail_level
        )
    
    st.success(f"Generated {len(lecture_sections)} sections")
    
    # Show script
    with st.expander("View Script"):
        st.text_area("Lecture Script", full_script, height=300)
    
    # Generate audio
    with st.spinner("Generating audio..."):
        audio_bytes, error = generate_tts_audio(full_script, voice=voice)
    
    if audio_bytes:
        st.success("Audio generated!")
        st.audio(audio_bytes, format="audio/mp3")
        
        st.download_button(
            "Download MP3",
            data=audio_bytes,
            file_name="test_lecture.mp3",
            mime="audio/mpeg"
        )
    else:
        st.error(f"Failed to generate audio: {error}")

st.info("This is a test interface for the audio lecture generator feature.")