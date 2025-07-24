#!/usr/bin/env python3
"""
Streamlit Interactive GraphRAG Multi-Document Explorer
Advanced interface to explore multiple document knowledge graphs
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import logging
import os
from datetime import datetime
import streamlit.components.v1 as components
from streamlit.components.v1 import html
from pyvis_graph import (
    create_pyvis_network, 
    render_pyvis_network,
    get_entity_type_options,
    get_degree_range,
    optimize_for_large_graph
)
from app_logic import (
    get_processor,
    get_processed_documents,
    get_all_documents,
    get_document_by_id,
    process_new_document,
    load_and_merge_graphs,
    delete_document,
    get_document_status,
    reprocess_failed_document,
    get_processing_logs,
    DocumentStatus,
    check_stuck_documents,
    save_enhanced_summary_to_metadata,
)
from query_logic import (
    query_documents,
    global_search,
    local_search,
    drift_search,
    basic_search,
    summary_search,
    enhanced_search,
    get_supported_methods,
    validate_query,
    add_to_chat_history,
    get_chat_history,
    clear_chat_history
)
from config_manager import check_api_key_availability
from openai_tts_simple import render_simple_tts_section

# Configure logging
logger = logging.getLogger(__name__)

def text_to_speech(text: str, rate: float = None, pitch: float = 1.0, voice_index: int = None):
    """
    Generates an invisible HTML component to trigger browser-based text-to-speech.
    
    Args:
        text (str): The text to be spoken.
        rate (float): The speed of the speech (0.1 to 10.0). Uses session state if None.
        pitch (float): The pitch of the speech (0.0 to 2.0). Defaults to 1.0.
        voice_index (int): Index of the voice to use. Uses session state if None.
    """
    # Use session state values if not provided
    if rate is None:
        rate = st.session_state.get('summary_tts_rate', 0.9)
    if voice_index is None:
        voice_index = st.session_state.get('selected_voice_index', 0)
    
    # Use json.dumps to safely escape the text for JavaScript
    safe_text = json.dumps(text)
    
    # The JavaScript code to be executed in the browser
    js_code = f"""
        <script>
            console.log('TTS: Script loaded, attempting to speak...');
            
            // Function to speak the text
            function speakText(text, rate, pitch) {{
                console.log('TTS: speakText called with text length:', text.length);
                console.log('TTS: Text preview:', text.substring(0, 50));
                
                // Check if speech synthesis is supported
                if (!('speechSynthesis' in window)) {{
                    console.error('TTS: Speech synthesis not supported in this browser');
                    alert('Text-to-Speech is not supported in your browser. Please try Chrome, Firefox, or Safari.');
                    return;
                }}
                
                console.log('TTS: Speech synthesis is supported');
                
                // Cancel any ongoing speech
                window.speechSynthesis.cancel();
                console.log('TTS: Previous speech cancelled');
                
                // Function to actually speak
                function doSpeak() {{
                    console.log('TTS: Creating utterance...');
                    
                    // Create a new SpeechSynthesisUtterance object
                    const utterance = new SpeechSynthesisUtterance(text);
                    
                    // Set properties
                    utterance.lang = 'en-US';
                    utterance.rate = rate;
                    utterance.pitch = pitch;
                    utterance.volume = 1.0;
                    
                    console.log('TTS: Utterance configured - rate:', rate, 'pitch:', pitch);
                    
                    // Add event handlers
                    utterance.onstart = function(event) {{
                        console.log('TTS: ✅ Speech STARTED successfully!');
                    }};
                    
                    utterance.onend = function(event) {{
                        console.log('TTS: Speech ended normally');
                    }};
                    
                    utterance.onerror = function(event) {{
                        console.error('TTS: ❌ Speech ERROR:', event.error, event);
                        alert('TTS Error: ' + event.error);
                    }};
                    
                    utterance.onpause = function(event) {{
                        console.log('TTS: Speech paused');
                    }};
                    
                    utterance.onresume = function(event) {{
                        console.log('TTS: Speech resumed');
                    }};
                    
                    utterance.onboundary = function(event) {{
                        console.log('TTS: Speech boundary reached');
                    }};
                    
                    // Get voices and log them
                    const voices = window.speechSynthesis.getVoices();
                    console.log('TTS: Available voices count:', voices.length);
                    
                    if (voices.length === 0) {{
                        console.warn('TTS: ⚠️ No voices available on this system');
                        alert(`TTS Error: No speech voices are available on your system.\\n\\nPossible solutions:\\n• On Linux: Install speech-dispatcher and espeak\\n• Try a different browser (Chrome/Edge usually work best)\\n• Check your system's accessibility settings\\n• Some corporate networks block TTS`);
                        return;
                    }}
                    
                    console.log('TTS: First few voices:');
                    voices.slice(0, Math.min(3, voices.length)).forEach((voice, index) => {{
                        console.log(`  ${{index}}: ${{voice.name}} (${{voice.lang}}) - ${{voice.localService ? 'Local' : 'Remote'}}`);
                    }});
                    
                    // Use the selected voice index or window selection
                    const voiceIndex = {voice_index} >= 0 ? {voice_index} : (window.selectedVoiceIndex || 0);
                    
                    if (voices.length > voiceIndex && voiceIndex >= 0) {{
                        utterance.voice = voices[voiceIndex];
                        console.log('TTS: Using selected voice:', voices[voiceIndex].name);
                    }} else {{
                        // Fallback to English voice
                        const englishVoice = voices.find(voice => voice.lang.includes('en')) || voices[0];
                        if (englishVoice) {{
                            utterance.voice = englishVoice;
                            console.log('TTS: Using fallback voice:', englishVoice.name);
                        }}
                    }}
                    
                    // Attempt to speak
                    console.log('TTS: 🎤 Calling speechSynthesis.speak()...');
                    try {{
                        window.speechSynthesis.speak(utterance);
                        console.log('TTS: speak() called successfully');
                        
                        // Check if it's actually speaking
                        setTimeout(() => {{
                            console.log('TTS: Speaking status check:', window.speechSynthesis.speaking);
                            console.log('TTS: Pending status check:', window.speechSynthesis.pending);
                        }}, 500);
                        
                    }} catch (error) {{
                        console.error('TTS: Exception in speak():', error);
                        alert('TTS Exception: ' + error.message);
                    }}
                }}
                
                // Wait for voices to load if needed
                if (window.speechSynthesis.getVoices().length === 0) {{
                    console.log('TTS: Waiting for voices to load...');
                    window.speechSynthesis.addEventListener('voiceschanged', function() {{
                        console.log('TTS: Voices loaded, attempting speech...');
                        doSpeak();
                    }}, {{ once: true }});
                    
                    // Fallback timeout
                    setTimeout(() => {{
                        if (window.speechSynthesis.getVoices().length === 0) {{
                            console.warn('TTS: Voices still not loaded, trying anyway...');
                            doSpeak();
                        }}
                    }}, 1000);
                }} else {{
                    console.log('TTS: Voices already available');
                    doSpeak();
                }}
            }}
            
            // Add a small delay then speak
            setTimeout(() => {{
                speakText({safe_text}, {rate}, {pitch});
            }}, 100);
        </script>
    """
    
    # Use html() to execute the JavaScript with invisible component
    html(js_code, height=0)

# Page configuration
st.set_page_config(
    page_title="GraphRAG Explorer - Interactive Knowledge Graphs",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/microsoft/graphrag',
        'Report a bug': "https://github.com/microsoft/graphrag/issues",
        'About': "# GraphRAG Explorer\nInteractive knowledge graph exploration powered by Microsoft GraphRAG"
    }
)

# Minimal CSS for basic styling
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Basic button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Force cache invalidation
if "ui_version" not in st.session_state:
    st.session_state.ui_version = "2.0"


# Simple header without complex styling
st.title("🧠 GraphRAG Explorer")
st.caption("Interactive knowledge graph exploration and document analysis")

# Initialize application
get_processor().init_db()

# Check API key availability
if not check_api_key_availability():
    st.error("⚠️ GraphRAG API key not found. Please set GRAPHRAG_API_KEY or OPENAI_API_KEY environment variable.")
    st.stop()

# Cache data loading
@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_json()})
def load_multi_document_data(selected_doc_ids):
    """Load GraphRAG output data from multiple documents."""
    if not selected_doc_ids:
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        entities, relationships = load_and_merge_graphs(selected_doc_ids)
        return entities, relationships
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_json()})
def create_networkx_graph(entities, relationships):
    """Create NetworkX graph from entities and relationships."""
    G = nx.Graph()
    
    # Add nodes
    for _, entity in entities.iterrows():
        G.add_node(
            entity['title'],
            type=entity.get('type', 'unknown'),
            description=entity.get('description', ''),
            degree=entity.get('degree', 0),
            frequency=entity.get('frequency', 0)
        )
    
    # Add edges
    for _, rel in relationships.iterrows():
        if rel['source'] in G and rel['target'] in G:
            G.add_edge(
                rel['source'],
                rel['target'],
                weight=rel.get('weight', 1),
                description=rel.get('description', ''),
                combined_degree=rel.get('combined_degree', 0)
            )
    
    return G

def load_community_reports(selected_doc_ids):
    """Load community reports from selected documents."""
    if not selected_doc_ids:
        return pd.DataFrame()
    
    all_reports = []
    
    try:
        # Get document info from database
        with get_processor().db_manager as cursor:
            placeholders = ','.join('?' for _ in selected_doc_ids)
            query = f"SELECT id, workspace_path, display_name FROM documents WHERE id IN ({placeholders}) AND status = ?"
            cursor.execute(query, selected_doc_ids + [DocumentStatus.COMPLETED])
            docs = cursor.fetchall()
        
        for doc_id, workspace_path, display_name in docs:
            workspace = Path(workspace_path)
            
            # Look for community reports file
            reports_file = workspace / "output" / "community_reports.parquet"
            
            # Try alternative location
            if not reports_file.exists():
                reports_file = workspace / "output" / "artifacts" / "community_reports.parquet"
            
            if reports_file.exists():
                reports_df = pd.read_parquet(reports_file)
                # Add source document information
                reports_df['source_document'] = display_name
                reports_df['source_doc_id'] = doc_id
                all_reports.append(reports_df)
                logger.info(f"Loaded {len(reports_df)} community reports from {display_name}")
        
        if all_reports:
            # Merge all reports
            merged_reports = pd.concat(all_reports, ignore_index=True)
            return merged_reports
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading community reports: {e}")
        return pd.DataFrame()

def generate_ai_document_summary(selected_doc_ids):
    """Generate AI-powered comprehensive document summary."""
    if not selected_doc_ids:
        return None
    
    try:
        # Get document info
        all_docs = get_all_documents()
        selected_docs = [doc for doc in all_docs if doc['id'] in selected_doc_ids]
        
        if not selected_docs:
            return None
        
        # Focus on the first document for now (can be enhanced for multiple docs later)
        doc = selected_docs[0]
        doc_id = doc['id']
        workspace_path = Path(doc['workspace_path'])
        
        # Get full document details including the initial summary
        doc_details = get_document_by_id(doc_id)
        initial_summary = doc_details.get('summary', '') if doc_details else ''
        
        # Check for existing summary cache
        summary_cache_file = workspace_path / "ai_document_summary_cache.json"
        if summary_cache_file.exists():
            try:
                with open(summary_cache_file, 'r') as f:
                    cached_data = json.load(f)
                # Check if cache is recent (less than 7 days old) and uses new format
                from datetime import timedelta
                cache_time = datetime.fromisoformat(cached_data.get('generated_at', '2000-01-01'))
                # Force regeneration if cache doesn't have the new user-familiar format
                has_new_format = 'Key Topics Explored' in cached_data.get('ai_summary', '')
                if (datetime.now() - cache_time < timedelta(days=7) and 
                    cached_data.get('includes_initial_summary', False) and
                    has_new_format):
                    return cached_data
            except (json.JSONDecodeError, ValueError, KeyError):
                pass  # Continue to regenerate if cache is invalid
        
        # Get document text excerpt (first 1000 characters)
        text_excerpt = ""
        input_dir = workspace_path / "input"
        for text_file in input_dir.glob("*.txt"):
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text_excerpt = f.read(2500)  # Increased from 1000 to provide more context
                break
            except Exception:
                continue
        
        # Get top community reports
        reports_df = load_community_reports([doc_id])
        top_reports = []
        if not reports_df.empty:
            # Sort by size and rank to get most comprehensive reports
            if 'size' in reports_df.columns and 'rank' in reports_df.columns:
                sorted_reports = reports_df.sort_values(['size', 'rank'], ascending=[False, False])
                top_reports = sorted_reports.head(10)  # Get top 10 largest communities
        
        # Gather document processing metrics
        entities, relationships = load_multi_document_data([doc_id])
        entity_count = len(entities) if not entities.empty else 0
        relationship_count = len(relationships) if not relationships.empty else 0
        entity_types = entities['type'].nunique() if not entities.empty and 'type' in entities.columns else 0
        
        # Count chunks from the summary vector store if available
        chunk_count = 0
        try:
            vector_cache_file = workspace_path / "cache" / "chunk_summaries.json"
            if vector_cache_file.exists():
                with open(vector_cache_file, 'r') as f:
                    chunk_data = json.load(f)
                    chunk_count = len(chunk_data.get("summaries", []))
        except:
            pass
        
        # Check for existing classification cache
        classification_data = ""
        doc_summary_cache = workspace_path / "document_summary_cache.json"
        if doc_summary_cache.exists():
            try:
                with open(doc_summary_cache, 'r') as f:
                    cache_data = json.load(f)
                    if 'overview' in cache_data:
                        classification_data = cache_data['overview']
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Create comprehensive prompt for AI summary
        prompt_parts = [
            f"You are analyzing a document for a user who is already familiar with it. Provide a practical analysis that highlights the most important aspects and insights discovered through processing.",
            f"\n## Document: {doc['display_name']}"
        ]
        
        # Include initial two-stage summary as the foundation
        if initial_summary:
            prompt_parts.append(f"\n## Initial Document Summary (Two-Stage Analysis):\n{initial_summary}")
        
        # Add document processing metrics
        prompt_parts.append(f"\n## Document Processing Metrics:")
        prompt_parts.append(f"- **Entities Extracted:** {entity_count:,}")
        prompt_parts.append(f"- **Relationships Identified:** {relationship_count:,}")
        prompt_parts.append(f"- **Entity Types:** {entity_types}")
        prompt_parts.append(f"- **Document Chunks:** {chunk_count}")
        prompt_parts.append(f"- **Community Reports:** {len(top_reports)}")
        prompt_parts.append(f"- **Processing Time:** {doc.get('processing_time', 'N/A')}s")
        
        if classification_data:
            prompt_parts.append(f"\n## Document Classification:\n{classification_data}")
        
        if text_excerpt and not initial_summary:  # Only include excerpt if no initial summary
            prompt_parts.append(f"\n## Document Content (First 2500 characters):\n{text_excerpt}")
        
        if len(top_reports) > 0:
            prompt_parts.append("\n## Key Community Insights (Top 10 Largest Communities):")
            for i, (_, report) in enumerate(top_reports.iterrows(), 1):
                title = report.get('title', 'Untitled')
                summary = report.get('summary', 'No summary available')
                size = report.get('size', 'Unknown')
                rank = report.get('rank', 0)
                prompt_parts.append(f"\n{i}. **{title}** (Size: {size}, Rank: {rank:.1f}): {summary}")
        
        prompt_parts.append("""
\n## Please provide an enhanced document analysis that assumes the user is familiar with their document:

### High-Level Summary
Provide a concise 2-3 sentence overview of what this document covers and its main purpose.

### Key Topics Explored
Identify and briefly explain 3-4 of the most significant topics or themes found in the document based on the analysis:
- Topic 1: Brief explanation of what was found
- Topic 2: Brief explanation of what was found  
- Topic 3: Brief explanation of what was found
- Topic 4: Brief explanation of what was found (if applicable)

### Key Insights & Important Points
List 6-8 bullet points highlighting:
- Most important information and conclusions from the document
- Significant patterns or relationships discovered through analysis
- Notable findings from the community analysis
- Any actionable insights or recommendations

Keep the entire response focused and practical, approximately 300-400 words total. Avoid redundant explanations of document type or basic structure.""")
        
        full_prompt = "".join(prompt_parts)
        
        # Use query_documents with o1-mini for enhanced reasoning
        result = query_documents(
            doc_ids=[doc_id],
            query=full_prompt,
            method="global",  # Use global search for comprehensive overview
            model="o1-mini"   # Use reasoning model for complex summarization
        )
        
        if result.success:
            summary_data = {
                "document_id": doc_id,
                "document_name": doc['display_name'],
                "ai_summary": result.response,
                "classification": classification_data,
                "generated_at": datetime.now().isoformat(),
                "method_used": "global_o1-mini",
                "source_reports_count": len(top_reports),
                "includes_initial_summary": bool(initial_summary),
                "metrics": {
                    "entities": entity_count,
                    "relationships": relationship_count,
                    "entity_types": entity_types,
                    "chunks": chunk_count,
                    "communities": len(top_reports),
                    "processing_time": doc.get('processing_time', 0)
                }
            }
            
            # Cache the result
            try:
                with open(summary_cache_file, 'w') as f:
                    json.dump(summary_data, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to cache AI summary: {e}")
            
            # Save enhanced summary to document metadata
            save_enhanced_summary_to_metadata(doc_id, summary_data)
            
            return summary_data
        else:
            logger.error(f"Failed to generate AI summary: {result.error_message}")
            return None
            
    except Exception as e:
        logger.error(f"Error generating AI document summary: {e}")
        return None

def create_plotly_graph(G, layout_type="spring"):
    """Create interactive Plotly graph visualization."""
    
    # Different layout algorithms
    if layout_type == "spring":
        pos = nx.spring_layout(G, k=3, iterations=50)
    elif layout_type == "circular":
        pos = nx.circular_layout(G)
    elif layout_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Edge info for hover
        edge_desc = G[edge[0]][edge[1]].get('description', 'No description')
        edge_info.append(f"{edge[0]} → {edge[1]}: {edge_desc}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    node_size = []
    node_color = []
    
    # Color mapping for entity types
    color_map = {
        'PERSON': '#FF6B6B',
        'ORGANIZATION': '#4ECDC4',
        'LOCATION': '#45B7D1',
        'EVENT': '#F7DC6F',
        'CONCEPT': '#82E0AA',
        'unknown': '#95A5A6'
    }
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node info
        node_data = G.nodes[node]
        node_type = node_data.get('type', 'unknown')
        degree = G.degree(node)
        description = node_data.get('description', 'No description')
        frequency = node_data.get('frequency', 0)
        
        # Node text (labels)
        node_text.append(node)
        
        # Hover info
        hover_text = f"<b>{node}</b><br>"
        hover_text += f"Type: {node_type}<br>"
        hover_text += f"Connections: {degree}<br>"
        hover_text += f"Frequency: {frequency}<br>"
        if description:
            hover_text += f"Description: {description[:200]}..."
        
        node_info.append(hover_text)
        
        # Node size based on degree
        node_size.append(max(degree * 10, 20))
        
        # Node color based on type
        node_color.append(color_map.get(node_type, color_map['unknown']))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="middle center",
        textfont=dict(size=10),
        hoverinfo='text',
        hovertext=node_info,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title="Baltimore City Knowledge Graph",
        title_font_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=600
    )
    
    return fig

def render_document_management_section():
    """Render document management section in sidebar."""
    # Enhanced sidebar header with custom styling
    st.sidebar.markdown('<p class="section-header">📄 Document Management</p>', unsafe_allow_html=True)
    
    # Upload new document with enhanced styling
    with st.sidebar.container():
        st.markdown("**Upload New Document**")
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=["pdf", "txt"],
            accept_multiple_files=False,
            help="Supported formats: PDF, TXT (max 200MB)"
        )
    
    if uploaded_file:
        with st.sidebar.container():
            display_name = st.text_input(
                "Document name (optional)",
                value=uploaded_file.name,
                key="doc_name",
                help="Custom name for easier identification"
            )
            
            if st.button("📤 Process Document", type="primary", use_container_width=True):
                with st.spinner("🔄 Starting document processing..."):
                    result = process_new_document(uploaded_file, display_name)
                    st.success(result)
                    st.rerun()
    
    # Show processing status with enhanced styling
    all_docs = get_processor().get_all_documents()
    processing_docs = [doc for doc in all_docs if doc['status'] == 'PROCESSING']
    failed_docs = [doc for doc in all_docs if doc['status'] == 'ERROR']
    
    if processing_docs:
        st.sidebar.markdown('<p class="section-header">⏳ Processing Status</p>', unsafe_allow_html=True)
        for doc in processing_docs:
            with st.sidebar.container():
                st.markdown(f'<div class="loading-pulse">🔄 Processing: **{doc["display_name"]}**</div>', unsafe_allow_html=True)
                st.progress(0.5)  # Indeterminate progress
    
    # Show failed documents with retry option
    if failed_docs:
        st.sidebar.markdown('<p class="section-header">❌ Failed Documents</p>', unsafe_allow_html=True)
        for doc in failed_docs:
            with st.sidebar.expander(f"❌ {doc['display_name']}", expanded=False):
                if doc['error_message']:
                    st.error(f"Error: {doc['error_message'][:100]}...")
                
                if st.button("🔄 Retry Processing", key=f"retry_{doc['id']}", type="secondary", use_container_width=True):
                    with st.spinner("🔄 Retrying processing..."):
                        result = reprocess_failed_document(doc['id'])
                        st.sidebar.success(result)
                        st.rerun()
    
    # Document list and selection with enhanced styling
    processed_docs = get_processed_documents()
    
    if processed_docs:
        st.sidebar.markdown('<p class="section-header">📋 Select Documents</p>', unsafe_allow_html=True)
        
        # Create options for multiselect with better formatting
        doc_options = {}
        for doc in processed_docs:
            label = f"📄 {doc['display_name']} ({doc['created_at'][:10]})"
            doc_options[label] = doc['id']
        
        with st.sidebar.container():
            selected_doc_names = st.multiselect(
                "Choose documents to explore:",
                options=doc_options.keys(),
                key="selected_docs",
                help=f"Select from {len(processed_docs)} available document(s)"
            )
        
        # Get selected document IDs
        selected_doc_ids = [doc_options[name] for name in selected_doc_names]
        
        # Document management buttons with enhanced styling
        if selected_doc_names:
            st.sidebar.markdown('<p class="section-header">🔧 Document Actions</p>', unsafe_allow_html=True)
            
            with st.sidebar.container():
                col1, col2 = st.columns(2)
                
                with col1:
                    # Show processing logs
                    if st.button("📋 Logs", use_container_width=True, help="View processing logs"):
                        st.session_state.show_logs = True
                        st.session_state.log_doc_ids = selected_doc_ids
                
                with col2:
                    # Delete selected documents
                    if st.button("🗑️ Delete", type="secondary", use_container_width=True, help="Delete selected documents"):
                        for doc_id in selected_doc_ids:
                            if delete_document(doc_id):
                                st.success("✅ Document deleted")
                            else:
                                st.error("❌ Failed to delete document")
                        st.rerun()
        
        
        return selected_doc_ids
    else:
        st.sidebar.info("No processed documents found. Upload a document to get started.")
        return []

def render_graph_controls(entities, _relationships):
    """Render graph control section in sidebar."""
    st.sidebar.markdown('<p class="section-header">🎛️ Graph Controls</p>', unsafe_allow_html=True)
    
    # Filtering controls with enhanced styling
    if not entities.empty:
        with st.sidebar.expander("🔍 Filters", expanded=True):
            # Entity type filter
            entity_types = get_entity_type_options(entities)
            selected_types = st.multiselect(
                "Entity Types",
                entity_types,
                default=entity_types,
                key="entity_types",
                help="Filter by entity types to focus your analysis"
            )
            
            # Degree range filter
            min_degree, max_degree = get_degree_range(entities)
            if min_degree < max_degree:
                degree_range = st.slider(
                    "Connection Range",
                    min_value=min_degree,
                    max_value=max_degree,
                    value=(min_degree, max_degree),
                    key="degree_range",
                    help="Filter entities by number of connections"
                )
            else:
                degree_range = (min_degree, max_degree)
            
            # Search filter
            search_term = st.text_input(
                "Search Entities", 
                key="search",
                placeholder="Type to search entities...",
                help="Search for specific entities by name"
            )
        
        with st.sidebar.expander("⚙️ Layout Settings", expanded=False):
            # Layout options
            layout_option = st.selectbox(
                "Layout Algorithm",
                ["barnes_hut", "force_atlas_2", "hierarchical"],
                index=0,
                key="layout",
                help="Choose visualization algorithm"
            )
            
            # Performance optimization
            optimize_graph = st.checkbox(
                "🚀 Performance Mode",
                value=len(entities) > 100,
                key="optimize",
                help="Show only top 100 most connected nodes for better performance"
            )
        
        return selected_types, degree_range, search_term, layout_option, optimize_graph
    else:
        st.sidebar.info("📊 Load documents to see graph controls")
        return [], (0, 0), "", "barnes_hut", False


def render_universal_sidebar():
    """Render universal sidebar with complete document management."""
    
    # 1. Document Upload Section
    st.sidebar.markdown('<p class="section-header">📤 Upload Document</p>', unsafe_allow_html=True)
    
    with st.sidebar.expander("Upload New Document", expanded=False):
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=["pdf", "txt"],
            accept_multiple_files=False,
            help="Supported formats: PDF, TXT (max 200MB)",
            key="sidebar_upload"
        )
        
        if uploaded_file:
            display_name = st.text_input(
                "Document name (optional)",
                value=uploaded_file.name,
                key="sidebar_doc_name",
                help="Custom name for easier identification"
            )
            
            if st.button("📤 Process Document", type="primary", use_container_width=True):
                with st.spinner("🔄 Starting document processing..."):
                    result = process_new_document(uploaded_file, display_name)
                    st.success(result)
                    st.rerun()
    
    # 2. Processing Status
    all_docs = get_all_documents()
    processing_docs = [doc for doc in all_docs if doc['status'] == 'PROCESSING']
    failed_docs = [doc for doc in all_docs if doc['status'] == 'ERROR']
    
    if processing_docs:
        st.sidebar.markdown('<p class="section-header">⏳ Processing Status</p>', unsafe_allow_html=True)
        for doc in processing_docs:
            with st.sidebar.container():
                st.sidebar.markdown(f'<div class="loading-pulse">🔄 Processing: **{doc["display_name"]}**</div>', unsafe_allow_html=True)
                st.sidebar.progress(0.5)  # Indeterminate progress
    
    if failed_docs:
        st.sidebar.markdown('<p class="section-header">❌ Failed Documents</p>', unsafe_allow_html=True)
        for doc in failed_docs:
            with st.sidebar.expander(f"❌ {doc['display_name']}", expanded=False):
                if doc['error_message']:
                    st.error(f"Error: {doc['error_message'][:100]}...")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Retry", key=f"retry_sidebar_{doc['id']}", help="Retry processing"):
                        with st.spinner("🔄 Retrying processing..."):
                            result = reprocess_failed_document(doc['id'])
                            st.success(result)
                            st.rerun()
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_sidebar_{doc['id']}", type="secondary", help="Delete document"):
                        if delete_document(doc['id']):
                            st.success("✅ Document deleted")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete")
    
    
    # 3. Document Selection
    processed_docs = get_processed_documents()
    
    if processed_docs:
        st.sidebar.markdown('<p class="section-header">📄 Select Documents</p>', unsafe_allow_html=True)
        
        # Create options for multiselect with better formatting
        doc_options = {}
        for doc in processed_docs:
            label = f"📄 {doc['display_name']} ({doc['created_at'][:10]})"
            doc_options[label] = doc['id']
        
        selected_doc_names = st.sidebar.multiselect(
            "Choose documents to analyze:",
            options=doc_options.keys(),
            key="universal_selected_docs",
            help=f"Select from {len(processed_docs)} available document(s)"
        )
        
        # Get selected document IDs
        selected_doc_ids = [doc_options[name] for name in selected_doc_names]
        
        if selected_doc_ids:
            # Show selected document info and actions
            st.sidebar.success(f"✅ {len(selected_doc_ids)} document(s) selected")
            
            # Document actions
            with st.sidebar.expander("🔧 Document Actions", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📋 View Logs", help="View processing logs", use_container_width=True):
                        st.session_state.show_logs = True
                        st.session_state.log_doc_ids = selected_doc_ids
                
                with col2:
                    if st.button("🗑️ Delete Selected", type="secondary", help="Delete selected documents", use_container_width=True):
                        for doc_id in selected_doc_ids:
                            if delete_document(doc_id):
                                st.success("✅ Document deleted")
                            else:
                                st.error("❌ Failed to delete document")
                        st.rerun()
        else:
            st.sidebar.info("👆 Select documents to start analyzing")
        
        return selected_doc_ids
    else:
        st.sidebar.info("📤 No documents available. Upload your first document above.")
        return []

def render_document_summary_tab(selected_doc_ids):
    """Render document summary tab with community reports and metadata."""
    if not selected_doc_ids:
        st.info("👈 Select documents from the sidebar to view their summaries.")
        return
    
    # Load document metadata
    all_docs = get_all_documents()
    selected_docs = [doc for doc in all_docs if doc['id'] in selected_doc_ids]
    
    st.title("📊 Document Summary")
    
    # First, check if we have an enhanced summary in metadata
    doc_details = get_document_by_id(selected_docs[0]['id']) if selected_docs else None
    has_enhanced_summary = (doc_details and 
                           doc_details.get('metadata') and 
                           doc_details['metadata'].get('enhanced_summary') and
                           doc_details['metadata']['enhanced_summary'].get('content'))
    
    # AI-Generated Document Overview Section (moved to top)
    st.markdown("---")
    
    # Display enhanced summary if available, otherwise generate it
    if has_enhanced_summary:
        # Show the cached enhanced summary
        st.subheader("🤖 Comprehensive Document Analysis")
        enhanced_summary = doc_details['metadata']['enhanced_summary']
        st.success(enhanced_summary['content'])
        
        # Show metrics
        if enhanced_summary.get('metrics'):
            metrics = enhanced_summary['metrics']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Entities", f"{metrics.get('entities', 0):,}")
            with col2:
                st.metric("Relationships", f"{metrics.get('relationships', 0):,}")
            with col3:
                st.metric("Communities", metrics.get('communities', 0))
            with col4:
                st.metric("Processing Time", f"{metrics.get('processing_time', 0)}s")
    else:
        # Generate enhanced summary
        st.subheader("🤖 Comprehensive Document Analysis")
        with st.spinner("Generating comprehensive document overview..."):
            ai_summary_data = generate_ai_document_summary(selected_doc_ids)
        
        if ai_summary_data:
            # Display the AI summary with better contrast
            st.success(ai_summary_data['ai_summary'])
            
            # Show metrics
            if ai_summary_data.get('metrics'):
                metrics = ai_summary_data['metrics']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Entities", f"{metrics.get('entities', 0):,}")
                with col2:
                    st.metric("Relationships", f"{metrics.get('relationships', 0):,}")
                with col3:
                    st.metric("Communities", metrics.get('communities', 0))
                with col4:
                    st.metric("Processing Time", f"{metrics.get('processing_time', 0)}s")
        else:
            st.warning("⚠️ Unable to generate AI summary. The document analysis may take a moment to complete.")
    
    # Browser TTS button
    if (has_enhanced_summary and doc_details['metadata']['enhanced_summary'].get('content')) or (ai_summary_data and ai_summary_data.get('ai_summary')):
        summary_text = (doc_details['metadata']['enhanced_summary']['content'] if has_enhanced_summary 
                       else ai_summary_data['ai_summary'])
        if st.button("🔊 Browser TTS", key="summary_browser_tts", help="Read summary with browser TTS"):
            text_to_speech(summary_text)
        
        # OpenAI TTS Section for Summary
        render_simple_tts_section(summary_text, "summary")
    
    # Two-Stage Document Summaries Section (now expandable)
    st.markdown("---")
    with st.expander("📄 Initial Document Summaries", expanded=False):
        st.caption("Two-stage summaries generated during document processing")
        
        # Display summaries for each selected document
        for doc in selected_docs:
            doc_details = get_document_by_id(doc['id'])
            if doc_details and doc_details.get('summary'):
                st.markdown(f"### {doc_details['display_name']}")
                st.markdown(doc_details['summary'])
                
                # Document metadata
                st.markdown("**Document Details:**")
                meta_col1, meta_col2, meta_col3 = st.columns(3)
                
                with meta_col1:
                    st.metric("Status", doc_details.get('status', 'Unknown'))
                    if doc_details.get('file_size'):
                        file_size_mb = round(doc_details['file_size'] / (1024 * 1024), 2)
                        st.metric("File Size", f"{file_size_mb} MB")
                
                with meta_col2:
                    if doc_details.get('processing_time'):
                        st.metric("Processing Time", f"{doc_details['processing_time']}s")
                    if doc_details.get('created_at'):
                        created_date = doc_details['created_at'][:10]
                        st.metric("Created", created_date)
                
                with meta_col3:
                    summary_length = len(doc_details['summary'])
                    st.metric("Summary Length", f"{summary_length:,} chars")
                    word_count = len(doc_details['summary'].split())
                    st.metric("Word Count", f"{word_count:,} words")
                    
            else:
                st.info(f"📝 No summary available for {doc['display_name']}.")
    
    # Section Summaries (now expandable by default)
    st.markdown("---")
    with st.expander("📑 Document Section Summaries", expanded=False):
        st.caption("Section summaries organized by document structure")
    
        # Display section summaries for each selected document
        for doc in selected_docs:
            doc_details = get_document_by_id(doc['id'])
            if doc_details and doc_details.get('section_summaries'):
                st.markdown(f"### {doc_details['display_name']} - {len(doc_details.get('section_summaries', {}))} sections")
                section_summaries = doc_details['section_summaries']
                
                # Create a hierarchical display of sections
                for i, (section_path, summary) in enumerate(section_summaries.items()):
                    # Section header with hierarchy indication
                    hierarchy_level = section_path.count(' > ')
                    indent = "&nbsp;" * (hierarchy_level * 4)
                    
                    # Display section with clean formatting
                    st.markdown(f"{indent}**{section_path}**")
                    st.markdown(f"{indent}{summary}")
                    
                    # Add subtle separator between sections
                    if i < len(section_summaries) - 1:
                        st.markdown(f"{indent}―――")
            else:
                st.info(f"No section summaries available for {doc['display_name']}.")
    
    
    # Document Metadata Section (now expandable)
    st.markdown("---")
    with st.expander("📄 Document Information", expanded=False):
        for doc in selected_docs:
            st.markdown(f"### {doc['display_name']}")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Status", doc['status'])
            with col2:
                file_size_mb = doc.get('file_size', 0) / (1024 * 1024) if doc.get('file_size') else 0
                st.metric("File Size", f"{file_size_mb:.2f} MB")
            with col3:
                processing_time = doc.get('processing_time', 0) 
                st.metric("Processing Time", f"{processing_time}s" if processing_time else "N/A")
            with col4:
                created_date = doc['created_at'][:10] if doc.get('created_at') else "Unknown"
                st.metric("Created", created_date)
            
            # Load graph data to get statistics
            entities, relationships = load_multi_document_data([doc['id']])
            
            if not entities.empty:
                st.markdown("**📊 Graph Statistics:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Entities", len(entities))
                with col2:
                    st.metric("Relationships", len(relationships))
                with col3:
                    # Get unique entity types
                    entity_types = entities['type'].value_counts() if 'type' in entities.columns else pd.Series()
                    st.metric("Entity Types", len(entity_types))
    
    # Community Reports Section (expandable)
    st.markdown("---")
    with st.expander("📋 Community Reports", expanded=False):
        # Load community reports
        with st.spinner("Loading community reports..."):
            reports_df = load_community_reports(selected_doc_ids)
        
        if reports_df.empty:
            st.info("No community reports found for the selected documents.")
        else:
            st.success(f"Found {len(reports_df)} community reports")
        
            # Sort reports by rank (descending) and then by level
            if 'rank' in reports_df.columns:
                reports_df = reports_df.sort_values(['rank', 'level'], ascending=[False, True])
            
            # Display reports
            for idx, report in reports_df.iterrows():
                # Create a nice header with rank/rating
                rank = report.get('rank', 0)
                
                with st.expander(f"📝 {report.get('title', 'Untitled Report')} (Rank: {rank:.1f})", expanded=(idx < 3)):
                    # Report metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Source:** {report.get('source_document', 'Unknown')}")
                    with col2:
                        st.markdown(f"**Level:** {report.get('level', 'N/A')}")
                    with col3:
                        st.markdown(f"**Community ID:** {report.get('community', 'N/A')}")
                    
                    # Summary
                    if 'summary' in report and pd.notna(report['summary']):
                        st.markdown("**Summary:**")
                        st.info(report['summary'])
                    
                    # Rating explanation
                    if 'rating_explanation' in report and pd.notna(report['rating_explanation']):
                        st.markdown(f"**Rating Explanation:** {report['rating_explanation']}")
                    
                    # Findings
                    if 'findings' in report and report['findings'] is not None:
                        findings = report['findings']
                        if isinstance(findings, (list, np.ndarray)) and len(findings) > 0:
                            st.markdown("**Key Findings:**")
                            for finding in findings:
                                if isinstance(finding, dict):
                                    if 'summary' in finding:
                                        st.markdown(f"• **{finding['summary']}**")
                                    if 'explanation' in finding:
                                        st.markdown(f"  {finding['explanation']}")
                        
                    # Full content (optional, collapsed by default)
                    if 'full_content' in report and pd.notna(report['full_content']):
                        with st.expander("View Full Report", expanded=False):
                            st.markdown(report['full_content'])
    
    # Simple Read Aloud section at the bottom
    if selected_docs:
        st.markdown("---")
        st.subheader("🔊 Read Aloud")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🔊 Read All Summaries", help="Read all document summaries aloud"):
                # Combine all document summaries
                all_summaries = []
                for doc in selected_docs:
                    doc_details = get_document_by_id(doc['id'])
                    if doc_details and doc_details.get('summary'):
                        all_summaries.append(f"{doc_details['display_name']}: {doc_details['summary']}")
                
                if all_summaries:
                    combined_text = ". ".join(all_summaries)
                    text_to_speech(combined_text)
        
        with col2:
            if st.button("⏹️ Stop Reading", help="Stop audio playback"):
                stop_speech_html = """
                <script>window.speechSynthesis.cancel();</script>
                """
                html(stop_speech_html, height=0)

def render_graph_tab(selected_doc_ids):
    """Render the Graph Explorer tab with embedded controls."""
    if not selected_doc_ids:
        st.info("👈 Select documents from the sidebar to explore the knowledge graph.")
        st.markdown("""
        ### 🔍 Graph Explorer
        
        This tab provides an interactive visualization of your document knowledge graphs:
        
        - **Interactive Network**: Explore entities and their relationships
        - **Dynamic Filtering**: Filter by entity types, connections, and search terms
        - **Entity Details**: Click nodes to see detailed information
        - **Multiple Layouts**: Choose from different visualization algorithms
        """)
        return
    
    # Load data for selected documents
    with st.spinner("🔄 Loading and merging graph data..."):
        entities, relationships = load_multi_document_data(selected_doc_ids)
    
    if entities.empty:
        st.warning("⚠️ No graph data found for the selected documents.")
        return
    
    # Get document names for display
    processed_docs = get_processed_documents()
    selected_doc_names = []
    for doc in processed_docs:
        if doc['id'] in selected_doc_ids:
            selected_doc_names.append(doc['display_name'])
    
    # Header with document info
    st.markdown(f"""<div style="background: linear-gradient(90deg, #4CAF50 0%, #45A049 100%); padding: 1rem 2rem; border-radius: 8px; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0;">🔍 Exploring: {', '.join(selected_doc_names)}</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">✅ {len(entities)} entities • {len(relationships)} relationships</p>
    </div>""", unsafe_allow_html=True)
    
    # Graph Controls Panel - Embedded in tab
    with st.expander("🎛️ Graph Controls", expanded=True):
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.markdown("**🔍 Filters**")
            # Entity type filter
            entity_types = get_entity_type_options(entities)
            selected_types = st.multiselect(
                "Entity Types",
                entity_types,
                default=entity_types,
                key="graph_entity_types",
                help="Filter by entity types to focus your analysis"
            )
            
            # Search filter
            search_term = st.text_input(
                "Search Entities", 
                key="graph_search_term",
                placeholder="Type to search entities...",
                help="Search for specific entities by name"
            )
        
        with col2:
            st.markdown("**⚙️ Layout Settings**")
            # Layout options
            layout_option = st.selectbox(
                "Layout Algorithm",
                ["barnes_hut", "force_atlas_2", "hierarchical"],
                index=0,
                key="graph_layout",
                help="Choose visualization algorithm"
            )
            
            # Degree range filter
            min_degree, max_degree = get_degree_range(entities)
            if min_degree < max_degree:
                degree_range = st.slider(
                    "Connection Range",
                    min_value=min_degree,
                    max_value=max_degree,
                    value=(min_degree, max_degree),
                    key="graph_degree_range",
                    help="Filter entities by number of connections"
                )
            else:
                degree_range = (min_degree, max_degree)
        
        with col3:
            st.markdown("**🚀 Performance**")
            # Performance optimization
            optimize_graph = st.checkbox(
                "Performance Mode",
                value=len(entities) > 100,
                key="graph_optimize",
                help="Show only top 100 most connected nodes for better performance"
            )
            
            # Create NetworkX graph for statistics
            G = create_networkx_graph(entities, relationships)
            
            st.metric("👥 Entities", len(entities))
            st.metric("🔗 Relations", len(relationships))
            st.metric("🔴 Components", nx.number_connected_components(G))
    
    # Optimize for large graphs if enabled
    display_entities = entities
    display_relationships = relationships
    
    if optimize_graph:
        display_entities, display_relationships = optimize_for_large_graph(
            entities, relationships, threshold=100
        )
        st.info(f"📈 Displaying top {len(display_entities)} entities for performance")
    
    # Create PyVis network
    try:
        with st.spinner("🔄 Creating interactive network..."):
            net = create_pyvis_network(
                display_entities,
                display_relationships,
                entity_filter=selected_types if selected_types else None,
                degree_range=degree_range,
                search_term=search_term if search_term else None,
                layout=layout_option
            )
            
            # Render network
            html_content = render_pyvis_network(net)
            
            # Display in Streamlit - full width
            components.html(html_content, height=700)
    except Exception as e:
        st.error(f"❌ Error creating graph: {e}")
        st.info("💡 Try adjusting the filters or switching to a different layout algorithm.")
    
    # Entity search and details section
    with st.expander("🔍 Entity Search & Details", expanded=False):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Entity search
            entity_search = st.text_input("Search for specific entity:", key="graph_entity_search")
            
            if entity_search:
                # Filter entities by search term
                filtered_entities = entities[
                    entities['title'].str.contains(entity_search, case=False, na=False)
                ]
                
                if not filtered_entities.empty:
                    selected_entity = st.selectbox(
                        "Select entity:",
                        filtered_entities['title'].tolist(),
                        key="graph_selected_entity"
                    )
                    
                    # Show entity details
                    entity_data = filtered_entities[filtered_entities['title'] == selected_entity].iloc[0]
                    
                    st.markdown(f"**Type:** {entity_data.get('type', 'Unknown')}")
                    st.markdown(f"**Source:** {entity_data.get('source_document', 'Unknown')}")
                    if 'description' in entity_data:
                        st.markdown(f"**Description:** {entity_data['description']}")
                else:
                    st.info("No entities found matching your search.")
        
        with col2:
            # Show sample entities
            if not entities.empty:
                st.markdown("**Sample Entities:**")
                sample_entities = entities.sample(min(8, len(entities)))
                for _, entity in sample_entities.iterrows():
                    source_doc = entity.get('source_document', 'Unknown')[:15]
                    st.markdown(f"• **{entity['title']}** ({source_doc}...)")


def render_audio_tab(selected_doc_ids):
    """Render the audio lecture generation tab."""
    # Get selected documents
    all_docs = get_all_documents()
    selected_docs = [doc for doc in all_docs if doc['id'] in selected_doc_ids]
    
    if not selected_docs:
        st.info("📄 Please select documents from the sidebar to generate audio lectures.")
        return
    
    st.header("🎙️ Audio Lecture Generator")
    st.caption("Generate comprehensive audio lectures from document content")
    
    # Import audio lecture generator
    try:
        from audio_lecture_generator import AudioLectureGenerator, LectureDetailLevel
        from openai_tts_simple import generate_tts_audio
        
        generator = AudioLectureGenerator()
        
        # Lecture controls
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Detail level slider
            detail_level = st.select_slider(
                "📏 Lecture Detail Level",
                options=[
                    LectureDetailLevel.OVERVIEW,
                    LectureDetailLevel.MAIN_POINTS,
                    LectureDetailLevel.STANDARD,
                    LectureDetailLevel.DETAILED,
                    LectureDetailLevel.COMPREHENSIVE
                ],
                value=LectureDetailLevel.STANDARD,
                format_func=lambda x: {
                    LectureDetailLevel.OVERVIEW: "Overview (Highest level only)",
                    LectureDetailLevel.MAIN_POINTS: "Main Points (Key sections)",
                    LectureDetailLevel.STANDARD: "Standard (All sections)",
                    LectureDetailLevel.DETAILED: "Detailed (+ Entities & Relations)",
                    LectureDetailLevel.COMPREHENSIVE: "Comprehensive (Everything)"
                }[x],
                help="Control how much detail to include in the lecture"
            )
        
        with col2:
            # Voice selection for lecture
            voice_options = {
                "alloy": "Alloy (Neutral)",
                "echo": "Echo (Clear)",
                "fable": "Fable (Warm)",
                "onyx": "Onyx (Deep)",
                "nova": "Nova (Bright)",
                "shimmer": "Shimmer (Soft)"
            }
            lecture_voice = st.selectbox(
                "🎤 Lecture Voice",
                options=list(voice_options.keys()),
                format_func=lambda x: voice_options[x],
                index=4,  # Default to nova
                help="Choose the voice for your audio lecture"
            )
        
        with col3:
            # Quality selection
            lecture_quality = st.selectbox(
                "🎧 Audio Quality",
                options=["tts-1", "tts-1-hd"],
                format_func=lambda x: "Standard" if x == "tts-1" else "HD (High Quality)",
                help="Higher quality takes longer but sounds better"
            )
        
        # Additional options
        with st.expander("🔧 Advanced Options", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                include_entities = st.checkbox(
                    "Include key entities",
                    value=detail_level.value >= 3,
                    help="Include important concepts and entities in the lecture"
                )
                include_relationships = st.checkbox(
                    "Include relationships",
                    value=detail_level.value >= 4,
                    help="Include connections between concepts"
                )
            
            with col2:
                include_bullet_points = st.checkbox(
                    "Include key takeaways",
                    value=True,
                    help="Include bullet point summaries as key takeaways"
                )
                add_intro_music = st.checkbox(
                    "Add intro/outro (coming soon)",
                    value=False,
                    disabled=True,
                    help="Add professional intro and outro music"
                )
            
            with col3:
                use_dynamic_format = st.checkbox(
                    "🤖 AI-Driven Format",
                    value=True,
                    help="Let AI analyze the document and choose the best lecture style (narrative, tutorial, academic, etc.)"
                )
                if use_dynamic_format:
                    st.info("AI will analyze your document and create a custom lecture format")
        
        # Show previously generated lectures
        if selected_docs:
            doc_id = selected_docs[0]['id']
            previous_lectures = generator.load_lecture_scripts(doc_id)
            
            if previous_lectures:
                with st.expander(f"📚 Previously Generated Lectures ({len(previous_lectures)})", expanded=False):
                    for lecture in previous_lectures[:5]:  # Show latest 5
                        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                        
                        with col1:
                            gen_time = lecture.get('generated_at', 'Unknown')
                            if gen_time != 'Unknown':
                                try:
                                    gen_dt = datetime.fromisoformat(gen_time)
                                    gen_time = gen_dt.strftime("%Y-%m-%d %H:%M")
                                except:
                                    pass
                            st.markdown(f"**Generated:** {gen_time}")
                        
                        with col2:
                            st.markdown(f"**Detail:** {lecture.get('detail_level', 'Unknown')}")
                        
                        with col3:
                            duration = lecture.get('duration_estimate', 0)
                            st.markdown(f"**Duration:** {duration} min")
                        
                        with col4:
                            # Download previous script button
                            if st.button("📄", key=f"download_script_{lecture.get('filename', '')}",
                                       help="Download script"):
                                script_text = lecture.get('full_script', '')
                                if script_text:
                                    st.download_button(
                                        label="💾",
                                        data=script_text,
                                        file_name=f"lecture_script_{lecture.get('detail_level', 'unknown')}.txt",
                                        mime="text/plain",
                                        key=f"dl_{lecture.get('filename', '')}"
                                    )
        
        # Generate lecture button
        if st.button("🎬 Generate Audio Lecture", type="primary", use_container_width=True):
            with st.spinner("🎙️ Preparing your audio lecture..."):
                # Gather all document data
                all_section_summaries = {}
                document_summary = None
                additional_context = {
                    'entities': [],
                    'relationships': [],
                    'bullet_points': [],
                    'community_reports': []
                }
                
                # Process each selected document
                for doc in selected_docs:
                    doc_details = get_document_by_id(doc['id'])
                    
                    if doc_details:
                        # Get document summary
                        if not document_summary and doc_details.get('summary'):
                            document_summary = {
                                'display_name': doc_details['display_name'],
                                'summary': doc_details['summary']
                            }
                        
                        # Get section summaries
                        if doc_details.get('section_summaries'):
                            all_section_summaries.update(doc_details['section_summaries'])
                        
                        # Get entities and relationships if needed
                        if include_entities or include_relationships:
                            entities, relationships = load_multi_document_data([doc['id']])
                            
                            if include_entities and not entities.empty:
                                # Get top entities
                                top_entities = entities.nlargest(20, 'degree', keep='all') if 'degree' in entities.columns else entities.head(20)
                                for _, entity in top_entities.iterrows():
                                    additional_context['entities'].append({
                                        'title': entity.get('title', ''),
                                        'type': entity.get('type', ''),
                                        'description': entity.get('description', '')[:200] if 'description' in entity else ''
                                    })
                            
                            if include_relationships and not relationships.empty:
                                # Get top relationships
                                for _, rel in relationships.head(15).iterrows():
                                    additional_context['relationships'].append({
                                        'source': rel.get('source', ''),
                                        'target': rel.get('target', ''),
                                        'description': rel.get('description', '')[:100] if 'description' in rel else ''
                                    })
                        
                        # Get community reports
                        try:
                            reports_df_local = load_community_reports([doc['id']])
                            if not reports_df_local.empty:
                                for _, report in reports_df_local.head(5).iterrows():
                                    if 'summary' in report and pd.notna(report['summary']):
                                        additional_context['community_reports'].append(report['summary'])
                        except Exception as e:
                            logger.warning(f"Could not load community reports for lecture: {e}")
                
                if not document_summary:
                    st.error("❌ No document summary available for lecture generation")
                    st.stop()
                
                # Generate the lecture script
                try:
                    lecture_script = generator.generate_comprehensive_lecture(
                        document_summary=document_summary,
                        section_summaries=all_section_summaries,
                        detail_level=detail_level,
                        additional_context=additional_context,
                        include_bullet_points=include_bullet_points,
                        use_dynamic_format=use_dynamic_format
                    )
                    
                    # Save the lecture script
                    saved_lecture = generator.save_lecture_script(
                        document_id=selected_docs[0]['id'],
                        script=lecture_script,
                        detail_level=detail_level.name
                    )
                    
                    if saved_lecture:
                        st.success("✅ Lecture script generated successfully!")
                        
                        # Display lecture info
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Duration Estimate:** {saved_lecture['duration_estimate']} minutes")
                            st.markdown(f"**Word Count:** {saved_lecture['word_count']:,} words")
                            st.markdown(f"**Detail Level:** {saved_lecture['detail_level']}")
                        
                        with col2:
                            # Download script button
                            st.download_button(
                                label="📥 Download Script",
                                data=lecture_script,
                                file_name=f"lecture_script_{detail_level.name.lower()}.txt",
                                mime="text/plain"
                            )
                        
                        # Show script preview
                        with st.expander("📄 Lecture Script Preview", expanded=True):
                            # Show first 1000 characters
                            preview_length = min(2000, len(lecture_script))
                            st.text_area(
                                "Script Preview:",
                                value=lecture_script[:preview_length] + ("..." if len(lecture_script) > preview_length else ""),
                                height=300,
                                disabled=True
                            )
                        
                        # Generate audio button
                        st.markdown("---")
                        st.subheader("🎤 Generate Audio File")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            if st.button("🔊 Generate Audio (OpenAI TTS)", type="secondary", use_container_width=True):
                                with st.spinner("🎵 Converting script to audio... This may take a few minutes..."):
                                    # Split the script into chunks if it's too long
                                    max_chunk_size = 4000  # OpenAI TTS limit
                                    
                                    if len(lecture_script) <= max_chunk_size:
                                        # Generate single audio file
                                        audio_bytes, error = generate_tts_audio(
                                            lecture_script,
                                            voice=lecture_voice,
                                            model=lecture_quality
                                        )
                                        
                                        if audio_bytes:
                                            st.success("✅ Audio generated successfully!")
                                            st.audio(audio_bytes, format="audio/mp3")
                                            
                                            # Download button
                                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            filename = f"lecture_{detail_level.name.lower()}_{timestamp}.mp3"
                                            
                                            st.download_button(
                                                label="💾 Download Audio",
                                                data=audio_bytes,
                                                file_name=filename,
                                                mime="audio/mpeg"
                                            )
                                        else:
                                            st.error(f"❌ Failed to generate audio: {error}")
                                    else:
                                        # Script is too long, need to chunk it
                                        st.warning(f"⚠️ Script is {len(lecture_script):,} characters. Will generate in parts...")
                                        
                                        # Split script into sentences and chunk them
                                        sentences = lecture_script.replace('. ', '.|').split('|')
                                        chunks = []
                                        current_chunk = ""
                                        
                                        for sentence in sentences:
                                            if len(current_chunk) + len(sentence) < max_chunk_size:
                                                current_chunk += sentence
                                            else:
                                                if current_chunk:
                                                    chunks.append(current_chunk)
                                                current_chunk = sentence
                                        
                                        if current_chunk:
                                            chunks.append(current_chunk)
                                        
                                        st.info(f"📊 Generating {len(chunks)} audio segments...")
                                        
                                        # Generate each chunk
                                        audio_segments = []
                                        for i, chunk in enumerate(chunks):
                                            with st.spinner(f"Generating segment {i+1}/{len(chunks)}..."):
                                                audio_bytes, error = generate_tts_audio(
                                                    chunk,
                                                    voice=lecture_voice,
                                                    model=lecture_quality
                                                )
                                                
                                                if audio_bytes:
                                                    audio_segments.append(audio_bytes)
                                                    st.success(f"✅ Segment {i+1} generated")
                                                else:
                                                    st.error(f"❌ Failed to generate segment {i+1}: {error}")
                                                    break
                                        
                                        if len(audio_segments) == len(chunks):
                                            st.success("✅ All segments generated successfully!")
                                            st.info("📥 Download each segment below:")
                                            
                                            # Provide download buttons for each segment
                                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            for i, audio_bytes in enumerate(audio_segments):
                                                col1, col2 = st.columns([1, 3])
                                                with col1:
                                                    st.audio(audio_bytes, format="audio/mp3")
                                                with col2:
                                                    filename = f"lecture_{detail_level.name.lower()}_part{i+1}_{timestamp}.mp3"
                                                    st.download_button(
                                                        label=f"💾 Part {i+1}/{len(chunks)}",
                                                        data=audio_bytes,
                                                        file_name=filename,
                                                        mime="audio/mpeg",
                                                        key=f"dl_part_{i}"
                                                    )
                        
                        with col2:
                            st.info("💡 Tip: For best results, use HD quality for important presentations")
                    
                except Exception as e:
                    st.error(f"❌ Error generating lecture: {str(e)}")
                    logger.error(f"Lecture generation error: {e}")
    
    except ImportError as e:
        st.error("❌ Audio lecture generator module not found. Please ensure all dependencies are installed.")
        logger.error(f"Import error in audio tab: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error in audio tab: {str(e)}")
        logger.error(f"Audio tab error: {e}")


def check_for_stuck_documents_periodically():
    """Check for stuck documents periodically and show notification."""
    # Check every 5 minutes (300 seconds) if the app has been running long enough
    current_time = datetime.now()
    
    # Initialize last check time in session state
    if 'last_stuck_check' not in st.session_state:
        st.session_state.last_stuck_check = current_time
    
    # Check if 5 minutes have passed since last check
    time_since_last_check = (current_time - st.session_state.last_stuck_check).total_seconds()
    
    if time_since_last_check >= 300:  # 5 minutes
        # Check for stuck documents
        stuck_docs = check_stuck_documents(timeout_minutes=30)
        
        if stuck_docs:
            st.warning(f"⚠️ Found {len(stuck_docs)} document(s) stuck in processing. They have been marked as ERROR.")
            logger.info(f"Auto-detected {len(stuck_docs)} stuck documents")
        
        # Update last check time
        st.session_state.last_stuck_check = current_time


def main():
    """Main Streamlit app with tab-based navigation."""
    
    # Universal sidebar for document selection
    selected_doc_ids = render_universal_sidebar()
    
    # Initialize processor and handle logs if requested
    get_processor().init_db()
    
    # Periodically check for stuck documents
    check_for_stuck_documents_periodically()
    
    if st.session_state.get('show_logs', False):
        render_processing_logs()
        return
    
    # Main tab navigation - only show if documents are selected
    if selected_doc_ids:
        tab_summary, tab_chat, tab_graph, tab_audio = st.tabs([
            "📊 Summary",
            "💬 Chat Assistant",
            "🔍 Graph Explorer",
            "🎙️ Audio"
        ])
        
        with tab_summary:
            render_document_summary_tab(selected_doc_ids)
        
        with tab_chat:
            render_chat_tab(selected_doc_ids)
        
        with tab_graph:
            render_graph_tab(selected_doc_ids)
        
        with tab_audio:
            render_audio_tab(selected_doc_ids)
    else:
        # Welcome screen when no documents selected
        st.info("👈 Select documents from the sidebar to begin exploring.")
        st.markdown("""
        ### 🧠 GraphRAG Explorer
        
        **Get started by:**
        1. **Upload documents** using the sidebar upload interface
        2. **Wait for processing** to complete (this may take several minutes)  
        3. **Select documents** to explore from your document library
        4. **Analyze** using the Graph Explorer or Chat Assistant
        
        ### Features Available:
        - **🔍 Graph Explorer**: Interactive knowledge graph visualization
        - **💬 Chat Assistant**: AI-powered document Q&A with multiple search methods
        """)

def render_processing_logs():
    """Render processing logs modal."""
    st.subheader("📋 Processing Logs")
    
    if st.button("❌ Close Logs"):
        st.session_state.show_logs = False
        st.rerun()
    
    # Get all docs for log display
    all_docs = get_processor().get_all_documents()
    
    for doc_id in st.session_state.get('log_doc_ids', []):
        logs = get_processor().get_processing_logs(doc_id)
        if logs:
            # Get document name and status
            doc_name = "Unknown"
            doc_status = "Unknown"
            for doc in all_docs:
                if doc['id'] == doc_id:
                    doc_name = doc['display_name']
                    doc_status = doc['status']
                    break
            
            status_emoji = {
                'COMPLETED': '✅',
                'PROCESSING': '⏳',
                'ERROR': '❌',
                'UPLOADED': '📤',
            }.get(doc_status, '❓')
            
            st.markdown(f"**{status_emoji} {doc_name}** ({doc_status})")
            
            # Create a formatted display of logs
            log_text = []
            for log in logs:
                timestamp = log['timestamp']
                stage = log['stage']
                message = log['message']
                level = log['level']
                
                level_emoji = {
                    'INFO': 'ℹ️',
                    'ERROR': '❌',
                    'WARNING': '⚠️'
                }.get(level, 'ℹ️')
                
                log_text.append(f"{level_emoji} [{timestamp}] {stage}: {message}")
            
            st.text_area(
                f"Logs for {doc_name}",
                value="\n".join(log_text),
                height=300,
                key=f"logs_{doc_id}"
            )
        else:
            st.info(f"No logs found for document {doc_id}")
    
    st.markdown("---")

def render_chat_tab(selected_doc_ids):
    """Render the Chat Assistant tab."""
    if not selected_doc_ids:
        st.info("👈 Select documents from the sidebar to start chatting.")
        available_methods = get_supported_methods()
        method_info = {
            "local": "**Local Search**: Find specific details and facts",
            "global": "**Global Search**: Discover themes and patterns",
            "drift": "**Drift Search**: Contextual exploration",
            "basic": "**Basic Search**: Simple keyword matching",
            "summary": "**Summary Search**: AI-enhanced semantic similarity using document summaries",
            "enhanced": "**Enhanced Search**: Two-stage retrieval with raw text embeddings and graph knowledge"
        }
        
        methods_list = "\n".join([f"- {method_info.get(method, method)}" for method in available_methods])
        
        st.markdown(f"""
        ### 💬 Chat Assistant
        
        Ask questions about your selected documents using advanced GraphRAG search methods:
        
        {methods_list}
        
        {'**🆕 Summary Search** uses AI-generated document summaries with vector embeddings for enhanced semantic understanding!' if 'summary' in available_methods else ''}
        """)
        return
    
    # Get document names for display
    processed_docs = get_processed_documents()
    selected_doc_names = [doc['display_name'] for doc in processed_docs if doc['id'] in selected_doc_ids]
    
    # Header with document info
    st.markdown(f"""<div style="background: linear-gradient(90deg, #4CAF50 0%, #45A049 100%); padding: 1rem 2rem; border-radius: 8px; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0;">💬 Chat with: {', '.join(selected_doc_names)}</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Ask questions about your documents using GraphRAG AI</p>
    </div>""", unsafe_allow_html=True)
    
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Query method, model selection, and TTS controls
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        # Get available search methods (includes summary if dependencies are available)
        available_methods = get_supported_methods()
        method_descriptions = {
            "local": "Local: specific details and precise answers",
            "global": "Global: themes, patterns, and broad insights", 
            "drift": "Drift: contextual and exploratory search",
            "basic": "Basic: simple keyword-based search",
            "summary": "Summary: AI-enhanced semantic similarity search",
            "enhanced": "Enhanced: raw text embeddings + graph knowledge"
        }
        
        query_method = st.selectbox(
            "Search Method:",
            options=available_methods,
            index=0,
            help="; ".join([method_descriptions.get(m, m) for m in available_methods]),
            key="chat_query_method"
        )
    
    with col2:
        # Initialize model selection in session state
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = "o4-mini"
        
        model_options = ["o4-mini", "gpt-4.1-mini", "gpt-4.1-nano", "o3-mini", "gpt-4o-mini", "gpt-4o-mini-audio-preview", "o1-mini"]
        
        # Set default if current selection is not in new options
        if st.session_state.selected_model not in model_options:
            st.session_state.selected_model = "o4-mini"
        
        selected_model = st.selectbox(
            "OpenAI Model:",
            options=model_options,
            index=model_options.index(st.session_state.selected_model),
            help="Choose the OpenAI model for Q&A responses",
            key="chat_model_selection"
        )
        
        # Update session state when model changes
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
    
    with col3:
        st.markdown("**🔊 TTS**")
        # Initialize TTS settings
        if "tts_enabled" not in st.session_state:
            st.session_state.tts_enabled = True
        if "tts_rate" not in st.session_state:
            st.session_state.tts_rate = 0.9
        
        # TTS toggle
        tts_enabled = st.checkbox(
            "Auto-read",
            value=st.session_state.tts_enabled,
            help="Automatically read new responses aloud",
            key="tts_toggle"
        )
        st.session_state.tts_enabled = tts_enabled
        
        # Stop all speech button
        if st.button("⏹️", help="Stop all speech", key="stop_speech"):
            stop_speech_html = """
            <script>
            window.speechSynthesis.cancel();
            console.log('TTS: Chat speech cancelled');
            </script>
            """
            html(stop_speech_html, height=0)
    
    # TTS Settings (expandable)
    with st.expander("🔊 Voice Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            # Speech rate control
            tts_rate = st.slider(
                "Speech Rate",
                min_value=0.3,
                max_value=2.0,
                value=st.session_state.get('tts_rate', 0.9),
                step=0.1,
                help="Adjust speaking speed",
                key="tts_rate_slider"
            )
            st.session_state.tts_rate = tts_rate
        
        with col2:
            # Test voice button
            if st.button("🎤 Test Voice", help="Test current voice settings"):
                test_text = "This is a test of the text to speech feature. You can adjust the speed and other settings."
                text_to_speech(test_text, rate=tts_rate)
        
        # Query form to handle input properly
        with st.form("query_form", clear_on_submit=True):
            query_input = st.text_input(
                "Ask a question about the selected documents:",
                placeholder="e.g., What are the main requirements mentioned in these documents?"
            )
            
            # Query buttons
            col1, col2 = st.columns([3, 1])
            with col1:
                ask_button = st.form_submit_button("🔍 Ask Question", type="primary")
            with col2:
                clear_button = st.form_submit_button("🗑️ Clear Chat")
            
            if ask_button and query_input.strip():
                # Store the query and trigger processing
                st.session_state.pending_query = query_input
                st.session_state.pending_method = query_method
                st.session_state.pending_model = selected_model
            
            elif ask_button and not query_input.strip():
                st.session_state.query_error = "Please enter a question first."
            
            if clear_button:
                st.session_state.chat_history = []
                st.session_state.clear_success = True
        
        # Process pending query outside the form
        if hasattr(st.session_state, 'pending_query'):
            query = st.session_state.pending_query
            method = st.session_state.pending_method
            model = st.session_state.pending_model
            
            # Clear the pending query
            del st.session_state.pending_query
            del st.session_state.pending_method
            del st.session_state.pending_model
            
            # Validate query
            is_valid, error_msg = validate_query(query)
            if not is_valid:
                st.error(f"Invalid query: {error_msg}")
            else:
                # Run query
                with st.spinner(f"Running {method} search..."):
                    try:
                        # Select appropriate search method
                        if method == "global":
                            result = global_search(selected_doc_ids, query, model=model)
                        elif method == "local":
                            result = local_search(selected_doc_ids, query, model=model)
                        elif method == "drift":
                            result = drift_search(selected_doc_ids, query, model=model)
                        elif method == "basic":
                            result = basic_search(selected_doc_ids, query, model=model)
                        elif method == "summary":
                            result = summary_search(selected_doc_ids, query, limit=10, model=model)
                        elif method == "enhanced":
                            # Enhanced search uses O1 model if selected model starts with 'o1'
                            use_o1 = model.startswith('o1')
                            result = enhanced_search(selected_doc_ids, query, use_o1_model=use_o1, limit=5)
                        else:
                            result = local_search(selected_doc_ids, query, model=model)
                        
                        # Search for relevant sections
                        from query_logic import search_relevant_sections
                        relevant_sections = search_relevant_sections(query, selected_doc_ids, limit=3)
                        
                        # Add to chat history
                        chat_entry = {
                            "query": query,
                            "response": result.response,
                            "method": method,
                            "model": model,
                            "success": result.success,
                            "source_documents": result.source_documents,
                            "error": result.error_message,
                            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                            "raw_response": result.raw_response,
                            "context_info": result.context_info,
                            "relevant_sections": relevant_sections
                        }
                        st.session_state.chat_history.append(chat_entry)
                        
                        # Show results
                        if result.success:
                            st.success(f"✅ Query completed using {method} search!")
                            
                            # Display the response
                            st.markdown("### 💬 Response")
                            st.markdown(result.response)
                            
                            # Display relevant sections if found
                            if relevant_sections:
                                st.markdown("### 📑 Relevant Document Sections")
                                for section in relevant_sections:
                                    with st.expander(f"📑 {section['section_path']} (Score: {section['relevance_score']:.2f})", expanded=False):
                                        st.markdown(f"**From:** {section['document_name']}")
                                        st.markdown(section['summary'])
                                        
                                        # TTS controls for section
                                        col1, col2 = st.columns([1, 5])
                                        with col1:
                                            if st.button("🔊 Read", key=f"tts_relevant_{hash(section['section_path'])}", 
                                                       help="Read this section"):
                                                text_to_speech(f"Section: {section['section_path']}. {section['summary']}")
                            
                            # OpenAI TTS Section (now outside the form)
                            render_simple_tts_section(result.response, "immediate")
                            
                            # Auto-read response if TTS is enabled
                            if st.session_state.get('tts_enabled', False):
                                tts_rate = st.session_state.get('tts_rate', 0.9)
                                text_to_speech(result.response, rate=tts_rate)
                        else:
                            st.error(f"❌ Query failed: {result.error_message}")
                            
                    except Exception as e:
                        st.error(f"Query failed: {str(e)}")
        
        # Show error or success messages
        if hasattr(st.session_state, 'query_error'):
            st.warning(st.session_state.query_error)
            del st.session_state.query_error
        
        if hasattr(st.session_state, 'clear_success'):
            st.success("Chat history cleared!")
            del st.session_state.clear_success
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💬 Conversation History")
            
            # Show recent chats first (reverse order)
            for i, entry in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10 entries
                with st.expander(f"🙋 {entry['query'][:50]}... ({entry['timestamp']})", expanded=(i == 0)):
                    if entry['success']:
                        st.markdown("**Response:**")
                        st.markdown(entry['response'])
                        
                        # Browser-based TTS
                        if st.button("🔊 Browser TTS", key=f"browser_tts_{i}_{hash(entry['response'])}", help="Read aloud with browser"):
                            text_to_speech(entry['response'])
                        
                        # OpenAI TTS Section
                        render_simple_tts_section(entry['response'], f"history_{i}_{hash(entry['response'])}")
                        
                        st.markdown(f"**Method:** {entry['method'].title()} | **Model:** {entry.get('model', 'o4-mini')}")
                        
                        if entry['source_documents']:
                            st.markdown("**Sources:**")
                            for doc in entry['source_documents']:
                                st.markdown(f"• {doc}")
                        
                        # Show relevant sections from history
                        if entry.get('relevant_sections'):
                            st.markdown("**📑 Relevant Sections:**")
                            for section in entry['relevant_sections']:
                                st.markdown(f"• **{section['section_path']}** - {section['summary'][:100]}...")
                        
                        # Show context information for debugging
                        if entry.get('context_info'):
                            with st.expander("🔍 Debug: GraphRAG Context Information", expanded=False):
                                context_info = entry['context_info']
                                
                                # Show data references (what GraphRAG is citing)
                                if context_info.get('data_references'):
                                    st.markdown("**📊 Data References (Sources Used):**")
                                    for ref in context_info['data_references']:
                                        st.code(f"[Data: {ref}]")
                                
                                # Show retrieved content
                                if context_info.get('retrieved_content'):
                                    st.markdown("**📄 Retrieved Content:**")
                                    for content in context_info['retrieved_content'][:5]:  # Show first 5 items
                                        st.text_area("Content:", content, height=100, key=f"content_{hash(content)}")
                                
                                # Show search context
                                if context_info.get('search_context'):
                                    st.markdown("**🔍 Search Context:**")
                                    for step, context in context_info['search_context'].items():
                                        st.text(f"{step}: {context}")
                                
                                if context_info.get('entities_found'):
                                    st.markdown("**👥 Entities Found:**")
                                    for entity in context_info['entities_found']:
                                        st.text(entity)
                                
                                if context_info.get('relationships_found'):
                                    st.markdown("**🔗 Relationships Found:**")
                                    for rel in context_info['relationships_found']:
                                        st.text(rel)
                                
                                if context_info.get('text_units_found'):
                                    st.markdown("**📝 Text Units Found:**")
                                    for text_unit in context_info['text_units_found']:
                                        st.text(text_unit)
                                
                                if context_info.get('community_reports_found'):
                                    st.markdown("**📋 Community Reports Found:**")
                                    for report in context_info['community_reports_found'][:3]:  # Show first 3
                                        st.text_area("Report:", report, height=150, key=f"report_{hash(report)}")
                                
                                if context_info.get('vector_search_results'):
                                    st.markdown("**🎯 Vector Search Results:**")
                                    for result in context_info['vector_search_results']:
                                        st.text(result)
                                
                                if context_info.get('configuration'):
                                    st.markdown("**⚙️ Configuration:**")
                                    for key, value in context_info['configuration'].items():
                                        st.text(f"{key}: {value}")
                        
                        # Show raw response for debugging
                        if entry.get('raw_response'):
                            with st.expander("📝 Debug: Raw GraphRAG Response", expanded=False):
                                st.text(entry['raw_response'])
                    else:
                        st.error(f"Query failed: {entry['error']}")
        else:
            st.info("Start a conversation by asking a question about your selected documents!")
    





if __name__ == "__main__":
    main()