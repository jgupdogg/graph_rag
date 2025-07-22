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
    processor,
    get_processed_documents,
    get_all_documents,
    process_new_document,
    load_and_merge_graphs,
    delete_document,
    get_document_status,
    reprocess_failed_document,
    get_processing_logs,
    DocumentStatus,
)
from query_logic import (
    query_documents,
    global_search,
    local_search,
    drift_search,
    basic_search,
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
processor.init_db()

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
        with processor.db_manager as cursor:
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
        
        # Check for existing summary cache
        summary_cache_file = workspace_path / "ai_document_summary_cache.json"
        if summary_cache_file.exists():
            try:
                with open(summary_cache_file, 'r') as f:
                    cached_data = json.load(f)
                # Check if cache is recent (less than 7 days old)
                from datetime import timedelta
                cache_time = datetime.fromisoformat(cached_data.get('generated_at', '2000-01-01'))
                if datetime.now() - cache_time < timedelta(days=7):
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
            "Please provide a comprehensive document summary with bullet points for major takeaways based on the following information:",
            f"\n## Document: {doc['display_name']}"
        ]
        
        if classification_data:
            prompt_parts.append(f"\n## Previous Classification:\n{classification_data}")
        
        if text_excerpt:
            prompt_parts.append(f"\n## Document Content (First 2500 characters):\n{text_excerpt}")
        
        if len(top_reports) > 0:
            prompt_parts.append("\n## Key Community Insights (Top 10 Largest Communities):")
            for i, (_, report) in enumerate(top_reports.iterrows(), 1):
                title = report.get('title', 'Untitled')
                summary = report.get('summary', 'No summary available')
                size = report.get('size', 'Unknown')
                prompt_parts.append(f"\n{i}. **{title}** (Size: {size}): {summary}")
        
        prompt_parts.append("""
\n## Please provide a concise summary with the following sections:

### Executive Summary
Provide a 2-3 sentence overview of the document.

### Major Takeaways
List 5-7 bullet points highlighting the most important information, insights, or requirements from the document. Focus on concrete, actionable items.

### Document Details
- **Type & Purpose**: What kind of document and its objective
- **Primary Focus**: The central theme or domain
- **Target Audience**: Who would use this document
- **Key Topics**: Main subject areas covered

Keep the entire response concise but informative, approximately 300-400 words total. Use bullet points where appropriate for clarity.""")
        
        full_prompt = "".join(prompt_parts)
        
        # Use query_documents to generate the summary
        result = query_documents(
            doc_ids=[doc_id],
            query=full_prompt,
            method="global"  # Use global search for comprehensive overview
        )
        
        if result.success:
            summary_data = {
                "document_id": doc_id,
                "document_name": doc['display_name'],
                "ai_summary": result.response,
                "classification": classification_data,
                "generated_at": datetime.now().isoformat(),
                "method_used": "global",
                "source_reports_count": len(top_reports)
            }
            
            # Cache the result
            try:
                with open(summary_cache_file, 'w') as f:
                    json.dump(summary_data, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to cache AI summary: {e}")
            
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
    all_docs = processor.get_all_documents()
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
    
    # Header with TTS controls
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("📊 Document Summary")
        st.caption("AI-powered overview, community insights, and document metadata")
    with col2:
        st.markdown("### 🔊 Audio")
        # Initialize TTS settings for summary tab if not exists
        if "summary_tts_enabled" not in st.session_state:
            st.session_state.summary_tts_enabled = False
        if "summary_tts_rate" not in st.session_state:
            st.session_state.summary_tts_rate = 0.8
        
        # TTS toggle for summaries
        summary_tts_enabled = st.checkbox(
            "Auto-read summaries",
            value=st.session_state.summary_tts_enabled,
            help="Automatically read document summaries aloud",
            key="summary_tts_toggle"
        )
        st.session_state.summary_tts_enabled = summary_tts_enabled
        
        # Stop speech button
        if st.button("⏹️ Stop", help="Stop all audio", key="summary_stop_speech"):
            stop_summary_speech_html = """
            <script>
            window.speechSynthesis.cancel();
            console.log('TTS: Speech cancelled');
            </script>
            """
            html(stop_summary_speech_html, height=0)
        
        # Test TTS button for debugging
        if st.button("🧪 Test TTS", help="Test text-to-speech with simple text", key="test_tts"):
            text_to_speech("Hello! This is a test of the text to speech feature. Can you hear me?")
        
        # Voice status check button
        if st.button("🔍 Check TTS Status", help="Check if text-to-speech is available", key="check_tts_status"):
            check_tts_html = """
            <script>
            function checkTTSStatus() {
                console.log('=== TTS STATUS CHECK ===');
                
                // Check basic support
                if (!('speechSynthesis' in window)) {
                    alert('❌ Speech Synthesis API not supported in this browser');
                    return;
                }
                
                console.log('✅ Speech Synthesis API is supported');
                
                // Check voices
                const voices = window.speechSynthesis.getVoices();
                console.log('Voices available:', voices.length);
                
                let statusMessage = `TTS Status Report:\\n`;
                statusMessage += `• Browser: ${navigator.userAgent.split(' ')[0]}\\n`;
                statusMessage += `• Speech API: ✅ Supported\\n`;
                statusMessage += `• Available voices: ${voices.length}\\n`;
                
                if (voices.length === 0) {
                    statusMessage += `\\n❌ NO VOICES AVAILABLE\\n\\nThis is the root cause of TTS failure.\\n\\nSolutions for Linux:\\n`;
                    statusMessage += `• Install: sudo apt install espeak espeak-data libespeak1 libespeak-dev\\n`;
                    statusMessage += `• Install: sudo apt install speech-dispatcher\\n`;
                    statusMessage += `• Try different browser (Chrome/Chromium usually work best)\\n`;
                    statusMessage += `• Check: sudo systemctl status speech-dispatcher\\n`;
                } else {
                    statusMessage += `\\n✅ VOICES AVAILABLE:\\n`;
                    voices.forEach((voice, i) => {
                        if (i < 5) { // Show first 5 voices
                            statusMessage += `  ${i+1}. ${voice.name} (${voice.lang})${voice.default ? ' [DEFAULT]' : ''}\\n`;
                        }
                    });
                }
                
                alert(statusMessage);
                console.log('=== END TTS STATUS ===');
            }
            
            // Wait for voices then check
            if (window.speechSynthesis.getVoices().length === 0) {
                window.speechSynthesis.addEventListener('voiceschanged', checkTTSStatus, { once: true });
                setTimeout(checkTTSStatus, 1000); // Fallback
            } else {
                checkTTSStatus();
            }
            </script>
            """
            html(check_tts_html, height=0)
    
    # Voice Settings (expandable)
    with st.expander("🎛️ Voice Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Speech rate control
            summary_tts_rate = st.slider(
                "Speech Speed",
                min_value=0.3,
                max_value=2.0,
                value=st.session_state.get('summary_tts_rate', 0.8),
                step=0.1,
                help="Adjust speaking speed (0.3 = very slow, 2.0 = very fast)",
                key="summary_tts_rate_slider"
            )
            st.session_state.summary_tts_rate = summary_tts_rate
            
            # Initialize voice selection
            if "selected_voice_index" not in st.session_state:
                st.session_state.selected_voice_index = 0
        
        with col2:
            # Voice selection dropdown (populated by JavaScript)
            voice_selection_html = f"""
            <div id="voice-selector-container">
                <select id="voice-selector" style="width: 100%; padding: 8px; border-radius: 4px; border: 1px solid #ddd;">
                    <option value="-1">Loading voices...</option>
                </select>
            </div>
            <script>
            function loadVoices() {{
                const voices = window.speechSynthesis.getVoices();
                const select = document.getElementById('voice-selector');
                
                if (voices.length > 0) {{
                    select.innerHTML = '';
                    
                    // Group voices by language
                    const englishVoices = voices.filter(v => v.lang.startsWith('en'));
                    const otherVoices = voices.filter(v => !v.lang.startsWith('en'));
                    
                    // Add English voices first
                    if (englishVoices.length > 0) {{
                        const optgroup = document.createElement('optgroup');
                        optgroup.label = 'English Voices';
                        
                        englishVoices.forEach((voice, index) => {{
                            const option = document.createElement('option');
                            option.value = voices.indexOf(voice);
                            option.text = voice.name + ' (' + voice.lang + ')';
                            if (voice.default) option.text += ' [DEFAULT]';
                            optgroup.appendChild(option);
                        }});
                        
                        select.appendChild(optgroup);
                    }}
                    
                    // Add other voices
                    if (otherVoices.length > 0) {{
                        const optgroup = document.createElement('optgroup');
                        optgroup.label = 'Other Languages';
                        
                        otherVoices.forEach(voice => {{
                            const option = document.createElement('option');
                            option.value = voices.indexOf(voice);
                            option.text = voice.name + ' (' + voice.lang + ')';
                            optgroup.appendChild(option);
                        }});
                        
                        select.appendChild(optgroup);
                    }}
                    
                    // Restore selected voice
                    select.value = {st.session_state.get('selected_voice_index', 0)};
                    
                    // Handle voice selection
                    select.onchange = function() {{
                        const selectedIndex = parseInt(select.value);
                        // Store in session state via hidden input
                        const hiddenInput = document.getElementById('selected-voice-index');
                        if (hiddenInput) {{
                            hiddenInput.value = selectedIndex;
                        }}
                        console.log('Selected voice index:', selectedIndex);
                        window.selectedVoiceIndex = selectedIndex;
                    }};
                    
                    // Set initial selection
                    window.selectedVoiceIndex = parseInt(select.value);
                    
                }} else {{
                    select.innerHTML = '<option value="-1">No voices available</option>';
                }}
            }}
            
            // Load voices on page load
            if (window.speechSynthesis.getVoices().length === 0) {{
                window.speechSynthesis.addEventListener('voiceschanged', loadVoices, {{ once: true }});
            }} else {{
                loadVoices();
            }}
            
            // Also try loading after a delay
            setTimeout(loadVoices, 500);
            </script>
            """
            
            st.markdown("**Select Voice:**")
            components.html(voice_selection_html, height=60)
            
            # Test button with current settings
            if st.button("🎤 Test Settings", help="Test with current voice and speed", key="test_voice_settings"):
                test_text = "Testing voice settings. This is how I will sound when reading your documents."
                test_settings_html = f"""
                <script>
                function testVoiceSettings() {{
                    const text = "{test_text}";
                    const rate = {summary_tts_rate};
                    const voiceIndex = window.selectedVoiceIndex || 0;
                    
                    window.speechSynthesis.cancel();
                    
                    const utterance = new SpeechSynthesisUtterance(text);
                    utterance.rate = rate;
                    utterance.pitch = 1.0;
                    utterance.volume = 1.0;
                    
                    const voices = window.speechSynthesis.getVoices();
                    if (voices.length > voiceIndex && voiceIndex >= 0) {{
                        utterance.voice = voices[voiceIndex];
                        console.log('Testing with voice:', voices[voiceIndex].name);
                    }}
                    
                    console.log('Testing with rate:', rate);
                    window.speechSynthesis.speak(utterance);
                }}
                
                // Wait for voices then test
                if (window.speechSynthesis.getVoices().length === 0) {{
                    setTimeout(testVoiceSettings, 500);
                }} else {{
                    testVoiceSettings();
                }}
                </script>
                """
                html(test_settings_html, height=0)
    
    # AI-Generated Document Overview Section
    st.subheader("🤖 AI-Generated Document Overview")
    
    with st.spinner("Generating comprehensive document overview..."):
        ai_summary_data = generate_ai_document_summary(selected_doc_ids)
    
    if ai_summary_data:
        # Display the AI summary with better contrast
        st.markdown("### 📋 Comprehensive Document Analysis")
        
        # Show the AI-generated summary with proper styling and TTS
        st.markdown("**AI Summary:**")
        st.success(ai_summary_data['ai_summary'])
        
        # Browser TTS button
        if st.button("🔊 Browser TTS", key="summary_browser_tts", help="Read summary with browser TTS"):
            text_to_speech(ai_summary_data['ai_summary'])
        
        # OpenAI TTS Section for Summary
        render_simple_tts_section(ai_summary_data['ai_summary'], "summary")
        
        # Auto-read summary if enabled
        if st.session_state.get('summary_tts_enabled', False):
            summary_text = ai_summary_data['ai_summary']
            # Use current voice settings for auto-read
            text_to_speech(summary_text)
        
        # Show additional metadata in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Analysis Method", ai_summary_data.get('method_used', 'N/A').title())
        with col2:
            st.metric("Source Reports", ai_summary_data.get('source_reports_count', 0))
        with col3:
            from datetime import datetime
            gen_time = ai_summary_data.get('generated_at', '')
            if gen_time:
                try:
                    gen_date = datetime.fromisoformat(gen_time).strftime("%Y-%m-%d %H:%M")
                    st.metric("Generated", gen_date)
                except:
                    st.metric("Generated", "Recently")
            else:
                st.metric("Generated", "Unknown")
        
        # Show original classification if available
        if ai_summary_data.get('classification'):
            with st.expander("📋 Previous Document Classification", expanded=False):
                st.markdown(ai_summary_data['classification'])
                
    else:
        st.warning("⚠️ Unable to generate AI summary. The document analysis may take a moment to complete, or there might be an issue with the AI service.")
        
        # Fallback: show basic document info
        doc_name = selected_docs[0]['display_name'] if selected_docs else "Unknown Document"
        st.info(f"📄 Document: **{doc_name}** - Use the chat feature for detailed analysis.")
    
    st.markdown("---")
    
    # Document Metadata Section
    st.subheader("📄 Document Information")
    
    for doc in selected_docs:
        with st.expander(f"📄 {doc['display_name']}", expanded=True):
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
    
    # Community Reports Section
    st.markdown("---")
    st.subheader("📋 Community Reports")
    
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


def main():
    """Main Streamlit app with tab-based navigation."""
    
    # Universal sidebar for document selection
    selected_doc_ids = render_universal_sidebar()
    
    # Initialize processor and handle logs if requested
    processor.init_db()
    
    if st.session_state.get('show_logs', False):
        render_processing_logs()
        return
    
    # Main tab navigation - only show if documents are selected
    if selected_doc_ids:
        tab_summary, tab_chat, tab_graph = st.tabs([
            "📊 Summary",
            "💬 Chat Assistant",
            "🔍 Graph Explorer"
        ])
        
        with tab_summary:
            render_document_summary_tab(selected_doc_ids)
        
        with tab_chat:
            render_chat_tab(selected_doc_ids)
        
        with tab_graph:
            render_graph_tab(selected_doc_ids)
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
    all_docs = processor.get_all_documents()
    
    for doc_id in st.session_state.get('log_doc_ids', []):
        logs = processor.get_processing_logs(doc_id)
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
        st.markdown("""
        ### 💬 Chat Assistant
        
        Ask questions about your selected documents using advanced GraphRAG search methods:
        
        - **Local Search**: Find specific details and facts
        - **Global Search**: Discover themes and patterns  
        - **Drift Search**: Contextual exploration
        - **Basic Search**: Simple keyword matching
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
        query_method = st.selectbox(
            "Search Method:",
            options=["local", "global", "drift", "basic"],
            index=0,
            help="Local: specific details, Global: themes/patterns, Drift: contextual search, Basic: simple search",
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
                        else:
                            result = local_search(selected_doc_ids, query, model=model)
                        
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
                            "context_info": result.context_info
                        }
                        st.session_state.chat_history.append(chat_entry)
                        
                        # Show results
                        if result.success:
                            st.success(f"✅ Query completed using {method} search!")
                            
                            # Display the response
                            st.markdown("### 💬 Response")
                            st.markdown(result.response)
                            
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