#!/usr/bin/env python3
"""
Streamlit Interactive GraphRAG Multi-Document Explorer
Advanced interface to explore multiple document knowledge graphs
"""

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import logging
import os
import streamlit.components.v1 as components
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
    cancel_document_processing
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

# Configure logging
logger = logging.getLogger(__name__)

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

# Custom CSS for professional styling - Version 2.0
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles - Force application */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        background-color: #F8F9FA !important;
    }
    
    /* Force main container styling */
    .main .block-container {
        padding-top: 1rem !important;
        max-width: none !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header styling - Force visibility */
    .main-header {
        background: linear-gradient(90deg, #4CAF50 0%, #45A049 100%) !important;
        padding: 1.5rem 2rem !important;
        border-radius: 12px !important;
        margin-bottom: 2rem !important;
        box-shadow: 0 4px 20px rgba(76, 175, 80, 0.15) !important;
        width: 100% !important;
        display: block !important;
        position: relative !important;
        z-index: 100 !important;
    }
    
    .main-header h1 {
        color: white !important;
        font-weight: 600;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Sidebar enhancements */
    .css-1d391kg {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
        border: none;
        padding: 0.5rem 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Primary button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4CAF50 0%, #45A049 100%);
        color: white;
    }
    
    /* Container styling */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    
    /* Success/error message styling */
    .stSuccess {
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }
    
    .stError {
        border-radius: 8px;
        border-left: 4px solid #F44336;
    }
    
    .stWarning {
        border-radius: 8px;
        border-left: 4px solid #FF9800;
    }
    
    .stInfo {
        border-radius: 8px;
        border-left: 4px solid #2196F3;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 500;
        border-radius: 8px;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #E0E0E0;
        transition: border-color 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.1);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    /* Multiselect styling */
    .stMultiSelect > div > div {
        border-radius: 8px;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #4CAF50;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: #4CAF50;
        border-radius: 4px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    /* Spinner customization */
    .stSpinner > div {
        border-top-color: #4CAF50;
    }
    
    /* Custom spacing */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        color: #2C3E50;
        border-bottom: 2px solid #E0E0E0;
        padding-bottom: 0.5rem;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading-pulse {
        animation: pulse 2s infinite;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Force cache invalidation
if "ui_version" not in st.session_state:
    st.session_state.ui_version = "2.0"

# Test CSS Application
st.markdown("""
<div style="background: red; padding: 10px; margin: 10px 0;">
    🚨 CSS Test Block - If you see this with a red background, HTML/CSS is working!
</div>
""", unsafe_allow_html=True)

# Enhanced header with custom styling and debugging
st.markdown(f"""
<div class="main-header" style="background: linear-gradient(90deg, #4CAF50 0%, #45A049 100%); padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(76, 175, 80, 0.15);">
    <h1 style="color: white; font-weight: 600; font-size: 2.5rem; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">🧠 GraphRAG Explorer</h1>
    <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem; margin: 0.5rem 0 0 0; font-weight: 400;">Interactive knowledge graph exploration and document analysis</p>
    <small style="color: rgba(255, 255, 255, 0.7);">UI Version: {st.session_state.ui_version}</small>
</div>
""", unsafe_allow_html=True)

# Initialize application
processor.init_db()

# Check API key availability
if not check_api_key_availability():
    st.error("⚠️ GraphRAG API key not found. Please set GRAPHRAG_API_KEY or OPENAI_API_KEY environment variable.")
    st.stop()

# Cache data loading
@st.cache_data
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

@st.cache_data
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
    cancelled_docs = [doc for doc in all_docs if doc['status'] == 'CANCELLED']
    
    if processing_docs:
        st.sidebar.markdown('<p class="section-header">⏳ Processing Status</p>', unsafe_allow_html=True)
        for doc in processing_docs:
            with st.sidebar.container():
                st.markdown(f'<div class="loading-pulse">🔄 Processing: **{doc["display_name"]}**</div>', unsafe_allow_html=True)
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.progress(0.5)  # Indeterminate progress
                with col2:
                    if st.button("❌", key=f"cancel_{doc['id']}", help="Cancel processing"):
                        if cancel_document_processing(doc['id']):
                            st.success("✅ Cancellation requested")
                            st.rerun()
                        else:
                            st.error("❌ Failed to cancel")
    
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
        
        # Show cancelled documents with retry option
        if cancelled_docs:
            st.sidebar.markdown('<p class="section-header">⏹️ Cancelled Documents</p>', unsafe_allow_html=True)
            for doc in cancelled_docs:
                with st.sidebar.expander(f"⏹️ {doc['display_name']}", expanded=False):
                    st.info("Processing was cancelled by user")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Retry", key=f"retry_cancelled_{doc['id']}", help="Retry processing"):
                            with st.spinner("🔄 Retrying processing..."):
                                # Reset cancellation flag
                                with processor.db_manager as cursor:
                                    cursor.execute("UPDATE documents SET cancellation_requested = 0 WHERE id = ?", (doc['id'],))
                                result = reprocess_failed_document(doc['id'])
                                st.success(result)
                                st.rerun()
                    with col2:
                        if st.button("🗑️ Delete", key=f"delete_cancelled_{doc['id']}", type="secondary", help="Delete document"):
                            if delete_document(doc['id']):
                                st.success("✅ Document deleted")
                                st.rerun()
                            else:
                                st.error("❌ Failed to delete")
        
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
    cancelled_docs = [doc for doc in all_docs if doc['status'] == 'CANCELLED']
    
    if processing_docs:
        st.sidebar.markdown('<p class="section-header">⏳ Processing Status</p>', unsafe_allow_html=True)
        for doc in processing_docs:
            with st.sidebar.container():
                st.sidebar.markdown(f'<div class="loading-pulse">🔄 Processing: **{doc["display_name"]}**</div>', unsafe_allow_html=True)
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    st.progress(0.5)  # Indeterminate progress
                with col2:
                    if st.button("❌", key=f"cancel_universal_{doc['id']}", help="Cancel processing"):
                        if cancel_document_processing(doc['id']):
                            st.success("✅ Cancellation requested")
                            st.rerun()
                        else:
                            st.error("❌ Failed to cancel")
    
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
    
    # Show cancelled documents with retry option
    if cancelled_docs:
        st.sidebar.markdown('<p class="section-header">⏹️ Cancelled Documents</p>', unsafe_allow_html=True)
        for doc in cancelled_docs:
            with st.sidebar.expander(f"⏹️ {doc['display_name']}", expanded=False):
                st.info("Processing was cancelled by user")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Retry", key=f"retry_universal_cancelled_{doc['id']}", help="Retry processing"):
                        with st.spinner("🔄 Retrying processing..."):
                            # Reset cancellation flag
                            with processor.db_manager as cursor:
                                cursor.execute("UPDATE documents SET cancellation_requested = 0 WHERE id = ?", (doc['id'],))
                            result = reprocess_failed_document(doc['id'])
                            st.success(result)
                            st.rerun()
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_universal_cancelled_{doc['id']}", type="secondary", help="Delete document"):
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
            "📊 Document Summary",
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
                'CANCELLED': '⏹️'
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

def load_document_overview(workspace_path):
    """Load document overview data from GraphRAG outputs."""
    workspace = Path(workspace_path)
    overview_data = {}
    
    try:
        # Load community reports for high-level summaries
        community_reports_file = workspace / "output" / "community_reports.parquet"
        if not community_reports_file.exists():
            community_reports_file = workspace / "output" / "artifacts" / "community_reports.parquet"
        
        if community_reports_file.exists():
            reports_df = pd.read_parquet(community_reports_file)
            # Get top-level summaries (highest rating)
            if not reports_df.empty and 'rating' in reports_df.columns:
                top_reports = reports_df.nlargest(3, 'rating')
                overview_data['top_summaries'] = top_reports[['title', 'summary']].to_dict('records')
            elif not reports_df.empty:
                # Fallback if no rating column
                overview_data['top_summaries'] = reports_df.head(3)[['title', 'summary']].to_dict('records')
        
        # Load document stats
        stats_file = workspace / "stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                overview_data['stats'] = json.load(f)
        
        return overview_data
    except Exception as e:
        logger.error(f"Error loading document overview: {e}")
        return {}

def load_key_entities(workspace_path, limit=10):
    """Load key entities from GraphRAG outputs."""
    workspace = Path(workspace_path)
    
    try:
        entities_file = workspace / "output" / "entities.parquet"
        if not entities_file.exists():
            entities_file = workspace / "output" / "artifacts" / "entities.parquet"
        
        if entities_file.exists():
            entities_df = pd.read_parquet(entities_file)
            if not entities_df.empty:
                # Sort by degree (most connected) or frequency
                if 'degree' in entities_df.columns:
                    top_entities = entities_df.nlargest(limit, 'degree')
                elif 'frequency' in entities_df.columns:
                    top_entities = entities_df.nlargest(limit, 'frequency')
                else:
                    top_entities = entities_df.head(limit)
                
                return top_entities[['title', 'type', 'description', 'degree']].to_dict('records') if 'degree' in entities_df.columns else top_entities[['title', 'type', 'description']].to_dict('records')
        
        return []
    except Exception as e:
        logger.error(f"Error loading key entities: {e}")
        return []

def generate_document_summary(workspace_path, max_chars=3000):
    """Generate a general summary from the beginning of the document."""
    workspace = Path(workspace_path)
    
    try:
        # Check for cached summary first
        cache_file = workspace / "document_summary_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached_summary = json.load(f)
                return cached_summary
        
        # Load text units to get document content
        text_units_file = workspace / "output" / "text_units.parquet"
        if not text_units_file.exists():
            text_units_file = workspace / "output" / "artifacts" / "text_units.parquet"
        
        if text_units_file.exists():
            text_units_df = pd.read_parquet(text_units_file)
            if not text_units_df.empty and 'text' in text_units_df.columns:
                # Get the first text unit (usually contains document beginning)
                first_text = text_units_df.iloc[0]['text']
                
                # Limit to max_chars but try to end at sentence boundary
                if len(first_text) > max_chars:
                    truncated_text = first_text[:max_chars]
                    # Find last sentence ending
                    last_period = truncated_text.rfind('.')
                    last_exclaim = truncated_text.rfind('!')
                    last_question = truncated_text.rfind('?')
                    last_sentence_end = max(last_period, last_exclaim, last_question)
                    
                    if last_sentence_end > max_chars * 0.7:  # If we found a reasonable sentence ending
                        text_for_summary = truncated_text[:last_sentence_end + 1]
                    else:
                        text_for_summary = truncated_text
                else:
                    text_for_summary = first_text
                
                # Generate AI summary using OpenAI
                try:
                    import openai
                    import os
                    
                    # Use the same API key as GraphRAG
                    api_key = os.getenv('GRAPHRAG_API_KEY') or os.getenv('OPENAI_API_KEY')
                    if not api_key:
                        return {"error": "No API key available for summary generation"}
                    
                    client = openai.OpenAI(api_key=api_key)
                    
                    prompt = f"""Please provide a concise summary of this document. Focus on:
1. The document's main purpose
2. Key topics or themes
3. Important context or background
4. Main points or objectives

Keep the summary to 3-4 sentences maximum.

Document text:
{text_for_summary}"""

                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that creates concise document summaries."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=200,
                        temperature=0.3
                    )
                    
                    summary_text = response.choices[0].message.content.strip()
                    
                    # Create summary object
                    summary_data = {
                        "overview": summary_text,
                        "chars_used": len(text_for_summary),
                        "total_chars": len(first_text),
                        "generated_at": pd.Timestamp.now().isoformat()
                    }
                    
                    # Cache the result
                    with open(cache_file, 'w') as f:
                        json.dump(summary_data, f, indent=2)
                    
                    return summary_data
                    
                except Exception as e:
                    logger.error(f"Error generating AI summary: {e}")
                    # Fallback: return first few sentences
                    sentences = text_for_summary.split('.')[:3]
                    fallback_summary = '. '.join(sentences).strip()
                    if fallback_summary and not fallback_summary.endswith('.'):
                        fallback_summary += '.'
                    
                    return {
                        "overview": fallback_summary or "Document content available but summary generation failed.",
                        "chars_used": len(text_for_summary),
                        "total_chars": len(first_text),
                        "generated_at": pd.Timestamp.now().isoformat(),
                        "fallback": True
                    }
        
        return {"error": "No text content found for summary generation"}
        
    except Exception as e:
        logger.error(f"Error generating document summary: {e}")
        return {"error": f"Failed to generate summary: {str(e)}"}

def get_top_communities(overview_data, limit=3):
    """Filter and return the most relevant community reports."""
    if not overview_data or 'top_summaries' not in overview_data:
        return []
    
    summaries = overview_data['top_summaries']
    
    # If we have ratings, sort by rating
    if summaries and 'rating' in summaries[0]:
        sorted_summaries = sorted(summaries, key=lambda x: x.get('rating', 0), reverse=True)
    else:
        # Otherwise, use the first few (they're usually ordered by importance)
        sorted_summaries = summaries
    
    return sorted_summaries[:limit]

def render_document_summary_tab(selected_doc_ids):
    """Render the Document Summary tab with metadata and AI-generated overviews."""
    if not selected_doc_ids:
        st.info("👈 Select documents from the sidebar to view document summaries.")
        st.markdown("""
        ### 📊 Document Summary
        
        This tab provides comprehensive overviews of your processed documents:
        
        - **Document Metadata**: File size, processing time, status
        - **AI-Generated Summaries**: High-level document overviews
        - **Key Entities**: Most important entities extracted from documents
        - **Processing Statistics**: Entity counts, relationship counts, and performance metrics
        """)
        return
    
    # Get document details
    all_docs = get_all_documents()
    
    # Filter selected documents
    selected_docs = []
    for doc in all_docs:
        if doc['id'] in selected_doc_ids:
            selected_docs.append(doc)
    
    if not selected_docs:
        st.warning("⚠️ No document data found for selected documents.")
        return
    
    # Header
    st.markdown(f"""<div style="background: linear-gradient(90deg, #4CAF50 0%, #45A049 100%); padding: 1rem 2rem; border-radius: 8px; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0;">📊 Document Summary</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Comprehensive overview of {len(selected_docs)} selected document(s)</p>
    </div>""", unsafe_allow_html=True)
    
    # Display each document
    for doc in selected_docs:
        with st.container():
            # Document header
            status_emoji = {
                'COMPLETED': '✅',
                'PROCESSING': '⏳',
                'ERROR': '❌',
                'UPLOADED': '📤',
                'CANCELLED': '⏹️'
            }.get(doc['status'], '❓')
            
            st.markdown(f"""
            <div style="background: white; border-left: 4px solid #4CAF50; padding: 1.5rem; border-radius: 8px; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h4 style="margin: 0 0 0.5rem 0; color: #2C3E50;">{status_emoji} {doc['display_name']}</h4>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">Status: {doc['status']} • Created: {doc['created_at'][:16]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Document metadata
                st.markdown("**📄 Document Information**")
                
                # Basic metadata
                metadata_info = f"""
                - **Original Filename**: {doc['original_filename']}
                - **File Size**: {doc.get('file_size', 'Unknown')} bytes
                - **Processing Time**: {doc.get('processing_time', 'Unknown')} seconds
                - **Created**: {doc['created_at']}
                - **Last Updated**: {doc['updated_at']}
                """
                st.markdown(metadata_info)
                
                # Load processing statistics if available
                if doc['status'] == 'COMPLETED':
                    try:
                        # Load entities and relationships count
                        entities, relationships = load_and_merge_graphs([doc['id']])
                        if not entities.empty:
                            st.markdown("**📊 Processing Results**")
                            metrics_col1, metrics_col2 = st.columns(2)
                            with metrics_col1:
                                st.metric("Entities", len(entities))
                            with metrics_col2:
                                st.metric("Relationships", len(relationships))
                    except Exception as e:
                        st.warning(f"Could not load processing statistics: {e}")
            
            with col2:
                # General Document Overview (NEW)
                if doc['status'] == 'COMPLETED':
                    st.markdown("**📝 General Document Overview**")
                    
                    # Generate document summary from beginning content
                    doc_summary = generate_document_summary(doc['workspace_path'])
                    
                    if doc_summary and 'overview' in doc_summary:
                        with st.expander("📄 Document Summary", expanded=True):
                            st.markdown(doc_summary['overview'])
                            
                            # Show metadata about the summary
                            if 'chars_used' in doc_summary:
                                st.caption(f"📊 Based on {doc_summary['chars_used']:,} characters from document beginning")
                            
                            if doc_summary.get('fallback'):
                                st.info("💡 Fallback summary (AI generation unavailable)")
                    elif doc_summary and 'error' in doc_summary:
                        st.warning(f"⚠️ Could not generate summary: {doc_summary['error']}")
                    else:
                        st.info("📋 Summary generation in progress...")
                else:
                    st.info(f"Document is {doc['status'].lower()}. Summary will be available once processing completes.")
            
            # Enhanced Community Insights Section (moved below)
            if doc['status'] == 'COMPLETED':
                st.markdown("**🏘️ Key Themes & Community Insights**")
                
                # Load document overview data
                overview_data = load_document_overview(doc['workspace_path'])
                
                if overview_data and overview_data.get('top_summaries'):
                    # Get top 3 most relevant communities
                    relevant_communities = get_top_communities(overview_data, limit=3)
                    
                    if relevant_communities:
                        for i, summary in enumerate(relevant_communities):
                            community_title = summary.get('title', f'Community {i+1}')
                            community_summary = summary.get('summary', 'No summary available')
                            
                            with st.expander(f"🏘️ {community_title}", expanded=(i == 0)):
                                st.markdown(community_summary)
                                
                                # Show rating if available
                                if 'rating' in summary:
                                    st.caption(f"📊 Relevance Score: {summary['rating']:.1f}/10")
                    else:
                        st.info("No community insights available.")
                else:
                    st.info("Community analysis not yet available.")
            
            # Key entities section
            if doc['status'] == 'COMPLETED':
                st.markdown("**👥 Key Entities**")
                
                key_entities = load_key_entities(doc['workspace_path'], limit=8)
                
                if key_entities:
                    # Display entities in a grid
                    entity_cols = st.columns(4)
                    for i, entity in enumerate(key_entities):
                        with entity_cols[i % 4]:
                            entity_type = entity.get('type', 'Unknown')
                            entity_name = entity.get('title', 'Unknown')
                            entity_desc = entity.get('description', 'No description')
                            
                            # Color coding for entity types
                            type_colors = {
                                'PERSON': '#FF6B6B',
                                'ORGANIZATION': '#4ECDC4',
                                'LOCATION': '#45B7D1',
                                'EVENT': '#F7DC6F',
                                'CONCEPT': '#82E0AA',
                            }
                            color = type_colors.get(entity_type, '#95A5A6')
                            
                            st.markdown(f"""
                            <div style="background: {color}; color: white; padding: 0.5rem; border-radius: 6px; margin: 0.25rem 0; text-align: center; font-size: 0.8rem;">
                                <strong>{entity_name}</strong><br>
                                <em>{entity_type}</em>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show description on hover/expander
                            if entity_desc and len(entity_desc) > 50:
                                with st.expander("ℹ️", expanded=False):
                                    st.caption(entity_desc[:200] + "..." if len(entity_desc) > 200 else entity_desc)
                else:
                    st.info("No key entities extracted yet.")
            
            # Error information if failed
            if doc['status'] == 'ERROR' and doc.get('error_message'):
                st.error(f"**Error**: {doc['error_message']}")
            
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
    
    # Query method selection
    col1, col2 = st.columns([3, 1])
    with col1:
        query_method = st.selectbox(
            "Search Method:",
            options=["local", "global", "drift", "basic"],
            index=0,
            help="Local: specific details, Global: themes/patterns, Drift: contextual search, Basic: simple search",
            key="chat_query_method"
        )
        
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
                # Validate query
                is_valid, error_msg = validate_query(query_input)
                if not is_valid:
                    st.error(f"Invalid query: {error_msg}")
                else:
                    # Run query
                    with st.spinner(f"Running {query_method} search..."):
                        try:
                            # Select appropriate search method
                            if query_method == "global":
                                result = global_search(selected_doc_ids, query_input)
                            elif query_method == "local":
                                result = local_search(selected_doc_ids, query_input)
                            elif query_method == "drift":
                                result = drift_search(selected_doc_ids, query_input)
                            elif query_method == "basic":
                                result = basic_search(selected_doc_ids, query_input)
                            else:
                                result = local_search(selected_doc_ids, query_input)
                            
                            # Add to chat history
                            chat_entry = {
                                "query": query_input,
                                "response": result.response,
                                "method": query_method,
                                "success": result.success,
                                "source_documents": result.source_documents,
                                "error": result.error_message,
                                "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                                "raw_response": result.raw_response,
                                "context_info": result.context_info
                            }
                            st.session_state.chat_history.append(chat_entry)
                            
                            # Show success message
                            if result.success:
                                st.success(f"✅ Query completed using {query_method} search!")
                            else:
                                st.error(f"❌ Query failed: {result.error_message}")
                                
                        except Exception as e:
                            st.error(f"Query failed: {str(e)}")
            
            elif ask_button and not query_input.strip():
                st.warning("Please enter a question first.")
            
            if clear_button:
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💬 Conversation History")
            
            # Show recent chats first (reverse order)
            for i, entry in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10 entries
                with st.expander(f"🙋 {entry['query'][:50]}... ({entry['timestamp']})", expanded=(i == 0)):
                    st.markdown(f"**Method:** {entry['method'].title()}")
                    
                    if entry['success']:
                        st.markdown("**Response:**")
                        st.markdown(entry['response'])
                        
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