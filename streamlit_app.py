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
    get_processing_logs
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

# Page configuration
st.set_page_config(
    page_title="GraphRAG Multi-Document Explorer",
    page_icon="📄",
    layout="wide"
)

# Title
st.title("📄 GraphRAG Multi-Document Explorer")
st.markdown("Upload and analyze multiple documents to build interactive knowledge graphs")

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
    st.sidebar.subheader("📄 Document Management")
    
    # Upload new document
    uploaded_file = st.sidebar.file_uploader(
        "Upload a new document",
        type=["pdf", "txt"],
        accept_multiple_files=False
    )
    
    if uploaded_file:
        display_name = st.sidebar.text_input(
            "Document name (optional)",
            value=uploaded_file.name,
            key="doc_name"
        )
        
        if st.sidebar.button("📤 Process Document"):
            with st.spinner("Starting document processing..."):
                result = process_new_document(uploaded_file, display_name)
                st.sidebar.success(result)
                st.rerun()
    
    # Show processing status
    all_docs = processor.get_all_documents()
    processing_docs = [doc for doc in all_docs if doc['status'] == 'PROCESSING']
    failed_docs = [doc for doc in all_docs if doc['status'] == 'ERROR']
    
    if processing_docs:
        st.sidebar.subheader("⏳ Processing Status")
        for doc in processing_docs:
            st.sidebar.info(f"Processing: {doc['display_name']}")
    
    # Show failed documents with retry option
    if failed_docs:
        st.sidebar.subheader("❌ Failed Documents")
        for doc in failed_docs:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.write(f"❌ {doc['display_name']}")
                if doc['error_message']:
                    st.caption(f"Error: {doc['error_message'][:50]}...")
            with col2:
                if st.button("🔄", key=f"retry_{doc['id']}", help="Retry processing"):
                    with st.spinner("Retrying..."):
                        result = reprocess_failed_document(doc['id'])
                        st.sidebar.success(result)
                        st.rerun()
    
    # Document list and selection
    processed_docs = get_processed_documents()
    
    if processed_docs:
        st.sidebar.subheader("📋 Select Documents")
        
        # Create options for multiselect
        doc_options = {}
        for doc in processed_docs:
            label = f"{doc['display_name']} ({doc['created_at'][:10]})"
            doc_options[label] = doc['id']
        
        selected_doc_names = st.sidebar.multiselect(
            "Choose documents to explore:",
            options=doc_options.keys(),
            key="selected_docs"
        )
        
        # Get selected document IDs
        selected_doc_ids = [doc_options[name] for name in selected_doc_names]
        
        # Document management buttons
        if selected_doc_names:
            st.sidebar.subheader("🔧 Document Actions")
            
            # Show processing logs
            if st.sidebar.button("📋 View Processing Logs"):
                st.session_state.show_logs = True
                st.session_state.log_doc_ids = selected_doc_ids
            
            # Delete selected documents
            if st.sidebar.button("🗑️ Delete Selected", type="secondary"):
                for doc_id in selected_doc_ids:
                    if delete_document(doc_id):
                        st.sidebar.success("Document deleted")
                    else:
                        st.sidebar.error("Failed to delete document")
                st.rerun()
        
        return selected_doc_ids
    else:
        st.sidebar.info("No processed documents found. Upload a document to get started.")
        return []

def render_graph_controls(entities, relationships):
    """Render graph control section in sidebar."""
    st.sidebar.subheader("🎛️ Graph Controls")
    
    # Filtering controls
    if not entities.empty:
        # Entity type filter
        entity_types = get_entity_type_options(entities)
        selected_types = st.sidebar.multiselect(
            "Entity Types",
            entity_types,
            default=entity_types,
            key="entity_types"
        )
        
        # Degree range filter
        min_degree, max_degree = get_degree_range(entities)
        if min_degree < max_degree:
            degree_range = st.sidebar.slider(
                "Degree Range",
                min_value=min_degree,
                max_value=max_degree,
                value=(min_degree, max_degree),
                key="degree_range"
            )
        else:
            degree_range = (min_degree, max_degree)
        
        # Search filter
        search_term = st.sidebar.text_input("Search Entities", key="search")
        
        # Layout options
        layout_option = st.sidebar.selectbox(
            "Layout Algorithm",
            ["barnes_hut", "force_atlas_2", "hierarchical"],
            index=0,
            key="layout"
        )
        
        # Performance optimization
        optimize_graph = st.sidebar.checkbox(
            "Optimize for Large Graphs (top 100 nodes)",
            value=len(entities) > 100,
            key="optimize"
        )
        
        return selected_types, degree_range, search_term, layout_option, optimize_graph
    else:
        return [], (0, 0), "", "barnes_hut", False

def render_graph_statistics(entities, relationships, selected_docs):
    """Render graph statistics section."""
    st.sidebar.subheader("📊 Graph Statistics")
    
    if not entities.empty:
        # Create NetworkX graph for statistics
        G = create_networkx_graph(entities, relationships)
        
        st.sidebar.metric("Documents Selected", len(selected_docs))
        st.sidebar.metric("Total Entities", len(entities))
        st.sidebar.metric("Total Relationships", len(relationships))
        st.sidebar.metric("Graph Density", f"{nx.density(G):.3f}")
        st.sidebar.metric("Connected Components", nx.number_connected_components(G))
        
        # Show entities by source document
        if 'source_document' in entities.columns:
            st.sidebar.subheader("📄 Entities by Document")
            doc_counts = entities['source_document'].value_counts()
            for doc, count in doc_counts.items():
                st.sidebar.metric(f"{doc[:20]}...", count)

def main():
    """Main Streamlit app."""
    
    # Sidebar - Document Management
    selected_doc_ids = render_document_management_section()
    
    # Show processing logs if requested
    if st.session_state.get('show_logs', False):
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
                    'UPLOADED': '📤'
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
    
    # Main content area
    if not selected_doc_ids:
        st.info("👈 Select one or more processed documents from the sidebar to begin exploring.")
        st.markdown("""
        ### Getting Started
        1. **Upload a document** using the file uploader in the sidebar
        2. **Wait for processing** to complete (this may take several minutes)
        3. **Select documents** to explore from the processed documents list
        4. **Explore the graph** using the interactive visualization
        """)
        return
    
    # Load data for selected documents
    with st.spinner("Loading and merging graph data..."):
        entities, relationships = load_multi_document_data(selected_doc_ids)
    
    if entities.empty:
        st.warning("No graph data found for the selected documents.")
        return
    
    # Get document names for display
    processed_docs = get_processed_documents()
    selected_doc_names = []
    for doc in processed_docs:
        if doc['id'] in selected_doc_ids:
            selected_doc_names.append(doc['display_name'])
    
    st.header(f"📊 Exploring: {', '.join(selected_doc_names)}")
    st.success(f"✅ Loaded {len(entities)} entities and {len(relationships)} relationships")
    
    # Sidebar - Graph Controls
    selected_types, degree_range, search_term, layout_option, optimize_graph = render_graph_controls(
        entities, relationships
    )
    
    # Sidebar - Statistics
    render_graph_statistics(entities, relationships, selected_doc_ids)
    
    # Main content - Graph visualization
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🔗 Interactive Knowledge Graph")
        
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
            with st.spinner("Creating interactive network..."):
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
                
                # Display in Streamlit
                components.html(html_content, height=600)
        except Exception as e:
            st.error(f"Error creating graph: {e}")
            st.info("Try adjusting the filters or switching to a different layout algorithm.")
    
    with col2:
        st.subheader("🔍 Entity Explorer")
        
        # Show entity details
        if not entities.empty:
            # Entity search
            entity_search = st.text_input("Search for specific entity:", key="entity_search")
            
            if entity_search:
                # Filter entities by search term
                filtered_entities = entities[
                    entities['title'].str.contains(entity_search, case=False, na=False)
                ]
                
                if not filtered_entities.empty:
                    selected_entity = st.selectbox(
                        "Select entity:",
                        filtered_entities['title'].tolist(),
                        key="selected_entity"
                    )
                    
                    # Show entity details
                    entity_data = filtered_entities[filtered_entities['title'] == selected_entity].iloc[0]
                    
                    st.markdown(f"**Type:** {entity_data.get('type', 'Unknown')}")
                    st.markdown(f"**Source:** {entity_data.get('source_document', 'Unknown')}")
                    if 'description' in entity_data:
                        st.markdown(f"**Description:** {entity_data['description']}")
                    
                    # Show related entities
                    if not relationships.empty:
                        related_rels = relationships[
                            (relationships['source'] == selected_entity) | 
                            (relationships['target'] == selected_entity)
                        ]
                        
                        if not related_rels.empty:
                            st.markdown("**Related Entities:**")
                            for _, rel in related_rels.head(10).iterrows():
                                other_entity = rel['target'] if rel['source'] == selected_entity else rel['source']
                                st.markdown(f"• {other_entity}")
                else:
                    st.info("No entities found matching your search.")
            
            # Show sample entities
            st.markdown("**Sample Entities:**")
            sample_entities = entities.sample(min(10, len(entities)))
            for _, entity in sample_entities.iterrows():
                source_doc = entity.get('source_document', 'Unknown')[:20]
                st.markdown(f"• **{entity['title']}** ({source_doc}...)")
        
        # Query interface
        st.subheader("💬 Document Chat")
        
        # Initialize chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Query method selection
        query_method = st.selectbox(
            "Search Method:",
            options=["local", "global", "drift", "basic"],
            index=0,
            help="Local: specific details, Global: themes/patterns, Drift: contextual search, Basic: simple search",
            key="query_method"
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