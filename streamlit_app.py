#!/usr/bin/env python3
"""
Streamlit Interactive GraphRAG Viewer
Simple interface to explore the Baltimore City knowledge graph
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

# Page configuration
st.set_page_config(
    page_title="Baltimore City GraphRAG Explorer",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Baltimore City GraphRAG Explorer")
st.markdown("Interactive exploration of the Baltimore City Engineering Specifications knowledge graph")

# Cache data loading
@st.cache_data
def load_graph_data():
    """Load GraphRAG output data."""
    try:
        # Load entities
        entities = pd.read_parquet('graphrag/output/entities.parquet')
        
        # Load relationships
        relationships = pd.read_parquet('graphrag/output/relationships.parquet')
        
        # Load communities
        communities = pd.read_parquet('graphrag/output/communities.parquet')
        
        # Load community reports
        community_reports = pd.read_parquet('graphrag/output/community_reports.parquet')
        
        return entities, relationships, communities, community_reports
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

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

def main():
    """Main Streamlit app."""
    
    # Load data
    entities, relationships, communities, community_reports = load_graph_data()
    
    if entities is None:
        st.error("Failed to load GraphRAG data. Make sure the GraphRAG processing has been completed.")
        return
    
    # Create NetworkX graph
    G = create_networkx_graph(entities, relationships)
    
    # Sidebar
    st.sidebar.header("Graph Controls")
    
    # Filtering controls
    st.sidebar.subheader("Filtering Options")
    
    # Entity type filter
    entity_types = get_entity_type_options(entities)
    selected_types = st.sidebar.multiselect(
        "Entity Types",
        entity_types,
        default=entity_types
    )
    
    # Degree range filter
    min_degree, max_degree = get_degree_range(entities)
    degree_range = st.sidebar.slider(
        "Degree Range",
        min_value=min_degree,
        max_value=max_degree,
        value=(min_degree, max_degree)
    )
    
    # Search filter
    search_term = st.sidebar.text_input("Search Entities")
    
    # Layout options
    layout_option = st.sidebar.selectbox(
        "Layout Algorithm",
        ["barnes_hut", "force_atlas_2", "hierarchical"],
        index=0
    )
    
    # Performance optimization
    optimize_graph = st.sidebar.checkbox(
        "Optimize for Large Graphs (top 100 nodes)",
        value=len(entities) > 100
    )
    
    # Graph statistics
    st.sidebar.subheader("Graph Statistics")
    st.sidebar.metric("Total Entities", len(entities))
    st.sidebar.metric("Total Relationships", len(relationships))
    st.sidebar.metric("Graph Density", f"{nx.density(G):.3f}")
    st.sidebar.metric("Connected Components", nx.number_connected_components(G))
    
    # Main content - Full width graph
    # Optimize for large graphs if enabled
    display_entities = entities
    display_relationships = relationships
    
    if optimize_graph:
        display_entities, display_relationships = optimize_for_large_graph(
            entities, relationships, threshold=100
        )
        st.info(f"Displaying top {len(display_entities)} entities for performance")
    
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
            
            # Display in Streamlit with full width
            components.html(html_content, height=850)
    except Exception as e:
        st.error(f"Error creating graph: {e}")
        st.info("Try adjusting the filters or switching to a different layout algorithm.")
    

if __name__ == "__main__":
    main()