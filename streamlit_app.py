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
    
    # Layout selection
    layout_type = st.sidebar.selectbox(
        "Layout Algorithm",
        ["spring", "circular", "kamada_kawai"],
        index=0
    )
    
    # Graph statistics
    st.sidebar.subheader("Graph Statistics")
    st.sidebar.metric("Total Entities", len(entities))
    st.sidebar.metric("Total Relationships", len(relationships))
    st.sidebar.metric("Graph Density", f"{nx.density(G):.3f}")
    st.sidebar.metric("Connected Components", nx.number_connected_components(G))
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Interactive Knowledge Graph")
        
        # Create and display graph
        fig = create_plotly_graph(G, layout_type)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Entity Details")
        
        # Entity type distribution
        entity_types = entities['type'].value_counts()
        fig_pie = px.pie(
            values=entity_types.values,
            names=entity_types.index,
            title="Entity Type Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Most connected entities
        st.subheader("Most Connected Entities")
        degrees = dict(G.degree())
        top_entities = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for entity, degree in top_entities:
            entity_type = G.nodes[entity].get('type', 'unknown')
            st.write(f"**{entity}** ({entity_type}): {degree} connections")
    
    # Detailed data views
    st.subheader("Detailed Data")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Entities", "Relationships", "Communities", "Community Reports"])
    
    with tab1:
        st.dataframe(entities[['title', 'type', 'description', 'degree', 'frequency']], use_container_width=True)
    
    with tab2:
        st.dataframe(relationships[['source', 'target', 'description', 'weight']], use_container_width=True)
    
    with tab3:
        if len(communities) > 0:
            st.dataframe(communities, use_container_width=True)
        else:
            st.info("No community data available")
    
    with tab4:
        if len(community_reports) > 0:
            for _, report in community_reports.iterrows():
                st.write(f"**Community {report.get('id', 'Unknown')}**")
                st.write(report.get('content', report.get('summary', 'No content available')))
                st.write("---")
        else:
            st.info("No community reports available")
    
    # Search functionality
    st.subheader("Search Entities")
    search_term = st.text_input("Search for entities:")
    
    if search_term:
        matches = entities[entities['title'].str.contains(search_term, case=False, na=False)]
        if len(matches) > 0:
            st.dataframe(matches[['title', 'type', 'description']], use_container_width=True)
        else:
            st.info("No entities found matching your search.")

if __name__ == "__main__":
    main()