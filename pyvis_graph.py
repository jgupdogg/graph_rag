#!/usr/bin/env python3
"""
PyVis Network Creation Module
Creates interactive PyVis networks from GraphRAG data with rich popups and styling
"""

import pandas as pd
import networkx as nx
from pyvis.network import Network
import json
from typing import Dict, List, Optional, Tuple
import streamlit as st
import tempfile
import os

# Entity type colors from the integration plan
ENTITY_COLORS = {
    'PERSON': '#FF6B6B',
    'ORGANIZATION': '#4ECDC4', 
    'LOCATION': '#45B7D1',
    'DOCUMENT': '#F7DC6F',
    'EVENT': '#F7DC6F',
    'CONCEPT': '#82E0AA',
    'unknown': '#95A5A6'
}

# Colors for different source documents (cycling through these)
DOCUMENT_COLORS = [
    '#FF6B6B',  # Red
    '#4ECDC4',  # Teal
    '#45B7D1',  # Blue
    '#F7DC6F',  # Yellow
    '#82E0AA',  # Green
    '#E67E22',  # Orange
    '#9B59B6',  # Purple
    '#E74C3C',  # Dark Red
    '#1ABC9C',  # Turquoise
    '#3498DB',  # Light Blue
]

# PyVis network configuration
PYVIS_CONFIG = {
    "physics": {
        "enabled": True,
        "solver": "barnesHut",
        "barnesHut": {
            "gravitationalConstant": -8000,
            "centralGravity": 0.3,
            "springLength": 95,
            "springConstant": 0.04,
            "damping": 0.09
        }
    },
    "interaction": {
        "hover": True,
        "tooltipDelay": 200,
        "hideEdgesOnDrag": False,
        "navigationButtons": True,
        "keyboard": True
    },
    "nodes": {
        "shape": "dot",
        "scaling": {
            "min": 10,
            "max": 30
        }
    }
}

def create_node_popup_html(entity: pd.Series, connections: List[str]) -> str:
    """Create simplified popup content for a node that works with PyVis."""
    entity_type = entity.get('type', 'unknown')
    entity_name = entity.get('title', 'Unknown')
    description = entity.get('description', 'No description available')
    degree = entity.get('degree', 0)
    frequency = entity.get('frequency', 0)
    source_document = entity.get('source_document', 'Unknown')
    
    # Limit description length
    if len(description) > 200:
        description = description[:200] + "..."
    
    # Build connections list (limit to first 5 for popup)
    connections_text = ""
    if connections:
        limited_connections = connections[:5]
        connections_text = ", ".join(limited_connections)
        if len(connections) > 5:
            connections_text += f" (and {len(connections) - 5} more)"
    
    # Create simple text-based popup (PyVis handles HTML differently)
    popup_text = f"""<b>{entity_name}</b><br/>
Type: {entity_type}<br/>
Source: {source_document}<br/>
Connections: {degree}<br/>
Frequency: {frequency}<br/>
<br/>
Description: {description}<br/>
{f'<br/>Connected to: {connections_text}' if connections_text else ''}
"""
    
    return popup_text

def get_document_color(source_document: str, document_color_map: Dict[str, str]) -> str:
    """Get color for a document, assigning a new one if not seen before."""
    if source_document not in document_color_map:
        color_index = len(document_color_map) % len(DOCUMENT_COLORS)
        document_color_map[source_document] = DOCUMENT_COLORS[color_index]
    return document_color_map[source_document]

def create_legend_html(document_color_map: Dict[str, str]) -> str:
    """Create HTML legend for document colors."""
    if not document_color_map:
        return ""
    
    legend_items = []
    for doc, color in document_color_map.items():
        doc_name = doc[:30] + "..." if len(doc) > 30 else doc
        legend_items.append(f'<span style="color: {color}; font-weight: bold;">● {doc_name}</span>')
    
    return f"""
    <div style="position: fixed; top: 10px; right: 10px; background: rgba(255,255,255,0.9); 
                padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                z-index: 1000; max-width: 300px;">
        <h4 style="margin: 0 0 10px 0; color: #333;">Document Sources</h4>
        <div style="font-size: 12px; line-height: 1.4;">
            {('<br/>'.join(legend_items))}
        </div>
    </div>
    """

def calculate_node_size(degree: int) -> int:
    """Calculate node size based on degree."""
    base_size = 15
    size_multiplier = 2
    max_size = 50
    min_size = 10
    
    calculated_size = base_size + (degree * size_multiplier)
    return max(min_size, min(calculated_size, max_size))

def create_pyvis_network(entities: pd.DataFrame, relationships: pd.DataFrame, 
                        entity_filter: Optional[List[str]] = None,
                        degree_range: Optional[Tuple[int, int]] = None,
                        search_term: Optional[str] = None,
                        layout: str = "barnes_hut") -> Network:
    """Create PyVis network from GraphRAG data with filtering options and multi-document support."""
    
    # Create NetworkX graph first for analysis
    G = nx.Graph()
    
    # Filter entities
    filtered_entities = entities.copy()
    
    if entity_filter:
        filtered_entities = filtered_entities[filtered_entities['type'].isin(entity_filter)]
    
    if degree_range:
        min_degree, max_degree = degree_range
        filtered_entities = filtered_entities[
            (filtered_entities['degree'] >= min_degree) & 
            (filtered_entities['degree'] <= max_degree)
        ]
    
    if search_term:
        filtered_entities = filtered_entities[
            filtered_entities['title'].str.contains(search_term, case=False, na=False)
        ]
    
    # Create document color mapping
    document_color_map = {}
    if 'source_document' in filtered_entities.columns:
        unique_docs = filtered_entities['source_document'].unique()
        for i, doc in enumerate(unique_docs):
            document_color_map[doc] = DOCUMENT_COLORS[i % len(DOCUMENT_COLORS)]
    
    # Add nodes to NetworkX graph
    for _, entity in filtered_entities.iterrows():
        G.add_node(
            entity['title'],
            type=entity.get('type', 'unknown'),
            description=entity.get('description', ''),
            degree=entity.get('degree', 0),
            frequency=entity.get('frequency', 0),
            source_document=entity.get('source_document', 'Unknown')
        )
    
    # Add edges, only if both nodes exist
    filtered_relationships = relationships[
        (relationships['source'].isin(filtered_entities['title'])) &
        (relationships['target'].isin(filtered_entities['title']))
    ]
    
    for _, rel in filtered_relationships.iterrows():
        if rel['source'] in G and rel['target'] in G:
            G.add_edge(
                rel['source'],
                rel['target'],
                weight=rel.get('weight', 1),
                description=rel.get('description', ''),
                source_document=rel.get('source_document', 'Unknown')
            )
    
    # Create PyVis network
    net = Network(
        height="800px",
        width="100%",
        bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        directed=False,
        notebook=False,
        cdn_resources='remote',
        select_menu=True,
        filter_menu=True
    )
    
    # Add nodes with popups
    for node in G.nodes():
        node_data = G.nodes[node]
        connections = list(G.neighbors(node))
        
        # Get corresponding entity data for popup
        entity_data = filtered_entities[filtered_entities['title'] == node]
        if not entity_data.empty:
            entity_series = entity_data.iloc[0]
        else:
            entity_series = pd.Series({
                'title': node,
                'type': node_data.get('type', 'unknown'),
                'description': node_data.get('description', ''),
                'degree': G.degree(node),
                'frequency': node_data.get('frequency', 0),
                'source_document': node_data.get('source_document', 'Unknown')
            })
        
        popup_html = create_node_popup_html(entity_series, connections)
        
        # Use document color if available, otherwise use entity type color
        if 'source_document' in node_data and node_data['source_document'] in document_color_map:
            node_color = document_color_map[node_data['source_document']]
        else:
            node_color = ENTITY_COLORS.get(node_data.get('type', 'unknown'), '#95A5A6')
        
        net.add_node(
            node,
            label=node if len(node) <= 20 else node[:17] + "...",
            title=popup_html,
            color=node_color,
            size=calculate_node_size(G.degree(node))
        )
    
    # Add edges with cross-document highlighting
    for edge in G.edges():
        edge_data = G[edge[0]][edge[1]]
        source_node_data = G.nodes[edge[0]]
        target_node_data = G.nodes[edge[1]]
        
        # Check if this is a cross-document relationship
        source_doc = source_node_data.get('source_document', 'Unknown')
        target_doc = target_node_data.get('source_document', 'Unknown')
        
        if source_doc != target_doc and source_doc != 'Unknown' and target_doc != 'Unknown':
            # Cross-document edge - make it thicker and different color
            edge_color = "#FF4444"
            edge_width = 3
            edge_title = f"Cross-document connection: {source_doc} ↔ {target_doc}"
        else:
            # Same document edge
            edge_color = "#888888"
            edge_width = 1
            edge_title = edge_data.get('description', '')
        
        net.add_edge(
            edge[0],
            edge[1],
            title=edge_title,
            weight=edge_data.get('weight', 1),
            color=edge_color,
            width=edge_width
        )
    
    # Apply configuration
    net.set_options(json.dumps(PYVIS_CONFIG))
    
    # Store document color mapping for legend
    net.document_color_map = document_color_map
    
    return net

def render_pyvis_network(net: Network) -> str:
    """Render PyVis network and return HTML content with document legend."""
    try:
        # Generate HTML directly without using show() to avoid template issues
        html_content = net.generate_html()
        
        # Add document legend if available
        legend_html = ""
        if hasattr(net, 'document_color_map') and net.document_color_map:
            legend_html = create_legend_html(net.document_color_map)
        
        # If generate_html() returns None, create a minimal HTML structure
        if html_content is None:
            # Fallback: create basic HTML structure
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Network Graph</title>
                <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            </head>
            <body>
                {legend_html}
                <div id="mynetworkid" style="width: 100%; height: 750px; border: 1px solid lightgray;"></div>
                <script>
                    var nodes = new vis.DataSet({net.get_nodes()});
                    var edges = new vis.DataSet({net.get_edges()});
                    var container = document.getElementById('mynetworkid');
                    var data = {{nodes: nodes, edges: edges}};
                    var options = {net.options};
                    var network = new vis.Network(container, data, options);
                </script>
            </body>
            </html>
            """
        else:
            # Insert legend into generated HTML
            if legend_html:
                # Find the body tag and insert legend after it
                body_start = html_content.find('<body>')
                if body_start != -1:
                    insert_pos = body_start + len('<body>')
                    html_content = html_content[:insert_pos] + legend_html + html_content[insert_pos:]
        
        return html_content
        
    except Exception as e:
        # If PyVis fails completely, return an error message
        return f"""
        <html>
        <body>
            <div style="padding: 20px; text-align: center; color: red;">
                <h3>Error rendering network visualization</h3>
                <p>Error: {str(e)}</p>
                <p>Please try switching to Plotly visualization.</p>
            </div>
        </body>
        </html>
        """

def get_entity_type_options(entities: pd.DataFrame) -> List[str]:
    """Get unique entity types for filtering."""
    return sorted(entities['type'].unique().tolist())

def get_degree_range(entities: pd.DataFrame) -> Tuple[int, int]:
    """Get min and max degree values."""
    min_degree = int(entities['degree'].min()) if not entities['degree'].empty else 0
    max_degree = int(entities['degree'].max()) if not entities['degree'].empty else 0
    return min_degree, max_degree

def optimize_for_large_graph(entities: pd.DataFrame, relationships: pd.DataFrame, 
                           threshold: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Optimize graph for large datasets by filtering to most important nodes."""
    if len(entities) <= threshold:
        return entities, relationships
    
    # Sort by degree and take top nodes
    top_entities = entities.nlargest(threshold, 'degree')
    
    # Filter relationships to only include connections between top entities
    filtered_relationships = relationships[
        (relationships['source'].isin(top_entities['title'])) &
        (relationships['target'].isin(top_entities['title']))
    ]
    
    return top_entities, filtered_relationships