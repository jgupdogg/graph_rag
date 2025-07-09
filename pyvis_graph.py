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
Connections: {degree}<br/>
Frequency: {frequency}<br/>
<br/>
Description: {description}<br/>
{f'<br/>Connected to: {connections_text}' if connections_text else ''}
"""
    
    return popup_text

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
    """Create PyVis network from GraphRAG data with filtering options."""
    
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
    
    # Add nodes to NetworkX graph
    for _, entity in filtered_entities.iterrows():
        G.add_node(
            entity['title'],
            type=entity.get('type', 'unknown'),
            description=entity.get('description', ''),
            degree=entity.get('degree', 0),
            frequency=entity.get('frequency', 0)
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
                description=rel.get('description', '')
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
        
        popup_html = create_node_popup_html(
            pd.Series({
                'title': node,
                'type': node_data.get('type', 'unknown'),
                'description': node_data.get('description', ''),
                'degree': G.degree(node),
                'frequency': node_data.get('frequency', 0)
            }),
            connections
        )
        
        net.add_node(
            node,
            label=node if len(node) <= 20 else node[:17] + "...",
            title=popup_html,
            color=ENTITY_COLORS.get(node_data.get('type', 'unknown'), '#95A5A6'),
            size=calculate_node_size(G.degree(node))
        )
    
    # Add edges
    for edge in G.edges():
        edge_data = G[edge[0]][edge[1]]
        net.add_edge(
            edge[0],
            edge[1],
            title=edge_data.get('description', ''),
            weight=edge_data.get('weight', 1),
            color="#888888"
        )
    
    # Apply configuration
    net.set_options(json.dumps(PYVIS_CONFIG))
    
    return net

def render_pyvis_network(net: Network) -> str:
    """Render PyVis network and return HTML content."""
    try:
        # Generate HTML directly without using show() to avoid template issues
        html_content = net.generate_html()
        
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