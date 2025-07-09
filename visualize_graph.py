#!/usr/bin/env python3
"""
Knowledge Graph Visualization for GraphRAG

This module provides utilities to visualize the knowledge graphs
created by GraphRAG from PDF documents.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import networkx as nx

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install for visualizations.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Install for interactive visualizations.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphRAGVisualizer:
    """Visualize GraphRAG knowledge graphs."""
    
    # Color scheme for different entity types
    ENTITY_COLORS = {
        "person": "#FF6B6B",
        "organization": "#4ECDC4",
        "location": "#45B7D1",
        "event": "#F7DC6F",
        "product": "#BB8FCE",
        "technology": "#85C1E2",
        "concept": "#82E0AA",
        "document": "#F8C471",
        "default": "#95A5A6"
    }
    
    def __init__(self, output_dir: str = "./graphrag/output"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory containing GraphRAG output
        """
        self.output_dir = Path(output_dir)
        self.graph = None
        self.entities = None
        self.relationships = None
    
    def load_graph_data(self) -> bool:
        """Load graph data from GraphRAG output."""
        try:
            # Load entities
            entities_path = self.output_dir / "entities.parquet"
            if entities_path.exists():
                self.entities = pd.read_parquet(entities_path)
                logger.info(f"Loaded {len(self.entities)} entities")
            else:
                logger.error(f"Entities file not found: {entities_path}")
                return False
            
            # Load relationships
            relationships_path = self.output_dir / "relationships.parquet"
            if relationships_path.exists():
                self.relationships = pd.read_parquet(relationships_path)
                logger.info(f"Loaded {len(self.relationships)} relationships")
            else:
                logger.error(f"Relationships file not found: {relationships_path}")
                return False
            
            # Create NetworkX graph
            self._create_networkx_graph()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load graph data: {e}")
            return False
    
    def _create_networkx_graph(self):
        """Create NetworkX graph from entities and relationships."""
        self.graph = nx.Graph()
        
        # Add nodes with attributes
        for _, entity in self.entities.iterrows():
            self.graph.add_node(
                entity['title'],
                type=entity.get('type', 'unknown'),
                description=entity.get('description', ''),
                degree=entity.get('degree', 0)
            )
        
        # Add edges with weights
        for _, rel in self.relationships.iterrows():
            if rel['source'] in self.graph and rel['target'] in self.graph:
                self.graph.add_edge(
                    rel['source'],
                    rel['target'],
                    weight=rel.get('weight', 1),
                    description=rel.get('description', '')
                )
    
    def get_subgraph(self, 
                     center_node: str, 
                     max_depth: int = 2,
                     max_nodes: int = 50) -> nx.Graph:
        """
        Extract a subgraph centered on a specific node.
        
        Args:
            center_node: The node to center the subgraph on
            max_depth: Maximum distance from center node
            max_nodes: Maximum number of nodes to include
            
        Returns:
            NetworkX subgraph
        """
        if center_node not in self.graph:
            logger.error(f"Node '{center_node}' not found in graph")
            return nx.Graph()
        
        # Get nodes within max_depth
        nodes = {center_node}
        for depth in range(max_depth):
            new_nodes = set()
            for node in nodes:
                new_nodes.update(self.graph.neighbors(node))
            nodes.update(new_nodes)
            if len(nodes) >= max_nodes:
                break
        
        # Limit to max_nodes
        if len(nodes) > max_nodes:
            # Prioritize by degree
            node_degrees = [(n, self.graph.degree(n)) for n in nodes]
            node_degrees.sort(key=lambda x: x[1], reverse=True)
            nodes = {n[0] for n in node_degrees[:max_nodes]}
            # Always include center node
            nodes.add(center_node)
        
        return self.graph.subgraph(nodes)
    
    def visualize_matplotlib(self,
                           subgraph: Optional[nx.Graph] = None,
                           title: str = "Knowledge Graph",
                           figsize: Tuple[int, int] = (15, 10),
                           save_path: Optional[str] = None):
        """
        Create a static visualization using matplotlib.
        
        Args:
            subgraph: Subgraph to visualize (uses full graph if None)
            title: Title for the visualization
            figsize: Figure size
            save_path: Path to save the figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib not available. Install with: pip install matplotlib")
            return
        
        G = subgraph or self.graph
        
        plt.figure(figsize=figsize)
        
        # Layout
        if len(G.nodes) < 50:
            pos = nx.spring_layout(G, k=2, iterations=50)
        else:
            pos = nx.kamada_kawai_layout(G)
        
        # Draw nodes by type
        for entity_type, color in self.ENTITY_COLORS.items():
            nodes = [n for n, d in G.nodes(data=True) 
                    if d.get('type', '').lower() == entity_type]
            if nodes:
                nx.draw_networkx_nodes(
                    G, pos, nodelist=nodes,
                    node_color=color,
                    node_size=[G.degree(n) * 100 for n in nodes],
                    alpha=0.8,
                    label=entity_type.capitalize()
                )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            alpha=0.5,
            width=[G[u][v]['weight'] * 0.5 for u, v in G.edges()]
        )
        
        # Draw labels for high-degree nodes
        labels = {}
        for node, degree in dict(G.degree()).items():
            if degree > 3 or len(G.nodes) < 20:
                labels[node] = node
        
        nx.draw_networkx_labels(
            G, pos, labels,
            font_size=8,
            font_weight='bold'
        )
        
        plt.title(title, fontsize=16)
        plt.legend(loc='upper left', fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to: {save_path}")
        
        plt.show()
    
    def visualize_plotly(self,
                        subgraph: Optional[nx.Graph] = None,
                        title: str = "Interactive Knowledge Graph",
                        save_path: Optional[str] = None):
        """
        Create an interactive visualization using plotly.
        
        Args:
            subgraph: Subgraph to visualize (uses full graph if None)
            title: Title for the visualization
            save_path: Path to save the HTML file
        """
        if not PLOTLY_AVAILABLE:
            logger.error("plotly not available. Install with: pip install plotly")
            return
        
        G = subgraph or self.graph
        
        # Layout
        if len(G.nodes) < 100:
            pos = nx.spring_layout(G, k=3, iterations=50)
        else:
            pos = nx.kamada_kawai_layout(G)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=G[edge[0]][edge[1]]['weight'] * 0.5, color='#888'),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create node traces by type
        node_traces = []
        for entity_type, color in self.ENTITY_COLORS.items():
            nodes = [n for n, d in G.nodes(data=True) 
                    if d.get('type', '').lower() == entity_type]
            
            if not nodes:
                continue
            
            node_x = [pos[node][0] for node in nodes]
            node_y = [pos[node][1] for node in nodes]
            node_text = []
            node_size = []
            
            for node in nodes:
                degree = G.degree(node)
                node_size.append(degree * 5 + 10)
                
                # Create hover text
                hover = f"<b>{node}</b><br>"
                hover += f"Type: {entity_type}<br>"
                hover += f"Connections: {degree}<br>"
                
                # Add description if available
                desc = G.nodes[node].get('description', '')
                if desc:
                    desc = desc[:100] + "..." if len(desc) > 100 else desc
                    hover += f"Description: {desc}"
                
                node_text.append(hover)
            
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=[n if G.degree(n) > 3 else '' for n in nodes],
                textposition="top center",
                textfont=dict(size=8),
                name=entity_type.capitalize(),
                hoverinfo='text',
                hovertext=node_text,
                marker=dict(
                    size=node_size,
                    color=color,
                    line=dict(width=2, color='white')
                )
            )
            node_traces.append(node_trace)
        
        # Create figure
        fig = go.Figure(data=edge_traces + node_traces)
        
        fig.update_layout(
            title=title,
            titlefont_size=20,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved interactive visualization to: {save_path}")
        
        fig.show()
    
    def create_report(self, save_path: str = "graph_report.md"):
        """
        Create a markdown report about the knowledge graph.
        
        Args:
            save_path: Path to save the report
        """
        report = ["# Knowledge Graph Report\n"]
        report.append(f"Generated from: {self.output_dir}\n")
        
        # Basic statistics
        report.append("## Graph Statistics\n")
        report.append(f"- **Total Entities**: {len(self.entities)}")
        report.append(f"- **Total Relationships**: {len(self.relationships)}")
        report.append(f"- **Graph Density**: {nx.density(self.graph):.4f}")
        report.append(f"- **Connected Components**: {nx.number_connected_components(self.graph)}")
        
        # Entity type distribution
        report.append("\n## Entity Type Distribution\n")
        type_counts = self.entities['type'].value_counts()
        for entity_type, count in type_counts.items():
            percentage = (count / len(self.entities)) * 100
            report.append(f"- **{entity_type}**: {count} ({percentage:.1f}%)")
        
        # Most connected entities
        report.append("\n## Most Connected Entities\n")
        degrees = dict(self.graph.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
        
        for node, degree in top_nodes:
            node_type = self.graph.nodes[node].get('type', 'unknown')
            report.append(f"- **{node}** ({node_type}): {degree} connections")
        
        # Relationship analysis
        report.append("\n## Relationship Analysis\n")
        if 'weight' in self.relationships.columns:
            report.append(f"- **Average Relationship Weight**: {self.relationships['weight'].mean():.2f}")
            report.append(f"- **Max Relationship Weight**: {self.relationships['weight'].max():.2f}")
        
        # Community detection
        report.append("\n## Community Structure\n")
        if len(self.graph.nodes) < 1000:  # Only for smaller graphs
            communities = list(nx.community.greedy_modularity_communities(self.graph))
            report.append(f"- **Number of Communities**: {len(communities)}")
            report.append(f"- **Modularity**: {nx.community.modularity(self.graph, communities):.4f}")
            
            # Top communities
            report.append("\n### Largest Communities\n")
            sorted_communities = sorted(communities, key=len, reverse=True)[:5]
            for i, community in enumerate(sorted_communities):
                report.append(f"\n**Community {i+1}** ({len(community)} members):")
                # Show top members by degree
                community_degrees = [(n, degrees[n]) for n in community]
                community_degrees.sort(key=lambda x: x[1], reverse=True)
                for node, deg in community_degrees[:5]:
                    node_type = self.graph.nodes[node].get('type', 'unknown')
                    report.append(f"  - {node} ({node_type}, degree: {deg})")
        
        # Save report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Saved report to: {save_path}")
        return '\n'.join(report)


def main():
    """Example usage of the visualizer."""
    visualizer = GraphRAGVisualizer(output_dir="./graphrag/output")
    
    # Load graph data
    if not visualizer.load_graph_data():
        logger.error("Failed to load graph data. Make sure GraphRAG has been run.")
        return
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # 1. Full graph visualization (if small enough)
    if len(visualizer.graph.nodes) < 100:
        visualizer.visualize_matplotlib(
            title="Complete Knowledge Graph",
            save_path="full_graph.png"
        )
    
    # 2. Find most connected node and visualize its neighborhood
    if visualizer.graph.nodes:
        degrees = dict(visualizer.graph.degree())
        center_node = max(degrees.items(), key=lambda x: x[1])[0]
        
        subgraph = visualizer.get_subgraph(center_node, max_depth=2, max_nodes=30)
        visualizer.visualize_matplotlib(
            subgraph=subgraph,
            title=f"Neighborhood of '{center_node}'",
            save_path=f"subgraph_{center_node.replace(' ', '_')}.png"
        )
        
        # Interactive version
        visualizer.visualize_plotly(
            subgraph=subgraph,
            title=f"Interactive: Neighborhood of '{center_node}'",
            save_path="interactive_graph.html"
        )
    
    # 3. Generate report
    visualizer.create_report()
    
    logger.info("Visualization complete!")


if __name__ == "__main__":
    main()