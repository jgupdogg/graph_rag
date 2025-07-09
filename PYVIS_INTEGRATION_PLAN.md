# PyVis Interactive Graph Integration Plan

## Overview

This document outlines the plan to integrate PyVis for creating a highly interactive knowledge graph visualization of the Baltimore City GraphRAG data. PyVis will provide node selection, popup details, and advanced graph interaction capabilities.

## Goals

1. **Interactive Node Selection**: Click nodes to view detailed information
2. **Rich Popups**: Display entity details, descriptions, and connections
3. **Dynamic Filtering**: Filter by entity type, degree, or search terms
4. **Visual Feedback**: Highlight selected nodes and their connections
5. **Export Capabilities**: Save graphs as HTML or images

## Technical Architecture

### Dependencies

```bash
pip install pyvis
pip install streamlit-components-v1
# Existing: streamlit, networkx, pandas, plotly
```

### File Structure

```
graph_rag/
├── pyvis_graph.py              # Core PyVis network creation
├── streamlit_pyvis_app.py      # New Streamlit app with PyVis
├── graph_utils.py              # Shared graph utilities
├── static/
│   ├── custom.css             # Custom styling for popups
│   └── graph_styles.json      # Visual configuration
└── configs/
    └── pyvis_config.json      # PyVis network settings
```

## Implementation Details

### 1. PyVis Network Configuration

```python
# pyvis_config.json
{
    "physics": {
        "enabled": true,
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
        "hover": true,
        "tooltipDelay": 200,
        "hideEdgesOnDrag": false,
        "navigationButtons": true,
        "keyboard": true
    },
    "nodes": {
        "shape": "dot",
        "scaling": {
            "min": 10,
            "max": 30
        }
    }
}
```

### 2. Node Popup Structure

Each node will have an HTML popup containing:

```html
<div class="node-popup">
    <div class="node-header">
        <h3>{entity_name}</h3>
        <span class="entity-type-badge {entity_type_class}">{entity_type}</span>
    </div>
    
    <div class="node-description">
        <p>{description}</p>
    </div>
    
    <div class="node-stats">
        <div class="stat-item">
            <span class="stat-label">Connections:</span>
            <span class="stat-value">{degree}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Frequency:</span>
            <span class="stat-value">{frequency}</span>
        </div>
    </div>
    
    <div class="node-relationships">
        <h4>Connected Entities:</h4>
        <ul class="connection-list">
            {for each connection}
            <li class="connection-item">
                <span class="connection-name">{connected_entity}</span>
                <span class="connection-type">{relationship_type}</span>
            </li>
            {/for}
        </ul>
    </div>
</div>
```

### 3. Visual Design System

#### Entity Type Colors
- **PERSON**: `#FF6B6B` (Coral Red)
- **ORGANIZATION**: `#4ECDC4` (Turquoise)
- **LOCATION**: `#45B7D1` (Sky Blue)
- **DOCUMENT**: `#F7DC6F` (Yellow)
- **CONCEPT**: `#82E0AA` (Mint Green)
- **DEFAULT**: `#95A5A6` (Gray)

#### Node Sizing
- Base size: 10px
- Size multiplier: `degree * 2`
- Maximum size: 50px
- Minimum size: 10px

#### Edge Styling
- Default color: `#888888` (Gray)
- Selected edge: `#FF6B6B` (Highlight)
- Width: Based on relationship weight (1-5px)
- Arrows: Enabled for directed relationships

### 4. Interactive Features

#### Node Interaction
1. **Hover**: Quick preview tooltip
2. **Click**: Full popup with details
3. **Double-click**: Center and zoom on node
4. **Drag**: Reposition node (physics temporarily disabled)

#### Filtering Controls
```python
# Sidebar filters
- Entity Type Checkboxes
  □ Person
  □ Organization  
  □ Location
  □ Document
  □ Concept

- Degree Range Slider
  Min: [1]  Max: [15]

- Search Box
  [Search entities...]

- Layout Options
  ○ Barnes-Hut (default)
  ○ Force Atlas 2
  ○ Hierarchical
  ○ Circular
```

#### Selection Behavior
1. Click node → Node highlighted in red
2. Connected nodes → Secondary highlight in orange
3. Connected edges → Highlighted and thickened
4. Non-connected elements → Faded (opacity 0.3)

### 5. Performance Optimizations

#### Large Graph Handling
```python
def optimize_large_graph(G, threshold=50):
    if len(G.nodes) > threshold:
        # Enable clustering
        options["clustering"] = {
            "enabled": True,
            "clusterByConnection": True
        }
        
        # Reduce physics iterations
        options["physics"]["stabilization"] = {
            "iterations": 100
        }
        
        # Enable node caching
        options["nodes"]["chosen"] = False
```

#### Progressive Loading
1. Initial load: Top 30 nodes by degree
2. User action: Load connected nodes on demand
3. Search: Dynamically add search results

### 6. Streamlit Integration

#### App Layout
```
┌─────────────────────────────────────────┐
│            Baltimore GraphRAG Explorer   │
├─────────┬───────────────────────────────┤
│         │                               │
│ Filters │    PyVis Network Graph        │
│   &     │                               │
│Controls │    (Interactive)              │
│         │                               │
│         ├───────────────────────────────┤
│         │   Selected Node Details       │
│         │   - Name, Type, Description   │
│         │   - Connections List          │
│         │   - Related Documents         │
└─────────┴───────────────────────────────┘
```

#### Component Structure
```python
# Main components
1. st.sidebar: Filtering and controls
2. col1 (70%): PyVis network
3. col2 (30%): Node details panel
4. Bottom tabs: Data tables and search
```

### 7. Advanced Features

#### Path Finding
```python
def find_shortest_path(G, source, target):
    """Find and highlight shortest path between nodes."""
    path = nx.shortest_path(G, source, target)
    highlight_path(path)
    return path
```

#### Community Detection
```python
def color_by_community(G):
    """Color nodes based on community detection."""
    communities = nx.community.louvain_communities(G)
    apply_community_colors(G, communities)
```

#### Subgraph Extraction
```python
def extract_subgraph(G, selected_nodes, depth=1):
    """Extract subgraph around selected nodes."""
    subgraph_nodes = set(selected_nodes)
    for _ in range(depth):
        for node in list(subgraph_nodes):
            subgraph_nodes.update(G.neighbors(node))
    return G.subgraph(subgraph_nodes)
```

### 8. Export Options

#### HTML Export
- Full interactive graph as standalone HTML
- Includes all JavaScript and CSS inline
- Shareable without server

#### Image Export
- PNG/SVG format
- Current view with zoom/pan state
- High resolution for printing

#### Data Export
- Selected nodes as CSV
- Subgraph as GraphML
- Filtered data as JSON

## Implementation Timeline

### Phase 1: Core PyVis Integration (Day 1)
- [ ] Install PyVis and dependencies
- [ ] Create basic PyVis network from GraphRAG data
- [ ] Implement node popups with entity information
- [ ] Add basic styling and colors

### Phase 2: Interactive Features (Day 2)
- [ ] Add filtering controls in sidebar
- [ ] Implement node selection and highlighting
- [ ] Add search functionality
- [ ] Create layout options

### Phase 3: Advanced Features (Day 3)
- [ ] Implement path finding
- [ ] Add community visualization
- [ ] Create subgraph extraction
- [ ] Add export capabilities

### Phase 4: Polish & Optimization (Day 4)
- [ ] Optimize for large graphs
- [ ] Add progressive loading
- [ ] Create custom CSS styling
- [ ] Write user documentation

## Success Metrics

1. **Interactivity**: Smooth node selection and popup display
2. **Performance**: Handle 100+ nodes without lag
3. **Usability**: Intuitive controls and clear visual feedback
4. **Information**: Rich node details easily accessible
5. **Export**: Multiple formats for sharing insights

## Example Usage

```python
# Initialize PyVis network
net = Network(height="750px", width="100%", bgcolor="#ffffff")

# Add nodes with popup HTML
for _, entity in entities.iterrows():
    popup_html = create_popup_html(entity)
    net.add_node(
        entity['title'],
        label=entity['title'],
        title=popup_html,
        color=ENTITY_COLORS.get(entity['type'], '#95A5A6'),
        size=calculate_node_size(entity['degree'])
    )

# Add edges with hover info
for _, rel in relationships.iterrows():
    net.add_edge(
        rel['source'],
        rel['target'],
        title=rel['description'],
        weight=rel['weight']
    )

# Apply physics and render
net.set_options(pyvis_config)
net.show("network.html")
```

## Next Steps

1. Review and approve this plan
2. Begin Phase 1 implementation
3. Test with Baltimore City data
4. Iterate based on user feedback
5. Deploy enhanced visualization

This PyVis integration will transform the Baltimore GraphRAG visualization into a truly interactive exploration tool, enabling users to discover insights through intuitive click-and-explore interactions.