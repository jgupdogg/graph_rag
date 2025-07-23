# Dynamic AI-Driven Lecture Format

## Overview

The Audio Lecture Generator now features dynamic, AI-driven formatting that analyzes each document and creates a custom lecture structure optimized for the content type and learning objectives.

## How It Works

### 1. **Document Analysis**
The AI examines:
- Document type and structure
- Content complexity
- Key themes and concepts
- Entity and relationship density
- Educational value

### 2. **Blueprint Generation**
Based on the analysis, the AI creates a lecture blueprint with:
- **Style Selection**: Chooses from narrative, academic, conversational, tutorial, analytical, or summary styles
- **Content Emphasis**: Determines what to focus on (concepts, relationships, examples, insights, technical details, or practical applications)
- **Custom Structure**: Creates sections tailored to the content
- **Duration Optimization**: Adjusts pacing based on target duration

### 3. **Dynamic Components**
The system generates:
- **Opening Hooks**: Engaging introductions tailored to the content
- **Section Types**: Theme explorations, example walkthroughs, concept explanations
- **Transitions**: Natural flow between topics based on style
- **Takeaways**: Key points formatted for the chosen style

## Lecture Styles

### **Narrative**
- Story-like flow
- Journey metaphors
- Engaging progression
- Best for: Historical documents, case studies

### **Academic**
- Traditional lecture format
- Formal structure
- Systematic coverage
- Best for: Technical papers, research

### **Conversational**
- Casual, friendly tone
- Direct engagement
- Relatable examples
- Best for: Tutorials, guides

### **Tutorial**
- Step-by-step approach
- Practical focus
- Clear instructions
- Best for: How-to documents, procedures

### **Analytical**
- Deep examination
- Critical thinking
- Multiple perspectives
- Best for: Analysis reports, critiques

### **Summary**
- High-level overview
- Key points focus
- Efficient coverage
- Best for: Executive summaries, briefs

## Benefits

1. **Optimal Pedagogy**: Each document gets the most effective teaching approach
2. **Better Engagement**: Style matches content for maximum impact
3. **Flexible Duration**: Adapts to time constraints while maintaining quality
4. **Natural Flow**: AI creates smooth transitions and logical progression
5. **Content-Aware**: Emphasizes what matters most in each document

## Usage

1. Enable "AI-Driven Format" in Advanced Options
2. The AI will analyze your document
3. A custom lecture blueprint is created
4. The lecture is generated following this blueprint
5. The final audio maintains the chosen style throughout

## Example Blueprints

### Technical Documentation → Tutorial Style
```json
{
  "style": "tutorial",
  "emphasis": ["technical", "practical"],
  "sections": [
    {"type": "introduction", "focus": "overview"},
    {"type": "example_walkthrough", "focus": "implementation"},
    {"type": "concept_explanation", "focus": "architecture"},
    {"type": "practical_tips", "focus": "best_practices"}
  ]
}
```

### Research Paper → Academic Style
```json
{
  "style": "academic",
  "emphasis": ["concepts", "insights"],
  "sections": [
    {"type": "introduction", "focus": "context"},
    {"type": "theme_exploration", "focus": "methodology"},
    {"type": "analysis", "focus": "findings"},
    {"type": "implications", "focus": "significance"}
  ]
}
```

## Section Summaries Fix

To fix missing section summaries for existing documents:

```bash
python3 fix_section_summaries.py
```

This will:
1. Check all documents for section summaries
2. Identify documents missing them
3. Regenerate section summaries from source files
4. Save them to the workspace cache

Section summaries are essential for high-quality lecture generation as they provide the hierarchical structure needed for coherent audio content.