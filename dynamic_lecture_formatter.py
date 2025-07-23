"""
Dynamic AI-Driven Lecture Format Generator
Uses AI to analyze document content and determine the best lecture structure.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import openai
from openai import OpenAI
from datetime import datetime

logger = logging.getLogger(__name__)


class LectureStyle(Enum):
    """Different lecture presentation styles"""
    NARRATIVE = "narrative"  # Story-like flow
    ACADEMIC = "academic"  # Traditional academic lecture
    CONVERSATIONAL = "conversational"  # Casual, engaging tone
    TUTORIAL = "tutorial"  # Step-by-step instructional
    ANALYTICAL = "analytical"  # Deep analysis and critique
    SUMMARY = "summary"  # High-level overview


class ContentEmphasis(Enum):
    """What to emphasize in the lecture"""
    CONCEPTS = "concepts"  # Focus on key concepts
    RELATIONSHIPS = "relationships"  # Focus on connections
    EXAMPLES = "examples"  # Focus on practical examples
    INSIGHTS = "insights"  # Focus on insights and implications
    TECHNICAL = "technical"  # Focus on technical details
    PRACTICAL = "practical"  # Focus on practical applications


@dataclass
class LectureBlueprint:
    """AI-generated blueprint for lecture structure"""
    title: str
    style: LectureStyle
    emphasis: List[ContentEmphasis]
    estimated_duration: float
    main_themes: List[str]
    suggested_sections: List[Dict[str, Any]]
    opening_hook: str
    closing_message: str
    key_takeaways: List[str]
    audience_level: str  # beginner, intermediate, advanced
    special_instructions: Dict[str, Any]


@dataclass
class LectureComponent:
    """Individual component of a lecture"""
    component_type: str  # introduction, theme, example, transition, etc.
    content: str
    order: int
    duration_estimate: float
    metadata: Dict[str, Any]


class DynamicLectureFormatter:
    """Creates dynamic, AI-driven lecture formats"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key) if api_key else None
        
    def analyze_document_for_lecture(
        self,
        document_summary: Dict[str, Any],
        section_summaries: Dict[str, str],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        community_reports: List[str],
        target_duration: Optional[float] = None
    ) -> LectureBlueprint:
        """
        Analyze document content and generate optimal lecture blueprint.
        """
        
        # Prepare analysis prompt
        analysis_prompt = self._create_analysis_prompt(
            document_summary,
            section_summaries,
            entities,
            relationships,
            community_reports,
            target_duration
        )
        
        try:
            # Get AI analysis
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert educational content designer. 
                        Analyze the document and create an optimal lecture structure.
                        Consider the content type, complexity, and best pedagogical approach.
                        Return a detailed JSON blueprint for the lecture."""
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            blueprint_data = json.loads(response.choices[0].message.content)
            
            # Parse into blueprint
            blueprint = LectureBlueprint(
                title=blueprint_data.get("title", document_summary.get("display_name", "Untitled")),
                style=LectureStyle(blueprint_data.get("style", "academic")),
                emphasis=[ContentEmphasis(e) for e in blueprint_data.get("emphasis", ["concepts"])],
                estimated_duration=blueprint_data.get("estimated_duration", 10.0),
                main_themes=blueprint_data.get("main_themes", []),
                suggested_sections=blueprint_data.get("suggested_sections", []),
                opening_hook=blueprint_data.get("opening_hook", ""),
                closing_message=blueprint_data.get("closing_message", ""),
                key_takeaways=blueprint_data.get("key_takeaways", []),
                audience_level=blueprint_data.get("audience_level", "intermediate"),
                special_instructions=blueprint_data.get("special_instructions", {})
            )
            
            return blueprint
            
        except Exception as e:
            logger.error(f"Failed to generate lecture blueprint: {e}")
            # Return default blueprint
            return self._create_default_blueprint(document_summary, section_summaries)
    
    def generate_lecture_components(
        self,
        blueprint: LectureBlueprint,
        document_data: Dict[str, Any]
    ) -> List[LectureComponent]:
        """
        Generate individual lecture components based on blueprint.
        """
        components = []
        order = 0
        
        # 1. Opening/Hook
        if blueprint.opening_hook:
            components.append(LectureComponent(
                component_type="opening_hook",
                content=self._enhance_content(blueprint.opening_hook, blueprint.style),
                order=order,
                duration_estimate=0.5,
                metadata={"style": blueprint.style.value}
            ))
            order += 1
        
        # 2. Introduction
        intro_content = self._generate_introduction(
            blueprint,
            document_data.get("document_summary", {})
        )
        components.append(LectureComponent(
            component_type="introduction",
            content=intro_content,
            order=order,
            duration_estimate=1.0,
            metadata={"themes": blueprint.main_themes}
        ))
        order += 1
        
        # 3. Main sections based on blueprint
        for section_config in blueprint.suggested_sections:
            section_components = self._generate_section_components(
                section_config,
                document_data,
                blueprint,
                order
            )
            components.extend(section_components)
            order += len(section_components)
        
        # 4. Key takeaways
        if blueprint.key_takeaways:
            takeaway_content = self._generate_takeaways_section(
                blueprint.key_takeaways,
                blueprint.style
            )
            components.append(LectureComponent(
                component_type="key_takeaways",
                content=takeaway_content,
                order=order,
                duration_estimate=1.0,
                metadata={"count": len(blueprint.key_takeaways)}
            ))
            order += 1
        
        # 5. Closing
        closing_content = self._generate_closing(
            blueprint,
            document_data.get("document_summary", {})
        )
        components.append(LectureComponent(
            component_type="closing",
            content=closing_content,
            order=order,
            duration_estimate=0.5,
            metadata={"style": blueprint.style.value}
        ))
        
        return components
    
    def _create_analysis_prompt(
        self,
        document_summary: Dict[str, Any],
        section_summaries: Dict[str, str],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        community_reports: List[str],
        target_duration: Optional[float]
    ) -> str:
        """Create prompt for AI analysis"""
        
        prompt = f"""Analyze this document and create an optimal lecture structure.

Document: {document_summary.get('display_name', 'Unknown')}
Summary: {document_summary.get('summary', 'No summary available')}

Sections ({len(section_summaries)}):
{self._summarize_sections(section_summaries)}

Key Entities ({len(entities)}):
{self._summarize_entities(entities[:10])}

Key Relationships ({len(relationships)}):
{self._summarize_relationships(relationships[:10])}

Community Insights:
{self._summarize_reports(community_reports[:3])}

Target Duration: {target_duration if target_duration else 'Flexible (5-15 minutes)'}

Create a JSON blueprint with:
1. title: Engaging lecture title
2. style: One of [narrative, academic, conversational, tutorial, analytical, summary]
3. emphasis: Array of focus areas [concepts, relationships, examples, insights, technical, practical]
4. estimated_duration: Realistic duration in minutes
5. main_themes: 3-5 main themes to cover
6. suggested_sections: Array of section objects with:
   - name: Section name
   - type: Section type (theme_exploration, example_walkthrough, concept_explanation, etc.)
   - content_focus: What to emphasize
   - duration: Estimated minutes
   - source_material: Which parts of document to use
7. opening_hook: Engaging opening statement/question
8. closing_message: Memorable closing
9. key_takeaways: 3-5 main takeaways
10. audience_level: beginner/intermediate/advanced
11. special_instructions: Any special formatting or emphasis instructions

Consider the document type, complexity, and create an engaging educational experience."""
        
        return prompt
    
    def _generate_section_components(
        self,
        section_config: Dict[str, Any],
        document_data: Dict[str, Any],
        blueprint: LectureBlueprint,
        start_order: int
    ) -> List[LectureComponent]:
        """Generate components for a section"""
        components = []
        
        # Section introduction
        section_intro = self._create_section_intro(
            section_config,
            blueprint.style
        )
        components.append(LectureComponent(
            component_type="section_intro",
            content=section_intro,
            order=start_order,
            duration_estimate=0.3,
            metadata={"section": section_config.get("name", "")}
        ))
        
        # Main content based on section type
        if section_config.get("type") == "theme_exploration":
            content = self._explore_theme(
                section_config,
                document_data,
                blueprint
            )
        elif section_config.get("type") == "example_walkthrough":
            content = self._create_example_walkthrough(
                section_config,
                document_data,
                blueprint
            )
        elif section_config.get("type") == "concept_explanation":
            content = self._explain_concept(
                section_config,
                document_data,
                blueprint
            )
        else:
            content = self._create_standard_section(
                section_config,
                document_data,
                blueprint
            )
        
        components.append(LectureComponent(
            component_type="section_content",
            content=content,
            order=start_order + 1,
            duration_estimate=section_config.get("duration", 2.0),
            metadata={
                "section": section_config.get("name", ""),
                "type": section_config.get("type", "standard")
            }
        ))
        
        # Add transition if not last section
        if section_config.get("add_transition", True):
            transition = self._create_transition(
                section_config.get("name", ""),
                blueprint.style
            )
            components.append(LectureComponent(
                component_type="transition",
                content=transition,
                order=start_order + 2,
                duration_estimate=0.1,
                metadata={}
            ))
        
        return components
    
    def compile_lecture_script(
        self,
        components: List[LectureComponent],
        blueprint: LectureBlueprint
    ) -> str:
        """Compile components into final lecture script"""
        
        # Sort by order
        sorted_components = sorted(components, key=lambda x: x.order)
        
        # Compile based on style
        if blueprint.style == LectureStyle.NARRATIVE:
            return self._compile_narrative_style(sorted_components)
        elif blueprint.style == LectureStyle.CONVERSATIONAL:
            return self._compile_conversational_style(sorted_components)
        elif blueprint.style == LectureStyle.TUTORIAL:
            return self._compile_tutorial_style(sorted_components)
        else:
            return self._compile_standard_style(sorted_components)
    
    def _compile_narrative_style(self, components: List[LectureComponent]) -> str:
        """Compile components in narrative style"""
        script_parts = []
        
        for component in components:
            if component.component_type == "transition":
                # Smooth narrative transitions
                script_parts.append(f"\n{component.content}\n")
            else:
                script_parts.append(component.content + " ")
        
        return " ".join(script_parts).strip()
    
    def _compile_conversational_style(self, components: List[LectureComponent]) -> str:
        """Compile components in conversational style"""
        script_parts = []
        
        for component in components:
            if component.component_type == "opening_hook":
                script_parts.append(f"{component.content}\n\n")
            elif component.component_type == "section_intro":
                script_parts.append(f"\nAlright, {component.content}\n")
            elif component.component_type == "key_takeaways":
                script_parts.append(f"\nSo, what should you remember? {component.content}\n")
            else:
                script_parts.append(component.content + " ")
        
        return " ".join(script_parts).strip()
    
    def _compile_tutorial_style(self, components: List[LectureComponent]) -> str:
        """Compile components in tutorial style"""
        script_parts = []
        step_number = 1
        
        for component in components:
            if component.component_type == "section_content" and component.metadata.get("type") != "introduction":
                script_parts.append(f"\nStep {step_number}: {component.content}\n")
                step_number += 1
            else:
                script_parts.append(component.content + " ")
        
        return " ".join(script_parts).strip()
    
    def _compile_standard_style(self, components: List[LectureComponent]) -> str:
        """Compile components in standard academic style"""
        script_parts = []
        
        for component in components:
            script_parts.append(component.content)
            if component.component_type in ["section_content", "key_takeaways"]:
                script_parts.append("\n\n")
            else:
                script_parts.append(" ")
        
        return " ".join(script_parts).strip()
    
    # Helper methods
    def _create_default_blueprint(
        self,
        document_summary: Dict[str, Any],
        section_summaries: Dict[str, str]
    ) -> LectureBlueprint:
        """Create a default blueprint when AI analysis fails"""
        
        sections = []
        for i, (section_name, _) in enumerate(section_summaries.items()):
            sections.append({
                "name": section_name,
                "type": "standard",
                "content_focus": "summary",
                "duration": 2.0,
                "source_material": section_name
            })
        
        return LectureBlueprint(
            title=f"Understanding {document_summary.get('display_name', 'This Document')}",
            style=LectureStyle.ACADEMIC,
            emphasis=[ContentEmphasis.CONCEPTS],
            estimated_duration=len(sections) * 2 + 3,
            main_themes=[s.split(':')[0] for s in list(section_summaries.keys())[:3]],
            suggested_sections=sections,
            opening_hook="Let's explore the key concepts in this document.",
            closing_message="We've covered the main points. Apply these concepts in your work.",
            key_takeaways=["Understand the main concepts", "Recognize key relationships", "Apply the knowledge"],
            audience_level="intermediate",
            special_instructions={}
        )
    
    def _enhance_content(self, content: str, style: LectureStyle) -> str:
        """Enhance content based on style"""
        if style == LectureStyle.CONVERSATIONAL:
            return content.replace("We will", "Let's").replace("This document", "What we're looking at")
        elif style == LectureStyle.NARRATIVE:
            return f"Our journey begins with a question: {content}"
        return content
    
    def _generate_introduction(
        self,
        blueprint: LectureBlueprint,
        document_summary: Dict[str, Any]
    ) -> str:
        """Generate introduction based on blueprint"""
        
        title = document_summary.get('display_name', 'this document')
        summary = document_summary.get('summary', '')
        
        if blueprint.style == LectureStyle.NARRATIVE:
            intro = f"Welcome to our exploration of {title}. "
            intro += f"Today's journey will take us through {len(blueprint.main_themes)} key themes: "
            intro += ", ".join(blueprint.main_themes) + ". "
            intro += summary
        elif blueprint.style == LectureStyle.CONVERSATIONAL:
            intro = f"Hey there! Today we're diving into {title}. "
            intro += f"We'll chat about {', '.join(blueprint.main_themes[:2])}, and more. "
            intro += f"Here's the gist: {summary}"
        elif blueprint.style == LectureStyle.TUTORIAL:
            intro = f"In this tutorial on {title}, you'll learn: "
            intro += "; ".join([f"How to understand {theme}" for theme in blueprint.main_themes[:3]]) + ". "
            intro += f"Let's start with the basics: {summary}"
        else:
            intro = f"This lecture examines {title}. "
            intro += f"We will cover {len(blueprint.main_themes)} main topics: "
            intro += ", ".join(blueprint.main_themes) + ". "
            intro += summary
        
        return intro
    
    def _generate_takeaways_section(
        self,
        takeaways: List[str],
        style: LectureStyle
    ) -> str:
        """Generate key takeaways section"""
        
        if style == LectureStyle.CONVERSATIONAL:
            content = "So, what are the big takeaways here? "
            for i, takeaway in enumerate(takeaways, 1):
                content += f"Number {i}: {takeaway}. "
        elif style == LectureStyle.TUTORIAL:
            content = "Key points to remember: "
            for takeaway in takeaways:
                content += f"✓ {takeaway}. "
        else:
            content = "In summary, the key takeaways are: "
            for i, takeaway in enumerate(takeaways, 1):
                content += f"First, {takeaway}. " if i == 1 else f"Additionally, {takeaway}. "
        
        return content
    
    def _generate_closing(
        self,
        blueprint: LectureBlueprint,
        document_summary: Dict[str, Any]
    ) -> str:
        """Generate closing based on blueprint"""
        
        if blueprint.closing_message:
            return blueprint.closing_message
        
        title = document_summary.get('display_name', 'this topic')
        
        if blueprint.style == LectureStyle.NARRATIVE:
            return f"Our exploration of {title} comes to an end, but your journey with these concepts has just begun."
        elif blueprint.style == LectureStyle.CONVERSATIONAL:
            return f"And that's a wrap on {title}! Hope this was helpful. Go forth and apply what you've learned!"
        elif blueprint.style == LectureStyle.TUTORIAL:
            return f"You've completed this tutorial on {title}. Practice these concepts to master them."
        else:
            return f"This concludes our examination of {title}. Thank you for your attention."
    
    def _summarize_sections(self, sections: Dict[str, str]) -> str:
        """Summarize sections for prompt"""
        summary = []
        for name, content in list(sections.items())[:5]:
            summary.append(f"- {name}: {content[:100]}...")
        return "\n".join(summary)
    
    def _summarize_entities(self, entities: List[Dict[str, Any]]) -> str:
        """Summarize entities for prompt"""
        if not entities:
            return "No entities available"
        summary = []
        for entity in entities[:5]:
            summary.append(f"- {entity.get('title', '')}: {entity.get('type', '')}")
        return "\n".join(summary)
    
    def _summarize_relationships(self, relationships: List[Dict[str, Any]]) -> str:
        """Summarize relationships for prompt"""
        if not relationships:
            return "No relationships available"
        summary = []
        for rel in relationships[:5]:
            summary.append(f"- {rel.get('source', '')} → {rel.get('target', '')}")
        return "\n".join(summary)
    
    def _summarize_reports(self, reports: List[str]) -> str:
        """Summarize community reports for prompt"""
        if not reports:
            return "No community insights available"
        return "\n".join([f"- {report[:150]}..." for report in reports])
    
    def _create_section_intro(self, section_config: Dict[str, Any], style: LectureStyle) -> str:
        """Create section introduction"""
        name = section_config.get("name", "next topic")
        
        if style == LectureStyle.CONVERSATIONAL:
            return f"let's talk about {name}"
        elif style == LectureStyle.NARRATIVE:
            return f"Our next chapter explores {name}"
        elif style == LectureStyle.TUTORIAL:
            return f"Now we'll learn about {name}"
        else:
            return f"We now turn to {name}"
    
    def _explore_theme(
        self,
        section_config: Dict[str, Any],
        document_data: Dict[str, Any],
        blueprint: LectureBlueprint
    ) -> str:
        """Create theme exploration content"""
        # Implementation would pull from relevant document data
        theme = section_config.get("name", "this theme")
        focus = section_config.get("content_focus", "concepts")
        
        content = f"Exploring {theme}, we focus on {focus}. "
        
        # Add relevant content from document data
        if focus == "relationships" and document_data.get("relationships"):
            content += "Key connections include: "
            for rel in document_data["relationships"][:3]:
                content += f"{rel.get('source', '')} relates to {rel.get('target', '')}. "
        
        return content
    
    def _create_example_walkthrough(
        self,
        section_config: Dict[str, Any],
        document_data: Dict[str, Any],
        blueprint: LectureBlueprint
    ) -> str:
        """Create example walkthrough content"""
        return f"Let's walk through a practical example of {section_config.get('name', 'this concept')}. "
    
    def _explain_concept(
        self,
        section_config: Dict[str, Any],
        document_data: Dict[str, Any],
        blueprint: LectureBlueprint
    ) -> str:
        """Create concept explanation content"""
        concept = section_config.get("name", "this concept")
        return f"To understand {concept}, we need to consider its fundamental aspects. "
    
    def _create_standard_section(
        self,
        section_config: Dict[str, Any],
        document_data: Dict[str, Any],
        blueprint: LectureBlueprint
    ) -> str:
        """Create standard section content"""
        # Pull from section summaries if available
        section_summaries = document_data.get("section_summaries", {})
        source = section_config.get("source_material", "")
        
        if source in section_summaries:
            return section_summaries[source]
        
        return f"In this section, we examine {section_config.get('name', 'this topic')}. "
    
    def _create_transition(self, from_section: str, style: LectureStyle) -> str:
        """Create transition between sections"""
        if style == LectureStyle.CONVERSATIONAL:
            return "Moving on,"
        elif style == LectureStyle.NARRATIVE:
            return "As we continue our journey,"
        elif style == LectureStyle.TUTORIAL:
            return "Next,"
        else:
            return "Furthermore,"