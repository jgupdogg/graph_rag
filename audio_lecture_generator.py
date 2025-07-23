"""
Audio Lecture Generator for GraphRAG Documents
Generates comprehensive audio lectures from document structures and summaries.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import re
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class LectureDetailLevel(Enum):
    """Detail levels for audio lectures"""
    OVERVIEW = 1  # High-level overview only
    MAIN_POINTS = 2  # Main sections + key points
    STANDARD = 3  # All sections + important entities
    DETAILED = 4  # Sections + entities + relationships
    COMPREHENSIVE = 5  # Everything with examples and details


@dataclass
class LectureSection:
    """Represents a section of the audio lecture"""
    title: str
    content: str
    level: int  # Hierarchy level (0 for intro, 1 for chapters, 2 for sections, etc.)
    section_type: str  # 'introduction', 'section', 'transition', 'conclusion'
    entities: List[str] = None
    relationships: List[str] = None
    key_points: List[str] = None


class AudioLectureGenerator:
    """Generates audio lecture scripts from document data"""
    
    def __init__(self):
        self.transition_phrases = [
            "Now, let's move on to",
            "Next, we'll explore",
            "Moving forward to",
            "Let's now discuss",
            "Turning our attention to",
            "The next important topic is"
        ]
        
        self.introduction_templates = [
            "Welcome to this audio lecture on {title}. {overview}",
            "In this lecture, we'll explore {title}. {overview}",
            "Today's topic is {title}. {overview}"
        ]
        
        self.conclusion_templates = [
            "In conclusion, we've covered {main_topics}. The key takeaways are: {key_points}",
            "To summarize this lecture on {title}, we've explored {main_topics}. Remember these important points: {key_points}",
            "We've now completed our exploration of {title}, covering {main_topics}. The main insights to remember are: {key_points}"
        ]
    
    def generate_lecture_script(
        self,
        document_summary: Dict[str, Any],
        section_summaries: Dict[str, str],
        detail_level: LectureDetailLevel = LectureDetailLevel.STANDARD,
        include_entities: bool = True,
        include_relationships: bool = True,
        include_bullet_points: bool = True,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[LectureSection], str]:
        """
        Generate a complete lecture script from document data.
        
        Returns:
            Tuple of (lecture_sections, full_script_text)
        """
        lecture_sections = []
        
        # 1. Create introduction
        intro_section = self._create_introduction(document_summary, section_summaries)
        lecture_sections.append(intro_section)
        
        # 2. Process each section based on detail level
        main_topics = []
        section_items = list(section_summaries.items())
        
        for i, (section_path, summary) in enumerate(section_items):
            # Determine section hierarchy level
            hierarchy_level = section_path.count('>') + 1
            
            # Skip lower-level sections if detail level is low
            if detail_level == LectureDetailLevel.OVERVIEW and hierarchy_level > 1:
                continue
            elif detail_level == LectureDetailLevel.MAIN_POINTS and hierarchy_level > 2:
                continue
            
            # Add transition if not the first section
            if i > 0 and hierarchy_level == 1:
                transition = self._create_transition(
                    section_items[i-1][0] if i > 0 else "introduction",
                    section_path
                )
                lecture_sections.append(transition)
            
            # Create section content
            section = self._create_section_content(
                section_path,
                summary,
                hierarchy_level,
                detail_level,
                additional_context
            )
            lecture_sections.append(section)
            
            # Track main topics
            if hierarchy_level == 1:
                main_topics.append(section_path.split('>')[0].strip())
        
        # 3. Add enhanced content based on detail level
        if detail_level.value >= LectureDetailLevel.STANDARD.value and additional_context:
            # Add entity information
            if include_entities and additional_context.get('entities'):
                entity_section = self._create_entity_section(additional_context['entities'])
                if entity_section:
                    lecture_sections.append(entity_section)
            
            # Add relationship information
            if include_relationships and additional_context.get('relationships'):
                relationship_section = self._create_relationship_section(additional_context['relationships'])
                if relationship_section:
                    lecture_sections.append(relationship_section)
            
            # Add bullet points
            if include_bullet_points and additional_context.get('bullet_points'):
                bullet_section = self._create_bullet_points_section(additional_context['bullet_points'])
                if bullet_section:
                    lecture_sections.append(bullet_section)
        
        # 4. Create conclusion
        conclusion = self._create_conclusion(
            document_summary,
            main_topics,
            lecture_sections,
            detail_level
        )
        lecture_sections.append(conclusion)
        
        # 5. Compile full script
        full_script = self._compile_full_script(lecture_sections)
        
        return lecture_sections, full_script
    
    def _create_introduction(
        self,
        document_summary: Dict[str, Any],
        section_summaries: Dict[str, str]
    ) -> LectureSection:
        """Create the introduction section"""
        title = document_summary.get('display_name', 'this document')
        overview = document_summary.get('summary', 'We will explore the key concepts and insights.')
        
        # Count main sections
        main_sections = [s for s in section_summaries.keys() if '>' not in s]
        
        intro_content = f"Welcome to this audio lecture on {title}. {overview} "
        intro_content += f"We'll be covering {len(main_sections)} main topics in this lecture. "
        
        # Add brief overview of main topics
        if main_sections:
            intro_content += "These include: " + ", ".join(main_sections[:5])
            if len(main_sections) > 5:
                intro_content += f", and {len(main_sections) - 5} more topics."
            else:
                intro_content += "."
        
        return LectureSection(
            title="Introduction",
            content=intro_content,
            level=0,
            section_type="introduction"
        )
    
    def _create_transition(self, from_section: str, to_section: str) -> LectureSection:
        """Create a transition between sections"""
        # Clean section names
        from_name = from_section.split('>')[-1].strip().replace(':', '')
        to_name = to_section.split('>')[0].strip().replace(':', '')
        
        # Select transition phrase
        import random
        transition_phrase = random.choice(self.transition_phrases)
        
        content = f"{transition_phrase} {to_name}."
        
        return LectureSection(
            title=f"Transition to {to_name}",
            content=content,
            level=0,
            section_type="transition"
        )
    
    def _create_section_content(
        self,
        section_path: str,
        summary: str,
        hierarchy_level: int,
        detail_level: LectureDetailLevel,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> LectureSection:
        """Create content for a document section"""
        # Clean up section title
        section_parts = section_path.split('>')
        section_title = section_parts[-1].strip().replace(':', '')
        
        # Start with section introduction
        if hierarchy_level == 1:
            content = f"Let's explore {section_title}. "
        else:
            content = f"Within this topic, we'll look at {section_title}. "
        
        # Add the summary
        content += summary
        
        # Add additional details based on detail level
        if detail_level.value >= LectureDetailLevel.DETAILED.value and additional_context:
            # Look for relevant entities in this section
            if additional_context.get('entities'):
                relevant_entities = self._find_relevant_items(
                    section_title,
                    summary,
                    additional_context['entities'],
                    'entity'
                )
                if relevant_entities:
                    content += f" Key concepts in this section include: {', '.join(relevant_entities[:3])}."
            
            # Add specific examples or numbers if comprehensive
            if detail_level == LectureDetailLevel.COMPREHENSIVE:
                # Extract any numbers or specific examples from the summary
                numbers = re.findall(r'\b\d+\.?\d*\b', summary)
                if numbers:
                    content += f" Important figures to note: {', '.join(numbers[:3])}."
        
        return LectureSection(
            title=section_title,
            content=content,
            level=hierarchy_level,
            section_type="section"
        )
    
    def _create_entity_section(self, entities: List[Dict[str, Any]]) -> Optional[LectureSection]:
        """Create a section about key entities"""
        if not entities:
            return None
        
        content = "Now, let's highlight some key concepts and entities discussed in this document. "
        
        # Group entities by type if available
        entity_groups = {}
        for entity in entities[:10]:  # Limit to top 10
            entity_type = entity.get('type', 'concept')
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(entity.get('title', ''))
        
        for entity_type, entity_list in entity_groups.items():
            content += f"Important {entity_type}s include: {', '.join(entity_list)}. "
        
        return LectureSection(
            title="Key Concepts and Entities",
            content=content,
            level=1,
            section_type="section",
            entities=[e.get('title', '') for e in entities[:10]]
        )
    
    def _create_relationship_section(self, relationships: List[Dict[str, Any]]) -> Optional[LectureSection]:
        """Create a section about key relationships"""
        if not relationships:
            return None
        
        content = "Let's examine some important relationships and connections in this document. "
        
        # Select most important relationships
        for rel in relationships[:5]:  # Limit to top 5
            source = rel.get('source', '')
            target = rel.get('target', '')
            description = rel.get('description', 'is related to')
            
            content += f"{source} {description} {target}. "
        
        return LectureSection(
            title="Key Relationships and Connections",
            content=content,
            level=1,
            section_type="section",
            relationships=[f"{r.get('source', '')} - {r.get('target', '')}" for r in relationships[:5]]
        )
    
    def _create_bullet_points_section(self, bullet_points: List[str]) -> Optional[LectureSection]:
        """Create a section for key takeaways"""
        if not bullet_points:
            return None
        
        content = "Here are the key takeaways from this document: "
        
        for i, point in enumerate(bullet_points[:7], 1):  # Limit to 7 points
            content += f"Point {i}: {point}. "
        
        return LectureSection(
            title="Key Takeaways",
            content=content,
            level=1,
            section_type="section",
            key_points=bullet_points[:7]
        )
    
    def _create_conclusion(
        self,
        document_summary: Dict[str, Any],
        main_topics: List[str],
        lecture_sections: List[LectureSection],
        detail_level: LectureDetailLevel
    ) -> LectureSection:
        """Create the conclusion section"""
        title = document_summary.get('display_name', 'this document')
        
        # Summarize main topics
        topics_summary = ", ".join(main_topics[:3])
        if len(main_topics) > 3:
            topics_summary += f" and {len(main_topics) - 3} other topics"
        
        # Extract key points from sections
        all_key_points = []
        for section in lecture_sections:
            if section.key_points:
                all_key_points.extend(section.key_points[:2])
        
        # Create conclusion content
        content = f"We've now completed our exploration of {title}. "
        content += f"We covered {topics_summary}. "
        
        if all_key_points:
            content += "The most important points to remember are: "
            content += ". ".join(all_key_points[:3]) + ". "
        
        content += "Thank you for listening to this audio lecture."
        
        return LectureSection(
            title="Conclusion",
            content=content,
            level=0,
            section_type="conclusion"
        )
    
    def _find_relevant_items(
        self,
        section_title: str,
        section_content: str,
        items: List[Dict[str, Any]],
        item_type: str
    ) -> List[str]:
        """Find items relevant to a specific section"""
        relevant = []
        
        section_text = (section_title + " " + section_content).lower()
        
        for item in items:
            item_title = item.get('title', '').lower()
            item_desc = item.get('description', '').lower()
            
            # Check if item is mentioned in section
            if item_title in section_text or any(word in section_text for word in item_title.split()):
                relevant.append(item.get('title', ''))
            elif item_desc and any(word in section_text for word in item_desc.split()[:5]):
                relevant.append(item.get('title', ''))
        
        return relevant[:5]  # Limit to 5 items per section
    
    def _compile_full_script(self, lecture_sections: List[LectureSection]) -> str:
        """Compile all sections into a full script"""
        script_parts = []
        
        for section in lecture_sections:
            # Add appropriate pauses between sections
            if section.section_type == "transition":
                script_parts.append(f"\n{section.content}\n")
            elif section.level == 0:  # Introduction/Conclusion
                script_parts.append(f"\n{section.content}\n")
            else:
                # Add section with appropriate formatting
                script_parts.append(f"{section.content} ")
        
        return "\n".join(script_parts)
    
    def estimate_lecture_duration(self, script_text: str, words_per_minute: int = 150) -> float:
        """Estimate lecture duration in minutes"""
        word_count = len(script_text.split())
        return round(word_count / words_per_minute, 1)
    
    def save_lecture_script(
        self,
        document_id: str,
        lecture_sections: List[LectureSection],
        full_script: str,
        detail_level: LectureDetailLevel,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save lecture script and metadata to workspace cache.
        
        Args:
            document_id: Unique document identifier
            lecture_sections: List of lecture sections
            full_script: Complete lecture script text
            detail_level: Detail level used for generation
            metadata: Additional metadata (voice, quality, etc.)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Find workspace directory
            workspace_base = Path("workspaces")
            workspace_dir = None
            
            # Search for workspace containing this document
            for workspace_path in workspace_base.glob("*"):
                if workspace_path.is_dir() and (workspace_path / "cache").exists():
                    # Check if this workspace has the document
                    cache_info_file = workspace_path / "cache" / "processing_info.json"
                    if cache_info_file.exists():
                        with open(cache_info_file, 'r') as f:
                            info = json.load(f)
                            if info.get('document_id') == document_id:
                                workspace_dir = workspace_path
                                break
            
            if not workspace_dir:
                # Fallback to document ID as workspace
                workspace_dir = workspace_base / document_id
            
            # Create lecture cache directory
            lecture_cache_dir = workspace_dir / "cache" / "audio_lectures"
            lecture_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lecture_{detail_level.name.lower()}_{timestamp}.json"
            filepath = lecture_cache_dir / filename
            
            # Prepare lecture data
            lecture_data = {
                "document_id": document_id,
                "generated_at": datetime.now().isoformat(),
                "detail_level": detail_level.name,
                "detail_value": detail_level.value,
                "duration_estimate": self.estimate_lecture_duration(full_script),
                "word_count": len(full_script.split()),
                "character_count": len(full_script),
                "sections": [
                    {
                        "title": section.title,
                        "content": section.content,
                        "level": section.level,
                        "section_type": section.section_type,
                        "entities": section.entities or [],
                        "relationships": section.relationships or [],
                        "key_points": section.key_points or []
                    }
                    for section in lecture_sections
                ],
                "full_script": full_script,
                "metadata": metadata or {}
            }
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(lecture_data, f, indent=2, ensure_ascii=False)
            
            # Also save a "latest" version for easy access
            latest_filepath = lecture_cache_dir / f"latest_{detail_level.name.lower()}.json"
            with open(latest_filepath, 'w', encoding='utf-8') as f:
                json.dump(lecture_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved lecture script to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save lecture script: {e}")
            return False
    
    def load_lecture_scripts(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Load all saved lecture scripts for a document.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            List of lecture script data dictionaries
        """
        try:
            # Find workspace directory
            workspace_base = Path("workspaces")
            lecture_scripts = []
            
            # Search all workspaces
            for workspace_path in workspace_base.glob("*"):
                lecture_cache_dir = workspace_path / "cache" / "audio_lectures"
                if lecture_cache_dir.exists():
                    # Load all lecture files
                    for lecture_file in lecture_cache_dir.glob("*.json"):
                        try:
                            with open(lecture_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if data.get('document_id') == document_id:
                                    data['filename'] = lecture_file.name
                                    data['filepath'] = str(lecture_file)
                                    lecture_scripts.append(data)
                        except Exception as e:
                            logger.warning(f"Failed to load lecture file {lecture_file}: {e}")
            
            # Sort by generation time (newest first)
            lecture_scripts.sort(key=lambda x: x.get('generated_at', ''), reverse=True)
            
            return lecture_scripts
            
        except Exception as e:
            logger.error(f"Failed to load lecture scripts: {e}")
            return []
    
    def get_latest_lecture_script(
        self,
        document_id: str,
        detail_level: Optional[LectureDetailLevel] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent lecture script for a document.
        
        Args:
            document_id: Unique document identifier
            detail_level: Optional specific detail level to retrieve
            
        Returns:
            Latest lecture script data or None
        """
        scripts = self.load_lecture_scripts(document_id)
        
        if detail_level:
            # Filter by detail level
            scripts = [s for s in scripts if s.get('detail_level') == detail_level.name]
        
        return scripts[0] if scripts else None