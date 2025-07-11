"""
Document Structure Parser
Extracts hierarchical structure (headings, sections) from documents
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a document section with its hierarchy"""
    level: int
    title: str
    start_pos: int
    end_pos: Optional[int] = None
    parent_path: List[str] = None
    content: str = ""
    
    def __post_init__(self):
        if self.parent_path is None:
            self.parent_path = []
    
    @property
    def full_path(self) -> List[str]:
        """Get the full hierarchical path including this section"""
        return self.parent_path + [self.title]
    
    @property
    def metadata(self) -> Dict[str, any]:
        """Get section metadata for embedding"""
        return {
            "section_title": self.title,
            "section_level": self.level,
            "section_path": self.full_path,
            "parent_sections": self.parent_path,
        }


class DocumentStructureParser:
    """Parses document structure and extracts hierarchical sections"""
    
    # Common heading patterns
    HEADING_PATTERNS = [
        # Numbered sections (1., 1.1, 1.1.1, etc.)
        (r'^(\d+(?:\.\d+)*)\s+(.+)$', 'numbered'),
        # ALL CAPS headings
        (r'^([A-Z][A-Z\s]+)$', 'caps'),
        # Markdown-style headings
        (r'^(#{1,6})\s+(.+)$', 'markdown'),
        # Underlined headings (text followed by === or ---)
        (r'^(.+)\n(={3,}|-{3,})$', 'underlined'),
        # Roman numerals
        (r'^([IVXLCDM]+(?:\.[IVXLCDM]+)*)\s+(.+)$', 'roman'),
        # Letter sections (A., B., A.1., etc.)
        (r'^([A-Z](?:\.\d+)*)\s+(.+)$', 'letter'),
    ]
    
    def __init__(self):
        self.compiled_patterns = [
            (re.compile(pattern, re.MULTILINE), pattern_type) 
            for pattern, pattern_type in self.HEADING_PATTERNS
        ]
    
    def parse_document(self, text: str) -> Tuple[List[Section], str]:
        """
        Parse document structure and return sections with their content
        
        Args:
            text: The document text to parse
            
        Returns:
            Tuple of (sections list, structured text with section markers)
        """
        sections = self._extract_sections(text)
        sections = self._build_hierarchy(sections)
        sections = self._extract_section_content(sections, text)
        
        # Create structured text with section markers
        structured_text = self._create_structured_text(sections, text)
        
        return sections, structured_text
    
    def _extract_sections(self, text: str) -> List[Section]:
        """Extract all potential section headings from text"""
        sections = []
        
        # Split text into lines for line-by-line analysis
        lines = text.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                current_pos += len(line) + 1
                continue
            
            # Check each heading pattern
            for pattern, pattern_type in self.compiled_patterns:
                match = pattern.match(line)
                if match:
                    section = self._create_section_from_match(
                        match, pattern_type, current_pos, i, lines
                    )
                    if section:
                        sections.append(section)
                        break
            
            current_pos += len(line) + 1
        
        return sections
    
    def _create_section_from_match(self, match, pattern_type: str, 
                                   pos: int, line_idx: int, lines: List[str]) -> Optional[Section]:
        """Create a Section object from a regex match"""
        if pattern_type == 'numbered':
            level = len(match.group(1).split('.'))
            title = match.group(2).strip()
        elif pattern_type == 'caps':
            # Check if it's really a heading (not just a random caps line)
            if len(match.group(0)) < 3 or len(match.group(0)) > 100:
                return None
            level = 1  # Top-level for all caps
            title = match.group(0).strip()
        elif pattern_type == 'markdown':
            level = len(match.group(1))
            title = match.group(2).strip()
        elif pattern_type == 'underlined':
            title = match.group(1).strip()
            underline = match.group(2)
            level = 1 if '=' in underline else 2
            # Skip if we're looking at the underline itself
            if line_idx > 0 and lines[line_idx-1].strip():
                return None
        elif pattern_type == 'roman':
            level = len(match.group(1).split('.'))
            title = match.group(2).strip()
        elif pattern_type == 'letter':
            level = len(match.group(1).split('.'))
            title = match.group(2).strip()
        else:
            return None
        
        return Section(level=level, title=title, start_pos=pos)
    
    def _build_hierarchy(self, sections: List[Section]) -> List[Section]:
        """Build hierarchical relationships between sections"""
        if not sections:
            return sections
        
        # Sort sections by position
        sections.sort(key=lambda s: s.start_pos)
        
        # Build hierarchy
        for i, section in enumerate(sections):
            # Find parent sections (previous sections with lower level)
            parent_path = []
            for j in range(i - 1, -1, -1):
                if sections[j].level < section.level:
                    # This is a parent at this level
                    parent_path = sections[j].full_path
                    break
            
            section.parent_path = parent_path
        
        return sections
    
    def _extract_section_content(self, sections: List[Section], text: str) -> List[Section]:
        """Extract content for each section"""
        if not sections:
            return sections
        
        # Set end positions
        for i in range(len(sections) - 1):
            sections[i].end_pos = sections[i + 1].start_pos
        sections[-1].end_pos = len(text)
        
        # Extract content
        for section in sections:
            section.content = text[section.start_pos:section.end_pos].strip()
        
        return sections
    
    def _create_structured_text(self, sections: List[Section], original_text: str) -> str:
        """Create text with section markers for better chunk context"""
        if not sections:
            return original_text
        
        # Build structured text with section markers
        structured_parts = []
        last_pos = 0
        
        for section in sections:
            # Add any text before this section
            if section.start_pos > last_pos:
                structured_parts.append(original_text[last_pos:section.start_pos])
            
            # Add section marker
            marker = f"\n[SECTION: {' > '.join(section.full_path)}]\n"
            structured_parts.append(marker)
            
            # Add section content
            structured_parts.append(section.content)
            
            last_pos = section.end_pos
        
        # Add any remaining text
        if last_pos < len(original_text):
            structured_parts.append(original_text[last_pos:])
        
        return ''.join(structured_parts)
    
    def get_section_for_position(self, sections: List[Section], position: int) -> Optional[Section]:
        """Find which section contains a given text position"""
        for section in sections:
            if section.start_pos <= position < section.end_pos:
                return section
        return None


def extract_document_structure(text: str) -> Tuple[List[Section], str, List[Dict]]:
    """
    Main function to extract document structure
    
    Returns:
        - List of Section objects
        - Structured text with section markers
        - List of section metadata dictionaries
    """
    parser = DocumentStructureParser()
    sections, structured_text = parser.parse_document(text)
    
    # Extract metadata for all sections
    metadata_list = [section.metadata for section in sections]
    
    logger.info(f"Extracted {len(sections)} sections from document")
    
    return sections, structured_text, metadata_list