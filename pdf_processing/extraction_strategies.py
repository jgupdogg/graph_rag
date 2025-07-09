#!/usr/bin/env python3
"""
PDF Extraction Strategies
Different approaches for extracting content from various PDF types
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class PDFType(Enum):
    """PDF document types"""
    TECHNICAL = "technical"
    LEGAL = "legal"
    ACADEMIC = "academic"
    GENERAL = "general"
    SCANNED = "scanned"


class ExtractionQuality(Enum):
    """Quality levels for extraction methods"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    FAILED = "failed"


@dataclass
class ExtractionResult:
    """Result from an extraction strategy"""
    text: str
    tables: List[Dict]
    structure: Dict
    metadata: Dict
    quality: ExtractionQuality
    method: str
    confidence: float


class ExtractionStrategy(ABC):
    """Base class for extraction strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def can_handle(self, pdf_info: Dict) -> bool:
        """Check if this strategy can handle the PDF"""
        pass
    
    @abstractmethod
    def extract(self, pdf_path: str, page_range: Tuple[int, int]) -> ExtractionResult:
        """Extract content from PDF pages"""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """Get strategy priority (higher = preferred)"""
        pass


class TechnicalDocumentStrategy(ExtractionStrategy):
    """
    Optimized for technical documents with specifications, tables, and references
    """
    
    def __init__(self):
        super().__init__()
        self.technical_patterns = [
            r'specification',
            r'requirements?',
            r'standard',
            r'section\\s+\\d+',
            r'\\d+\\.\\d+\\s+[A-Z]',
            r'astm|iso|ansi|aashto',
            r'concrete|steel|asphalt',
            r'psi|mpa|kg/m',
        ]
    
    def can_handle(self, pdf_info: Dict) -> bool:
        """Check if PDF appears to be a technical document"""
        title = pdf_info.get('title', '').lower()
        subject = pdf_info.get('subject', '').lower()
        
        # Check for technical keywords
        technical_score = 0
        for pattern in self.technical_patterns:
            if re.search(pattern, title + ' ' + subject, re.IGNORECASE):
                technical_score += 1
        
        return technical_score >= 2
    
    def extract(self, pdf_path: str, page_range: Tuple[int, int]) -> ExtractionResult:
        """Extract content optimized for technical documents"""
        try:
            # Import here to avoid dependency issues
            import fitz
            
            text_parts = []
            tables = []
            structure = {'sections': [], 'specifications': [], 'references': []}
            
            with fitz.open(pdf_path) as doc:
                for page_num in range(page_range[0], min(page_range[1], doc.page_count)):
                    page = doc[page_num]
                    page_text = page.get_text()
                    
                    if page_text.strip():
                        # Process technical content
                        processed_text = self._process_technical_content(page_text, page_num + 1)
                        text_parts.append(processed_text)
                        
                        # Extract specifications
                        specs = self._extract_specifications(page_text, page_num + 1)
                        structure['specifications'].extend(specs)
                        
                        # Extract references
                        refs = self._extract_references(page_text, page_num + 1)
                        structure['references'].extend(refs)
                        
                        # Extract sections
                        sections = self._extract_sections(page_text, page_num + 1)
                        structure['sections'].extend(sections)
            
            final_text = '\\n'.join(text_parts)
            confidence = self._calculate_confidence(final_text, structure)
            
            return ExtractionResult(
                text=final_text,
                tables=tables,
                structure=structure,
                metadata={'strategy': 'technical', 'patterns_found': len(structure['specifications'])},
                quality=ExtractionQuality.HIGH if confidence > 0.8 else ExtractionQuality.MEDIUM,
                method='technical_strategy',
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Technical extraction failed: {e}")
            return ExtractionResult(
                text="", tables=[], structure={}, metadata={},
                quality=ExtractionQuality.FAILED, method='technical_strategy', confidence=0.0
            )
    
    def _process_technical_content(self, text: str, page_num: int) -> str:
        """Process text for technical documents"""
        lines = text.split('\\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Mark different types of content
            if self._is_section_header(line):
                processed_lines.append(f"[SECTION: {line}]")
            elif self._is_specification(line):
                processed_lines.append(f"[SPEC: {line}]")
            elif self._is_reference(line):
                processed_lines.append(f"[REF: {line}]")
            else:
                processed_lines.append(line)
        
        return f"[PAGE: {page_num}]\\n" + "\\n".join(processed_lines) + "\\n\\n"
    
    def _is_section_header(self, text: str) -> bool:
        """Check if text is a section header"""
        patterns = [
            r'^\\d+\\.\\d*\\s+[A-Z]',
            r'^SECTION\\s+\\d+',
            r'^[A-Z][A-Z\\s]{5,}$',
            r'^\\d+\\s+[A-Z]'
        ]
        return any(re.match(pattern, text) for pattern in patterns)
    
    def _is_specification(self, text: str) -> bool:
        """Check if text contains specifications"""
        patterns = [
            r'shall\\s+(be|have|comply|conform)',
            r'minimum\\s+\\d+',
            r'maximum\\s+\\d+',
            r'\\d+\\s*(psi|mpa|kg|lb)',
            r'type\\s+[A-Z]\\s+(concrete|steel|asphalt)',
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def _is_reference(self, text: str) -> bool:
        """Check if text contains references"""
        patterns = [
            r'(astm|iso|ansi|aashto)\\s*[A-Z]?\\d+',
            r'see\\s+section\\s+\\d+',
            r'per\\s+section\\s+\\d+',
            r'in\\s+accordance\\s+with',
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def _extract_specifications(self, text: str, page_num: int) -> List[Dict]:
        """Extract specifications from text"""
        specs = []
        
        # Pattern for specifications
        spec_patterns = [
            r'(minimum|maximum)\\s+(\\d+(?:\\.\\d+)?)\\s*(psi|mpa|kg|lb|inch|mm)',
            r'type\\s+([A-Z])\\s+(concrete|steel|asphalt)',
            r'(\\w+)\\s+shall\\s+(be|have|comply|conform)\\s+([^.]+)',
        ]
        
        for pattern in spec_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                specs.append({
                    'page': page_num,
                    'text': match.group(0),
                    'type': 'specification',
                    'groups': match.groups()
                })
        
        return specs
    
    def _extract_references(self, text: str, page_num: int) -> List[Dict]:
        """Extract references from text"""
        refs = []
        
        # Standard references
        std_pattern = r'(astm|iso|ansi|aashto)\\s*([A-Z]?\\d+(?:[.-]\\d+)?)'
        matches = re.finditer(std_pattern, text, re.IGNORECASE)
        
        for match in matches:
            refs.append({
                'page': page_num,
                'standard': match.group(1).upper(),
                'number': match.group(2),
                'full_text': match.group(0),
                'type': 'standard_reference'
            })
        
        # Cross-references
        xref_patterns = [
            r'see\\s+section\\s+(\\d+(?:\\.\\d+)?)',
            r'per\\s+section\\s+(\\d+(?:\\.\\d+)?)',
            r'section\\s+(\\d+(?:\\.\\d+)?)\\s+(?:above|below)',
        ]
        
        for pattern in xref_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                refs.append({
                    'page': page_num,
                    'section': match.group(1),
                    'full_text': match.group(0),
                    'type': 'cross_reference'
                })
        
        return refs
    
    def _extract_sections(self, text: str, page_num: int) -> List[Dict]:
        """Extract section headers"""
        sections = []
        lines = text.split('\\n')
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if self._is_section_header(line):
                # Try to determine section level
                level = self._get_section_level(line)
                sections.append({
                    'page': page_num,
                    'line': line_num,
                    'text': line,
                    'level': level,
                    'type': 'section_header'
                })
        
        return sections
    
    def _get_section_level(self, text: str) -> int:
        """Determine section header level"""
        if re.match(r'^\\d+\\s+[A-Z]', text):
            return 1
        elif re.match(r'^\\d+\\.\\d+\\s+[A-Z]', text):
            return 2
        elif re.match(r'^\\d+\\.\\d+\\.\\d+\\s+[A-Z]', text):
            return 3
        elif re.match(r'^[A-Z][A-Z\\s]{5,}$', text):
            return 1
        else:
            return 2
    
    def _calculate_confidence(self, text: str, structure: Dict) -> float:
        """Calculate extraction confidence based on found patterns"""
        if not text:
            return 0.0
        
        confidence_factors = []
        
        # Check for technical patterns
        pattern_score = sum(
            1 for pattern in self.technical_patterns
            if re.search(pattern, text, re.IGNORECASE)
        ) / len(self.technical_patterns)
        confidence_factors.append(pattern_score)
        
        # Check for extracted structures
        specs_score = min(len(structure.get('specifications', [])) / 10, 1.0)
        confidence_factors.append(specs_score)
        
        refs_score = min(len(structure.get('references', [])) / 5, 1.0)
        confidence_factors.append(refs_score)
        
        sections_score = min(len(structure.get('sections', [])) / 3, 1.0)
        confidence_factors.append(sections_score)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def get_priority(self) -> int:
        return 90  # High priority for technical documents


class GeneralDocumentStrategy(ExtractionStrategy):
    """
    General-purpose extraction for standard documents
    """
    
    def can_handle(self, pdf_info: Dict) -> bool:
        """Always can handle general documents"""
        return True
    
    def extract(self, pdf_path: str, page_range: Tuple[int, int]) -> ExtractionResult:
        """Extract content using general approach"""
        try:
            import fitz
            
            text_parts = []
            structure = {'pages': []}
            
            with fitz.open(pdf_path) as doc:
                for page_num in range(page_range[0], min(page_range[1], doc.page_count)):
                    page = doc[page_num]
                    page_text = page.get_text()
                    
                    if page_text.strip():
                        text_parts.append(f"[PAGE: {page_num + 1}]\\n{page_text}\\n\\n")
                        structure['pages'].append({
                            'page': page_num + 1,
                            'char_count': len(page_text)
                        })
            
            final_text = ''.join(text_parts)
            confidence = 0.7 if final_text.strip() else 0.0
            
            return ExtractionResult(
                text=final_text,
                tables=[],
                structure=structure,
                metadata={'strategy': 'general'},
                quality=ExtractionQuality.MEDIUM if confidence > 0.5 else ExtractionQuality.LOW,
                method='general_strategy',
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"General extraction failed: {e}")
            return ExtractionResult(
                text="", tables=[], structure={}, metadata={},
                quality=ExtractionQuality.FAILED, method='general_strategy', confidence=0.0
            )
    
    def get_priority(self) -> int:
        return 10  # Low priority, fallback option


class TableFocusedStrategy(ExtractionStrategy):
    """
    Strategy focused on documents with many tables
    """
    
    def can_handle(self, pdf_info: Dict) -> bool:
        """Check if document likely contains many tables"""
        # This would require analyzing the PDF structure
        # For now, return False to avoid conflicts
        return False
    
    def extract(self, pdf_path: str, page_range: Tuple[int, int]) -> ExtractionResult:
        """Extract content with focus on tables"""
        try:
            import pdfplumber
            
            text_parts = []
            tables = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num in range(page_range[0], min(page_range[1], len(pdf.pages))):
                    page = pdf.pages[page_num]
                    
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"[PAGE: {page_num + 1}]\\n{page_text}\\n")
                    
                    # Extract tables with high precision
                    page_tables = page.extract_tables()
                    for i, table in enumerate(page_tables or []):
                        if table and len(table) > 1:
                            table_text = self._format_table_detailed(table, page_num + 1, i)
                            text_parts.append(table_text)
                            tables.append({
                                'page': page_num + 1,
                                'index': i,
                                'rows': len(table),
                                'cols': len(table[0]) if table else 0,
                                'data': table
                            })
            
            final_text = '\\n'.join(text_parts)
            confidence = 0.8 if tables else 0.5
            
            return ExtractionResult(
                text=final_text,
                tables=tables,
                structure={'table_count': len(tables)},
                metadata={'strategy': 'table_focused', 'tables_found': len(tables)},
                quality=ExtractionQuality.HIGH if len(tables) > 0 else ExtractionQuality.MEDIUM,
                method='table_strategy',
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Table extraction failed: {e}")
            return ExtractionResult(
                text="", tables=[], structure={}, metadata={},
                quality=ExtractionQuality.FAILED, method='table_strategy', confidence=0.0
            )
    
    def _format_table_detailed(self, table: List[List[str]], page_num: int, table_index: int) -> str:
        """Format table with detailed structure"""
        if not table:
            return ""
        
        lines = [f"\\n[TABLE_START: Page {page_num}, Table {table_index + 1}]"]
        lines.append(f"[TABLE_DIMS: {len(table)} rows × {len(table[0]) if table else 0} columns]")
        
        # Headers
        if table[0]:
            headers = [str(cell) if cell else "" for cell in table[0]]
            lines.append(f"[HEADERS: {' | '.join(headers)}]")
            lines.append(" | ".join(headers))
            lines.append("-" * (sum(len(h) for h in headers) + len(headers) * 3))
        
        # Data rows
        for row in table[1:]:
            if any(cell for cell in row):
                cells = [str(cell) if cell else "" for cell in row]
                lines.append(" | ".join(cells))
        
        lines.append("[TABLE_END]\\n")
        return "\\n".join(lines)
    
    def get_priority(self) -> int:
        return 70  # High priority when tables are detected


class StrategySelector:
    """
    Selects the best extraction strategy for a given PDF
    """
    
    def __init__(self):
        self.strategies = [
            TechnicalDocumentStrategy(),
            TableFocusedStrategy(),
            GeneralDocumentStrategy(),
        ]
        self.logger = logging.getLogger(__name__)
    
    def select_strategy(self, pdf_info: Dict) -> ExtractionStrategy:
        """Select the best strategy for the PDF"""
        suitable_strategies = []
        
        for strategy in self.strategies:
            if strategy.can_handle(pdf_info):
                suitable_strategies.append(strategy)
        
        if not suitable_strategies:
            # Fallback to general strategy
            return GeneralDocumentStrategy()
        
        # Sort by priority and return the best one
        suitable_strategies.sort(key=lambda s: s.get_priority(), reverse=True)
        selected = suitable_strategies[0]
        
        self.logger.info(f"Selected strategy: {selected.__class__.__name__}")
        return selected
    
    def extract_with_fallback(self, pdf_path: str, page_range: Tuple[int, int], 
                            pdf_info: Dict) -> ExtractionResult:
        """
        Extract content with automatic fallback to other strategies
        """
        primary_strategy = self.select_strategy(pdf_info)
        
        # Try primary strategy
        result = primary_strategy.extract(pdf_path, page_range)
        
        if result.quality != ExtractionQuality.FAILED and result.confidence > 0.3:
            return result
        
        # Try fallback strategies
        self.logger.warning(f"Primary strategy failed, trying fallbacks")
        
        for strategy in self.strategies:
            if strategy == primary_strategy:
                continue
                
            if strategy.can_handle(pdf_info):
                self.logger.info(f"Trying fallback strategy: {strategy.__class__.__name__}")
                result = strategy.extract(pdf_path, page_range)
                
                if result.quality != ExtractionQuality.FAILED:
                    return result
        
        # Return the best result we got, even if failed
        return result


def detect_pdf_type(pdf_info: Dict, sample_text: str = "") -> PDFType:
    """
    Detect the type of PDF document
    """
    title = pdf_info.get('title', '').lower()
    subject = pdf_info.get('subject', '').lower()
    content = (title + ' ' + subject + ' ' + sample_text).lower()
    
    # Technical document patterns
    technical_keywords = ['specification', 'standard', 'engineering', 'construction', 
                         'material', 'concrete', 'steel', 'requirements']
    
    # Legal document patterns
    legal_keywords = ['contract', 'agreement', 'legal', 'law', 'statute', 'regulation']
    
    # Academic document patterns
    academic_keywords = ['research', 'study', 'analysis', 'journal', 'university', 'paper']
    
    # Count keyword matches
    technical_score = sum(1 for kw in technical_keywords if kw in content)
    legal_score = sum(1 for kw in legal_keywords if kw in content)
    academic_score = sum(1 for kw in academic_keywords if kw in content)
    
    # Determine type based on highest score
    scores = {
        PDFType.TECHNICAL: technical_score,
        PDFType.LEGAL: legal_score,
        PDFType.ACADEMIC: academic_score
    }
    
    best_type = max(scores, key=scores.get)
    
    # Check if it's likely a scanned document (would need OCR)
    if not content.strip() or len(content.strip()) < 50:
        return PDFType.SCANNED
    
    return best_type if scores[best_type] > 0 else PDFType.GENERAL