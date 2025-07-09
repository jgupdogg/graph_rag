#!/usr/bin/env python3
"""
Quality Validation for PDF Processing
Validates extraction quality and provides recommendations
"""

import re
import logging
import statistics
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import hashlib


class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"
    NORMAL = "normal"
    LENIENT = "lenient"


class IssueType(Enum):
    """Types of quality issues"""
    MISSING_TEXT = "missing_text"
    GARBLED_TEXT = "garbled_text"
    INCOMPLETE_TABLES = "incomplete_tables"
    BROKEN_STRUCTURE = "broken_structure"
    ENCODING_ISSUES = "encoding_issues"
    LAYOUT_PROBLEMS = "layout_problems"


@dataclass
class QualityIssue:
    """Represents a quality issue found during validation"""
    issue_type: IssueType
    severity: str  # "critical", "warning", "info"
    page: Optional[int]
    description: str
    suggestion: str
    confidence: float


@dataclass
class QualityMetrics:
    """Quality metrics for extracted content"""
    text_coverage: float  # Percentage of expected text extracted
    character_accuracy: float  # Accuracy of character extraction
    word_accuracy: float  # Accuracy of word extraction
    table_detection_rate: float  # Percentage of tables detected
    structure_preservation: float  # How well structure is maintained
    entity_readiness: float  # How ready the text is for entity extraction
    overall_score: float  # Overall quality score (0-1)


@dataclass
class ValidationReport:
    """Complete validation report"""
    metrics: QualityMetrics
    issues: List[QualityIssue]
    recommendations: List[str]
    processing_recommendations: Dict[str, Any]
    graphrag_readiness: str  # "ready", "needs_work", "poor"


class QualityValidator:
    """
    Validates the quality of PDF extraction results
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.NORMAL):
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
        
        # Define quality thresholds based on validation level
        self.thresholds = self._get_thresholds(validation_level)
        
        # Common issues patterns
        self.garbled_patterns = [
            r'[^\\w\\s\\-.,;:()\\[\\]{}"\\'\\n\\t]{3,}',  # Random characters
            r'\\w{50,}',  # Very long words (likely garbled)
            r'[A-Z]{10,}',  # Too many consecutive capitals
            r'\\d{20,}',  # Very long numbers
        ]
        
        self.encoding_patterns = [
            r'[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f-\\xff]',  # Control characters
            r'Ã|â|€|™|'|'|"|"',  # Common encoding issues
        ]
    
    def _get_thresholds(self, level: ValidationLevel) -> Dict[str, float]:
        """Get quality thresholds based on validation level"""
        if level == ValidationLevel.STRICT:
            return {
                'text_coverage': 0.95,
                'character_accuracy': 0.98,
                'word_accuracy': 0.95,
                'table_detection': 0.90,
                'structure_preservation': 0.85,
                'entity_readiness': 0.80,
                'overall_minimum': 0.85
            }
        elif level == ValidationLevel.NORMAL:
            return {
                'text_coverage': 0.85,
                'character_accuracy': 0.90,
                'word_accuracy': 0.85,
                'table_detection': 0.70,
                'structure_preservation': 0.70,
                'entity_readiness': 0.70,
                'overall_minimum': 0.75
            }
        else:  # LENIENT
            return {
                'text_coverage': 0.70,
                'character_accuracy': 0.80,
                'word_accuracy': 0.75,
                'table_detection': 0.50,
                'structure_preservation': 0.60,
                'entity_readiness': 0.60,
                'overall_minimum': 0.65
            }
    
    def validate_extraction(self, 
                          extracted_text: str,
                          pdf_path: str,
                          processing_metrics: Dict,
                          tables: List[Dict] = None,
                          structure: Dict = None) -> ValidationReport:
        """
        Validate the quality of PDF extraction
        
        Args:
            extracted_text: The extracted text content
            pdf_path: Path to original PDF
            processing_metrics: Metrics from the extraction process
            tables: Extracted tables
            structure: Extracted structure information
            
        Returns:
            ValidationReport with quality assessment
        """
        self.logger.info(f"Validating extraction quality for {pdf_path}")
        
        issues = []
        recommendations = []
        
        # Calculate quality metrics
        metrics = self._calculate_quality_metrics(
            extracted_text, pdf_path, processing_metrics, tables, structure
        )
        
        # Validate text quality
        text_issues = self._validate_text_quality(extracted_text)
        issues.extend(text_issues)
        
        # Validate table extraction
        if tables:
            table_issues = self._validate_table_quality(tables)
            issues.extend(table_issues)
        
        # Validate structure preservation
        if structure:
            structure_issues = self._validate_structure_quality(structure)
            issues.extend(structure_issues)
        
        # Validate GraphRAG readiness
        entity_issues = self._validate_entity_readiness(extracted_text)
        issues.extend(entity_issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, issues)
        
        # Determine processing recommendations
        processing_recs = self._generate_processing_recommendations(metrics, issues)
        
        # Determine GraphRAG readiness
        readiness = self._assess_graphrag_readiness(metrics, issues)
        
        return ValidationReport(
            metrics=metrics,
            issues=issues,
            recommendations=recommendations,
            processing_recommendations=processing_recs,
            graphrag_readiness=readiness
        )
    
    def _calculate_quality_metrics(self, 
                                 extracted_text: str,
                                 pdf_path: str,
                                 processing_metrics: Dict,
                                 tables: List[Dict],
                                 structure: Dict) -> QualityMetrics:
        """Calculate quality metrics"""
        
        # Text coverage (estimated based on file size and extracted length)
        text_coverage = self._estimate_text_coverage(extracted_text, pdf_path, processing_metrics)
        
        # Character accuracy (check for garbled text)
        character_accuracy = self._calculate_character_accuracy(extracted_text)
        
        # Word accuracy (check for broken words)
        word_accuracy = self._calculate_word_accuracy(extracted_text)
        
        # Table detection rate (estimated)
        table_detection_rate = self._estimate_table_detection_rate(tables, extracted_text)
        
        # Structure preservation
        structure_preservation = self._calculate_structure_preservation(structure, extracted_text)
        
        # Entity readiness
        entity_readiness = self._calculate_entity_readiness(extracted_text)
        
        # Overall score (weighted average)
        overall_score = self._calculate_overall_score({
            'text_coverage': text_coverage,
            'character_accuracy': character_accuracy,
            'word_accuracy': word_accuracy,
            'table_detection_rate': table_detection_rate,
            'structure_preservation': structure_preservation,
            'entity_readiness': entity_readiness
        })
        
        return QualityMetrics(
            text_coverage=text_coverage,
            character_accuracy=character_accuracy,
            word_accuracy=word_accuracy,
            table_detection_rate=table_detection_rate,
            structure_preservation=structure_preservation,
            entity_readiness=entity_readiness,
            overall_score=overall_score
        )
    
    def _estimate_text_coverage(self, text: str, pdf_path: str, metrics: Dict) -> float:
        """Estimate what percentage of text was extracted"""
        try:
            import os
            
            # Rough estimation based on file size and extracted content
            file_size = os.path.getsize(pdf_path)
            total_pages = metrics.get('total_pages', 1)
            processed_pages = metrics.get('pages_processed', 1)
            
            # Estimate expected characters per page (rough heuristic)
            # Technical documents: ~2000 chars/page, General: ~3000 chars/page
            estimated_chars_per_page = 2000  # Conservative estimate
            expected_total_chars = total_pages * estimated_chars_per_page
            
            # Account for pages processed
            expected_chars = expected_total_chars * (processed_pages / total_pages)
            
            # Calculate coverage
            actual_chars = len(text)
            coverage = min(actual_chars / expected_chars, 1.0) if expected_chars > 0 else 0.0
            
            # Adjust for very large files (PDFs might have less text density)
            if file_size > 50 * 1024 * 1024:  # > 50MB
                coverage = min(coverage * 1.2, 1.0)  # Be more lenient
            
            return coverage
            
        except Exception as e:
            self.logger.warning(f"Could not estimate text coverage: {e}")
            return 0.8  # Default to reasonable coverage
    
    def _calculate_character_accuracy(self, text: str) -> float:
        """Calculate character-level accuracy"""
        if not text:
            return 0.0
        
        total_chars = len(text)
        problematic_chars = 0
        
        # Count garbled characters
        for pattern in self.garbled_patterns:
            matches = re.findall(pattern, text)
            problematic_chars += sum(len(match) for match in matches)
        
        # Count encoding issues
        for pattern in self.encoding_patterns:
            matches = re.findall(pattern, text)
            problematic_chars += sum(len(match) for match in matches)
        
        # Calculate accuracy
        accuracy = 1.0 - (problematic_chars / total_chars)
        return max(0.0, accuracy)
    
    def _calculate_word_accuracy(self, text: str) -> float:
        """Calculate word-level accuracy"""
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        problematic_words = 0
        
        for word in words:
            word = word.strip('.,;:()[]{}"\\'')
            
            # Check for very long words (likely broken)
            if len(word) > 30:
                problematic_words += 1
                continue
            
            # Check for words with too many numbers
            if len(word) > 5 and sum(c.isdigit() for c in word) / len(word) > 0.7:
                problematic_words += 1
                continue
            
            # Check for words with random capitalization
            if len(word) > 3 and sum(c.isupper() for c in word) > len(word) * 0.8:
                problematic_words += 1
                continue
        
        accuracy = 1.0 - (problematic_words / len(words))
        return max(0.0, accuracy)
    
    def _estimate_table_detection_rate(self, tables: List[Dict], text: str) -> float:
        """Estimate table detection rate"""
        if not text:
            return 0.0
        
        # Count table indicators in text
        table_indicators = [
            r'\\[TABLE_START\\]',  # Our table markers
            r'\\|[^\\n]*\\|[^\\n]*\\|',  # Pipe-separated tables
            r'\\t[^\\n]*\\t[^\\n]*\\t',  # Tab-separated content
            r'^\\s*\\w+\\s+\\w+\\s+\\w+\\s*$',  # Three-column layout
        ]
        
        estimated_tables = 0
        for pattern in table_indicators:
            matches = re.findall(pattern, text, re.MULTILINE)
            estimated_tables += len(matches)
        
        # Avoid double counting
        estimated_tables = max(estimated_tables // 2, 1)
        
        detected_tables = len(tables) if tables else 0
        
        # Calculate detection rate
        if estimated_tables == 0:
            return 1.0 if detected_tables == 0 else 0.5
        
        detection_rate = min(detected_tables / estimated_tables, 1.0)
        return detection_rate
    
    def _calculate_structure_preservation(self, structure: Dict, text: str) -> float:
        """Calculate how well document structure is preserved"""
        if not structure or not text:
            return 0.5  # Neutral score if no structure data
        
        score_factors = []
        
        # Check for page markers
        page_markers = len(re.findall(r'\\[PAGE:', text))
        if page_markers > 0:
            score_factors.append(0.8)
        
        # Check for section markers
        section_markers = len(re.findall(r'\\[SECTION:', text))
        if section_markers > 0:
            score_factors.append(0.9)
        
        # Check for table markers
        table_markers = len(re.findall(r'\\[TABLE_START:', text))
        if table_markers > 0:
            score_factors.append(0.8)
        
        # Check structure data
        if 'sections' in structure and structure['sections']:
            score_factors.append(0.9)
        
        if 'pages' in structure and structure['pages']:
            score_factors.append(0.7)
        
        return statistics.mean(score_factors) if score_factors else 0.5
    
    def _calculate_entity_readiness(self, text: str) -> float:
        """Calculate how ready the text is for entity extraction"""
        if not text:
            return 0.0
        
        readiness_factors = []
        
        # Check for clear sentence structure
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s for s in sentences if len(s.strip().split()) >= 3]
        if sentences:
            sentence_quality = len(valid_sentences) / len(sentences)
            readiness_factors.append(sentence_quality)
        
        # Check for entity-like patterns
        potential_entities = 0
        
        # Capitalized words (potential proper nouns)
        capitalized = re.findall(r'\\b[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*\\b', text)
        potential_entities += len(capitalized)
        
        # Numbers with units (measurements, specifications)
        measurements = re.findall(r'\\d+(?:\\.\\d+)?\\s*(?:psi|mpa|inch|mm|kg|lb|%)', text, re.IGNORECASE)
        potential_entities += len(measurements)
        
        # Standards and references
        standards = re.findall(r'\\b(?:ASTM|ISO|ANSI|AASHTO)\\s*[A-Z]?\\d+', text, re.IGNORECASE)
        potential_entities += len(standards)
        
        # Entity density (entities per 1000 words)
        words = len(text.split())
        if words > 0:
            entity_density = (potential_entities / words) * 1000
            # Good density is between 10-50 entities per 1000 words
            density_score = min(entity_density / 30, 1.0)
            readiness_factors.append(density_score)
        
        # Check for relationship indicators
        relationship_indicators = [
            r'\\b(?:is|are|was|were)\\s+(?:a|an|the)\\b',
            r'\\b(?:has|have|had)\\b',
            r'\\b(?:according\\s+to|per|in\\s+accordance\\s+with)\\b',
            r'\\b(?:shall|must|should)\\b',
        ]
        
        relationship_count = 0
        for pattern in relationship_indicators:
            relationship_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        if words > 0:
            relationship_density = (relationship_count / words) * 1000
            relationship_score = min(relationship_density / 20, 1.0)
            readiness_factors.append(relationship_score)
        
        return statistics.mean(readiness_factors) if readiness_factors else 0.5
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        weights = {
            'text_coverage': 0.25,
            'character_accuracy': 0.20,
            'word_accuracy': 0.15,
            'table_detection_rate': 0.15,
            'structure_preservation': 0.15,
            'entity_readiness': 0.10
        }
        
        weighted_sum = sum(scores[metric] * weights[metric] for metric in weights)
        return weighted_sum
    
    def _validate_text_quality(self, text: str) -> List[QualityIssue]:
        """Validate text quality and identify issues"""
        issues = []
        
        if not text or len(text.strip()) < 100:
            issues.append(QualityIssue(
                issue_type=IssueType.MISSING_TEXT,
                severity="critical",
                page=None,
                description="Very little or no text extracted",
                suggestion="Check PDF format, try OCR if scanned, verify extraction method",
                confidence=0.9
            ))
            return issues
        
        # Check for garbled text
        for pattern in self.garbled_patterns:
            matches = re.findall(pattern, text)
            if len(matches) > 10:  # Too many garbled sequences
                issues.append(QualityIssue(
                    issue_type=IssueType.GARBLED_TEXT,
                    severity="warning",
                    page=None,
                    description=f"Found {len(matches)} sequences of garbled text",
                    suggestion="Try different extraction method or check PDF encoding",
                    confidence=0.8
                ))
        
        # Check for encoding issues
        encoding_issues = 0
        for pattern in self.encoding_patterns:
            encoding_issues += len(re.findall(pattern, text))
        
        if encoding_issues > 20:
            issues.append(QualityIssue(
                issue_type=IssueType.ENCODING_ISSUES,
                severity="warning",
                page=None,
                description=f"Found {encoding_issues} encoding-related issues",
                suggestion="Check PDF encoding, try UTF-8 conversion",
                confidence=0.7
            ))
        
        # Check for broken words (very long sequences without spaces)
        long_words = re.findall(r'\\S{50,}', text)
        if len(long_words) > 5:
            issues.append(QualityIssue(
                issue_type=IssueType.LAYOUT_PROBLEMS,
                severity="info",
                page=None,
                description=f"Found {len(long_words)} very long word sequences",
                suggestion="May indicate layout extraction issues, check spacing",
                confidence=0.6
            ))
        
        return issues
    
    def _validate_table_quality(self, tables: List[Dict]) -> List[QualityIssue]:
        """Validate table extraction quality"""
        issues = []
        
        if not tables:
            return issues
        
        # Check for incomplete tables
        incomplete_tables = 0
        for table in tables:
            data = table.get('data', [])
            if not data or len(data) < 2:  # Less than header + 1 row
                incomplete_tables += 1
            else:
                # Check for too many empty cells
                total_cells = sum(len(row) for row in data)
                empty_cells = sum(1 for row in data for cell in row if not str(cell).strip())
                if total_cells > 0 and empty_cells / total_cells > 0.5:
                    incomplete_tables += 1
        
        if incomplete_tables > len(tables) * 0.3:  # More than 30% incomplete
            issues.append(QualityIssue(
                issue_type=IssueType.INCOMPLETE_TABLES,
                severity="warning",
                page=None,
                description=f"{incomplete_tables}/{len(tables)} tables appear incomplete",
                suggestion="Try different table extraction method or manual review",
                confidence=0.7
            ))
        
        return issues
    
    def _validate_structure_quality(self, structure: Dict) -> List[QualityIssue]:
        """Validate document structure preservation"""
        issues = []
        
        # Check if basic structure elements are present
        if not structure.get('pages') and not structure.get('sections'):
            issues.append(QualityIssue(
                issue_type=IssueType.BROKEN_STRUCTURE,
                severity="info",
                page=None,
                description="No document structure information extracted",
                suggestion="Enable structure extraction or use different method",
                confidence=0.6
            ))
        
        return issues
    
    def _validate_entity_readiness(self, text: str) -> List[QualityIssue]:
        """Validate readiness for entity extraction"""
        issues = []
        
        if not text:
            return issues
        
        # Check sentence structure
        sentences = re.split(r'[.!?]+', text)
        short_sentences = [s for s in sentences if len(s.strip().split()) < 3]
        
        if sentences and len(short_sentences) / len(sentences) > 0.7:
            issues.append(QualityIssue(
                issue_type=IssueType.BROKEN_STRUCTURE,
                severity="info",
                page=None,
                description="Many incomplete sentences found",
                suggestion="May affect entity extraction quality",
                confidence=0.5
            ))
        
        return issues
    
    def _generate_recommendations(self, metrics: QualityMetrics, issues: List[QualityIssue]) -> List[str]:
        """Generate recommendations based on metrics and issues"""
        recommendations = []
        
        # Based on overall score
        if metrics.overall_score < self.thresholds['overall_minimum']:
            recommendations.append(
                f"Overall quality score ({metrics.overall_score:.2f}) is below threshold "
                f"({self.thresholds['overall_minimum']:.2f}). Consider reprocessing with different settings."
            )
        
        # Specific metric recommendations
        if metrics.text_coverage < self.thresholds['text_coverage']:
            recommendations.append(
                f"Text coverage ({metrics.text_coverage:.2f}) is low. Try different extraction method or check for scanned pages."
            )
        
        if metrics.character_accuracy < self.thresholds['character_accuracy']:
            recommendations.append(
                f"Character accuracy ({metrics.character_accuracy:.2f}) is low. Check for encoding issues or try OCR."
            )
        
        if metrics.table_detection_rate < self.thresholds['table_detection']:
            recommendations.append(
                f"Table detection rate ({metrics.table_detection_rate:.2f}) is low. Consider using Camelot or manual table extraction."
            )
        
        if metrics.entity_readiness < self.thresholds['entity_readiness']:
            recommendations.append(
                f"Entity readiness ({metrics.entity_readiness:.2f}) is low. Text may need preprocessing before GraphRAG."
            )
        
        # Issue-based recommendations
        critical_issues = [issue for issue in issues if issue.severity == "critical"]
        if critical_issues:
            recommendations.append(
                f"Found {len(critical_issues)} critical issues that should be addressed before using with GraphRAG."
            )
        
        return recommendations
    
    def _generate_processing_recommendations(self, metrics: QualityMetrics, issues: List[QualityIssue]) -> Dict[str, Any]:
        """Generate specific processing recommendations"""
        recommendations = {
            'chunk_size': 1200,  # Default
            'overlap': 100,      # Default
            'preprocessing': [],
            'graphrag_settings': {}
        }
        
        # Adjust chunk size based on quality
        if metrics.structure_preservation < 0.7:
            recommendations['chunk_size'] = 800  # Smaller chunks for better structure
            recommendations['overlap'] = 200    # More overlap
            recommendations['preprocessing'].append('structure_enhancement')
        
        if metrics.entity_readiness < 0.7:
            recommendations['preprocessing'].append('entity_preprocessing')
            recommendations['graphrag_settings']['max_gleanings'] = 2  # More extraction passes
        
        if metrics.table_detection_rate < 0.7:
            recommendations['preprocessing'].append('table_enhancement')
        
        # Entity types based on content
        if any('technical' in issue.description.lower() for issue in issues):
            recommendations['graphrag_settings']['entity_types'] = [
                'specification', 'material', 'standard', 'equipment', 'procedure',
                'organization', 'location'
            ]
        
        return recommendations
    
    def _assess_graphrag_readiness(self, metrics: QualityMetrics, issues: List[QualityIssue]) -> str:
        """Assess readiness for GraphRAG processing"""
        critical_issues = [issue for issue in issues if issue.severity == "critical"]
        
        if critical_issues:
            return "poor"
        
        if metrics.overall_score >= self.thresholds['overall_minimum']:
            return "ready"
        elif metrics.overall_score >= self.thresholds['overall_minimum'] * 0.8:
            return "needs_work"
        else:
            return "poor"


def main():
    """Example usage"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Validate PDF extraction quality")
    parser.add_argument("text_file", help="Path to extracted text file")
    parser.add_argument("pdf_file", help="Path to original PDF file")
    parser.add_argument("--level", choices=['strict', 'normal', 'lenient'], 
                       default='normal', help="Validation level")
    parser.add_argument("--output", help="Output validation report to JSON file")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Read extracted text
    with open(args.text_file, 'r', encoding='utf-8') as f:
        extracted_text = f.read()
    
    # Create validator
    level = ValidationLevel(args.level)
    validator = QualityValidator(validation_level=level)
    
    # Mock processing metrics (would come from actual processor)
    processing_metrics = {
        'total_pages': 100,  # Placeholder
        'pages_processed': 100,
        'processing_time': 120,
        'errors': 0
    }
    
    # Validate
    report = validator.validate_extraction(
        extracted_text=extracted_text,
        pdf_path=args.pdf_file,
        processing_metrics=processing_metrics
    )
    
    # Print report
    print(f"\\nValidation Report for {args.pdf_file}")
    print("=" * 50)
    print(f"Overall Score: {report.metrics.overall_score:.2f}")
    print(f"GraphRAG Readiness: {report.graphrag_readiness}")
    print()
    
    print("Quality Metrics:")
    print(f"  Text Coverage: {report.metrics.text_coverage:.2f}")
    print(f"  Character Accuracy: {report.metrics.character_accuracy:.2f}")
    print(f"  Word Accuracy: {report.metrics.word_accuracy:.2f}")
    print(f"  Table Detection: {report.metrics.table_detection_rate:.2f}")
    print(f"  Structure Preservation: {report.metrics.structure_preservation:.2f}")
    print(f"  Entity Readiness: {report.metrics.entity_readiness:.2f}")
    print()
    
    if report.issues:
        print(f"Issues Found ({len(report.issues)}):")
        for issue in report.issues:
            print(f"  [{issue.severity.upper()}] {issue.description}")
            print(f"    Suggestion: {issue.suggestion}")
        print()
    
    if report.recommendations:
        print("Recommendations:")
        for rec in report.recommendations:
            print(f"  • {rec}")
        print()
    
    # Save detailed report
    if args.output:
        report_data = {
            'metrics': {
                'text_coverage': report.metrics.text_coverage,
                'character_accuracy': report.metrics.character_accuracy,
                'word_accuracy': report.metrics.word_accuracy,
                'table_detection_rate': report.metrics.table_detection_rate,
                'structure_preservation': report.metrics.structure_preservation,
                'entity_readiness': report.metrics.entity_readiness,
                'overall_score': report.metrics.overall_score
            },
            'issues': [
                {
                    'type': issue.issue_type.value,
                    'severity': issue.severity,
                    'page': issue.page,
                    'description': issue.description,
                    'suggestion': issue.suggestion,
                    'confidence': issue.confidence
                }
                for issue in report.issues
            ],
            'recommendations': report.recommendations,
            'processing_recommendations': report.processing_recommendations,
            'graphrag_readiness': report.graphrag_readiness
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"Detailed report saved to {args.output}")


if __name__ == "__main__":
    main()