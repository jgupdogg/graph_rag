#!/usr/bin/env python3
"""
Advanced Table Extraction for PDF Processing
Specialized handling for tables in technical documents
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class TableType(Enum):
    """Types of tables found in documents"""
    SPECIFICATION = "specification"
    MATERIAL = "material" 
    MEASUREMENT = "measurement"
    REFERENCE = "reference"
    GENERAL = "general"


@dataclass
class TableMetadata:
    """Metadata about extracted table"""
    page: int
    index: int
    rows: int
    cols: int
    table_type: TableType
    confidence: float
    headers: List[str]
    title: Optional[str] = None


@dataclass
class ExtractedTable:
    """Complete table extraction result"""
    data: List[List[str]]
    metadata: TableMetadata
    formatted_text: str
    raw_text: str
    entities: List[Dict]  # Pre-identified entities for GraphRAG


class TableExtractor:
    """
    Advanced table extraction with type detection and formatting
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Patterns for different table types
        self.specification_patterns = [
            r'specification',
            r'requirement',
            r'shall\\s+(be|have)',
            r'minimum|maximum',
            r'type\\s+[A-Z]',
        ]
        
        self.material_patterns = [
            r'concrete|steel|asphalt|aggregate',
            r'\\d+\\s*(psi|mpa|kg|lb)',
            r'material',
            r'grade\\s+\\d+',
        ]
        
        self.measurement_patterns = [
            r'\\d+\\.?\\d*\\s*(inch|mm|cm|ft)',
            r'\\d+\\.?\\d*\\s*(psi|mpa|kpa)',
            r'thickness|width|height|diameter',
            r'tolerance',
        ]
        
    def extract_tables_pymupdf(self, pdf_path: str, page_range: Tuple[int, int]) -> List[ExtractedTable]:
        """Extract tables using PyMuPDF"""
        try:
            import fitz
            
            tables = []
            
            with fitz.open(pdf_path) as doc:
                for page_num in range(page_range[0], min(page_range[1], doc.page_count)):
                    page = doc[page_num]
                    
                    # Try to find tables using text blocks
                    blocks = page.get_text("dict")["blocks"]
                    page_tables = self._detect_tables_from_blocks(blocks, page_num + 1)
                    tables.extend(page_tables)
                    
            return tables
            
        except Exception as e:
            self.logger.error(f"PyMuPDF table extraction failed: {e}")
            return []
    
    def extract_tables_pdfplumber(self, pdf_path: str, page_range: Tuple[int, int]) -> List[ExtractedTable]:
        """Extract tables using pdfplumber"""
        try:
            import pdfplumber
            
            tables = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num in range(page_range[0], min(page_range[1], len(pdf.pages))):
                    page = pdf.pages[page_num]
                    
                    # Extract tables with different strategies
                    page_tables = []
                    
                    # Strategy 1: Default table extraction
                    default_tables = page.extract_tables()
                    if default_tables:
                        for i, table_data in enumerate(default_tables):
                            if table_data and len(table_data) > 1:
                                table = self._process_table_data(table_data, page_num + 1, i)
                                if table:
                                    page_tables.append(table)
                    
                    # Strategy 2: Custom table detection for technical docs
                    if not page_tables:
                        custom_tables = self._detect_technical_tables(page, page_num + 1)
                        page_tables.extend(custom_tables)
                    
                    tables.extend(page_tables)
                    
            return tables
            
        except Exception as e:
            self.logger.error(f"pdfplumber table extraction failed: {e}")
            return []
    
    def extract_tables_camelot(self, pdf_path: str, page_range: Tuple[int, int]) -> List[ExtractedTable]:
        """Extract tables using Camelot"""
        try:
            import camelot
            
            tables = []
            
            for page_num in range(page_range[0] + 1, page_range[1] + 1):  # Camelot uses 1-based indexing
                try:
                    # Try lattice method first (for tables with lines)
                    camelot_tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='lattice')
                    
                    if not camelot_tables:
                        # Try stream method (for tables without lines)
                        camelot_tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='stream')
                    
                    for i, table in enumerate(camelot_tables):
                        if table.df is not None and not table.df.empty:
                            # Convert DataFrame to list of lists
                            table_data = table.df.values.tolist()
                            if table.df.columns.tolist()[0] != '0':  # Has headers
                                table_data.insert(0, table.df.columns.tolist())
                            
                            processed_table = self._process_table_data(table_data, page_num, i)
                            if processed_table:
                                # Add Camelot-specific metadata
                                processed_table.metadata.confidence = table.accuracy / 100.0
                                tables.append(processed_table)
                                
                except Exception as e:
                    self.logger.debug(f"Camelot failed for page {page_num}: {e}")
                    continue
                    
            return tables
            
        except Exception as e:
            self.logger.error(f"Camelot table extraction failed: {e}")
            return []
    
    def _detect_tables_from_blocks(self, blocks: List[Dict], page_num: int) -> List[ExtractedTable]:
        """Detect tables from PyMuPDF text blocks"""
        tables = []
        
        # Look for tabular patterns in text blocks
        table_candidates = []
        
        for block in blocks:
            if "lines" in block:
                block_text = ""
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        block_text += span.get("text", "") + " "
                
                # Check if block looks like a table
                if self._is_tabular_text(block_text):
                    table_candidates.append({
                        'text': block_text.strip(),
                        'bbox': block.get('bbox', []),
                        'block': block
                    })
        
        # Process table candidates
        for i, candidate in enumerate(table_candidates):
            table_data = self._parse_tabular_text(candidate['text'])
            if table_data and len(table_data) > 1:
                table = self._process_table_data(table_data, page_num, i)
                if table:
                    tables.append(table)
        
        return tables
    
    def _is_tabular_text(self, text: str) -> bool:
        """Check if text appears to be tabular"""
        lines = text.strip().split('\\n')
        
        if len(lines) < 2:
            return False
        
        # Count common tabular indicators
        tabular_score = 0
        
        # Check for consistent spacing/alignment
        if self._has_consistent_columns(lines):
            tabular_score += 2
        
        # Check for table-like patterns
        for line in lines[:5]:  # Check first few lines
            # Multiple spaces or tabs
            if re.search(r'\\s{3,}|\\t', line):
                tabular_score += 1
            
            # Numbers in columns
            if re.search(r'\\d+\\.?\\d*\\s+\\d+\\.?\\d*', line):
                tabular_score += 1
            
            # Measurement units
            if re.search(r'\\d+\\s*(psi|mpa|inch|mm|kg|lb)', line):
                tabular_score += 1
        
        return tabular_score >= 3
    
    def _has_consistent_columns(self, lines: List[str]) -> bool:
        """Check if lines have consistent column structure"""
        if len(lines) < 3:
            return False
        
        # Find common whitespace patterns
        patterns = []
        for line in lines[:5]:
            # Find positions of multiple spaces
            spaces = [m.start() for m in re.finditer(r'\\s{2,}', line)]
            if spaces:
                patterns.append(spaces)
        
        if len(patterns) < 2:
            return False
        
        # Check if patterns are similar
        base_pattern = patterns[0]
        similar_count = 0
        
        for pattern in patterns[1:]:
            if self._patterns_similar(base_pattern, pattern):
                similar_count += 1
        
        return similar_count >= len(patterns) * 0.7
    
    def _patterns_similar(self, pattern1: List[int], pattern2: List[int], tolerance: int = 5) -> bool:
        """Check if two spacing patterns are similar"""
        if abs(len(pattern1) - len(pattern2)) > 1:
            return False
        
        matches = 0
        for i, pos1 in enumerate(pattern1):
            for j, pos2 in enumerate(pattern2):
                if abs(pos1 - pos2) <= tolerance:
                    matches += 1
                    break
        
        return matches >= min(len(pattern1), len(pattern2)) * 0.8
    
    def _parse_tabular_text(self, text: str) -> List[List[str]]:
        """Parse tabular text into rows and columns"""
        lines = [line.strip() for line in text.split('\\n') if line.strip()]
        
        if len(lines) < 2:
            return []
        
        # Try different separation strategies
        table_data = []
        
        # Strategy 1: Multiple spaces or tabs
        for line in lines:
            if re.search(r'\\s{2,}|\\t', line):
                # Split on multiple spaces/tabs
                cells = re.split(r'\\s{2,}|\\t+', line)
                cells = [cell.strip() for cell in cells if cell.strip()]
                if cells:
                    table_data.append(cells)
        
        # Strategy 2: Fixed positions (if strategy 1 fails)
        if not table_data or len(table_data) < 2:
            # Try to detect column positions
            positions = self._detect_column_positions(lines)
            if positions:
                for line in lines:
                    cells = self._split_by_positions(line, positions)
                    if cells:
                        table_data.append(cells)
        
        return table_data if len(table_data) >= 2 else []
    
    def _detect_column_positions(self, lines: List[str]) -> List[int]:
        """Detect column separator positions"""
        # Find common word boundaries across lines
        position_counts = {}
        
        for line in lines[:10]:  # Analyze first 10 lines
            words = line.split()
            if len(words) > 1:
                pos = 0
                for word in words[:-1]:  # Don't count the last word
                    pos = line.find(word, pos) + len(word)
                    # Find next non-space position
                    while pos < len(line) and line[pos] == ' ':
                        pos += 1
                    
                    position_counts[pos] = position_counts.get(pos, 0) + 1
        
        # Get positions that appear frequently
        min_frequency = max(1, len(lines) * 0.3)
        positions = [pos for pos, count in position_counts.items() if count >= min_frequency]
        
        return sorted(positions)
    
    def _split_by_positions(self, line: str, positions: List[int]) -> List[str]:
        """Split line by fixed positions"""
        if not positions:
            return [line.strip()] if line.strip() else []
        
        cells = []
        prev_pos = 0
        
        for pos in positions:
            if pos < len(line):
                cell = line[prev_pos:pos].strip()
                if cell:
                    cells.append(cell)
                prev_pos = pos
        
        # Add remaining text
        if prev_pos < len(line):
            cell = line[prev_pos:].strip()
            if cell:
                cells.append(cell)
        
        return cells
    
    def _detect_technical_tables(self, page, page_num: int) -> List[ExtractedTable]:
        """Detect tables specific to technical documents"""
        tables = []
        
        # Extract text and look for specification patterns
        text = page.extract_text()
        if not text:
            return tables
        
        lines = text.split('\\n')
        
        # Look for specification tables
        spec_table = self._extract_specification_table(lines, page_num)
        if spec_table:
            tables.append(spec_table)
        
        # Look for material property tables  
        material_table = self._extract_material_table(lines, page_num)
        if material_table:
            tables.append(material_table)
        
        return tables
    
    def _extract_specification_table(self, lines: List[str], page_num: int) -> Optional[ExtractedTable]:
        """Extract specification tables from text lines"""
        spec_lines = []
        
        for line in lines:
            line = line.strip()
            # Look for specification patterns
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in self.specification_patterns):
                # Check if line has tabular structure
                if self._is_specification_line(line):
                    spec_lines.append(line)
        
        if len(spec_lines) < 2:
            return None
        
        # Try to parse as table
        table_data = []
        headers = ['Specification', 'Requirement', 'Standard']
        table_data.append(headers)
        
        for line in spec_lines:
            row = self._parse_specification_line(line)
            if row:
                table_data.append(row)
        
        if len(table_data) > 1:
            return self._process_table_data(table_data, page_num, 0, TableType.SPECIFICATION)
        
        return None
    
    def _is_specification_line(self, line: str) -> bool:
        """Check if line contains specification data"""
        # Look for patterns like "Material shall be minimum 4000 psi"
        patterns = [
            r'\\w+\\s+shall\\s+(be|have)\\s+[^.]+',
            r'minimum\\s+\\d+\\s*\\w+',
            r'maximum\\s+\\d+\\s*\\w+',
            r'type\\s+[A-Z]\\s+\\w+',
        ]
        
        return any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns)
    
    def _parse_specification_line(self, line: str) -> Optional[List[str]]:
        """Parse specification line into table columns"""
        # Try to extract: [item] [specification] [standard/reference]
        
        # Pattern 1: "Material shall be minimum X unit per Standard"
        match = re.search(r'(\\w+)\\s+shall\\s+(be|have)\\s+([^.]+?)\\s+(?:per|in accordance with|conforming to)\\s+([A-Z\\d\\s-]+)', 
                         line, re.IGNORECASE)
        if match:
            return [match.group(1), match.group(3), match.group(4)]
        
        # Pattern 2: "Material shall be minimum X unit"
        match = re.search(r'(\\w+)\\s+shall\\s+(be|have)\\s+([^.]+)', line, re.IGNORECASE)
        if match:
            return [match.group(1), match.group(3), '']
        
        # Pattern 3: Simple split on multiple spaces
        parts = re.split(r'\\s{3,}', line)
        if len(parts) >= 2:
            return parts[:3] + [''] * (3 - len(parts))
        
        return None
    
    def _extract_material_table(self, lines: List[str], page_num: int) -> Optional[ExtractedTable]:
        """Extract material property tables"""
        material_lines = []
        
        for line in lines:
            line = line.strip()
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in self.material_patterns):
                if self._is_material_line(line):
                    material_lines.append(line)
        
        if len(material_lines) < 2:
            return None
        
        # Parse as material properties table
        table_data = [['Material', 'Property', 'Value', 'Unit']]
        
        for line in material_lines:
            row = self._parse_material_line(line)
            if row:
                table_data.append(row)
        
        if len(table_data) > 1:
            return self._process_table_data(table_data, page_num, 0, TableType.MATERIAL)
        
        return None
    
    def _is_material_line(self, line: str) -> bool:
        """Check if line contains material property data"""
        # Look for material + number + unit patterns
        return bool(re.search(r'\\w+\\s+\\d+\\.?\\d*\\s*(psi|mpa|kg|lb|inch|mm)', line, re.IGNORECASE))
    
    def _parse_material_line(self, line: str) -> Optional[List[str]]:
        """Parse material property line"""
        # Pattern: "Concrete compressive strength 4000 psi"
        match = re.search(r'(\\w+)\\s+(\\w+(?:\\s+\\w+)*)\\s+(\\d+\\.?\\d*)\\s*(\\w+)', line, re.IGNORECASE)
        if match:
            return [match.group(1), match.group(2), match.group(3), match.group(4)]
        
        return None
    
    def _process_table_data(self, table_data: List[List[str]], page_num: int, index: int, 
                           table_type: TableType = None) -> Optional[ExtractedTable]:
        """Process raw table data into ExtractedTable"""
        if not table_data or len(table_data) < 2:
            return None
        
        # Clean table data
        cleaned_data = []
        for row in table_data:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            if any(cleaned_row):  # Skip empty rows
                cleaned_data.append(cleaned_row)
        
        if len(cleaned_data) < 2:
            return None
        
        # Determine table type if not provided
        if table_type is None:
            table_type = self._classify_table(cleaned_data)
        
        # Extract headers
        headers = cleaned_data[0] if cleaned_data else []
        
        # Create metadata
        metadata = TableMetadata(
            page=page_num,
            index=index,
            rows=len(cleaned_data),
            cols=len(cleaned_data[0]) if cleaned_data else 0,
            table_type=table_type,
            confidence=self._calculate_table_confidence(cleaned_data, table_type),
            headers=headers
        )
        
        # Format table text
        formatted_text = self._format_table_for_graphrag(cleaned_data, metadata)
        
        # Extract entities
        entities = self._extract_table_entities(cleaned_data, table_type)
        
        # Create raw text representation
        raw_text = self._create_raw_table_text(cleaned_data)
        
        return ExtractedTable(
            data=cleaned_data,
            metadata=metadata,
            formatted_text=formatted_text,
            raw_text=raw_text,
            entities=entities
        )
    
    def _classify_table(self, table_data: List[List[str]]) -> TableType:
        """Classify table type based on content"""
        content = ' '.join(' '.join(row) for row in table_data).lower()
        
        # Count type indicators
        spec_score = sum(1 for pattern in self.specification_patterns 
                        if re.search(pattern, content, re.IGNORECASE))
        
        material_score = sum(1 for pattern in self.material_patterns 
                           if re.search(pattern, content, re.IGNORECASE))
        
        measurement_score = sum(1 for pattern in self.measurement_patterns 
                              if re.search(pattern, content, re.IGNORECASE))
        
        # Determine type based on highest score
        scores = {
            TableType.SPECIFICATION: spec_score,
            TableType.MATERIAL: material_score,
            TableType.MEASUREMENT: measurement_score
        }
        
        best_type = max(scores, key=scores.get)
        return best_type if scores[best_type] > 0 else TableType.GENERAL
    
    def _calculate_table_confidence(self, table_data: List[List[str]], table_type: TableType) -> float:
        """Calculate confidence score for table extraction"""
        if not table_data:
            return 0.0
        
        confidence_factors = []
        
        # Factor 1: Table structure consistency
        col_counts = [len(row) for row in table_data]
        if col_counts:
            max_cols = max(col_counts)
            consistent_cols = sum(1 for count in col_counts if count == max_cols)
            structure_score = consistent_cols / len(col_counts)
            confidence_factors.append(structure_score)
        
        # Factor 2: Content quality (non-empty cells)
        total_cells = sum(len(row) for row in table_data)
        non_empty_cells = sum(1 for row in table_data for cell in row if str(cell).strip())
        content_score = non_empty_cells / total_cells if total_cells > 0 else 0
        confidence_factors.append(content_score)
        
        # Factor 3: Type-specific patterns
        content = ' '.join(' '.join(row) for row in table_data).lower()
        if table_type == TableType.SPECIFICATION:
            pattern_score = sum(1 for pattern in self.specification_patterns 
                              if re.search(pattern, content, re.IGNORECASE)) / len(self.specification_patterns)
        elif table_type == TableType.MATERIAL:
            pattern_score = sum(1 for pattern in self.material_patterns 
                              if re.search(pattern, content, re.IGNORECASE)) / len(self.material_patterns)
        else:
            pattern_score = 0.5  # Neutral score for general tables
        
        confidence_factors.append(min(pattern_score, 1.0))
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _format_table_for_graphrag(self, table_data: List[List[str]], metadata: TableMetadata) -> str:
        """Format table for GraphRAG processing"""
        lines = []
        
        # Add table header with metadata
        lines.append(f"\\n[TABLE_START: Page {metadata.page}, Table {metadata.index + 1}]")
        lines.append(f"[TABLE_TYPE: {metadata.table_type.value}]")
        lines.append(f"[TABLE_SIZE: {metadata.rows} rows × {metadata.cols} columns]")
        lines.append(f"[CONFIDENCE: {metadata.confidence:.2f}]")
        
        if metadata.headers:
            lines.append(f"[HEADERS: {' | '.join(metadata.headers)}]")
        
        lines.append("")  # Empty line before table
        
        # Format table content
        if table_data:
            # Headers
            if metadata.headers:
                lines.append(" | ".join(metadata.headers))
                lines.append("-" * (sum(len(h) for h in metadata.headers) + len(metadata.headers) * 3))
            
            # Data rows (skip first row if it's headers)
            start_row = 1 if metadata.headers else 0
            for row in table_data[start_row:]:
                if any(str(cell).strip() for cell in row):  # Skip empty rows
                    formatted_row = " | ".join(str(cell) if cell else "" for cell in row)
                    lines.append(formatted_row)
        
        lines.append("")  # Empty line after table
        lines.append("[TABLE_END]\\n")
        
        return "\\n".join(lines)
    
    def _extract_table_entities(self, table_data: List[List[str]], table_type: TableType) -> List[Dict]:
        """Extract entities from table for GraphRAG"""
        entities = []
        
        if not table_data:
            return entities
        
        # Extract based on table type
        if table_type == TableType.SPECIFICATION:
            entities.extend(self._extract_specification_entities(table_data))
        elif table_type == TableType.MATERIAL:
            entities.extend(self._extract_material_entities(table_data))
        elif table_type == TableType.MEASUREMENT:
            entities.extend(self._extract_measurement_entities(table_data))
        
        # Extract general entities from all table types
        entities.extend(self._extract_general_entities(table_data))
        
        return entities
    
    def _extract_specification_entities(self, table_data: List[List[str]]) -> List[Dict]:
        """Extract specification-specific entities"""
        entities = []
        
        for row_idx, row in enumerate(table_data[1:], 1):  # Skip header
            for col_idx, cell in enumerate(row):
                cell_text = str(cell).strip()
                if not cell_text:
                    continue
                
                # Extract standards (ASTM, ISO, etc.)
                standards = re.findall(r'(astm|iso|ansi|aashto)\\s*([A-Z]?\\d+(?:[.-]\\d+)?)', 
                                     cell_text, re.IGNORECASE)
                for standard in standards:
                    entities.append({
                        'text': f"{standard[0].upper()} {standard[1]}",
                        'type': 'standard',
                        'row': row_idx,
                        'col': col_idx,
                        'category': 'specification'
                    })
                
                # Extract requirements
                if re.search(r'shall\\s+(be|have)', cell_text, re.IGNORECASE):
                    entities.append({
                        'text': cell_text,
                        'type': 'requirement',
                        'row': row_idx,
                        'col': col_idx,
                        'category': 'specification'
                    })
        
        return entities
    
    def _extract_material_entities(self, table_data: List[List[str]]) -> List[Dict]:
        """Extract material-specific entities"""
        entities = []
        
        for row_idx, row in enumerate(table_data[1:], 1):
            for col_idx, cell in enumerate(row):
                cell_text = str(cell).strip()
                if not cell_text:
                    continue
                
                # Extract materials
                materials = re.findall(r'(concrete|steel|asphalt|aggregate|cement)', 
                                     cell_text, re.IGNORECASE)
                for material in materials:
                    entities.append({
                        'text': material.lower(),
                        'type': 'material',
                        'row': row_idx,
                        'col': col_idx,
                        'category': 'material'
                    })
                
                # Extract measurements
                measurements = re.findall(r'(\\d+(?:\\.\\d+)?)\\s*(psi|mpa|kg|lb|inch|mm)', 
                                        cell_text, re.IGNORECASE)
                for value, unit in measurements:
                    entities.append({
                        'text': f"{value} {unit.lower()}",
                        'type': 'measurement',
                        'value': value,
                        'unit': unit.lower(),
                        'row': row_idx,
                        'col': col_idx,
                        'category': 'material'
                    })
        
        return entities
    
    def _extract_measurement_entities(self, table_data: List[List[str]]) -> List[Dict]:
        """Extract measurement-specific entities"""
        entities = []
        
        for row_idx, row in enumerate(table_data[1:], 1):
            for col_idx, cell in enumerate(row):
                cell_text = str(cell).strip()
                if not cell_text:
                    continue
                
                # Extract all measurements
                measurements = re.findall(r'(\\d+(?:\\.\\d+)?)\\s*(\\w+)', cell_text)
                for value, unit in measurements:
                    if unit.lower() in ['psi', 'mpa', 'kpa', 'kg', 'lb', 'inch', 'mm', 'cm', 'ft']:
                        entities.append({
                            'text': f"{value} {unit.lower()}",
                            'type': 'measurement',
                            'value': value,
                            'unit': unit.lower(),
                            'row': row_idx,
                            'col': col_idx,
                            'category': 'measurement'
                        })
        
        return entities
    
    def _extract_general_entities(self, table_data: List[List[str]]) -> List[Dict]:
        """Extract general entities from any table"""
        entities = []
        
        for row_idx, row in enumerate(table_data[1:], 1):
            for col_idx, cell in enumerate(row):
                cell_text = str(cell).strip()
                if not cell_text:
                    continue
                
                # Extract numbers that might be important
                numbers = re.findall(r'\\d+(?:\\.\\d+)?', cell_text)
                for number in numbers:
                    if float(number) > 1:  # Skip small numbers that are likely not significant
                        entities.append({
                            'text': number,
                            'type': 'number',
                            'value': number,
                            'row': row_idx,
                            'col': col_idx,
                            'category': 'general'
                        })
        
        return entities
    
    def _create_raw_table_text(self, table_data: List[List[str]]) -> str:
        """Create raw text representation of table"""
        lines = []
        for row in table_data:
            line = "\\t".join(str(cell) if cell else "" for cell in row)
            lines.append(line)
        return "\\n".join(lines)


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract tables from PDF")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--method", choices=['pymupdf', 'pdfplumber', 'camelot'], 
                       default='pdfplumber', help="Extraction method")
    parser.add_argument("--pages", help="Page range (e.g., '1-10')")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse page range
    if args.pages:
        start, end = map(int, args.pages.split('-'))
    else:
        start, end = 0, 999
    
    # Extract tables
    extractor = TableExtractor()
    
    if args.method == 'pymupdf':
        tables = extractor.extract_tables_pymupdf(args.pdf_path, (start, end))
    elif args.method == 'pdfplumber':
        tables = extractor.extract_tables_pdfplumber(args.pdf_path, (start, end))
    elif args.method == 'camelot':
        tables = extractor.extract_tables_camelot(args.pdf_path, (start, end))
    
    print(f"Found {len(tables)} tables using {args.method}")
    
    for i, table in enumerate(tables):
        print(f"\\nTable {i + 1}:")
        print(f"  Page: {table.metadata.page}")
        print(f"  Type: {table.metadata.table_type.value}")
        print(f"  Size: {table.metadata.rows}×{table.metadata.cols}")
        print(f"  Confidence: {table.metadata.confidence:.2f}")
        print(f"  Entities: {len(table.entities)}")
        print("  Preview:")
        print(table.formatted_text[:200] + "..." if len(table.formatted_text) > 200 else table.formatted_text)


if __name__ == "__main__":
    main()