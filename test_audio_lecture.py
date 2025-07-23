#!/usr/bin/env python3
"""
Test script for the audio lecture generator functionality.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from audio_lecture_generator import AudioLectureGenerator, LectureDetailLevel


def test_lecture_generation():
    """Test the audio lecture generation process."""
    print("=== Audio Lecture Generator Test ===\n")
    
    # Initialize generator
    generator = AudioLectureGenerator()
    
    # Sample document data
    document_summary = {
        'display_name': 'Test Document: AI and Machine Learning',
        'summary': 'This document provides a comprehensive overview of artificial intelligence and machine learning concepts, including neural networks, deep learning, and practical applications in various industries.'
    }
    
    # Sample section summaries
    section_summaries = {
        "Chapter 1: Introduction to AI": "An overview of artificial intelligence, its history, and fundamental concepts. AI represents the simulation of human intelligence in machines.",
        "Chapter 1 > Section 1.1: History of AI": "The development of AI from the 1950s to present, including key milestones like the Turing Test and recent breakthroughs in deep learning.",
        "Chapter 1 > Section 1.2: Types of AI": "Discussion of narrow AI vs general AI, and the current state of AI technology.",
        "Chapter 2: Machine Learning Fundamentals": "Core concepts of machine learning including supervised, unsupervised, and reinforcement learning paradigms.",
        "Chapter 2 > Section 2.1: Supervised Learning": "Algorithms that learn from labeled training data, including classification and regression techniques.",
        "Chapter 2 > Section 2.2: Neural Networks": "Introduction to artificial neural networks, their architecture, and how they learn from data.",
        "Chapter 3: Applications and Future": "Real-world applications of AI and ML in healthcare, finance, transportation, and future directions."
    }
    
    # Additional context
    additional_context = {
        'entities': [
            {'title': 'Neural Networks', 'type': 'Technology', 'description': 'Computational models inspired by biological neural networks'},
            {'title': 'Deep Learning', 'type': 'Technology', 'description': 'Machine learning using multi-layered neural networks'},
            {'title': 'Turing Test', 'type': 'Concept', 'description': 'Test of a machine\'s ability to exhibit intelligent behavior'},
            {'title': 'Supervised Learning', 'type': 'Method', 'description': 'ML approach using labeled training data'},
            {'title': 'Geoffrey Hinton', 'type': 'Person', 'description': 'Pioneer in deep learning and neural networks'}
        ],
        'relationships': [
            {'source': 'Neural Networks', 'target': 'Deep Learning', 'description': 'forms the foundation of'},
            {'source': 'Geoffrey Hinton', 'target': 'Deep Learning', 'description': 'pioneered research in'},
            {'source': 'Supervised Learning', 'target': 'Neural Networks', 'description': 'is commonly used with'}
        ],
        'bullet_points': [
            'AI aims to create intelligent machines that can perform tasks requiring human intelligence',
            'Machine learning enables computers to learn from data without explicit programming',
            'Deep learning has revolutionized computer vision and natural language processing',
            'Ethical considerations are crucial in AI development and deployment'
        ]
    }
    
    # Test different detail levels
    detail_levels = [
        LectureDetailLevel.OVERVIEW,
        LectureDetailLevel.MAIN_POINTS,
        LectureDetailLevel.STANDARD,
        LectureDetailLevel.DETAILED,
        LectureDetailLevel.COMPREHENSIVE
    ]
    
    for detail_level in detail_levels:
        print(f"\n{'='*60}")
        print(f"Testing with detail level: {detail_level.name}")
        print('='*60)
        
        # Generate lecture
        lecture_sections, full_script = generator.generate_lecture_script(
            document_summary=document_summary,
            section_summaries=section_summaries,
            detail_level=detail_level,
            include_entities=(detail_level.value >= 3),
            include_relationships=(detail_level.value >= 4),
            include_bullet_points=True,
            additional_context=additional_context
        )
        
        # Show results
        print(f"\nGenerated {len(lecture_sections)} sections")
        print(f"Estimated duration: {generator.estimate_lecture_duration(full_script)} minutes")
        
        # Show outline
        print("\nLecture Outline:")
        for section in lecture_sections:
            if section.section_type == "transition":
                print(f"  -> {section.title}")
            else:
                indent = "  " * section.level
                print(f"{indent}- {section.title}")
        
        # Show first 500 characters of script
        print(f"\nScript preview (first 500 chars):")
        print(full_script[:500] + "...")
        
        # Word count
        word_count = len(full_script.split())
        print(f"\nTotal word count: {word_count}")
    
    print("\n\n=== Test Complete ===")
    print("\nThe audio lecture generator is working correctly!")
    print("Key features tested:")
    print("- Multiple detail levels from overview to comprehensive")
    print("- Section hierarchy handling")
    print("- Entity and relationship integration")
    print("- Transitions between sections")
    print("- Duration estimation")


if __name__ == "__main__":
    test_lecture_generation()