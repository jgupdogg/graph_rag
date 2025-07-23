# Audio Lecture Generator Feature

## Overview

The Audio Lecture Generator is a new feature in GraphRAG Explorer that transforms document content into comprehensive audio lectures. It leverages the various document representations (summaries, entities, relationships, community reports) to create engaging, educational audio content.

## Features

### 1. **Lecture Script Persistence**
- All generated lecture scripts are automatically saved to the document's workspace
- Scripts are stored as JSON files with complete metadata
- Multiple versions can be saved with different detail levels
- Previous lectures can be viewed and downloaded
- Scripts include:
  - Full text content
  - Section breakdown
  - Generation metadata (voice, quality, timestamp)
  - Duration estimates
  - Word/character counts

### 2. **Multiple Detail Levels**
- **Overview**: High-level summary only (1-2 minutes)
- **Main Points**: Key sections and main topics (2-4 minutes)
- **Standard**: All sections with summaries (4-8 minutes)
- **Detailed**: Includes entities and relationships (8-12 minutes)
- **Comprehensive**: Everything including examples and specific details (12+ minutes)

### 2. **Smart Content Integration**
The system intelligently combines:
- Document-level summaries as introduction
- Section summaries in hierarchical order
- Key entities and concepts
- Important relationships between concepts
- Community insights and reports
- Bullet point takeaways

### 3. **Professional Audio Output**
- Uses OpenAI's TTS API for high-quality speech
- Multiple voice options (Alloy, Echo, Fable, Onyx, Nova, Shimmer)
- Standard and HD quality options
- Automatic duration estimation
- MP3 download capability

### 4. **Intelligent Script Generation**
- Natural transitions between sections
- Contextual introductions and conclusions
- Hierarchical content organization
- Relevant entity and relationship mentions
- Key takeaways and summaries

## How It Works

### Document Representations Used

1. **Text Vectors**: For finding relevant content
2. **Section Summaries**: Main content structure
3. **Bullet Points**: Key takeaways
4. **Community Reports**: High-level insights
5. **Entity Graph**: Important concepts
6. **Relationships**: Connections between concepts
7. **Document Summaries**: Overall context

### Generation Process

1. **Content Collection**
   - Gathers all document representations
   - Organizes by hierarchy and relevance
   - Filters based on detail level

2. **Script Generation**
   - Creates introduction from document summary
   - Processes sections hierarchically
   - Adds transitions between major sections
   - Integrates entities and relationships contextually
   - Generates conclusion with key takeaways

3. **Audio Creation**
   - Sends script to OpenAI TTS API
   - Generates high-quality MP3 file
   - Provides playback and download options

## Usage

1. Navigate to the **Document Summary** tab
2. Scroll to the **Audio Lecture Generator** section
3. View any previously generated lectures in the expandable section
4. Select your desired detail level
5. Choose voice and quality settings
6. Configure advanced options if needed
7. Click **Generate Audio Lecture**
8. Review the outline and script
9. Download both the audio file (MP3) and script (TXT)

### Persistence Features

- **Automatic Saving**: Every generated lecture script is automatically saved to the document's workspace
- **Version History**: View and download previously generated lectures
- **Multiple Formats**: Download as audio (MP3) or text script (TXT)
- **Workspace Storage**: Scripts are stored in `workspaces/{doc_id}/cache/audio_lectures/`

## Configuration

### Required Setup
- OpenAI API key in `.env` file
- Processed documents with summaries
- Section summaries (generated during processing)

### Optional Enhancements
- Entity extraction enabled
- Relationship extraction enabled
- Community reports generated
- Bullet point summaries available

## API Components

### AudioLectureGenerator Class
```python
generator = AudioLectureGenerator()
lecture_sections, full_script = generator.generate_lecture_script(
    document_summary=document_summary,
    section_summaries=section_summaries,
    detail_level=LectureDetailLevel.STANDARD,
    include_entities=True,
    include_relationships=True,
    include_bullet_points=True,
    additional_context=context_dict
)
```

### LectureDetailLevel Enum
- `OVERVIEW` (1): Highest level only
- `MAIN_POINTS` (2): Main sections
- `STANDARD` (3): All sections
- `DETAILED` (4): + Entities & relationships
- `COMPREHENSIVE` (5): Everything

### LectureSection Class
Represents individual sections with:
- Title
- Content
- Hierarchy level
- Section type (introduction, section, transition, conclusion)
- Associated entities, relationships, and key points

## Future Enhancements

1. **Audio Features**
   - Background music for intro/outro
   - Sound effects for transitions
   - Multiple speaker voices for dialogue

2. **Content Features**
   - Custom section selection
   - Time-based summaries (5min, 10min, etc.)
   - Q&A format lectures
   - Interactive transcripts

3. **Export Options**
   - Podcast format with metadata
   - Chapter markers
   - Transcript generation
   - Multiple language support

## Testing

Run the test scripts:
```bash
# Test core functionality
python3 test_audio_lecture.py

# Test Streamlit integration
streamlit run test_audio_lecture_streamlit.py
```

## Troubleshooting

### Common Issues

1. **No audio generated**
   - Check OpenAI API key is set
   - Verify document has summaries
   - Ensure section summaries exist

2. **Poor audio quality**
   - Use HD quality option
   - Try different voices
   - Check text formatting

3. **Script too long**
   - Reduce detail level
   - Process fewer documents
   - Use section selection (future)

### Debug Information
The feature includes detailed error messages and logging to help diagnose issues.