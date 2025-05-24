from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from utils import common
from modules.linkedin import google_search_linkedin
import json
import logging
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Pydantic models for transcript enhancement
class EnhancedSegment(BaseModel):
    """Represents an enhanced transcript segment with named speakers and corrected text"""
    speaker: str  # Named speaker (e.g., "Andre" instead of "SPEAKER_00")
    text: str  # Corrected text
    start: float  # Start time in seconds
    end: float  # End time in seconds

class EnhancedTranscript(BaseModel):
    """Represents an enhanced transcript with named speakers and corrected text"""
    transcript_id: Optional[str] = None
    text: str  # Full corrected transcript text
    segments: List[EnhancedSegment]  # List of enhanced segments
    confidence: float = Field(description="Confidence score for the enhancement")
    
class MindMapNode(BaseModel):
    """Represents a node in a mermaid mindmap"""
    id: int
    label: str
    parent_id: Optional[int] = None
    
class MindMap(BaseModel):
    """Represents a mermaid mindmap structure"""
    nodes: List[MindMapNode]
    
class ConversationSummary(BaseModel):
    """Represents a summary of a conversation with a mindmap"""
    summary: str  # Markdown-formatted summary
    mindmap: MindMap  # Mermaid mindmap structure

# Initialize Gemini client
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))


def find_linkedin_profile(transcript_text: str) -> list[str]:
    """
    Find a LinkedIn profile URL for a given name.
    
    Args:
        transcript_text (str): The full text of the transcript to search for on LinkedIn using context clues from the transcript
        
    Returns:
        list[str]: A list of LinkedIn profile URLs if found that might match the profile, otherwise an empty list
    """
    query_config = types.GenerateContentConfig(
        tools=[google_search_linkedin],
        response_mime_type='text/plain',
        temperature=1.0
    )

    base_prompt = """
    You are an AI assistant specialized in finding LinkedIn profiles based on context clues from a transcript.
    
    ## IMPORTANT RULES:
    - If a speaker's name is clearly mentioned in the conversation (e.g., "Hi, I'm John Smith"), use that name to search for the profile
    - Do not invent names that aren't mentioned in the conversation
    - Correct obvious speech recognition errors but preserve the authentic style of spoken language
    - Do not add new content that wasn't in the original transcript
    
    ## TASK:
    Find a LinkedIn profile URL for the following transcript text:
    
    {}
    
    ## RESPONSE:
    [
        {
            "name": "John Smith",
            "linkedin_url": "https://www.linkedin.com/in/gmcast",
            "job_title": "Student",
            "description": "Head of CS club, interested in AI"
        },
        ...
    ]
    """
    
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[base_prompt.format(transcript_text)],
        config=query_config
    )

    # parse the response for the urls using regex
    urls = re.findall(r'https://www.linkedin.com/in/[^\s]+', response.text)

    # TODO: Reformat to proper response
    # [
    #     {
    #         "name": "John Smith",
    #         "linkedin_url": "https://www.linkedin.com/in/gmcast",
    #         "job_title": "Student",
    #         "description": "Head of CS club, interested in AI"
    #     },
    #     ...
    # ]
    
    return urls

def _fix_split_sentences(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyzes and fixes unnaturally split sentences across multiple speaker segments.
    
    Args:
        segments: List of speaker segments from Gemini's response
        
    Returns:
        A processed list of speaker segments with fixed split sentences
    """
    if not segments or len(segments) < 2:
        return segments
        
    # Make a copy to avoid modifying the original
    processed = segments.copy()
    i = 0
    
    while i < len(processed) - 1:
        curr_segment = processed[i]
        next_segment = processed[i + 1]
        
        curr_text = curr_segment.get("text", "").strip()
        next_text = next_segment.get("text", "").strip()
        
        # Check if current segment ends with an incomplete sentence
        # Look for signs like ending with prepositions, no punctuation, etc.
        incomplete_endings = [" of", " the", " a", " an", " to", " in", " on", " at", " with", " by"]
        abrupt_ending = not any(curr_text.endswith(p) for p in [".", "!", "?", "...", ";", ":"])
        ends_with_preposition = any(curr_text.endswith(ending) for ending in incomplete_endings)
        
        # Check if these should be merged
        if (ends_with_preposition or abrupt_ending) and len(curr_text.split()) < 5:
            # It's likely a split sentence, merge with the next segment
            # Determine which speaker was likely speaking based on context
            # For simplicity, we'll keep the speaker of the first segment for now
            speaker = curr_segment.get("speaker")
            
            # Merge the text
            merged_text = f"{curr_text} {next_text}"
            
            # Keep the start time of first segment and end time of second segment
            start = curr_segment.get("start", 0)
            end = next_segment.get("end", 0)
            
            # Create a new merged segment
            merged_segment = {
                "speaker": speaker,
                "text": merged_text,
                "start": start,
                "end": end
            }
            
            # Replace current segment with merged segment and remove next segment
            processed[i] = merged_segment
            processed.pop(i + 1)
            
            # Don't increment i since we need to check if the merged segment
            # should be merged with the next segment too
        else:
            i += 1
    
    return processed

async def enhance_transcript(transcript_data: Dict[str, Any]) -> EnhancedTranscript:
    """
    Enhances a transcript by identifying speaker names and correcting text.
    
    Args:
        transcript_data: The original transcript data with speaker diarization
        
    Returns:
        An enhanced transcript with named speakers and corrected text
    """
    try:
        logger.info("Enhancing transcript with Gemini")
        
        # Create a string representation of the segments for the prompt
        segments_text = ""
        try:
            for i, segment in enumerate(transcript_data.get("segments", [])):
                segment_num = i + 1
                speaker = segment.get('speaker', 'Unknown')
                text = segment.get('text', '')
                start_time = float(segment.get('start', 0))
                end_time = float(segment.get('end', 0))
                
                segments_text += "Segment {}:\n".format(segment_num)
                segments_text += "Speaker: {}\n".format(speaker)
                segments_text += "Text: {}\n".format(text)
                segments_text += "Time: {:.2f}s - {:.2f}s\n\n".format(start_time, end_time)
        except Exception as e:
            logger.error(f"Error formatting segments: {str(e)}")
            raise ValueError(f"Error preparing transcript segments: {str(e)}")
            
        # Build the prompt using string concatenation to avoid formatting issues
        full_text = transcript_data.get('text', '')
        
        base_prompt = """
        You are an AI assistant specialized in enhancing conversation transcripts. Your task is to analyze the following conversation transcript produced by automatic speech recognition with speaker diarization, and create an enhanced version with named speakers and corrected text.
        
        ## IMPORTANT RULES:
        - If a speaker's name is clearly mentioned in the conversation (e.g., "Hi, I'm John"), use that name instead of the generic label
        - If a speaker's name is not mentioned or unclear, keep the original speaker label
        - Do not invent names that aren't mentioned in the conversation
        - Correct obvious speech recognition errors but preserve the authentic style of spoken language
        - Do not add new content that wasn't in the original transcript
        - Do not remove any segments or significantly alter the meaning
        - Keep all original timing information exactly the same
        
        ## FIXING SPLIT SENTENCES:
        - Look for incomplete sentences that are split between consecutive speaker segments
        - If a sentence appears to be unnaturally split (e.g., "University of" then "Pennsylvania"), attribute the whole sentence to the most likely speaker
        - When fixing split sentences, use your judgment to determine which speaker was actually talking
        - Pay attention to context, question-answer patterns, and topic continuity
        
        ## ORIGINAL TRANSCRIPT:
        
        Full Text: 
        {}
        
        Segments:
        {}
        
        ## OUTPUT FORMAT:
        Return your response ONLY as a simple JSON list of enhanced segments. Each segment should be a JSON object with these fields:
        - "speaker": the identified speaker name or original speaker label if no name was identified
        - "text": the corrected text for this segment
        - "start": the start time in seconds (must match the original)
        - "end": the end time in seconds (must match the original)
        
        Example format:
        [
          {{
            "speaker": "John",
            "text": "Hi, my name is John. How are you today?",
            "start": 0.0,
            "end": 3.5
          }},
          {{
            "speaker": "Mary",
            "text": "I'm doing well, thank you.",
            "start": 3.7,
            "end": 5.2
          }}
        ]
        
        Ensure you use the actual names if mentioned, otherwise use the original speaker IDs. Return ONLY the JSON, with no additional text.
        """.format(full_text, segments_text)
        
        # Define schema for Gemini structured output
        segment_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string", "description": "The identified speaker name or original speaker label"},
                    "text": {"type": "string", "description": "The corrected text for this segment, with any split sentences fixed"},
                    "start": {"type": "number", "description": "The start time in seconds (must match original)"},
                    "end": {"type": "number", "description": "The end time in seconds (must match original)"}
                },
                "required": ["speaker", "text", "start", "end"]
            }
        }
        
        # Configure Gemini to return structured output
        segment_config = types.GenerateContentConfig(
            response_schema=segment_schema,
            response_mime_type='application/json',
            temperature=0.2
        )
        
        # Make the API call to Gemini with structured output
        logger.info("Making API call to Gemini with structured output schema")
        response = await client.aio.models.generate_content(
            model='gemini-2.0-flash',
            contents=[base_prompt],
            config=segment_config
        )
        
        # Parse the response - with structured output, we get JSON directly
        logger.info("Received structured response from Gemini")
        enhanced_segments = json.loads(response.text)
        
        # Post-process segments to identify and fix any remaining split sentences that Gemini might have missed
        processed_segments = _fix_split_sentences(enhanced_segments)
        
        # Build the enhanced transcript
        # Create valid EnhancedSegment objects from processed segments
        valid_segments = []
        full_text = []
        
        for segment in processed_segments:
            # Skip invalid segments
            if not isinstance(segment, dict):
                continue
                
            # Extract fields with validation
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "")
            
            try:
                start = float(segment.get("start", 0))
                end = float(segment.get("end", 0))
            except (ValueError, TypeError):
                start = 0
                end = 0
                
            # Add to valid segments
            valid_segments.append(EnhancedSegment(
                speaker=speaker,
                text=text,
                start=start,
                end=end
            ))
            
            # Build the full text
            full_text.append(text)
        
        # Create the enhanced transcript
        enhanced_transcript = EnhancedTranscript(
            transcript_id=transcript_data.get("transcript_id"),
            text=" ".join(full_text),
            segments=valid_segments,
            confidence=0.85  # Default confidence value
        )
        
        # Preserve the original transcript ID if it exists
        if "transcript_id" in transcript_data:
            enhanced_transcript.transcript_id = transcript_data["transcript_id"]
            
        logger.info(f"Successfully enhanced transcript with named speakers")
        return enhanced_transcript
        
    except Exception as e:
        logger.error(f"Error enhancing transcript: {str(e)}")
        # Get detailed exception info for debugging
        import traceback
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        # If enhancement fails, return a minimally enhanced version of the original
        return create_fallback_enhancement(transcript_data)

async def generate_summary(transcript_text: str, duration: int) -> ConversationSummary:
    """
    Generate detailed meeting notes with HTML formatting and a mermaid mindmap for a conversation transcript.
    
    Args:
        transcript_text: The full text of the enhanced transcript
        
    Returns:
        A ConversationSummary with HTML-formatted meeting notes and mermaid mindmap
    """
    try:
        duration = common.convert_seconds_to_readable(duration)
        
        logger.info("Generating conversation summary and mindmap with Gemini")
        
        # Get current date
        from datetime import datetime
        current_date = datetime.now().strftime('%B %d, %Y')
        
        # Create the prompt for meeting notes and mindmap generation
        base_prompt = f"""
        You are an AI assistant that creates detailed meeting notes and mindmaps of conversations. Your task is to analyze the following transcript and create two outputs:

        1. Detailed meeting notes in HTML format that captures:
           - The main topics discussed
           - Key points made by each speaker
           - Important details, decisions, or action items
           - The flow and context of the conversation
           - Format this as if a participant was taking notes during the meeting

        Your meeting notes should use HTML tags for formatting with this structure:
        <h2><strong>Conversation Title</strong></h2>
        <p><strong>Date:</strong> {current_date}</p>
        <p><strong>Duration:</strong> {duration}</p>
        
        <h3><strong>Summary:</strong></h3>
        <p>Brief overview of the conversation</p>
        
        <h3><strong>Key Points Discussed:</strong></h3>
        <h4><strong>Topic Category:</strong></h4>
        <ul>
          <li>Important point 1</li>
          <li>Important point 2</li>
        </ul>
        
        <h4><strong>Another Topic Category:</strong></h4>
        <ul>
          <li>Another point 1</li>
          <li>Another point 2</li>
        </ul>
        
        <h3><strong>Action Items:</strong></h3>
        <ul>
          <li>Action item 1</li>
          <li>Action item 2</li>
        </ul>
        
        <h3><strong>Participant Insights:</strong></h3>
        <p>Brief commentary about the participants' contributions</p>

        2. A mermaid mindmap structure that visually represents the conversation topics and subtopics.

        ## CONVERSATION TRANSCRIPT:
        {transcript_text}

        The duration of the transcript is {duration}. The date the transcript was recorded is {current_date}.

        ## OUTPUT FORMAT:
        Respond with TWO clearly separated sections:
        
        SECTION 1 - MEETING NOTES:
        <Begin your detailed HTML-formatted meeting notes here>
        
        SECTION 2 - MINDMAP NODES:
        <Provide a list of mindmap nodes in this format>
        1. Conversation (root)
        2. Topic 1 (parent: 1)
        3. Subtopic A (parent: 2)
        4. Subtopic B (parent: 2)
        5. Topic 2 (parent: 1)
        """
        
        # Make the API call to Gemini without a schema
        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2048
        )
        
        response = await client.aio.models.generate_content(
            model='gemini-2.0-flash',
            contents=[base_prompt],
            config=config
        )
        
        # Extract the text content
        response_text = response.text
        
        # Split the response into meeting notes and mindmap sections
        if "SECTION 1 - MEETING NOTES:" in response_text and "SECTION 2 - MINDMAP NODES:" in response_text:
            # Split by sections
            parts = response_text.split("SECTION 2 - MINDMAP NODES:")
            summary_part = parts[0].replace("SECTION 1 - MEETING NOTES:", "").strip()
            mindmap_part = parts[1].strip()
            
            # Parse the mindmap nodes
            mindmap_nodes = []
            node_lines = mindmap_part.split("\n")
            for line in node_lines:
                if not line.strip():
                    continue
                    
                # Parse lines like: "1. Conversation (root)" or "2. Topic 1 (parent: 1)"
                try:
                    # Extract the node number, label and parent
                    parts = line.split(".", 1)
                    if len(parts) < 2:
                        continue
                        
                    node_id = int(parts[0].strip())
                    label_part = parts[1].strip()
                    
                    # Check if there's parent info
                    parent_id = None
                    if "(parent:" in label_part:
                        label_and_parent = label_part.split("(parent:")
                        label = label_and_parent[0].strip()
                        parent_id_str = label_and_parent[1].replace(")", "").strip()
                        parent_id = int(parent_id_str)
                    elif "(root)" in label_part:
                        label = label_part.replace("(root)", "").strip()
                    else:
                        label = label_part
                    
                    # Add the node
                    mindmap_nodes.append(MindMapNode(
                        id=node_id,
                        label=label,
                        parent_id=parent_id
                    ))
                except Exception as e:
                    logger.warning(f"Error parsing mindmap node: {line}, error: {e}")
                    continue
            
            # If no nodes were parsed successfully, create a default root node
            if not mindmap_nodes:
                mindmap_nodes.append(MindMapNode(id=1, label="Conversation"))
        else:
            # Fallback if the expected sections aren't found
            summary_part = response_text
            mindmap_nodes = [MindMapNode(id=1, label="Conversation")]
        
        # Create and return the summary
        return ConversationSummary(
            summary=summary_part,
            mindmap=MindMap(nodes=mindmap_nodes)
        )
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        import traceback
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        
        # Return a fallback summary if generation fails
        return ConversationSummary(
            summary="Unable to generate summary. Please try again later.",
            mindmap=MindMap(nodes=[MindMapNode(id=1, label="Conversation")])
        )

def create_fallback_enhancement(transcript_data: Dict[str, Any]) -> EnhancedTranscript:
    """
    Creates a fallback enhancement if the Gemini API call fails.
    Simply copies the original transcript data with minimal changes.
    """
    segments = []
    speakers = set()
    for segment in transcript_data.get("segments", []):
        speaker = segment.get("speaker", "Unknown")
        speakers.add(speaker)
        segments.append(EnhancedSegment(
            speaker=speaker,
            text=segment.get("text", ""),
            start=float(segment.get("start", 0)),
            end=float(segment.get("end", 0))
        ))
    
    return EnhancedTranscript(
        transcript_id=transcript_data.get("transcript_id"),
        text=transcript_data.get("text", ""),
        segments=segments,
        confidence=0.0
    )
