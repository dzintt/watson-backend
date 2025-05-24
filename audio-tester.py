#!/usr/bin/env python

"""
Audio Transcription and Speaker Diarization Tester

This script tests the transcription and speaker diarization functionality
without having to go through the API endpoints.

Usage:
    python audio-tester.py [audio_file_path]

If no audio file path is provided, it will use the default test_audio.mp3.
"""

import asyncio
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Import the transcription module
from utils import transcription

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


async def test_transcription_and_diarization(audio_path: str):
    """Test the transcription and speaker diarization on a given audio file."""
    try:
        logger.info(f"Starting test on audio file: {audio_path}")
        start_time = datetime.now()
        
        # Get the Hugging Face token from environment variables
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning("No HF_TOKEN found in environment variables. Diarization might fail.")
        
        # Process the audio file
        logger.info("Processing audio file...")
        result = await transcription.process_audio_file(audio_path, hf_token=hf_token)
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Log results
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        logger.info(f"Detected {len(result.get('speakers', []))} speakers")
        logger.info(f"Generated {len(result.get('segments', []))} speech segments")
        
        # Create output directory for results
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save results to JSON file
        output_file = output_dir / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # Display sample of the results
        print("\n" + "=" * 50)
        print("TRANSCRIPTION AND DIARIZATION RESULTS:")
        print("=" * 50)
        print(f"Full transcript: {result.get('text', '')[:200]}...")
        print("\nSpeaker segments:")
        
        # Print first few speaker segments
        for i, segment in enumerate(result.get('segments', [])[:5]):
            print(f"[{segment.get('speaker')}] {segment.get('start'):.2f}s - {segment.get('end'):.2f}s: {segment.get('text')}")
        
        if len(result.get('segments', [])) > 5:
            print(f"... and {len(result.get('segments', [])) - 5} more segments")
            
        print("\n" + "=" * 50)
        print(f"Confidence score: {result.get('confidence', 0):.2f}")
        print("=" * 50)
        
        return result
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        raise


async def test_individual_components(audio_path: str):
    """Test the individual components of the transcription and diarization pipeline."""
    try:
        logger.info("Testing individual components...")
        
        # Test TranscriptionService
        logger.info("Testing transcription service...")
        transcription_service = transcription.TranscriptionService(model_name="base")
        
        if not transcription_service.is_available():
            logger.error("Transcription service is not available. Check dependencies.")
            return
            
        transcription_result = await transcription_service.transcribe_audio(audio_path)
        logger.info(f"Transcription successful: {len(transcription_result.get('text', ''))} characters")
        
        # Test SpeakerDiarizationService
        logger.info("Testing speaker diarization service...")
        hf_token = os.getenv("HF_TOKEN")
        diarization_service = transcription.SpeakerDiarizationService(hf_token=hf_token)
        
        if not diarization_service.is_available():
            logger.error("Speaker diarization service is not available. Check dependencies.")
            return
            
        diarization_result = await diarization_service.diarize_audio(audio_path)
        logger.info(f"Diarization successful: {len(diarization_result.speakers)} speakers detected")
        
        logger.info("Individual component tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error during individual component test: {str(e)}")
        raise


async def main():
    # Get audio file path from command line args or use default
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = os.path.join(os.getcwd(), "test_audio.mp3")
    
    # Check if file exists
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        
        # Try an alternative path (for different working directory scenarios)
        alternative_path = os.path.abspath("test_audio.mp3")
        if os.path.exists(alternative_path):
            logger.info(f"Found audio file at alternative path: {alternative_path}")
            audio_path = alternative_path
        else:
            logger.error("Could not find test_audio.mp3 in any location")
            return
    
    logger.info(f"Using audio file: {audio_path}")
    
    # Verify file is readable
    try:
        with open(audio_path, 'rb') as f:
            # Just read a small chunk to verify access
            f.read(1024)
        logger.info("Audio file is accessible and readable")
    except Exception as e:
        logger.error(f"Cannot access audio file: {str(e)}")
        return
    
    try:
        logger.info("Running test suite for audio transcription and diarization")
        
        # Test individual components first
        await test_individual_components(audio_path)
        
        # Then test the full pipeline
        await test_transcription_and_diarization(audio_path)
        
        logger.info("All tests completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
