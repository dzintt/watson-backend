import os
import logging
import asyncio
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

# Import for audio processing
import librosa

# Import for transcription
import whisper
from whisper.audio import pad_or_trim, log_mel_spectrogram

# Import for speaker diarization
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Global model instances (loaded only once for efficiency)
_whisper_model = None
_diarization_pipeline = None

# Supported audio formats
SUPPORTED_FORMATS = [".wav", ".mp3", ".m4a"]

def get_whisper_model():
    """Lazy loading of Whisper model to save memory."""
    global _whisper_model
    if _whisper_model is None:
        model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
        logger.info(f"Loading Whisper model (size: {model_size})")
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model

def get_diarization_pipeline():
    """Lazy loading of Pyannote diarization pipeline."""
    global _diarization_pipeline
    if _diarization_pipeline is None:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is required for speaker diarization")
            
        logger.info("Loading pyannote.audio diarization pipeline")
        _diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Use GPU if available
        if torch.cuda.is_available():
            _diarization_pipeline = _diarization_pipeline.to(torch.device("cuda"))
            
    return _diarization_pipeline

async def load_audio_with_librosa(audio_path: str) -> np.ndarray:
    """Load audio file using librosa."""
    logger.info(f"Loading audio file with librosa: {audio_path}")
    loop = asyncio.get_event_loop()
    
    # Load and preprocess audio file in a separate thread
    audio, sr = await loop.run_in_executor(
        None,
        lambda: librosa.load(audio_path, sr=16000, mono=True)
    )
    
    return audio

async def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio file using Whisper with librosa for audio loading."""
    logger.info(f"Transcribing audio file: {audio_path}")
    
    # Load audio using librosa
    audio = await load_audio_with_librosa(audio_path)
    
    # Run transcription in a separate thread to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    model = get_whisper_model()
    
    # Preprocess audio for Whisper
    def process_and_transcribe():
        # Pad or trim audio to 30 seconds (required by Whisper)
        audio_padded = pad_or_trim(audio)
        
        # Generate log-mel spectrogram
        mel = log_mel_spectrogram(audio_padded)
        
        # Transcribe with Whisper model
        result = model.transcribe(
            audio, 
            verbose=False,
            condition_on_previous_text=True,
            initial_prompt="This is an audio conversation with multiple speakers."
        )
        
        return result
    
    # Offload the CPU-intensive transcription to a thread pool
    result = await loop.run_in_executor(None, process_and_transcribe)
    
    logger.info(f"Transcription completed for {audio_path}")
    return result

async def perform_diarization(audio_path: str) -> dict:
    """Perform speaker diarization on an audio file."""
    logger.info(f"Performing speaker diarization on {audio_path}")
    
    # Run diarization in a separate thread to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    pipeline = get_diarization_pipeline()
    
    # Offload the CPU/GPU-intensive diarization to a thread pool
    diarization = await loop.run_in_executor(
        None,
        lambda: pipeline(audio_path)
    )
    
    logger.info(f"Diarization completed for {audio_path}")
    return diarization

def map_transcription_to_speakers(transcription: dict, diarization) -> List[dict]:
    """Map transcribed segments to speakers identified in diarization."""
    transcript_segments = []
    
    # Process each segment from the transcription
    for segment in transcription["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"].strip()
        
        # Create a segment object for the current transcription segment
        current_segment = Segment(start_time, end_time)
        
        # Find which speaker was active during this segment
        speaker_times = {}
        
        # Iterate through diarization results
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # If there's an overlap between this speaker turn and our segment
            overlap = current_segment.intersect(turn)
            
            if overlap:
                # Calculate the duration of overlap
                overlap_duration = overlap.duration
                
                # Add or update the speaker's time
                if speaker in speaker_times:
                    speaker_times[speaker] += overlap_duration
                else:
                    speaker_times[speaker] = overlap_duration
        
        # Determine the dominant speaker for this segment
        if speaker_times:
            dominant_speaker = max(speaker_times, key=speaker_times.get)
        else:
            dominant_speaker = "Unknown Speaker"
        
        # Add the segment with speaker information
        transcript_segments.append({
            "speaker": dominant_speaker,
            "start": start_time,
            "end": end_time,
            "text": text
        })
    
    return transcript_segments

async def perform_transcription_and_diarization(file_path: str) -> dict:
    """Main function to perform both transcription and speaker diarization."""
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {file_extension}. Supported formats: {SUPPORTED_FORMATS}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Perform transcription and diarization concurrently
        transcription_task = asyncio.create_task(transcribe_audio(file_path))
        diarization_task = asyncio.create_task(perform_diarization(file_path))
        
        # Wait for both tasks to complete
        transcription = await transcription_task
        diarization = await diarization_task
        
        # Map transcription segments to speakers
        transcript_with_speakers = map_transcription_to_speakers(transcription, diarization)
        
        # Create the final transcript data structure
        transcript_data = {
            "transcript_id": Path(file_path).stem,  # Use filename without extension as ID
            "text": transcript_with_speakers,
            "metadata": {
                "duration": transcription.get("duration", 0),
                "language": transcription.get("language", "en"),
                "processed_at": datetime.utcnow().isoformat()
            }
        }
        
        return transcript_data
        
    except Exception as e:
        logger.error(f"Error in transcription and diarization: {str(e)}")
        raise
