import logging
import tempfile
import os
import asyncio
import shutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path

try:
    import torch
    import whisper
    from pyannote.audio import Pipeline
    import librosa
    import soundfile as sf
    import ffmpeg
    from pydub import AudioSegment

    TRANSCRIPTION_AVAILABLE = True
    DIARIZATION_AVAILABLE = True
except ImportError:
    TRANSCRIPTION_AVAILABLE = False
    DIARIZATION_AVAILABLE = False
    logging.warning(
        "Transcription or diarization dependencies not installed. Install with: pip install -r requirements_audio.txt"
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DiarizedTranscript:
    """Represents a transcript with speaker information"""

    segments: List[Dict[str, Any]]
    speakers: List[str]
    total_duration: float
    confidence_score: float


class TranscriptionService:
    """Service for transcribing audio files using Whisper"""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Whisper model"""
        if not TRANSCRIPTION_AVAILABLE:
            logger.error("Transcription dependencies not available")
            return

        try:
            logger.info(f"Initializing Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            
            # Use GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to(torch.device("cuda"))
                logger.info("Using GPU for transcription")
            else:
                logger.info("Using CPU for transcription")
                
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {str(e)}")
            self.model = None

    def is_available(self) -> bool:
        """Check if transcription is available"""
        return TRANSCRIPTION_AVAILABLE and self.model is not None

    async def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio file using Whisper"""
        if not self.is_available():
            raise RuntimeError("Transcription not available. Install dependencies first.")

        logger.info(f"Starting transcription for: {audio_path}")
        
        # Ensure the path is absolute and properly normalized
        abs_audio_path = os.path.abspath(os.path.normpath(audio_path))
        logger.info(f"Using normalized path: {abs_audio_path}")
        
        # Verify file exists and is readable
        if not os.path.isfile(abs_audio_path):
            raise FileNotFoundError(f"Audio file not found: {abs_audio_path}")
            
        # Load audio data into memory first
        try:
            logger.info("Loading audio file into memory")
            # For whisper, it's better to load the file directly without preprocessing
            # This helps avoid file path issues
            audio_data = abs_audio_path
            
            # Run transcription in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self.model.transcribe(audio_data, word_timestamps=True)
            )

            logger.info(f"Transcription completed: {len(result['text'])} characters")
            return result
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise


class SpeakerDiarizationService:
    """Service for performing speaker diarization on audio files."""

    def __init__(self, model_name: str = "pyannote/speaker-diarization-3.1", hf_token: Optional[str] = None):
        self.model_name = model_name
        self.hf_token = hf_token
        self.pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the diarization pipeline"""
        if not DIARIZATION_AVAILABLE:
            logger.error("Speaker diarization dependencies not available")
            return

        try:
            logger.info(f"Initializing speaker diarization pipeline: {self.model_name}")

            self.pipeline = Pipeline.from_pretrained(
                self.model_name, use_auth_token=self.hf_token
            )

            # Use GPU if available
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to(torch.device("cuda"))
                logger.info("Using GPU for speaker diarization")
            else:
                logger.info("Using CPU for speaker diarization")

        except Exception as e:
            logger.error(f"Failed to initialize diarization pipeline: {str(e)}")
            self.pipeline = None

    def is_available(self) -> bool:
        """Check if speaker diarization is available"""
        return DIARIZATION_AVAILABLE and self.pipeline is not None

    async def diarize_audio(self, audio_path: str, num_speakers: Optional[int] = None) -> DiarizedTranscript:
        """Perform speaker diarization on an audio file."""
        if not self.is_available():
            raise RuntimeError("Speaker diarization not available. Install dependencies first.")

        logger.info(f"Starting speaker diarization for: {audio_path}")
        
        # Ensure the path is absolute and properly normalized
        abs_audio_path = os.path.abspath(os.path.normpath(audio_path))
        logger.info(f"Using normalized path: {abs_audio_path}")
        
        # Verify file exists and is readable
        if not os.path.isfile(abs_audio_path):
            raise FileNotFoundError(f"Audio file not found: {abs_audio_path}")
            
        try:
            logger.info("Verifying audio file access")
            with open(abs_audio_path, 'rb') as f:
                # Just check we can read it
                f.read(1024)

            # Run in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Preprocess audio (in thread pool)
            logger.info("Preprocessing audio file")
            processed_audio_path = await loop.run_in_executor(
                None, lambda: self._preprocess_audio(abs_audio_path)
            )
            logger.info(f"Audio preprocessed: {processed_audio_path}")

            # Perform diarization (in thread pool)
            logger.info("Running diarization pipeline")
            diarization = await loop.run_in_executor(
                None, lambda: self.pipeline(processed_audio_path, num_speakers=num_speakers)
            )

            # Process results
            segments = []
            speakers = set()

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = {
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                    "duration": turn.end - turn.start,
                }
                segments.append(segment)
                speakers.add(speaker)

            # Calculate total duration
            total_duration = max([seg["end"] for seg in segments]) if segments else 0.0

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(segments)

            result = DiarizedTranscript(
                segments=segments,
                speakers=sorted(list(speakers)),
                total_duration=total_duration,
                confidence_score=confidence_score,
            )

            logger.info(f"Diarization completed: {len(speakers)} speakers, {len(segments)} segments")
            
            # Clean up temporary file
            if processed_audio_path != abs_audio_path and os.path.exists(processed_audio_path):
                os.unlink(processed_audio_path)
                
            return result

        except Exception as e:
            logger.error(f"Error during speaker diarization: {str(e)}")
            logger.error(f"Audio path that failed: {abs_audio_path}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _preprocess_audio(self, audio_path: str) -> str:
        """Preprocess audio file for better diarization results."""
        try:
            logger.info(f"Preprocessing audio: {audio_path}")
            
            # Verify file exists before attempting to load it
            if not os.path.isfile(audio_path):
                raise FileNotFoundError(f"Audio file not found for preprocessing: {audio_path}")
                
            # Load audio with librosa - this can handle various audio formats
            logger.info("Loading audio with librosa")
            audio, sr = librosa.load(audio_path, sr=16000)  # Resample to 16kHz

            # Apply basic preprocessing
            logger.info("Applying audio preprocessing")
            # Remove silence at beginning and end
            audio, _ = librosa.effects.trim(audio, top_db=20)

            # Normalize audio
            audio = librosa.util.normalize(audio)

            # Create a consistent temp directory
            temp_dir = os.path.join(os.getcwd(), "temp_audio")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save processed audio to a named temporary file
            temp_filename = f"processed_{os.path.basename(audio_path)}.wav"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            logger.info(f"Saving preprocessed audio to: {temp_path}")
            sf.write(temp_path, audio, sr)
            
            return temp_path

        except Exception as e:
            logger.warning(f"Audio preprocessing failed, using original: {str(e)}")
            import traceback
            logger.warning(f"Preprocessing traceback: {traceback.format_exc()}")
            return audio_path

    def _calculate_confidence_score(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate a confidence score for the diarization results."""
        if not segments:
            return 0.0

        # Simple heuristic: longer segments and fewer speaker changes indicate higher confidence
        total_duration = sum(seg["duration"] for seg in segments)
        avg_segment_duration = total_duration / len(segments)

        # Normalize based on typical speech patterns
        # Segments of 2-10 seconds are considered optimal
        if avg_segment_duration >= 2.0:
            duration_score = min(1.0, avg_segment_duration / 10.0)
        else:
            duration_score = avg_segment_duration / 2.0

        # Factor in number of speaker changes
        speaker_changes = 0
        for i in range(1, len(segments)):
            if segments[i]["speaker"] != segments[i - 1]["speaker"]:
                speaker_changes += 1

        change_ratio = speaker_changes / len(segments) if segments else 0
        change_score = max(0.0, 1.0 - change_ratio * 2)  # Penalize too many changes

        # Combine scores
        confidence = duration_score * 0.6 + change_score * 0.4
        return min(1.0, max(0.0, confidence))


def convert_audio_to_mp3(audio_path: str) -> str:
    """
    Convert any audio file to MP3 format for consistent processing.
    
    Args:
        audio_path: Path to the original audio file
        
    Returns:
        Path to the converted MP3 file
    """
    try:
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.getcwd(), "temp_audio")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Get file info
        original_path = Path(audio_path)
        file_stem = original_path.stem
        file_ext = original_path.suffix.lower()
        
        # If already an MP3 file, return the original path
        if file_ext == ".mp3":
            logger.info(f"File is already in MP3 format: {audio_path}")
            return audio_path
            
        # Create output path for the MP3 file
        mp3_filename = f"{file_stem}.mp3"
        mp3_path = os.path.join(temp_dir, mp3_filename)
        
        logger.info(f"Converting {file_ext} file to MP3: {mp3_path}")
        
        # Try using pydub for conversion (handles most formats)
        try:
            # Load audio file with pydub (automatically detects format)
            audio = AudioSegment.from_file(audio_path)
            # Export as MP3
            audio.export(mp3_path, format="mp3")
            logger.info(f"Successfully converted to MP3 using pydub: {mp3_path}")
            return mp3_path
        except Exception as e:
            logger.warning(f"Pydub conversion failed: {str(e)}. Trying ffmpeg...")
            
            # Fallback to ffmpeg directly
            try:
                # Use ffmpeg to convert the file
                (ffmpeg
                    .input(audio_path)
                    .output(mp3_path, acodec='libmp3lame', ac=1, ar='16k')
                    .run(capture_stdout=True, capture_stderr=True, overwrite_output=True))
                logger.info(f"Successfully converted to MP3 using ffmpeg: {mp3_path}")
                return mp3_path
            except Exception as ffmpeg_error:
                logger.error(f"FFmpeg conversion failed: {str(ffmpeg_error)}")
                # If all conversions fail, return original path
                return audio_path
    except Exception as e:
        logger.error(f"Error converting audio file: {str(e)}")
        return audio_path

async def process_audio_file(audio_path: str, hf_token: Optional[str] = None) -> Dict[str, Any]:
    """Process audio file for transcription and speaker diarization."""
    try:
        # Convert audio to MP3 format if needed
        logger.info(f"Processing audio file: {audio_path}")
        mp3_audio_path = convert_audio_to_mp3(audio_path)
        logger.info(f"Using audio file for processing: {mp3_audio_path}")
        
        # Initialize services
        transcription_service = TranscriptionService(model_name="base")
        diarization_service = SpeakerDiarizationService(hf_token=hf_token)
        
        # Check if services are available
        if not transcription_service.is_available():
            raise RuntimeError("Transcription service not available")
            
        if not diarization_service.is_available():
            raise RuntimeError("Speaker diarization service not available")
        
        # Run transcription and diarization concurrently
        transcription_task = asyncio.create_task(transcription_service.transcribe_audio(mp3_audio_path))
        diarization_task = asyncio.create_task(diarization_service.diarize_audio(mp3_audio_path))
        
        # Wait for both tasks to complete
        transcription_result = await transcription_task
        diarization_result = await diarization_task
        
        # Combine results
        result = await combine_transcription_with_diarization(
            transcription_result, diarization_result
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        raise


async def combine_transcription_with_diarization(
    transcription: Dict[str, Any], diarization: DiarizedTranscript
) -> Dict[str, Any]:
    """Combine transcription and diarization results."""
    try:
        logger.info("Combining transcription and diarization results")
        
        # Get word-level timestamps from transcription
        word_timestamps = transcription.get("segments", [])
        words_with_timestamps = []
        
        for segment in word_timestamps:
            for word in segment.get("words", []):
                words_with_timestamps.append({
                    "word": word.get("word", ""),
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                })
        
        # Map words to speakers
        diarized_segments = []
        current_speaker = None
        current_text = []
        current_start = None
        
        for word_info in words_with_timestamps:
            word = word_info.get("word", "")
            start_time = word_info.get("start", 0)
            end_time = word_info.get("end", 0)
            
            # Find which speaker is active at this time
            speaker = _find_speaker_at_time(start_time, diarization.segments)
            
            if speaker != current_speaker:
                # Speaker change detected
                if current_text and current_speaker:
                    diarized_segments.append({
                        "speaker": current_speaker,
                        "text": "".join(current_text).strip(),
                        "start": current_start,
                        "end": start_time,
                    })
                    
                current_speaker = speaker
                current_text = [word]
                current_start = start_time
            else:
                current_text.append(word)
        
        # Add final segment
        if current_text and current_speaker:
            diarized_segments.append({
                "speaker": current_speaker,
                "text": "".join(current_text).strip(),
                "start": current_start,
                "end": words_with_timestamps[-1].get("end", 0) if words_with_timestamps else 0,
            })
        
        # Return combined result
        return {
            "transcript_id": None,  # Will be set by the database
            "text": transcription.get("text", ""),
            "segments": diarized_segments,
            "speakers": diarization.speakers,
            "confidence": diarization.confidence_score,
        }
        
    except Exception as e:
        logger.error(f"Error combining transcription and diarization: {str(e)}")
        raise


def _find_speaker_at_time(timestamp: float, segments: List[Dict[str, Any]]) -> str:
    """Find which speaker is active at a given timestamp."""
    for segment in segments:
        if segment["start"] <= timestamp <= segment["end"]:
            return segment["speaker"]
    
    # If no exact match, find closest segment
    if segments:
        closest_segment = min(segments, key=lambda s: min(
            abs(s["start"] - timestamp), 
            abs(s["end"] - timestamp)
        ))
        return closest_segment["speaker"]
    
    return "SPEAKER_UNKNOWN"
