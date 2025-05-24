import logging
import uuid
import os
from fastapi import APIRouter, status, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from modules import database
from utils import transcription

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class ConversationUpload(BaseModel):
    audio_file: bytes

class ConversationUploadResponse(BaseModel):
    upload_id: str
    status: str

@router.post(
    "/upload",
    response_model=ConversationUploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_conversation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    upload_id = str(uuid.uuid4())
    logger.info(f"Received audio upload with ID: {upload_id}")
    
    upload_dir = os.path.join(os.getcwd(), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, f"{upload_id}_{file.filename}")
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    await database.store_upload(upload_id, file_path)
    
    background_tasks.add_task(process_audio, upload_id, file_path)
    
    return {"upload_id": upload_id, "status": "uploaded"}


async def process_audio(upload_id: str, file_path: str):
    """Process the audio file asynchronously after upload."""
    try:
        logger.info(f"Starting audio processing for upload: {upload_id}")
        
        # Update status to processing
        await database.update_upload_status(upload_id, "processing")
        
        # Step 1: Transcription and Speaker Diarization
        logger.info(f"Starting transcription and diarization for upload: {upload_id}")
        
        # Get HF token from environment if available
        hf_token = os.getenv("HF_TOKEN")
        
        # Process audio file with transcription and diarization
        transcript_data = await transcription.process_audio_file(file_path, hf_token=hf_token)
        
        # Set transcript ID to match upload ID for easier reference
        transcript_data["transcript_id"] = upload_id
        
        # Store transcript data in database
        await database.store_transcript(upload_id, transcript_data)
        
        # Update status to indicate transcription is complete
        await database.update_upload_status(upload_id, "transcribed")
        
        # Step 2: Summarization and Mindmap Generation
        # TODO: Implement summary and mindmap generation
        # summary_data = await generate_summary_and_mindmap(transcript_data)
        # await database.store_summary(upload_id, summary_data)
        
        # Step 3: Contact Inference and LinkedIn Lookup
        # TODO: Implement contact inference and LinkedIn lookup via Dex MCP
        # contact_data = await perform_contact_inference(transcript_data)
        # await database.store_contact_data(upload_id, contact_data)
        
        logger.info(f"Completed audio processing for upload: {upload_id}")
        await database.update_upload_status(upload_id, "processed")
        
    except Exception as e:
        logger.error(f"Error processing audio for upload {upload_id}: {str(e)}")
        await database.update_upload_status(upload_id, "error", error_message=str(e))