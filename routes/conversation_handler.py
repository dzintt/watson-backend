import logging
import uuid
import os
from fastapi import APIRouter, status, UploadFile, File, BackgroundTasks, HTTPException, Path
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

class TranscriptSummary(BaseModel):
    summary: str
    mindmap: dict
    
class ContactInference(BaseModel):
    contact: dict

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


@router.get(
    "/summary/{transcript_id}",
    response_model=TranscriptSummary,
    status_code=status.HTTP_200_OK,
)
async def get_summary(transcript_id: str = Path(..., description="The transcript ID to generate or retrieve a summary for")):
    """Get or generate a summary and mindmap for a transcript"""
    try:
        # First, check if summary already exists
        existing_summary = await database.get_summary_by_upload_id(transcript_id)
        if existing_summary:
            logger.info(f"Found existing summary for transcript {transcript_id}")
            return {
                "summary": existing_summary.get("summary", ""),
                "mindmap": existing_summary.get("mindmap", {"nodes": [], "edges": []})
            }
            
        # If no summary exists, check if transcript exists
        transcript = await database.get_transcript_by_id(transcript_id)
        if not transcript:
            # Try looking up by upload_id as well
            transcript = await database.get_transcript_by_upload_id(transcript_id)
            
        if not transcript:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transcript with ID {transcript_id} not found"
            )
            
        # TODO: Implement actual summarization and mindmap generation
        # For now, return a placeholder
        logger.info(f"Generating placeholder summary for transcript {transcript_id}")
        summary_data = {
            "summary": "Summary generation is not yet implemented. This is a placeholder.",
            "mindmap": {
                "nodes": [{ "id": 1, "label": "Conversation" }],
                "edges": []
            }
        }
        
        # Store the summary
        await database.store_summary(transcript_id, summary_data)
        
        return summary_data
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error generating summary for transcript {transcript_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating summary: {str(e)}"
        )

@router.get(
    "/contact-inference/{transcript_id}",
    response_model=ContactInference,
    status_code=status.HTTP_200_OK,
)
async def get_contact_inference(transcript_id: str = Path(..., description="The transcript ID to perform contact inference on")):
    """Extract contact information and perform LinkedIn lookup"""
    try:
        # First, check if contact info already exists
        existing_contact = await database.get_contact_by_upload_id(transcript_id)
        if existing_contact:
            logger.info(f"Found existing contact data for transcript {transcript_id}")
            return {
                "contact": existing_contact.get("contact", {})
            }
            
        # If no contact info exists, check if transcript exists
        transcript = await database.get_transcript_by_id(transcript_id)
        if not transcript:
            # Try looking up by upload_id as well
            transcript = await database.get_transcript_by_upload_id(transcript_id)
            
        if not transcript:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transcript with ID {transcript_id} not found"
            )
            
        # TODO: Implement actual contact inference and LinkedIn lookup
        # For now, return a placeholder
        logger.info(f"Generating placeholder contact data for transcript {transcript_id}")
        contact_data = {
            "contact": {
                "name": "Contact extraction not yet implemented",
                "title": "Placeholder",
                "company": "Example Corp",
                "linkedin_url": "https://linkedin.com/"
            }
        }
        
        # Store the contact data
        await database.store_contact_data(transcript_id, contact_data)
        
        return contact_data
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error performing contact inference for transcript {transcript_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing contact inference: {str(e)}"
        )

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
        # For now, we'll generate a placeholder summary
        logger.info(f"Generating placeholder summary for upload: {upload_id}")
        summary_data = {
            "summary": "Automatic summary generation is not yet implemented.",
            "mindmap": {
                "nodes": [{ "id": 1, "label": "Conversation" }],
                "edges": []
            }
        }
        await database.store_summary(upload_id, summary_data)
        await database.update_upload_status(upload_id, "summarized")
        
        # Step 3: Contact Inference and LinkedIn Lookup
        # TODO: Implement contact inference and LinkedIn lookup via Dex MCP
        # For now, we'll generate a placeholder contact
        logger.info(f"Generating placeholder contact data for upload: {upload_id}")
        contact_data = {
            "contact": {
                "name": "Contact extraction not yet implemented",
                "title": "Placeholder",
                "company": "Example Corp",
                "linkedin_url": "https://linkedin.com/"
            }
        }
        await database.store_contact_data(upload_id, contact_data)
        await database.update_upload_status(upload_id, "contact_identified")
        
        logger.info(f"Completed audio processing for upload: {upload_id}")
        await database.update_upload_status(upload_id, "processed")
        
    except Exception as e:
        logger.error(f"Error processing audio for upload {upload_id}: {str(e)}")
        await database.update_upload_status(upload_id, "error", error_message=str(e))