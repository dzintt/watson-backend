from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from dotenv import load_dotenv
from uuid import uuid4
import time
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# MongoDB connection setup
client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
database = client[os.getenv("DATABASE_NAME")]

# Define collections
uploads = database["uploads"]
transcripts = database["transcripts"]
summaries = database["summaries"]
contacts = database["contacts"]

# Upload related functions
async def store_upload(upload_id: str, file_path: str) -> str:
    """
    Store information about an uploaded audio file.
    
    Args:
        upload_id: Unique identifier for the upload
        file_path: Path to the stored audio file
        
    Returns:
        The upload ID
    """
    try:
        upload_doc = {
            "_id": upload_id,
            "filename": os.path.basename(file_path),
            "status": "uploaded",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "processing_history": [
                {"status": "uploaded", "timestamp": datetime.utcnow()}
            ]
        }
        
        await uploads.insert_one(upload_doc)
        logger.info(f"Stored upload with ID: {upload_id}")
        return upload_id
    except Exception as e:
        logger.error(f"Error storing upload {upload_id}: {str(e)}")
        raise

async def update_upload_status(upload_id: str, status: str, error_message: str = None) -> None:
    """
    Update the status of an upload and add to processing history.
    
    Args:
        upload_id: Unique identifier for the upload
        status: New status value
        error_message: Optional error message if status is 'error'
    """
    try:
        now = datetime.utcnow()
        update_data = {
            "status": status,
            "updated_at": now,
            "$push": {
                "processing_history": {
                    "status": status,
                    "timestamp": now
                }
            }
        }
        
        if error_message:
            update_data["error_message"] = error_message
            # Add error to history too
            update_data["$push"]["processing_history"]["error_message"] = error_message
        
        await uploads.update_one(
            {"_id": upload_id},
            {"$set": {k: v for k, v in update_data.items() if k != "$push"}}
        )
        
        # Add to history in a separate operation
        if "$push" in update_data:
            await uploads.update_one(
                {"_id": upload_id},
                {"$push": update_data["$push"]}
            )
            
        logger.info(f"Updated upload {upload_id} status to {status}")
    except Exception as e:
        logger.error(f"Error updating upload status for {upload_id}: {str(e)}")
        raise

async def get_upload_by_id(upload_id: str) -> dict:
    """
    Get upload information by ID.
    
    Args:
        upload_id: Unique identifier for the upload
        
    Returns:
        The upload document or None if not found
    """
    try:
        upload_doc = await uploads.find_one({"_id": upload_id})
        return upload_doc
    except Exception as e:
        logger.error(f"Error retrieving upload {upload_id}: {str(e)}")
        raise

# Transcript related functions
async def store_transcript(upload_id: str, transcript_data: dict) -> str:
    """
    Store the transcript data for an upload.
    
    Args:
        upload_id: Unique identifier for the upload
        transcript_data: Transcript data including speaker segments
        
    Returns:
        The transcript document ID
    """
    try:
        # Generate a transcript ID if not provided
        transcript_id = transcript_data.get("transcript_id") or str(uuid4())
        
        # Prepare the transcript document
        transcript_doc = {
            "_id": transcript_id,
            "upload_id": upload_id,
            "text": transcript_data.get("text", ""),
            "segments": transcript_data.get("segments", []),
            "speakers": transcript_data.get("speakers", []),
            "confidence": transcript_data.get("confidence", 0),
            "created_at": datetime.utcnow()
        }
        
        # Insert the transcript document
        await transcripts.insert_one(transcript_doc)
        logger.info(f"Stored transcript for upload {upload_id} with ID {transcript_id}")
        return transcript_id
    except Exception as e:
        logger.error(f"Error storing transcript for upload {upload_id}: {str(e)}")
        raise

async def get_transcript_by_id(transcript_id: str) -> dict:
    """
    Get transcript information by ID.
    
    Args:
        transcript_id: Unique identifier for the transcript
        
    Returns:
        The transcript document or None if not found
    """
    try:
        transcript_doc = await transcripts.find_one({"_id": transcript_id})
        return transcript_doc
    except Exception as e:
        logger.error(f"Error retrieving transcript {transcript_id}: {str(e)}")
        raise

async def get_transcript_by_upload_id(upload_id: str) -> dict:
    """
    Get transcript information by upload ID.
    
    Args:
        upload_id: Unique identifier for the upload
        
    Returns:
        The transcript document or None if not found
    """
    try:
        transcript_doc = await transcripts.find_one({"upload_id": upload_id})
        return transcript_doc
    except Exception as e:
        logger.error(f"Error retrieving transcript for upload {upload_id}: {str(e)}")
        raise

# Summary related functions
async def store_summary(upload_id: str, summary_data: dict) -> str:
    """
    Store the summary and mindmap data for an upload.
    
    Args:
        upload_id: Unique identifier for the upload
        summary_data: Summary and mindmap data
        
    Returns:
        The summary document ID
    """
    try:
        summary_id = str(uuid4())
        
        # Prepare the summary document
        summary_doc = {
            "_id": summary_id,
            "upload_id": upload_id,
            "summary": summary_data.get("summary", ""),
            "mindmap": summary_data.get("mindmap", {"nodes": [], "edges": []}),
            "created_at": datetime.utcnow()
        }
        
        # Insert the summary document
        await summaries.insert_one(summary_doc)
        logger.info(f"Stored summary for upload {upload_id} with ID {summary_id}")
        return summary_id
    except Exception as e:
        logger.error(f"Error storing summary for upload {upload_id}: {str(e)}")
        raise

async def get_summary_by_upload_id(upload_id: str) -> dict:
    """
    Get summary information by upload ID.
    
    Args:
        upload_id: Unique identifier for the upload
        
    Returns:
        The summary document or None if not found
    """
    try:
        summary_doc = await summaries.find_one({"upload_id": upload_id})
        return summary_doc
    except Exception as e:
        logger.error(f"Error retrieving summary for upload {upload_id}: {str(e)}")
        raise

# Contact inference related functions
async def store_contact_data(upload_id: str, contact_data: dict) -> str:
    """
    Store the contact inference data for an upload.
    
    Args:
        upload_id: Unique identifier for the upload
        contact_data: Contact information data
        
    Returns:
        The contact document ID
    """
    try:
        contact_id = str(uuid4())
        
        # Prepare the contact document
        contact_doc = {
            "_id": contact_id,
            "upload_id": upload_id,
            "contact": contact_data.get("contact", {}),
            "raw_entities": contact_data.get("raw_entities", []),
            "created_at": datetime.utcnow()
        }
        
        # Insert the contact document
        await contacts.insert_one(contact_doc)
        logger.info(f"Stored contact data for upload {upload_id} with ID {contact_id}")
        return contact_id
    except Exception as e:
        logger.error(f"Error storing contact data for upload {upload_id}: {str(e)}")
        raise

async def get_contact_by_upload_id(upload_id: str) -> dict:
    """
    Get contact information by upload ID.
    
    Args:
        upload_id: Unique identifier for the upload
        
    Returns:
        The contact document or None if not found
    """
    try:
        contact_doc = await contacts.find_one({"upload_id": upload_id})
        return contact_doc
    except Exception as e:
        logger.error(f"Error retrieving contact data for upload {upload_id}: {str(e)}")
        raise
