import logging
import uuid
import os
from fastapi import (
    APIRouter,
    status,
    UploadFile,
    File,
    BackgroundTasks,
    HTTPException,
    Path,
)
from pydantic import BaseModel
from modules import database, gemini
from utils import transcription

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


class ConversationUpload(BaseModel):
    audio_file: bytes


class ConversationUploadResponse(BaseModel):
    upload_id: str
    status: str


class ComprehensiveConversationResponse(BaseModel):
    upload_id: str
    status: str
    filename: str
    created_at: str
    updated_at: str
    summary: dict | None = None
    contact: dict | None = None


class TranscriptSummary(BaseModel):
    summary: str
    mindmap: dict


class ContactInference(BaseModel):
    contact: dict


class EnhancedTranscriptResponse(BaseModel):
    transcript_id: str
    text: str
    segments: list


@router.post(
    "/upload",
    response_model=ConversationUploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_conversation(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
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
    "/",
    response_model=list[ComprehensiveConversationResponse],
    status_code=status.HTTP_200_OK,
)
async def get_upload():
    """
    Retrieve all conversation uploads with summaries and contacts

    Returns a list of all conversation uploads in the system, including their
    upload IDs, current processing status, summaries, and contact information.

    Returns:
        list[ComprehensiveConversationResponse]: A list containing all conversation uploads
                                               with their upload_id, status, summary, and contact data
    """
    # GET ALL CONVERSATIONS WITH DETAILS
    conversations = await database.get_all_conversations_with_details()

    # Transform the database response to match the response model
    formatted_conversations = []
    for conversation in conversations:
        # Extract summary data
        summary_data = None
        if conversation.get("summary"):
            summary_data = {
                "summary": conversation["summary"].get("summary"),
                "mindmap": conversation["summary"].get("mindmap"),
            }

        # Extract contact data
        contact_data = None
        if conversation.get("contact"):
            contact_data = conversation["contact"].get("contact", {})

        # Format the response
        formatted_conversation = {
            "upload_id": conversation.get("_id"),
            "status": conversation.get("status"),
            "filename": conversation.get("filename"),
            "created_at": (
                conversation.get("created_at").isoformat()
                if conversation.get("created_at")
                else None
            ),
            "updated_at": (
                conversation.get("updated_at").isoformat()
                if conversation.get("updated_at")
                else None
            ),
            "summary": summary_data,
            "contact": contact_data,
        }

        formatted_conversations.append(formatted_conversation)

    return formatted_conversations


@router.get(
    "/summary/{transcript_id}",
    response_model=TranscriptSummary,
    status_code=status.HTTP_200_OK,
)
async def get_summary(
    transcript_id: str = Path(
        ..., description="The transcript ID to generate or retrieve a summary for"
    )
):
    """Get or generate a summary and mindmap for a transcript"""
    try:
        # First, check if summary already exists
        existing_summary = await database.get_summary_by_upload_id(transcript_id)
        if existing_summary:
            logger.info(f"Found existing summary for transcript {transcript_id}")
            return {
                "summary": existing_summary.get("summary", ""),
                "mindmap": existing_summary.get("mindmap", {"nodes": [], "edges": []}),
            }

        # If no summary exists, check if transcript exists
        transcript = await database.get_transcript_by_id(transcript_id)
        if not transcript:
            # Try looking up by upload_id as well
            transcript = await database.get_transcript_by_upload_id(transcript_id)

        if not transcript:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transcript with ID {transcript_id} not found",
            )

        # TODO: Implement actual summarization and mindmap generation
        # For now, return a placeholder
        logger.info(f"Generating placeholder summary for transcript {transcript_id}")
        summary_data = {
            "summary": "Summary generation is not yet implemented. This is a placeholder.",
            "mindmap": {"nodes": [{"id": 1, "label": "Conversation"}], "edges": []},
        }

        # Store the summary
        await database.store_summary(transcript_id, summary_data)

        return summary_data

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            f"Error generating summary for transcript {transcript_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating summary: {str(e)}",
        )


@router.get(
    "/enhanced-transcript/{upload_id}",
    response_model=EnhancedTranscriptResponse,
    status_code=status.HTTP_200_OK,
)
async def get_enhanced_transcript(
    upload_id: str = Path(
        ..., description="The upload ID to retrieve enhanced transcript for"
    )
):
    """Get enhanced transcript with named speakers and corrected text"""
    try:
        # Check if enhanced transcript exists
        enhanced = await database.get_enhanced_transcript_by_upload_id(upload_id)
        if enhanced:
            logger.info(f"Found existing enhanced transcript for upload {upload_id}")
            return {
                "transcript_id": enhanced.get("_id"),
                "text": enhanced.get("text", ""),
                "segments": enhanced.get("segments", []),
            }

        # If no enhanced transcript exists, check if upload is still processing
        upload = await database.get_upload_by_id(upload_id)
        if not upload:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Upload with ID {upload_id} not found",
            )

        # Check upload status
        if upload.get("status") in ["uploaded", "processing", "enhancing"]:
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail=f"Upload is still being processed. Current status: {upload.get('status')}",
            )

        # If enhancement failed, try to generate it now
        if (
            upload.get("status") == "enhancement_failed"
            or upload.get("status") == "transcribed"
        ):
            # Get the transcript
            transcript = await database.get_transcript_by_upload_id(upload_id)
            if not transcript:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Transcript for upload {upload_id} not found",
                )

            # Try to enhance the transcript now
            logger.info(f"Attempting to enhance transcript now for upload: {upload_id}")
            enhanced_transcript = await gemini.enhance_transcript(transcript)

            # Store the enhanced transcript
            enhanced_transcript_data = enhanced_transcript.dict()
            enhanced_transcript_data["original_transcript_id"] = upload_id

            # Store in database
            enhanced_id = await database.store_enhanced_transcript(
                upload_id, enhanced_transcript_data
            )

            # Update upload status
            await database.update_upload_status(upload_id, "enhanced")

            return {
                "transcript_id": enhanced_id,
                "text": enhanced_transcript.text,
                "segments": enhanced_transcript.segments,
            }

        # If we got here, something unexpected happened
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to retrieve or generate enhanced transcript. Upload status: {upload.get('status')}",
        )

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            f"Error retrieving enhanced transcript for upload {upload_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving enhanced transcript: {str(e)}",
        )


@router.get(
    "/contact-inference/{transcript_id}",
    response_model=ContactInference,
    status_code=status.HTTP_200_OK,
)
async def get_contact_inference(
    transcript_id: str = Path(
        ..., description="The transcript ID to perform contact inference on"
    )
):
    """Extract contact information and perform LinkedIn lookup"""
    try:
        # First, check if contact info already exists
        existing_contact = await database.get_contact_by_upload_id(transcript_id)
        if existing_contact:
            logger.info(f"Found existing contact data for transcript {transcript_id}")
            return {"contact": existing_contact.get("contact", {})}

        # If no contact info exists, check if transcript exists
        transcript = await database.get_transcript_by_id(transcript_id)
        if not transcript:
            # Try looking up by upload_id as well
            transcript = await database.get_transcript_by_upload_id(transcript_id)

        if not transcript:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transcript with ID {transcript_id} not found",
            )

        # TODO: Implement actual contact inference and LinkedIn lookup
        # For now, return a placeholder
<<<<<<< Updated upstream
        # logger.info(f"Generating placeholder contact data for transcript {transcript_id}")
        # contact_data = {
        #     ...
        # }
        
=======
        logger.info(
            f"Generating placeholder contact data for transcript {transcript_id}"
        )
        contact_data = {
            "contact": {
                "name": "Contact extraction not yet implemented",
                "title": "Placeholder",
                "company": "Example Corp",
                "linkedin_url": "https://linkedin.com/",
            }
        }

>>>>>>> Stashed changes
        # Store the contact data
        await database.store_contact_data(transcript_id, contact_data)

        return contact_data

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            f"Error performing contact inference for transcript {transcript_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing contact inference: {str(e)}",
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
        transcript_data = await transcription.process_audio_file(
            file_path, hf_token=hf_token
        )

        # Set transcript ID to match upload ID for easier reference
        transcript_data["transcript_id"] = upload_id

        # Store original transcript data in database
        await database.store_transcript(upload_id, transcript_data)

        # Update status to indicate transcription is complete
        await database.update_upload_status(upload_id, "transcribed")

        # Enhanced transcript with Gemini - identify speakers by name and correct text
        logger.info(f"Enhancing transcript with Gemini for upload: {upload_id}")
        await database.update_upload_status(upload_id, "enhancing")

        try:
            # Use Gemini to enhance the transcript
            enhanced_transcript = await gemini.enhance_transcript(transcript_data)

            # Store the enhanced transcript
            enhanced_transcript_data = enhanced_transcript.dict()
            enhanced_transcript_data["original_transcript_id"] = upload_id

            # Store enhanced transcript in the database
            enhanced_id = await database.store_enhanced_transcript(
                upload_id, enhanced_transcript_data
            )

            logger.info(
                f"Successfully enhanced transcript with Gemini for upload: {upload_id}"
            )
            await database.update_upload_status(upload_id, "enhanced")

        except Exception as e:
            logger.error(f"Error enhancing transcript with Gemini: {str(e)}")
            # Continue with processing even if enhancement fails
            await database.update_upload_status(upload_id, "enhancement_failed")

        # Step 2: Summarization and Mindmap Generation
        try:
            logger.info(f"Generating summary and mindmap for upload: {upload_id}")
            await database.update_upload_status(upload_id, "summarizing")

            # Get the enhanced transcript if available, otherwise use original transcript
            enhanced_transcript = await database.get_enhanced_transcript_by_upload_id(
                upload_id
            )
            if enhanced_transcript:
                logger.info(f"Using enhanced transcript for summary generation")
                transcript_text = enhanced_transcript.get("text", "")
            else:
                logger.info(f"Enhanced transcript not found, using original transcript")
                original_transcript = await database.get_transcript_by_upload_id(
                    upload_id
                )
                if not original_transcript:
                    logger.error(f"No transcript found for upload {upload_id}")
                    raise ValueError(f"No transcript found for upload {upload_id}")
                transcript_text = original_transcript.get("text", "")

            # Generate summary and mindmap using Gemini
            summary_result = await gemini.generate_summary(
                transcript_text, transcript_data.get("segments", [])[-1].get("end", 0)
            )

            # Convert the mindmap nodes to the format expected by the database
            mindmap_data = {"nodes": [], "edges": []}

            # Add nodes
            for node in summary_result.mindmap.nodes:
                mindmap_data["nodes"].append({"id": node.id, "label": node.label})

                # Add edge if this node has a parent
                if node.parent_id is not None:
                    mindmap_data["edges"].append(
                        {"from": node.parent_id, "to": node.id}
                    )

            # Prepare and store summary data
            summary_data = {"summary": summary_result.summary, "mindmap": mindmap_data}

            await database.store_summary(upload_id, summary_data)
            await database.update_upload_status(upload_id, "summarized")
            logger.info(
                f"Successfully generated summary and mindmap for upload: {upload_id}"
            )

        except Exception as e:
            logger.error(f"Error generating summary for upload {upload_id}: {str(e)}")
            # Create a minimal fallback summary
            summary_data = {
                "summary": "Unable to generate summary. Please try again later.",
                "mindmap": {"nodes": [{"id": 1, "label": "Conversation"}], "edges": []},
            }
            await database.store_summary(upload_id, summary_data)
            await database.update_upload_status(upload_id, "summary_failed")

        # Step 3: Contact Inference and LinkedIn Lookup
        # TODO: Implement contact inference and LinkedIn lookup via Dex MCP
        # For now, we'll generate a placeholder contact
<<<<<<< Updated upstream
        # logger.info(f"Generating placeholder contact data for upload: {upload_id}")
        # contact_data = {

        # }
        # await database.store_contact_data(upload_id, contact_data)
        # await database.update_upload_status(upload_id, "contact_identified")
        
=======
        logger.info(f"Generating placeholder contact data for upload: {upload_id}")
        contact_data = {
            "contact": {
                "name": "Contact extraction not yet implemented",
                "title": "Placeholder",
                "company": "Example Corp",
                "linkedin_url": "https://linkedin.com/",
            }
        }
        await database.store_contact_data(upload_id, contact_data)
        await database.update_upload_status(upload_id, "contact_identified")

>>>>>>> Stashed changes
        logger.info(f"Completed audio processing for upload: {upload_id}")
        await database.update_upload_status(upload_id, "processed")

    except Exception as e:
        logger.error(f"Error processing audio for upload {upload_id}: {str(e)}")
        await database.update_upload_status(upload_id, "error", error_message=str(e))
