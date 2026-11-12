import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Body
from pydantic import BaseModel

from app.config import settings
from app.db.database import (
    get_connection,
    get_student,
    create_session,
    get_session_by_id,
    add_conversation_message,
    get_conversation_history,
    get_subject,
    get_topic_by_id,
    get_entire_subject_hierarchy,
    get_or_create_student_subject_progress,
    mark_topic_completed,
    PROTOTYPE_STUDENT_ID
)
from app.services.study_document_service import study_doc_service
from app.services.tutor_service import tutor_service

router = APIRouter(prefix="/api/v1/study", tags=["Study Agent & Material-Based Teaching"])


class StudyChatRequest(BaseModel):
    message: str
    student_id: int = PROTOTYPE_STUDENT_ID
    session_id: Optional[str] = None
    subject_id: Optional[int] = None
    topic_id: Optional[int] = None
    action: Optional[str] = None  # e.g., "explain_simply", "example", "quiz", "next_topic", "mark_complete"


class StudyChatResponse(BaseModel):
    success: bool = True
    reply_text: str
    intent: str
    current_subject: str
    current_topic: str
    session_id: str
    is_outside_syllabus: bool = False
    mastery_score: Optional[int] = 0


@router.post("/upload")
async def upload_study_material(
    file: UploadFile = File(...),
    student_id: Optional[int] = Query(default=None),
    subject_id: Optional[int] = Query(default=None),
    student_id_form: Optional[int] = Form(default=None, alias="student_id"),
    subject_id_form: Optional[int] = Form(default=None, alias="subject_id")
):
    """
    Upload and index syllabus notes or PDF textbooks.
    Extracts Units, Topics, and Subtopics strictly from source.
    """
    final_student_id = student_id if student_id is not None else (student_id_form if student_id_form is not None else PROTOTYPE_STUDENT_ID)
    final_subject_id = subject_id if subject_id is not None else subject_id_form
    filename = file.filename or "syllabus.pdf"
    content_bytes = await file.read()
    file_size = len(content_bytes)

    # 1. Validation: file type & size
    is_valid, err_msg = study_doc_service.validate_file(filename, file_size)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err_msg)

    # 2. Save file temporarily
    safe_name = f"{uuid.uuid4().hex[:6]}_{filename}"
    saved_path = settings.UPLOAD_DIR / safe_name
    with open(saved_path, "wb") as f:
        f.write(content_bytes)

    # 3. Process, index, and load into syllabus hierarchy
    try:
        result = study_doc_service.index_and_load_study_material(
            file_path=saved_path,
            original_filename=filename,
            student_id=final_student_id,
            target_subject_id=final_subject_id
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process study material: {str(e)}")


@router.get("/documents")
def list_study_documents(subject_id: int = Query(...)):
    """Lists all uploaded documents for a specific subject."""
    return study_doc_service.list_study_documents(subject_id=subject_id)


@router.post("/chat", response_model=StudyChatResponse)
async def study_chat(req: StudyChatRequest):
    """
    Dedicated Study Agent Chat Endpoint.
    Grounded strictly in the uploaded syllabus and study materials.
    Maintains isolated study conversation state without polluting Normal Chat.
    """
    student_id = req.student_id or PROTOTYPE_STUDENT_ID
    session_id = req.session_id

    # Ensure a dedicated Study Session exists (mode="tutor")
    if not session_id:
        sess = create_session(student_id=student_id, mode="tutor")
        session_id = sess["session_id"]
    else:
        existing = get_session_by_id(session_id)
        if not existing:
            sess = create_session(student_id=student_id, mode="tutor", session_id=session_id)
            session_id = sess["session_id"]

    # Bind active subject and topic if specified
    if req.subject_id:
        tutor_service.set_active_subject_and_topic(
            student_id=student_id,
            session_id=session_id,
            subject_name_or_id=req.subject_id,
            topic_name_or_id=req.topic_id
        )

    # Record incoming user message in conversation history
    add_conversation_message(
        student_id=student_id,
        session_id=session_id,
        role="user",
        content=req.message,
        learning_mode="tutor"
    )

    # Process turn with strict source grounding & boundary checks
    tutor_turn = await tutor_service.process_tutor_turn(
        student_id=student_id,
        session_id=session_id,
        user_text=req.message,
        current_difficulty=1
    )

    reply_text = tutor_turn["reply_text"]
    intent = tutor_turn.get("intent", "TEACH")
    curr_subj = tutor_turn.get("current_subject", "")
    curr_top = tutor_turn.get("current_topic", "")
    is_outside = tutor_turn.get("is_outside_syllabus", False)

    # Save assistant response to conversation history
    add_conversation_message(
        student_id=student_id,
        session_id=session_id,
        role="assistant",
        content=reply_text,
        learning_mode="tutor",
        intent=intent
    )

    # Fetch updated subject progress
    prog = {}
    if req.subject_id:
        prog = get_or_create_student_subject_progress(student_id, req.subject_id)

    return StudyChatResponse(
        success=True,
        reply_text=reply_text,
        intent=intent,
        current_subject=curr_subj,
        current_topic=curr_top,
        session_id=session_id,
        is_outside_syllabus=is_outside,
        mastery_score=prog.get("mastery_score", 0)
    )


@router.post("/session/new")
def new_study_session(student_id: int = Query(default=PROTOTYPE_STUDENT_ID)):
    """
    Starts a fresh study session without erasing uploaded materials, subjects, or progress.
    """
    sess = create_session(student_id=student_id, mode="tutor")
    tutor_service.reset_study_session(student_id=student_id)
    return {
        "success": True,
        "session_id": sess["session_id"],
        "message": "Fresh study session started."
    }


@router.post("/topics/{topic_id}/complete")
def complete_study_topic(
    topic_id: int,
    subject_id: int = Query(...),
    student_id: int = Query(default=PROTOTYPE_STUDENT_ID)
):
    """
    Explicitly marks a topic complete and updates subject mastery.
    """
    updated_prog = mark_topic_completed(student_id=student_id, subject_id=subject_id, topic_name_or_id=topic_id)
    return {
        "success": True,
        "topic_id": topic_id,
        "progress": updated_prog
    }


@router.get("/history")
def get_study_history(session_id: str = Query(...), student_id: int = Query(default=PROTOTYPE_STUDENT_ID)):
    """Returns conversation history for this study session."""
    messages = get_conversation_history(student_id=student_id, session_id=session_id)
    return {"messages": messages}
