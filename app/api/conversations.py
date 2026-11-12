from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional, Dict, Any
from app.db.database import (
    PROTOTYPE_STUDENT_ID,
    get_conversation_history,
    create_session,
    delete_chat,
    reset_session_context,
    get_student
)
from app.services.tutor_service import SubjectTutorService
from app.db.models import NewChatRequest

router = APIRouter(prefix="/api/v1/conversations", tags=["Conversations"])


@router.get("")
async def list_conversations(
    student_id: Optional[int] = Query(None, description="Filter by student ID"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    limit: int = Query(50, description="Max messages to retrieve"),
    search: Optional[str] = Query(None, description="Search message contents")
) -> List[Dict[str, Any]]:
    """Retrieve conversation history and evaluation records."""
    target_id = student_id if student_id is not None else PROTOTYPE_STUDENT_ID
    return get_conversation_history(student_id=target_id, session_id=session_id, limit=limit, search=search)


@router.get("/student/{student_id}")
async def list_student_conversations(
    student_id: int,
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    limit: int = Query(50, description="Max messages to retrieve"),
    search: Optional[str] = Query(None, description="Search message contents")
) -> List[Dict[str, Any]]:
    """Retrieve conversation history for a specific student."""
    return get_conversation_history(student_id=student_id, session_id=session_id, limit=limit, search=search)


@router.post("/new-chat")
async def start_new_chat(payload: Optional[NewChatRequest] = None) -> Dict[str, Any]:
    """
    Creates a new isolated conversation / chat session.
    - Generates a fresh session_id.
    - Starts with empty visible message history.
    - Preserves onboarding profile and saved durable memories.
    - Clears in-memory tutor/context state so old topics do not leak.
    """
    sid = payload.student_id if payload and payload.student_id is not None else PROTOTYPE_STUDENT_ID
    mode = payload.mode if payload and payload.mode else "coach"

    session = create_session(student_id=sid, mode=mode)
    sess_id = session["session_id"]

    # Reset in-memory cached context and tutor state for the new chat session
    reset_session_context(sess_id)
    SubjectTutorService.reset_study_session(student_id=sid, session_id=sess_id)

    return {
        "status": "created",
        "session_id": sess_id,
        "student_id": sid,
        "mode": mode,
        "created_at": session.get("start_time")
    }


@router.delete("/chats/{session_id}")
@router.delete("/{session_id}")
async def delete_chat_endpoint(
    session_id: str,
    student_id: Optional[int] = Query(None, description="Student ID")
) -> Dict[str, Any]:
    """
    Deletes only the specified chat's messages and session context.
    Strictly preserves the student profile, onboarding status, assessments, and saved memories.
    """
    res = delete_chat(session_id=session_id, student_id=student_id)
    return {
        "status": "deleted",
        "session_id": session_id,
        "deleted_messages": res.get("deleted_messages", 0) if isinstance(res, dict) else int(res or 0),
        "active_session": res.get("active_session") if isinstance(res, dict) else None
    }
