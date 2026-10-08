from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Body

from app.db.database import (
    create_or_get_speech_session,
    get_speech_session_by_id,
    get_active_speech_session,
    update_speech_session,
    get_previous_speech_attempts,
    PROTOTYPE_STUDENT_ID
)
from app.db.models import SpeechSessionResponse
from app.services.speech_service import speech_service

router = APIRouter(prefix="/api/v1/speech", tags=["Speech & Seminar Practice"])


@router.get("/session", response_model=Dict[str, Any])
def get_current_speech_session(
    student_id: int = Query(default=PROTOTYPE_STUDENT_ID),
    session_id: str = Query(default="speech_default")
):
    """Returns the current speech practice session state."""
    sess = create_or_get_speech_session(student_id=student_id, session_id=session_id)
    return sess


@router.post("/setup", response_model=Dict[str, Any])
def setup_speech_session(
    student_id: int = Query(default=PROTOTYPE_STUDENT_ID),
    session_id: str = Query(default="speech_default"),
    topic: str = Body(...),
    required_points: List[str] = Body(...),
    time_limit_seconds: int = Body(default=180)
):
    """Explicitly configures the topic, points, and time limit."""
    sess = create_or_get_speech_session(student_id=student_id, session_id=session_id)
    updated = update_speech_session(sess["id"], {
        "topic": topic.strip(),
        "required_points": required_points,
        "time_limit_seconds": time_limit_seconds,
        "state": speech_service.STATE_READY
    })
    return {"success": True, "session": updated}


@router.post("/start", response_model=Dict[str, Any])
def start_speech_recording(
    student_id: int = Query(default=PROTOTYPE_STUDENT_ID),
    session_id: str = Query(default="speech_default")
):
    """Transitions state from READY to RECORDING."""
    sess = create_or_get_speech_session(student_id=student_id, session_id=session_id)
    if sess.get("state") != speech_service.STATE_READY:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start recording from state '{sess.get('state')}'. Complete SETUP first."
        )
    updated = update_speech_session(sess["id"], {"state": speech_service.STATE_RECORDING})
    return {"success": True, "session": updated}


@router.post("/finish", response_model=Dict[str, Any])
async def finish_speech_and_analyze(
    student_id: int = Query(default=PROTOTYPE_STUDENT_ID),
    session_id: str = Query(default="speech_default"),
    transcript: str = Body(...),
    duration_seconds: Optional[float] = Body(default=None)
):
    """Finishes speech attempt, processes transcript, and generates structured feedback."""
    res = await speech_service.process_speech_turn(
        student_id=student_id,
        session_id=session_id,
        user_text=transcript,
        duration_seconds=duration_seconds
    )
    return res


@router.get("/history", response_model=List[Dict[str, Any]])
def get_speech_history(
    topic: str = Query(...),
    student_id: int = Query(default=PROTOTYPE_STUDENT_ID)
):
    """Returns past attempts and feedback for comparison."""
    return get_previous_speech_attempts(student_id=student_id, topic=topic)


@router.post("/analyze-transcript", response_model=Dict[str, Any])
async def analyze_speech_transcript_endpoint(data: Dict[str, Any] = Body(...)):
    """Convenience endpoint for frontend and scripts to submit transcripts for rubric feedback."""
    student_id = data.get("student_id", PROTOTYPE_STUDENT_ID)
    session_id = data.get("session_id", "speech_default")
    transcript = data.get("transcript", "")
    duration_seconds = data.get("duration_seconds")
    res = await speech_service.process_speech_turn(
        student_id=student_id,
        session_id=session_id,
        user_text=transcript,
        duration_seconds=duration_seconds
    )
    return res

