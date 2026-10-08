from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from app.db.database import (
    update_student,
    create_session,
    get_active_session,
    get_session_by_id,
    get_recent_mistakes,
    reset_student_to_fresh,
    restart_onboarding,
    delete_learner_data,
    get_student_assessments,
    get_assessment_by_id,
    get_student_historical_progress,
    reset_current_session,
    PROTOTYPE_STUDENT_ID,
    get_prototype_student,
)
from app.services.personalization_service import personalization_service
from app.db.models import (
    StudentCreate,
    StudentUpdate,
    StudentResponse,
    SessionCreate,
    SessionResponse,
    StudentMistake,
    AssessmentResponse,
    HistoricalProgressResponse
)

router = APIRouter(prefix="/api/v1/students", tags=["Student Management"])


def _primary_student_or_503() -> Dict[str, Any]:
    student = get_prototype_student()
    if not student:
        raise HTTPException(status_code=503, detail="Primary student profile (Student #1) is missing")
    return student


@router.get("", response_model=List[StudentResponse])
async def list_students():
    """Retrieve all student profiles."""
    return [_primary_student_or_503()]


@router.get("/{student_id}", response_model=StudentResponse)
async def get_student_profile(student_id: int):
    """Retrieve a single student profile and learning metrics."""
    return _primary_student_or_503()


@router.post("", response_model=StudentResponse)
async def add_student(payload: StudentCreate):
    """Student creation is disabled for the single-learner prototype."""
    raise HTTPException(status_code=403, detail="Mizo prototype uses the existing Student #1 profile")


@router.put("/{student_id}", response_model=StudentResponse)
async def edit_student(student_id: int, payload: StudentUpdate):
    """Update the sole prototype learner profile."""
    _primary_student_or_503()
    updated = update_student(PROTOTYPE_STUDENT_ID, payload.model_dump(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=404, detail="Student not found")
    return updated


@router.post("/{student_id}/restart-onboarding", response_model=StudentResponse)
@router.post("/{student_id}/onboarding/restart", response_model=StudentResponse)
async def restart_student_onboarding(student_id: int):
    """
    Restarts onboarding for a new run:
    - Archives current profile into profile_archives snapshot.
    - Resets active profile state back to ask_name.
    - Preserves all past sessions, conversation dialogues, and assessments.
    """
    _primary_student_or_503()
    return restart_onboarding(PROTOTYPE_STUDENT_ID)


@router.delete("/{student_id}", response_model=StudentResponse)
@router.delete("/{student_id}/data", response_model=StudentResponse)
async def delete_student_learner_data(student_id: int):
    """
    Destructive action requiring explicit confirmation:
    - Permanently deletes student profile, conversations, assessments, mistakes, and sessions.
    - Strictly preserves system_settings, .env, and device API keys.
    """
    _primary_student_or_503()
    return delete_learner_data(PROTOTYPE_STUDENT_ID)


@router.post("/{student_id}/complete-text-onboarding", response_model=StudentResponse)
async def complete_text_only_onboarding(student_id: int):
    """
    Completes onboarding without audio speaking assessment when microphone is unavailable.
    Leaves speaking assessment pending and does not invent fake baseline scores.
    """
    _primary_student_or_503()
    return personalization_service.complete_text_onboarding_without_audio(PROTOTYPE_STUDENT_ID)


@router.post("/{student_id}/reset", response_model=StudentResponse)
async def reset_student(student_id: int):
    """Resets student profile, zeroes metrics, and sets onboarding to fresh start (Factory / Admin Reset)."""
    _primary_student_or_503()
    return reset_student_to_fresh(PROTOTYPE_STUDENT_ID)


# -------------------------------------------------------------
# Session Lifecycle Endpoints
# -------------------------------------------------------------
@router.post("/{student_id}/sessions", response_model=SessionResponse)
@router.post("/{student_id}/sessions/new", response_model=SessionResponse)
async def start_new_session(student_id: int, payload: Optional[SessionCreate] = None):
    """
    Creates a new learning session for the student.
    Current session metrics are initialized to ZERO without deleting historical progress.
    """
    mode = payload.mode if payload else "coach"
    _primary_student_or_503()
    session_data = create_session(student_id=PROTOTYPE_STUDENT_ID, mode=mode)
    return session_data


@router.get("/{student_id}/sessions/active", response_model=SessionResponse)
async def get_student_active_session(student_id: int, mode: Optional[str] = Query("coach")):
    """Retrieves current active session or starts one if none is active."""
    _primary_student_or_503()
    return get_active_session(student_id=PROTOTYPE_STUDENT_ID, default_mode=mode or "coach")


# -------------------------------------------------------------
# Persistent Mistakes & Weakness Memory
# -------------------------------------------------------------
@router.get("/{student_id}/mistakes", response_model=List[StudentMistake])
async def get_student_mistakes_history(student_id: int, limit: int = Query(10)):
    """Retrieves logged mistakes and grammatical errors for the student."""
    _primary_student_or_503()
    return get_recent_mistakes(student_id=PROTOTYPE_STUDENT_ID, limit=limit)


# -------------------------------------------------------------
# Detailed Speaking Assessments & Evidence
# -------------------------------------------------------------
@router.get("/{student_id}/assessments", response_model=List[AssessmentResponse])
async def get_student_assessments_history(student_id: int, limit: int = Query(20)):
    """Retrieves all past speaking assessment records and reasoning for the student."""
    _primary_student_or_503()
    return get_student_assessments(student_id=PROTOTYPE_STUDENT_ID, limit=limit)


@router.get("/{student_id}/assessments/{assessment_id}", response_model=AssessmentResponse)
async def get_single_assessment(student_id: int, assessment_id: int):
    """Retrieves a specific speaking assessment record with detailed dimension scores and evidence."""
    assessment = get_assessment_by_id(assessment_id)
    if not assessment or assessment.get("student_id") != PROTOTYPE_STUDENT_ID:
        raise HTTPException(status_code=404, detail="Assessment record not found")
    return assessment


# -------------------------------------------------------------
# Historical Progress & Session Reset Endpoints (Part 2, 21, 23, 24)
# -------------------------------------------------------------
@router.get("/{student_id}/historical-progress", response_model=HistoricalProgressResponse)
async def get_learner_historical_progress(student_id: int):
    """
    Retrieves cumulative historical progress calculated strictly from completed assessments.
    Contains full data-source provenance and never outputs fake/default numbers.
    """
    _primary_student_or_503()
    return get_student_historical_progress(PROTOTYPE_STUDENT_ID)


@router.post("/{student_id}/sessions/reset-current", response_model=SessionResponse)
async def reset_student_current_session(student_id: int):
    """
    Clears only the active session's metrics and conversation messages.
    Does NOT delete historical progress, learner profile, or completed assessments.
    """
    _primary_student_or_503()
    return reset_current_session(student_id=PROTOTYPE_STUDENT_ID)


