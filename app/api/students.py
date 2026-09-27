from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from app.db.database import (
    get_all_students,
    get_student,
    create_student,
    update_student,
    create_session,
    get_active_session,
    get_session_by_id,
    get_recent_mistakes,
    reset_student_to_fresh,
    restart_onboarding,
    delete_learner_data,
    get_student_assessments,
    get_assessment_by_id
)
from app.services.personalization_service import personalization_service
from app.db.models import (
    StudentCreate,
    StudentUpdate,
    StudentResponse,
    SessionCreate,
    SessionResponse,
    StudentMistake,
    AssessmentResponse
)

router = APIRouter(prefix="/api/v1/students", tags=["Student Management"])


@router.get("", response_model=List[StudentResponse])
async def list_students():
    """Retrieve all student profiles."""
    return get_all_students()


@router.get("/{student_id}", response_model=StudentResponse)
async def get_student_profile(student_id: int):
    """Retrieve a single student profile and learning metrics."""
    student = get_student(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


@router.post("", response_model=StudentResponse)
async def add_student(payload: StudentCreate):
    """Register a new student profile."""
    student = create_student(payload.model_dump())
    return student


@router.put("/{student_id}", response_model=StudentResponse)
async def edit_student(student_id: int, payload: StudentUpdate):
    """Update an existing student profile."""
    updated = update_student(student_id, payload.model_dump(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=404, detail="Student not found")
    return updated


@router.post("/{student_id}/restart-onboarding", response_model=StudentResponse)
async def restart_student_onboarding(student_id: int):
    """
    Restarts onboarding for a new run:
    - Archives current profile into profile_archives snapshot.
    - Resets active profile state back to ask_name.
    - Preserves all past sessions, conversation dialogues, and assessments.
    """
    student = get_student(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return restart_onboarding(student_id)


@router.delete("/{student_id}/data", response_model=StudentResponse)
async def delete_student_learner_data(student_id: int):
    """
    Destructive action requiring explicit confirmation:
    - Permanently deletes student profile, conversations, assessments, mistakes, and sessions.
    - Strictly preserves system_settings, .env, and device API keys.
    """
    student = get_student(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return delete_learner_data(student_id)


@router.post("/{student_id}/complete-text-onboarding", response_model=StudentResponse)
async def complete_text_only_onboarding(student_id: int):
    """
    Completes onboarding without audio speaking assessment when microphone is unavailable.
    Leaves speaking assessment pending and does not invent fake baseline scores.
    """
    student = get_student(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return personalization_service.complete_text_onboarding_without_audio(student_id)


@router.post("/{student_id}/reset", response_model=StudentResponse)
async def reset_student(student_id: int):
    """Resets student profile, zeroes metrics, and sets onboarding to fresh start (Factory / Admin Reset)."""
    student = get_student(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return reset_student_to_fresh(student_id)


# -------------------------------------------------------------
# Session Lifecycle Endpoints
# -------------------------------------------------------------
@router.post("/{student_id}/sessions", response_model=SessionResponse)
async def start_new_session(student_id: int, payload: Optional[SessionCreate] = None):
    """
    Creates a new learning session for the student.
    Current session metrics are initialized to ZERO without deleting historical progress.
    """
    student = get_student(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    mode = payload.mode if payload else "coach"
    session_data = create_session(student_id=student_id, mode=mode)
    return session_data


@router.get("/{student_id}/sessions/active", response_model=SessionResponse)
async def get_student_active_session(student_id: int, mode: Optional[str] = Query("coach")):
    """Retrieves current active session or starts one if none is active."""
    student = get_student(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return get_active_session(student_id=student_id, default_mode=mode or "coach")


# -------------------------------------------------------------
# Persistent Mistakes & Weakness Memory
# -------------------------------------------------------------
@router.get("/{student_id}/mistakes", response_model=List[StudentMistake])
async def get_student_mistakes_history(student_id: int, limit: int = Query(10)):
    """Retrieves logged mistakes and grammatical errors for the student."""
    return get_recent_mistakes(student_id=student_id, limit=limit)


# -------------------------------------------------------------
# Detailed Speaking Assessments & Evidence
# -------------------------------------------------------------
@router.get("/{student_id}/assessments", response_model=List[AssessmentResponse])
async def get_student_assessments_history(student_id: int, limit: int = Query(20)):
    """Retrieves all past speaking assessment records and reasoning for the student."""
    return get_student_assessments(student_id=student_id, limit=limit)


@router.get("/{student_id}/assessments/{assessment_id}", response_model=AssessmentResponse)
async def get_single_assessment(student_id: int, assessment_id: int):
    """Retrieves a specific speaking assessment record with detailed dimension scores and evidence."""
    assessment = get_assessment_by_id(assessment_id)
    if not assessment or assessment.get("student_id") != student_id:
        raise HTTPException(status_code=404, detail="Assessment record not found")
    return assessment

