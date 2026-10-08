from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Body

from app.db.database import (
    create_subject,
    get_subject,
    list_all_subjects,
    delete_subject,
    add_syllabus_unit,
    add_syllabus_topic,
    get_entire_subject_hierarchy,
    get_or_create_student_subject_progress,
    update_student_subject_progress,
    load_complete_syllabus,
    PROTOTYPE_STUDENT_ID
)
from app.db.models import (
    SubjectCreate,
    SubjectResponse,
    SyllabusHierarchyResponse,
    StudentSubjectProgressResponse
)
from app.services.tutor_service import tutor_service

router = APIRouter(prefix="/api/v1/subjects", tags=["Academic Subjects & Syllabus"])


@router.post("", response_model=Dict[str, Any])
def create_new_subject(data: SubjectCreate):
    """Creates a new academic subject."""
    sub_id = create_subject(name=data.name, description=data.description or "")
    sub = get_subject(sub_id)
    return {"success": True, "subject": sub}


@router.get("", response_model=List[Dict[str, Any]])
def list_subjects():
    """Lists all registered academic subjects."""
    return list_all_subjects()


@router.get("/{subject_id}", response_model=Dict[str, Any])
def get_subject_details(subject_id: int):
    """Returns a subject and its complete hierarchical syllabus (units & topics)."""
    hierarchy = get_entire_subject_hierarchy(subject_id)
    if not hierarchy:
        raise HTTPException(status_code=404, detail="Subject not found")
    return hierarchy


@router.post("/load", response_model=Dict[str, Any])
def load_syllabus_data(syllabus: Dict[str, Any] = Body(...)):
    """Loads a complete subject with its nested units and topics from JSON."""
    sub_id = load_complete_syllabus(syllabus)
    hierarchy = get_entire_subject_hierarchy(sub_id)
    return {"success": True, "subject_id": sub_id, "syllabus": hierarchy}


@router.post("/{subject_id}/select", response_model=Dict[str, Any])
def select_active_subject(
    subject_id: int,
    student_id: int = Query(default=PROTOTYPE_STUDENT_ID),
    topic_id: Optional[int] = Query(default=None)
):
    """Sets the active subject and optional topic for the student."""
    sub, top = tutor_service.set_active_subject_and_topic(
        student_id=student_id,
        session_id=None,
        subject_name_or_id=subject_id,
        topic_name_or_id=topic_id
    )
    if not sub:
        raise HTTPException(status_code=404, detail="Subject not found")
    return {
        "success": True,
        "subject": sub["name"],
        "topic": top["topic_name"] if top else None
    }


@router.get("/{subject_id}/progress", response_model=Dict[str, Any])
def get_subject_progress(
    subject_id: int,
    student_id: int = Query(default=PROTOTYPE_STUDENT_ID)
):
    """Returns the student's learning progress on this subject."""
    progress = get_or_create_student_subject_progress(student_id=student_id, subject_id=subject_id)
    return progress


@router.delete("/{subject_id}", response_model=Dict[str, Any])
def remove_subject(subject_id: int):
    """Deletes a subject and all its units/topics."""
    success = delete_subject(subject_id)
    if not success:
        raise HTTPException(status_code=404, detail="Subject not found")
    return {"success": True, "message": "Subject deleted"}


@router.post("/session/reset", response_model=Dict[str, Any])
def reset_study_session_endpoint(student_id: int = Query(default=PROTOTYPE_STUDENT_ID)):
    """Resets the study session state so old tutor context is cleared."""
    tutor_service.reset_study_session(student_id=student_id)
    return {"success": True, "message": "Study session reset successfully"}

