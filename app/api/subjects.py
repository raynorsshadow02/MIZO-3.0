from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Body, UploadFile, File

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


@router.post("/{subject_id}/upload-material")
async def upload_subject_material(
    subject_id: int,
    file: UploadFile = File(...),
    student_id: int = Query(default=PROTOTYPE_STUDENT_ID)
):
    """Uploads and indexes study material directly for this subject."""
    import uuid
    from app.config import settings
    from app.services.study_document_service import study_doc_service

    filename = file.filename or "notes.pdf"
    content_bytes = await file.read()
    file_size = len(content_bytes)

    is_valid, err_msg = study_doc_service.validate_file(filename, file_size)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err_msg)

    safe_name = f"{uuid.uuid4().hex[:6]}_{filename}"
    saved_path = settings.UPLOAD_DIR / safe_name
    with open(saved_path, "wb") as f:
        f.write(content_bytes)

    try:
        res = study_doc_service.index_and_load_study_material(
            file_path=saved_path,
            original_filename=filename,
            student_id=student_id,
            target_subject_id=subject_id
        )
        return res
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.get("/{subject_id}/documents")
def get_subject_documents(subject_id: int):
    """Lists all uploaded documents for a subject."""
    from app.services.study_document_service import study_doc_service
    docs = study_doc_service.list_study_documents(subject_id=subject_id)
    return {"subject_id": subject_id, "documents": docs}


@router.post("/{subject_id}/topics/{topic_id}/complete")
def mark_topic_complete_endpoint(
    subject_id: int,
    topic_id: int,
    student_id: int = Query(default=PROTOTYPE_STUDENT_ID)
):
    """Explicitly marks a topic complete and updates mastery."""
    from app.db.database import mark_topic_completed
    prog = mark_topic_completed(student_id=student_id, subject_id=subject_id, topic_name_or_id=topic_id)
    return {"success": True, "progress": prog}

