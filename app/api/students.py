from fastapi import APIRouter, HTTPException
from typing import List
from app.db.database import get_all_students, get_student, create_student, update_student
from app.db.models import StudentCreate, StudentUpdate, StudentResponse

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
