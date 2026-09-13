from fastapi import APIRouter, Query
from typing import List, Optional, Dict, Any
from app.db.database import get_conversation_history

router = APIRouter(prefix="/api/v1/conversations", tags=["Conversations"])


@router.get("")
async def list_conversations(
    student_id: Optional[int] = Query(None, description="Filter by student ID"),
    limit: int = Query(50, description="Max messages to retrieve")
) -> List[Dict[str, Any]]:
    """Retrieve conversation history and evaluation records."""
    return get_conversation_history(student_id=student_id, limit=limit)
