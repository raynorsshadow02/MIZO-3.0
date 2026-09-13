import uuid
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.config import settings
from app.db.database import get_all_knowledge_docs, delete_knowledge_doc
from app.db.models import KnowledgeDocumentResponse
from app.services.rag_service import rag_service

router = APIRouter(prefix="/api/v1/knowledge", tags=["Knowledge Base & RAG"])


@router.get("/documents")
async def list_documents():
    """List all indexed knowledge documents."""
    return get_all_knowledge_docs()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or text document for chunking and RAG indexing."""
    filename = file.filename or "document.pdf"
    safe_name = f"{uuid.uuid4().hex[:6]}_{filename}"
    saved_path = settings.UPLOAD_DIR / safe_name

    with open(saved_path, "wb") as f:
        f.write(await file.read())

    try:
        doc_id = rag_service.index_document(saved_path, filename)
        return {
            "success": True,
            "message": f"Document '{filename}' uploaded and indexed successfully.",
            "doc_id": doc_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.get("/search")
async def search_knowledge(q: str = Query(..., description="Search query")):
    """Query knowledge base for semantic chunk matches."""
    chunks = rag_service.query_knowledge(q, top_k=5)
    return {"query": q, "results": chunks}


@router.delete("/documents/{doc_id}")
async def remove_document(doc_id: int):
    """Delete a document and its indexed chunks from the database."""
    delete_knowledge_doc(doc_id)
    return {"success": True, "message": f"Document {doc_id} deleted."}
