import math
import re
from pathlib import Path
from typing import List, Dict, Any
from app.db.database import add_knowledge_doc, add_knowledge_chunk, get_all_chunks, get_all_knowledge_docs, delete_knowledge_doc


class RAGService:
    """Document processing and semantic retrieval for Mikaza knowledge grounding."""

    @staticmethod
    def extract_text_from_pdf(pdf_path: Path) -> str:
        """Extract plain text from PDF using pypdf."""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(pdf_path))
            text_parts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
            return "\n".join(text_parts)
        except Exception as e:
            print(f"[RAG] PDF extraction failed: {e}")
            return ""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping clean chunks."""
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunks.append(" ".join(chunk_words))
            i += (chunk_size - overlap)
            if i >= len(words):
                break
        return chunks if chunks else [text]

    @classmethod
    def index_document(cls, file_path: Path, original_filename: str) -> int:
        """Parse, chunk, and index a document into SQLite."""
        if file_path.suffix.lower() == ".pdf":
            content = cls.extract_text_from_pdf(file_path)
        else:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                content = ""

        if not content.strip():
            content = f"Empty or unreadable document: {original_filename}"

        chunks = cls.chunk_text(content)
        title = original_filename.replace("_", " ").rsplit(".", 1)[0]
        doc_id = add_knowledge_doc(
            filename=original_filename,
            file_path=str(file_path),
            title=title,
            chunks_count=len(chunks)
        )

        for idx, chunk_text in enumerate(chunks):
            add_knowledge_chunk(doc_id=doc_id, chunk_index=idx, content=chunk_text)

        return doc_id

    @classmethod
    def query_knowledge(cls, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top relevant chunks matching the user query using keyword/BM25-style scoring."""
        chunks = get_all_chunks()
        if not chunks:
            return []

        query_terms = set(re.findall(r'\w+', query.lower()))
        if not query_terms:
            return []

        scored_chunks = []
        for ch in chunks:
            content = ch["content"]
            chunk_words = re.findall(r'\w+', content.lower())
            if not chunk_words:
                continue

            # Calculate match score
            score = 0
            for term in query_terms:
                count = chunk_words.count(term)
                if count > 0:
                    score += (1 + math.log(count)) * (1.0 / (1.0 + math.log(len(chunk_words))))

            if score > 0:
                scored_chunks.append({
                    "id": ch["id"],
                    "doc_id": ch["doc_id"],
                    "content": content,
                    "score": round(score, 4)
                })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]


rag_service = RAGService()
