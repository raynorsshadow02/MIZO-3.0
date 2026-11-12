import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from app.config import settings
from app.db.database import (
    get_connection,
    create_subject,
    get_subject,
    add_syllabus_unit,
    add_syllabus_topic,
    get_syllabus_units,
    get_all_topics_for_subject,
    load_complete_syllabus
)


class StudyDocumentService:
    """
    Authoritative document ingestion and syllabus extraction service.
    Parses uploaded PDFs, text files, and markdown into a structured hierarchy:
    Subject -> Unit -> Topic -> Subtopics.
    Preserves exact source definitions, terminology, and listed subtopics.
    """

    ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".json"}
    MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

    @classmethod
    def validate_file(cls, filename: str, file_size: int) -> Tuple[bool, Optional[str]]:
        """Validates file extension and size."""
        ext = Path(filename).suffix.lower()
        if ext not in cls.ALLOWED_EXTENSIONS:
            return False, f"Unsupported file type '{ext}'. Allowed types: PDF, TXT, MD, JSON."
        if file_size > cls.MAX_FILE_SIZE_BYTES:
            return False, f"File size exceeds the 20MB limit ({file_size / (1024*1024):.1f}MB)."
        return True, None

    @classmethod
    def extract_text_from_pdf(cls, file_path: Path) -> Tuple[str, int, bool]:
        """
        Extracts plain text and page count from a PDF file.
        Returns: (extracted_text, page_count, is_scanned_image)
        """
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            pages = reader.pages
            page_count = len(pages)
            extracted_pages = []
            for page in pages:
                txt = page.extract_text() or ""
                if txt.strip():
                    extracted_pages.append(txt.strip())

            full_text = "\n\n".join(extracted_pages)
            is_scanned = (page_count > 0 and len(full_text.strip()) < 50)
            return full_text, page_count, is_scanned
        except Exception as e:
            print(f"[StudyDocService] PDF parse error: {e}")
            return "", 0, False

    @classmethod
    def parse_document_text_to_hierarchy(
        cls,
        text: str,
        document_title: str
    ) -> Dict[str, Any]:
        """
        Extracts hierarchical units, topics, and subtopics from text.
        Guarantees:
        - Source headings define units/topics.
        - Listed bullet points/sub-items become subtopics.
        - No extra subtopics or general knowledge facts are invented.
        """
        clean_text = text.strip()
        if not clean_text:
            return {
                "name": document_title,
                "description": f"Syllabus extracted from {document_title}",
                "units": []
            }

        lines = [line.strip() for line in clean_text.splitlines() if line.strip()]

        # Pattern detectors for units and topics
        unit_regex = re.compile(
            r"^(?:unit|chapter|module|section|part)\s*(\d+)[\s:.-]+(.+)$",
            re.IGNORECASE
        )
        topic_regex = re.compile(
            r"^(?:topic|lesson|\d+\.\d+)\s*[:.-]?\s*(.+)$",
            re.IGNORECASE
        )
        subtopic_bullet_regex = re.compile(
            r"^[-*•]\s+(.+)$|^\(?([a-z]|[0-9]+)\)\s+(.+)$",
            re.IGNORECASE
        )

        units: List[Dict[str, Any]] = []
        current_unit: Optional[Dict[str, Any]] = None
        current_topic: Optional[Dict[str, Any]] = None
        current_body_lines: List[str] = []

        def flush_topic():
            nonlocal current_topic, current_body_lines
            if current_topic:
                body = " ".join(current_body_lines).strip()
                if not current_topic.get("content"):
                    current_topic["content"] = body or f"Details for {current_topic['topic_name']} from {document_title}."
                if not current_topic.get("summary"):
                    current_topic["summary"] = body[:200] + ("..." if len(body) > 200 else "") or current_topic["content"]
                if not current_topic.get("simple_explanation"):
                    current_topic["simple_explanation"] = current_topic["summary"]
                current_body_lines = []

        def flush_unit():
            nonlocal current_unit
            flush_topic()
            if current_unit and current_unit.get("topics"):
                units.append(current_unit)

        for line in lines:
            # Check Unit header
            u_match = unit_regex.match(line)
            if u_match:
                flush_unit()
                unit_num = int(u_match.group(1))
                unit_title = u_match.group(2).strip()
                current_unit = {
                    "unit_number": unit_num,
                    "title": f"Unit {unit_num}: {unit_title}",
                    "description": f"Material from {document_title}",
                    "topics": []
                }
                current_topic = None
                continue

            # Markdown H1 / H2 headers as units if no explicit "Unit X"
            if line.startswith("# ") or line.startswith("## "):
                h_title = re.sub(r"^#+\s*", "", line).strip()
                if not current_unit or len(current_unit["topics"]) > 0:
                    flush_unit()
                    unit_num = len(units) + 1
                    current_unit = {
                        "unit_number": unit_num,
                        "title": h_title,
                        "description": f"Extracted section from {document_title}",
                        "topics": []
                    }
                    current_topic = None
                    continue

            # Check Topic header (e.g. "### Topic Name", "Topic: Name", "1.1 Name")
            t_match = topic_regex.match(line) or (line.startswith("### ") or line.startswith("#### "))
            if t_match:
                flush_topic()
                if isinstance(t_match, re.Match):
                    t_title = t_match.group(1).strip()
                else:
                    t_title = re.sub(r"^#+\s*", "", line).strip()

                if not current_unit:
                    current_unit = {
                        "unit_number": 1,
                        "title": f"Unit 1: {document_title}",
                        "description": f"Extracted from {document_title}",
                        "topics": []
                    }

                current_topic = {
                    "topic_name": t_title,
                    "content": "",
                    "summary": "",
                    "simple_explanation": "",
                    "subtopics": [],
                    "examples": [],
                    "key_terms": [],
                    "sample_questions": [],
                    "order_index": len(current_unit["topics"])
                }
                current_unit["topics"].append(current_topic)
                current_body_lines = []
                continue

            # Check Subtopic bullet points under current topic
            sb_match = subtopic_bullet_regex.match(line)
            if sb_match and current_topic:
                sub_title = (sb_match.group(1) or sb_match.group(3) or "").strip()
                if sub_title and sub_title not in current_topic["subtopics"]:
                    # Preserve exact listed subtopic
                    current_topic["subtopics"].append(sub_title)
                    continue

            # Check Key Terms (e.g., "Term: Meaning" or "Definition: ...")
            if ":" in line and current_topic and not line.startswith("http"):
                parts = line.split(":", 1)
                term = parts[0].strip()
                meaning = parts[1].strip()
                if 2 < len(term) < 40 and len(meaning) > 5:
                    current_topic["key_terms"].append({"term": term, "meaning": meaning})

            # Normal content line
            if current_topic:
                current_body_lines.append(line)
            else:
                # Content before any unit/topic - create initial unit & topic if needed
                if not current_unit:
                    current_unit = {
                        "unit_number": 1,
                        "title": f"Unit 1: Core Syllabus",
                        "description": f"Extracted from {document_title}",
                        "topics": []
                    }
                if not current_topic:
                    current_topic = {
                        "topic_name": line if len(line) < 60 else "Overview",
                        "content": "",
                        "summary": "",
                        "simple_explanation": "",
                        "subtopics": [],
                        "examples": [],
                        "key_terms": [],
                        "sample_questions": [],
                        "order_index": len(current_unit["topics"])
                    }
                    current_unit["topics"].append(current_topic)
                current_body_lines.append(line)

        flush_unit()

        # If nothing was parsed into units, create a single clean topic from full text
        if not units and clean_text:
            first_line = lines[0] if lines else document_title
            t_name = first_line[:60].strip() or document_title
            units = [{
                "unit_number": 1,
                "title": f"Unit 1: {document_title}",
                "description": f"Material from {document_title}",
                "topics": [{
                    "topic_name": t_name,
                    "content": clean_text,
                    "summary": clean_text[:250] + ("..." if len(clean_text) > 250 else ""),
                    "simple_explanation": clean_text[:250],
                    "subtopics": [],
                    "examples": [],
                    "key_terms": [],
                    "sample_questions": [],
                    "order_index": 0
                }]
            }]

        return {
            "name": document_title,
            "description": f"Syllabus extracted from {document_title}",
            "units": units
        }

    @classmethod
    def index_and_load_study_material(
        cls,
        file_path: Path,
        original_filename: str,
        student_id: int,
        target_subject_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Parses document, saves document record, and registers units & topics into the syllabus database.
        Returns detailed summary including unit and topic counts.
        """
        ext = file_path.suffix.lower()
        doc_title = Path(original_filename).stem.replace("_", " ").title()

        if ext == ".json":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    syllabus_json = json.load(f)
                sub_id = load_complete_syllabus(syllabus_json)
                cls._record_study_document(student_id, sub_id, original_filename, str(file_path), doc_title, 1)
                return {
                    "success": True,
                    "subject_id": sub_id,
                    "message": f"Syllabus '{doc_title}' loaded successfully from JSON.",
                    "units_count": len(syllabus_json.get("units", [])),
                    "topics_count": sum(len(u.get("topics", [])) for u in syllabus_json.get("units", []))
                }
            except Exception as e:
                raise ValueError(f"Invalid JSON syllabus format: {str(e)}")

        # Extract text from PDF or Text/Markdown
        is_scanned = False
        page_count = 1
        if ext == ".pdf":
            text, page_count, is_scanned = cls.extract_text_from_pdf(file_path)
            if is_scanned:
                raise ValueError(
                    "This PDF appears to be a scanned image without a selectable text layer. "
                    "OCR is currently unavailable in this environment, so text could not be extracted."
                )
        else:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception as e:
                raise ValueError(f"Failed to read file: {str(e)}")

        if not text.strip():
            raise ValueError(f"The document '{original_filename}' is empty or no readable text could be extracted.")

        # Parse text into structured hierarchy
        hierarchy = cls.parse_document_text_to_hierarchy(text, doc_title)
        units = hierarchy.get("units", [])
        if not units:
            raise ValueError("No syllabus units or topics could be extracted from this document.")

        # Determine target subject
        if target_subject_id:
            sub = get_subject(target_subject_id)
            sub_id = target_subject_id if sub else create_subject(doc_title, f"Extracted from {original_filename}")
        else:
            sub_id = create_subject(doc_title, f"Extracted from {original_filename}")

        # Record study document
        doc_id = cls._record_study_document(
            student_id=student_id,
            subject_id=sub_id,
            filename=original_filename,
            file_path=str(file_path),
            title=doc_title,
            page_count=page_count
        )

        # Clear old units for this subject on fresh document upload to avoid duplicates
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM syllabus_units WHERE subject_id = ?", (sub_id,))
        conn.commit()
        conn.close()

        total_topics = 0
        for u in units:
            unit_num = u.get("unit_number", 1)
            u_title = u.get("title", f"Unit {unit_num}")
            u_desc = u.get("description", "")
            unit_id = add_syllabus_unit(sub_id, u_title, unit_num, u_desc)

            for idx, t in enumerate(u.get("topics", [])):
                t_name = t.get("topic_name") or t.get("title") or f"Topic {idx+1}"
                t_content = t.get("content") or f"Content for {t_name} from {original_filename}."
                t_summary = t.get("summary") or t_content[:200]
                t_simple = t.get("simple_explanation") or t_summary
                subtopics = t.get("subtopics") or []
                examples = t.get("examples") or []
                key_terms = t.get("key_terms") or []
                sample_questions = t.get("sample_questions") or []

                cls._add_topic_with_subtopics(
                    unit_id=unit_id,
                    topic_name=t_name,
                    summary=t_summary,
                    content=t_content,
                    simple_explanation=t_simple,
                    subtopics=subtopics,
                    examples=examples,
                    key_terms=key_terms,
                    sample_questions=sample_questions,
                    order_index=idx,
                    source_doc_id=doc_id
                )
                total_topics += 1

        return {
            "success": True,
            "subject_id": sub_id,
            "document_id": doc_id,
            "message": f"Successfully indexed '{original_filename}': {len(units)} units and {total_topics} topics.",
            "units_count": len(units),
            "topics_count": total_topics
        }

    @classmethod
    def _record_study_document(
        cls,
        student_id: int,
        subject_id: int,
        filename: str,
        file_path: str,
        title: str,
        page_count: int
    ) -> int:
        """Stores or updates study_documents table."""
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS study_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                subject_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                title TEXT NOT NULL,
                page_count INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            INSERT INTO study_documents (student_id, subject_id, filename, file_path, title, page_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (student_id, subject_id, filename, file_path, title, page_count))
        doc_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return doc_id

    @classmethod
    def _add_topic_with_subtopics(
        cls,
        unit_id: int,
        topic_name: str,
        summary: str,
        content: str,
        simple_explanation: str,
        subtopics: List[str],
        examples: List[str],
        key_terms: List[Dict[str, str]],
        sample_questions: List[Dict[str, Any]],
        order_index: int,
        source_doc_id: Optional[int] = None
    ) -> int:
        """Inserts a syllabus topic with subtopics and source doc reference."""
        conn = get_connection()
        cursor = conn.cursor()
        # Ensure columns exist in syllabus_topics
        try:
            cursor.execute("ALTER TABLE syllabus_topics ADD COLUMN subtopics TEXT DEFAULT '[]'")
        except Exception:
            pass
        try:
            cursor.execute("ALTER TABLE syllabus_topics ADD COLUMN source_document_id INTEGER DEFAULT NULL")
        except Exception:
            pass

        cursor.execute("""
            INSERT INTO syllabus_topics (
                unit_id, topic_name, summary, content, simple_explanation,
                subtopics, examples, key_terms, sample_questions, order_index, source_document_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            unit_id,
            topic_name.strip(),
            summary.strip(),
            content.strip(),
            simple_explanation.strip(),
            json.dumps(subtopics or []),
            json.dumps(examples or []),
            json.dumps(key_terms or []),
            json.dumps(sample_questions or []),
            order_index,
            source_doc_id
        ))
        top_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return top_id

    @classmethod
    def list_study_documents(cls, subject_id: int) -> List[Dict[str, Any]]:
        """Returns all documents uploaded for a specific subject."""
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT * FROM study_documents WHERE subject_id = ? ORDER BY id DESC
            """, (subject_id,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
        finally:
            conn.close()


study_doc_service = StudyDocumentService()
