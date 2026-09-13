import sqlite3
import json
import secrets
from datetime import datetime
from typing import Dict, Any, List, Optional
from app.config import settings
from app.db.models import SystemSettings, StudentResponse, DeviceResponse


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.DATABASE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they do not exist and seed default records."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. System Settings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_settings (
            id INTEGER PRIMARY KEY,
            active_provider TEXT DEFAULT 'groq',
            groq_api_key TEXT DEFAULT '',
            openai_api_key TEXT DEFAULT '',
            qwen_api_key TEXT DEFAULT '',
            ollama_base_url TEXT DEFAULT 'http://localhost:11434',
            groq_model TEXT DEFAULT 'llama-3.3-70b-versatile',
            openai_model TEXT DEFAULT 'gpt-4o-mini',
            qwen_model TEXT DEFAULT 'qwen/qwen-2.5-72b-instruct',
            ollama_model TEXT DEFAULT 'llama3:latest',
            system_prompt TEXT,
            coaching_mode TEXT DEFAULT 'coach',
            grammar_strictness TEXT DEFAULT 'balanced',
            tts_voice TEXT DEFAULT 'en-US-GuyNeural',
            tts_rate TEXT DEFAULT '+0%',
            tts_pitch TEXT DEFAULT '+0Hz',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Students
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            grade TEXT DEFAULT 'Grade 8',
            native_language TEXT DEFAULT 'English',
            target_level TEXT DEFAULT 'Intermediate',
            interests TEXT DEFAULT 'Science, Robotics, Storytelling',
            learning_goals TEXT DEFAULT 'Improve speaking fluency and master science concepts',
            grammar_score REAL DEFAULT 75.0,
            vocabulary_score REAL DEFAULT 70.0,
            fluency_score REAL DEFAULT 80.0,
            confidence_score REAL DEFAULT 78.0,
            total_sessions INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 3. Conversations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            audio_path TEXT,
            grammar_errors TEXT,
            vocabulary_suggestions TEXT,
            fluency_score REAL,
            learning_mode TEXT DEFAULT 'coach',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE
        )
    """)

    # 4. Knowledge Documents & Chunks
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            title TEXT NOT NULL,
            chunks_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES knowledge_docs (id) ON DELETE CASCADE
        )
    """)

    # 5. ESP32 Device Authorization
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS device_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_name TEXT NOT NULL,
            api_key TEXT UNIQUE NOT NULL,
            is_active INTEGER DEFAULT 1,
            last_seen_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()

    # Seed default system settings if empty
    cursor.execute("SELECT COUNT(*) FROM system_settings")
    if cursor.fetchone()[0] == 0:
        default_settings = SystemSettings(
            groq_api_key=settings.GROQ_API_KEY or "",
            openai_api_key=settings.OPENAI_API_KEY or "",
            qwen_api_key=settings.QWEN_API_KEY or "",
            ollama_base_url=settings.OLLAMA_BASE_URL,
            groq_model=settings.GROQ_MODEL,
            openai_model=settings.OPENAI_MODEL,
            qwen_model=settings.QWEN_MODEL,
            tts_voice=settings.DEFAULT_TTS_VOICE,
        )
        cursor.execute("""
            INSERT INTO system_settings (
                id, active_provider, groq_api_key, openai_api_key, qwen_api_key,
                ollama_base_url, groq_model, openai_model, qwen_model,
                system_prompt, coaching_mode, grammar_strictness, tts_voice, tts_rate, tts_pitch
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            default_settings.active_provider,
            default_settings.groq_api_key,
            default_settings.openai_api_key,
            default_settings.qwen_api_key,
            default_settings.ollama_base_url,
            default_settings.groq_model,
            default_settings.openai_model,
            default_settings.qwen_model,
            default_settings.system_prompt,
            default_settings.coaching_mode,
            default_settings.grammar_strictness,
            default_settings.tts_voice,
            default_settings.tts_rate,
            default_settings.tts_pitch
        ))
        conn.commit()

    # Seed default student if empty
    cursor.execute("SELECT COUNT(*) FROM students")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO students (
                name, grade, native_language, target_level, interests, learning_goals
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "Alex Mercer",
            "Grade 8",
            "English (Learner)",
            "Intermediate",
            "Robotics, Space, Science, Storytelling",
            "Improve speaking fluency, eliminate grammar hesitations, and master physics concepts"
        ))
        conn.commit()

    # Seed default ESP32 Device Key if empty
    cursor.execute("SELECT COUNT(*) FROM device_keys")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO device_keys (device_name, api_key, is_active)
            VALUES (?, ?, 1)
        """, ("ESP32-S3 Classroom Robot", settings.DEFAULT_DEVICE_KEY))
        conn.commit()

    conn.close()


# -------------------------------------------------------------------
# Database Operations: System Settings
# -------------------------------------------------------------------
def get_db_settings() -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM system_settings WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return SystemSettings().model_dump()


def update_db_settings(updates: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()

    valid_fields = [
        "active_provider", "groq_api_key", "openai_api_key", "qwen_api_key",
        "ollama_base_url", "groq_model", "openai_model", "qwen_model", "ollama_model",
        "system_prompt", "coaching_mode", "grammar_strictness", "tts_voice",
        "tts_rate", "tts_pitch"
    ]

    set_clauses = []
    params = []
    for k, v in updates.items():
        if k in valid_fields and v is not None:
            set_clauses.append(f"{k} = ?")
            params.append(v)

    if set_clauses:
        query = f"UPDATE system_settings SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP WHERE id = 1"
        cursor.execute(query, params)
        conn.commit()

    conn.close()
    return get_db_settings()


# -------------------------------------------------------------------
# Database Operations: Students
# -------------------------------------------------------------------
def get_all_students() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_student(student_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def create_student(data: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO students (name, grade, native_language, target_level, interests, learning_goals)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        data.get("name", "Student"),
        data.get("grade", "Grade 8"),
        data.get("native_language", "English"),
        data.get("target_level", "Intermediate"),
        data.get("interests", "General"),
        data.get("learning_goals", "Learn English and STEM")
    ))
    student_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_student(student_id)  # type: ignore


def update_student(student_id: int, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    valid_fields = ["name", "grade", "native_language", "target_level", "interests", "learning_goals"]
    clauses = []
    params = []
    for k, v in updates.items():
        if k in valid_fields and v is not None:
            clauses.append(f"{k} = ?")
            params.append(v)
    if clauses:
        params.append(student_id)
        cursor.execute(f"UPDATE students SET {', '.join(clauses)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?", params)
        conn.commit()
    conn.close()
    return get_student(student_id)


def update_student_scores(student_id: int, grammar_delta: float, vocab_delta: float, fluency_val: float):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT grammar_score, vocabulary_score, fluency_score, confidence_score, total_sessions FROM students WHERE id = ?", (student_id,))
    row = cursor.fetchone()
    if row:
        g = min(100.0, max(10.0, row["grammar_score"] + grammar_delta))
        v = min(100.0, max(10.0, row["vocabulary_score"] + vocab_delta))
        # Weighted average for fluency
        fl = min(100.0, max(10.0, (row["fluency_score"] * 0.8) + (fluency_val * 0.2)))
        conf = round((g * 0.3) + (v * 0.3) + (fl * 0.4), 1)
        sessions = row["total_sessions"] + 1

        cursor.execute("""
            UPDATE students
            SET grammar_score = ?, vocabulary_score = ?, fluency_score = ?, confidence_score = ?, total_sessions = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (round(g, 1), round(v, 1), round(fl, 1), conf, sessions, student_id))
        conn.commit()
    conn.close()


# -------------------------------------------------------------------
# Database Operations: Conversations
# -------------------------------------------------------------------
def add_conversation_message(
    student_id: int,
    session_id: str,
    role: str,
    content: str,
    audio_path: Optional[str] = None,
    grammar_errors: Optional[List[Any]] = None,
    vocabulary_suggestions: Optional[List[str]] = None,
    fluency_score: Optional[float] = None,
    learning_mode: str = "coach"
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversations (
            student_id, session_id, role, content, audio_path,
            grammar_errors, vocabulary_suggestions, fluency_score, learning_mode
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        student_id,
        session_id,
        role,
        content,
        audio_path,
        json.dumps(grammar_errors or []),
        json.dumps(vocabulary_suggestions or []),
        fluency_score,
        learning_mode
    ))
    msg_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return msg_id


def get_conversation_history(student_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    if student_id:
        cursor.execute("""
            SELECT * FROM conversations WHERE student_id = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (student_id, limit))
    else:
        cursor.execute("SELECT * FROM conversations ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in reversed(rows):
        item = dict(r)
        try:
            item["grammar_errors"] = json.loads(item.get("grammar_errors") or "[]")
        except Exception:
            item["grammar_errors"] = []
        try:
            item["vocabulary_suggestions"] = json.loads(item.get("vocabulary_suggestions") or "[]")
        except Exception:
            item["vocabulary_suggestions"] = []
        results.append(item)
    return results


# -------------------------------------------------------------------
# Database Operations: Knowledge Base
# -------------------------------------------------------------------
def add_knowledge_doc(filename: str, file_path: str, title: str, chunks_count: int) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO knowledge_docs (filename, file_path, title, chunks_count)
        VALUES (?, ?, ?, ?)
    """, (filename, file_path, title, chunks_count))
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doc_id


def add_knowledge_chunk(doc_id: int, chunk_index: int, content: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO knowledge_chunks (doc_id, chunk_index, content)
        VALUES (?, ?, ?)
    """, (doc_id, chunk_index, content))
    conn.commit()
    conn.close()


def get_all_knowledge_docs() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM knowledge_docs ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_chunks() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, doc_id, chunk_index, content FROM knowledge_chunks")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_knowledge_doc(doc_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM knowledge_chunks WHERE doc_id = ?", (doc_id,))
    cursor.execute("DELETE FROM knowledge_docs WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()


# -------------------------------------------------------------------
# Database Operations: Devices
# -------------------------------------------------------------------
def get_all_devices() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM device_keys ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_device(device_name: str) -> Dict[str, Any]:
    api_key = f"mizo_{secrets.token_hex(16)}"
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO device_keys (device_name, api_key, is_active)
        VALUES (?, ?, 1)
    """, (device_name, api_key))
    dev_id = cursor.lastrowid
    conn.commit()
    cursor.execute("SELECT * FROM device_keys WHERE id = ?", (dev_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row)


def validate_device_key(api_key: str) -> bool:
    if not api_key:
        return False
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM device_keys WHERE api_key = ? AND is_active = 1", (api_key,))
    row = cursor.fetchone()
    if row:
        cursor.execute("UPDATE device_keys SET last_seen_at = CURRENT_TIMESTAMP WHERE id = ?", (row["id"],))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False


def delete_device(device_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM device_keys WHERE id = ?", (device_id,))
    conn.commit()
    conn.close()
