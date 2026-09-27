import sqlite3
import json
import secrets
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from app.config import settings
from app.db.models import SystemSettings, StudentResponse, DeviceResponse


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.DATABASE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(cursor: sqlite3.Cursor, table: str, column: str, col_type: str, default: Any = None):
    """Safely adds column to SQLite table if it doesn't exist."""
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    if column not in columns:
        def_clause = f" DEFAULT {default}" if default is not None else ""
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}{def_clause}")


def init_db():
    """Create tables if they do not exist and ensure columns exist."""
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

    # 2. Students (Unified Learner Profile & Long-Term Memory)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            education TEXT DEFAULT 'Student',
            grade TEXT DEFAULT 'Grade 8',
            native_language TEXT DEFAULT 'English',
            target_level TEXT DEFAULT 'Intermediate',
            interests TEXT DEFAULT 'Science, Robotics, Storytelling',
            learning_goals TEXT DEFAULT 'Improve speaking fluency and master communication',
            onboarding_completed INTEGER DEFAULT 1,
            onboarding_step TEXT DEFAULT 'completed',
            baseline_grammar REAL DEFAULT 75.0,
            baseline_vocabulary REAL DEFAULT 70.0,
            baseline_fluency REAL DEFAULT 80.0,
            baseline_pronunciation REAL DEFAULT 75.0,
            baseline_confidence REAL DEFAULT 75.0,
            baseline_communication REAL DEFAULT 75.0,
            grammar_score REAL DEFAULT 75.0,
            vocabulary_score REAL DEFAULT 70.0,
            fluency_score REAL DEFAULT 80.0,
            pronunciation_score REAL DEFAULT 75.0,
            confidence_score REAL DEFAULT 78.0,
            communication_score REAL DEFAULT 75.0,
            total_sessions INTEGER DEFAULT 0,
            strengths TEXT DEFAULT '["Active participation", "Good pronunciation"]',
            weaknesses TEXT DEFAULT '["Subject-verb agreement", "Filler words"]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Check for newly added columns in students table
    _ensure_column(cursor, "students", "education", "TEXT", "'Student'")
    _ensure_column(cursor, "students", "self_reported_level", "TEXT", "'Not sure'")
    _ensure_column(cursor, "students", "learning_topics", "TEXT", "''")
    _ensure_column(cursor, "students", "profile_version", "INTEGER", 1)
    _ensure_column(cursor, "students", "speaking_assessment_completed", "INTEGER", 0)
    _ensure_column(cursor, "students", "onboarding_completed", "INTEGER", 1)
    _ensure_column(cursor, "students", "onboarding_step", "TEXT", "'completed'")
    _ensure_column(cursor, "students", "baseline_grammar", "REAL", 75.0)
    _ensure_column(cursor, "students", "baseline_vocabulary", "REAL", 70.0)
    _ensure_column(cursor, "students", "baseline_fluency", "REAL", 80.0)
    _ensure_column(cursor, "students", "baseline_pronunciation", "REAL", 75.0)
    _ensure_column(cursor, "students", "baseline_confidence", "REAL", 75.0)
    _ensure_column(cursor, "students", "baseline_communication", "REAL", 75.0)
    _ensure_column(cursor, "students", "pronunciation_score", "REAL", 75.0)
    _ensure_column(cursor, "students", "communication_score", "REAL", 75.0)
    _ensure_column(cursor, "students", "strengths", "TEXT", "'[]'")
    _ensure_column(cursor, "students", "weaknesses", "TEXT", "'[]'")

    # Profile Archives / Snapshots Table (Preserves historical profile versions across restarts)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS profile_archives (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            profile_version INTEGER NOT NULL,
            name TEXT,
            education TEXT,
            learning_goals TEXT,
            learning_topics TEXT,
            weaknesses TEXT,
            strengths TEXT,
            self_reported_level TEXT,
            baseline_grammar REAL,
            baseline_vocabulary REAL,
            baseline_fluency REAL,
            baseline_pronunciation REAL,
            baseline_confidence REAL,
            baseline_communication REAL,
            snapshot_reason TEXT DEFAULT 'restart_onboarding',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    """)

    # 3. Learning Sessions (Tracks active session metrics separately from history)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            student_id INTEGER NOT NULL,
            mode TEXT DEFAULT 'coach',
            status TEXT DEFAULT 'active',
            session_grammar REAL DEFAULT 0.0,
            session_fluency REAL DEFAULT 0.0,
            session_vocabulary REAL DEFAULT 0.0,
            session_pronunciation REAL DEFAULT 0.0,
            session_confidence REAL DEFAULT 0.0,
            session_communication REAL DEFAULT 0.0,
            session_pacing REAL DEFAULT 0.0,
            message_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE
        )
    """)
    _ensure_column(cursor, "learning_sessions", "session_pronunciation", "REAL", 0.0)
    _ensure_column(cursor, "learning_sessions", "session_communication", "REAL", 0.0)

    # 4. Student Mistakes Log (Granular memory for personalized guidance)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_mistakes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            utterance TEXT NOT NULL,
            mistake_type TEXT DEFAULT 'grammar',
            error_text TEXT NOT NULL,
            correction TEXT NOT NULL,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE
        )
    """)

    # 5. Assessments (Speaking Assessments, Evidence, Reasoning & Baselines)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            topic TEXT,
            transcript TEXT NOT NULL,
            duration_seconds REAL DEFAULT 0.0,
            grammar_score REAL DEFAULT 0.0,
            vocabulary_score REAL DEFAULT 0.0,
            fluency_score REAL DEFAULT 0.0,
            pronunciation_score REAL DEFAULT 0.0,
            confidence_score REAL DEFAULT 0.0,
            communication_score REAL DEFAULT 0.0,
            overall_level TEXT DEFAULT 'Intermediate',
            grammar_feedback TEXT,
            vocabulary_feedback TEXT,
            fluency_feedback TEXT,
            pronunciation_feedback TEXT,
            confidence_feedback TEXT,
            communication_feedback TEXT,
            strengths TEXT DEFAULT '[]',
            weaknesses TEXT DEFAULT '[]',
            raw_analysis_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE
        )
    """)

    # 6. Conversations
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
            pacing_score REAL,
            confidence_score REAL,
            learning_mode TEXT DEFAULT 'coach',
            onboarding_state TEXT,
            intent TEXT,
            provider_used TEXT DEFAULT 'groq',
            is_fallback INTEGER DEFAULT 0,
            latency_ms REAL DEFAULT 0.0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE
        )
    """)

    _ensure_column(cursor, "conversations", "pacing_score", "REAL", 0.0)
    _ensure_column(cursor, "conversations", "confidence_score", "REAL", 0.0)
    _ensure_column(cursor, "conversations", "onboarding_state", "TEXT", "NULL")
    _ensure_column(cursor, "conversations", "intent", "TEXT", "NULL")
    _ensure_column(cursor, "conversations", "provider_used", "TEXT", "'groq'")
    _ensure_column(cursor, "conversations", "is_fallback", "INTEGER", 0)
    _ensure_column(cursor, "conversations", "latency_ms", "REAL", 0.0)

    # 7. Knowledge Documents & Chunks
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

    # 8. ESP32 Device Authorization
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
                name, grade, native_language, target_level, interests, learning_goals,
                onboarding_completed, onboarding_step, baseline_grammar, baseline_vocabulary, baseline_fluency,
                grammar_score, vocabulary_score, fluency_score, confidence_score, total_sessions,
                strengths, weaknesses
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "New Learner",
            "Grade 8",
            "English (Learner)",
            "Intermediate",
            "Science, General Learning",
            "Improve speaking fluency and master concepts",
            0,
            "ask_name",
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            json.dumps([]),
            json.dumps([])
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
# Database Operations: System Settings & .env Sync
# -------------------------------------------------------------------
def _sync_keys_to_env_file(updates: Dict[str, Any]):
    """Persists API keys and provider configurations to .env file so they survive all updates."""
    import os
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return

    env_path = settings.BASE_DIR / ".env"
    existing_lines = []
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()

    env_dict = {}
    for line in existing_lines:
        line_str = line.strip()
        if line_str and not line_str.startswith("#") and "=" in line_str:
            k, v = line_str.split("=", 1)
            env_dict[k.strip()] = v.strip()

    key_map = {
        "groq_api_key": "GROQ_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
        "qwen_api_key": "QWEN_API_KEY",
        "ollama_base_url": "OLLAMA_BASE_URL",
        "groq_model": "GROQ_MODEL",
        "openai_model": "OPENAI_MODEL",
        "qwen_model": "QWEN_MODEL",
        "ollama_model": "OLLAMA_MODEL",
        "active_provider": "DEFAULT_AI_PROVIDER",
        "tts_voice": "DEFAULT_TTS_VOICE",
        "tts_rate": "DEFAULT_TTS_RATE",
    }

    import os
    for db_key, env_key in key_map.items():
        if db_key in updates and updates[db_key] is not None:
            val = str(updates[db_key]).strip()
            if val:
                env_dict[env_key] = val
                os.environ[env_key] = val
                if hasattr(settings, env_key):
                    setattr(settings, env_key, val)

    # Write back to .env
    with open(env_path, "w", encoding="utf-8") as f:
        for k, v in env_dict.items():
            f.write(f"{k}={v}\n")


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
            # Prevent empty strings from overwriting an already configured API key
            if k in ["groq_api_key", "openai_api_key", "qwen_api_key"] and not str(v).strip():
                continue
            set_clauses.append(f"{k} = ?")
            params.append(v)

    if set_clauses:
        query = f"UPDATE system_settings SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP WHERE id = 1"
        cursor.execute(query, params)
        conn.commit()

    conn.close()

    # Synchronize keys permanently into .env and runtime settings
    _sync_keys_to_env_file(updates)

    return get_db_settings()


# -------------------------------------------------------------------
# Database Operations: Students & Onboarding
# -------------------------------------------------------------------
def _parse_student_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(d)
    w = item.get("weaknesses")
    if isinstance(w, str):
        try:
            item["weaknesses"] = json.loads(w)
        except Exception:
            item["weaknesses"] = [x.strip() for x in w.split(",") if x.strip()]
    elif not isinstance(w, list):
        item["weaknesses"] = []

    s = item.get("strengths")
    if isinstance(s, str):
        try:
            item["strengths"] = json.loads(s)
        except Exception:
            item["strengths"] = [x.strip() for x in s.split(",") if x.strip()]
    elif not isinstance(s, list):
        item["strengths"] = []

    item["onboarding_completed"] = bool(item.get("onboarding_completed", 0))
    item["education"] = item.get("education") or "Student"
    item["self_reported_level"] = item.get("self_reported_level") or "Not sure"
    item["learning_topics"] = item.get("learning_topics") or ""
    item["profile_version"] = int(item.get("profile_version", 1) or 1)
    item["speaking_assessment_completed"] = bool(item.get("speaking_assessment_completed", 0))
    item["baseline_pronunciation"] = item.get("baseline_pronunciation", 0.0) or 0.0
    item["baseline_confidence"] = item.get("baseline_confidence", 0.0) or 0.0
    item["baseline_communication"] = item.get("baseline_communication", 0.0) or 0.0
    item["pronunciation_score"] = item.get("pronunciation_score", 0.0) or 0.0
    item["communication_score"] = item.get("communication_score", 0.0) or 0.0
    return item


def get_all_students() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [_parse_student_dict(dict(r)) for r in rows]


def get_student(student_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    row = cursor.fetchone()
    conn.close()
    return _parse_student_dict(dict(row)) if row else None


def create_student(data: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    is_onboarding = data.get("onboarding_completed", False)
    onboarding_step = data.get("onboarding_step", "completed" if is_onboarding else "ask_name")
    cursor.execute("""
        INSERT INTO students (
            name, education, grade, native_language, target_level, self_reported_level,
            interests, learning_goals, learning_topics,
            onboarding_completed, onboarding_step, speaking_assessment_completed, profile_version,
            baseline_grammar, baseline_vocabulary, baseline_fluency, baseline_pronunciation, baseline_confidence, baseline_communication,
            grammar_score, vocabulary_score, fluency_score, pronunciation_score, confidence_score, communication_score, total_sessions,
            strengths, weaknesses
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, '[]', '[]')
    """, (
        data.get("name", "Student"),
        data.get("education", "Student"),
        data.get("grade", "Grade 8"),
        data.get("native_language", "English"),
        data.get("target_level", "Intermediate"),
        data.get("self_reported_level", "Pending"),
        data.get("interests", "General Topics"),
        data.get("learning_goals", "Improve communication skills"),
        data.get("learning_topics", "General Topics"),
        1 if is_onboarding else 0,
        onboarding_step
    ))
    student_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_student(student_id)  # type: ignore


def update_student(student_id: int, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    valid_fields = [
        "name", "education", "grade", "native_language", "target_level", "self_reported_level",
        "interests", "learning_goals", "learning_topics", "onboarding_completed", "onboarding_step",
        "speaking_assessment_completed", "profile_version",
        "baseline_grammar", "baseline_vocabulary", "baseline_fluency",
        "baseline_pronunciation", "baseline_confidence", "baseline_communication",
        "grammar_score", "vocabulary_score", "fluency_score", "pronunciation_score",
        "confidence_score", "communication_score",
        "total_sessions", "strengths", "weaknesses"
    ]
    clauses = []
    params = []
    for k, v in updates.items():
        if k in valid_fields and v is not None:
            if k in ["strengths", "weaknesses"]:
                if isinstance(v, list):
                    v = json.dumps(v)
                elif isinstance(v, str) and not v.startswith("["):
                    v = json.dumps([x.strip() for x in v.split(",") if x.strip()])
            elif k in ["onboarding_completed", "speaking_assessment_completed"]:
                v = 1 if v else 0
            clauses.append(f"{k} = ?")
            params.append(v)
    if clauses:
        params.append(student_id)
        cursor.execute(f"UPDATE students SET {', '.join(clauses)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?", params)
        conn.commit()
    conn.close()
    return get_student(student_id)


def record_onboarding_baseline(
    student_id: int,
    grammar: float,
    vocab: float,
    fluency: float,
    pronunciation: float = 75.0,
    confidence: float = 75.0,
    communication: float = 75.0,
    overall_level: str = "Intermediate",
    strengths: Optional[List[str]] = None,
    weaknesses: Optional[List[str]] = None
):
    """Finalizes student onboarding and establishes permanent baseline scores."""
    conn = get_connection()
    cursor = conn.cursor()
    s_list = strengths or ["Active voice participation"]
    w_list = weaknesses or ["Grammar consistency"]
    cursor.execute("""
        UPDATE students
        SET onboarding_completed = 1,
            onboarding_step = 'completed',
            speaking_assessment_completed = 1,
            target_level = ?,
            baseline_grammar = ?,
            baseline_vocabulary = ?,
            baseline_fluency = ?,
            baseline_pronunciation = ?,
            baseline_confidence = ?,
            baseline_communication = ?,
            grammar_score = ?,
            vocabulary_score = ?,
            fluency_score = ?,
            pronunciation_score = ?,
            confidence_score = ?,
            communication_score = ?,
            total_sessions = 1,
            strengths = ?,
            weaknesses = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (
        overall_level,
        round(grammar, 1), round(vocab, 1), round(fluency, 1),
        round(pronunciation, 1), round(confidence, 1), round(communication, 1),
        round(grammar, 1), round(vocab, 1), round(fluency, 1),
        round(pronunciation, 1), round(confidence, 1), round(communication, 1),
        json.dumps(s_list), json.dumps(w_list),
        student_id
    ))
    conn.commit()
    conn.close()


def save_assessment(data: Dict[str, Any]) -> Dict[str, Any]:
    """Persists a detailed speaking assessment record with evidence and reasoning."""
    conn = get_connection()
    cursor = conn.cursor()
    strengths = data.get("strengths", [])
    weaknesses = data.get("weaknesses", [])
    if isinstance(strengths, list):
        strengths_str = json.dumps(strengths)
    else:
        strengths_str = json.dumps([x.strip() for x in str(strengths).split(",") if x.strip()])

    if isinstance(weaknesses, list):
        weaknesses_str = json.dumps(weaknesses)
    else:
        weaknesses_str = json.dumps([x.strip() for x in str(weaknesses).split(",") if x.strip()])

    raw_analysis = data.get("raw_analysis_json")
    if isinstance(raw_analysis, dict):
        raw_analysis = json.dumps(raw_analysis)

    cursor.execute("""
        INSERT INTO assessments (
            student_id, session_id, topic, transcript, duration_seconds,
            grammar_score, vocabulary_score, fluency_score, pronunciation_score,
            confidence_score, communication_score, overall_level,
            grammar_feedback, vocabulary_feedback, fluency_feedback,
            pronunciation_feedback, confidence_feedback, communication_feedback,
            strengths, weaknesses, raw_analysis_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("student_id", 1),
        data.get("session_id", ""),
        data.get("topic", "Self Introduction & Experience"),
        data.get("transcript", ""),
        data.get("duration_seconds", 0.0),
        data.get("grammar_score", 0.0),
        data.get("vocabulary_score", 0.0),
        data.get("fluency_score", 0.0),
        data.get("pronunciation_score", 0.0),
        data.get("confidence_score", 0.0),
        data.get("communication_score", 0.0),
        data.get("overall_level", "Intermediate"),
        data.get("grammar_feedback", ""),
        data.get("vocabulary_feedback", ""),
        data.get("fluency_feedback", ""),
        data.get("pronunciation_feedback", ""),
        data.get("confidence_feedback", ""),
        data.get("communication_feedback", ""),
        strengths_str,
        weaknesses_str,
        raw_analysis
    ))
    assessment_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_assessment_by_id(assessment_id) or {}


def get_assessment_by_id(assessment_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM assessments WHERE id = ?", (assessment_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    item = dict(row)
    try:
        item["strengths"] = json.loads(item.get("strengths") or "[]")
    except Exception:
        item["strengths"] = []
    try:
        item["weaknesses"] = json.loads(item.get("weaknesses") or "[]")
    except Exception:
        item["weaknesses"] = []
    return item


def get_student_assessments(student_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM assessments
        WHERE student_id = ?
        ORDER BY created_at DESC LIMIT ?
    """, (student_id, limit))
    rows = cursor.fetchall()
    conn.close()
    results = []
    for r in rows:
        item = dict(r)
        try:
            item["strengths"] = json.loads(item.get("strengths") or "[]")
        except Exception:
            item["strengths"] = []
        try:
            item["weaknesses"] = json.loads(item.get("weaknesses") or "[]")
        except Exception:
            item["weaknesses"] = []
        results.append(item)
    return results


def restart_onboarding(student_id: int) -> Dict[str, Any]:
    """
    Restarts onboarding for a new run:
    - Archives a snapshot of the current profile into profile_archives.
    - Increments profile_version.
    - Resets active fields: name='New Learner', onboarding_completed=0, onboarding_step='ask_name',
      learning_goals='', learning_topics='', interests='', weaknesses='[]', strengths='[]',
      self_reported_level='Not sure', speaking_assessment_completed=0.
    - PRESERVES all past sessions, conversations, assessments, and mistakes so they remain reviewable in History.
    - Creates a new active session for the fresh onboarding run.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    row = cursor.fetchone()
    if row:
        curr = dict(row)
        ver = curr.get("profile_version", 1) or 1
        cursor.execute("""
            INSERT INTO profile_archives (
                student_id, profile_version, name, education, learning_goals,
                learning_topics, weaknesses, strengths, self_reported_level,
                baseline_grammar, baseline_vocabulary, baseline_fluency,
                baseline_pronunciation, baseline_confidence, baseline_communication,
                snapshot_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'restart_onboarding')
        """, (
            student_id, ver, curr.get("name"), curr.get("education"), curr.get("learning_goals"),
            curr.get("learning_topics", ""), str(curr.get("weaknesses", "[]")), str(curr.get("strengths", "[]")),
            curr.get("self_reported_level", "Not sure"), curr.get("baseline_grammar", 0.0),
            curr.get("baseline_vocabulary", 0.0), curr.get("baseline_fluency", 0.0),
            curr.get("baseline_pronunciation", 0.0), curr.get("baseline_confidence", 0.0),
            curr.get("baseline_communication", 0.0)
        ))

        cursor.execute("""
            UPDATE students
            SET name = 'New Learner',
                education = 'Student',
                learning_goals = '',
                learning_topics = '',
                interests = '',
                self_reported_level = 'Pending',
                onboarding_completed = 0,
                onboarding_step = 'ask_name',
                speaking_assessment_completed = 0,
                profile_version = profile_version + 1,
                baseline_grammar = 0.0,
                baseline_vocabulary = 0.0,
                baseline_fluency = 0.0,
                baseline_pronunciation = 0.0,
                baseline_confidence = 0.0,
                baseline_communication = 0.0,
                strengths = '[]',
                weaknesses = '[]',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (student_id,))
        conn.commit()
    conn.close()

    # Create fresh active session for the restarted onboarding
    create_session(student_id=student_id, mode="coach")

    return get_student(student_id) or {}


def delete_learner_data(student_id: int) -> Dict[str, Any]:
    """
    Explicitly destructive action:
    - Permanently deletes student profile, conversations, assessments, mistakes, sessions, and profile archives.
    - Strictly preserves system_settings, .env, and device_keys.
    - Re-creates a fresh default student at id=1 with onboarding_step='ask_name'.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM student_mistakes WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM conversations WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM assessments WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM learning_sessions WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM profile_archives WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))

    # Re-insert clean default student
    cursor.execute("""
        INSERT INTO students (
            id, name, education, grade, native_language, target_level, self_reported_level,
            interests, learning_goals, learning_topics, onboarding_completed, onboarding_step,
            speaking_assessment_completed, profile_version,
            baseline_grammar, baseline_vocabulary, baseline_fluency, baseline_pronunciation, baseline_confidence, baseline_communication,
            grammar_score, vocabulary_score, fluency_score, pronunciation_score, confidence_score, communication_score, total_sessions,
            strengths, weaknesses
        ) VALUES (
            ?, 'New Learner', 'Student', 'Grade 8', 'English', 'Intermediate', 'Pending',
            '', '', '', 0, 'ask_name', 0, 1,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,
            '[]', '[]'
        )
    """, (student_id,))
    conn.commit()
    conn.close()

    create_session(student_id=student_id, mode="coach")
    return get_student(student_id) or {}


def reset_student_to_fresh(student_id: int) -> Dict[str, Any]:
    """Backward compatibility alias for tests and admin reset."""
    return delete_learner_data(student_id)


def update_student_historical_scores(
    student_id: int,
    grammar_val: float,
    vocab_val: float,
    fluency_val: float,
    pacing_val: float = 80.0,
    pronunciation_val: float = 75.0,
    communication_val: float = 75.0
):
    """
    Updates cumulative historical student metrics using exponential moving averages.
    Historical performance stays intact and evolves over time.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    row = cursor.fetchone()
    if row:
        row_dict = dict(row)
        curr_g = row_dict.get("grammar_score", 75.0) or 75.0
        curr_v = row_dict.get("vocabulary_score", 70.0) or 70.0
        curr_f = row_dict.get("fluency_score", 80.0) or 80.0
        curr_p = row_dict.get("pronunciation_score", 75.0) or 75.0
        curr_c = row_dict.get("communication_score", 75.0) or 75.0

        # Exponential moving average (80% historical + 20% recent performance)
        new_g = min(100.0, max(10.0, (curr_g * 0.8) + (grammar_val * 0.2)))
        new_v = min(100.0, max(10.0, (curr_v * 0.8) + (vocab_val * 0.2)))
        new_f = min(100.0, max(10.0, (curr_f * 0.8) + (fluency_val * 0.2)))
        new_p = min(100.0, max(10.0, (curr_p * 0.8) + (pronunciation_val * 0.2)))
        new_c = min(100.0, max(10.0, (curr_c * 0.8) + (communication_val * 0.2)))
        new_conf = round((new_g * 0.25) + (new_v * 0.2) + (new_f * 0.25) + (new_p * 0.15) + (new_c * 0.15), 1)

        cursor.execute("""
            UPDATE students
            SET grammar_score = ?, vocabulary_score = ?, fluency_score = ?,
                pronunciation_score = ?, communication_score = ?, confidence_score = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (round(new_g, 1), round(new_v, 1), round(new_f, 1), round(new_p, 1), round(new_c, 1), new_conf, student_id))
        conn.commit()
    conn.close()


# -------------------------------------------------------------------
# Database Operations: Sessions Lifecycle
# -------------------------------------------------------------------
def create_session(student_id: int, mode: str = "coach") -> Dict[str, Any]:
    """
    Creates a brand new learning session.
    Current session metrics are initialized to ZERO without deleting historical data.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Mark existing active sessions for this student as completed
    cursor.execute("""
        UPDATE learning_sessions
        SET status = 'completed', updated_at = CURRENT_TIMESTAMP
        WHERE student_id = ? AND status = 'active'
    """, (student_id,))

    # Increment student total_sessions count
    cursor.execute("UPDATE students SET total_sessions = total_sessions + 1 WHERE id = ?", (student_id,))

    session_id = f"ses_{uuid.uuid4().hex[:12]}"
    cursor.execute("""
        INSERT INTO learning_sessions (
            session_id, student_id, mode, status,
            session_grammar, session_fluency, session_vocabulary, session_pacing, session_confidence, message_count
        ) VALUES (?, ?, ?, 'active', 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    """, (session_id, student_id, mode))
    conn.commit()
    conn.close()
    return get_session_by_id(session_id)  # type: ignore


def get_active_session(student_id: int, default_mode: str = "coach") -> Dict[str, Any]:
    """Retrieves active session or creates a new one if none exists."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM learning_sessions
        WHERE student_id = ? AND status = 'active'
        ORDER BY created_at DESC LIMIT 1
    """, (student_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return _format_session_dict(dict(row))
    return create_session(student_id=student_id, mode=default_mode)


def get_session_by_id(session_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM learning_sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    return _format_session_dict(dict(row)) if row else None


def _format_session_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row.get("id"),
        "session_id": row.get("session_id"),
        "student_id": row.get("student_id"),
        "mode": row.get("mode"),
        "status": row.get("status"),
        "metrics": {
            "grammar_accuracy": row.get("session_grammar", 0.0),
            "fluency_score": row.get("session_fluency", 0.0),
            "vocabulary_richness": row.get("session_vocabulary", 0.0),
            "pacing_score": row.get("session_pacing", 0.0),
            "confidence_score": row.get("session_confidence", 0.0),
            "message_count": row.get("message_count", 0),
        },
        "created_at": str(row.get("created_at")),
        "updated_at": str(row.get("updated_at"))
    }


def update_session_metrics(
    session_id: str,
    grammar_score: float,
    fluency_score: float,
    vocab_score: float,
    pacing_score: float = 80.0
) -> Dict[str, Any]:
    """Updates the ongoing metrics for the current active session."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM learning_sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return {}

    msg_count = row["message_count"] + 1
    if msg_count == 1:
        new_g = grammar_score
        new_f = fluency_score
        new_v = vocab_score
        new_p = pacing_score
    else:
        # Running average across messages in this session
        new_g = ((row["session_grammar"] * (msg_count - 1)) + grammar_score) / msg_count
        new_f = ((row["session_fluency"] * (msg_count - 1)) + fluency_score) / msg_count
        new_v = ((row["session_vocabulary"] * (msg_count - 1)) + vocab_score) / msg_count
        new_p = ((row["session_pacing"] * (msg_count - 1)) + pacing_score) / msg_count

    conf = round((new_g * 0.3) + (new_v * 0.3) + (new_f * 0.4), 1)

    cursor.execute("""
        UPDATE learning_sessions
        SET session_grammar = ?, session_fluency = ?, session_vocabulary = ?, session_pacing = ?, session_confidence = ?, message_count = ?, updated_at = CURRENT_TIMESTAMP
        WHERE session_id = ?
    """, (round(new_g, 1), round(new_f, 1), round(new_v, 1), round(new_p, 1), conf, msg_count, session_id))
    conn.commit()
    conn.close()
    return get_session_by_id(session_id) or {}


# -------------------------------------------------------------------
# Database Operations: Mistakes & Memory Engine
# -------------------------------------------------------------------
def log_student_mistake(
    student_id: int,
    session_id: str,
    utterance: str,
    mistake_type: str,
    error_text: str,
    correction: str,
    explanation: Optional[str] = None
):
    """Stores granular mistakes so future AI prompts can reference past patterns."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO student_mistakes (student_id, session_id, utterance, mistake_type, error_text, correction, explanation)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (student_id, session_id, utterance, mistake_type, error_text, correction, explanation))
    conn.commit()
    conn.close()


def get_recent_mistakes(student_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM student_mistakes
        WHERE student_id = ?
        ORDER BY created_at DESC LIMIT ?
    """, (student_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_student_memory_context(student_id: int) -> Dict[str, Any]:
    """Builds a rich context payload of student history for the LLM."""
    student = get_student(student_id)
    if not student:
        return {}

    recent_mistakes = get_recent_mistakes(student_id, limit=4)
    history = get_conversation_history(student_id, limit=6)

    return {
        "student": student,
        "recent_mistakes": recent_mistakes,
        "recent_dialogue": history
    }


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
    pacing_score: Optional[float] = None,
    confidence_score: Optional[float] = None,
    learning_mode: str = "coach",
    onboarding_state: Optional[str] = None,
    intent: Optional[str] = None,
    provider_used: Optional[str] = "groq",
    is_fallback: bool = False,
    latency_ms: float = 0.0
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversations (
            student_id, session_id, role, content, audio_path,
            grammar_errors, vocabulary_suggestions, fluency_score, pacing_score, confidence_score,
            learning_mode, onboarding_state, intent, provider_used, is_fallback, latency_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        student_id,
        session_id,
        role,
        content,
        audio_path,
        json.dumps(grammar_errors or []),
        json.dumps(vocabulary_suggestions or []),
        fluency_score,
        pacing_score,
        confidence_score,
        learning_mode,
        onboarding_state,
        intent,
        provider_used,
        1 if is_fallback else 0,
        latency_ms
    ))
    msg_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return msg_id


def handle_device_start_or_reset(student_id: int = 1) -> Dict[str, Any]:
    """
    Handles physical ESP32 Start / Reset button event:
    - If student exists and onboarding is complete:
      -> Returning user! Loads profile, creates a NEW session (current metrics = 0), preserves all historical data.
    - If student exists and onboarding is NOT complete:
      -> Continues onboarding from the current incomplete state.
    - If student does not exist:
      -> Creates student and begins onboarding from ASK_NAME.
    """
    student = get_student(student_id)
    if not student:
        student = create_student({"name": "New Learner", "onboarding_completed": False, "onboarding_step": "ASK_NAME"})
        student_id = student["id"]

    is_onboarding_done = student.get("onboarding_completed", False)
    current_step = student.get("onboarding_step", "ASK_NAME")

    if is_onboarding_done:
        # Returning user -> Create new session with 0 metrics, preserve historical data
        new_session = create_session(student_id=student_id, mode="coach")
        return {
            "status": "returning_user",
            "student": student,
            "session": new_session,
            "onboarding_active": False,
            "onboarding_step": "completed"
        }
    else:
        # Onboarding in progress -> get or create session, resume current step
        session = get_active_session(student_id=student_id, default_mode="coach")
        return {
            "status": "onboarding_active",
            "student": student,
            "session": session,
            "onboarding_active": True,
            "onboarding_step": current_step
        }


def get_conversation_history(
    student_id: Optional[int] = None,
    session_id: Optional[str] = None,
    limit: int = 50,
    search: Optional[str] = None
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()

    conditions = []
    params: List[Any] = []

    if session_id:
        conditions.append("session_id = ?")
        params.append(session_id)
    elif student_id:
        conditions.append("student_id = ?")
        params.append(student_id)

    if search and search.strip():
        conditions.append("content LIKE ?")
        params.append(f"%{search.strip()}%")

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"SELECT * FROM conversations {where_clause} ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    # Return in chronological order
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
        item["is_fallback"] = bool(item.get("is_fallback", 0))
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


# Ensure database and tables exist on module import
init_db()


