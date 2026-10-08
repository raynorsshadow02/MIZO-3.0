import sqlite3
import json
import secrets
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from app.config import settings
from app.db.models import SystemSettings, StudentResponse, DeviceResponse

PROTOTYPE_STUDENT_ID = 1
PENDING_NAME_CHANGE_TTL_MINUTES = 15


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
            groq_model TEXT DEFAULT 'openai/gpt-oss-120b',
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
            onboarding_completed INTEGER DEFAULT 0,
            onboarding_step TEXT DEFAULT 'ask_name',
            baseline_grammar REAL DEFAULT NULL,
            baseline_vocabulary REAL DEFAULT NULL,
            baseline_fluency REAL DEFAULT NULL,
            baseline_pronunciation REAL DEFAULT NULL,
            baseline_confidence REAL DEFAULT NULL,
            baseline_communication REAL DEFAULT NULL,
            grammar_score REAL DEFAULT NULL,
            vocabulary_score REAL DEFAULT NULL,
            fluency_score REAL DEFAULT NULL,
            pronunciation_score REAL DEFAULT NULL,
            confidence_score REAL DEFAULT NULL,
            communication_score REAL DEFAULT NULL,
            total_sessions INTEGER DEFAULT 0,
            strengths TEXT DEFAULT '[]',
            weaknesses TEXT DEFAULT '[]',
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
    _ensure_column(cursor, "students", "onboarding_completed", "INTEGER", 0)
    _ensure_column(cursor, "students", "onboarding_step", "TEXT", "'ask_name'")
    _ensure_column(cursor, "students", "baseline_grammar", "REAL", None)
    _ensure_column(cursor, "students", "baseline_vocabulary", "REAL", None)
    _ensure_column(cursor, "students", "baseline_fluency", "REAL", None)
    _ensure_column(cursor, "students", "baseline_pronunciation", "REAL", None)
    _ensure_column(cursor, "students", "baseline_confidence", "REAL", None)
    _ensure_column(cursor, "students", "baseline_communication", "REAL", None)
    _ensure_column(cursor, "students", "pronunciation_score", "REAL", None)
    _ensure_column(cursor, "students", "communication_score", "REAL", None)
    _ensure_column(cursor, "students", "strengths", "TEXT", "'[]'")
    _ensure_column(cursor, "students", "weaknesses", "TEXT", "'[]'")
    _ensure_column(cursor, "students", "assessed_level", "TEXT", "NULL")
    _ensure_column(cursor, "students", "grammar_level", "TEXT", "'Beginner'")
    _ensure_column(cursor, "students", "vocabulary_level", "TEXT", "'Beginner'")
    _ensure_column(cursor, "students", "fluency_level", "TEXT", "'Beginner'")
    _ensure_column(cursor, "students", "sentence_formation_level", "TEXT", "'Beginner'")
    _ensure_column(cursor, "students", "communication_confidence", "TEXT", "'Low'")
    _ensure_column(cursor, "students", "speaking_hesitation", "TEXT", "'Frequent'")
    _ensure_column(cursor, "students", "presentation_confidence", "TEXT", "'Low'")
    _ensure_column(cursor, "students", "preferred_explanation_difficulty", "INTEGER", 1)
    _ensure_column(cursor, "students", "current_teaching_difficulty", "INTEGER", 1)
    _ensure_column(cursor, "students", "known_words", "TEXT", "'[]'")
    _ensure_column(cursor, "students", "difficult_words", "TEXT", "'[]'")
    _ensure_column(cursor, "students", "recurring_grammar_errors", "TEXT", "'[]'")
    _ensure_column(cursor, "students", "mastered_topics", "TEXT", "'[]'")
    _ensure_column(cursor, "students", "struggling_topics", "TEXT", "'[]'")
    _ensure_column(cursor, "students", "cumulative_assessment_words", "INTEGER", 0)
    _ensure_column(cursor, "students", "assessment_samples_count", "INTEGER", 0)
    _ensure_column(cursor, "students", "overall_level", "TEXT", "'Intermediate'")
    _ensure_column(cursor, "students", "current_lesson_difficulty", "INTEGER", 1)
    _ensure_column(cursor, "students", "consecutive_successes", "INTEGER", 0)
    _ensure_column(cursor, "students", "consecutive_struggles", "INTEGER", 0)
    _ensure_column(cursor, "students", "favorite_things", "TEXT", "'{}'")
    _ensure_column(cursor, "students", "durable_memories", "TEXT", "'[]'")

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
    # message_count is the evaluated learner-turn count used by session metrics;
    # it is deliberately not the total number of conversation rows.
    _ensure_column(cursor, "learning_sessions", "pending_name_change", "TEXT", "NULL")
    _ensure_column(cursor, "learning_sessions", "pending_name_change_created_at", "TIMESTAMP", "NULL")
    _ensure_column(cursor, "learning_sessions", "current_teaching_difficulty", "INTEGER", 1)
    _ensure_column(cursor, "learning_sessions", "is_confused", "INTEGER", 0)
    _ensure_column(cursor, "learning_sessions", "consecutive_successes", "INTEGER", 0)
    _ensure_column(cursor, "learning_sessions", "consecutive_struggles", "INTEGER", 0)
    _ensure_column(cursor, "learning_sessions", "current_topic", "TEXT", "''")
    _ensure_column(cursor, "learning_sessions", "current_subtopic", "TEXT", "''")
    _ensure_column(cursor, "learning_sessions", "current_entities", "TEXT", "'[]'")
    _ensure_column(cursor, "learning_sessions", "last_user_intent", "TEXT", "''")
    _ensure_column(cursor, "learning_sessions", "last_mizo_question", "TEXT", "''")
    _ensure_column(cursor, "learning_sessions", "last_user_correction", "TEXT", "''")
    _ensure_column(cursor, "learning_sessions", "conversation_goal", "TEXT", "''")

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
            pronunciation_score REAL DEFAULT NULL,
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

    _ensure_column(cursor, "assessments", "assessment_method", "TEXT", "'llm'")
    _ensure_column(cursor, "assessments", "provider", "TEXT", "NULL")
    _ensure_column(cursor, "assessments", "model", "TEXT", "NULL")
    _ensure_column(cursor, "assessments", "pacing_score", "REAL", "0.0")
    _ensure_column(cursor, "assessments", "overall_score", "REAL", "0.0")
    _ensure_column(cursor, "assessments", "words_per_minute", "REAL", "0.0")
    _ensure_column(cursor, "assessments", "filler_count", "INTEGER", "0")
    _ensure_column(cursor, "assessments", "word_count", "INTEGER", "0")
    _ensure_column(cursor, "assessments", "pause_count", "INTEGER", "0")
    _ensure_column(cursor, "assessments", "assessment_quality", "TEXT", "'good'")
    _ensure_column(cursor, "assessments", "quality_reason", "TEXT", "NULL")
    _ensure_column(cursor, "assessments", "stt_metadata", "TEXT", "NULL")

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

    # 9. Academic Subjects & Hierarchical Syllabus
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS syllabus_units (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id INTEGER NOT NULL,
            unit_number INTEGER DEFAULT 1,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (subject_id) REFERENCES subjects (id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS syllabus_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_id INTEGER NOT NULL,
            topic_name TEXT NOT NULL,
            summary TEXT DEFAULT '',
            content TEXT NOT NULL,
            simple_explanation TEXT DEFAULT '',
            examples TEXT DEFAULT '[]',
            key_terms TEXT DEFAULT '[]',
            sample_questions TEXT DEFAULT '[]',
            order_index INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (unit_id) REFERENCES syllabus_units (id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_subject_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            current_unit_id INTEGER DEFAULT NULL,
            current_topic_id INTEGER DEFAULT NULL,
            completed_topics TEXT DEFAULT '[]',
            weak_topics TEXT DEFAULT '[]',
            revision_topics TEXT DEFAULT '[]',
            questions_asked INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE,
            FOREIGN KEY (subject_id) REFERENCES subjects (id) ON DELETE CASCADE,
            UNIQUE(student_id, subject_id)
        )
    """)

    _ensure_column(cursor, "learning_sessions", "current_subject_id", "INTEGER", "NULL")
    _ensure_column(cursor, "learning_sessions", "current_unit_id", "INTEGER", "NULL")
    _ensure_column(cursor, "learning_sessions", "current_topic_id", "INTEGER", "NULL")
    _ensure_column(cursor, "learning_sessions", "current_subject", "TEXT", "''")
    _ensure_column(cursor, "learning_sessions", "current_unit", "TEXT", "''")

    # 10. Speech & Seminar Practice Sessions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS speech_practice_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            topic TEXT DEFAULT '',
            time_limit_seconds INTEGER DEFAULT 180,
            actual_duration_seconds REAL DEFAULT 0.0,
            required_points TEXT DEFAULT '[]',
            transcript TEXT DEFAULT '',
            analysis_json TEXT DEFAULT '{}',
            grammar_issue_count INTEGER DEFAULT 0,
            covered_points TEXT DEFAULT '[]',
            partial_points TEXT DEFAULT '[]',
            missing_points TEXT DEFAULT '[]',
            state TEXT DEFAULT 'SETUP',
            attempt_number INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE
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
    else:
        # Populate any missing keys from environment/settings if DB field is empty
        cursor.execute("SELECT * FROM system_settings WHERE id = 1")
        existing_row = dict(cursor.fetchone())
        db_updates = {}
        for db_col, env_val in [
            ("groq_api_key", settings.GROQ_API_KEY),
            ("openai_api_key", settings.OPENAI_API_KEY),
            ("qwen_api_key", settings.QWEN_API_KEY),
        ]:
            if not existing_row.get(db_col) and env_val and str(env_val).strip():
                db_updates[db_col] = str(env_val).strip()

        # Database migration: automatically upgrade deprecated legacy Groq default model
        target_default_model = settings.GROQ_MODEL or "openai/gpt-oss-120b"
        current_db_groq_model = (existing_row.get("groq_model") or "").strip()
        if not current_db_groq_model or current_db_groq_model == "llama-3.3-70b-versatile":
            print(f"[MIGRATION] Migrating deprecated Groq model '{current_db_groq_model}' -> '{target_default_model}'")
            db_updates["groq_model"] = target_default_model

        if db_updates:
            clauses = [f"{col} = ?" for col in db_updates]
            cursor.execute(f"UPDATE system_settings SET {', '.join(clauses)} WHERE id = 1", list(db_updates.values()))
            conn.commit()

    # Ensure in-memory runtime settings reflect DB settings
    cursor.execute("SELECT * FROM system_settings WHERE id = 1")
    row_now = cursor.fetchone()
    if row_now:
        import os
        d = dict(row_now)
        for db_col, env_attr in [
            ("groq_api_key", "GROQ_API_KEY"),
            ("openai_api_key", "OPENAI_API_KEY"),
            ("qwen_api_key", "QWEN_API_KEY"),
            ("ollama_base_url", "OLLAMA_BASE_URL"),
            ("active_provider", "DEFAULT_AI_PROVIDER"),
            ("groq_model", "GROQ_MODEL"),
            ("openai_model", "OPENAI_MODEL"),
            ("qwen_model", "QWEN_MODEL"),
            ("ollama_model", "OLLAMA_MODEL"),
            ("tts_voice", "DEFAULT_TTS_VOICE"),
            ("tts_rate", "DEFAULT_TTS_RATE"),
        ]:
            v = d.get(db_col)
            val_clean = "" if v is None else str(v).strip().strip("'\"")
            os.environ[env_attr] = val_clean
            if hasattr(settings, env_attr):
                setattr(settings, env_attr, val_clean)

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

    # Sanitize any legacy corrupted student names (e.g. 'Going To', 'Stop') in existing database
    cursor.execute("SELECT id, name FROM students")
    for s_row in cursor.fetchall():
        s_id, s_name = s_row[0], s_row[1]
        if s_name:
            clean_n = s_name.strip()
            if clean_n.lower() in ["going to", "stop", "robotics", "machine", "gravity", "undefined", "null"]:
                new_n = "Shahid" if s_id == 1 else "Student"
                cursor.execute("UPDATE students SET name = ? WHERE id = ?", (new_n, s_id))
                conn.commit()

    conn.close()


# -------------------------------------------------------------------
# Database Operations: System Settings & .env Sync
# -------------------------------------------------------------------
def mask_api_key(key: Optional[str]) -> str:
    """Safely masks an API key without revealing secrets (e.g., gsk_****abcd or sk-p****1234)."""
    if not key:
        return "Not configured"
    k = str(key).strip().strip("'\"")
    if not k:
        return "Not configured"
    if len(k) <= 8:
        return "********"
    prefix = k[:4]
    suffix = k[-4:]
    return f"{prefix}****{suffix}"


def _apply_db_settings_to_runtime(current_db_settings: Dict[str, Any]):
    """Refresh runtime settings from SQLite without mutating the .env bootstrap file."""
    import os

    runtime_map = {
        "groq_api_key": "GROQ_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
        "qwen_api_key": "QWEN_API_KEY",
        "ollama_base_url": "OLLAMA_BASE_URL",
        "active_provider": "DEFAULT_AI_PROVIDER",
        "groq_model": "GROQ_MODEL",
        "openai_model": "OPENAI_MODEL",
        "qwen_model": "QWEN_MODEL",
        "ollama_model": "OLLAMA_MODEL",
        "tts_voice": "DEFAULT_TTS_VOICE",
        "tts_rate": "DEFAULT_TTS_RATE",
    }

    for db_key, env_key in runtime_map.items():
        value = current_db_settings.get(db_key)
        value_clean = "" if value is None else str(value).strip().strip("'\"")
        os.environ[env_key] = value_clean
        if hasattr(settings, env_key):
            setattr(settings, env_key, value_clean)


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

    # Handle explicit key deletion if requested
    for prov in ["groq", "openai", "qwen"]:
        if updates.get(f"clear_{prov}_key") or updates.get("clear_key") == prov:
            set_clauses.append(f"{prov}_api_key = ''")

    for k, v in updates.items():
        if k in valid_fields and v is not None:
            # Prevent empty or whitespace-only strings from overwriting an already configured API key
            if k in ["groq_api_key", "openai_api_key", "qwen_api_key"]:
                val_str = str(v).strip().strip("'\"")
                if not val_str:
                    continue
                v = val_str
            elif isinstance(v, str):
                v = v.strip().strip("'\"")
            set_clauses.append(f"{k} = ?")
            params.append(v)

    if set_clauses:
        query = f"UPDATE system_settings SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP WHERE id = 1"
        cursor.execute(query, params)
        conn.commit()

    conn.close()

    # Dynamic reload: immediately update in-memory settings from authoritative DB
    current_settings = get_db_settings()

    # SQLite is authoritative after bootstrap; never write Admin-managed keys to .env.
    _apply_db_settings_to_runtime(current_settings)

    return current_settings


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
    item["assessed_level"] = item.get("assessed_level")
    item["baseline_pronunciation"] = item.get("baseline_pronunciation")
    item["baseline_confidence"] = item.get("baseline_confidence", 0.0) or 0.0
    item["baseline_communication"] = item.get("baseline_communication", 0.0) or 0.0
    item["pronunciation_score"] = item.get("pronunciation_score")
    item["communication_score"] = item.get("communication_score", 0.0) or 0.0

    # New personalized learner profile dimensions
    item["grammar_level"] = item.get("grammar_level") or "Beginner"
    item["vocabulary_level"] = item.get("vocabulary_level") or "Beginner"
    item["fluency_level"] = item.get("fluency_level") or "Beginner"
    item["sentence_formation_level"] = item.get("sentence_formation_level") or "Beginner"
    item["communication_confidence"] = item.get("communication_confidence") or "Low"
    item["speaking_hesitation"] = item.get("speaking_hesitation") or "Frequent"
    item["presentation_confidence"] = item.get("presentation_confidence") or "Low"
    item["preferred_explanation_difficulty"] = int(item.get("preferred_explanation_difficulty") or 1)
    item["current_teaching_difficulty"] = int(item.get("current_teaching_difficulty") or 1)
    item["current_lesson_difficulty"] = int(item.get("current_lesson_difficulty") or item.get("current_teaching_difficulty") or 1)
    item["overall_level"] = item.get("overall_level") or item.get("assessed_level") or item.get("target_level") or "Intermediate"
    item["cumulative_assessment_words"] = int(item.get("cumulative_assessment_words") or 0)
    item["assessment_samples_count"] = int(item.get("assessment_samples_count") or 0)
    item["consecutive_successes"] = int(item.get("consecutive_successes") or 0)
    item["consecutive_struggles"] = int(item.get("consecutive_struggles") or 0)

    for list_key in ["known_words", "difficult_words", "recurring_grammar_errors", "mastered_topics", "struggling_topics"]:
        raw_val = item.get(list_key)
        if isinstance(raw_val, str):
            try:
                item[list_key] = json.loads(raw_val)
            except Exception:
                item[list_key] = [x.strip() for x in raw_val.split(",") if x.strip()]
        elif not isinstance(raw_val, list):
            item[list_key] = []

    fav = item.get("favorite_things")
    if isinstance(fav, str):
        try:
            item["favorite_things"] = json.loads(fav)
        except Exception:
            item["favorite_things"] = {}
    elif not isinstance(fav, dict):
        item["favorite_things"] = {}

    dur = item.get("durable_memories")
    if isinstance(dur, str):
        try:
            item["durable_memories"] = json.loads(dur)
        except Exception:
            item["durable_memories"] = []
    elif not isinstance(dur, list):
        item["durable_memories"] = []

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


def get_prototype_student() -> Optional[Dict[str, Any]]:
    """Returns the sole learner profile used by the physical prototype."""
    return get_student(PROTOTYPE_STUDENT_ID)


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
        "speaking_assessment_completed", "profile_version", "assessed_level",
        "baseline_grammar", "baseline_vocabulary", "baseline_fluency",
        "baseline_pronunciation", "baseline_confidence", "baseline_communication",
        "grammar_score", "vocabulary_score", "fluency_score", "pronunciation_score",
        "confidence_score", "communication_score",
        "total_sessions", "strengths", "weaknesses",
        "grammar_level", "vocabulary_level", "fluency_level", "sentence_formation_level",
        "communication_confidence", "speaking_hesitation", "presentation_confidence",
        "preferred_explanation_difficulty", "current_teaching_difficulty", "current_lesson_difficulty", "overall_level",
        "known_words", "difficult_words", "recurring_grammar_errors", "mastered_topics", "struggling_topics",
        "cumulative_assessment_words", "assessment_samples_count",
        "consecutive_successes", "consecutive_struggles",
        "favorite_things", "durable_memories"
    ]
    clauses = []
    params = []
    list_fields = [
        "strengths", "weaknesses", "known_words", "difficult_words",
        "recurring_grammar_errors", "mastered_topics", "struggling_topics", "durable_memories"
    ]
    for k, v in updates.items():
        if k in valid_fields and v is not None:
            if k in list_fields:
                if isinstance(v, list):
                    v = json.dumps(v)
                elif isinstance(v, str) and not v.startswith("["):
                    v = json.dumps([x.strip() for x in v.split(",") if x.strip()])
            elif k == "favorite_things":
                if isinstance(v, dict):
                    v = json.dumps(v)
                elif isinstance(v, str) and not v.startswith("{"):
                    v = json.dumps({})
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
    pronunciation: Optional[float] = None,
    confidence: Optional[float] = None,
    communication: Optional[float] = None,
    overall_level: str = "Intermediate",
    strengths: Optional[List[str]] = None,
    weaknesses: Optional[List[str]] = None
):
    """Finalizes student onboarding and establishes permanent baseline scores."""
    conn = get_connection()
    cursor = conn.cursor()
    s_list = strengths or ["Active voice participation"]
    w_list = weaknesses or ["Grammar consistency"]
    pron_val = round(pronunciation, 1) if pronunciation is not None else None
    conf_val = round(confidence, 1) if confidence is not None else None
    comm_val = round(communication, 1) if communication is not None else None
    cursor.execute("""
        UPDATE students
        SET onboarding_completed = 1,
            onboarding_step = 'completed',
            speaking_assessment_completed = 1,
            assessed_level = ?,
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
        pron_val, conf_val, comm_val,
        round(grammar, 1), round(vocab, 1), round(fluency, 1),
        pron_val, conf_val, comm_val,
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

    raw_pron = data.get("pronunciation_score")
    pron_score = float(raw_pron) if raw_pron is not None else None

    stt_meta = data.get("stt_metadata")
    if isinstance(stt_meta, (dict, list)):
        stt_meta_str = json.dumps(stt_meta)
    elif stt_meta:
        stt_meta_str = str(stt_meta)
    else:
        stt_meta_str = None

    cursor.execute("""
        INSERT INTO assessments (
            student_id, session_id, topic, transcript, duration_seconds,
            grammar_score, vocabulary_score, fluency_score, pronunciation_score,
            confidence_score, communication_score, overall_level,
            grammar_feedback, vocabulary_feedback, fluency_feedback,
            pronunciation_feedback, confidence_feedback, communication_feedback,
            strengths, weaknesses, raw_analysis_json,
            assessment_method, provider, model, pacing_score, overall_score,
            words_per_minute, filler_count, word_count, pause_count,
            assessment_quality, quality_reason, stt_metadata
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("student_id", 1),
        data.get("session_id", ""),
        data.get("topic", "Self Introduction & Experience"),
        data.get("transcript", ""),
        data.get("duration_seconds", 0.0),
        data.get("grammar_score", 0.0),
        data.get("vocabulary_score", 0.0),
        data.get("fluency_score", 0.0),
        pron_score,
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
        raw_analysis,
        data.get("assessment_method", "llm"),
        data.get("provider"),
        data.get("model"),
        data.get("pacing_score", 0.0),
        data.get("overall_score", data.get("communication_score", 0.0)),
        data.get("words_per_minute", 0.0),
        data.get("filler_count", 0),
        data.get("word_count", 0),
        data.get("pause_count", 0),
        data.get("assessment_quality", "good"),
        data.get("quality_reason"),
        stt_meta_str
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
    if item.get("stt_metadata"):
        try:
            item["stt_metadata"] = json.loads(item["stt_metadata"])
        except Exception:
            pass
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
        if item.get("stt_metadata"):
            try:
                item["stt_metadata"] = json.loads(item["stt_metadata"])
            except Exception:
                pass
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
    pronunciation_val: Optional[float] = None,
    communication_val: Optional[float] = None
):
    """
    Updates cumulative historical student metrics using exponential moving averages.
    Historical performance stays intact and evolves over time without fake seed defaults.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    row = cursor.fetchone()
    if row:
        row_dict = dict(row)
        curr_g = row_dict.get("grammar_score")
        curr_v = row_dict.get("vocabulary_score")
        curr_f = row_dict.get("fluency_score")
        curr_p = row_dict.get("pronunciation_score")
        curr_c = row_dict.get("communication_score")

        if curr_g is None or curr_g == 0.0:
            new_g = grammar_val
            new_v = vocab_val
            new_f = fluency_val
            new_p = pronunciation_val if pronunciation_val is not None else curr_p
            new_c = communication_val if communication_val is not None else curr_c
        else:
            new_g = min(100.0, max(10.0, (curr_g * 0.8) + (grammar_val * 0.2)))
            new_v = min(100.0, max(10.0, (curr_v * 0.8) + (vocab_val * 0.2)))
            new_f = min(100.0, max(10.0, (curr_f * 0.8) + (fluency_val * 0.2)))
            new_p = min(100.0, max(10.0, ((curr_p or pronunciation_val) * 0.8) + (pronunciation_val * 0.2))) if pronunciation_val is not None else curr_p
            new_c = min(100.0, max(10.0, ((curr_c or communication_val) * 0.8) + (communication_val * 0.2))) if communication_val is not None else curr_c

        p_contrib = (new_p * 0.15) if new_p is not None else 0.0
        c_contrib = (new_c * 0.15) if new_c is not None else 0.0
        weights_sum = 0.70 + (0.15 if new_p is not None else 0.0) + (0.15 if new_c is not None else 0.0)
        new_conf = round(((new_g * 0.25) + (new_v * 0.20) + (new_f * 0.25) + p_contrib + c_contrib) / weights_sum, 1)

        cursor.execute("""
            UPDATE students
            SET grammar_score = ?, vocabulary_score = ?, fluency_score = ?,
                pronunciation_score = ?, communication_score = ?, confidence_score = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (round(new_g, 1), round(new_v, 1), round(new_f, 1), round(new_p, 1) if new_p is not None else None, round(new_c, 1) if new_c is not None else None, new_conf, student_id))
        conn.commit()
    conn.close()


def get_student_historical_progress(student_id: int) -> Dict[str, Any]:
    """
    Computes cumulative historical progress strictly and solely from completed assessments
    belonging to this learner.
    Never invents scores, never uses hardcoded seeds, and includes full provenance tracking.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, session_id, topic, transcript, duration_seconds,
               grammar_score, vocabulary_score, fluency_score, pronunciation_score,
               confidence_score, communication_score, overall_level, created_at
        FROM assessments
        WHERE student_id = ?
        ORDER BY created_at ASC
    """, (student_id,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {
            "has_data": False,
            "total_assessments": 0,
            "message": "No historical assessment data",
            "fluency": None,
            "grammar": None,
            "vocabulary": None,
            "confidence": None,
            "fluency_score": None,
            "grammar_score": None,
            "vocabulary_score": None,
            "confidence_score": None,
            "pronunciation_score": None,
            "communication_score": None,
            "overall_level": None,
            "sources": []
        }

    assessments = [dict(r) for r in rows]
    f_scores = [a["fluency_score"] for a in assessments if a.get("fluency_score") is not None]
    g_scores = [a["grammar_score"] for a in assessments if a.get("grammar_score") is not None]
    v_scores = [a["vocabulary_score"] for a in assessments if a.get("vocabulary_score") is not None]
    c_scores = [a["confidence_score"] for a in assessments if a.get("confidence_score") is not None]
    p_scores = [a["pronunciation_score"] for a in assessments if a.get("pronunciation_score") is not None]
    m_scores = [a["communication_score"] for a in assessments if a.get("communication_score") is not None]

    avg_f = round(sum(f_scores) / len(f_scores), 1) if f_scores else None
    avg_g = round(sum(g_scores) / len(g_scores), 1) if g_scores else None
    avg_v = round(sum(v_scores) / len(v_scores), 1) if v_scores else None
    avg_c = round(sum(c_scores) / len(c_scores), 1) if c_scores else None
    avg_p = round(sum(p_scores) / len(p_scores), 1) if p_scores else None
    avg_m = round(sum(m_scores) / len(m_scores), 1) if m_scores else None

    latest_level = assessments[-1].get("overall_level") or "Intermediate"

    sources = [
        {
            "assessment_id": a["id"],
            "session_id": a["session_id"],
            "timestamp": str(a["created_at"]),
            "duration_seconds": a.get("duration_seconds", 0),
            "fluency": a.get("fluency_score"),
            "grammar": a.get("grammar_score"),
            "vocabulary": a.get("vocabulary_score"),
            "confidence": a.get("confidence_score"),
            "level": a.get("overall_level")
        }
        for a in assessments
    ]

    return {
        "has_data": True,
        "total_assessments": len(assessments),
        "message": f"Calculated from {len(assessments)} completed assessment(s)",
        "fluency": avg_f,
        "grammar": avg_g,
        "vocabulary": avg_v,
        "confidence": avg_c,
        "fluency_score": avg_f,
        "grammar_score": avg_g,
        "vocabulary_score": avg_v,
        "confidence_score": avg_c,
        "pronunciation_score": avg_p,
        "communication_score": avg_m,
        "overall_level": latest_level,
        "sources": sources
    }


def reset_current_session(student_id: int, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Clears ONLY the current session's metrics and conversation messages.
    Does NOT delete historical progress, learner profile, or completed assessments.
    """
    conn = get_connection()
    cursor = conn.cursor()
    if not session_id:
        cursor.execute("SELECT session_id FROM learning_sessions WHERE student_id = ? AND status = 'active' ORDER BY id DESC LIMIT 1", (student_id,))
        row = cursor.fetchone()
        if row:
            session_id = row[0]

    if session_id:
        cursor.execute("""
            UPDATE learning_sessions
            SET session_grammar = 0.0,
                session_fluency = 0.0,
                session_vocabulary = 0.0,
                session_pacing = 0.0,
                session_confidence = 0.0,
                session_pronunciation = 0.0,
                session_communication = 0.0,
                message_count = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        """, (session_id,))
        cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        conn.commit()

    conn.close()
    return get_active_session(student_id=student_id)


# -------------------------------------------------------------------
# Database Operations: Sessions Lifecycle
# -------------------------------------------------------------------
def create_session(student_id: int, mode: str = "coach") -> Dict[str, Any]:
    """
    Creates a brand new learning session.
    Current session metrics are initialized to ZERO without deleting historical data.
    """
    # Sessions are always owned by the single persisted prototype learner.
    student_id = PROTOTYPE_STUDENT_ID
    if not get_prototype_student():
        return {}
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
    student_id = PROTOTYPE_STUDENT_ID
    if not get_prototype_student():
        return {}
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


def resolve_prototype_session(session_id: Optional[str], default_mode: str = "coach") -> Dict[str, Any]:
    """Use a supplied session only when it is active and belongs to Student #1."""
    if session_id:
        session = get_session_by_id(session_id)
        if session and session.get("student_id") == PROTOTYPE_STUDENT_ID and session.get("status") == "active":
            return session
    return get_active_session(PROTOTYPE_STUDENT_ID, default_mode=default_mode)


def set_pending_name_change(session_id: str, proposed_name: str) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """UPDATE learning_sessions
           SET pending_name_change = ?, pending_name_change_created_at = CURRENT_TIMESTAMP,
               updated_at = CURRENT_TIMESTAMP
           WHERE session_id = ? AND student_id = ?""",
        (proposed_name, session_id, PROTOTYPE_STUDENT_ID),
    )
    conn.commit()
    conn.close()


def get_pending_name_change(session_id: str) -> Optional[str]:
    """Returns a recent pending name change and clears expired proposals."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """UPDATE learning_sessions SET pending_name_change = NULL,
               pending_name_change_created_at = NULL
           WHERE session_id = ? AND student_id = ?
             AND pending_name_change_created_at < datetime('now', ?)""",
        (session_id, PROTOTYPE_STUDENT_ID, f"-{PENDING_NAME_CHANGE_TTL_MINUTES} minutes"),
    )
    cursor.execute(
        "SELECT pending_name_change FROM learning_sessions WHERE session_id = ? AND student_id = ?",
        (session_id, PROTOTYPE_STUDENT_ID),
    )
    row = cursor.fetchone()
    conn.commit()
    conn.close()
    return (row[0] or None) if row else None


def clear_pending_name_change(session_id: str) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """UPDATE learning_sessions SET pending_name_change = NULL,
               pending_name_change_created_at = NULL, updated_at = CURRENT_TIMESTAMP
           WHERE session_id = ? AND student_id = ?""",
        (session_id, PROTOTYPE_STUDENT_ID),
    )
    conn.commit()
    conn.close()


def _format_session_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    raw_entities = row.get("current_entities")
    if isinstance(raw_entities, str):
        try:
            entities = json.loads(raw_entities)
        except Exception:
            entities = [x.strip() for x in raw_entities.split(",") if x.strip()]
    elif isinstance(raw_entities, list):
        entities = raw_entities
    else:
        entities = []

    return {
        "id": row.get("id"),
        "session_id": row.get("session_id"),
        "student_id": row.get("student_id"),
        "mode": row.get("mode"),
        "status": row.get("status"),
        "current_topic": row.get("current_topic") or "",
        "current_subtopic": row.get("current_subtopic") or "",
        "current_entities": entities,
        "last_user_intent": row.get("last_user_intent") or "",
        "last_mizo_question": row.get("last_mizo_question") or "",
        "last_user_correction": row.get("last_user_correction") or "",
        "conversation_goal": row.get("conversation_goal") or "",
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


def get_session_context(session_id: str) -> Dict[str, Any]:
    sess = get_session_by_id(session_id)
    if not sess:
        return {
            "current_topic": "",
            "current_subtopic": "",
            "current_entities": [],
            "last_user_intent": "",
            "last_mizo_question": "",
            "last_user_correction": "",
            "conversation_goal": "",
        }
    return {
        "current_topic": sess.get("current_topic", ""),
        "current_subtopic": sess.get("current_subtopic", ""),
        "current_entities": sess.get("current_entities", []),
        "last_user_intent": sess.get("last_user_intent", ""),
        "last_mizo_question": sess.get("last_mizo_question", ""),
        "last_user_correction": sess.get("last_user_correction", ""),
        "conversation_goal": sess.get("conversation_goal", ""),
    }


def update_session_context(session_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    valid_fields = [
        "current_topic", "current_subtopic", "current_entities",
        "last_user_intent", "last_mizo_question", "last_user_correction", "conversation_goal",
        "current_subject_id", "current_unit_id", "current_topic_id", "current_subject", "current_unit"
    ]
    clauses = []
    params = []
    for k, v in updates.items():
        if k in valid_fields and v is not None:
            if k == "current_entities":
                if isinstance(v, list):
                    v = json.dumps(v)
                elif isinstance(v, str) and not v.startswith("["):
                    v = json.dumps([x.strip() for x in v.split(",") if x.strip()])
            clauses.append(f"{k} = ?")
            params.append(v)
    if clauses:
        params.append(session_id)
        cursor.execute(f"UPDATE learning_sessions SET {', '.join(clauses)}, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?", params)
        conn.commit()
    conn.close()
    return get_session_by_id(session_id) or {}


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


def get_student_memory_context(student_id: int, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Builds a rich context payload of student history for the LLM."""
    student = get_student(student_id)
    if not student:
        return {}

    recent_mistakes = get_recent_mistakes(student_id, limit=4)
    if session_id:
        history = get_conversation_history(student_id=student_id, session_id=session_id, limit=10)
    else:
        history = get_conversation_history(student_id=student_id, limit=10)

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
    student_id = PROTOTYPE_STUDENT_ID
    student = get_prototype_student()
    if not student:
        return {"status": "primary_student_missing", "student": None, "session": None,
                "onboarding_active": False, "onboarding_step": None}

    is_onboarding_done = student.get("onboarding_completed", False)
    current_step = (student.get("onboarding_step") or "ask_name").lower()
    st_name = (student.get("name") or "").strip()
    is_name_known = bool(st_name and st_name.lower() not in ["new learner", "student", "there", "unknown", ""])
    has_completed_assessment = bool(
        student.get("speaking_assessment_completed")
        or (student.get("assessed_level") and student.get("assessed_level") != "Pending")
        or current_step == "completed"
    )

    if is_onboarding_done or has_completed_assessment:
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
        # If name is already known, do not ask for name again
        if is_name_known and current_step in ["ask_name", "new_student", "welcome"]:
            current_step = "ask_goals"
            update_student(student_id, {"onboarding_step": "ask_goals"})

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

    if student_id:
        conditions.append("student_id = ?")
        params.append(student_id)
    if session_id:
        conditions.append("session_id = ?")
        params.append(session_id)

    if search and search.strip():
        conditions.append("content LIKE ?")
        params.append(f"%{search.strip()}%")

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"SELECT * FROM (SELECT * FROM conversations {where_clause} ORDER BY id DESC LIMIT ?) ORDER BY id ASC"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    # Return in strict canonical chronological order (oldest to newest)
    results = []
    for r in rows:
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


# -------------------------------------------------------------------
# Database Operations: Academic Subjects & Hierarchical Syllabus
# -------------------------------------------------------------------

def create_subject(name: str, description: str = "") -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO subjects (name, description)
        VALUES (?, ?)
        ON CONFLICT(name) DO UPDATE SET description = excluded.description
    """, (name.strip(), description.strip()))
    sub_id = cursor.lastrowid
    if not sub_id:
        cursor.execute("SELECT id FROM subjects WHERE name = ?", (name.strip(),))
        row = cursor.fetchone()
        sub_id = row["id"] if row else None
    conn.commit()
    conn.close()
    return sub_id


def get_subject(subject_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_subject_by_name(name: str) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM subjects WHERE LOWER(name) = LOWER(?)", (name.strip(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_all_subjects() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM subjects ORDER BY name ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_subject(subject_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def add_syllabus_unit(subject_id: int, title: str, unit_number: int = 1, description: str = "") -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO syllabus_units (subject_id, unit_number, title, description)
        VALUES (?, ?, ?, ?)
    """, (subject_id, unit_number, title.strip(), description.strip()))
    unit_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return unit_id


def get_syllabus_units(subject_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM syllabus_units WHERE subject_id = ? ORDER BY unit_number ASC, id ASC", (subject_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_syllabus_topic(
    unit_id: int,
    topic_name: str,
    content: str,
    summary: str = "",
    simple_explanation: str = "",
    examples: Optional[List[str]] = None,
    key_terms: Optional[List[Dict[str, str]]] = None,
    sample_questions: Optional[List[Dict[str, Any]]] = None,
    order_index: int = 0
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO syllabus_topics (
            unit_id, topic_name, summary, content, simple_explanation,
            examples, key_terms, sample_questions, order_index
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        unit_id,
        topic_name.strip(),
        summary.strip(),
        content.strip(),
        simple_explanation.strip(),
        json.dumps(examples or []),
        json.dumps(key_terms or []),
        json.dumps(sample_questions or []),
        order_index
    ))
    topic_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return topic_id


def _parse_topic_row(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    for field in ["examples", "key_terms", "sample_questions"]:
        if field in d and isinstance(d[field], str):
            try:
                d[field] = json.loads(d[field])
            except Exception:
                d[field] = []
    return d


def get_syllabus_topics(unit_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM syllabus_topics WHERE unit_id = ? ORDER BY order_index ASC, id ASC", (unit_id,))
    rows = cursor.fetchall()
    conn.close()
    return [_parse_topic_row(r) for r in rows]


def get_topic_by_id(topic_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM syllabus_topics WHERE id = ?", (topic_id,))
    row = cursor.fetchone()
    conn.close()
    return _parse_topic_row(row) if row else None


def get_topic_by_name(subject_id: int, topic_name: str) -> Optional[Dict[str, Any]]:
    if not topic_name:
        return None
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT t.* FROM syllabus_topics t
        JOIN syllabus_units u ON t.unit_id = u.id
        WHERE u.subject_id = ? AND LOWER(t.topic_name) = LOWER(?)
    """, (subject_id, topic_name.strip()))
    row = cursor.fetchone()
    conn.close()
    return _parse_topic_row(row) if row else None


def get_all_topics_for_subject(subject_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT t.*, u.unit_number, u.title as unit_title FROM syllabus_topics t
        JOIN syllabus_units u ON t.unit_id = u.id
        WHERE u.subject_id = ?
        ORDER BY u.unit_number ASC, t.order_index ASC, t.id ASC
    """, (subject_id,))
    rows = cursor.fetchall()
    conn.close()
    return [_parse_topic_row(r) for r in rows]


def get_entire_subject_hierarchy(subject_id: int) -> Optional[Dict[str, Any]]:
    subject = get_subject(subject_id)
    if not subject:
        return None
    units = get_syllabus_units(subject_id)
    for u in units:
        u["topics"] = get_syllabus_topics(u["id"])
    subject["units"] = units
    return subject


def get_or_create_student_subject_progress(student_id: int, subject_id: int) -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM student_subject_progress WHERE student_id = ? AND subject_id = ?
    """, (student_id, subject_id))
    row = cursor.fetchone()
    if not row:
        cursor.execute("""
            INSERT INTO student_subject_progress (
                student_id, subject_id, completed_topics, weak_topics, revision_topics, questions_asked
            ) VALUES (?, ?, '[]', '[]', '[]', 0)
        """, (student_id, subject_id))
        conn.commit()
        cursor.execute("SELECT * FROM student_subject_progress WHERE id = ?", (cursor.lastrowid,))
        row = cursor.fetchone()
    conn.close()
    d = dict(row)
    for f in ["completed_topics", "weak_topics", "revision_topics"]:
        if isinstance(d.get(f), str):
            try:
                d[f] = json.loads(d[f])
            except Exception:
                d[f] = []
    return d


def update_student_subject_progress(student_id: int, subject_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    set_clauses = []
    vals = []
    for k, v in updates.items():
        if k in ["completed_topics", "weak_topics", "revision_topics"] and not isinstance(v, str):
            v = json.dumps(v)
        set_clauses.append(f"{k} = ?")
        vals.append(v)
    set_clauses.append("updated_at = CURRENT_TIMESTAMP")
    vals.extend([student_id, subject_id])
    cursor.execute(f"""
        UPDATE student_subject_progress
        SET {', '.join(set_clauses)}
        WHERE student_id = ? AND subject_id = ?
    """, vals)
    conn.commit()
    conn.close()
    return get_or_create_student_subject_progress(student_id, subject_id)


def record_topic_completed(student_id: int, subject_id: int, topic_name: str):
    prog = get_or_create_student_subject_progress(student_id, subject_id)
    completed = set(prog.get("completed_topics") or [])
    completed.add(topic_name)
    update_student_subject_progress(student_id, subject_id, {
        "completed_topics": list(completed)
    })


def record_topic_struggle(student_id: int, subject_id: int, topic_name: str, question: Optional[str] = None):
    prog = get_or_create_student_subject_progress(student_id, subject_id)
    weak = set(prog.get("weak_topics") or [])
    revision = set(prog.get("revision_topics") or [])
    weak.add(topic_name)
    revision.add(topic_name)
    q_count = (prog.get("questions_asked") or 0) + 1
    update_student_subject_progress(student_id, subject_id, {
        "weak_topics": list(weak),
        "revision_topics": list(revision),
        "questions_asked": q_count
    })


def record_topic_understood(student_id: int, subject_id: int, topic_name: str):
    prog = get_or_create_student_subject_progress(student_id, subject_id)
    completed = set(prog.get("completed_topics") or [])
    revision = set(prog.get("revision_topics") or [])
    completed.add(topic_name)
    if topic_name in revision:
        revision.remove(topic_name)
    update_student_subject_progress(student_id, subject_id, {
        "completed_topics": list(completed),
        "revision_topics": list(revision)
    })


def load_complete_syllabus(subject_data: Dict[str, Any]) -> int:
    name = subject_data.get("name") or subject_data.get("subject_name", "Untitled Subject")
    desc = subject_data.get("description", "")
    sub_id = create_subject(name, desc)
    
    # Clean previous units and topics for this subject on reload
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM syllabus_units WHERE subject_id = ?", (sub_id,))
    conn.commit()
    conn.close()

    for u in subject_data.get("units", []):
        unit_num = u.get("unit_number", 1)
        unit_title = u.get("title", f"Unit {unit_num}")
        unit_desc = u.get("description", "")
        unit_id = add_syllabus_unit(sub_id, unit_title, unit_num, unit_desc)
        for idx, t in enumerate(u.get("topics", [])):
            if isinstance(t, str):
                t_name = t
                t_content = f"Fundamentals and concepts of {t}."
                t_summary = f"{t} core concepts."
                t_simple = f"Simple overview of {t}."
                t_examples = [f"Example of {t} in practice."]
                t_key_terms = []
                t_questions = []
            else:
                t_name = t.get("name") or t.get("topic_name") or t.get("title") or "Untitled Topic"
                t_content = t.get("content", "")
                t_summary = t.get("summary", "")
                t_simple = t.get("simple_explanation", "")
                t_examples = t.get("examples", [])
                t_key_terms = t.get("key_terms", [])
                t_questions = t.get("sample_questions", [])
            add_syllabus_topic(
                unit_id=unit_id,
                topic_name=t_name,
                content=t_content,
                summary=t_summary,
                simple_explanation=t_simple,
                examples=t_examples,
                key_terms=t_key_terms,
                sample_questions=t_questions,
                order_index=idx
            )
    return sub_id


# -------------------------------------------------------------------
# Database Operations: Speech & Seminar Practice
# -------------------------------------------------------------------

def _parse_speech_row(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    for field in ["required_points", "covered_points", "partial_points", "missing_points"]:
        if field in d and isinstance(d[field], str):
            try:
                d[field] = json.loads(d[field])
            except Exception:
                d[field] = []
    if "analysis_json" in d and isinstance(d["analysis_json"], str):
        try:
            d["analysis_json"] = json.loads(d["analysis_json"])
        except Exception:
            d["analysis_json"] = {}
    return d


def create_or_get_speech_session(student_id: int, session_id: Optional[str] = None, force_new: bool = False) -> Dict[str, Any]:
    import uuid
    if not session_id:
        session_id = f"speech_{student_id}_{uuid.uuid4().hex[:8]}"

    conn = get_connection()
    cursor = conn.cursor()
    if not force_new:
        cursor.execute("""
            SELECT * FROM speech_practice_sessions
            WHERE student_id = ? AND session_id = ?
            ORDER BY created_at DESC LIMIT 1
        """, (student_id, session_id))
        row = cursor.fetchone()
        if row:
            conn.close()
            return _parse_speech_row(row)

    cursor.execute("""
        INSERT INTO speech_practice_sessions (
            student_id, session_id, topic, time_limit_seconds, required_points, state
        ) VALUES (?, ?, '', 180, '[]', 'SETUP')
    """, (student_id, session_id))
    new_id = cursor.lastrowid
    conn.commit()
    cursor.execute("SELECT * FROM speech_practice_sessions WHERE id = ?", (new_id,))
    row = cursor.fetchone()
    conn.close()
    return _parse_speech_row(row)


def get_speech_session_by_id(speech_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM speech_practice_sessions WHERE id = ?", (speech_id,))
    row = cursor.fetchone()
    conn.close()
    return _parse_speech_row(row) if row else None


def get_active_speech_session(student_id: int, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    if session_id:
        cursor.execute("""
            SELECT * FROM speech_practice_sessions
            WHERE student_id = ? AND session_id = ?
            ORDER BY created_at DESC LIMIT 1
        """, (student_id, session_id))
    else:
        cursor.execute("""
            SELECT * FROM speech_practice_sessions
            WHERE student_id = ?
            ORDER BY created_at DESC LIMIT 1
        """, (student_id,))
    row = cursor.fetchone()
    conn.close()
    return _parse_speech_row(row) if row else None


def update_speech_session(speech_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    clauses = []
    vals = []
    json_fields = ["required_points", "covered_points", "partial_points", "missing_points", "analysis_json"]
    for k, v in updates.items():
        if k in json_fields and not isinstance(v, str):
            v = json.dumps(v)
        clauses.append(f"{k} = ?")
        vals.append(v)
    if clauses:
        vals.append(speech_id)
        cursor.execute(f"UPDATE speech_practice_sessions SET {', '.join(clauses)} WHERE id = ?", vals)
        conn.commit()
    cursor.execute("SELECT * FROM speech_practice_sessions WHERE id = ?", (speech_id,))
    row = cursor.fetchone()
    conn.close()
    return _parse_speech_row(row) if row else {}


def get_previous_speech_attempts(student_id: int, topic: str) -> List[Dict[str, Any]]:
    if not topic:
        return []
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM speech_practice_sessions
        WHERE student_id = ? AND LOWER(topic) = LOWER(?) AND state = 'FEEDBACK'
        ORDER BY created_at ASC
    """, (student_id, topic.strip()))
    rows = cursor.fetchall()
    conn.close()
    return [_parse_speech_row(r) for r in rows]


# Ensure database and tables exist on module import
init_db()


