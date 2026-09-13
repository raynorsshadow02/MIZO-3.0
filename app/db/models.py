from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# -------------------------------------------------------------
# System & Settings Models
# -------------------------------------------------------------
class SystemSettingsUpdate(BaseModel):
    active_provider: Optional[str] = None  # "groq" | "openai" | "qwen" | "ollama"
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    qwen_api_key: Optional[str] = None
    ollama_base_url: Optional[str] = None
    groq_model: Optional[str] = None
    openai_model: Optional[str] = None
    qwen_model: Optional[str] = None
    ollama_model: Optional[str] = None
    system_prompt: Optional[str] = None
    coaching_mode: Optional[str] = None  # "coach" | "tutor" | "speech"
    grammar_strictness: Optional[str] = None  # "gentle" | "balanced" | "strict"
    tts_voice: Optional[str] = None
    tts_rate: Optional[str] = None
    tts_pitch: Optional[str] = None


class SystemSettings(BaseModel):
    active_provider: str = "groq"
    groq_api_key: str = ""
    openai_api_key: str = ""
    qwen_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    groq_model: str = "llama-3.3-70b-versatile"
    openai_model: str = "gpt-4o-mini"
    qwen_model: str = "qwen/qwen-2.5-72b-instruct"
    ollama_model: str = "llama3:latest"
    system_prompt: str = (
        "You are Mikaza (Mizo), an empathetic, encouraging, and highly intelligent AI learning tutor. "
        "Your mission is to boost students' communication confidence, expand vocabulary, correct grammatical errors gently, "
        "and explain academic topics clearly. Keep spoken responses concise and conversational (2-4 sentences max) "
        "suitable for clear voice output on a robotic tutor speaker."
    )
    coaching_mode: str = "coach"
    grammar_strictness: str = "balanced"
    tts_voice: str = "en-US-GuyNeural"
    tts_rate: str = "+0%"
    tts_pitch: str = "+0Hz"


# -------------------------------------------------------------
# Student Profile & Progress Models
# -------------------------------------------------------------
class StudentCreate(BaseModel):
    name: str
    grade: Optional[str] = "Grade 8"
    native_language: Optional[str] = "English"
    target_level: Optional[str] = "Intermediate"
    interests: Optional[str] = "Science, Robotics, Storytelling"
    learning_goals: Optional[str] = "Improve speaking fluency and master science concepts"


class StudentUpdate(BaseModel):
    name: Optional[str] = None
    grade: Optional[str] = None
    native_language: Optional[str] = None
    target_level: Optional[str] = None
    interests: Optional[str] = None
    learning_goals: Optional[str] = None


class StudentResponse(BaseModel):
    id: int
    name: str
    grade: str
    native_language: str
    target_level: str
    interests: str
    learning_goals: str
    grammar_score: float = 75.0
    vocabulary_score: float = 70.0
    fluency_score: float = 80.0
    confidence_score: float = 78.0
    total_sessions: int = 0
    created_at: str
    updated_at: str


# -------------------------------------------------------------
# Conversation Models
# -------------------------------------------------------------
class ConversationMessage(BaseModel):
    id: Optional[int] = None
    student_id: int
    session_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    audio_path: Optional[str] = None
    grammar_errors: Optional[List[Dict[str, Any]]] = None
    vocabulary_suggestions: Optional[List[str]] = None
    fluency_score: Optional[float] = None
    learning_mode: str = "coach"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# -------------------------------------------------------------
# Knowledge Base Models
# -------------------------------------------------------------
class KnowledgeDocumentResponse(BaseModel):
    id: int
    filename: str
    title: str
    chunks_count: int
    created_at: str


# -------------------------------------------------------------
# Device Authorization
# -------------------------------------------------------------
class DeviceCreate(BaseModel):
    device_name: str


class DeviceResponse(BaseModel):
    id: int
    device_name: str
    api_key: str
    is_active: bool
    last_seen_at: Optional[str] = None
    created_at: str


# -------------------------------------------------------------
# ESP32 Interaction Request / Response
# -------------------------------------------------------------
class ESP32ChatRequest(BaseModel):
    student_id: Optional[int] = 1
    session_id: Optional[str] = "esp32_session"
    message: str
    mode: Optional[str] = None  # coach | tutor | speech


class ESP32ChatResponse(BaseModel):
    transcript: str
    reply_text: str
    audio_url: Optional[str] = None
    learning_mode: str
    corrections: List[str] = []
    student_stats: Dict[str, float] = {}
