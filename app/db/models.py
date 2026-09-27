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
        "You are Mizo, an empathetic, encouraging, and highly intelligent AI learning tutor. "
        "Your mission is to boost students' communication confidence, expand vocabulary, correct grammatical errors gently, "
        "and explain academic topics clearly. Keep spoken responses concise and conversational (2-4 sentences max) "
        "suitable for clear voice output on a robotic tutor speaker."
    )
    coaching_mode: str = "coach"
    grammar_strictness: str = "balanced"
    tts_voice: str = "en-US-GuyNeural"
    tts_rate: str = "+0%"
    tts_pitch: str = "+0Hz"


class ProviderStatusItem(BaseModel):
    provider: str
    configured: bool
    status: str  # "connected" | "error" | "not_configured"
    model: str
    details: Optional[str] = None
    latency_ms: Optional[float] = None


class SystemStatusResponse(BaseModel):
    active_provider: str
    primary_provider: str
    providers: List[ProviderStatusItem]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# -------------------------------------------------------------
# Student Profile & Progress Models
# -------------------------------------------------------------
class StudentCreate(BaseModel):
    name: str
    education: Optional[str] = "Student"
    grade: Optional[str] = "Grade 8"
    native_language: Optional[str] = "English"
    target_level: Optional[str] = "Intermediate"
    self_reported_level: Optional[str] = "Not sure"
    interests: Optional[str] = "Science, Robotics, Storytelling"
    learning_goals: Optional[str] = "Improve speaking fluency and communication"
    learning_topics: Optional[str] = "General Knowledge, STEM"


class StudentUpdate(BaseModel):
    name: Optional[str] = None
    education: Optional[str] = None
    grade: Optional[str] = None
    native_language: Optional[str] = None
    target_level: Optional[str] = None
    self_reported_level: Optional[str] = None
    interests: Optional[str] = None
    learning_goals: Optional[str] = None
    learning_topics: Optional[str] = None
    onboarding_completed: Optional[bool] = None
    onboarding_step: Optional[str] = None
    speaking_assessment_completed: Optional[bool] = None
    strengths: Optional[List[str]] = None
    weaknesses: Optional[List[str]] = None
    baseline_grammar: Optional[float] = None
    baseline_vocabulary: Optional[float] = None
    baseline_fluency: Optional[float] = None
    baseline_pronunciation: Optional[float] = None
    baseline_confidence: Optional[float] = None
    baseline_communication: Optional[float] = None


class StudentResponse(BaseModel):
    id: int
    name: str
    education: Optional[str] = "Student"
    grade: str
    native_language: str
    target_level: str
    self_reported_level: Optional[str] = "Not sure"
    interests: str
    learning_goals: str
    learning_topics: Optional[str] = ""
    onboarding_completed: bool = False
    onboarding_step: str = "ASK_NAME"
    speaking_assessment_completed: bool = False
    profile_version: int = 1
    baseline_grammar: float = 0.0
    baseline_vocabulary: float = 0.0
    baseline_fluency: float = 0.0
    baseline_pronunciation: float = 0.0
    baseline_confidence: float = 0.0
    baseline_communication: float = 0.0
    grammar_score: float = 0.0
    vocabulary_score: float = 0.0
    fluency_score: float = 0.0
    pronunciation_score: float = 0.0
    confidence_score: float = 0.0
    communication_score: float = 0.0
    total_sessions: int = 0
    strengths: List[str] = []
    weaknesses: List[str] = []
    created_at: str
    updated_at: str


# -------------------------------------------------------------
# Assessment Models
# -------------------------------------------------------------
class AssessmentResponse(BaseModel):
    id: int
    student_id: int
    session_id: str
    topic: Optional[str] = None
    transcript: str
    duration_seconds: float = 0.0
    grammar_score: float = 0.0
    vocabulary_score: float = 0.0
    fluency_score: float = 0.0
    pronunciation_score: float = 0.0
    confidence_score: float = 0.0
    communication_score: float = 0.0
    overall_level: str = "Intermediate"
    grammar_feedback: Optional[str] = None
    vocabulary_feedback: Optional[str] = None
    fluency_feedback: Optional[str] = None
    pronunciation_feedback: Optional[str] = None
    confidence_feedback: Optional[str] = None
    communication_feedback: Optional[str] = None
    strengths: List[str] = []
    weaknesses: List[str] = []
    created_at: str


# -------------------------------------------------------------
# Session Models
# -------------------------------------------------------------
class SessionCreate(BaseModel):
    student_id: int = 1
    mode: Optional[str] = "coach"


class SessionMetrics(BaseModel):
    grammar_accuracy: float = 0.0
    fluency_score: float = 0.0
    vocabulary_richness: float = 0.0
    pronunciation_score: float = 0.0
    confidence_score: float = 0.0
    communication_score: float = 0.0
    pacing_score: float = 0.0
    message_count: int = 0


class SessionResponse(BaseModel):
    id: int
    session_id: str
    student_id: int
    mode: str
    status: str  # "active" | "completed"
    metrics: SessionMetrics
    created_at: str
    updated_at: str


# -------------------------------------------------------------
# Mistakes & Learning History Models
# -------------------------------------------------------------
class StudentMistake(BaseModel):
    id: Optional[int] = None
    student_id: int
    session_id: str
    utterance: str
    mistake_type: str  # "grammar" | "vocabulary" | "pronunciation" | "structure"
    error_text: str
    correction: str
    explanation: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


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
    pacing_score: Optional[float] = None
    confidence_score: Optional[float] = None
    learning_mode: str = "coach"
    onboarding_state: Optional[str] = None
    intent: Optional[str] = None
    provider_used: Optional[str] = None
    is_fallback: bool = False
    latency_ms: Optional[float] = None
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
    session_id: Optional[str] = None
    message: str
    mode: Optional[str] = None  # coach | tutor | speech


class ESP32ChatResponse(BaseModel):
    transcript: str
    reply_text: str
    audio_url: Optional[str] = None
    learning_mode: str
    session_id: str
    provider_used: str
    is_fallback: bool = False
    onboarding_active: bool = False
    onboarding_step: Optional[str] = None
    corrections: List[str] = []
    vocab_suggestions: List[str] = []
    session_metrics: Dict[str, float] = {}
    historical_metrics: Dict[str, float] = {}
    follow_up_hint: Optional[str] = None
    topic: Optional[str] = None
    assessment: Optional[Dict[str, Any]] = None
    is_control_command: Optional[bool] = False
    command: Optional[str] = None
    voice_state: Optional[str] = None


class ESP32StartRequest(BaseModel):
    student_id: Optional[int] = 1
    action: Optional[str] = "start"  # "start" | "reset"


class ESP32StartResponse(BaseModel):
    success: bool = True
    action: str = "start"
    is_returning_user: bool = False
    student_id: int
    student_name: str
    reply_text: str
    audio_url: Optional[str] = None
    onboarding_active: bool = False
    onboarding_step: Optional[str] = None
    session_id: str
    topic: Optional[str] = None


