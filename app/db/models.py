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
    clear_groq_key: Optional[bool] = None
    clear_openai_key: Optional[bool] = None
    clear_qwen_key: Optional[bool] = None
    clear_key: Optional[str] = None


class SystemSettings(BaseModel):
    active_provider: str = "groq"
    groq_api_key: str = ""
    openai_api_key: str = ""
    qwen_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    groq_model: str = "openai/gpt-oss-120b"
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
    healthy: bool = False
    status: str  # "READY" | "AUTHENTICATED" | "INVALID_API_KEY" | "QUOTA_EXCEEDED" | "RATE_LIMITED" | "MODEL_UNAVAILABLE" | "PERMISSION_DENIED" | "NETWORK_ERROR" | "NOT_CONFIGURED"
    auth_status: Optional[str] = None  # "AUTHENTICATED" | "INVALID_API_KEY" | "PERMISSION_DENIED" | "NOT_CONFIGURED" | "NETWORK_ERROR"
    generation_ready: Optional[bool] = False
    reason: Optional[str] = None
    display_status: Optional[str] = None
    model: str
    details: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    available_models: Optional[List[str]] = None


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
    assessed_level: Optional[str] = None
    baseline_grammar: Optional[float] = None
    baseline_vocabulary: Optional[float] = None
    baseline_fluency: Optional[float] = None
    baseline_pronunciation: Optional[float] = None
    baseline_confidence: Optional[float] = None
    baseline_communication: Optional[float] = None
    grammar_score: Optional[float] = None
    vocabulary_score: Optional[float] = None
    fluency_score: Optional[float] = None
    pronunciation_score: Optional[float] = None
    confidence_score: Optional[float] = None
    communication_score: Optional[float] = None
    total_sessions: int = 0
    strengths: List[str] = []
    weaknesses: List[str] = []
    created_at: str
    updated_at: str


# -------------------------------------------------------------
# Assessment Models & Historical Progress Provenance
# -------------------------------------------------------------
class AssessmentSourceItem(BaseModel):
    assessment_id: int
    session_id: str
    timestamp: str
    duration_seconds: float = 0.0
    fluency: Optional[float] = None
    grammar: Optional[float] = None
    vocabulary: Optional[float] = None
    confidence: Optional[float] = None
    level: Optional[str] = None


class HistoricalProgressResponse(BaseModel):
    has_data: bool
    total_assessments: int
    message: str = "Historical progress"
    fluency: Optional[float] = None
    grammar: Optional[float] = None
    vocabulary: Optional[float] = None
    confidence: Optional[float] = None
    fluency_score: Optional[float] = None
    grammar_score: Optional[float] = None
    vocabulary_score: Optional[float] = None
    confidence_score: Optional[float] = None
    pronunciation_score: Optional[float] = None
    communication_score: Optional[float] = None
    overall_level: Optional[str] = None
    sources: List[AssessmentSourceItem] = []


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
    pronunciation_score: Optional[float] = None
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
    assessment_method: Optional[str] = "llm"
    provider: Optional[str] = None
    model: Optional[str] = None
    pacing_score: Optional[float] = 0.0
    overall_score: Optional[float] = 0.0
    words_per_minute: Optional[float] = 0.0
    filler_count: Optional[int] = 0
    word_count: Optional[int] = 0
    pause_count: Optional[int] = 0
    stt_metadata: Optional[Any] = None
    assessment_quality: Optional[str] = "good"
    quality_reason: Optional[str] = None
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
    pronunciation_score: Optional[float] = None
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
# Academic Subjects & Syllabus Models
# -------------------------------------------------------------
class SubjectCreate(BaseModel):
    name: str
    description: Optional[str] = ""


class SubjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = ""
    created_at: str


class SyllabusTopicCreate(BaseModel):
    topic_name: str
    content: str
    summary: Optional[str] = ""
    simple_explanation: Optional[str] = ""
    examples: Optional[List[str]] = []
    key_terms: Optional[List[Dict[str, str]]] = []
    sample_questions: Optional[List[Dict[str, Any]]] = []
    order_index: Optional[int] = 0


class SyllabusTopicResponse(BaseModel):
    id: int
    unit_id: int
    topic_name: str
    summary: Optional[str] = ""
    content: str
    simple_explanation: Optional[str] = ""
    examples: List[str] = []
    key_terms: List[Dict[str, str]] = []
    sample_questions: List[Dict[str, Any]] = []
    order_index: int = 0


class SyllabusUnitCreate(BaseModel):
    title: str
    unit_number: int = 1
    description: Optional[str] = ""
    topics: Optional[List[SyllabusTopicCreate]] = []


class SyllabusUnitResponse(BaseModel):
    id: int
    subject_id: int
    unit_number: int
    title: str
    description: Optional[str] = ""
    topics: List[SyllabusTopicResponse] = []


class SyllabusHierarchyResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = ""
    units: List[SyllabusUnitResponse] = []


class StudentSubjectProgressResponse(BaseModel):
    id: int
    student_id: int
    subject_id: int
    current_unit_id: Optional[int] = None
    current_topic_id: Optional[int] = None
    completed_topics: List[str] = []
    weak_topics: List[str] = []
    revision_topics: List[str] = []
    questions_asked: int = 0
    updated_at: str


# -------------------------------------------------------------
# Speech & Seminar Practice Models
# -------------------------------------------------------------
class SpeechSessionResponse(BaseModel):
    id: int
    student_id: int
    session_id: str
    topic: str
    time_limit_seconds: int
    actual_duration_seconds: float
    required_points: List[str] = []
    transcript: str
    grammar_issue_count: int = 0
    covered_points: List[str] = []
    partial_points: List[str] = []
    missing_points: List[str] = []
    state: str
    attempt_number: int = 1
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
    duration_seconds: Optional[float] = None
    audio_metadata: Optional[Dict[str, Any]] = None
    stt_metadata: Optional[Dict[str, Any]] = None
    is_typed_text: Optional[bool] = False


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
    route: Optional[str] = None
    active_section: Optional[str] = None


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


