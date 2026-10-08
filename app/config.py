import os
from pathlib import Path
from typing import Optional
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="allow")

    APP_NAME: str = "Mizo 3.0 Cloud Server"
    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Storage Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    AUDIO_CACHE_DIR: Path = BASE_DIR / "data" / "audio_cache"
    _raw_db_path: str = os.getenv("DATABASE_PATH", "data/mizo.db").strip("\"'")
    DATABASE_PATH: Path = (
        Path(_raw_db_path)
        if Path(_raw_db_path).is_absolute()
        else BASE_DIR / _raw_db_path
    )

    # Primary & Fallback AI Settings
    DEFAULT_AI_PROVIDER: str = "groq"
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
    QWEN_API_KEY: Optional[str] = os.getenv("QWEN_API_KEY", "")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # LLM Models
    # Configuration Precedence: Database explicit model > Environment variable (.env) > Application default
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    QWEN_MODEL: str = os.getenv("QWEN_MODEL", "qwen/qwen-2.5-72b-instruct")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3:latest")

    # STT Models
    GROQ_WHISPER_MODEL: str = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")
    OPENAI_WHISPER_MODEL: str = os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")

    # Centralized TTS Settings
    DEFAULT_TTS_VOICE: str = os.getenv("DEFAULT_TTS_VOICE", "en-US-GuyNeural")
    DEFAULT_TTS_RATE: str = os.getenv("DEFAULT_TTS_RATE", "+0%")
    DEFAULT_TTS_PITCH: str = os.getenv("DEFAULT_TTS_PITCH", "+0Hz")
    TTS_LANGUAGE: str = os.getenv("TTS_LANGUAGE", "en-US")
    TTS_VOICE: str = os.getenv("DEFAULT_TTS_VOICE", "en-US-GuyNeural")
    TTS_RATE: str = os.getenv("DEFAULT_TTS_RATE", "+0%")
    TTS_PITCH: str = os.getenv("DEFAULT_TTS_PITCH", "+0Hz")

    # Voice State Machine & Audio Timing Settings
    SILENCE_TIMEOUT_SECONDS: float = 3.0
    MAX_ASSESSMENT_DURATION_SECONDS: float = 120.0
    MIN_ASSESSMENT_WORDS: int = 12

    # Hardware & Device Settings
    DEFAULT_DEVICE_KEY: str = os.getenv("DEFAULT_DEVICE_KEY", "mizo_esp32_secret_key_123")
    AUDIO_SAMPLE_RATE: int = 16000  # 16kHz audio for ESP32 INMP441 & MAX98357A
    AUDIO_CHANNELS: int = 1         # Mono


settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

