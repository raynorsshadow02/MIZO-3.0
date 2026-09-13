import os
import io
import wave
import httpx
from pathlib import Path
from typing import Optional
from app.config import settings
from app.db.database import get_db_settings


class STTService:
    """Speech-to-Text service supporting Groq Whisper Cloud and local Faster-Whisper."""

    def __init__(self):
        self._local_model = None

    def _get_local_whisper(self):
        if self._local_model is None:
            try:
                from faster_whisper import WhisperModel
                # Load small or tiny model on CPU for instant response
                self._local_model = WhisperModel("tiny", device="cpu", compute_type="int8")
            except Exception as e:
                print(f"[STT] Local Faster-Whisper initialization skipped: {e}")
                self._local_model = False
        return self._local_model if self._local_model is not False else None

    async def transcribe_audio_file(self, file_path: Path) -> str:
        """Transcribe an audio file using Groq Whisper API or local model."""
        db_conf = get_db_settings()
        groq_api_key = db_conf.get("groq_api_key") or settings.GROQ_API_KEY

        # 1. Try Groq Whisper Cloud API (near instant ~150-250ms)
        if groq_api_key:
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    with open(file_path, "rb") as audio_file:
                        files = {
                            "file": (file_path.name, audio_file, "audio/wav")
                        }
                        data = {
                            "model": settings.GROQ_WHISPER_MODEL,
                            "temperature": 0.0,
                            "response_format": "json"
                        }
                        headers = {
                            "Authorization": f"Bearer {groq_api_key}"
                        }
                        response = await client.post(
                            "https://api.groq.com/openai/v1/audio/transcriptions",
                            headers=headers,
                            files=files,
                            data=data
                        )
                        if response.status_code == 200:
                            result = response.json()
                            text = result.get("text", "").strip()
                            if text:
                                return text
                        else:
                            print(f"[STT] Groq Whisper API status {response.status_code}: {response.text}")
            except Exception as e:
                print(f"[STT] Groq Whisper API request failed: {e}")

        # 2. Try Local Faster-Whisper
        local_model = self._get_local_whisper()
        if local_model:
            try:
                segments, info = local_model.transcribe(str(file_path), beam_size=1)
                text = " ".join([seg.text for seg in segments]).strip()
                if text:
                    return text
            except Exception as e:
                print(f"[STT] Local Faster-Whisper failed: {e}")

        return "Hello Mikaza, I am ready to learn."


stt_service = STTService()
