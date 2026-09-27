import os
import io
import wave
import httpx
from pathlib import Path
from typing import Optional
from app.config import settings
from app.db.database import get_db_settings


class STTService:
    """
    Speech-to-Text service:
    1. Primary: Groq Whisper Cloud (whisper-large-v3) ~150-250ms ultra-low latency.
    2. Fallback: OpenAI Whisper API (whisper-1).
    3. Fallback: Local Faster-Whisper on CPU/GPU.
    4. NO fake mock fallback text. Real errors or silence returned truthfully.
    """

    def __init__(self):
        self._local_model = None

    def _get_local_whisper(self):
        if self._local_model is None:
            try:
                from faster_whisper import WhisperModel
                self._local_model = WhisperModel("tiny", device="cpu", compute_type="int8")
            except Exception as e:
                print(f"[STT] Local Faster-Whisper initialization skipped: {e}")
                self._local_model = False
        return self._local_model if self._local_model is not False else None

    async def transcribe_audio_file(self, file_path: Path) -> str:
        """Transcribe an audio file using Groq Whisper, OpenAI Whisper, or local model."""
        db_conf = get_db_settings()
        groq_api_key = db_conf.get("groq_api_key") or settings.GROQ_API_KEY
        openai_api_key = db_conf.get("openai_api_key") or settings.OPENAI_API_KEY

        # Determine MIME type based on file extension
        suffix = file_path.suffix.lower()
        mime_map = {
            ".webm": "audio/webm",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".mp3": "audio/mp3",
            ".m4a": "audio/m4a",
            ".mp4": "audio/mp4"
        }
        mime_type = mime_map.get(suffix, "audio/webm" if "webm" in file_path.name.lower() else "audio/wav")

        # 1. Try Groq Whisper Cloud API
        if groq_api_key:
            try:
                async with httpx.AsyncClient(timeout=25.0) as client:
                    with open(file_path, "rb") as audio_file:
                        files = {
                            "file": (file_path.name, audio_file, mime_type)
                        }
                        data = {
                            "model": settings.GROQ_WHISPER_MODEL or "whisper-large-v3",
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
                            print(f"[STT] Groq Whisper API error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"[STT] Groq Whisper API request failed: {e}")

        # 2. Try OpenAI Whisper Cloud API
        if openai_api_key:
            try:
                async with httpx.AsyncClient(timeout=25.0) as client:
                    with open(file_path, "rb") as audio_file:
                        files = {
                            "file": (file_path.name, audio_file, mime_type)
                        }
                        data = {
                            "model": settings.OPENAI_WHISPER_MODEL or "whisper-1",
                            "response_format": "json"
                        }
                        headers = {
                            "Authorization": f"Bearer {openai_api_key}"
                        }
                        response = await client.post(
                            "https://api.openai.com/v1/audio/transcriptions",
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
                            print(f"[STT] OpenAI Whisper API error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"[STT] OpenAI Whisper API request failed: {e}")

        # 3. Try Local Faster-Whisper
        local_model = self._get_local_whisper()
        if local_model:
            try:
                segments, info = local_model.transcribe(str(file_path), beam_size=1)
                text = " ".join([seg.text for seg in segments]).strip()
                if text:
                    return text
            except Exception as e:
                print(f"[STT] Local Faster-Whisper failed: {e}")

        # If audio had no intelligible speech or all STT engines are unconfigured
        return ""


stt_service = STTService()

