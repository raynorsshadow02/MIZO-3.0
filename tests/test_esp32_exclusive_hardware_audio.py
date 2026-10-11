import io
import wave
import struct
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.api.esp32 import _esp32_hardware_state, record_esp32_activity
from app.services.llm_service import LLMResult

client = TestClient(app)


def test_hardware_status_initial_disconnected():
    """When no hardware has communicated recently, hardware_status reports disconnected."""
    _esp32_hardware_state["last_seen_timestamp"] = None
    _esp32_hardware_state["last_ip"] = None

    res = client.get("/api/v1/esp32/hardware_status")
    assert res.status_code == 200
    data = res.json()
    assert data["connected"] is False
    assert data["exclusive_mode"] is True
    assert data["laptop_fallback_enabled"] is False
    assert "INMP441" in data["input_device"]
    assert "MAX98357A" in data["output_device"]
    assert "disconnected" in data["status_message"].lower()


def test_hardware_status_after_ping():
    """When ESP32 sends ping, hardware_status updates to connected."""
    ping_res = client.get("/api/v1/esp32/ping")
    assert ping_res.status_code == 200
    ping_data = ping_res.json()
    assert ping_data["status"] == "online"
    assert ping_data["laptop_fallback"] is False

    res = client.get("/api/v1/esp32/hardware_status")
    assert res.status_code == 200
    data = res.json()
    assert data["connected"] is True
    assert data["last_seen_seconds_ago"] is not None
    assert data["last_seen_seconds_ago"] < 5.0
    assert "connected" in data["status_message"].lower()


def test_speaker_test_tone_endpoint():
    """Speaker test endpoint returns a valid 16kHz 16-bit mono WAV tone for MAX98357A."""
    res = client.get("/api/v1/esp32/speaker_test")
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"

    # Verify WAV header
    wav_bytes = io.BytesIO(res.content)
    with wave.open(wav_bytes, 'rb') as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # 16-bit
        assert wf.getframerate() == 16000
        n_frames = wf.getnframes()
        assert n_frames == 16000  # 1.0 second duration


@pytest.mark.asyncio
async def test_esp32_audio_format_audio_for_hardware_playback():
    """When ESP32 requests format=audio, server returns direct audio/wav stream for MAX98357A."""
    sample_rate = 16000
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for _ in range(8000):
            frames.extend(struct.pack('<h', 0))
        wf.writeframes(frames)
    wav_io.seek(0)

    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:

        mock_stt.return_value = {
            "text": "What is kinematics?",
            "duration_seconds": 0.5,
            "language": "en"
        }
        mock_llm.return_value = LLMResult(
            text="Kinematics describes the motion of points, bodies, and systems without considering the forces.",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False,
            latency_ms=100.0
        )

        files = {"audio": ("inmp441.wav", wav_io, "audio/wav")}
        data = {"student_id": "1", "mode": "coach"}
        res = client.post("/api/v1/esp32/audio?format=audio", files=files, data=data)

        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/wav"
        assert len(res.content) > 44  # Valid WAV response with header and PCM samples


def test_esp32_audio_status_metadata():
    """GET /api/v1/esp32/audio returns metadata confirming exclusive hardware setup."""
    res = client.get("/api/v1/esp32/audio")
    assert res.status_code == 200
    data = res.json()
    assert data["exclusive_hardware"] is True
    assert data["laptop_fallback_enabled"] is False
    assert "INMP441" in data["input_device"]
    assert "MAX98357A" in data["output_device"]
