import os
import io
import wave
import struct
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.services.llm_service import LLMResult

client = TestClient(app)


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Mizo 3.0" in data["app"]
    assert "primary_llm" in data


def test_admin_settings_and_provider_status():
    # Get settings
    get_res = client.get("/api/v1/admin/settings")
    assert get_res.status_code == 200
    settings_data = get_res.json()
    assert "active_provider" in settings_data

    # Update settings
    post_res = client.post("/api/v1/admin/settings", json={
        "coaching_mode": "tutor",
        "grammar_strictness": "strict"
    })
    assert post_res.status_code == 200
    assert post_res.json()["success"] is True

    # Check provider status endpoint
    status_res = client.get("/api/v1/admin/settings/provider-status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert "providers" in status_data
    assert len(status_data["providers"]) == 4


def test_students_and_session_lifecycle():
    # List students
    res = client.get("/api/v1/students")
    assert res.status_code == 200
    students = res.json()
    assert len(students) > 0
    student_id = students[0]["id"]

    # Get single student
    single_res = client.get(f"/api/v1/students/{student_id}")
    assert single_res.status_code == 200
    assert single_res.json()["name"] == students[0]["name"]

    # Start a NEW session for the student
    new_sess_res = client.post(f"/api/v1/students/{student_id}/sessions", json={
        "student_id": student_id,
        "mode": "coach"
    })
    assert new_sess_res.status_code == 200
    new_sess = new_sess_res.json()
    assert new_sess["status"] == "active"
    # Verify current session metrics start at ZERO
    assert new_sess["metrics"]["grammar_accuracy"] == 0.0
    assert new_sess["metrics"]["fluency_score"] == 0.0
    assert new_sess["metrics"]["message_count"] == 0

    # Get active session
    active_res = client.get(f"/api/v1/students/{student_id}/sessions/active")
    assert active_res.status_code == 200
    assert active_res.json()["session_id"] == new_sess["session_id"]

    # Get mistakes history
    mistakes_res = client.get(f"/api/v1/students/{student_id}/mistakes")
    assert mistakes_res.status_code == 200
    assert isinstance(mistakes_res.json(), list)


def test_knowledge_base():
    # Upload text knowledge
    sample_text = b"Photosynthesis is the process by which green plants convert sunlight into chemical energy using chlorophyll."
    files = {"file": ("biology_notes.txt", io.BytesIO(sample_text), "text/plain")}
    upload_res = client.post("/api/v1/knowledge/upload", files=files)
    assert upload_res.status_code == 200
    doc_id = upload_res.json()["doc_id"]

    # Search knowledge
    search_res = client.get("/api/v1/knowledge/search?q=photosynthesis+chlorophyll")
    assert search_res.status_code == 200
    results = search_res.json()["results"]
    assert len(results) > 0


@pytest.mark.asyncio
async def test_esp32_chat_with_real_llm_flow():
    from app.db.database import update_student
    update_student(1, {"onboarding_completed": True, "onboarding_step": "completed"})

    # Mock LLM generation to return structured LLMResult
    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLMResult(
            text="Hello Alex! Gravity is the invisible force pulling objects together.",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False,
            latency_ms=120.5
        )

        res = client.post("/api/v1/esp32/chat", json={
            "message": "He don't know how gravity works.",
            "student_id": 1,
            "mode": "coach"
        })
        assert res.status_code == 200
        data = res.json()
        assert "Gravity is the invisible force" in data["reply_text"]
        assert "groq" in data["provider_used"]
        assert data["is_fallback"] is False
        assert len(data["corrections"]) > 0  # Detected "he don't" -> "he doesn't"
        assert "session_metrics" in data
        assert "historical_metrics" in data


@pytest.mark.asyncio
async def test_esp32_audio_pipeline():
    # Create 16kHz 16-bit mono WAV in memory
    sample_rate = 16000
    duration = 0.5
    num_samples = int(sample_rate * duration)
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(num_samples):
            frames.extend(struct.pack('<h', 0))
        wf.writeframes(frames)
    wav_io.seek(0)

    # Mock STT and LLM to test entire audio routing
    with patch("app.services.stt_service.stt_service.transcribe_audio_file", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:
        
        mock_stt.return_value = "Can you teach me about black holes?"
        mock_llm.return_value = LLMResult(
            text="A black hole is a cosmic region where gravity is so strong that nothing can escape.",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False,
            latency_ms=180.0
        )

        files = {"audio": ("mic_recording.wav", wav_io, "audio/wav")}
        data = {"student_id": "1", "mode": "tutor"}
        res = client.post("/api/v1/esp32/audio?format=json", files=files, data=data)

        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        assert body["transcript"] == "Can you teach me about black holes?"
        assert "A black hole is a cosmic region" in body["reply_text"]
        assert "audio_url" in body
        assert "session_metrics" in body

