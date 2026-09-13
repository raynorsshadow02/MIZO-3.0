import os
import io
import wave
import struct
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Mizo 3.0" in data["app"]


def test_admin_settings():
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


def test_students_api():
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


def test_knowledge_base():
    # Upload text knowledge
    sample_text = b"Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water."
    files = {"file": ("biology_notes.txt", io.BytesIO(sample_text), "text/plain")}
    upload_res = client.post("/api/v1/knowledge/upload", files=files)
    assert upload_res.status_code == 200
    doc_id = upload_res.json()["doc_id"]

    # Search knowledge
    search_res = client.get("/api/v1/knowledge/search?q=photosynthesis+sunlight")
    assert search_res.status_code == 200
    results = search_res.json()["results"]
    assert len(results) > 0


def test_esp32_chat():
    # Test text chat endpoint
    res = client.post("/api/v1/esp32/chat", json={
        "message": "Hello Mikaza! Can you help me practice speaking?",
        "student_id": 1,
        "mode": "coach"
    })
    assert res.status_code == 200
    data = res.json()
    assert "reply_text" in data
    assert "audio_url" in data
    assert data["learning_mode"] == "coach"


def test_esp32_audio_pipeline():
    # Synthesize a simple 16kHz 16-bit mono WAV in memory
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
            frames.extend(struct.pack('<h', 0)) # Silence
        wf.writeframes(frames)
    wav_io.seek(0)

    # Post to audio endpoint
    files = {"audio": ("mic_recording.wav", wav_io, "audio/wav")}
    data = {"student_id": "1", "session_id": "test_session", "mode": "coach"}
    res = client.post("/api/v1/esp32/audio?format=json", files=files, data=data)

    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert "reply_text" in body
    assert "audio_url" in body
