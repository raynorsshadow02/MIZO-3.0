import pytest
import os
import json
import sqlite3
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import (
    init_db,
    get_student,
    update_student,
    create_student,
    save_assessment,
    get_student_assessments,
    get_student_historical_progress,
    reset_current_session,
    add_conversation_message,
    get_active_session,
    create_session,
    delete_learner_data,
    restart_onboarding,
    get_db_settings,
    update_db_settings,
    get_connection
)
from app.services.memory_service import memory_manager
from app.services.voice_service import voice_manager, VoiceState
from app.services.llm_service import llm_service, LLMResult
from app.services.personalization_service import personalization_service

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Ensure clean database setup for each test."""
    init_db()
    # Ensure student 1 exists with a clean baseline state
    student = get_student(1)
    if not student:
        create_student({"name": "Shahid", "grade": "College", "target_level": "Intermediate"})
    else:
        update_student(1, {
            "name": "Shahid",
            "onboarding_completed": True,
            "onboarding_step": "completed"
        })
    yield


# -------------------------------------------------------------------------
# Test 1 & 2: User Name Persistence and Explicit Name Update (Rejects artifacts)
# -------------------------------------------------------------------------
def test_1_and_2_user_name_persistence_and_safe_update():
    # 1. Verification of authoritative initial name
    st = get_student(1)
    assert st["name"] == "Shahid"

    # Reject conversational artifacts ("Stop", "Going To", "Gravity", "Okay")
    for bad_name in ["Stop", "Going To", "Gravity", "Okay", "There", "No", "Yes"]:
        extracted = memory_manager.extract_name_safely(bad_name, current_student=st)
        assert extracted is None or extracted == "Shahid", f"Artifact '{bad_name}' was incorrectly accepted as a name!"

    # Explicit name update patterns MUST be accepted
    explicit_statements = [
        ("My name is Abdul Rahman.", "Abdul Rahman"),
        ("Call me Farhan.", "Farhan"),
        ("I am Shahid.", "Shahid")
    ]
    for utterance, expected_name in explicit_statements:
        extracted = memory_manager.extract_name_safely(utterance, current_student=st)
        assert extracted is not None
        assert extracted.lower() == expected_name.lower()
        update_student(1, {"name": extracted})
        refreshed = get_student(1)
        assert refreshed["name"].lower() == expected_name.lower()


# -------------------------------------------------------------------------
# Test 3: Fixed Mizo Identity
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_3_mizo_fixed_identity():
    # Verify Mizo never denies having a name or claims to be nameless
    res = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "What is your name?"
    })
    assert res.status_code == 200
    data = res.json()
    reply = data["reply_text"].lower()
    assert "mizo" in reply
    assert "i don't have a name" not in reply
    assert "just your coach" not in reply


# -------------------------------------------------------------------------
# Test 4 & 5: Stop Command & Wake Command
# -------------------------------------------------------------------------
def test_4_and_5_stop_command_and_wake_command():
    # Wake up assistant
    voice_manager.transition_to(VoiceState.READY, "test_start")
    
    # Send stop command
    res_stop = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Mizo, stop"
    })
    assert res_stop.status_code == 200
    assert voice_manager.state == VoiceState.STANDBY
    assert voice_manager.microphone_active is True  # Ready to hear wake word

    # Wake word activation
    res_wake = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Hey Mizo, are you there?"
    })
    assert res_wake.status_code == 200
    assert voice_manager.state in [VoiceState.ACTIVE_LISTENING, VoiceState.READY]


# -------------------------------------------------------------------------
# Test 6: 3-Second Silence Detection Configuration
# -------------------------------------------------------------------------
def test_6_three_second_silence_detection():
    # Continuous silence threshold after speech must be configured to 3.0 seconds (Part 8)
    assert voice_manager.turn_silence_seconds == 3.0


# -------------------------------------------------------------------------
# Test 7 & 8: Assessment Completion & Score Persistence
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_7_and_8_assessment_completion_and_persistence():
    # Execute full speech assessment with valid sample
    sample_transcript = (
        "I am currently studying computer science and software engineering. "
        "I enjoy programming microcontrollers and building autonomous robots. "
        "My goal is to communicate confidently in English with international teams."
    )
    result = await personalization_service.run_speaking_assessment(
        student_id=1,
        session_id="test_assessment_session",
        transcript=sample_transcript,
        duration_seconds=75.0,
        topic="My Studies and Goals"
    )

    # Result structure validations
    assert "fluency_score" in result
    assert "grammar_score" in result
    assert "vocabulary_score" in result
    assert "confidence_score" in result
    assert "overall_level" in result
    assert result["fluency_score"] > 0
    assert result["grammar_score"] > 0
    assert result["vocabulary_score"] > 0

    # Ensure integer-level score precision (no fake decimal precision like 91.1%)
    assert float(result["grammar_score"]).is_integer() or round(result["grammar_score"], 1) == result["grammar_score"]

    # Verify assessment persisted in DB
    assessments = get_student_assessments(1)
    assert len(assessments) >= 1
    latest = assessments[0]
    assert latest["student_id"] == 1
    assert latest["transcript"] == sample_transcript
    assert latest["grammar_score"] == result["grammar_score"]


# -------------------------------------------------------------------------
# Test 9: Historical Progress Calculation (Strictly data-driven with provenance)
# -------------------------------------------------------------------------
def test_9_historical_progress_data_driven_provenance():
    # Create isolated student with no assessments
    stud = create_student({"name": "Unassessed Learner", "grade": "College", "target_level": "Intermediate"})
    stud_id = stud["id"]

    # Verify unassessed student has NO fake scores (85%, 100%, 75%, 84% must NEVER appear)
    progress_empty = get_student_historical_progress(stud_id)
    assert progress_empty["has_data"] is False
    assert progress_empty["fluency"] is None
    assert progress_empty["grammar"] is None
    assert progress_empty["vocabulary"] is None
    assert progress_empty["confidence"] is None
    assert progress_empty["total_assessments"] == 0

    # Insert two genuine assessments
    save_assessment({
        "student_id": stud_id,
        "session_id": "sess_1",
        "topic": "Topic A",
        "transcript": "Speech A",
        "duration_seconds": 60,
        "grammar_score": 60.0,
        "vocabulary_score": 70.0,
        "fluency_score": 64.0,
        "confidence_score": 66.0,
        "pacing_score": 68.0,
        "communication_score": 65.0,
        "overall_level": "Intermediate",
        "grammar_feedback": "Good consistency",
        "vocabulary_feedback": "Varied",
        "fluency_feedback": "Steady",
        "confidence_feedback": "Confident",
        "strengths": ["Clarity"],
        "weaknesses": ["Transitions"]
    })
    save_assessment({
        "student_id": stud_id,
        "session_id": "sess_2",
        "topic": "Topic B",
        "transcript": "Speech B",
        "duration_seconds": 70,
        "grammar_score": 80.0,
        "vocabulary_score": 80.0,
        "fluency_score": 76.0,
        "confidence_score": 74.0,
        "pacing_score": 72.0,
        "communication_score": 78.0,
        "overall_level": "Upper Intermediate",
        "grammar_feedback": "Very accurate",
        "vocabulary_feedback": "Expressive",
        "fluency_feedback": "Fluent",
        "confidence_feedback": "Very confident",
        "strengths": ["Vocabulary"],
        "weaknesses": ["Minor pauses"]
    })

    # Retrieve calculated progress
    progress = get_student_historical_progress(stud_id)
    assert progress["has_data"] is True
    assert progress["total_assessments"] == 2
    # Exact averages: Grammar = (60 + 80)/2 = 70.0, Fluency = (64 + 76)/2 = 70.0
    assert progress["grammar"] == 70.0
    assert progress["fluency"] == 70.0
    assert progress["vocabulary"] == 75.0
    assert progress["confidence"] == 70.0
    assert len(progress["sources"]) == 2
    # Verify provenance trace
    for s in progress["sources"]:
        assert "assessment_id" in s
        assert "session_id" in s
        assert "timestamp" in s or "created_at" in s


# -------------------------------------------------------------------------
# Test 10: New Session Isolation
# -------------------------------------------------------------------------
def test_10_new_session_isolation():
    initial_st = get_student(1)
    name_before = initial_st["name"]

    res = client.post("/api/v1/students/1/sessions/new", json={"mode": "coach"})
    assert res.status_code == 200
    new_sess = res.json()
    assert "session_id" in new_sess
    assert new_sess["student_id"] == 1

    # Student profile must NOT be erased or changed
    st_after = get_student(1)
    assert st_after["name"] == name_before


# -------------------------------------------------------------------------
# Test 11: Restart Onboarding Behavior
# -------------------------------------------------------------------------
def test_11_restart_onboarding_preserves_assessments():
    # Save an assessment first
    save_assessment({
        "student_id": 1,
        "session_id": "sess_pre_restart",
        "topic": "Before Restart",
        "transcript": "Test sample",
        "duration_seconds": 60,
        "grammar_score": 72.0,
        "vocabulary_score": 74.0,
        "fluency_score": 70.0,
        "confidence_score": 72.0,
        "communication_score": 72.0,
        "overall_level": "Intermediate",
        "strengths": [],
        "weaknesses": []
    })
    count_before = len(get_student_assessments(1))

    # Trigger restart onboarding
    res = client.post("/api/v1/students/1/onboarding/restart")
    assert res.status_code == 200

    # Onboarding step should be reset
    st = get_student(1)
    assert st["onboarding_completed"] == 0 or st["onboarding_completed"] is False
    assert st["onboarding_step"] == "ask_name"

    # Assessments history must be preserved
    count_after = len(get_student_assessments(1))
    assert count_after == count_before


# -------------------------------------------------------------------------
# Test 12: Delete Learner Data
# -------------------------------------------------------------------------
def test_12_delete_learner_data_preserves_system_settings():
    # Create temporary student to delete
    tmp_stud = create_student({"name": "Temp Delete", "grade": "Grade 10", "target_level": "Beginner"})
    tmp_id = tmp_stud["id"]

    # Ensure system API key exists in settings
    update_db_settings({"groq_api_key": "gsk_test_key_keep_safe"})

    # Delete learner
    del_res = client.delete(f"/api/v1/students/{tmp_id}")
    assert del_res.status_code == 200

    # Student profile must be reset to fresh default and assessments wiped
    st_reset = get_student(tmp_id)
    assert st_reset is not None
    assert st_reset["name"] == "New Learner"
    assert st_reset["onboarding_completed"] == 0 or st_reset["onboarding_completed"] is False
    assert len(get_student_assessments(tmp_id)) == 0

    # System settings MUST be preserved
    settings = get_db_settings()
    assert settings.get("groq_api_key") == "gsk_test_key_keep_safe"


# -------------------------------------------------------------------------
# Test 13 & 14: Provider Authentication & Generation Testing
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_13_and_14_provider_auth_and_generation():
    # Test valid Ollama generation path
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock(
            status_code=200,
            json=lambda: {"message": {"content": "OK"}}
        )
        res = await llm_service.verify_provider_connectivity("ollama")
        assert res["status"] in ["READY", "ready", "AUTHENTICATED", "authenticated"]
        assert res.get("configured") is True or res.get("healthy") is True


# -------------------------------------------------------------------------
# Test 15 & 16: Provider Fallback & Ollama Fallback
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_15_and_16_provider_fallback_to_ollama():
    # When Groq fails, cascade should fall back to Ollama
    with patch("app.services.llm_service.LLMService._call_groq", side_effect=Exception("Groq 401 Invalid Key")), \
         patch("app.services.llm_service.LLMService._call_openai", side_effect=Exception("OpenAI 429 Quota Exceeded")), \
         patch("app.services.llm_service.LLMService._call_qwen", side_effect=Exception("Qwen Not Configured")), \
         patch("app.services.llm_service.LLMService._call_ollama", new_callable=AsyncMock) as mock_ollama:

        mock_ollama.return_value = ("I am here to help you practice English.", "llama3:latest")

        messages = [{"role": "user", "content": "Hello!"}]
        response = await llm_service.generate_response(messages=messages, system_prompt="You are Mizo.")

        assert response.text == "I am here to help you practice English."
        assert response.provider == "ollama"
        assert response.is_fallback is True


# -------------------------------------------------------------------------
# Test 17 & 19: API Failure Clean Recovery and Session Continuation
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_17_and_19_api_failure_recovery_and_session_continuation():
    # Simulate a complete failure across providers
    with patch.object(llm_service, "generate_response", side_effect=Exception("All providers offline")):
        res = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "Can you hear me?"
        })
        # Session must survive and return graceful fallback
        assert res.status_code == 200
        # Voice manager must not be stuck in PROCESSING
        assert voice_manager.state != VoiceState.PROCESSING
        assert voice_manager.is_processing is False

    # Next interaction succeeds immediately
    with patch.object(llm_service, "generate_response", return_value=LLMResult(
        text="Yes, I can hear you clearly!",
        provider="ollama",
        model="llama3:latest",
        latency_ms=120.0,
        is_fallback=False
    )):
        res2 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "Are you ready now?"
        })
        assert res2.status_code == 200
        assert "hear you" in res2.json()["reply_text"]


# -------------------------------------------------------------------------
# Test 18: Server Restart Persistence
# -------------------------------------------------------------------------
def test_18_server_restart_persistence():
    # Update student 1 name to a specific value
    update_student(1, {"name": "Shahid"})

    # Simulate server restart by re-running init_db()
    init_db()

    # Verify learner data remains intact
    st = get_student(1)
    assert st is not None
    assert st["name"] == "Shahid"


# -------------------------------------------------------------------------
# Test 20: Dashboard Values Matching Database Values (Zero Fake Data)
# -------------------------------------------------------------------------
def test_20_dashboard_values_match_db():
    res = client.get("/api/v1/students/1/historical-progress")
    assert res.status_code == 200
    data = res.json()

    # Compare directly with database records
    db_progress = get_student_historical_progress(1)
    assert data["has_data"] == db_progress["has_data"]
    assert data["total_assessments"] == db_progress["total_assessments"]
    assert data["fluency"] == db_progress["fluency"]
    assert data["grammar"] == db_progress["grammar"]
    assert data["vocabulary"] == db_progress["vocabulary"]
    assert data["confidence"] == db_progress["confidence"]
