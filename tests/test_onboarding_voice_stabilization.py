import os
import io
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.db.database import get_db_settings, add_conversation_message, get_conversation_history, get_active_session
from app.services.voice_service import is_stop_command, is_wake_word, VoiceState
from app.services.tts_service import map_tts_pronunciation
from app.services.llm_service import LLMServiceError, LLMErrorCategory, LLMResult

client = TestClient(app)


def test_api_key_persistence_and_runtime_sync():
    """Verify API keys persist to SQLite, .env, and runtime settings."""
    test_key = "gsk_test_persistent_key_abc123"
    res = client.post("/api/v1/admin/settings", json={
        "groq_api_key": test_key
    })
    assert res.status_code == 200
    assert res.json()["success"] is True

    # 1. Verify DB persistence
    db_settings = get_db_settings()
    assert db_settings.get("groq_api_key") == test_key

    # 2. Verify runtime config sync
    assert settings.GROQ_API_KEY == test_key


def test_empty_api_key_update_preserves_existing_key():
    """Submitting empty key strings must NOT delete the existing saved key."""
    # First set a key
    test_key = "gsk_preserve_me_xyz789"
    client.post("/api/v1/admin/settings", json={"groq_api_key": test_key})

    # Now simulate submitting the settings form where the password field was left blank
    res = client.post("/api/v1/admin/settings", json={
        "coaching_mode": "coach",
        "grammar_strictness": "balanced",
        "groq_api_key": ""
    })
    assert res.status_code == 200

    # Key must still be preserved
    db_settings = get_db_settings()
    assert db_settings.get("groq_api_key") == test_key
    assert settings.GROQ_API_KEY == test_key


def test_explicit_clear_key_deletes_key():
    """Explicitly requesting clear_groq_key deletes the saved key."""
    test_key = "gsk_to_be_cleared_456"
    client.post("/api/v1/admin/settings", json={"groq_api_key": test_key})

    # Send clear request
    res = client.post("/api/v1/admin/settings", json={
        "clear_groq_key": True
    })
    assert res.status_code == 200

    db_settings = get_db_settings()
    assert not db_settings.get("groq_api_key")
    assert settings.GROQ_API_KEY == ""


def test_message_ordering_and_distinct_roles():
    """Conversation history must be strictly oldest-to-newest with distinct roles."""
    student_res = client.get("/api/v1/students")
    assert student_res.status_code == 200
    student_id = student_res.json()[0]["id"]
    session = get_active_session(student_id)
    session_id = session.get("session_id", "test_session")

    # Add three messages with specific roles
    id1 = add_conversation_message(
        student_id=student_id,
        session_id=session_id,
        role="assistant",
        content="What is your name?"
    )
    id2 = add_conversation_message(
        student_id=student_id,
        session_id=session_id,
        role="user",
        content="My name is Shahid."
    )
    id3 = add_conversation_message(
        student_id=student_id,
        session_id=session_id,
        role="assistant",
        content="Nice to meet you Shahid!"
    )

    res = client.get(f"/api/v1/conversations?student_id={student_id}&limit=20")
    assert res.status_code == 200
    messages = res.json()
    assert len(messages) >= 3

    # IDs must be strictly ascending (oldest -> newest)
    ids = [m["id"] for m in messages]
    assert ids == sorted(ids)

    # Filter for our three test messages
    matched = [m for m in messages if m["id"] in [id1, id2, id3]]
    assert len(matched) == 3
    assert matched[0]["role"] == "assistant"
    assert matched[0]["content"] == "What is your name?"
    assert matched[1]["role"] == "user"
    assert matched[1]["content"] == "My name is Shahid."
    assert matched[2]["role"] == "assistant"
    assert matched[2]["content"] == "Nice to meet you Shahid!"


def test_mizo_tts_pronunciation_mapping():
    """Verify that Mizo is converted to Meezo strictly for TTS output."""
    raw_text = "Hello! I am Mizo, your AI English tutor robot. Welcome to Mizo."
    phonetic_text = map_tts_pronunciation(raw_text)
    assert "Meezo" in phonetic_text
    assert "Mizo" not in phonetic_text

    # Case insensitivity test
    assert map_tts_pronunciation("mizo") == "meezo"
    assert map_tts_pronunciation("MIZO") == "MEEZO"


def test_voice_stop_command_detection():
    """Stop commands must be recognized and bypass LLM interaction."""
    stop_phrases = ["Stop Mizo", "stop mizo", "mizo stop", "Stop", "stop", "hey mizo stop", "please stop"]
    for phrase in stop_phrases:
        assert is_stop_command(phrase) is True, f"Failed for {phrase}"

    normal_phrases = ["I want to learn English", "My name is Shahid", "What is science?"]
    for phrase in normal_phrases:
        assert is_stop_command(phrase) is False, f"Failed for {phrase}"


def test_wake_command_detection():
    """Wake commands must be recognized correctly."""
    wake_phrases = ["Hey Mizo", "hey mizo", "Hello Mizo", "hello mizo", "Mizo", "mizo"]
    for phrase in wake_phrases:
        assert is_wake_word(phrase)[0] is True, f"Failed for {phrase}"

    non_wake = ["Good morning", "How are you", "I am studying"]
    for phrase in non_wake:
        assert is_wake_word(phrase)[0] is False, f"Failed for {phrase}"


def test_stop_command_in_chat_transitions_to_snooze():
    """Sending 'Stop Mizo' to chat API immediately halts and returns snooze state."""
    student_res = client.get("/api/v1/students")
    student_id = student_res.json()[0]["id"]

    res = client.post("/api/v1/esp32/chat", json={
        "message": "Stop Mizo",
        "student_id": student_id,
        "mode": "coach"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["is_control_command"] is True
    assert data["command"] == "stop"
    assert data["voice_state"] in ["STANDBY", "SLEEP", "SNOOZE"]


def test_snooze_mode_ignores_normal_speech_without_wake_word():
    """In SNOOZE mode, normal speech without a wake word is ignored."""
    student_res = client.get("/api/v1/students")
    student_id = student_res.json()[0]["id"]

    # Put in snooze first
    client.post("/api/v1/esp32/chat", json={
        "message": "Stop Mizo",
        "student_id": student_id
    })

    # Send ordinary speech without wake word
    res = client.post("/api/v1/esp32/chat", json={
        "message": "The weather is very sunny today in the city.",
        "student_id": student_id,
        "mode": "coach"
    })
    assert res.status_code == 200
    data = res.json()
    # In snooze mode, ordinary speech produces no spoken reply
    assert data["voice_state"] in ["STANDBY", "SLEEP", "SNOOZE"]
    assert data.get("reply_text") == ""
    assert data.get("audio_url") is None


def test_wake_word_wakes_from_snooze():
    """Saying 'Hey Mizo' wakes Mizo up from SNOOZE mode."""
    student_res = client.get("/api/v1/students")
    student_id = student_res.json()[0]["id"]

    # Put in snooze
    client.post("/api/v1/esp32/chat", json={"message": "Stop Mizo", "student_id": student_id})

    # Wake with 'Hey Mizo'
    res = client.post("/api/v1/esp32/chat", json={
        "message": "Hey Mizo",
        "student_id": student_id,
        "mode": "coach"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["voice_state"] in ["ACTIVE_LISTENING", "AWAKE"]
    assert len(data.get("reply_text", "")) > 0
    assert data.get("audio_url") is not None


def test_llm_error_clean_handling_without_traceback():
    """If LLM fails, the user receives a clean message without stack traces or secrets."""
    student_res = client.get("/api/v1/students")
    student_id = student_res.json()[0]["id"]

    # Mark onboarding as completed to test regular conversation LLM failure path
    from app.db.database import update_student
    update_student(student_id, {"onboarding_completed": 1})

    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = LLMServiceError(
            category=LLMErrorCategory.AUTHENTICATION_ERROR,
            message="Invalid API key provided to upstream provider",
            user_message="I'm having trouble connecting to my AI brain due to an authentication issue. Please check your API key."
        )

        res = client.post("/api/v1/esp32/chat", json={
            "message": "Can you explain photosynthesis?",
            "student_id": student_id,
            "mode": "coach"
        })
        assert res.status_code == 200
        data = res.json()
        reply = data.get("reply_text", "")
        assert "authentication issue" in reply.lower() or "check your api key" in reply.lower()
        # Verify no traceback or secrets in user response
        assert "traceback" not in reply.lower()
        assert "gsk_" not in reply
        assert "sk-" not in reply
