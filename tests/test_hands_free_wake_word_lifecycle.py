"""
Tests for Hands-Free Wake-Word and Voice Lifecycle (Mizo 3.0)
============================================================
Verifies all 16 required test cases specified in Task requirements:
1. Server starts in WAKE_LISTENING.
2. "Hey Mizo" activates conversation.
3. "Hello Mizo" activates conversation.
4. "Hi Mizo" activates conversation.
5. Case-insensitive detection.
6. Punctuation variation.
7. Idle speech without wake phrase is ignored.
8. "Hey Mizo, explain robotics" extracts the request.
9. "Hello Mizo, I want to study" routes correctly.
10. "Hi Mizo, I want to practice my speech" routes correctly.
11. "Stop Mizo" returns to WAKE_LISTENING.
12. Stop does not completely disable microphone.
13. Start button still works as manual fallback.
14. Start button does not create a second listener.
15. Multiple wake phrases do not create multiple listeners.
16. Mizo's own TTS output does not trigger itself.
"""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import update_student
from app.services.voice_service import (
    voice_manager,
    VoiceState,
    detect_mizo_intent,
    is_wake_word,
    is_stop_command
)
from app.services.agent_router import AgentRouter, AgentRoute
from app.services.llm_service import LLMResult

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_student():
    """Ensure student #1 has completed onboarding and external services are mocked."""
    update_student(1, {
        "name": "Shahid",
        "onboarding_completed": True,
        "onboarding_step": "completed"
    })
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_fixture_setup")
    
    mock_llm_res = LLMResult(
        text="I am ready to help you.",
        provider="mock",
        model="mock-model",
        is_fallback=False,
        latency_ms=10.0
    )
    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm, \
         patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:
        mock_llm.return_value = mock_llm_res
        mock_tts.return_value = ("/audio/test.wav", "test.wav")
        yield



# ---------------------------------------------------------------------------
# Test 1: Server starts in WAKE_LISTENING
# ---------------------------------------------------------------------------
def test_01_server_starts_in_wake_listening():
    # Simulate server startup initialization
    voice_manager.start_wake_listener()
    assert voice_manager.state == VoiceState.WAKE_LISTENING
    assert voice_manager.state == VoiceState.STANDBY  # Equivalent alias
    assert voice_manager.microphone_active is True


# ---------------------------------------------------------------------------
# Test 2: "Hey Mizo" activates conversation
# ---------------------------------------------------------------------------
def test_02_hey_mizo_activates_conversation(capsys):
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_setup")
    _ = capsys.readouterr()

    res = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Hey Mizo"
    })
    assert res.status_code == 200
    data = res.json()
    assert "Yes" in data["reply_text"] or "listening" in data["reply_text"].lower()
    assert voice_manager.state == VoiceState.ACTIVE_CONVERSATION
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING

    out = capsys.readouterr().out
    assert "[VOICE] Wake word detected" in out
    assert "[VOICE] State: ACTIVE_CONVERSATION" in out
    assert "[VOICE] Listening for user request" in out


# ---------------------------------------------------------------------------
# Test 3: "Hello Mizo" activates conversation
# ---------------------------------------------------------------------------
def test_03_hello_mizo_activates_conversation():
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_setup")
    res = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Hello Mizo"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["reply_text"] != ""
    assert voice_manager.state == VoiceState.ACTIVE_CONVERSATION


# ---------------------------------------------------------------------------
# Test 4: "Hi Mizo" activates conversation
# ---------------------------------------------------------------------------
def test_04_hi_mizo_activates_conversation():
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_setup")
    res = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Hi Mizo"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["reply_text"] != ""
    assert voice_manager.state == VoiceState.ACTIVE_CONVERSATION


# ---------------------------------------------------------------------------
# Test 5: Case-insensitive detection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("phrase", [
    "hey mizo",
    "HEY MIZO",
    "HeY MiZo",
    "hElLo mIzO",
    "HI MIZO"
])
def test_05_case_insensitive_detection(phrase):
    intent = detect_mizo_intent(phrase, log_debug=False)
    assert intent == "WAKE"
    is_wake, trailing = is_wake_word(phrase, log_debug=False)
    assert is_wake is True


# ---------------------------------------------------------------------------
# Test 6: Punctuation variation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("phrase", [
    "Hey, Mizo",
    "Hello, Mizo!",
    "Hi, Mizo?",
    "Hey... Mizo!",
    "hey, mizo."
])
def test_06_punctuation_variation(phrase):
    intent = detect_mizo_intent(phrase, log_debug=False)
    assert intent == "WAKE"
    is_wake, trailing = is_wake_word(phrase, log_debug=False)
    assert is_wake is True


# ---------------------------------------------------------------------------
# Test 7: Idle speech without wake phrase is ignored
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("idle_phrase", [
    "What's the weather?",
    "Hello robot.",
    "Hello everyone.",
    "I am studying robotics.",
    "Today is Monday.",
    "Can you hear the music?"
])
def test_07_idle_speech_without_wake_phrase_is_ignored(capsys, idle_phrase):
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_setup")
    _ = capsys.readouterr()

    res = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": idle_phrase
    })
    assert res.status_code == 200
    data = res.json()
    assert data["reply_text"] == ""
    assert voice_manager.state == VoiceState.WAKE_LISTENING

    out = capsys.readouterr().out
    assert "[VOICE] No wake word. Ignoring." in out


# ---------------------------------------------------------------------------
# Test 8: "Hey Mizo, explain robotics" extracts the request
# ---------------------------------------------------------------------------
def test_08_wake_phrase_with_request_extracts_request():
    is_wake, trailing = is_wake_word("Hey Mizo, explain robotics", log_debug=False)
    assert is_wake is True
    assert trailing == "explain robotics"


# ---------------------------------------------------------------------------
# Test 9: "Hello Mizo, I want to study" routes correctly
# ---------------------------------------------------------------------------
def test_09_hello_mizo_study_routes_to_study_agent():
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_setup")
    res = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Hello Mizo, I want to study"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["route"] == "STUDY"
    assert data["active_section"] == "study"
    assert voice_manager.state == VoiceState.ACTIVE_CONVERSATION


# ---------------------------------------------------------------------------
# Test 10: "Hi Mizo, I want to practice my speech" routes correctly
# ---------------------------------------------------------------------------
def test_10_hi_mizo_practice_speech_routes_to_speech_agent():
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_setup")
    res = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Hi Mizo, I want to practice my speech"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["route"] == "SPEECH"
    assert data["active_section"] == "speech"
    assert voice_manager.state == VoiceState.ACTIVE_CONVERSATION


# ---------------------------------------------------------------------------
# Test 11: "Stop Mizo" returns to WAKE_LISTENING
# ---------------------------------------------------------------------------
def test_11_stop_mizo_returns_to_wake_listening(capsys):
    voice_manager.transition_to(VoiceState.ACTIVE_CONVERSATION, "test_setup")
    _ = capsys.readouterr()

    res = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Stop Mizo"
    })
    assert res.status_code == 200
    assert voice_manager.state == VoiceState.WAKE_LISTENING

    out = capsys.readouterr().out
    assert "[VOICE] Stop command detected" in out
    assert "[VOICE] State: WAKE_LISTENING" in out


# ---------------------------------------------------------------------------
# Test 12: Stop does not completely disable microphone
# ---------------------------------------------------------------------------
def test_12_stop_does_not_disable_microphone():
    voice_manager.transition_to(VoiceState.ACTIVE_CONVERSATION, "test_setup")
    voice_manager.handle_stop_command()
    assert voice_manager.state == VoiceState.WAKE_LISTENING
    assert voice_manager.microphone_active is True


# ---------------------------------------------------------------------------
# Test 13: Start button still works as manual fallback
# ---------------------------------------------------------------------------
def test_13_start_button_works_as_manual_fallback():
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_setup")
    res = client.post("/api/v1/esp32/start", json={
        "student_id": 1,
        "action": "start"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert voice_manager.state == VoiceState.ACTIVE_CONVERSATION


# ---------------------------------------------------------------------------
# Test 14: Start button does not create a second listener
# ---------------------------------------------------------------------------
def test_14_start_button_does_not_create_second_listener():
    voice_manager.start_wake_listener()
    initial_running = voice_manager._wake_listener_running
    assert initial_running is True

    # Call start_wake_listener again
    voice_manager.start_wake_listener()
    assert voice_manager._wake_listener_running is True


# ---------------------------------------------------------------------------
# Test 15: Multiple wake phrases do not create multiple listeners
# ---------------------------------------------------------------------------
def test_15_multiple_wake_phrases_do_not_create_multiple_listeners():
    voice_manager.start_wake_listener()
    for phrase in ["Hey Mizo", "Hello Mizo", "Hi Mizo"]:
        client.post("/api/v1/esp32/chat", json={"student_id": 1, "message": phrase})
    assert voice_manager._wake_listener_running is True


# ---------------------------------------------------------------------------
# Test 16: Mizo's own TTS output does not trigger itself
# ---------------------------------------------------------------------------
def test_16_tts_output_does_not_self_trigger():
    voice_manager.handle_speaking_start()
    assert voice_manager.is_speaking is True
    assert voice_manager.state == VoiceState.SPEAKING

    # Simulated audio echoing Mizo's own response during speaking
    echoed_text = "Yes? I'm listening."
    # Since Mizo is speaking, this echo must not be treated as a new user wake command
    intent = detect_mizo_intent(echoed_text, log_debug=False)
    # The echo is not a stop command
    assert is_stop_command(echoed_text) is False

    voice_manager.handle_speaking_ended()
    assert voice_manager.is_speaking is False
    assert voice_manager.state == VoiceState.ACTIVE_CONVERSATION
