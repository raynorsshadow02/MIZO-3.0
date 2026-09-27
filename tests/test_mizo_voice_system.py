"""
Comprehensive Verification Suite for Mizo Voice System
======================================================
Tests all requirements:
1. Rename assistant to 'Mizo'
2. Wake-up works continuously without any button
3. Wake word = 'Mizo' + natural addressing variations
4. Whisper phonetic misrecognition handling ('Meeso', 'Meso', 'Mezo', 'Mizzo', 'Miso', 'Meiso')
5. Conservative matching: do NOT wake from random speech
6. Standby listener loop is continuous and recovers from timeouts/errors
7. Sleep/snooze behavior: 10s inactivity -> SLEEP (mic monitoring active, ignores normal speech, wakes on 'Mizo')
8. Button no longer required anywhere in cycle
9. Immediate Test A, B, C, D, E
10. Explicit debugging logs formatted per specification
"""

import io
import sys
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import get_student, update_student, reset_student_to_fresh
from app.services.voice_service import (
    voice_manager,
    VoiceState,
    ASSISTANT_NAME,
    detect_mizo_intent,
    is_wake_word,
    is_stop_command,
    run_continuous_standby_listener
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Canonical Assistant Identity
# ---------------------------------------------------------------------------
def test_assistant_name_is_mizo():
    assert ASSISTANT_NAME == "Mizo"


# ---------------------------------------------------------------------------
# 2. Centralized Intent Detection: Natural Wake Words
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("phrase", [
    "Mizo",
    "Hey Mizo",
    "Hello Mizo",
    "Hi Mizo",
    "Hey, Mizo",
    "Mizo, are you there?",
    "What's up, Mizo?",
    "Can you hear me, Mizo?",
    "Mizo, wake up",
    "Wake up, Mizo",
    "Wake up Mizo",
    "mizo",
    "hey mizo",
    "hello mizo",
    "hi mizo",
    "hey, mizo"
])
def test_detect_mizo_intent_natural_wake_phrases(phrase):
    intent = detect_mizo_intent(phrase)
    assert intent == "WAKE", f"Expected WAKE for phrase '{phrase}', got {intent}"


# ---------------------------------------------------------------------------
# 3. Whisper Phonetic Misrecognitions Handling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("whisper_misrecognition", [
    "Meeso",
    "Meso",
    "Mezo",
    "Mizzo",
    "Miso",
    "Meiso",
    "Hey Meeso",
    "Hello Meso",
    "Hi Mezo",
    "Mizzo, wake up",
    "Miso, are you there?",
    "Can you hear me, Meiso?"
])
def test_detect_mizo_intent_whisper_phonetic_variations(whisper_misrecognition):
    intent = detect_mizo_intent(whisper_misrecognition)
    assert intent == "WAKE", f"Expected WAKE for phonetic misrecognition '{whisper_misrecognition}', got {intent}"


# ---------------------------------------------------------------------------
# 4. Conservative Matching: Do NOT Wake from Random Speech
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("random_speech", [
    "The weather is nice today.",
    "I am studying robotics.",
    "Can you explain Python?",
    "Hello robot.",
    "I want to learn Python.",
    "What is your goal?",
    "Today is Monday.",
    "Good morning everyone."
])
def test_detect_mizo_intent_ignores_random_speech(random_speech):
    intent = detect_mizo_intent(random_speech)
    assert intent == "NORMAL", f"Expected NORMAL for random speech '{random_speech}', got {intent}"


# ---------------------------------------------------------------------------
# 5. Stop Commands Detection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stop_phrase", [
    "Stop Mizo",
    "Mizo stop",
    "Stop Meeso",
    "Meeso stop",
    "Stop Miso",
    "Miso stop",
    "Stop",
    "stop",
    "Stop please",
    "Please stop",
    "Stop listening",
    "Stop talking",
    "Stop now",
    "Stop Mikaza",
    "Mikaza stop"
])
def test_detect_mizo_intent_stop_commands(stop_phrase):
    intent = detect_mizo_intent(stop_phrase)
    assert intent == "STOP", f"Expected STOP for phrase '{stop_phrase}', got {intent}"


# ---------------------------------------------------------------------------
# 6. Empty / Inaudible Input
# ---------------------------------------------------------------------------
def test_detect_mizo_intent_empty_or_whitespace():
    assert detect_mizo_intent("") == "UNKNOWN"
    assert detect_mizo_intent("   ") == "UNKNOWN"


# ---------------------------------------------------------------------------
# 7. Debug Output Logs Verification (Requirement 10)
# ---------------------------------------------------------------------------
def test_debugging_logs_format(capsys):
    # Test WAKE logging
    detect_mizo_intent("Hey Miso")
    captured = capsys.readouterr().out
    assert '[WAKE] Raw transcription: "Hey Miso"' in captured
    assert '[WAKE] Normalized: "hey miso"' in captured
    assert '[WAKE] Similarity to "mizo":' in captured
    assert '[WAKE] Intent: WAKE' in captured

    # Test SLEEP state logging
    voice_manager.transition_to(VoiceState.STANDBY, "test_sleep")
    captured_sleep = capsys.readouterr().out
    assert "[VOICE] State: SLEEP" in captured_sleep
    assert "[VOICE] Wake listener: ACTIVE" in captured_sleep
    assert "[VOICE] Microphone: ACTIVE" in captured_sleep


# ---------------------------------------------------------------------------
# 8. Immediate Tests A, B, C, D, E (Requirement 9)
# ---------------------------------------------------------------------------
def test_immediate_test_a_mizo_wakes_without_button(capsys):
    """
    Test A:
    Start program in SLEEP. Do NOT press button.
    Say: 'Mizo'
    Expected:
    [VOICE] Wake word detected: Mizo
    [VOICE] Waking assistant
    [VOICE] State: LISTENING
    """
    voice_manager.transition_to(VoiceState.STANDBY, "test_setup")
    _ = capsys.readouterr()  # Clear buffer

    res = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Mizo"
    })
    assert res.status_code == 200
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING

    captured = capsys.readouterr().out
    assert "[VOICE] Wake word detected: Mizo" in captured
    assert "[VOICE] Waking assistant" in captured
    assert "[VOICE] State: LISTENING" in captured


def test_immediate_test_b_hey_mizo_wakes_without_button():
    """Test B: Standby + 'Hey Mizo' -> Wake."""
    voice_manager.transition_to(VoiceState.STANDBY, "test_setup")
    res = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Hey Mizo"
    })
    assert res.status_code == 200
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING


def test_immediate_test_c_hello_mizo_wakes_without_button():
    """Test C: Standby + 'Hello Mizo' -> Wake."""
    voice_manager.transition_to(VoiceState.STANDBY, "test_setup")
    res = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Hello Mizo"
    })
    assert res.status_code == 200
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING


def test_immediate_test_d_mizo_are_you_there_wakes_without_button():
    """Test D: Standby + 'Mizo, are you there?' -> Wake."""
    voice_manager.transition_to(VoiceState.STANDBY, "test_setup")
    res = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Mizo, are you there?"
    })
    assert res.status_code == 200
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING


def test_immediate_test_e_hello_robot_does_not_wake():
    """Test E: Standby + 'Hello robot.' -> No wake, remains in STANDBY / SLEEP."""
    voice_manager.transition_to(VoiceState.STANDBY, "test_setup")
    res = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Hello robot."
    })
    assert res.status_code == 200
    data = res.json()
    assert data["reply_text"] == ""
    assert voice_manager.state == VoiceState.STANDBY


# ---------------------------------------------------------------------------
# 9. Continuous Standby Listener Loop Recovery (Requirement 6)
# ---------------------------------------------------------------------------
def test_continuous_standby_listener_loop_recovery():
    """
    Standby listener loop:
    - Continues after TimeoutError
    - Continues after recoverable microphone error
    - Does NOT permanently exit on 3s, 10s, 12s timeout
    - Wakes upon 'Mizo'
    """
    iteration = 0
    woken = []
    recovered_count = 0

    def mock_capture():
        nonlocal iteration
        iteration += 1
        if iteration == 1:
            raise TimeoutError("Microphone read timeout")
        elif iteration == 2:
            raise RuntimeError("Audio device buffer underrun")
        elif iteration == 3:
            return b"dummy_audio_bytes"
        return None

    def mock_transcribe(audio):
        return "Mizo"

    def mock_recover():
        nonlocal recovered_count
        recovered_count += 1

    def on_wake(trailing):
        woken.append(trailing)

    def should_stop():
        return len(woken) > 0 or iteration >= 5

    run_continuous_standby_listener(
        capture_audio_func=mock_capture,
        transcribe_func=mock_transcribe,
        on_wake_callback=on_wake,
        recover_microphone_func=mock_recover,
        should_stop_func=should_stop
    )

    assert len(woken) == 1
    assert recovered_count >= 1
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING


# ---------------------------------------------------------------------------
# 10. Sleep/Snooze Behavior (Requirement 7)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_snooze_behavior_ignores_normal_speech_until_mizo():
    """
    10 seconds inactivity -> SLEEP mode:
    - Mic monitoring is active
    - Normal speech ('I want to learn Python.') is ignored
    - 'Mizo' wakes the assistant
    - Once awake, speech is processed normally
    """
    reset_student_to_fresh(1)
    update_student(1, {"onboarding_step": "ask_goals", "name": "Shahid"})

    # Enter sleep after 10s inactivity
    voice_manager.enter_snooze()
    assert voice_manager.state == VoiceState.STANDBY

    # User speaks ordinary speech during sleep -> IGNORED
    res_ignored = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "I want to learn Python."
    })
    assert res_ignored.status_code == 200
    assert res_ignored.json()["reply_text"] == ""
    assert voice_manager.state == VoiceState.STANDBY

    # User says wake name 'Mizo' -> WAKES
    res_wake = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Mizo"
    })
    assert res_wake.status_code == 200
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING

    # Now that it's ACTIVE_LISTENING, user's input is processed
    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm, \
         patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        from app.services.llm_service import LLMResult
        mock_llm.return_value = LLMResult(
            text="Python is a fantastic goal! What challenges do you face?",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False,
            latency_ms=100.0
        )
        mock_tts.return_value = ("/audio/goal.wav", "goal.wav")

        res_answer = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "I want to learn Python."
        })
        assert res_answer.status_code == 200
        assert res_answer.json()["reply_text"] != ""
