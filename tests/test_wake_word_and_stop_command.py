"""
Tests for Mizo Wake Word ("Hey Mizo") and Stop Command ("Stop Mizo")
===================================================================
Covers the requirements specified in Section 6:
1. Wake phrase recognized while Mizo is idle.
2. Wake phrase starts listening only once.
3. A normal phrase that does not match the wake word does not trigger Mizo.
4. Stop command interrupts active speech without playing reply audio.
5. Stop command is not sent to normal question answering.
6. Wake detection resumes after stopping.
7. Wake detection recovers after errors and timeouts.
8. Wake phrase with trailing question extracts query and answers.
9. Microphone remains active throughout idle and stopping.
"""

import io
import time
import wave
import struct
from unittest.mock import patch, AsyncMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import update_student, PROTOTYPE_STUDENT_ID
from app.services.voice_service import (
    voice_manager,
    VoiceState,
    is_wake_word,
    is_stop_command,
    classify_command_or_intent,
    detect_mizo_intent
)
from app.services.llm_service import LLMResult

client = TestClient(app)


def _generate_mock_wav(duration_seconds: float = 0.5, sample_rate: int = 16000) -> io.BytesIO:
    """Generates a valid 16kHz 16-bit mono PCM WAV file in memory."""
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        num_frames = int(sample_rate * duration_seconds)
        wf.writeframes(bytearray(num_frames * 2))
    wav_io.seek(0)
    return wav_io


@pytest.fixture(autouse=True)
def setup_test_student():
    """Ensure student #1 has completed onboarding and voice_manager is in WAKE_LISTENING."""
    update_student(PROTOTYPE_STUDENT_ID, {
        "name": "Shahid",
        "onboarding_completed": True,
        "onboarding_step": "completed"
    })
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_fixture_setup")
    voice_manager.last_interaction_timestamp = time.time()
    yield


# -----------------------------------------------------------------------------
# Test 1: Wake phrase recognized while Mizo is idle
# -----------------------------------------------------------------------------
def test_wake_phrase_recognized_while_idle(capsys):
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_setup")
    _ = capsys.readouterr()

    # Intent detection check
    intent = detect_mizo_intent("Hey Mizo", log_debug=True)
    assert intent == "WAKE"

    # API upload with "Hey Mizo"
    wav_io = _generate_mock_wav()
    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        mock_stt.return_value = {"text": "Hey Mizo", "duration_seconds": 0.5, "language": "en"}
        mock_tts.return_value = ("/audio/mock_wake.wav", "mock_wake.wav")

        files = {"audio": ("inmp441.wav", wav_io, "audio/wav")}
        data = {"student_id": "1", "mode": "coach"}
        res = client.post("/api/v1/esp32/audio?format=json", files=files, data=data)

        assert res.status_code == 200
        data_res = res.json()
        assert data_res["activated"] is True
        assert "Yes, Shahid? I'm listening." in data_res["reply_text"] or "listening" in data_res["reply_text"].lower()
        assert voice_manager.state == VoiceState.ACTIVE_LISTENING
        assert voice_manager.microphone_active is True

    out = capsys.readouterr().out
    assert "[VOICE] Wake word detected" in out


# -----------------------------------------------------------------------------
# Test 2: Wake phrase starts listening only once
# -----------------------------------------------------------------------------
def test_wake_phrase_starts_listening_only_once():
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_setup")
    voice_manager.start_wake_listener()
    initial_running = voice_manager._wake_listener_running

    # Call start_wake_listener multiple times
    voice_manager.start_wake_listener()
    voice_manager.start_wake_listener()

    assert voice_manager._wake_listener_running is True
    assert voice_manager.microphone_active is True


# -----------------------------------------------------------------------------
# Test 3: Normal phrase that does not match wake word does not trigger Mizo
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("non_wake_text", [
    "What is the weather outside?",
    "I am talking to my colleague.",
    "Can you hear the car driving by?",
    "Good morning everyone.",
    "The test score is eighty percent."
])
def test_non_wake_phrase_ignored_in_standby(non_wake_text):
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_setup")

    wav_io = _generate_mock_wav()
    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:

        mock_stt.return_value = {"text": non_wake_text, "duration_seconds": 1.0, "language": "en"}

        files = {"audio": ("inmp441.wav", wav_io, "audio/wav")}
        data = {"student_id": "1", "mode": "coach"}
        res = client.post("/api/v1/esp32/audio?format=audio", files=files, data=data)

        assert res.status_code == 200
        # Crucial: Must return application/json so MAX98357A speaker receives zero audio bytes
        assert "application/json" in res.headers["content-type"].lower()
        payload = res.json()
        assert payload.get("ignored") is True
        assert payload.get("reply_text") == ""
        # LLM must NEVER have been called
        mock_llm.assert_not_called()
        # Voice state must remain in idle wake listening
        assert voice_manager.state == VoiceState.WAKE_LISTENING


# -----------------------------------------------------------------------------
# Test 4: Stop command interrupts active speech and does NOT return audio
# -----------------------------------------------------------------------------
def test_stop_command_interrupts_and_returns_no_audio(capsys):
    voice_manager.transition_to(VoiceState.SPEAKING, "tts_active")
    _ = capsys.readouterr()

    wav_io = _generate_mock_wav()
    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:

        mock_stt.return_value = {"text": "Stop Mizo", "duration_seconds": 0.5, "language": "en"}

        files = {"audio": ("inmp441.wav", wav_io, "audio/wav")}
        data = {"student_id": "1", "mode": "coach"}
        res = client.post("/api/v1/esp32/audio?format=audio", files=files, data=data)

        assert res.status_code == 200
        # Crucial: Must NOT return audio/wav when stopping!
        assert "application/json" in res.headers["content-type"].lower()
        payload = res.json()
        assert payload["is_control_command"] is True
        assert payload["command"] == "stop"
        assert payload["voice_state"] in ["PAUSED", "STANDBY"]
        assert payload["audio_url"] is None
        mock_llm.assert_not_called()

        # State must return to WAKE_LISTENING
        assert voice_manager.state == VoiceState.WAKE_LISTENING
        assert voice_manager.microphone_active is True

    out = capsys.readouterr().out
    assert "[VOICE] Stop command detected" in out
    assert "[VOICE] Playback interrupted" in out
    assert "[VOICE] Wake listener resumed" in out


# -----------------------------------------------------------------------------
# Test 5: Stop command is not sent to normal question answering
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("stop_phrase", [
    "Stop Mizo",
    "mizo stop",
    "stop please",
    "please stop mizo",
    "stop",
    "میزو سٹاپ"
])
def test_stop_command_bypasses_llm_qa(stop_phrase):
    voice_manager.transition_to(VoiceState.ACTIVE_CONVERSATION, "in_conversation")

    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:
        res = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": stop_phrase
        })

        assert res.status_code == 200
        data = res.json()
        assert data["is_control_command"] is True
        assert data["command"] == "stop"
        assert data["voice_state"] in ["PAUSED", "STANDBY"]
        assert data["reply_text"] == ""
        # Zero LLM calls
        mock_llm.assert_not_called()


# -----------------------------------------------------------------------------
# Test 6: Wake detection resumes after stopping
# -----------------------------------------------------------------------------
def test_wake_detection_resumes_after_stopping():
    # 1. Start in active conversation
    voice_manager.transition_to(VoiceState.ACTIVE_CONVERSATION, "active")

    # 2. Issue stop command
    voice_manager.handle_stop_command()
    assert voice_manager.state == VoiceState.WAKE_LISTENING
    assert voice_manager.microphone_active is True

    # 3. Issue wake command again
    is_wake, trailing = is_wake_word("Hey Mizo", log_debug=False)
    assert is_wake is True
    assert trailing == ""

    # Simulate standby upload with Hey Mizo
    wav_io = _generate_mock_wav()
    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        mock_stt.return_value = {"text": "Hey Mizo", "duration_seconds": 0.5, "language": "en"}
        mock_tts.return_value = ("/audio/mock.wav", "mock.wav")

        files = {"audio": ("inmp441.wav", wav_io, "audio/wav")}
        data = {"student_id": "1", "mode": "coach"}
        res = client.post("/api/v1/esp32/audio?format=json", files=files, data=data)

        assert res.status_code == 200
        data_res = res.json()
        assert data_res["activated"] is True
        assert voice_manager.state == VoiceState.ACTIVE_LISTENING


# -----------------------------------------------------------------------------
# Test 7: Wake detection recovers after errors and timeouts
# -----------------------------------------------------------------------------
def test_wake_detection_recovers_after_errors_and_timeouts():
    # A. Inactivity timeout recovery
    voice_manager.transition_to(VoiceState.ACTIVE_CONVERSATION, "active")
    voice_manager.last_interaction_timestamp = time.time() - 25.0  # Simulated 25s idle
    timed_out = voice_manager.check_inactivity_timeout(timeout_seconds=15.0)
    assert timed_out is True
    assert voice_manager.state == VoiceState.WAKE_LISTENING
    assert voice_manager.microphone_active is True

    # B. STT Error does not crash state machine or disable microphone
    wav_io = _generate_mock_wav()
    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt:
        mock_stt.side_effect = RuntimeError("STT Engine temporarily unavailable")

        files = {"audio": ("inmp441.wav", wav_io, "audio/wav")}
        data = {"student_id": "1", "mode": "coach"}
        res = client.post("/api/v1/esp32/audio?format=json", files=files, data=data)

        assert res.status_code == 200
        payload = res.json()
        assert payload.get("ignored") is True
        # State machine remains healthy and waiting for wake word
        assert voice_manager.state == VoiceState.WAKE_LISTENING
        assert voice_manager.microphone_active is True


# -----------------------------------------------------------------------------
# Test 8: Wake phrase with trailing question extracts query and answers
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_wake_phrase_with_trailing_question_extracts_and_answers():
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_setup")

    wav_io = _generate_mock_wav()
    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm, \
         patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        mock_stt.return_value = {
            "text": "Hey Mizo, what is quantum computing?",
            "duration_seconds": 1.2,
            "language": "en"
        }
        mock_llm.return_value = LLMResult(
            text="Quantum computing utilizes quantum mechanics principles like superposition and entanglement.",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False,
            latency_ms=100.0
        )
        mock_tts.return_value = ("/audio/mock_quantum.wav", "mock_quantum.wav")

        files = {"audio": ("inmp441.wav", wav_io, "audio/wav")}
        data = {"student_id": "1", "mode": "coach"}
        res = client.post("/api/v1/esp32/audio?format=json", files=files, data=data)

        assert res.status_code == 200
        payload = res.json()
        assert payload["success"] is True
        assert "Quantum computing utilizes" in payload["reply_text"]
        # LLM was invoked with the cleaned trailing question
        mock_llm.assert_called_once()
        call_repr = str(mock_llm.call_args).lower()
        assert "what is quantum computing" in call_repr


# -----------------------------------------------------------------------------
# Test 10: Regression Tests for Wake Words, Variations & Lifecycle
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("wake_phrase", [
    "Hey Mizo",
    "Hello Mizo",
    "Hi Mizo",
    "Hey, Mizo!",
    "Mizo",
    "Hey Mizu",
    "Hi Mizu",
    "He Mizu",
    "Hay Mizo",
    "Ay Mizu",
    "Hello Mizu",
    "Mizu"
])
def test_all_wake_greetings_and_transcription_variations(wake_phrase):
    """Verifies that all required greetings and common Whisper variations are recognized as WAKE."""
    intent = detect_mizo_intent(wake_phrase, log_debug=False)
    assert intent == "WAKE", f"Failed for phrase: '{wake_phrase}'"
    is_wake, trailing = is_wake_word(wake_phrase, log_debug=False)
    assert is_wake is True, f"is_wake_word failed for: '{wake_phrase}'"


def test_greeting_followed_by_question_photosynthesis():
    """Verifies 'Hello Mizo, explain photosynthesis' strips wake phrase and processes request immediately."""
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_setup")

    is_wake, trailing = is_wake_word("Hello Mizo, explain photosynthesis", log_debug=False)
    assert is_wake is True
    assert trailing.lower() == "explain photosynthesis"

    wav_io = _generate_mock_wav()
    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.tutor_service.tutor_service.process_tutor_turn", new_callable=AsyncMock) as mock_tutor, \
         patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        mock_stt.return_value = {
            "text": "Hello Mizo, explain photosynthesis",
            "duration_seconds": 1.5,
            "language": "en"
        }
        mock_tutor.return_value = {
            "reply_text": "Photosynthesis is the process plants use to convert sunlight into chemical energy.",
            "provider": "groq",
            "intent": "EXPLAIN",
            "current_subject": "Biology",
            "current_topic": "Photosynthesis"
        }
        mock_tts.return_value = ("/audio/mock_photo.wav", "mock_photo.wav")

        files = {"audio": ("inmp441.wav", wav_io, "audio/wav")}
        data = {"student_id": "1", "mode": "coach"}
        res = client.post("/api/v1/esp32/audio?format=json", files=files, data=data)

        assert res.status_code == 200
        payload = res.json()
        assert payload["success"] is True
        assert "photosynthesis" in payload["reply_text"].lower()
        mock_tutor.assert_called_once()
        tutor_arg_str = str(mock_tutor.call_args).lower()
        assert "explain photosynthesis" in tutor_arg_str
        assert "hello mizo" not in tutor_arg_str


def test_greeting_alone_followed_by_second_utterance_without_wake_word():
    """
    Verifies that 'Hi Mizo' alone wakes up and invites user to speak,
    and subsequent utterance is processed without repeating wake phrase.
    """
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_setup")

    # Step 1: Greeting alone
    wav_io1 = _generate_mock_wav()
    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        mock_stt.return_value = {"text": "Hi Mizo", "duration_seconds": 0.6, "language": "en"}
        mock_tts.return_value = ("/audio/mock_hi.wav", "mock_hi.wav")

        files = {"audio": ("inmp441.wav", wav_io1, "audio/wav")}
        data = {"student_id": "1", "mode": "coach"}
        res = client.post("/api/v1/esp32/audio?format=json", files=files, data=data)

        assert res.status_code == 200
        data1 = res.json()
        assert data1["activated"] is True
        assert "listening" in data1["reply_text"].lower()
        assert voice_manager.state in [VoiceState.ACTIVE_LISTENING, VoiceState.READY]

    # Step 2: User asks question directly without wake word
    wav_io2 = _generate_mock_wav()
    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm, \
         patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        mock_stt.return_value = {
            "text": "What is the capital of France?",
            "duration_seconds": 1.0,
            "language": "en"
        }
        mock_llm.return_value = LLMResult(
            text="The capital of France is Paris.",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False,
            latency_ms=90.0
        )
        mock_tts.return_value = ("/audio/mock_paris.wav", "mock_paris.wav")

        files = {"audio": ("inmp441.wav", wav_io2, "audio/wav")}
        data = {"student_id": "1", "mode": "coach"}
        res2 = client.post("/api/v1/esp32/audio?format=json", files=files, data=data)

        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["success"] is True
        assert "Paris" in data2["reply_text"]
        mock_llm.assert_called_once()


def test_standby_ignores_silence_and_noise_without_resetting_timer():
    """Verifies silence or background noise in standby is ignored without changing state or resetting interaction timestamp."""
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_setup")
    old_ts = 1000000.0
    voice_manager.last_interaction_timestamp = old_ts

    wav_io = _generate_mock_wav()
    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt:
        mock_stt.return_value = {"text": "", "duration_seconds": 0.5, "language": "en"}

        files = {"audio": ("inmp441.wav", wav_io, "audio/wav")}
        data = {"student_id": "1", "mode": "coach"}
        res = client.post("/api/v1/esp32/audio?format=json", files=files, data=data)

        assert res.status_code == 200
        payload = res.json()
        assert payload.get("ignored") is True
        assert voice_manager.state == VoiceState.WAKE_LISTENING
        # Timer was NOT touched
        assert voice_manager.last_interaction_timestamp == old_ts


def test_configurable_inactivity_timeout_default_60s():
    """Verifies conversation inactivity timeout defaults to 60s and returns to standby."""
    voice_manager.transition_to(VoiceState.ACTIVE_CONVERSATION, "test_active")
    now = time.time()

    # 30 seconds idle: should NOT timeout
    voice_manager.last_interaction_timestamp = now - 30.0
    timed_out = voice_manager.check_inactivity_timeout()
    assert timed_out is False
    assert voice_manager.state == VoiceState.ACTIVE_CONVERSATION

    # 60.5 seconds idle: should timeout to WAKE_LISTENING
    voice_manager.last_interaction_timestamp = now - 60.5
    timed_out = voice_manager.check_inactivity_timeout()
    assert timed_out is True
    assert voice_manager.state == VoiceState.WAKE_LISTENING
    assert voice_manager.microphone_active is True

    # After returning to standby, wake recognition continues operational
    is_wake, _ = is_wake_word("Hey Mizo")
    assert is_wake is True


def test_timer_does_not_timeout_during_active_turn_processing_or_tts():
    """Verifies that Mizo never snoozes while a turn is processing, speaking, or generating an answer."""
    voice_manager.transition_to(VoiceState.ACTIVE_CONVERSATION, "test_active")
    now = time.time()
    voice_manager.last_interaction_timestamp = now - 90.0  # 90s idle

    # Case 1: Active turn in progress
    voice_manager.begin_turn()
    assert voice_manager._turn_in_progress > 0
    assert voice_manager.check_inactivity_timeout() is False
    voice_manager.end_turn()

    # Case 2: TTS speaking active
    voice_manager.handle_speaking_start()
    assert voice_manager.is_speaking is True
    assert voice_manager.check_inactivity_timeout() is False

    # Case 3: When speaking finishes, timer is reset and state resumes listening
    voice_manager.handle_speaking_ended()
    assert voice_manager.is_speaking is False
    # Fresh interaction timestamp after TTS completes
    assert time.time() - voice_manager.last_interaction_timestamp < 1.0


def test_active_mode_does_not_reset_timer_on_silence():
    """Verifies that receiving silence/noise in active conversation does NOT reset the inactivity timer."""
    voice_manager.transition_to(VoiceState.ACTIVE_CONVERSATION, "test_active")
    base_ts = time.time() - 40.0
    voice_manager.last_interaction_timestamp = base_ts

    wav_io = _generate_mock_wav()
    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        mock_stt.return_value = {"text": "", "duration_seconds": 0.5, "language": "en"}
        mock_tts.return_value = ("/audio/mock_silence.wav", "mock_silence.wav")

        files = {"audio": ("inmp441.wav", wav_io, "audio/wav")}
        data = {"student_id": "1", "mode": "coach"}
        res = client.post("/api/v1/esp32/audio?format=json", files=files, data=data)

        assert res.status_code == 200
        # Inactivity timestamp was NOT reset by silence
        assert voice_manager.last_interaction_timestamp == base_ts


# -----------------------------------------------------------------------------
# Test 11: Negative Fuzzy Match and Unrelated Speech Guard
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("unrelated_phrase", [
    "Miso soup is delicious",
    "miso paste",
    "miso ramen",
    "Hey Gizmo",
    "Missouri weather",
    "miscellaneous items",
    "robot tutor",
    "The weather is nice today",
    "Python programming language",
    "Good morning teacher"
])
def test_negative_fuzzy_match_cases_are_rejected(unrelated_phrase):
    """Verifies that words merely resembling 'mizo' or food phrases like 'miso soup' do NOT wake Mizo."""
    intent = detect_mizo_intent(unrelated_phrase, log_debug=False)
    assert intent != "WAKE", f"False wake detected for: '{unrelated_phrase}'"
    is_wake, _ = is_wake_word(unrelated_phrase, log_debug=False)
    assert is_wake is False, f"False wake detected by is_wake_word for: '{unrelated_phrase}'"


# -----------------------------------------------------------------------------
# Test 12: Hyphenated, Spaced and Compound Token Variants
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("compound_phrase", [
    "Hey Mi-zo",
    "Hey Mi-zu",
    "Hey Mi Zo",
    "heymizo",
    "heymizu",
    "Hey, Mi-zo!",
    "He Mi-zu",
    "Hi Mi-zo"
])
def test_hyphenated_and_compound_wake_variations(compound_phrase):
    """Verifies that common Whisper tokenizations (hyphenated, split syllables, or joined words) trigger WAKE."""
    intent = detect_mizo_intent(compound_phrase, log_debug=False)
    assert intent == "WAKE", f"Failed for compound phrase: '{compound_phrase}'"
    is_wake, _ = is_wake_word(compound_phrase, log_debug=False)
    assert is_wake is True, f"Failed in is_wake_word for compound phrase: '{compound_phrase}'"


# -----------------------------------------------------------------------------
# Test 13: Concurrent Upload & Duplicate Turn Protection
# -----------------------------------------------------------------------------
def test_concurrent_upload_and_active_turn_protection():
    """Verifies active turn tracking prevents multiple concurrent turns from falsely resetting or snoozing."""
    voice_manager.transition_to(VoiceState.ACTIVE_CONVERSATION, "test_active")
    assert voice_manager._turn_in_progress == 0

    voice_manager.begin_turn()
    assert voice_manager._turn_in_progress == 1
    # Second concurrent turn arrives
    voice_manager.begin_turn()
    assert voice_manager._turn_in_progress == 2

    # Inactivity check must NOT snooze while any turn is active
    voice_manager.last_interaction_timestamp = time.time() - 100.0
    assert voice_manager.check_inactivity_timeout() is False

    voice_manager.end_turn()
    assert voice_manager._turn_in_progress == 1
    assert voice_manager.check_inactivity_timeout() is False

    voice_manager.end_turn()
    assert voice_manager._turn_in_progress == 0
    # Now that turns are done and idle time exceeded, timeout succeeds
    assert voice_manager.check_inactivity_timeout() is True
    assert voice_manager.state == VoiceState.WAKE_LISTENING


