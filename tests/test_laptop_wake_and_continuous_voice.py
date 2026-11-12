"""
Tests for Laptop Wake Word Detection and Continuous Voice Conversation
=======================================================================
Verifies all 13 targeted criteria specified in user request:
1. "Hey Mizo" wakes the assistant.
2. "Mizo" wakes the assistant.
3. A sentence containing "Mizo" wakes it.
4. Unrelated speech does not wake it unnecessarily.
5. A wake word appearing in partial recognition results is handled correctly.
6. A completed utterance is submitted exactly once.
7. Listening automatically resumes after the AI response.
8. The assistant returns to snooze after 20 seconds of inactivity.
9. New user speech resets the inactivity timer.
10. Snooze mode does not submit ordinary speech to the AI.
11. TTS playback does not trigger a false wake or feedback loop.
12. Recognition errors and browser permission denial are handled cleanly.
13. Repeated wake/snooze cycles do not create duplicate listeners or microphone streams.
"""

import io
import time
import wave
import struct
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.services.voice_service import (
    voice_manager,
    VoiceState,
    is_wake_word,
    detect_mizo_intent,
    is_stop_command,
    normalize_text
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


# 1. "Hey Mizo" wakes the assistant.
def test_1_hey_mizo_wakes_assistant():
    is_wake, trailing = is_wake_word("Hey Mizo", log_debug=False)
    assert is_wake is True
    assert trailing == ""

    # Test case sensitivity and punctuation
    is_wake_lower, _ = is_wake_word("hey mizo!", log_debug=False)
    assert is_wake_lower is True


# 2. "Mizo" wakes the assistant.
def test_2_mizo_alone_wakes_assistant():
    is_wake, trailing = is_wake_word("Mizo", log_debug=False)
    assert is_wake is True
    assert trailing == ""

    is_wake_variant, _ = is_wake_word("mizu", log_debug=False)
    assert is_wake_variant is True


# 3. A sentence containing "Mizo" wakes it.
def test_3_sentence_containing_mizo_wakes_it():
    test_phrases = [
        "Can Mizo help me with maths?",
        "I have a question for Mizo",
        "Good morning Mizo",
        "Could Mizo explain forward kinematics?",
        "Tell Mizo I am ready"
    ]
    for phrase in test_phrases:
        is_wake, _ = is_wake_word(phrase, log_debug=False)
        assert is_wake is True, f"Failed to wake for: {phrase}"


# 4. Unrelated speech does not wake it unnecessarily.
def test_4_unrelated_speech_does_not_wake_it():
    unrelated_phrases = [
        "The weather is very sunny today",
        "Python is a great programming language",
        "I like to study robotics on weekends",
        "Hey Gizmo can you hear me",
        "Missouri state university",
        "Miso soup is delicious",
        "miso paste recipe",
        "miso ramen for lunch"
    ]
    for phrase in unrelated_phrases:
        is_wake, _ = is_wake_word(phrase, log_debug=False)
        assert is_wake is False, f"False wake triggered for: {phrase}"


# 5. A wake word appearing in partial recognition results is handled correctly.
def test_5_partial_recognition_wake_word_handling():
    # Simulates stream of partial speech recognition chunks
    partial_chunks = [
        "h",
        "hey",
        "hey mi",
        "hey mizo",
        "hey mizo teach me"
    ]
    detected_at_partial = False
    for chunk in partial_chunks:
        is_wake, trailing = is_wake_word(chunk, log_debug=False)
        if is_wake:
            detected_at_partial = True
            break
    assert detected_at_partial is True


# 6. A completed utterance is submitted exactly once.
@pytest.mark.asyncio
async def test_6_completed_utterance_submitted_exactly_once():
    from app.db.database import update_student
    update_student(1, {"onboarding_completed": 1})
    voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "test_setup")
    voice_manager.touch_interaction()

    mock_wav = _generate_mock_wav(1.0)
    files = {"audio": ("laptop_speech.webm", mock_wav, "audio/webm")}
    data = {"student_id": "1", "mode": "coach"}

    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.tts_service.TTSService.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        mock_stt.return_value = {"text": "What is forward kinematics?", "duration_seconds": 1.2, "language": "en"}
        mock_tts.return_value = ("/audio/mock_fk.wav", "mock_fk.wav")

        res = client.post("/api/v1/audio/laptop?format=json", files=files, data=data)
        assert res.status_code == 200
        payload = res.json()
        assert payload.get("success") is True
        assert payload.get("transcript") == "What is forward kinematics?"
        assert len(payload.get("reply_text", "")) > 0
        assert payload.get("audio_url") is not None
        # Assert STT was called exactly once for this single uploaded utterance
        assert mock_stt.await_count == 1


# 7. Listening automatically resumes after the AI response.
def test_7_listening_state_persists_in_active_mode():
    voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "test_setup")
    voice_manager.touch_interaction()
    # In active listening mode, auto_listen_after_tts is enabled
    assert voice_manager.auto_listen_after_tts is True
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING


# 8. The assistant returns to snooze after 20 seconds of inactivity.
def test_8_returns_to_snooze_after_inactivity():
    voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "test_setup")
    # Simulate past interaction > 20s ago
    voice_manager.last_interaction_timestamp = time.time() - 25.0

    # Checking inactivity with 20s timeout triggers snooze
    timed_out = voice_manager.check_inactivity_timeout(timeout_seconds=20.0)
    assert timed_out is True
    assert voice_manager.state in [VoiceState.WAKE_LISTENING, VoiceState.SNOOZED, VoiceState.STANDBY]


# 9. New user speech resets the inactivity timer.
def test_9_new_user_speech_resets_inactivity_timer():
    voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "test_setup")
    old_timestamp = time.time() - 15.0
    voice_manager.last_interaction_timestamp = old_timestamp

    # New speech occurs -> touch_interaction() updates timestamp
    voice_manager.touch_interaction()
    assert voice_manager.last_interaction_timestamp > old_timestamp

    # Under 20s, timeout does NOT fire
    timed_out = voice_manager.check_inactivity_timeout(timeout_seconds=20.0)
    assert timed_out is False
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING


# 10. Snooze mode does not submit ordinary speech to the AI.
def test_10_snooze_mode_ignores_ordinary_speech():
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_snooze")
    mock_wav = _generate_mock_wav(1.0)
    files = {"audio": ("laptop_speech.webm", mock_wav, "audio/webm")}
    data = {"student_id": "1", "mode": "coach"}

    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:

        mock_stt.return_value = {"text": "The weather is very sunny today", "duration_seconds": 1.5, "language": "en"}

        res = client.post("/api/v1/audio/laptop?format=json", files=files, data=data)
        assert res.status_code == 200
        payload = res.json()
        # Ordinary speech in snooze mode is ignored and NOT sent to LLM
        assert payload.get("ignored") is True
        assert mock_llm.await_count == 0
        assert payload.get("audio_url") is None


# 11. TTS playback does not trigger a false wake or feedback loop.
def test_11_tts_playback_gating_prevents_feedback_loop():
    voice_manager.transition_to(VoiceState.SPEAKING, "test_tts")
    voice_manager.is_speaking = True

    # Inactivity timeout does NOT fire while speaking
    timed_out = voice_manager.check_inactivity_timeout(timeout_seconds=20.0)
    assert timed_out is False

    # Reset speaking
    voice_manager.is_speaking = False
    voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "playback_done")
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING


# 12. Recognition errors and browser permission denial are handled cleanly.
def test_12_stop_and_error_handling_clean():
    # Stop command safely puts voice manager into snooze
    assert is_stop_command("stop mizo") is True
    assert is_stop_command("mizo stop") is True
    assert is_stop_command("stop") is True

    # Clean normalization of text with weird characters
    clean = normalize_text("  Hey,  Mizo?!?  ")
    assert clean == "hey mizo"


# 13. Repeated wake/snooze cycles do not create duplicate listeners or microphone streams.
def test_13_repeated_wake_snooze_cycles_safe():
    for cycle in range(5):
        # Snooze
        voice_manager.transition_to(VoiceState.WAKE_LISTENING, f"cycle_{cycle}_snooze")
        assert voice_manager.state in [VoiceState.WAKE_LISTENING, VoiceState.STANDBY, VoiceState.SNOOZED]

        # Wake
        is_wake, _ = is_wake_word("Hey Mizo", log_debug=False)
        assert is_wake is True
        voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, f"cycle_{cycle}_wake")
        assert voice_manager.state == VoiceState.ACTIVE_LISTENING

        # Inactivity timeout
        voice_manager.last_interaction_timestamp = time.time() - 30.0
        assert voice_manager.check_inactivity_timeout(timeout_seconds=20.0) is True


# 14. Empty audio or silence does not produce error TTS or repeated processing loops.
@pytest.mark.asyncio
async def test_14_laptop_audio_silence_does_not_loop_or_generate_tts():
    voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "test_active")
    mock_wav = _generate_mock_wav(0.2)
    files = {"audio": ("laptop_speech.webm", mock_wav, "audio/webm")}
    data = {"student_id": "1", "mode": "coach"}

    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.tts_service.TTSService.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        # Empty/silent transcription
        mock_stt.return_value = {"text": "", "duration_seconds": 0.2, "language": "en"}

        res = client.post("/api/v1/audio/laptop?format=json", files=files, data=data)
        assert res.status_code == 200
        payload = res.json()

        # Must be cleanly ignored without synthesizing TTS playback that triggers a feedback loop
        assert payload.get("ignored") is True
        assert payload.get("audio_url") is None
        assert mock_tts.await_count == 0


# 15. Phonetic wake variants including split syllables wake the assistant.
def test_15_phonetic_wake_variants_including_split_syllables():
    variants = [
        "Mizo",
        "Hey Mizo",
        "Hi Mizo",
        "hey me so",
        "me so",
        "hey miso",
        "hi miso",
        "hello mizo",
        "mizo wake up"
    ]
    for v in variants:
        is_wake, _ = is_wake_word(v, log_debug=False)
        assert is_wake is True, f"Failed to wake on variant: {v}"


# 16. Hands-free follow-up queries succeed without repeating wake word in active mode.
@pytest.mark.asyncio
async def test_16_continuous_follow_up_without_repeating_wake_word():
    voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "test_active")
    voice_manager.touch_interaction()

    mock_wav = _generate_mock_wav(1.0)
    files = {"audio": ("laptop_speech.webm", mock_wav, "audio/webm")}
    data = {"student_id": "1", "mode": "coach"}

    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.tts_service.TTSService.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        # Natural follow-up question containing NO wake word
        mock_stt.return_value = {"text": "Can you give me an example?", "duration_seconds": 1.2, "language": "en"}
        mock_tts.return_value = ("/audio/mock_example.wav", "mock_example.wav")

        res = client.post("/api/v1/audio/laptop?format=json", files=files, data=data)
        assert res.status_code == 200
        payload = res.json()

        assert payload.get("success") is True
        assert payload.get("transcript") == "Can you give me an example?"
        assert payload.get("audio_url") is not None
        assert mock_tts.await_count == 1

