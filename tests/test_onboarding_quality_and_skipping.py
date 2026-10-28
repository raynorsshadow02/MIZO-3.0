import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import (
    get_student,
    update_student,
    get_prototype_student,
    PROTOTYPE_STUDENT_ID
)
from app.services.voice_service import (
    is_assessment_skip_command,
    is_unclear_or_low_confidence_speech,
    classify_command_or_intent
)
from app.services.personalization_service import personalization_service

client = TestClient(app)


def test_assessment_skip_phrase_detection():
    """Verify skip phrases are deterministically recognized."""
    assert is_assessment_skip_command("skip the assessment") is True
    assert is_assessment_skip_command("let's talk normally") is True
    assert is_assessment_skip_command("we can talk about it") is True
    assert is_assessment_skip_command("not now") is True
    assert is_assessment_skip_command("can we just talk") is True
    assert is_assessment_skip_command("no assessment") is True
    assert is_assessment_skip_command("skip") is True
    assert classify_command_or_intent("skip the assessment") == "SKIP_ASSESSMENT"

    # Regular phrases are not skip commands
    assert is_assessment_skip_command("I am ready for the test") is False
    assert is_assessment_skip_command("My name is Alex") is False


def test_unreliable_speech_detection_and_safeguards():
    """Verify inaudible/hallucinated speech is flagged while valid short answers are preserved."""
    # Valid short answers must NOT be flagged as unclear
    assert is_unclear_or_low_confidence_speech("yes")[0] is False
    assert is_unclear_or_low_confidence_speech("no")[0] is False
    assert is_unclear_or_low_confidence_speech("stop")[0] is False
    assert is_unclear_or_low_confidence_speech("ok")[0] is False
    assert is_unclear_or_low_confidence_speech("beginner")[0] is False
    assert is_unclear_or_low_confidence_speech("Shahid")[0] is False
    assert is_unclear_or_low_confidence_speech("Hey Mizo")[0] is False

    # Whisper silence hallucinations must be flagged
    assert is_unclear_or_low_confidence_speech("Thank you for watching.")[0] is True
    assert is_unclear_or_low_confidence_speech("Takk for at du så på")[0] is True
    assert is_unclear_or_low_confidence_speech("Terima kasih telah menonton")[0] is True

    # Punctuation only or empty
    assert is_unclear_or_low_confidence_speech("...")[0] is True
    assert is_unclear_or_low_confidence_speech("")[0] is True

    # Foreign language metadata flag
    unreliable, reason = is_unclear_or_low_confidence_speech(
        "noe rart",
        stt_metadata={"language": "no"}
    )
    assert unreliable is True


def test_assessment_skip_via_chat_endpoint():
    """Verify that saying 'skip the assessment' defers assessment and exits onboarding into normal conversation."""
    update_student(PROTOTYPE_STUDENT_ID, {
        "name": "Jordan",
        "onboarding_completed": False,
        "onboarding_step": "speech_test_prompt",
        "assessment_deferred": 0
    })

    with patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:
        mock_tts.return_value = ("/api/v1/esp32/audio/cache/skip.wav", "data/audio_cache/skip.wav")

        res = client.post("/api/v1/esp32/chat", json={
            "student_id": PROTOTYPE_STUDENT_ID,
            "message": "can we skip the assessment and let's talk normally"
        })

        assert res.status_code == 200
        data = res.json()
        assert data["onboarding_active"] is False
        assert data["onboarding_step"] == "completed"
        assert "skip the assessment" in data["reply_text"].lower() or "talk normally" in data["reply_text"].lower()

        # Verify DB state
        st = get_student(PROTOTYPE_STUDENT_ID)
        assert st["onboarding_completed"] is True
        assert st["onboarding_step"] == "completed"
        assert st["assessment_deferred"] == 1


def test_we_can_talk_about_it_does_not_restart_assessment():
    """Verify 'we can talk about it' is treated as normal conversation, not restarting assessment."""
    update_student(PROTOTYPE_STUDENT_ID, {
        "name": "Jordan",
        "onboarding_completed": False,
        "onboarding_step": "speech_test_prompt",
        "assessment_deferred": 0
    })

    with patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:
        mock_tts.return_value = ("/api/v1/esp32/audio/cache/skip.wav", "data/audio_cache/skip.wav")

        res = client.post("/api/v1/esp32/chat", json={
            "student_id": PROTOTYPE_STUDENT_ID,
            "message": "we can talk about it"
        })

        assert res.status_code == 200
        st = get_student(PROTOTYPE_STUDENT_ID)
        assert st["assessment_deferred"] == 1
        assert st["onboarding_completed"] is True
        assert st["onboarding_step"] == "completed"


def test_unclear_speech_returns_polite_repeat_prompt():
    """Verify unclear speech in audio upload triggers 'Sorry, I didn't catch that. Could you repeat it?'."""
    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        # Whisper hallucinates Norwegian closing phrase from mic noise
        mock_stt.return_value = {
            "text": "Takk for at du så på.",
            "duration_seconds": 1.2,
            "language": "no",
            "segments": [],
            "words": []
        }
        mock_tts.return_value = ("/api/v1/esp32/audio/cache/repeat.wav", "data/audio_cache/repeat.wav")

        fake_wav = b"RIFF" + b"\x00" * 40
        res = client.post(
            "/api/v1/esp32/audio?format=json&test_mode=true",
            files={"audio": ("audio.wav", fake_wav, "audio/wav")},
            data={"student_id": "1"}
        )

        assert res.status_code == 200
        data = res.json()
        assert data["reply_text"] == "Sorry, I didn't catch that. Could you repeat it?"


def test_empathy_and_context_in_onboarding_turn_prompt():
    """Verify build_onboarding_turn_prompt includes stage, profile, history, and empathy directives."""
    st = {
        "name": "Maria",
        "education": "Computer Science student",
        "learning_goals": "Improve fluency for job interviews",
        "strengths": ["Reading", "Vocabulary"],
        "weaknesses": ["Hesitation when speaking", "Lack of confidence"],
        "self_reported_level": "Intermediate",
        "assessment_deferred": 0,
        "onboarding_completed": False
    }

    history = [
        {"role": "assistant", "content": "Tell me a little about yourself—what are your main goals?"},
        {"role": "user", "content": "I am studying CS and want to prepare for interviews."}
    ]

    prompt = personalization_service.build_onboarding_turn_prompt(
        student=st,
        current_step="ask_weaknesses",
        next_step="ask_self_level",
        user_answer="I lack confidence and often hesitate when speaking.",
        recent_history=history,
        assessment_status="pending"
    )

    assert "CURRENT ONBOARDING STAGE: ask_weaknesses" in prompt
    assert "NEXT ONBOARDING STAGE: ask_self_level" in prompt
    assert "Maria" in prompt
    assert "Computer Science student" in prompt
    assert "Improve fluency for job interviews" in prompt
    assert "RECENT CONVERSATION TURNS" in prompt
    assert "EMPATHY FOR NERVOUSNESS & CONFIDENCE" in prompt
    assert "ANSWER THE STUDENT'S ACTUAL MESSAGE FIRST" in prompt
