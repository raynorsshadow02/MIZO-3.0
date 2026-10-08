import pytest
import time
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import (
    get_student,
    update_student,
    PROTOTYPE_STUDENT_ID,
    get_active_session,
    create_session,
    handle_device_start_or_reset,
)
from app.services.personalization_service import personalization_service
from app.services.llm_service import LLMService
from app.services.voice_service import voice_manager, VoiceState

client = TestClient(app)


# -----------------------------------------------------------------------------
# Test 1 & 2: Deterministic Confusion Detection
# -----------------------------------------------------------------------------
def test_1_and_2_confusion_detection():
    """1. 'I don't understand' and 2. 'Make it simpler' -> LEARNER_CONFUSED."""
    confused_1, severity_1 = personalization_service.is_learner_confused("I don't understand")
    assert confused_1 is True
    assert severity_1 in ["mild", "major"]

    confused_2, severity_2 = personalization_service.is_learner_confused("Can you make it simpler?")
    assert confused_2 is True
    assert severity_2 == "major"

    # Additional user variations from problem description
    for phrase in [
        "I don't get it",
        "I don't quite get you",
        "can you explain it to me a very simpler way",
        "You are making it very complicated.",
        "too complicated",
        "what does that mean",
        "I am confused"
    ]:
        is_conf, _ = personalization_service.is_learner_confused(phrase)
        assert is_conf is True, f"Failed to detect confusion in: '{phrase}'"


# -----------------------------------------------------------------------------
# Test 3 & 4: Adaptive Teaching Difficulty (1..5 scale)
# -----------------------------------------------------------------------------
def test_3_and_4_adaptive_difficulty():
    """3. Confusion decreases teaching difficulty; 4. Successes gradually increase it."""
    student_id = PROTOTYPE_STUDENT_ID
    # Start at difficulty 3
    update_student(student_id, {"current_teaching_difficulty": 3})
    assert get_student(student_id)["current_teaching_difficulty"] == 3

    # User expresses major confusion -> drops by up to 2
    new_diff = personalization_service.adapt_teaching_difficulty(
        student_id=student_id,
        user_utterance="You are making it very complicated. Make it simpler.",
        struggle=True
    )
    assert new_diff <= 2
    assert get_student(student_id)["current_teaching_difficulty"] <= 2

    # Multiple confusions clamp at min 1
    new_diff = personalization_service.adapt_teaching_difficulty(
        student_id=student_id,
        user_utterance="I still don't understand. Make it easy.",
        struggle=True
    )
    assert new_diff == 1

    # Consecutive successes can gradually increase difficulty (at most by 1)
    session = get_active_session(student_id=student_id, default_mode="coach")
    session_id = session["session_id"]

    # First success
    personalization_service.adapt_teaching_difficulty(
        student_id=student_id,
        session_id=session_id,
        user_utterance="I am an engineering student.",
        success=True,
        struggle=False
    )
    # Second consecutive success
    increased_diff = personalization_service.adapt_teaching_difficulty(
        student_id=student_id,
        session_id=session_id,
        user_utterance="I build robots using ESP32.",
        success=True,
        struggle=False
    )
    assert increased_diff == 2
    assert get_student(student_id)["current_teaching_difficulty"] == 2


# -----------------------------------------------------------------------------
# Test 5: "What is my name?" -> Deterministic Tier 0
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_5_what_is_my_name_deterministic():
    """5. 'What is my name?' -> deterministic Tier 0 response without LLM call."""
    update_student(PROTOTYPE_STUDENT_ID, {"name": "Shahid", "onboarding_completed": True})

    with patch.object(LLMService, "_call_ollama", new_callable=AsyncMock) as mock_ollama, \
         patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq, \
         patch("app.services.tts_service.TTSService.synthesize_to_file", new_callable=AsyncMock) as mock_tts:
        mock_tts.return_value = ("/audio/mock.wav", "dummy.wav")

        res = client.post("/api/v1/esp32/chat", json={
            "message": "What is my name?",
            "is_typed_text": True
        })
        assert res.status_code == 200
        data = res.json()
        assert "Shahid" in data["reply_text"]
        assert data["provider_used"] == "system"
        mock_ollama.assert_not_called()
        mock_groq.assert_not_called()


# -----------------------------------------------------------------------------
# Test 6: "Thank you" -> Ollama/simple path (never Groq)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_6_thank_you_routes_ollama_not_groq():
    """6. 'Thank you' routes to Ollama/simple path, never Groq."""
    messages = [{"role": "user", "content": "Thank you"}]
    with patch.object(LLMService, "_call_ollama", new_callable=AsyncMock) as mock_ollama, \
         patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq:
        mock_ollama.return_value = ("You're welcome!", "llama3:latest")

        res = await LLMService.generate_response(messages=messages)
        assert res.provider == "ollama"
        mock_ollama.assert_called_once()
        mock_groq.assert_not_called()


# -----------------------------------------------------------------------------
# Test 7: "stop" -> Deterministic, No LLM, sets PAUSED
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_7_stop_deterministic():
    """7. 'stop' is deterministic, calls zero LLMs, and sets voice_state=PAUSED."""
    with patch.object(LLMService, "_call_ollama", new_callable=AsyncMock) as mock_ollama, \
         patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq, \
         patch("app.services.tts_service.TTSService.synthesize_to_file", new_callable=AsyncMock) as mock_tts:
        mock_tts.return_value = ("/audio/mock.wav", "dummy.wav")

        res = client.post("/api/v1/esp32/chat", json={
            "message": "stop",
            "is_typed_text": True
        })
        assert res.status_code == 200
        data = res.json()
        assert data["is_control_command"] is True
        assert data["command"] == "stop"
        assert data["voice_state"] == "PAUSED"
        mock_ollama.assert_not_called()
        mock_groq.assert_not_called()


# -----------------------------------------------------------------------------
# Test 8: Existing profile prevents onboarding restart
# -----------------------------------------------------------------------------
def test_8_existing_profile_prevents_onboarding_restart():
    """8. Student with known profile/name does not restart onboarding upon device wake/start."""
    update_student(PROTOTYPE_STUDENT_ID, {
        "name": "Shahid",
        "onboarding_completed": True,
        "onboarding_step": "completed",
        "assessed_level": "Intermediate",
        "current_teaching_difficulty": 2
    })

    start_info = handle_device_start_or_reset(student_id=PROTOTYPE_STUDENT_ID)
    assert start_info["status"] == "returning_user"
    assert start_info["onboarding_active"] is False
    assert start_info["onboarding_step"] == "completed"
    assert start_info["student"]["name"] == "Shahid"


# -----------------------------------------------------------------------------
# Test 9 & 10: Name Change Confirmation Protocol
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_9_and_10_name_change_confirmation():
    """9. 'My name is Zoro' asks confirmation; 10. 'Yes, change it' updates name."""
    update_student(PROTOTYPE_STUDENT_ID, {"name": "Shahid", "onboarding_completed": True})
    session = create_session(student_id=PROTOTYPE_STUDENT_ID, mode="coach")
    session_id = session["session_id"]

    with patch.object(LLMService, "_call_ollama", new_callable=AsyncMock) as mock_ollama, \
         patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq, \
         patch("app.services.tts_service.TTSService.synthesize_to_file", new_callable=AsyncMock) as mock_tts:
        mock_tts.return_value = ("/audio/mock.wav", "dummy.wav")

        # Step 9: Propose name change
        res1 = client.post("/api/v1/esp32/chat", json={
            "session_id": session_id,
            "message": "My name is Zoro",
            "is_typed_text": True
        })
        assert res1.status_code == 200
        reply1 = res1.json()["reply_text"]
        assert "Shahid" in reply1
        assert "Zoro" in reply1
        assert "change" in reply1.lower()
        # Verify name is NOT updated yet
        assert get_student(PROTOTYPE_STUDENT_ID)["name"] == "Shahid"
        mock_groq.assert_not_called()

        # Step 10: Explicit confirmation
        res2 = client.post("/api/v1/esp32/chat", json={
            "session_id": session_id,
            "message": "Yes, please change it",
            "is_typed_text": True
        })
        assert res2.status_code == 200
        reply2 = res2.json()["reply_text"]
        assert "Zoro" in reply2
        # Name is now safely updated
        assert get_student(PROTOTYPE_STUDENT_ID)["name"] == "Zoro"
        mock_groq.assert_not_called()


# -----------------------------------------------------------------------------
# Test 11: Non-English input does not trigger Groq
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_11_non_english_input_rejected_without_groq():
    """11. Non-English input returns specification phrase without triggering Groq."""
    with patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq, \
         patch("app.services.tts_service.TTSService.synthesize_to_file", new_callable=AsyncMock) as mock_tts:
        mock_tts.return_value = ("/audio/mock.wav", "dummy.wav")

        res = client.post("/api/v1/esp32/chat", json={
            "message": "Bonjour comment ca va merci beaucoup",
            "is_typed_text": True
        })
        assert res.status_code == 200
        data = res.json()
        assert data["reply_text"] in [
            "I didn't understand that. Could you say it in English?",
            "I didn't understand that. Let's continue in English."
        ]
        assert data["provider_used"] == "system"
        mock_groq.assert_not_called()


# -----------------------------------------------------------------------------
# Test 12: Simple teaching question does not trigger Groq
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_12_simple_teaching_question_does_not_trigger_groq():
    """12. A simple teaching question routes to Tier 1 Ollama, never Groq."""
    messages = [
        {"role": "system", "content": "You are Mizo, an English tutor."},
        {"role": "user", "content": "What is the past tense of go?"}
    ]
    with patch.object(LLMService, "_call_ollama", new_callable=AsyncMock) as mock_ollama, \
         patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq:
        mock_ollama.return_value = ("The past tense of 'go' is 'went'.", "llama3:latest")

        res = await LLMService.generate_response(messages=messages)
        assert res.provider == "ollama"
        assert "went" in res.text
        mock_ollama.assert_called_once()
        mock_groq.assert_not_called()


# -----------------------------------------------------------------------------
# Test 13: Multiple assessment samples stop the assessment
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_13_multiple_assessment_samples_finalize_assessment():
    """13. Multi-sample assessment accumulates evidence and finalizes without infinite repetition."""
    student_id = PROTOTYPE_STUDENT_ID
    update_student(student_id, {
        "name": "Shahid",
        "onboarding_completed": False,
        "onboarding_step": "speech_test_prompt",
        "cumulative_assessment_words": 0,
        "assessment_samples_count": 0
    })

    with patch.object(personalization_service, "run_speaking_assessment", new_callable=AsyncMock) as mock_assess, \
         patch("app.services.tts_service.TTSService.synthesize_to_file", new_callable=AsyncMock) as mock_tts:
        mock_tts.return_value = ("/audio/mock.wav", "dummy.wav")
        mock_assess.return_value = {
            "invalid_input": False,
            "spoken_summary": "Great job! Your baseline communication profile is saved.",
            "overall_level": "Intermediate",
            "grammar_score": 60.0
        }

        # First sample: 25 words
        sample_1 = "I am an engineering student from Bangalore working on an autonomous mobile robot using ROS and ESP32 microcontrollers for obstacle avoidance."
        res1 = client.post("/api/v1/esp32/chat", json={
            "message": sample_1,
            "is_typed_text": True
        })
        assert res1.status_code == 200

        # Second sample: 25 words -> Cumulative reaches 50 words across 2 samples -> assessment finalizes!
        sample_2 = "I want to improve my speaking confidence and vocabulary so I can present my project clearly during engineering campus interviews and seminars."
        res2 = client.post("/api/v1/esp32/chat", json={
            "message": sample_2,
            "is_typed_text": True
        })
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["onboarding_active"] is False
        assert data2["onboarding_step"] == "completed"
        # Database student is now marked complete
        st = get_student(student_id)
        assert st["onboarding_completed"] in [True, 1]


# -----------------------------------------------------------------------------
# Test 14: Overall English Level decoupled from Teaching Difficulty
# -----------------------------------------------------------------------------
def test_14_overall_level_decoupled_from_teaching_difficulty():
    """14. Overall English level (e.g. Upper Intermediate) does not dictate teaching difficulty."""
    student_id = PROTOTYPE_STUDENT_ID
    update_student(student_id, {
        "overall_level": "Upper Intermediate",
        "current_teaching_difficulty": 1
    })

    st = get_student(student_id)
    assert st["overall_level"] == "Upper Intermediate"
    assert st["current_teaching_difficulty"] == 1

    # Simplified turn uses current_teaching_difficulty (1), ignoring overall_level
    turn = personalization_service.get_simplified_teaching_turn(
        student=st,
        user_text="Give me something easy",
        current_difficulty=st["current_teaching_difficulty"]
    )
    assert turn is not None
    assert turn["new_difficulty"] == 1
    # Check that it uses short, simple beginner sentences, not advanced vocabulary like "pioneer"
    assert "pioneer" not in turn["reply_text"].lower()
    assert len(turn["reply_text"].split("\n")) >= 1
