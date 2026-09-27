import pytest
import io
import wave
import struct
import json
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import (
    get_student,
    create_student,
    update_student,
    reset_student_to_fresh,
    create_session,
    get_session_by_id,
    get_student_assessments,
    get_recent_mistakes,
    get_conversation_history,
    handle_device_start_or_reset
)
from app.services.llm_service import LLMResult
from app.services.personalization_service import personalization_service

client = TestClient(app)


def test_esp32_reset_and_start_behavior():
    """
    1. Reset button behavior:
       - Normal reset NEVER deletes historical student data.
       - Starts a new interaction/session.
       - If student already onboarded, recognizes returning user and retrieves stored profile.
    """
    # Create student with completed profile
    reset_student_to_fresh(1)
    update_student(1, {
        "name": "Shahid",
        "education": "Final-year Robotics Student",
        "learning_goals": "Improve English communication during presentations",
        "weaknesses": ["Grammar", "Fluency"],
        "strengths": ["Technical robotics knowledge", "Good pronunciation"],
        "baseline_grammar": 65.0,
        "baseline_fluency": 60.0,
        "baseline_vocabulary": 70.0,
        "baseline_pronunciation": 75.0,
        "baseline_confidence": 62.0,
        "baseline_communication": 68.0,
        "grammar_score": 65.0,
        "fluency_score": 60.0,
        "vocabulary_score": 70.0,
        "onboarding_completed": True,
        "onboarding_step": "completed"
    })

    # Trigger ESP32 normal start / reset interaction
    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm, \
         patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        mock_llm.return_value = LLMResult(
            text="Welcome back, Shahid! I remember that we're working on your grammar and presentation fluency. Let's continue.",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False
        )
        mock_tts.return_value = ("/api/v1/esp32/audio/cache/welcome.wav", "data/audio_cache/welcome.wav")

        res = client.post("/api/v1/esp32/start", json={"student_id": 1, "action": "start"})
        assert res.status_code == 200
        data = res.json()

        assert data["is_returning_user"] is True
        assert data["student_name"] == "Shahid"
        assert "Welcome back, Shahid" in data["reply_text"]
        assert data["onboarding_active"] is False

        # Verify historical data in database remained 100% intact
        student = get_student(1)
        assert student["name"] == "Shahid"
        assert student["baseline_grammar"] == 65.0
        assert student["baseline_fluency"] == 60.0
        assert "Grammar" in student["weaknesses"]

        # Verify new session was created with 0 session metrics
        session = get_session_by_id(data["session_id"])
        assert session is not None
        assert session["metrics"]["grammar_accuracy"] == 0.0
        assert session["metrics"]["fluency_score"] == 0.0
        assert session["metrics"]["message_count"] == 0


def test_multi_information_extraction_in_single_utterance():
    """
    Extracts name, education, weakness, and goals all from a single natural answer:
    'I'm Shahid, a final-year robotics student and I struggle with speaking English during presentations.'
    """
    reset_student_to_fresh(1)
    text = "I'm Shahid, a final-year robotics student and I struggle with speaking English during presentations."

    extracted = personalization_service.extract_student_info(text)
    assert extracted["name"] == "Shahid"
    assert "robotics student" in extracted["education"].lower()
    assert any("presentation" in w.lower() or "speaking" in w.lower() for w in extracted["weaknesses"])

    # Update student memory directly
    updated = personalization_service.extract_and_update_student_memory(1, text)
    assert updated["name"] == "Shahid"
    assert "robotics" in updated["education"].lower()


@pytest.mark.asyncio
async def test_complete_onboarding_and_speaking_assessment_pipeline():
    """
    Full Acceptance Test:
    1. Reset student to fresh (Factory reset).
    2. Start onboarding from ask_name.
    3. User provides multi-info: name & background.
    4. State machine progresses without asking already-known facts.
    5. User provides speaking assessment sample.
    6. Assessment Engine evaluates 6 dimensions, evidence, mistakes.
    7. Baseline established and saved to database.
    8. History, mistakes, and assessment records retrievable from SQLite.
    """
    # 1. Fresh start
    reset_res = client.post("/api/v1/students/1/reset")
    assert reset_res.status_code == 200

    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm, \
         patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        mock_tts.return_value = ("/api/v1/esp32/audio/cache/reply.wav", "data/audio_cache/reply.wav")

        # Step 1: User gives name & background in one natural sentence
        mock_llm.return_value = LLMResult(
            text="Nice to meet you, Shahid! Since you're studying robotics, what are your main goals and challenges when speaking English?",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False
        )
        res1 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "My name is Shahid, and I am a final-year robotics student."
        })
        assert res1.status_code == 200
        st1 = get_student(1)
        assert st1["name"] == "Shahid"
        assert "robotics" in st1["education"].lower()

        # Step 2: User states goals & weaknesses
        mock_llm.return_value = LLMResult(
            text="Understood, Shahid! Now I'd like to evaluate your baseline speaking level. Tell me about a robotics project you've worked on.",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False
        )
        res2 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "I struggle with speaking during technical presentations, and my goal is to speak fluently and confidently."
        })
        assert res2.status_code == 200
        st2 = get_student(1)
        assert st2["onboarding_step"] in ["ask_weaknesses", "speech_test_prompt"]

        # Advance to speech assessment prompt
        update_student(1, {"onboarding_step": "speech_test_prompt"})

        # Step 3: Speaking Assessment Execution (1-2 minute continuous speech sample)
        assessment_analysis_json = {
            "grammar_score": 62.0,
            "vocabulary_score": 68.0,
            "fluency_score": 58.0,
            "pronunciation_score": 72.0,
            "confidence_score": 60.0,
            "communication_score": 65.0,
            "overall_level": "Elementary",
            "grammar_feedback": "Subject-verb agreement issues (he go instead of he goes). Missing articles.",
            "vocabulary_feedback": "Good technical terminology; opportunities for descriptive adjectives.",
            "fluency_feedback": "Pauses between technical descriptions and frequent filler words.",
            "pronunciation_feedback": "Intelligible delivery of technical terms.",
            "confidence_feedback": "Hesitant delivery when formulating complex sentences.",
            "communication_feedback": "Effectively explained the robot's functionality and challenges.",
            "strengths": ["Solid technical domain vocabulary", "Clear problem explanation"],
            "weaknesses": ["Subject-verb agreement", "Filler words", "Sentence linking"],
            "mistakes": [
                {
                    "utterance": "He go to college and he don't know the answer.",
                    "error_text": "He go",
                    "correction": "He goes",
                    "mistake_type": "grammar",
                    "explanation": "Third-person singular subjects require 'goes'."
                }
            ],
            "spoken_summary": "Great job on your assessment, Shahid! Your communication score is 65% and your starting level is Elementary. We will focus on grammar agreement and smooth presentation delivery."
        }

        mock_llm.return_value = LLMResult(
            text=json.dumps(assessment_analysis_json),
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False
        )

        speech_transcript = "Last semester I build an autonomous obstacle avoiding robot using Arduino and ultrasonic sensors. He go to college and he don't know the answer."
        res3 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": speech_transcript
        })
        assert res3.status_code == 200
        data3 = res3.json()

        # Verify onboarding is complete
        assert data3["onboarding_active"] is False
        assert data3["onboarding_step"] == "completed"

        # Verify baseline scores saved to SQLite
        st3 = get_student(1)
        assert st3["onboarding_completed"] is True
        assert st3["baseline_grammar"] == 62.0
        assert st3["baseline_vocabulary"] == 68.0
        assert st3["baseline_fluency"] == 58.0
        assert st3["baseline_pronunciation"] == 72.0
        assert st3["baseline_confidence"] == 60.0
        assert st3["baseline_communication"] == 65.0
        assert st3["target_level"] == "Elementary"

        # Verify assessment record in assessments table
        assessments = get_student_assessments(1)
        assert len(assessments) >= 1
        ass = assessments[0]
        assert ass["grammar_score"] == 62.0
        assert "Subject-verb agreement" in ass["grammar_feedback"]
        assert len(ass["strengths"]) > 0

        # Verify mistake logged in student_mistakes table
        mistakes = get_recent_mistakes(1)
        assert len(mistakes) >= 1
        assert any("He go" in m["error_text"] or "goes" in m["correction"] for m in mistakes)

        # Verify conversation history logged
        history = get_conversation_history(student_id=1)
        assert len(history) >= 2


def test_baseline_preserved_when_new_session_starts():
    """
    Verifies that starting a new session initializes session metrics to 0
    without modifying the permanent baseline or historical record.
    """
    student = get_student(1)
    orig_baseline_grammar = student["baseline_grammar"]
    orig_baseline_fluency = student["baseline_fluency"]

    # Start brand new session
    res = client.post("/api/v1/students/1/sessions", json={"student_id": 1, "mode": "coach"})
    assert res.status_code == 200
    session_data = res.json()

    # Session metrics start at 0
    assert session_data["metrics"]["grammar_accuracy"] == 0.0
    assert session_data["metrics"]["fluency_score"] == 0.0
    assert session_data["metrics"]["message_count"] == 0

    # Permanent baseline remains intact
    refreshed_student = get_student(1)
    assert refreshed_student["baseline_grammar"] == orig_baseline_grammar
    assert refreshed_student["baseline_fluency"] == orig_baseline_fluency
