import json
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import (
    get_student,
    update_student,
    create_student,
    reset_student_to_fresh,
    restart_onboarding,
    get_db_settings,
    update_db_settings,
    add_conversation_message,
    get_conversation_history,
    save_assessment,
    get_student_assessments,
    get_connection
)
from app.services.personalization_service import personalization_service

client = TestClient(app)


# ---------------------------------------------------------------------------
# Criterion 1: A fresh or restarted onboarding asks for my name first
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_criterion_1_fresh_or_restarted_asks_name_first():
    # Restart onboarding
    restart_res = client.post("/api/v1/students/1/restart-onboarding")
    assert restart_res.status_code == 200

    student = get_student(1)
    assert student["onboarding_completed"] is False
    assert student["onboarding_step"] == "ask_name"

    # Verify onboarding prompt mandates asking name first
    prompt = personalization_service._build_onboarding_prompt(student, "ask_name")
    assert "asking for their name MUST be the first onboarding question" in prompt
    assert "PURPOSE: Welcome the student warmly and ask for their name as your first question." in prompt


# ---------------------------------------------------------------------------
# Criterion 2: Later questions generated from missing info, not fixed list
# ---------------------------------------------------------------------------
def test_criterion_2_questions_generated_from_missing_info():
    # Student has name and goals, but missing weaknesses, hobbies, level
    test_student = {
        "id": 1,
        "name": "Maria",
        "education": "College Student",
        "learning_goals": "Prepare for medical conference presentations",
        "learning_topics": "",
        "interests": "",
        "weaknesses": [],
        "strengths": [],
        "self_reported_level": "Pending",
        "onboarding_completed": False,
        "onboarding_step": "ask_goals"
    }

    missing = personalization_service.get_missing_onboarding_fields(test_student)
    assert "weaknesses or challenges in English" in missing
    assert "self-reported English level (or whether you are not sure)" in missing
    assert "name" not in missing
    assert "English or communication goals" not in missing

    prompt = personalization_service._build_onboarding_prompt(test_student, "ask_goals")
    assert "INFORMATION STILL MISSING TO COVER:" in prompt
    assert "weaknesses or challenges in English" in prompt
    assert "Do NOT use a fixed question sequence or canned questions." in prompt


# ---------------------------------------------------------------------------
# Criterion 3: One answer can fill multiple fields; completed not re-asked
# ---------------------------------------------------------------------------
def test_criterion_3_one_answer_fills_multiple_fields_without_repetition():
    reset_student_to_fresh(1)

    multi_answer = (
        "Hi Mikaza! My name is David. I work as a data scientist, and my goal is to speak fluently in team meetings. "
        "I am interested in machine learning and robotics, but I often hesitate when speaking."
    )

    updated = personalization_service.extract_and_update_student_memory(1, multi_answer)

    assert updated["name"] == "David"
    assert "data scientist" in updated["education"].lower()
    assert "meetings" in updated["learning_goals"].lower()
    assert "machine learning" in (updated["learning_topics"] + updated["interests"]).lower()
    assert len(updated["weaknesses"]) > 0

    # Fields filled should be removed from missing list
    missing = personalization_service.get_missing_onboarding_fields(updated)
    assert "name" not in missing
    assert "English or communication goals" not in missing
    assert "subjects or topics you want to learn" not in missing


# ---------------------------------------------------------------------------
# Criterion 4: Application restart preserves onboarding progress & API settings
# ---------------------------------------------------------------------------
def test_criterion_4_app_restart_preserves_onboarding_and_settings():
    # Save specific admin settings
    update_db_settings({
        "active_provider": "groq",
        "groq_model": "llama-3.3-70b-versatile",
        "system_prompt": "You are Mikaza, custom test persona.",
        "tts_voice": "en-US-GuyNeural"
    })

    # Set student to mid-onboarding step
    update_student(1, {
        "name": "Elena",
        "onboarding_step": "ask_weaknesses",
        "onboarding_completed": False
    })

    # Simulate app reboot: read directly from fresh DB connection
    reloaded_settings = get_db_settings()
    assert reloaded_settings["active_provider"] == "groq"
    assert reloaded_settings["groq_model"] == "llama-3.3-70b-versatile"
    assert reloaded_settings["system_prompt"] == "You are Mikaza, custom test persona."

    reloaded_student = get_student(1)
    assert reloaded_student["name"] == "Elena"
    assert reloaded_student["onboarding_step"] == "ask_weaknesses"
    assert reloaded_student["onboarding_completed"] is False


# ---------------------------------------------------------------------------
# Criterion 5: Restarting onboarding preserves prior history and assessments
# ---------------------------------------------------------------------------
def test_criterion_5_restart_onboarding_preserves_history_and_assessments():
    # Log past conversation message
    add_conversation_message(
        student_id=1,
        session_id="ses_prev_run",
        role="user",
        content="This was from my earlier onboarding run."
    )

    # Save past assessment record
    save_assessment({
        "student_id": 1,
        "session_id": "ses_prev_run",
        "topic": "Previous technology topic",
        "transcript": "I spoke for 60 seconds about my interest in software engineering.",
        "overall_level": "Intermediate",
        "communication_score": 76.0
    })

    # Now trigger explicit restart onboarding
    restart_onboarding(1)

    student = get_student(1)
    assert student["onboarding_step"] == "ask_name"
    assert student["onboarding_completed"] is False

    # History must be preserved
    history = get_conversation_history(student_id=1, limit=50)
    user_contents = [m["content"] for m in history if m["role"] == "user"]
    assert "This was from my earlier onboarding run." in user_contents

    # Assessment records must be preserved
    assessments = get_student_assessments(1)
    topics = [a["topic"] for a in assessments]
    assert "Previous technology topic" in topics


# ---------------------------------------------------------------------------
# Criterion 6: Standby ignores ordinary speech; 'Hey Mikaza' activates
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_criterion_6_wake_phrase_activation_and_prefix_stripping():
    # When user starts speech with 'Hey Mikaza', prefix is removed and remaining words processed
    with patch("app.services.llm_service.LLMService._call_groq", new_callable=AsyncMock) as mock_groq, \
         patch("app.services.tts_service.TTSService.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        mock_groq.return_value = ("Nice to meet you Carlos! What are your goals?", "llama-3.3-70b-versatile")
        mock_tts.return_value = ("/api/v1/esp32/audio/cache/test.wav", "data/audio_cache/test.wav")

        res = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "Hey Mikaza, my name is Carlos"
        })
        assert res.status_code == 200
        st = get_student(1)
        assert st["name"] == "Carlos"


# ---------------------------------------------------------------------------
# Criterion 7: Turn silence timer is 3.0s and configurable
# ---------------------------------------------------------------------------
def test_criterion_7_turn_silence_timer_configuration():
    # In app.js: turnEndingSilenceSeconds = 3.0
    # In index.html: input #setting-turn-silence has min=1, max=10, step=0.5, value=3.0
    with open("app/static/js/app.js", "r", encoding="utf-8") as f:
        js_content = f.read()
    assert "let turnEndingSilenceSeconds = 3.0;" in js_content
    assert "turnEndingSilenceSeconds" in js_content

    with open("app/static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    assert 'id="setting-turn-silence"' in html_content
    assert 'value="3.0"' in html_content


# ---------------------------------------------------------------------------
# Criterion 8: Inactivity timeout is 12s (configurable between 10-15s)
# ---------------------------------------------------------------------------
def test_criterion_8_inactivity_timeout_configuration():
    with open("app/static/js/app.js", "r", encoding="utf-8") as f:
        js_content = f.read()
    assert ("let inactivityTimeoutSeconds = 10.0;" in js_content or "let inactivityTimeoutSeconds = 12.0;" in js_content)
    assert "startInactivityCountdown()" in js_content
    assert "inactivityTimeoutSeconds * 1000" in js_content

    with open("app/static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    assert 'id="setting-inactivity-timeout"' in html_content
    assert 'value="12"' in html_content


# ---------------------------------------------------------------------------
# Criterion 9: 'Mikaza stop' halts playback/processing & returns to WAKE_ONLY
# ---------------------------------------------------------------------------
def test_criterion_9_mikaza_stop_command():
    res = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Mikaza stop"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["is_control_command"] is True
    assert data["command"] == "stop"
    assert data["voice_state"] in ["STANDBY", "WAKE_ONLY"]
    assert data["reply_text"] == ""
    assert data["audio_url"] is None


# ---------------------------------------------------------------------------
# Criterion 10: Failed or too-short audio does not create fake transcripts/scores
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_criterion_10_too_short_audio_rejects_without_fake_baseline():
    update_student(1, {
        "onboarding_step": "speech_test_prompt",
        "onboarding_completed": False,
        "speaking_assessment_completed": False,
        "baseline_grammar": 0.0
    })

    with patch("app.services.tts_service.TTSService.synthesize_to_file", new_callable=AsyncMock) as mock_tts:
        mock_tts.return_value = ("/api/v1/esp32/audio/cache/retry.wav", "data/audio_cache/retry.wav")

        # User only says "yes hello" (2 words, far below ~60s requirement)
        res = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "yes hello"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["onboarding_step"] == "speech_test_prompt"
        assert "too short" in data["reply_text"].lower()

        # Confirm no baseline was recorded
        st = get_student(1)
        assert st["onboarding_completed"] is False
        assert st["speaking_assessment_completed"] is False
        assert st["baseline_grammar"] == 0.0


# ---------------------------------------------------------------------------
# Criterion 11: Speaking assessment saved as baseline; later sessions don't overwrite it
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_criterion_11_baseline_saved_and_immutable_in_later_sessions():
    update_student(1, {
        "onboarding_step": "speech_test_prompt",
        "onboarding_completed": False,
        "speaking_assessment_completed": False
    })

    valid_transcript = (
        "I am very excited about studying machine learning and computer vision. "
        "In my free time, I like building small robots and working on automation projects. "
        "My goal is to be able to present technical topics in English clearly without searching for words."
    )

    with patch("app.services.llm_service.LLMService._call_groq", new_callable=AsyncMock) as mock_groq, \
         patch("app.services.tts_service.TTSService.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        # Mock structured LLM assessment JSON
        mock_groq.return_value = (json.dumps({
            "grammar_score": 82.0,
            "vocabulary_score": 80.0,
            "fluency_score": 78.0,
            "pronunciation_score": 75.0,
            "confidence_score": 76.0,
            "communication_score": 80.0,
            "overall_level": "Intermediate",
            "grammar_feedback": "Accurate sentence construction.",
            "vocabulary_feedback": "Technical domain vocabulary well utilized.",
            "fluency_feedback": "Good delivery pace.",
            "pronunciation_feedback": "Clear intelligible transcript.",
            "confidence_feedback": "Observable delivery flow.",
            "communication_feedback": "Effectively explained goals and interests.",
            "strengths": ["Domain vocabulary", "Clear intent"],
            "weaknesses": ["Minor preposition usage"],
            "mistakes": [],
            "spoken_summary": "Great job on your assessment! Your communication baseline is 80%."
        }), "llama-3.3-70b-versatile")
        mock_tts.return_value = ("/api/v1/esp32/audio/cache/summary.wav", "data/audio_cache/summary.wav")

        res = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": valid_transcript
        })
        assert res.status_code == 200

        # Verify baseline established
        st = get_student(1)
        assert st["onboarding_completed"] is True
        assert st["speaking_assessment_completed"] is True
        assert st["baseline_grammar"] == 82.0
        assert st["baseline_vocabulary"] == 80.0
        assert st["baseline_fluency"] == 78.0
        baseline_gram = st["baseline_grammar"]

        # Later session interaction: should update current scores without touching baseline
        mock_groq.return_value = ("That is an interesting thought! Let's explore that further.", "llama-3.3-70b-versatile")
        res_later = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "Today I want to practice vocabulary for job interviews."
        })
        assert res_later.status_code == 200

        st_later = get_student(1)
        assert st_later["baseline_grammar"] == baseline_gram
        assert st_later["speaking_assessment_completed"] is True
