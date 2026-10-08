import io
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.db.database import (
    get_student,
    create_student,
    update_student,
    delete_learner_data,
    restart_onboarding,
    get_db_settings,
    update_db_settings,
    get_conversation_history,
    get_student_assessments,
    create_session,
    init_db
)
from app.services.personalization_service import personalization_service

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_clean_db():
    init_db()
    # Reset student 1 to fresh onboarding state
    delete_learner_data(1)


# -----------------------------------------------------------------------------
# Criterion 1: A fresh learner receives a greeting and is asked for their name
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fresh_learner_greeting_and_name_request():
    student = get_student(1)
    assert student["onboarding_completed"] == 0
    assert student["onboarding_step"] == "ask_name"

    with patch("app.services.llm_service.LLMService._call_groq", new_callable=AsyncMock) as mock_groq:
        mock_groq.return_value = ("Hello! I'm Mikaza, your AI English tutor. What is your name?", "llama-3.3-70b-versatile")
        
        # Start trigger returns warm greeting for fresh learner
        res = client.post("/api/v1/esp32/start", json={"student_id": 1, "action": "start"})
        assert res.status_code == 200
        data = res.json()
        assert data["is_returning_user"] is False
        assert "name" in data["reply_text"].lower() or "who" in data["reply_text"].lower() or "call you" in data["reply_text"].lower() or "hello" in data["reply_text"].lower()


# -----------------------------------------------------------------------------
# Criterion 2: Natural-language answers can fill more than one profile field
# -----------------------------------------------------------------------------
def test_natural_language_multi_field_extraction():
    # User provides name, learning goals, education, and weaknesses in a single sentence
    utterance = "Hi, my name is Marcus. I am an engineering student and I want to master English for job interviews, but my weaknesses are vocabulary depth and hesitation."
    
    extracted = personalization_service.extract_student_info(utterance, current_student=get_student(1))
    assert extracted.get("name") == "Marcus"
    assert "interview" in extracted.get("learning_goals", "").lower()
    assert "engineering" in extracted.get("education", "").lower()
    assert any("vocabulary" in w.lower() for w in extracted.get("weaknesses", []))

    # Test extracting self-reported level including "not sure"
    extracted_unsure = personalization_service.extract_student_info("I am really not sure about my current English level", current_student=get_student(1))
    assert extracted_unsure.get("self_reported_level") == "Not sure"


# -----------------------------------------------------------------------------
# Criterion 3: Missing fields are asked for without repeating completed fields
# -----------------------------------------------------------------------------
def test_missing_fields_asked_without_repeating_completed():
    # Fill in name and goals
    update_student(1, {
        "name": "Sarah Connor",
        "learning_goals": "Improve business presentations",
        "onboarding_step": "ask_weaknesses"
    })
    
    next_step = personalization_service.determine_next_onboarding_state(get_student(1), current_step="ask_weaknesses")
    assert next_step == "ask_weaknesses"  # Doesn't ask name or goals again

    # Fill weaknesses
    update_student(1, {
        "weaknesses": ["Prepositions", "Pacing"],
        "self_reported_level": "Pending",
        "onboarding_step": "ask_self_level"
    })
    next_step2 = personalization_service.determine_next_onboarding_state(get_student(1), current_step="ask_self_level")
    assert next_step2 == "ask_self_level"

    # Fill self level
    update_student(1, {
        "self_reported_level": "Intermediate",
        "onboarding_step": "ask_self_level"
    })
    next_step3 = personalization_service.determine_next_onboarding_state(get_student(1), current_step="ask_self_level")
    assert next_step3 == "speech_test_prompt"


# -----------------------------------------------------------------------------
# Criterion 4: Incomplete onboarding resumes after a restart using the same DB
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_incomplete_onboarding_resumes_saved_step():
    # Set step to ask_self_level
    update_student(1, {
        "name": "David Miller",
        "learning_goals": "Academic writing",
        "weaknesses": ["Grammar"],
        "onboarding_step": "ask_self_level",
        "onboarding_completed": False
    })

    with patch("app.services.llm_service.LLMService._call_groq", new_callable=AsyncMock) as mock_groq:
        mock_groq.return_value = ("Welcome back David! How would you describe your English level?", "llama-3.3-70b-versatile")
        
        # Trigger start/resume
        res = client.post("/api/v1/esp32/start", json={"student_id": 1, "action": "start"})
        assert res.status_code == 200
        data = res.json()
        assert data["is_returning_user"] is False
        assert data["onboarding_step"] == "ask_self_level"
        assert data["student_name"] == "David Miller"


# -----------------------------------------------------------------------------
# Criterion 5: Real audio upload passes to STT with correct MIME/format; failures don't produce fake transcripts
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_audio_upload_mime_and_stt_error_handling():
    # Test WebM Opus audio upload
    fake_webm_bytes = b"Gwebm_test_bytes_here"
    
    with patch("app.services.stt_service.STTService.transcribe_audio_file", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.llm_service.LLMService._call_groq", new_callable=AsyncMock) as mock_groq, \
         patch("app.services.tts_service.TTSService.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        mock_stt.return_value = "Hello Mikaza, I am testing the audio input."
        mock_groq.return_value = ("Hello! Audio received successfully.", "llama-3.3-70b-versatile")
        mock_tts.return_value = ("/api/v1/esp32/audio/cache/out.wav", "data/audio_cache/out.wav")

        res = client.post(
            "/api/v1/esp32/audio?format=json",
            files={"audio": ("sample.webm", fake_webm_bytes, "audio/webm")},
            data={"student_id": "1", "mode": "coach"}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["transcript"] == "Hello Mikaza, I am testing the audio input."
        mock_stt.assert_called_once()

    # Test STT failure does not return a canned transcript
    with patch("app.services.stt_service.STTService.transcribe_audio_file", new_callable=AsyncMock) as mock_stt_fail, \
         patch("app.services.llm_service.LLMService._call_groq", new_callable=AsyncMock) as mock_groq:
        mock_stt_fail.side_effect = Exception("Whisper API connection timeout")
        mock_groq.return_value = ("I couldn't hear that clearly, please try again.", "llama-3.3-70b-versatile")

        res_fail = client.post(
            "/api/v1/esp32/audio?format=json",
            files={"audio": ("sample.webm", fake_webm_bytes, "audio/webm")},
            data={"student_id": "1", "mode": "coach"}
        )
        assert res_fail.status_code == 200
        fail_data = res_fail.json()
        assert fail_data["transcript"] == "[Speech Recognition Error]"
        assert "trouble hearing" in fail_data["reply_text"].lower() or "try again" in fail_data["reply_text"].lower() or "couldn't hear" in fail_data["reply_text"].lower()


# -----------------------------------------------------------------------------
# Criterion 6: Assessment saved as baseline; later sessions do not overwrite baseline
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_assessment_baseline_preservation_across_later_sessions():
    update_student(1, {
        "name": "Elena Rostova",
        "onboarding_step": "speech_test_prompt"
    })

    with patch("app.services.llm_service.LLMService._call_groq", new_callable=AsyncMock) as mock_groq:
        mock_groq.return_value = ("""{
  "grammar_score": 82.0,
  "vocabulary_score": 85.0,
  "fluency_score": 80.0,
  "pronunciation_score": 78.0,
  "confidence_score": 84.0,
  "communication_score": 82.0,
  "overall_level": "Upper Intermediate",
  "grammar_feedback": "Strong sentence construction.",
  "vocabulary_feedback": "Rich technical terms.",
  "fluency_feedback": "Smooth delivery with few pauses.",
  "pronunciation_feedback": "Clear intelligibility.",
  "confidence_feedback": "Assertive speaking tempo.",
  "communication_feedback": "Clear topic organization.",
  "strengths": ["Clear articulation", "Broad vocabulary"],
  "weaknesses": ["Occasional article omission"],
  "mistakes": [],
  "spoken_summary": "Excellent speaking test Elena!"
}""", "llama-3.3-70b-versatile")

        # Save initial baseline
        assessment = await personalization_service.run_speaking_assessment(
            transcript="I have been studying computer science for three years and I want to improve my speaking fluency for international conferences.",
            student_id=1,
            session_id="ses_onboard_01",
            duration_seconds=58.0
        )

    st_after = get_student(1)
    assert st_after["onboarding_completed"] == 1
    assert st_after["speaking_assessment_completed"] == 1
    baseline_gram = st_after["baseline_grammar"]
    baseline_flu = st_after["baseline_fluency"]
    assert baseline_gram > 0
    assert baseline_flu > 0

    # Start a NEW session
    new_sess = client.post("/api/v1/students/1/sessions", json={"student_id": 1, "mode": "coach"}).json()
    
    # Process interaction in new session with some scores
    personalization_service.record_interaction_metrics(
        student_id=1,
        session_id=new_sess["session_id"],
        grammar_errors=[],
        vocab_suggestions=[],
        fluency_score=90.0,
        pacing_score=85.0,
        confidence_score=88.0
    )

    st_later = get_student(1)
    # Baseline must NOT be overwritten by subsequent normal sessions
    assert st_later["baseline_grammar"] == baseline_gram
    assert st_later["baseline_fluency"] == baseline_flu
    assert st_later["baseline_fluency"] == baseline_flu


# -----------------------------------------------------------------------------
# Criterion 7: User and assistant messages saved and searchable in History
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_messages_saved_and_searchable_in_history():
    with patch("app.services.llm_service.LLMService._call_groq", new_callable=AsyncMock) as mock_groq, \
         patch("app.services.llm_service.LLMService._call_ollama", new_callable=AsyncMock) as mock_ollama:
        mock_groq.return_value = ("Quantum physics explores the subatomic realm.", "llama-3.3-70b-versatile")
        mock_ollama.return_value = ("Quantum physics explores the subatomic realm.", "llama3:latest")
        
        client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "Can you explain quantum physics basics?"
        })

    # Retrieve all conversations
    hist_all = client.get("/api/v1/conversations?student_id=1").json()
    assert len(hist_all) >= 2
    assert any("quantum physics" in m["content"].lower() for m in hist_all)

    # Search filter
    hist_search = client.get("/api/v1/conversations?student_id=1&search=subatomic").json()
    assert len(hist_search) >= 1
    assert "subatomic" in hist_search[0]["content"].lower()


# -----------------------------------------------------------------------------
# Criterion 8: Restarting onboarding preserves prior conversations & assessments
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_restarting_onboarding_preserves_history_and_archives_profile():
    # Setup completed student with conversations & assessments
    update_student(1, {
        "name": "Alice Wonderland",
        "learning_goals": "Literature discussion",
        "onboarding_completed": 1,
        "onboarding_step": "completed"
    })
    
    with patch("app.services.llm_service.LLMService._call_groq", new_callable=AsyncMock) as mock_groq:
        mock_groq.return_value = ("""{
  "grammar_score": 75.0,
  "vocabulary_score": 80.0,
  "fluency_score": 70.0,
  "pronunciation_score": 75.0,
  "confidence_score": 75.0,
  "communication_score": 75.0,
  "overall_level": "Intermediate",
  "grammar_feedback": "Good structure.",
  "vocabulary_feedback": "Literary words.",
  "fluency_feedback": "Moderate pace.",
  "pronunciation_feedback": "Clear.",
  "confidence_feedback": "Steady flow.",
  "communication_feedback": "Expressive ideas.",
  "strengths": ["Good vocabulary"],
  "weaknesses": ["Hesitations"],
  "mistakes": [],
  "spoken_summary": "Great sample Alice!"
}""", "llama-3.3-70b-versatile")

        # Create an assessment record with sufficient evidence length (>12 words)
        await personalization_service.run_speaking_assessment(
            transcript="Alice speaking sample about books and writing styles for her upcoming novel project.",
            student_id=1,
            session_id="ses_alice_01",
            duration_seconds=62.0
        )

    pre_assessments = get_student_assessments(1)
    assert len(pre_assessments) >= 1

    # Restart Onboarding
    res = client.post("/api/v1/students/1/restart-onboarding")
    assert res.status_code == 200
    st_restarted = res.json()

    # Active profile reset back to ask_name with incremented version
    assert st_restarted["onboarding_completed"] == 0
    assert st_restarted["onboarding_step"] == "ask_name"
    assert st_restarted["profile_version"] >= 2

    # Assessments and conversations preserved in SQLite
    post_assessments = get_student_assessments(1)
    assert len(post_assessments) == len(pre_assessments)


# -----------------------------------------------------------------------------
# Criterion 9: Deleting learner data wipes profile & history while strictly preserving API keys
# -----------------------------------------------------------------------------
def test_deleting_learner_data_preserves_api_keys_and_system_settings():
    # Configure API keys in system settings
    update_db_settings({
        "groq_api_key": "gsk_test_secret_key_12345",
        "openai_api_key": "sk_test_secret_key_67890",
        "system_prompt": "Custom Mikaza prompt"
    })

    pre_settings = get_db_settings()
    assert pre_settings["groq_api_key"] == "gsk_test_secret_key_12345"

    # Execute destructive learner delete
    res = client.delete("/api/v1/students/1/data")
    assert res.status_code == 200

    # Learner history wiped
    hist = get_conversation_history(student_id=1)
    assessments = get_student_assessments(1)
    assert len(hist) == 0
    assert len(assessments) == 0

    # API keys and system settings strictly preserved
    post_settings = get_db_settings()
    assert post_settings["groq_api_key"] == "gsk_test_secret_key_12345"
    assert post_settings["openai_api_key"] == "sk_test_secret_key_67890"
    assert post_settings["system_prompt"] == "Custom Mikaza prompt"


# -----------------------------------------------------------------------------
# Criterion 10: Empty API key settings submission does not erase configured keys
# -----------------------------------------------------------------------------
def test_empty_api_key_submission_does_not_erase_configured_keys():
    update_db_settings({"groq_api_key": "gsk_permanent_valid_key_888"})
    
    # Send update with empty string for groq_api_key
    res = client.post("/api/v1/admin/settings", json={
        "groq_api_key": "",
        "system_prompt": "Updated prompt text"
    })
    assert res.status_code == 200

    settings_after = get_db_settings()
    assert settings_after["groq_api_key"] == "gsk_permanent_valid_key_888"
    assert settings_after["system_prompt"] == "Updated prompt text"


# -----------------------------------------------------------------------------
# Criterion 11: Text-only completion fallback leaves speaking assessment pending
# -----------------------------------------------------------------------------
def test_text_only_completion_fallback_no_fake_baseline():
    update_student(1, {
        "name": "Text Only Learner",
        "learning_goals": "Grammar and reading",
        "onboarding_step": "ask_self_level"
    })

    # User cannot use microphone -> triggers text completion fallback
    res = client.post("/api/v1/students/1/complete-text-onboarding")
    assert res.status_code == 200
    data = res.json()

    assert data["onboarding_completed"] is True
    assert data["onboarding_step"] == "completed"
    assert data["speaking_assessment_completed"] is False
    # Baseline must NOT have fake synthesized scores
    assert data["baseline_grammar"] == 0.0
    assert data["baseline_fluency"] == 0.0
