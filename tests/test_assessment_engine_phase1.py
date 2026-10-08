import pytest
import json
from unittest.mock import patch, AsyncMock
from app.db.database import (
    get_student,
    update_student,
    get_assessment_by_id,
    get_student_assessments,
    reset_student_to_fresh,
    init_db
)
from app.services.personalization_service import personalization_service
from app.services.llm_service import LLMResult


@pytest.fixture(autouse=True)
def setup_db():
    init_db()
    reset_student_to_fresh(1)
    yield
    reset_student_to_fresh(1)


@pytest.mark.asyncio
async def test_criterion_a_duration_not_hardcoded_to_60():
    """Criterion A: Assessment duration is not hardcoded to 60 seconds."""
    sample_text = (
        "I have been studying computer science and software development. "
        "I build web applications and robotics projects in my spare time. "
        "My goal is to express technical ideas clearly and accurately."
    )
    word_count = len(sample_text.split())

    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLMResult(
            text=json.dumps({
                "grammar_score": 85.0,
                "vocabulary_score": 88.0,
                "fluency_score": 82.0,
                "confidence_score": 80.0,
                "communication_score": 84.0,
                "overall_level": "Upper Intermediate",
                "grammar_feedback": "Strong grammatical consistency.",
                "vocabulary_feedback": "Rich technical terms.",
                "fluency_feedback": "Natural continuity.",
                "confidence_feedback": "Assertive speaking delivery.",
                "communication_feedback": "Clear explanation of goals.",
                "strengths": ["Technical vocabulary"],
                "weaknesses": ["Minor rhythm adjustments"],
                "mistakes": [],
                "spoken_summary": "Great work on your speaking assessment!"
            }),
            provider="groq",
            model="llama-3.3-70b-versatile"
        )

        # Test case 1: Measured duration = 38.5 seconds (not 60s)
        res_38 = await personalization_service.run_speaking_assessment(
            transcript=sample_text,
            student_id=1,
            session_id="test_session_38",
            duration_seconds=38.5
        )
        assert res_38["duration_seconds"] == 38.5
        expected_wpm_38 = round((word_count / 38.5) * 60, 1)
        assert res_38["words_per_minute"] == expected_wpm_38
        assert res_38["words_per_minute"] != round((word_count / 60.0) * 60, 1)

        # Verify saved in SQLite assessments record
        rec_38 = get_assessment_by_id(res_38["assessment_id"])
        assert rec_38 is not None
        assert rec_38["duration_seconds"] == 38.5

        # Test case 2: Measured duration = 115.0 seconds (not 60s)
        res_115 = await personalization_service.run_speaking_assessment(
            transcript=sample_text,
            student_id=1,
            session_id="test_session_115",
            duration_seconds=115.0
        )
        assert res_115["duration_seconds"] == 115.0
        expected_wpm_115 = round((word_count / 115.0) * 60, 1)
        assert res_115["words_per_minute"] == expected_wpm_115
        assert res_115["words_per_minute"] != round((word_count / 60.0) * 60, 1)

        # Verify saved in SQLite assessments record
        rec_115 = get_assessment_by_id(res_115["assessment_id"])
        assert rec_115 is not None
        assert rec_115["duration_seconds"] == 115.0


@pytest.mark.asyncio
async def test_criterion_b_stt_timing_and_timestamps_preserved():
    """Criterion B: STT timing information and Whisper timestamps are preserved."""
    sample_text = (
        "I am presenting my final year project on automated embedded controllers. "
        "The project integrates acoustic sensors with a microcontroller."
    )
    whisper_segments = [
        {"id": 0, "start": 0.0, "end": 4.2, "text": "I am presenting my final year project on automated embedded controllers."},
        {"id": 1, "start": 4.5, "end": 8.9, "text": "The project integrates acoustic sensors with a microcontroller."}
    ]
    whisper_words = [
        {"word": "I", "start": 0.0, "end": 0.3},
        {"word": "am", "start": 0.3, "end": 0.5},
        {"word": "presenting", "start": 0.5, "end": 1.2}
    ]
    stt_meta = {
        "duration_seconds": 9.2,
        "segments": whisper_segments,
        "words": whisper_words,
        "language": "en"
    }

    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLMResult(
            text=json.dumps({
                "grammar_score": 80.0,
                "vocabulary_score": 82.0,
                "fluency_score": 78.0,
                "confidence_score": 79.0,
                "communication_score": 81.0,
                "overall_level": "Intermediate",
                "grammar_feedback": "Accurate phrasing.",
                "vocabulary_feedback": "Good embedded systems vocabulary.",
                "fluency_feedback": "Consistent flow.",
                "confidence_feedback": "Steady presentation delivery.",
                "communication_feedback": "Clear explanation.",
                "strengths": ["Domain terminology"],
                "weaknesses": ["Pacing consistency"],
                "mistakes": [],
                "spoken_summary": "Solid technical presentation."
            }),
            provider="groq",
            model="llama-3.3-70b-versatile"
        )

        res = await personalization_service.run_speaking_assessment(
            transcript=sample_text,
            student_id=1,
            session_id="test_session_timing",
            duration_seconds=9.2,
            audio_metadata=stt_meta
        )

        # 1. Preserved in returned result
        assert res.get("stt_metadata") is not None
        assert res["stt_metadata"]["duration_seconds"] == 9.2
        assert len(res["stt_metadata"]["segments"]) == 2
        assert res["stt_metadata"]["segments"][0]["start"] == 0.0
        assert res["stt_metadata"]["segments"][0]["end"] == 4.2
        assert len(res["stt_metadata"]["words"]) == 3

        # 2. Preserved in SQLite database record
        rec = get_assessment_by_id(res["assessment_id"])
        assert rec is not None
        assert rec.get("stt_metadata") is not None
        saved_meta = rec["stt_metadata"]
        assert saved_meta["duration_seconds"] == 9.2
        assert len(saved_meta["segments"]) == 2
        assert saved_meta["segments"][1]["start"] == 4.5
        assert saved_meta["segments"][1]["end"] == 8.9


@pytest.mark.asyncio
async def test_criterion_c_pronunciation_not_falsely_scored_from_text():
    """Criterion C: Pronunciation is not falsely scored from transcript-only data."""
    sample_text = (
        "I enjoy reading science fiction books and practicing public speaking. "
        "Every morning I practice reading out loud to build confidence."
    )

    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLMResult(
            text=json.dumps({
                "grammar_score": 84.0,
                "vocabulary_score": 80.0,
                "fluency_score": 82.0,
                "confidence_score": 85.0,
                "communication_score": 83.0,
                "overall_level": "Upper Intermediate",
                "grammar_feedback": "Accurate tenses and agreement.",
                "vocabulary_feedback": "Good conversational range.",
                "fluency_feedback": "Smooth flow without long pauses.",
                "confidence_feedback": "Clear delivery.",
                "communication_feedback": "Clear message conveying daily habits.",
                "strengths": ["Self-directed practice"],
                "weaknesses": ["Lexical complexity"],
                "mistakes": [],
                "spoken_summary": "Great speaking practice habits!"
            }),
            provider="groq",
            model="llama-3.3-70b-versatile"
        )

        res = await personalization_service.run_speaking_assessment(
            transcript=sample_text,
            student_id=1,
            session_id="test_session_pron",
            duration_seconds=35.0
        )

        # 1. Result must explicitly indicate pronunciation is None
        assert res["pronunciation_score"] is None
        assert "disabled for transcript-only" in res["pronunciation_feedback"].lower()

        # 2. Database assessment record must have pronunciation_score as NULL (None)
        rec = get_assessment_by_id(res["assessment_id"])
        assert rec is not None
        assert rec["pronunciation_score"] is None

        # 3. Student baseline in DB must NOT have a fake pronunciation score
        student = get_student(1)
        assert student["baseline_pronunciation"] is None
        assert student["pronunciation_score"] is None


@pytest.mark.asyncio
async def test_criterion_d_target_learning_level_not_overwritten_by_assessed_level():
    """Criterion D: Target learning level is not overwritten by current assessed level."""
    # Set student's target learning level to Advanced
    update_student(1, {
        "name": "Maria",
        "target_level": "Advanced",
        "onboarding_completed": False
    })
    st_before = get_student(1)
    assert st_before["target_level"] == "Advanced"
    assert st_before["assessed_level"] is None

    # Student speaks with Beginner/Elementary level English
    speech_sample = "Yesterday I go market. I buy many fruit and very happy. He don't know my house."

    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLMResult(
            text=json.dumps({
                "grammar_score": 45.0,
                "vocabulary_score": 48.0,
                "fluency_score": 42.0,
                "confidence_score": 40.0,
                "communication_score": 44.0,
                "overall_level": "Elementary",
                "grammar_feedback": "Frequent past tense and agreement errors (I go, he don't).",
                "vocabulary_feedback": "Basic everyday vocabulary.",
                "fluency_feedback": "Short fragmented sentences.",
                "confidence_feedback": "Hesitant delivery.",
                "communication_feedback": "Basic meaning understood despite errors.",
                "strengths": ["Communicated core message"],
                "weaknesses": ["Past tense verbs", "Subject-verb agreement"],
                "mistakes": [
                    {
                        "utterance": "Yesterday I go market",
                        "error_text": "I go",
                        "correction": "I went to the",
                        "mistake_type": "grammar",
                        "explanation": "Past time markers require the past tense 'went'."
                    }
                ],
                "spoken_summary": "Good effort Maria! We will focus on past tense verbs and agreement."
            }),
            provider="groq",
            model="llama-3.3-70b-versatile"
        )

        res = await personalization_service.run_speaking_assessment(
            transcript=speech_sample,
            student_id=1,
            session_id="test_session_target_level",
            duration_seconds=18.0
        )

        assert res["overall_level"] == "Elementary"

        # Check student record in database
        st_after = get_student(1)
        assert st_after is not None

        # The assessed current level must be Elementary
        assert st_after["assessed_level"] == "Elementary"

        # CRITICAL: The student's target learning level MUST REMAIN "Advanced", NOT overwritten!
        assert st_after["target_level"] == "Advanced"


@pytest.mark.asyncio
async def test_criterion_e_no_score_anchor_bias_in_prompt():
    """Criterion E: Assessment prompt no longer contains example numerical scores that could anchor the model."""
    captured_messages = []
    captured_system_prompt = []

    async def mock_generate(messages, system_prompt="", **kwargs):
        captured_messages.extend(messages)
        captured_system_prompt.append(system_prompt)
        return LLMResult(
            text=json.dumps({
                "grammar_score": 75.0,
                "vocabulary_score": 75.0,
                "fluency_score": 75.0,
                "confidence_score": 75.0,
                "communication_score": 75.0,
                "overall_level": "Intermediate",
                "grammar_feedback": "Clear grammar.",
                "vocabulary_feedback": "Adequate range.",
                "fluency_feedback": "Good continuity.",
                "confidence_feedback": "Confident delivery.",
                "communication_feedback": "Clear explanation.",
                "strengths": ["Good clarity"],
                "weaknesses": ["Complex sentence structures"],
                "mistakes": [],
                "spoken_summary": "Good work on your assessment!"
            }),
            provider="groq",
            model="llama-3.3-70b-versatile"
        )

    with patch("app.services.llm_service.llm_service.generate_response", side_effect=mock_generate):
        await personalization_service.run_speaking_assessment(
            transcript="I want to improve my English speaking confidence and learn vocabulary for my career.",
            student_id=1,
            session_id="test_prompt_audit",
            duration_seconds=22.0
        )

    assert len(captured_system_prompt) == 1
    sys_prompt = captured_system_prompt[0]
    user_prompt = captured_messages[0]["content"]

    # 1. No numerical anchoring ranges in CEFR criteria
    assert "scores < 40" not in sys_prompt
    assert "scores 40-54" not in sys_prompt
    assert "scores 55-69" not in sys_prompt
    assert "scores 70-84" not in sys_prompt
    assert "scores 85-100" not in sys_prompt
    assert "< 40" not in sys_prompt
    assert "40-54" not in sys_prompt
    assert "55-69" not in sys_prompt
    assert "70-84" not in sys_prompt
    assert "85-100" not in sys_prompt

    # 2. No example numerical numbers in user prompt template
    assert "e.g. 75" not in user_prompt
    assert "e.g. 80" not in user_prompt
    assert "75.0" not in user_prompt
    assert "85.0" not in user_prompt

    # 3. Explicit separation of objective evaluation and encouraging coaching
    assert "SEPARATION OF OBJECTIVE EVALUATION AND ENCOURAGING COACHING" in sys_prompt
    assert "STRICT OBJECTIVITY" in sys_prompt
    assert "ENCOURAGING FEEDBACK ISOLATION" in sys_prompt

    # 4. Pronunciation scoring is disabled and not requested from model
    assert "PRONUNCIATION" in sys_prompt
    assert "pronunciation_score" not in user_prompt
