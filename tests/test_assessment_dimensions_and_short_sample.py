import pytest
import json
from unittest.mock import patch, AsyncMock
from app.services.personalization_service import personalization_service
from app.services.llm_service import LLMResult
from app.db.database import get_student, update_student, PROTOTYPE_STUDENT_ID


@pytest.mark.asyncio
async def test_short_speaking_sample_rejected_without_scores():
    """Verify that an extremely short sample (under 12 words) does NOT generate meaningful scores or calibrate baseline."""
    update_student(PROTOTYPE_STUDENT_ID, {
        "name": "Shahid",
        "baseline_grammar": 0.0,
        "assessed_level": None,
        "onboarding_completed": False
    })
    
    # 5-word sample
    short_transcript = "I like playing cricket today."
    
    res = await personalization_service.run_speaking_assessment(
        transcript=short_transcript,
        student_id=PROTOTYPE_STUDENT_ID,
        session_id="short_test_session"
    )
    
    # Must be marked as insufficient sample with no scores
    assert res.get("insufficient_sample") is True
    assert res["grammar_score"] is None
    assert res["vocabulary_score"] is None
    assert res["fluency_score"] is None
    assert res["coherence_score"] is None
    assert res["communication_score"] is None
    assert res["overall_level"] == "Uncalibrated"
    assert "too short" in res["spoken_summary"].lower()
    
    # Baseline must NOT be calibrated
    student_after = get_student(PROTOTYPE_STUDENT_ID)
    assert student_after["baseline_grammar"] == 0.0
    assert student_after["assessed_level"] is None


@pytest.mark.asyncio
async def test_proper_communication_assessment_separated_dimensions():
    """Verify that a valid speaking sample separates grammar, vocabulary, fluency, coherence, and communication."""
    full_transcript = (
        "I am currently studying robotics and computer engineering at my university. "
        "I build autonomous mobile systems and write firmware in Python and C++. "
        "My goal is to express technical ideas clearly during international conferences."
    )
    
    # Test with rule-based fallback evaluator (offline safe)
    with patch("app.services.llm_service.llm_service.generate_response", side_effect=Exception("LLM offline")):
        res = await personalization_service.run_speaking_assessment(
            transcript=full_transcript,
            student_id=PROTOTYPE_STUDENT_ID,
            session_id="valid_assessment_session",
            duration_seconds=45.0
        )
    
    # Verify dimensions are cleanly separated
    assert "grammar_score" in res and res["grammar_score"] is not None
    assert "vocabulary_score" in res and res["vocabulary_score"] is not None
    assert "fluency_score" in res and res["fluency_score"] is not None
    assert "coherence_score" in res and res["coherence_score"] is not None
    assert "confidence_score" in res and res["confidence_score"] is not None
    assert "communication_score" in res and res["communication_score"] is not None
    assert "overall_level" in res and res["overall_level"] != "Uncalibrated"
    
    # Verify feedback strings for each dimension
    assert "grammar_feedback" in res and len(res["grammar_feedback"]) > 0
    assert "vocabulary_feedback" in res and len(res["vocabulary_feedback"]) > 0
    assert "fluency_feedback" in res and len(res["fluency_feedback"]) > 0
    assert "coherence_feedback" in res and len(res["coherence_feedback"]) > 0
    assert "communication_feedback" in res and len(res["communication_feedback"]) > 0


@pytest.mark.asyncio
async def test_deterministic_name_preservation():
    """Verify that an existing student name is protected and cannot be randomly overwritten by casual dialogue."""
    update_student(PROTOTYPE_STUDENT_ID, {"name": "Shahid"})
    
    # Student says something casual containing other nouns
    casual_utterances = [
        "I was talking to Alex about robotics.",
        "My friend John is studying mechanical engineering.",
        "I am studying computer science."
    ]
    
    for utt in casual_utterances:
        updated = personalization_service.extract_and_update_student_memory(PROTOTYPE_STUDENT_ID, utt)
        student_now = get_student(PROTOTYPE_STUDENT_ID)
        assert student_now["name"] == "Shahid", f"Name was unexpectedly changed by utterance: {utt}"
