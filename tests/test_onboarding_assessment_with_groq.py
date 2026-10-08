import pytest
from unittest.mock import patch, AsyncMock
from app.services.personalization_service import personalization_service
from app.db.database import get_student, reset_student_to_fresh, get_db_settings
from app.services.llm_service import LLMService


@pytest.mark.asyncio
async def test_onboarding_assessment_calls_groq_and_advances_student():
    """Verify onboarding speech assessment calls configured Groq model, parses result, and advances student."""
    reset_student_to_fresh(1)
    transcript = "Hello my name is Alex Mercer and I am learning English with Mizo tutor."

    mock_json = (
        '{\n'
        '  "grammar_score": 88.0,\n'
        '  "vocabulary_score": 85.0,\n'
        '  "fluency_score": 82.0,\n'
        '  "pronunciation_score": 80.0,\n'
        '  "confidence_score": 85.0,\n'
        '  "communication_score": 84.0,\n'
        '  "overall_level": "Intermediate",\n'
        '  "grammar_feedback": "Good sentence structure with clear subject-verb agreement.",\n'
        '  "vocabulary_feedback": "Effective vocabulary suitable for self-introduction.",\n'
        '  "fluency_feedback": "Natural flow with minimal hesitations.",\n'
        '  "pronunciation_feedback": "Clear phonetic articulation.",\n'
        '  "confidence_feedback": "Demonstrates strong spoken communication confidence.",\n'
        '  "communication_feedback": "Engaging introduction.",\n'
        '  "strengths": ["Clear articulation", "Good introductory vocabulary"],\n'
        '  "weaknesses": ["Can expand complex sentence structures"],\n'
        '  "mistakes": [],\n'
        '  "spoken_summary": "Great job, Alex! You have a solid foundation in spoken English."\n'
        '}'
    )

    with patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq:
        mock_groq.return_value = (mock_json, "llama-3.3-70b-versatile")

        result = await personalization_service.run_speaking_assessment(
            transcript=transcript,
            student_id=1,
            session_id="test_ses_99",
            topic="Self Introduction & Experience",
            duration_seconds=45.0
        )

        assert result["overall_level"] == "Intermediate"
        assert result["grammar_score"] == 88.0
        assert result["communication_score"] == 84.0

        student = get_student(1)
        assert student["onboarding_completed"] == 1
        assert student["onboarding_step"] == "completed"
        assert student["baseline_grammar"] == 88.0
        mock_groq.assert_called_once()
