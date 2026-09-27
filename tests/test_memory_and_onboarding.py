import pytest
from app.db.database import (
    create_student,
    get_student,
    update_student,
    record_onboarding_baseline,
    create_session,
    get_session_by_id,
    log_student_mistake,
    get_recent_mistakes
)
from app.services.personalization_service import PersonalizationService


def test_onboarding_baseline_flow():
    """Verify dynamic onboarding baseline calibration and storage."""
    student_data = create_student({
        "name": "Maya Lin",
        "grade": "Grade 9",
        "native_language": "Spanish",
        "target_level": "Beginner",
        "interests": "Astronomy, Art",
        "learning_goals": "Speak English confidently",
        "onboarding_completed": False
    })
    student_id = student_data["id"]
    assert student_data["onboarding_completed"] is False
    assert student_data["onboarding_step"] in ["ask_name", "greeting"]

    # Advance onboarding step
    update_student(student_id, {"onboarding_step": "goals"})
    updated = get_student(student_id)
    assert updated["onboarding_step"] == "goals"

    # Finalize onboarding baseline
    record_onboarding_baseline(
        student_id=student_id,
        grammar=82.0,
        vocab=75.0,
        fluency=78.0,
        strengths=["Clear articulation", "Curious"],
        weaknesses=["Prepositions", "Verb tenses"]
    )

    final_student = get_student(student_id)
    assert final_student["onboarding_completed"] is True
    assert final_student["baseline_grammar"] == 82.0
    assert final_student["baseline_fluency"] == 78.0
    assert "Clear articulation" in final_student["strengths"]
    assert "Prepositions" in final_student["weaknesses"]


def test_session_lifecycle_and_metric_separation():
    """Verify that starting a new session resets current session metrics while historical data remains intact."""
    student = create_student({"name": "David", "onboarding_completed": True})
    student_id = student["id"]

    # Initial historical scores exist
    initial_historical_grammar = student["grammar_score"]

    # Start Session 1
    session_1 = create_session(student_id=student_id, mode="coach")
    session_1_id = session_1["session_id"]
    assert session_1["metrics"]["grammar_accuracy"] == 0.0
    assert session_1["metrics"]["message_count"] == 0

    # Record performance in Session 1
    PersonalizationService.record_interaction_metrics(
        student_id=student_id,
        session_id=session_1_id,
        grammar_errors=[{"error": "he don't", "correction": "he doesn't"}],
        vocab_suggestions=["Instead of 'good', try 'beneficial'"],
        fluency_score=85.0,
        pacing_score=88.0,
        confidence_score=82.0
    )

    s1_updated = get_session_by_id(session_1_id)
    assert s1_updated["metrics"]["message_count"] == 1
    assert s1_updated["metrics"]["fluency_score"] == 85.0

    # Start Session 2 (New learning session)
    session_2 = create_session(student_id=student_id, mode="tutor")
    session_2_id = session_2["session_id"]
    assert session_2_id != session_1_id

    # Verify Session 2 metrics are initialized to ZERO
    assert session_2["metrics"]["grammar_accuracy"] == 0.0
    assert session_2["metrics"]["fluency_score"] == 0.0
    assert session_2["metrics"]["message_count"] == 0

    # Verify historical data on the student profile is preserved
    refetched_student = get_student(student_id)
    assert refetched_student["total_sessions"] >= 2


def test_utterance_evaluation_and_mistake_logging():
    """Verify grammar flaw identification, filler penalties, and persistent mistake recording."""
    student = create_student({"name": "Lucas", "onboarding_completed": True})
    student_id = student["id"]
    session = create_session(student_id=student_id, mode="speech")
    session_id = session["session_id"]

    sample_utterance = "Um, like, I seen the teacher and he don't know the answer."
    grammar_errors, vocab_suggestions, fluency, pacing, confidence = PersonalizationService.evaluate_utterance(
        text=sample_utterance,
        student_id=student_id,
        session_id=session_id
    )

    # Check grammar corrections
    err_texts = [e["error"] for e in grammar_errors]
    assert any("seen" in e for e in err_texts)
    assert any("don't" in e for e in err_texts)

    # Check mistakes saved in DB
    recent_mistakes = get_recent_mistakes(student_id=student_id, limit=5)
    assert len(recent_mistakes) >= 2
    logged_errors = [m["error_text"] for m in recent_mistakes]
    assert any("seen" in le for le in logged_errors)
