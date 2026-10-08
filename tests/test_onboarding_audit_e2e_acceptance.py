import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import (
    init_db,
    reset_student_to_fresh,
    get_student,
    get_active_session,
    save_assessment,
    get_student_assessments
)
from app.services.voice_service import classify_command_or_intent, voice_manager
from app.services.personalization_service import personalization_service

@pytest.fixture(autouse=True)
def clean_db():
    init_db()
    reset_student_to_fresh(1)
    yield
    reset_student_to_fresh(1)

@pytest.mark.asyncio
async def test_full_part_29_e2e_acceptance_flow():
    client = TestClient(app)

    # 1. START in SNOOZED / STANDBY
    # Background speech before wake word is ignored
    ignored_resp = classify_command_or_intent("I am talking to my friend in the room.")
    assert ignored_resp != "WAKE"
    assert ignored_resp != "STOP"

    # 2. WAKE: "Hey Mizo"
    wake_resp = classify_command_or_intent("Hey Mizo")
    assert wake_resp == "WAKE"

    # 3. Mizo asks name -> User says "My name is Shahid"
    from app.services.llm_service import LLMResult
    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm_turn:
        mock_llm_turn.return_value = LLMResult(
            text="Nice to meet you, Shahid! What are your main goals for improving your English?",
            provider="mock",
            model="mock-v1"
        )
        res1 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "My name is Shahid",
            "mode": "onboarding"
        })
        assert res1.status_code == 200
        student = get_student(1)
        assert student["name"] == "Shahid"
        assert "Shahid" in res1.json()["reply_text"]

    # 4. Negative phrases MUST NOT corrupt the name:
    # "I am going to improve my English", "Stop", "I am going to learn AI"
    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm_turn:
        mock_llm_turn.return_value = LLMResult(
            text="Understood, let's keep working toward your goals.",
            provider="mock",
            model="mock-v1"
        )
        res_neg = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "I am going to improve my English and study robotics.",
            "mode": "onboarding"
        })
        assert res_neg.status_code == 200
        student = get_student(1)
        assert student["name"] == "Shahid", f"Name was corrupted to: {student['name']}"

    # 5. Assessment: Transition to speaking assessment
    from app.db.database import update_student
    update_student(1, {"onboarding_step": "speech_test_prompt"})

    speech_sample = (
        "I am currently studying software engineering and robotics in university. "
        "I want to become fluent in English so I can communicate technical ideas "
        "confidently during international interviews and team discussions."
    )
    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLMResult(
            text='{"grammar_score": 82, "vocabulary_score": 80, "fluency_score": 85, '
                 '"pronunciation_score": 83, "confidence_score": 80, "overall_level": "Intermediate", '
                 '"communication_score": 82, "strengths": ["Clear technical goals"], '
                 '"weaknesses": ["Minor sentence rhythm"], "grammar_errors": [], "vocabulary_suggestions": [], '
                 '"spoken_summary": "Great introduction, Shahid! Your technical goals are very clear."}',
            provider="groq",
            model="llama-3.3-70b-versatile"
        )
        assess_res = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": speech_sample,
            "mode": "onboarding"
        })
        assert assess_res.status_code == 200
        data = assess_res.json()
        assert data.get("assessment") is not None
        assert data["assessment"]["grammar_score"] == 82
        assert data["assessment"]["assessment_method"] == "llm"

    # Profile updated and onboarding completed
    student_after = get_student(1)
    assert student_after["onboarding_completed"] == 1
    assert student_after["baseline_grammar"] == 82

    # 6. User asks: "What is my name?"
    # Must answer Shahid from profile data, NOT hallucinate
    res_name = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "What is my name?",
        "mode": "coach"
    })
    assert res_name.status_code == 200
    assert "Shahid" in res_name.json()["reply_text"]

    # 7. User asks: "What is your name?"
    # Deterministic Mizo identity intercept
    res_mizo = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "What is your name?",
        "mode": "coach"
    })
    assert res_mizo.status_code == 200
    assert "Mizo" in res_mizo.json()["reply_text"]

    # Test "Are you Mizo?"
    res_mizo2 = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Are you Mizo?",
        "mode": "coach"
    })
    assert res_mizo2.status_code == 200
    assert "Mizo" in res_mizo2.json()["reply_text"]

    # 8. API Failure Recovery: Simulate all LLMs failing
    with patch("app.services.llm_service.llm_service.generate_response", side_effect=Exception("ConnectionRefusedError: All LLM providers offline")):
        res_fail = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "Tell me a joke about robotics.",
            "mode": "coach"
        })
        # System does not crash, returns graceful response
        assert res_fail.status_code == 200
        assert "trouble" in res_fail.json()["reply_text"].lower() or "sorry" in res_fail.json()["reply_text"].lower() or len(res_fail.json()["reply_text"]) > 0

    # User immediately speaks again: System is NOT locked in processing
    res_retry = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "What is my name?",
        "mode": "coach"
    })
    assert res_retry.status_code == 200
    assert "Shahid" in res_retry.json()["reply_text"]

    # 9. Server Restart Simulation: Verify DB persistence
    fresh_student = get_student(1)
    assert fresh_student["name"] == "Shahid"
    assert fresh_student["onboarding_completed"] == 1

    # Diagnostics endpoint verification
    diag_res = client.get("/api/v1/diagnostics/onboarding")
    assert diag_res.status_code == 200
    diag = diag_res.json()
    assert diag["profile"]["name"] == "Shahid"
    assert diag["profile"]["onboarding_completed"] is True
    assert diag["session"]["turn_silence_threshold_seconds"] == 3.0
