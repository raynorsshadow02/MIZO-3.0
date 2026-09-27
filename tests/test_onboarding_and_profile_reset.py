import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.db.database import get_student, reset_student_to_fresh
from app.services.personalization_service import personalization_service

client = TestClient(app)


def test_student_reset_and_editable_profile():
    # 1. Reset Student to Fresh
    reset_res = client.post("/api/v1/students/1/reset")
    assert reset_res.status_code == 200
    data = reset_res.json()
    assert data["name"] == "New Learner"
    assert data["onboarding_completed"] is False
    assert data["onboarding_step"] == "ask_name"
    assert data["fluency_score"] == 0.0
    assert data["grammar_score"] == 0.0
    assert data["vocabulary_score"] == 0.0
    assert data["confidence_score"] == 0.0
    assert data["weaknesses"] == []

    # 2. Edit Profile via PUT
    edit_payload = {
        "name": "Sameer Sam",
        "grade": "Grade 10",
        "target_level": "Advanced",
        "learning_goals": "Master fluent debate and clear pronunciations",
        "weaknesses": ["Filler words", "Complex sentence timing"],
        "interests": "Robotics, Astrophysics, Machine Learning",
        "strengths": ["Enthusiastic speaker", "Quick comprehension"]
    }
    put_res = client.put("/api/v1/students/1", json=edit_payload)
    assert put_res.status_code == 200
    updated_data = put_res.json()
    assert updated_data["name"] == "Sameer Sam"
    assert updated_data["grade"] == "Grade 10"
    assert "Robotics" in updated_data["interests"]
    assert "Filler words" in updated_data["weaknesses"]

    # Verify DB persistence directly
    st = get_student(1)
    assert st is not None
    assert st["name"] == "Sameer Sam"
    assert st["grade"] == "Grade 10"


@pytest.mark.asyncio
async def test_full_7_step_onboarding_flow():
    # Reset student to start fresh at Step 1
    reset_student_to_fresh(1)
    student = get_student(1)
    assert student["onboarding_step"] == "ask_name"

    with patch("app.services.llm_service.LLMService._call_groq", new_callable=AsyncMock) as mock_groq, \
         patch("app.services.tts_service.TTSService.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        mock_groq.return_value = ("Nice to meet you! What difficulties do you face?", "llama-3.3-70b-versatile")
        mock_tts.return_value = ("/api/v1/esp32/audio/cache/test.wav", "data/audio_cache/test.wav")

        # Step 1: User gives name
        res1 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "My name is John Doe"
        })
        assert res1.status_code == 200
        st1 = get_student(1)
        assert st1["name"] == "John Doe"
        assert st1["onboarding_step"] in ["ask_goals", "ask_problems"]

        # Step 2: User states problems / goals
        mock_groq.return_value = ("I understand. What areas or challenges are your biggest weaknesses?", "llama-3.3-70b-versatile")
        res2 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "I struggle to speak smoothly without hesitations."
        })
        assert res2.status_code == 200
        st2 = get_student(1)
        assert "hesitations" in st2["learning_goals"].lower()
        assert st2["onboarding_step"] == "ask_weaknesses"

        # Step 3: User states weaknesses
        mock_groq.return_value = ("Got it. How would you describe your current English level?", "llama-3.3-70b-versatile")
        res3 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "My weaknesses are chemistry terms and using past tense verbs."
        })
        assert res3.status_code == 200
        st3 = get_student(1)
        assert st3["onboarding_step"] == "ask_self_level"

        # Step 4: User provides self-reported level
        mock_groq.return_value = ("Please talk for about one minute on your goals or a topic you love so I can assess your speaking.", "llama-3.3-70b-versatile")
        res_lvl = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "I am not sure about my level, maybe intermediate."
        })
        assert res_lvl.status_code == 200
        st_lvl = get_student(1)
        assert st_lvl["onboarding_step"] == "speech_test_prompt"

        # Step 5: User provides 1-2 minute speaking test response
        mock_groq.return_value = ("Great job! Your fluency is strong. Let's begin our first lesson!", "llama-3.3-70b-versatile")
        speech_sample = "Yesterday I built a small robotic car with my friend and we programmed it using Arduino. It was very exciting and taught us about electronics."
        res4 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": speech_sample
        })
        assert res4.status_code == 200
        st4 = get_student(1)
        assert st4["baseline_fluency"] > 0
        assert st4["grammar_score"] > 0
        assert st4["onboarding_step"] in ["speech_evaluation", "completed"]

        # Step 6 & 7: Final transition to teaching
        res5 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "Teach me how robotic sensors work."
        })
        assert res5.status_code == 200
        st5 = get_student(1)
        assert st5["onboarding_completed"] is True
        assert st5["onboarding_step"] == "completed"
