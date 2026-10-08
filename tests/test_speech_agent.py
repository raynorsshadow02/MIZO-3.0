import pytest
import asyncio
from fastapi.testclient import TestClient

from app.main import create_app
from app.db.database import (
    PROTOTYPE_STUDENT_ID,
    get_student,
    create_or_get_speech_session,
    get_speech_session_by_id,
    get_active_speech_session,
    update_speech_session,
    get_previous_speech_attempts,
    update_student
)
from app.services.speech_service import speech_service, SpeechState
from app.services.personalization_service import personalization_service

app = create_app()
client = TestClient(app)


# ---------------------------------------------------------------------------
# Test 1: Create speech session
# ---------------------------------------------------------------------------
def test_01_create_speech_session():
    session = speech_service.get_or_create_session(student_id=PROTOTYPE_STUDENT_ID)
    assert session is not None
    assert session["student_id"] == PROTOTYPE_STUDENT_ID
    assert session["state"] in [SpeechState.SETUP.value, SpeechState.READY.value, SpeechState.OPTIONAL_RETRY.value]
    assert "session_id" in session
    assert session["attempt_number"] >= 1


# ---------------------------------------------------------------------------
# Test 2: Extract topic
# ---------------------------------------------------------------------------
def test_02_extract_topic():
    text1 = "I have 3 minutes to talk about robotics. I want to explain what robotics is, applications, advantages and future."
    ext1 = speech_service.extract_setup_information(text1)
    assert ext1["topic"] is not None
    assert "robotics" in ext1["topic"].lower()

    text2 = "My topic is Artificial Intelligence in Healthcare."
    ext2 = speech_service.extract_setup_information(text2)
    assert ext2["topic"] == "Artificial Intelligence in Healthcare"


# ---------------------------------------------------------------------------
# Test 3: Extract time limit
# ---------------------------------------------------------------------------
def test_03_extract_time_limit():
    assert speech_service.extract_setup_information("5 minutes")["time_limit_seconds"] == 300
    assert speech_service.extract_setup_information("180 seconds")["time_limit_seconds"] == 180
    assert speech_service.extract_setup_information("3 mins")["time_limit_seconds"] == 180
    assert speech_service.extract_setup_information("3:30")["time_limit_seconds"] == 210
    assert speech_service.extract_setup_information("I need 2 minutes for this talk")["time_limit_seconds"] == 120


# ---------------------------------------------------------------------------
# Test 4: Extract required points
# ---------------------------------------------------------------------------
def test_04_extract_required_points():
    text = "Definition, applications, benefits, risks and future scope."
    ext = speech_service.extract_setup_information(text)
    pts = ext["required_points"]
    assert len(pts) >= 4
    pts_lower = [p.lower() for p in pts]
    assert any("definition" in p for p in pts_lower)
    assert any("applications" in p for p in pts_lower)
    assert any("benefits" in p for p in pts_lower)
    assert any("risks" in p for p in pts_lower)

    text_all_in_one = "I have 3 minutes to talk about robotics. I want to explain what robotics is, applications, advantages and future."
    ext_all = speech_service.extract_setup_information(text_all_in_one)
    assert len(ext_all["required_points"]) >= 3


# ---------------------------------------------------------------------------
# Test 5: Missing setup information
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_05_missing_setup_information():
    # Reset session for clean testing
    session = speech_service.get_or_create_session(student_id=PROTOTYPE_STUDENT_ID, force_new=True)
    sid = session["session_id"]

    # Provide only topic
    res1 = await speech_service.process_speech_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sid,
        user_text="My topic is Renewable Energy."
    )
    assert res1["state"] == SpeechState.SETUP.value
    assert "topic" in res1["reply_text"].lower() or "points" in res1["reply_text"].lower() or "time" in res1["reply_text"].lower()

    # Provide required points
    res2 = await speech_service.process_speech_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sid,
        user_text="Solar power, wind turbines, grid storage, and environmental impact."
    )
    assert res2["state"] == SpeechState.SETUP.value
    # Now only time limit should be missing
    assert "time" in res2["reply_text"].lower() or "minute" in res2["reply_text"].lower()


# ---------------------------------------------------------------------------
# Test 6: Start recording
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_06_start_recording():
    session = speech_service.get_or_create_session(student_id=PROTOTYPE_STUDENT_ID, force_new=True)
    sid = session["session_id"]

    # Complete setup in one go
    res_setup = await speech_service.process_speech_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sid,
        user_text="Topic is Space Exploration. 3 minutes. Points: Rockets, Moon base, Mars mission, Space telescopes."
    )
    assert res_setup["state"] == SpeechState.READY.value
    assert "start" in res_setup["reply_text"].lower()

    # User says "start"
    res_start = await speech_service.process_speech_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sid,
        user_text="start"
    )
    assert res_start["state"] == SpeechState.RECORDING.value
    assert "listening" in res_start["reply_text"].lower() or "started" in res_start["reply_text"].lower()


# ---------------------------------------------------------------------------
# Test 7: Finish early
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_07_finish_early():
    session = speech_service.get_or_create_session(student_id=PROTOTYPE_STUDENT_ID, force_new=True)
    sid = session["session_id"]

    # Setup session directly
    update_speech_session(session["id"], {
        "topic": "Autonomous Vehicles",
        "time_limit_seconds": 180,
        "required_points": ["Lidar sensors", "Computer vision", "Safety benefits"],
        "state": SpeechState.RECORDING.value
    })

    # Explicit finish command with speech
    finish_res = await speech_service.process_speech_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sid,
        user_text="Autonomous vehicles use lidar sensors and computer vision for navigation. This improves safety benefits on highways. I'm done.",
        duration_seconds=45.0
    )
    assert finish_res["state"] == SpeechState.FEEDBACK.value
    assert "SPEECH REVIEW" in finish_res["reply_text"]
    assert finish_res["analysis"] is not None


# ---------------------------------------------------------------------------
# Test 8: Automatic timeout
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_08_automatic_timeout():
    session = speech_service.get_or_create_session(student_id=PROTOTYPE_STUDENT_ID, force_new=True)
    sid = session["session_id"]

    update_speech_session(session["id"], {
        "topic": "Internet of Things",
        "time_limit_seconds": 60,
        "required_points": ["Sensors", "Connectivity", "Smart homes"],
        "state": SpeechState.RECORDING.value
    })

    # Speech delivered at or exceeding allowed time (65s on 60s limit)
    res = await speech_service.process_speech_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sid,
        user_text="The Internet of Things connects everyday devices. Sensors collect temperature and motion data. Connectivity allows smart homes to control lights automatically.",
        duration_seconds=65.0
    )
    assert res["state"] == SpeechState.FEEDBACK.value
    timing = res["analysis"]["timing"]
    assert timing["status"] in ["within_time", "slightly_over", "within time", "slightly over"]


# ---------------------------------------------------------------------------
# Test 9: Empty transcript
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_09_empty_transcript():
    session = speech_service.get_or_create_session(student_id=PROTOTYPE_STUDENT_ID, force_new=True)
    sid = session["session_id"]

    update_speech_session(session["id"], {
        "topic": "Quantum Computing",
        "time_limit_seconds": 120,
        "required_points": ["Qubits", "Superposition"],
        "state": SpeechState.RECORDING.value
    })

    # Near empty / inaudible transcript
    res = await speech_service.process_speech_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sid,
        user_text="um uh",
        duration_seconds=10.0
    )
    assert res["state"] in [SpeechState.READY.value, SpeechState.RECORDING.value]
    assert "couldn't get enough speech" in res["reply_text"].lower()
    # Ensure no fake feedback was generated
    assert "SPEECH REVIEW" not in res["reply_text"]


# ---------------------------------------------------------------------------
# Test 10: Full transcript
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_10_full_transcript():
    speech_text = (
        "Good morning everyone. Today I will discuss Artificial Intelligence in Healthcare. "
        "First, AI in healthcare refers to algorithms that analyze complex medical data. "
        "Second, applications include radiology imaging and robotic surgery. "
        "Third, benefits include faster diagnoses and reduced medical errors. "
        "However, risks involve data privacy and algorithmic bias. "
        "Finally, future scope will see autonomous clinics. In conclusion, AI transforms healthcare."
    )
    analysis = await speech_service.analyze_speech_attempt(
        topic="Artificial Intelligence in Healthcare",
        required_points=["Definition", "Applications", "Benefits", "Risks", "Future scope"],
        time_limit_seconds=180,
        actual_duration_seconds=120.0,
        transcript=speech_text
    )
    assert analysis is not None
    assert len(analysis["coverage"]) == 5
    assert analysis["structure"]["introduction"] != ""
    assert analysis["structure"]["conclusion"] != ""
    assert analysis["timing"]["actual_seconds"] == 120.0


# ---------------------------------------------------------------------------
# Test 11: Covered points
# ---------------------------------------------------------------------------
def test_11_covered_points():
    transcript = "Robotics definition is automated mechanical systems. Modern applications include manufacturing, logistics, and healthcare."
    cov = speech_service.evaluate_point_coverage(
        required_points=["Definition", "Applications"],
        transcript=transcript
    )
    assert cov[0]["status"] == "covered"
    assert cov[1]["status"] == "covered"
    assert cov[0]["evidence"] != ""


# ---------------------------------------------------------------------------
# Test 12: Partially covered points
# ---------------------------------------------------------------------------
def test_12_partially_covered_points():
    transcript = "We must briefly consider the future scope when thinking about automation."
    cov = speech_service.evaluate_point_coverage(
        required_points=["Future scope"],
        transcript=transcript
    )
    # Mentioned superficially
    assert cov[0]["status"] in ["covered", "partial"]


# ---------------------------------------------------------------------------
# Test 13: Missing points
# ---------------------------------------------------------------------------
def test_13_missing_points():
    transcript = "I will only talk about definition and advantages today. Definition is smart code. Advantages are high speed."
    cov = speech_service.evaluate_point_coverage(
        required_points=["Definition", "Advantages", "Risks and Safety", "Deep Space Missions"],
        transcript=transcript
    )
    cov_dict = {c["point"]: c["status"] for c in cov}
    assert cov_dict["Definition"] == "covered"
    assert cov_dict["Advantages"] == "covered"
    assert cov_dict["Risks and Safety"] == "missing"
    assert cov_dict["Deep Space Missions"] == "missing"


# ---------------------------------------------------------------------------
# Test 14: Grammar issue detection
# ---------------------------------------------------------------------------
def test_14_grammar_issue_detection():
    transcript = "AI have many applications. He do not understand the system. They is working on new models."
    issues = speech_service.detect_grammar_issues(transcript)
    assert len(issues) >= 2
    err_texts = [i["original"].lower() for i in issues]
    assert any("ai have" in e or "he do" in e or "they is" in e for e in err_texts)


# ---------------------------------------------------------------------------
# Test 15: Timing analysis
# ---------------------------------------------------------------------------
def test_15_timing_analysis():
    t_short = speech_service.evaluate_timing(actual_seconds=30.0, allowed_seconds=180.0, finished_intentionally=True)
    assert t_short["status"] == "too_short"

    t_good = speech_service.evaluate_timing(actual_seconds=170.0, allowed_seconds=180.0)
    assert t_good["status"] == "within_time"

    t_slight = speech_service.evaluate_timing(actual_seconds=195.0, allowed_seconds=180.0)
    assert t_slight["status"] == "slightly_over"

    t_over = speech_service.evaluate_timing(actual_seconds=250.0, allowed_seconds=180.0)
    assert t_over["status"] == "significantly_over"


# ---------------------------------------------------------------------------
# Test 16: Retry same topic
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_16_retry_same_topic():
    session = speech_service.get_or_create_session(student_id=PROROT_ID if 'PROROT_ID' in locals() else PROTOTYPE_STUDENT_ID, force_new=True)
    sid = session["session_id"]

    # Put session into FEEDBACK state
    update_speech_session(session["id"], {
        "topic": "Machine Learning",
        "time_limit_seconds": 180,
        "required_points": ["Supervised", "Unsupervised", "Reinforcement"],
        "state": SpeechState.FEEDBACK.value,
        "attempt_number": 1,
        "actual_duration_seconds": 150.0,
        "grammar_issue_count": 3
    })

    # User asks to try again
    res = await speech_service.process_speech_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sid,
        user_text="try again"
    )
    assert res["state"] == SpeechState.READY.value
    assert "Attempt #2" in res["reply_text"] or "Attempt 2" in res["reply_text"]
    assert "Machine Learning" in res["reply_text"]
    assert "start" in res["reply_text"].lower()


# ---------------------------------------------------------------------------
# Test 17: New topic
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_17_new_topic():
    session = speech_service.get_or_create_session(student_id=PROTOTYPE_STUDENT_ID, force_new=True)
    sid = session["session_id"]

    update_speech_session(session["id"], {
        "topic": "Old Topic",
        "state": SpeechState.FEEDBACK.value
    })

    res = await speech_service.process_speech_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sid,
        user_text="new topic"
    )
    assert res["state"] == SpeechState.SETUP.value
    assert "What is your topic?" in res["reply_text"]


# ---------------------------------------------------------------------------
# Test 18: Stop/cancel session
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_18_stop_cancel_session():
    session = speech_service.get_or_create_session(student_id=PROTOTYPE_STUDENT_ID, force_new=True)
    sid = session["session_id"]

    res = await speech_service.process_speech_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sid,
        user_text="cancel session"
    )
    assert res["state"] == SpeechState.SETUP.value
    assert "cancelled" in res["reply_text"].lower() or "reset" in res["reply_text"].lower()


# ---------------------------------------------------------------------------
# Test 19: Invalid state transition
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_19_invalid_state_transition():
    session = speech_service.get_or_create_session(student_id=PROTOTYPE_STUDENT_ID, force_new=True)
    sid = session["session_id"]

    # Clear setup fields
    update_speech_session(session["id"], {
        "topic": "",
        "required_points": [],
        "time_limit_seconds": 0,
        "state": SpeechState.SETUP.value
    })

    # Trying to say "start" without completing setup
    res = await speech_service.process_speech_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sid,
        user_text="start"
    )
    assert res["state"] == SpeechState.SETUP.value
    assert "first" in res["reply_text"].lower() or "need" in res["reply_text"].lower() or "topic" in res["reply_text"].lower()


# ---------------------------------------------------------------------------
# Test 20: Routing to speech agent
# ---------------------------------------------------------------------------
def test_20_routing_to_speech_agent():
    # 1. Test intent detection routes to "speech"
    mode1 = personalization_service.detect_intent_and_mode("I want to practice a seminar.")
    assert mode1 == "speech"

    mode2 = personalization_service.detect_intent_and_mode("I want speech practice for my presentation")
    assert mode2 == "speech"

    # 2. Test that normal conversation still routes to "coach"
    mode3 = personalization_service.detect_intent_and_mode("I like watching anime on weekends", current_mode="coach")
    assert mode3 == "coach"

    # 3. Test that tutor still routes to "tutor"
    mode4 = personalization_service.detect_intent_and_mode("Teach me about photosynthesis in biology")
    assert mode4 == "tutor"

    # 4. Test API endpoint routing to speech mode
    update_student(PROTOTYPE_STUDENT_ID, {"onboarding_completed": True, "onboarding_step": "completed"})
    from app.services.voice_service import voice_manager, VoiceState
    voice_manager.transition_to(VoiceState.READY, "test_speech_routing")
    res = client.post(
        "/api/v1/esp32/chat",
        json={
            "student_id": 1,
            "message": "I want to practice a seminar.",
            "mode": "coach",
            "is_typed_text": True
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["learning_mode"] == "speech"
    assert "topic" in data["reply_text"].lower()
