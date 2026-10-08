import pytest
from app.db.database import (
    get_student,
    update_student,
    create_session,
    load_complete_syllabus,
    PROTOTYPE_STUDENT_ID
)
from app.services.agent_router import agent_router, AgentRoute
from app.services.onboarding_service import onboarding_service
from app.services.normal_chat_service import normal_chat_service
from app.services.tutor_service import tutor_service
from app.services.speech_service import speech_service


# ---------------------------------------------------------------------------
# 1. New user -> onboarding active
# ---------------------------------------------------------------------------
def test_01_new_user_onboarding_active():
    update_student(PROTOTYPE_STUDENT_ID, {"name": "Newbie", "onboarding_completed": False, "onboarding_step": "name"})
    is_onboarded = onboarding_service.is_onboarded(PROTOTYPE_STUDENT_ID)
    assert not is_onboarded, "New student must start with onboarding incomplete"
    st = get_student(PROTOTYPE_STUDENT_ID)
    assert st["onboarding_step"] == "name"


# ---------------------------------------------------------------------------
# 2. Completed onboarding -> main chat active
# ---------------------------------------------------------------------------
def test_02_completed_onboarding_main_chat_active():
    update_student(PROTOTYPE_STUDENT_ID, {
        "name": "Shahid",
        "native_language": "Hindi",
        "english_level": "Intermediate",
        "interests": "Anime, Robotics",
        "onboarding_completed": True,
        "onboarding_step": "completed"
    })
    is_onboarded = onboarding_service.is_onboarded(PROTOTYPE_STUDENT_ID)
    assert is_onboarded, "Onboarding must be marked complete"
    profile = onboarding_service.get_learner_profile(PROTOTYPE_STUDENT_ID)
    assert profile["onboarding_completed"] is True


# ---------------------------------------------------------------------------
# 3. Normal intent -> Normal Chat
# ---------------------------------------------------------------------------
def test_03_normal_intent_routes_to_normal_chat():
    for utterance in [
        "Let's talk about anime.",
        "I want to improve my English.",
        "Can we discuss my day?",
        "Hey Mizo, let's talk about One Piece."
    ]:
        res = agent_router.route_intent(utterance)
        assert res["route"] == AgentRoute.NORMAL_CHAT.value, f"Failed for: {utterance}"


# ---------------------------------------------------------------------------
# 4. Study intent -> Study section
# ---------------------------------------------------------------------------
def test_04_study_intent_routes_to_study():
    for utterance in [
        "I uploaded my notes.",
        "Teach me this chapter.",
        "I want to prepare for my exam from these notes.",
        "I have my syllabus and want to study.",
        "Teach me Forward Kinematics.",
        "I want to study for my robotics exam. I have notes."
    ]:
        res = agent_router.route_intent(utterance)
        assert res["route"] == AgentRoute.STUDY.value, f"Failed for: {utterance}"


# ---------------------------------------------------------------------------
# 5. Speech intent -> Speech section
# ---------------------------------------------------------------------------
def test_05_speech_intent_routes_to_speech():
    for utterance in [
        "I want to practice my presentation.",
        "I have a seminar tomorrow and want to practice.",
        "Listen to my speech and give me feedback.",
        "practice my speech",
        "seminar practice",
        "presentation rehearsal"
    ]:
        res = agent_router.route_intent(utterance)
        assert res["route"] == AgentRoute.SPEECH.value, f"Failed for: {utterance}"


# ---------------------------------------------------------------------------
# 6. Normal -> Study transition (Immediate intent switch, no lag)
# ---------------------------------------------------------------------------
def test_06_normal_to_study_transition():
    r1 = agent_router.route_intent("Let's talk about anime.")
    assert r1["route"] == AgentRoute.NORMAL_CHAT.value

    r2 = agent_router.route_intent("I want to study my syllabus.")
    assert r2["route"] == AgentRoute.STUDY.value


# ---------------------------------------------------------------------------
# 7. Study -> Normal transition (Router does not hold onto tutor topic)
# ---------------------------------------------------------------------------
def test_07_study_to_normal_transition():
    r1 = agent_router.route_intent("Teach me Forward Kinematics from my syllabus.")
    assert r1["route"] == AgentRoute.STUDY.value

    r2 = agent_router.route_intent("I want to talk about One Piece.")
    assert r2["route"] == AgentRoute.NORMAL_CHAT.value


# ---------------------------------------------------------------------------
# 8. Normal -> Speech transition
# ---------------------------------------------------------------------------
def test_08_normal_to_speech_transition():
    r1 = agent_router.route_intent("How are you today Mizo?")
    assert r1["route"] == AgentRoute.NORMAL_CHAT.value

    r2 = agent_router.route_intent("I want to practice my seminar presentation.")
    assert r2["route"] == AgentRoute.SPEECH.value


# ---------------------------------------------------------------------------
# 9. Study -> Speech transition
# ---------------------------------------------------------------------------
def test_09_study_to_speech_transition():
    r1 = agent_router.route_intent("Quiz me on robotics.")
    assert r1["route"] == AgentRoute.STUDY.value

    r2 = agent_router.route_intent("Now listen to my speech rehearsal.")
    assert r2["route"] == AgentRoute.SPEECH.value


# ---------------------------------------------------------------------------
# 10. Speech -> Study transition
# ---------------------------------------------------------------------------
def test_10_speech_to_study_transition():
    r1 = agent_router.route_intent("Give me feedback on my seminar.")
    assert r1["route"] == AgentRoute.SPEECH.value

    r2 = agent_router.route_intent("Let's go back and study chapter 2 notes.")
    assert r2["route"] == AgentRoute.STUDY.value


# ---------------------------------------------------------------------------
# 11. Agent-specific state isolation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_11_agent_state_isolation():
    student_id = PROTOTYPE_STUDENT_ID
    session = create_session(student_id=student_id, mode="coach")

    # Load syllabus and set active topic in Study Agent
    syllabus_data = {
        "subject_name": "Robotics",
        "description": "Robotics curriculum",
        "units": [
            {
                "unit_order": 1,
                "title": "Kinematics",
                "topics": [
                    {"topic_order": 1, "title": "Forward Kinematics", "content": "FK equations"}
                ]
            }
        ]
    }
    load_complete_syllabus(syllabus_data)
    tutor_service.set_active_subject_and_topic(
        student_id=student_id,
        session_id=session["session_id"],
        subject_name_or_id="Robotics",
        topic_name_or_id="Forward Kinematics"
    )

    # Verify Study Agent state
    study_state = tutor_service.get_tutor_session_state(student_id, session["session_id"])
    assert study_state["active_topic"] == "Forward Kinematics"

    # Process turn through Normal Chat Agent with user talking about anime
    res = await normal_chat_service.process_chat_turn(
        student_id=student_id,
        session_id=session["session_id"],
        user_text="Tell me what you think about Luffy from One Piece."
    )

    spoken = res["response"]
    # CRITICAL: Forward Kinematics must NOT appear in normal chat
    assert "Forward Kinematics" not in spoken
    assert "Kinematics" not in spoken
    # Normal chat agent has separate working context
    assert len(spoken) > 5


# ---------------------------------------------------------------------------
# 12. Shared profile memory (Name & profile accessible across agents)
# ---------------------------------------------------------------------------
def test_12_shared_profile_memory_accessible_across_agents():
    student_id = PROTOTYPE_STUDENT_ID
    update_student(
        student_id,
        {
            "name": "Shahid",
            "native_language": "Hindi",
            "english_level": "Intermediate",
            "interests": "Robotics, Anime",
            "onboarding_completed": True
        }
    )

    profile = onboarding_service.get_learner_profile(student_id)
    assert profile["name"] == "Shahid"
    assert "Robotics" in profile["interests"]

    # Verify tutor service can access student profile
    tutor_sub, tutor_top = tutor_service.get_current_subject_and_topic(student_id)
    assert student_id == profile["id"]


# ---------------------------------------------------------------------------
# 13. Name-change confirmation flow
# ---------------------------------------------------------------------------
def test_13_name_change_confirmation():
    student_id = PROTOTYPE_STUDENT_ID
    session_id = "test_sess_confirm"
    update_student(student_id, {"name": "Shahid", "onboarding_completed": True})

    # Step A: User requests name change
    cand = onboarding_service.detect_name_change_intent("Change my name to Migaza.")
    assert cand == "Migaza"

    prompt = onboarding_service.handle_name_change_flow(student_id, session_id, "Change my name to Migaza.")
    assert prompt == "Your current name is Shahid. You asked me to change it to Migaza. Should I replace your name and call you Migaza?"

    # Before confirmation, name in DB remains Shahid
    p = get_student(student_id)
    assert p["name"] == "Shahid"

    # Step B: User confirms
    confirm_reply = onboarding_service.handle_name_change_flow(student_id, session_id, "Yes, please replace it and call me Migaza.")
    assert "updated your name to Migaza" in confirm_reply

    # After confirmation, DB is updated
    p_after = get_student(student_id)
    assert p_after["name"] == "Migaza"


# ---------------------------------------------------------------------------
# 14. No stale tutor context on fresh study session reset
# ---------------------------------------------------------------------------
def test_14_no_stale_tutor_context():
    student_id = PROTOTYPE_STUDENT_ID
    syllabus_data = {
        "subject_name": "Robotics",
        "description": "Robotics curriculum",
        "units": [
            {
                "unit_order": 1,
                "title": "Kinematics",
                "topics": [
                    {"topic_order": 1, "title": "Inverse Kinematics", "content": "IK equations"}
                ]
            }
        ]
    }
    load_complete_syllabus(syllabus_data)
    tutor_service.set_active_subject_and_topic(
        student_id=student_id,
        session_id=None,
        subject_name_or_id="Robotics",
        topic_name_or_id="Inverse Kinematics"
    )
    assert tutor_service.get_tutor_session_state(student_id)["active_topic"] == "Inverse Kinematics"

    # Reset study session
    tutor_service.reset_study_session()
    assert tutor_service.get_tutor_session_state(student_id)["active_topic"] is None


# ---------------------------------------------------------------------------
# 15. No stale speech context
# ---------------------------------------------------------------------------
def test_15_no_stale_speech_context():
    student_id = PROTOTYPE_STUDENT_ID
    # Setup speech
    speech_service.setup_speech_session(
        student_id=student_id,
        topic="Microservices vs Monolith",
        required_points=["Scalability", "Deployment", "Latency"],
        allowed_duration_seconds=120
    )
    state = speech_service.get_active_session_state(student_id)
    assert state["topic"] == "Microservices vs Monolith"

    # Clear speech session
    speech_service.clear_speech_session(student_id)
    state_after = speech_service.get_active_session_state(student_id)
    assert state_after["topic"] is None
    assert state_after["required_points"] == []


# ---------------------------------------------------------------------------
# 16. No stale normal-chat topic
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_16_no_stale_normal_chat_topic():
    student_id = PROTOTYPE_STUDENT_ID
    session = create_session(student_id=student_id, mode="coach")

    # Normal chat on cooking
    await normal_chat_service.process_chat_turn(student_id, session["session_id"], "I love making pasta at home.")
    # Clear working context
    normal_chat_service.clear_working_context(student_id, session["session_id"])
    ctx = normal_chat_service.get_working_context(student_id, session["session_id"])
    assert ctx["recent_user_turns"] == []
    assert not ctx["current_topic"]



# ---------------------------------------------------------------------------
# 17. Ambiguous intent handling rule
# ---------------------------------------------------------------------------
def test_17_ambiguous_intent_handling():
    # Rule 12: "I want to practice."
    res = agent_router.route_intent("I want to practice.")
    assert res["route"] == AgentRoute.AMBIGUOUS.value
    assert res["is_ambiguous"] is True
    assert res["clarification_prompt"] == "Sure. Do you want to practice a conversation, study from your notes, or practice a speech/presentation?"

    # But obvious intent must NOT trigger ambiguity
    assert agent_router.route_intent("I want to practice my seminar presentation.")["route"] == AgentRoute.SPEECH.value
    assert agent_router.route_intent("I want to practice speaking English with you.")["route"] == AgentRoute.NORMAL_CHAT.value
