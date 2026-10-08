import pytest
import asyncio
from fastapi.testclient import TestClient

from app.main import create_app
from app.db.database import (
    create_subject,
    get_subject,
    get_subject_by_name,
    list_all_subjects,
    add_syllabus_unit,
    add_syllabus_topic,
    get_syllabus_units,
    get_syllabus_topics,
    get_all_topics_for_subject,
    get_entire_subject_hierarchy,
    load_complete_syllabus,
    get_or_create_student_subject_progress,
    update_student_subject_progress,
    record_topic_completed,
    record_topic_struggle,
    record_topic_understood,
    PROTOTYPE_STUDENT_ID,
    get_student,
    update_student
)
from app.services.tutor_service import tutor_service
from app.services.llm_service import llm_service


# Standard Test Syllabus fixture based on Section 16 specification
SAMPLE_ROBOTICS_SYLLABUS = {
    "name": "Robotics",
    "description": "Introduction to robotics kinematics and geometry.",
    "units": [
        {
            "unit_number": 1,
            "title": "Unit 1: Robot Kinematics",
            "description": "Spatial descriptions, transformations, and arm geometry.",
            "topics": [
                {
                    "topic_name": "Forward Kinematics",
                    "summary": "Forward kinematics calculates the position and orientation of the robot's end-effector from known joint angles.",
                    "content": "Forward kinematics means finding the position and orientation of a robot's end-effector from joint angles and link parameters. For example, knowing the angles of shoulder and elbow joints allows us to compute where the robotic hand is in 3D space.",
                    "simple_explanation": "Forward kinematics means finding where the robot's hand is when we know the angles of its joints.",
                    "examples": [
                        "If we know the angles of a robot arm's joints, we can calculate where its hand will be in space."
                    ],
                    "key_terms": [
                        {
                            "term": "Forward Kinematics",
                            "meaning": "It means finding where the robot's hand is when we know the angles of its joints."
                        }
                    ],
                    "sample_questions": [
                        {
                            "question": "What does forward kinematics calculate for a robot arm?",
                            "expected_keywords": ["position", "hand", "end", "effector", "angles", "joints", "where"],
                            "answer": "Forward kinematics calculates the position of the robot hand using its joint angles."
                        }
                    ]
                },
                {
                    "topic_name": "Inverse Kinematics",
                    "summary": "Inverse kinematics calculates the required joint angles to place the robot's end-effector at a desired target position.",
                    "content": "Inverse kinematics solves the reverse problem: given a target position for the robot hand, what angles should the joints take?",
                    "simple_explanation": "Inverse kinematics means finding what joint angles are needed so the robot's hand can reach a target point.",
                    "examples": [
                        "If you want a robot hand to pick up a cup on a table, inverse kinematics tells the motor joints which angles to turn."
                    ],
                    "key_terms": [
                        {
                            "term": "Inverse Kinematics",
                            "meaning": "Finding the joint angles needed to place the robot hand at a target spot."
                        }
                    ],
                    "sample_questions": [
                        {
                            "question": "What is the difference between forward and inverse kinematics?",
                            "expected_keywords": ["angles", "position", "reverse", "target"],
                            "answer": "Forward kinematics finds hand position from angles, while inverse kinematics finds angles needed for a target position."
                        }
                    ]
                },
                {
                    "topic_name": "DH Parameters",
                    "summary": "Denavit-Hartenberg (DH) parameters are four parameters associated with a particular convention for attaching reference frames to the links of a spatial kinematic chain.",
                    "content": "DH parameters are four numbers used to describe how consecutive robot links and joints are connected.",
                    "simple_explanation": "DH parameters are four numbers that describe the shape and connection between robot joints.",
                    "examples": [
                        "They describe link length, link twist, link offset, and joint angle for each segment of the arm."
                    ],
                    "key_terms": [
                        {
                            "term": "DH Parameters",
                            "meaning": "Four standard numbers that describe how robot joints and links connect."
                        }
                    ],
                    "sample_questions": [
                        {
                            "question": "How many parameters are used in DH parameters convention?",
                            "expected_keywords": ["four", "4"],
                            "answer": "Four parameters: link length, link twist, link offset, and joint angle."
                        }
                    ]
                }
            ]
        }
    ]
}


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def setup_robotics_syllabus():
    """Ensure Robotics syllabus is loaded and student progress is clean."""
    tutor_service.clear_active_quizzes()
    sub_id = load_complete_syllabus(SAMPLE_ROBOTICS_SYLLABUS)
    # Reset student subject progress for clean test state
    update_student_subject_progress(PROTOTYPE_STUDENT_ID, sub_id, {
        "completed_topics": [],
        "weak_topics": [],
        "revision_topics": [],
        "questions_asked": 0
    })
    # Set completed onboarding and default teaching difficulty
    update_student(PROTOTYPE_STUDENT_ID, {
        "name": "Shahid",
        "onboarding_completed": 1,
        "onboarding_step": "completed",
        "current_teaching_difficulty": 1
    })
    yield sub_id


# -------------------------------------------------------------
# Test 1: Subject can be loaded
# -------------------------------------------------------------
def test_1_subject_can_be_loaded():
    sub = get_subject_by_name("Robotics")
    assert sub is not None
    assert sub["name"] == "Robotics"
    assert "kinematics" in sub["description"].lower()


# -------------------------------------------------------------
# Test 2: Syllabus can be loaded
# -------------------------------------------------------------
def test_2_syllabus_can_be_loaded():
    sub = get_subject_by_name("Robotics")
    assert sub is not None
    hierarchy = get_entire_subject_hierarchy(sub["id"])
    assert hierarchy is not None
    assert len(hierarchy["units"]) >= 1
    unit1 = hierarchy["units"][0]
    assert "Unit 1" in unit1["title"]
    topics = unit1["topics"]
    assert len(topics) == 3
    topic_names = [t["topic_name"] for t in topics]
    assert "Forward Kinematics" in topic_names
    assert "Inverse Kinematics" in topic_names
    assert "DH Parameters" in topic_names


# -------------------------------------------------------------
# Test 3: Chapter/topic can be selected
# -------------------------------------------------------------
def test_3_topic_can_be_selected():
    sub = get_subject_by_name("Robotics")
    sub_res, top_res = tutor_service.set_active_subject_and_topic(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id="test_session_tutor",
        subject_name_or_id="Robotics",
        topic_name_or_id="Forward Kinematics"
    )
    assert sub_res is not None
    assert sub_res["name"] == "Robotics"
    assert top_res is not None
    assert top_res["topic_name"] == "Forward Kinematics"

    # Verify retrieval
    curr_sub, curr_top = tutor_service.get_current_subject_and_topic(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id="test_session_tutor"
    )
    assert curr_sub["name"] == "Robotics"
    assert curr_top["topic_name"] == "Forward Kinematics"


# -------------------------------------------------------------
# Test 4: Tutor stays within syllabus
# -------------------------------------------------------------
def test_4_tutor_stays_within_syllabus():
    sub = get_subject_by_name("Robotics")
    in_bounds, matched_t, out_topic = tutor_service.check_syllabus_boundary(sub, "What is forward kinematics?")
    assert in_bounds is True
    assert out_topic is None
    assert matched_t is not None
    assert matched_t["topic_name"] == "Forward Kinematics"


# -------------------------------------------------------------
# Test 5: Out-of-syllabus question is rejected politely
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_5_out_of_syllabus_question_rejected_politely():
    sub = get_subject_by_name("Robotics")
    tutor_service.set_active_subject_and_topic(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id="test_sess_5",
        subject_name_or_id=sub["id"],
        topic_name_or_id="Forward Kinematics"
    )

    res = await tutor_service.process_tutor_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id="test_sess_5",
        user_text="Teach me reinforcement learning."
    )
    assert res["intent"] == "OUT_OF_SYLLABUS"
    # Must politely reject and mention the topic is not in the material
    assert "reinforcement learning" in res["reply_text"].lower()
    assert "current syllabus/material" in res["reply_text"].lower()
    assert "would you like to add" in res["reply_text"].lower() or "please provide that chapter" in res["reply_text"].lower()


# -------------------------------------------------------------
# Test 6: Simple explanation works
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_6_simple_explanation_works():
    sub = get_subject_by_name("Robotics")
    tutor_service.set_active_subject_and_topic(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id="test_sess_6",
        subject_name_or_id=sub["id"],
        topic_name_or_id="Forward Kinematics"
    )

    res = await tutor_service.process_tutor_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id="test_sess_6",
        user_text="What is forward kinematics?"
    )
    assert res["intent"] in ["TEACH", "ASK_QUESTION"]
    # Check that explanation contains simple meaning and practical example
    text = res["reply_text"].lower()
    assert "forward kinematics" in text
    assert "hand" in text or "joint" in text or "angles" in text
    assert "simple" in text or "example" in text


# -------------------------------------------------------------
# Test 7: "Explain simply" lowers difficulty
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_7_explain_simply_lowers_difficulty():
    sub = get_subject_by_name("Robotics")
    tutor_service.set_active_subject_and_topic(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id="test_sess_7",
        subject_name_or_id=sub["id"],
        topic_name_or_id="Forward Kinematics"
    )

    res = await tutor_service.process_tutor_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id="test_sess_7",
        user_text="Explain it very simply.",
        current_difficulty=3
    )
    assert res["intent"] == "EXPLAIN_SIMPLY"
    assert res["new_difficulty"] == 1
    assert "super simple" in res["reply_text"].lower() or "step by step" in res["reply_text"].lower()


# -------------------------------------------------------------
# Test 8: "I don't understand" triggers simplification
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_8_i_dont_understand_triggers_simplification():
    sub = get_subject_by_name("Robotics")
    tutor_service.set_active_subject_and_topic(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id="test_sess_8",
        subject_name_or_id=sub["id"],
        topic_name_or_id="Forward Kinematics"
    )

    res = await tutor_service.process_tutor_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id="test_sess_8",
        user_text="I don't understand.",
        current_difficulty=2
    )
    assert res["intent"] == "EXPLAIN_SIMPLY"
    assert res["new_difficulty"] == 1
    assert len(res["reply_text"].split("\n")) > 1


# -------------------------------------------------------------
# Test 9: User can request an example
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_9_user_can_request_example():
    sub = get_subject_by_name("Robotics")
    tutor_service.set_active_subject_and_topic(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id="test_sess_9",
        subject_name_or_id=sub["id"],
        topic_name_or_id="Forward Kinematics"
    )

    res = await tutor_service.process_tutor_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id="test_sess_9",
        user_text="Give me an example."
    )
    assert res["intent"] == "GIVE_EXAMPLE"
    assert "example" in res["reply_text"].lower()
    assert "robot" in res["reply_text"].lower()


# -------------------------------------------------------------
# Test 10: User can request a quiz
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_10_user_can_request_quiz():
    sub = get_subject_by_name("Robotics")
    tutor_service.set_active_subject_and_topic(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id="test_sess_10",
        subject_name_or_id=sub["id"],
        topic_name_or_id="Forward Kinematics"
    )

    res = await tutor_service.process_tutor_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id="test_sess_10",
        user_text="Quiz me."
    )
    assert res["intent"] == "QUIZ"
    assert "?" in res["reply_text"]
    assert "Forward Kinematics" in res["reply_text"]


# -------------------------------------------------------------
# Test 11: User can request revision
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_11_user_can_request_revision():
    sub = get_subject_by_name("Robotics")
    tutor_service.set_active_subject_and_topic(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id="test_sess_11",
        subject_name_or_id=sub["id"],
        topic_name_or_id="Forward Kinematics"
    )

    res = await tutor_service.process_tutor_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id="test_sess_11",
        user_text="Revise this."
    )
    assert res["intent"] == "REVISE"
    assert "revision" in res["reply_text"].lower()


# -------------------------------------------------------------
# Test 12: Tutor tracks current topic
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_12_tutor_tracks_current_topic():
    sub = get_subject_by_name("Robotics")
    sess_id = "test_sess_12"
    tutor_service.set_active_subject_and_topic(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sess_id,
        subject_name_or_id=sub["id"],
        topic_name_or_id="Forward Kinematics"
    )

    curr_sub, curr_top = tutor_service.get_current_subject_and_topic(PROTOTYPE_STUDENT_ID, sess_id)
    assert curr_top["topic_name"] == "Forward Kinematics"

    # Advance to next topic
    res = await tutor_service.process_tutor_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sess_id,
        user_text="Move to the next topic."
    )
    assert res["intent"] == "NEXT_TOPIC"
    assert res["current_topic"] == "Inverse Kinematics"

    curr_sub, curr_top = tutor_service.get_current_subject_and_topic(PROTOTYPE_STUDENT_ID, sess_id)
    assert curr_top["topic_name"] == "Inverse Kinematics"


# -------------------------------------------------------------
# Test 13: Tutor remembers weak topics
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_13_tutor_remembers_weak_topics():
    sub = get_subject_by_name("Robotics")
    sess_id = "test_sess_13"
    tutor_service.set_active_subject_and_topic(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sess_id,
        subject_name_or_id=sub["id"],
        topic_name_or_id="Forward Kinematics"
    )

    # 1. Ask for quiz
    await tutor_service.process_tutor_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sess_id,
        user_text="Quiz me."
    )

    # 2. Give an incorrect answer
    res = await tutor_service.process_tutor_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sess_id,
        user_text="It measures the color and temperature of the robot battery."
    )
    assert res["intent"] == "QUIZ_RESULT"
    assert "not quite" in res["reply_text"].lower()

    # 3. Verify weak_topics and revision_topics in database
    prog = get_or_create_student_subject_progress(PROTOTYPE_STUDENT_ID, sub["id"])
    assert "Forward Kinematics" in prog["weak_topics"]
    assert "Forward Kinematics" in prog["revision_topics"]
    assert prog["questions_asked"] >= 1


# -------------------------------------------------------------
# Test 14: Simple questions do not use Groq
# -------------------------------------------------------------
def test_14_simple_questions_do_not_use_groq():
    test_queries = [
        "What is forward kinematics?",
        "Explain it simply.",
        "What is DH parameters?",
        "Give me an example of kinematics.",
        "What is a robot arm?"
    ]
    for q in test_queries:
        messages = [{"role": "user", "content": q}]
        intent, complexity, primary, cascade = llm_service.classify_route(messages)
        assert primary != "groq", f"Simple query '{q}' should not route primary to groq! Got {primary}"
        assert primary == "ollama", f"Simple query '{q}' must route primary to local Ollama! Got {primary}"


# -------------------------------------------------------------
# Test 15: Tutor does not inject unrelated personal interests
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_15_tutor_does_not_inject_unrelated_personal_interests():
    # Update student profile to have personal interests like anime, cooking
    update_student(PROTOTYPE_STUDENT_ID, {
        "interests": "Anime, Naruto, Cooking, Italian cuisine",
        "favorite_things": '{"favorite_anime": "Naruto", "favorite_food": "Pasta"}'
    })

    sub = get_subject_by_name("Robotics")
    sess_id = "test_sess_15"
    tutor_service.set_active_subject_and_topic(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sess_id,
        subject_name_or_id=sub["id"],
        topic_name_or_id="Forward Kinematics"
    )

    res = await tutor_service.process_tutor_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sess_id,
        user_text="What is forward kinematics?"
    )
    reply = res["reply_text"].lower()
    # Must NOT mention Naruto, Anime, Cooking, Pasta in the robotics explanation
    assert "naruto" not in reply
    assert "anime" not in reply
    assert "pasta" not in reply
    assert "cooking" not in reply


# -------------------------------------------------------------
# Test 16: Tutor does not invent missing syllabus content
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_16_tutor_does_not_invent_missing_syllabus_content():
    sub = get_subject_by_name("Robotics")
    sess_id = "test_sess_16"
    tutor_service.set_active_subject_and_topic(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sess_id,
        subject_name_or_id=sub["id"],
        topic_name_or_id="Forward Kinematics"
    )

    # Ask about a topic not in the syllabus (e.g. quantum entanglement)
    res = await tutor_service.process_tutor_turn(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=sess_id,
        user_text="What is quantum entanglement?"
    )
    assert res["intent"] == "OUT_OF_SYLLABUS"
    # Does not invent a physics lecture about qubits
    assert "isn't in the current syllabus/material" in res["reply_text"] or "i don't have that topic" in res["reply_text"].lower()
    assert "qubit" not in res["reply_text"].lower()


# -------------------------------------------------------------
# Section 16: Complete End-to-End Test Sequence
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_section_16_complete_end_to_end_sequence(client):
    """
    Step 1. Select Robotics.
    Step 2. Select Forward Kinematics.
    Step 3. Ask: 'What is forward kinematics?'
    Step 4. Ask: 'Explain it very simply.'
    Step 5. Ask: 'Give me an example.'
    Step 6. Ask: 'Quiz me.'
    Step 7. Give an incorrect answer.
    Step 8. Verify correction.
    Step 9. Ask: 'Teach me reinforcement learning.'
    Step 10. Verify that the tutor does NOT teach it because it is outside the supplied syllabus.
    """
    session_id = "e2e_tutor_session_robotics"

    # Step 1: Select Robotics
    res1 = client.post(
        "/api/v1/esp32/chat",
        json={
            "student_id": PROTOTYPE_STUDENT_ID,
            "session_id": session_id,
            "message": "Select Robotics",
            "mode": "tutor",
            "is_typed_text": True
        }
    )
    assert res1.status_code == 200
    data1 = res1.json()
    assert "Robotics" in data1["reply_text"]

    # Step 2: Select Forward Kinematics
    res2 = client.post(
        "/api/v1/esp32/chat",
        json={
            "student_id": PROTOTYPE_STUDENT_ID,
            "session_id": session_id,
            "message": "Select Forward Kinematics",
            "mode": "tutor",
            "is_typed_text": True
        }
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert "Forward Kinematics" in data2["reply_text"]

    # Step 3: Ask: "What is forward kinematics?"
    res3 = client.post(
        "/api/v1/esp32/chat",
        json={
            "student_id": PROTOTYPE_STUDENT_ID,
            "session_id": session_id,
            "message": "What is forward kinematics?",
            "mode": "tutor",
            "is_typed_text": True
        }
    )
    assert res3.status_code == 200
    data3 = res3.json()
    assert "hand" in data3["reply_text"].lower() or "angles" in data3["reply_text"].lower()

    # Step 4: Ask: "Explain it very simply."
    res4 = client.post(
        "/api/v1/esp32/chat",
        json={
            "student_id": PROTOTYPE_STUDENT_ID,
            "session_id": session_id,
            "message": "Explain it very simply.",
            "mode": "tutor",
            "is_typed_text": True
        }
    )
    assert res4.status_code == 200
    data4 = res4.json()
    assert "simple" in data4["reply_text"].lower()

    # Step 5: Ask: "Give me an example."
    res5 = client.post(
        "/api/v1/esp32/chat",
        json={
            "student_id": PROTOTYPE_STUDENT_ID,
            "session_id": session_id,
            "message": "Give me an example.",
            "mode": "tutor",
            "is_typed_text": True
        }
    )
    assert res5.status_code == 200
    data5 = res5.json()
    assert "example" in data5["reply_text"].lower()

    # Step 6: Ask: "Quiz me."
    res6 = client.post(
        "/api/v1/esp32/chat",
        json={
            "student_id": PROTOTYPE_STUDENT_ID,
            "session_id": session_id,
            "message": "Quiz me.",
            "mode": "tutor",
            "is_typed_text": True
        }
    )
    assert res6.status_code == 200
    data6 = res6.json()
    assert "?" in data6["reply_text"]

    # Step 7: Give an incorrect answer
    # Step 8: Verify correction
    res7 = client.post(
        "/api/v1/esp32/chat",
        json={
            "student_id": PROTOTYPE_STUDENT_ID,
            "session_id": session_id,
            "message": "It measures how hot the robot motor is.",
            "mode": "tutor",
            "is_typed_text": True
        }
    )
    assert res7.status_code == 200
    data7 = res7.json()
    assert "not quite" in data7["reply_text"].lower()
    assert "revision" in data7["reply_text"].lower() or "hand" in data7["reply_text"].lower()

    # Step 9: Ask: "Teach me reinforcement learning."
    # Step 10: Verify tutor does NOT teach it because outside supplied syllabus
    res9 = client.post(
        "/api/v1/esp32/chat",
        json={
            "student_id": PROTOTYPE_STUDENT_ID,
            "session_id": session_id,
            "message": "Teach me reinforcement learning.",
            "mode": "tutor",
            "is_typed_text": True
        }
    )
    assert res9.status_code == 200
    data9 = res9.json()
    reply9 = data9["reply_text"].lower()
    assert "reinforcement learning" in reply9
    assert "current syllabus/material" in reply9 or "isn't in the current" in reply9
    # Must NOT teach Q-learning, rewards, or policy gradients
    assert "q-learning" not in reply9
    assert "policy gradient" not in reply9
    assert "reward function" not in reply9
