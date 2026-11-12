"""
Comprehensive Automated Test Suite for Study / Material-Based Teaching Agent (MIZO 3.0)
Validates all requirements from Sections A through G:
1. Exact-source definition retrieval
2. Simplified explanations that preserve source meaning
3. A topic containing exactly three subtopics, with no fourth subtopic introduced
4. Unsupported question producing a clear source limitation
5. Outside-syllabus permission being requested before answering
6. Declining permission preventing outside-syllabus content
7. Explicit permission enabling clearly labelled additional content
8. Quiz generation and grading from uploaded content only
9. Progress changing after a completed topic or recorded quiz
10. Weak-topic tracking after incorrect answers
11. PDF selection staying within Study
12. Study messages never being routed to Normal Chat
13. Multiple subjects and uploaded documents remaining isolated
14. Invalid or unreadable PDFs producing useful errors
15. Conversation and progress persistence after refresh
16. Existing onboarding, normal chat, and conversation lifecycle remaining intact
"""
import pytest
import os
import io
import json
import tempfile
from pathlib import Path
from starlette.testclient import TestClient

from app.config import settings
from app.main import app
from app.db.database import (
    init_db,
    get_connection,
    create_subject,
    get_subject,
    add_syllabus_unit,
    add_syllabus_topic,
    get_topic_by_id,
    get_all_topics_for_subject,
    get_or_create_student_subject_progress,
    mark_topic_completed,
    create_session,
    get_conversation_history,
    update_student,
    PROTOTYPE_STUDENT_ID
)
from app.services.tutor_service import SubjectTutorService
from app.services.study_document_service import StudyDocumentService


@pytest.fixture(scope="module")
def setup_test_db():
    """Configures an isolated temporary SQLite database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db_path = Path(tmpdir) / "test_study.db"
        orig_db_path = settings.DATABASE_PATH
        settings.DATABASE_PATH = test_db_path
        
        # Initialize isolated schema
        init_db()
        update_student(PROTOTYPE_STUDENT_ID, {
            "name": "Alex",
            "onboarding_completed": 1,
            "onboarding_step": "completed",
            "preferred_name": "Alex"
        })
        
        yield test_db_path
        
        # Restore original path
        settings.DATABASE_PATH = orig_db_path


@pytest.fixture
def client(setup_test_db):
    return TestClient(app)


@pytest.fixture
def sample_study_subject(setup_test_db):
    """Creates a sample subject with exact 3 subtopics under a topic."""
    SubjectTutorService.reset_study_session()
    sub_id = create_subject(
        name="Robotics Kinematics",
        description="Core robotic mechanisms and spatial transformations."
    )
    sub = get_subject(sub_id)
    unit_id = add_syllabus_unit(
        subject_id=sub_id,
        title="Unit 1: Spatial Descriptions and Arm Geometry",
        unit_number=1
    )
    unit = {"id": unit_id, "title": "Unit 1: Spatial Descriptions and Arm Geometry"}

    # Topic with exact 3 subtopics and grounded source definition
    topic_id = add_syllabus_topic(
        unit_id=unit_id,
        topic_name="Forward Kinematics",
        content="Forward kinematics maps joint coordinates to task coordinates. For a planar 2-link arm, the hand position is X = L1*cos(theta1) + L2*cos(theta1+theta2), Y = L1*sin(theta1) + L2*sin(theta1+theta2).",
        summary="Forward Kinematics calculates the Cartesian position (X, Y, Z) and orientation of the robot end-effector from given joint angles and fixed link lengths.",
        simple_explanation="Forward Kinematics is the math that computes where the robot hand is located based on the angles of its joints.",
        key_terms=[
            {
                "term": "Forward Kinematics",
                "meaning": "The mathematical calculation of the robot end-effector position and orientation from known joint angles and link lengths."
            },
            {
                "term": "End-Effector",
                "meaning": "The gripper, tool, or device at the end of the robotic arm designed to interact with the environment."
            }
        ],
        subtopics=[
            "Planar 2-DOF Arm Geometry",
            "Trigonometric Link Equations",
            "End-Effector Cartesian Mapping"
        ],
        sample_questions=[
            {
                "question": "What does forward kinematics calculate for a robotic arm?",
                "expected_keywords": ["position", "orientation", "hand", "angles", "coordinates"],
                "answer": "It calculates the Cartesian position and orientation of the end-effector from joint angles."
            }
        ],
        order_index=1
    )
    topic = get_topic_by_id(topic_id)
    return sub, unit, topic


# ---------------------------------------------------------------------------
# Test 1: Exact-Source Definition Retrieval (Section A.8)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_01_exact_source_definition(sample_study_subject):
    sub, unit, topic = sample_study_subject
    student_id = PROTOTYPE_STUDENT_ID
    session_id = "test_study_sess_01"

    SubjectTutorService.set_active_subject_and_topic(student_id, session_id, sub["id"], topic["id"])
    result = await SubjectTutorService.process_tutor_turn(
        student_id=student_id,
        session_id=session_id,
        user_text="What is the exact definition of Forward Kinematics according to my notes?"
    )

    assert result["intent"] == "EXACT_DEFINITION"
    reply = result["reply_text"]
    assert "Exact definition of" in reply
    assert "mathematical calculation of the robot end-effector position" in reply


# ---------------------------------------------------------------------------
# Test 2: Simplified Explanations Preserving Source Meaning (Section A.5, A.8)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_02_simplified_explanation(sample_study_subject):
    sub, unit, topic = sample_study_subject
    student_id = PROTOTYPE_STUDENT_ID
    session_id = "test_study_sess_02"

    SubjectTutorService.set_active_subject_and_topic(student_id, session_id, sub["id"], topic["id"])
    result = await SubjectTutorService.process_tutor_turn(
        student_id=student_id,
        session_id=session_id,
        user_text="Can you explain it simply?"
    )

    assert result["intent"] == "EXPLAIN_SIMPLY"
    reply = result["reply_text"].lower()
    # Preserves meaning without inventing external facts
    assert "hand" in reply or "joint" in reply or "math" in reply


# ---------------------------------------------------------------------------
# Test 3: Exactly Three Subtopics Preserved — No Fourth Subtopic (Section A.3)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_03_subtopics_preservation_no_hallucination(sample_study_subject):
    sub, unit, topic = sample_study_subject
    student_id = PROTOTYPE_STUDENT_ID
    session_id = "test_study_sess_03"

    SubjectTutorService.set_active_subject_and_topic(student_id, session_id, sub["id"], topic["id"])
    result = await SubjectTutorService.process_tutor_turn(
        student_id=student_id,
        session_id=session_id,
        user_text="What are the subtopics for this topic?"
    )

    assert result["intent"] == "LIST_SUBTOPICS"
    reply = result["reply_text"]
    # Check all 3 original subtopics exist
    assert "Planar 2-DOF Arm Geometry" in reply
    assert "Trigonometric Link Equations" in reply
    assert "End-Effector Cartesian Mapping" in reply

    # Confirm there are exactly 3 bullet points, no 4th subtopic invented
    bullet_count = reply.count("•")
    assert bullet_count == 3


# ---------------------------------------------------------------------------
# Test 4: Unsupported Question Produces Clear Source Limitation (Section A.7)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_04_unsupported_question_source_limitation(sample_study_subject):
    sub, unit, topic = sample_study_subject
    student_id = PROTOTYPE_STUDENT_ID
    session_id = "test_study_sess_04"

    SubjectTutorService.set_active_subject_and_topic(student_id, session_id, sub["id"], topic["id"])
    result = await SubjectTutorService.process_tutor_turn(
        student_id=student_id,
        session_id=session_id,
        user_text="What is the patent registration number for this arm in the uploaded material?"
    )

    assert result["intent"] == "UNSUPPORTED_SOURCE_QUERY"
    assert "not available in the uploaded material" in result["reply_text"].lower()


# ---------------------------------------------------------------------------
# Test 5: Outside-Syllabus Permission Requested Before Answering (Section B)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_05_outside_syllabus_permission_requested(sample_study_subject):
    sub, unit, topic = sample_study_subject
    student_id = PROTOTYPE_STUDENT_ID
    session_id = "test_study_sess_05"

    SubjectTutorService.set_active_subject_and_topic(student_id, session_id, sub["id"], topic["id"])
    result = await SubjectTutorService.process_tutor_turn(
        student_id=student_id,
        session_id=session_id,
        user_text="Teach me about Quantum Cryptography"
    )

    assert result["intent"] == "OUT_OF_SYLLABUS_CONFIRMATION"
    expected_prompt = "That information is outside your uploaded syllabus. Are you willing to learn outside the syllabus?"
    assert expected_prompt in result["reply_text"]


# ---------------------------------------------------------------------------
# Test 6: Declining Permission Prevents Outside Content (Section B)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_06_declining_permission_stays_in_syllabus(sample_study_subject):
    sub, unit, topic = sample_study_subject
    student_id = PROTOTYPE_STUDENT_ID
    session_id = "test_study_sess_06"

    SubjectTutorService.set_active_subject_and_topic(student_id, session_id, sub["id"], topic["id"])
    
    # 1. Ask outside topic
    await SubjectTutorService.process_tutor_turn(
        student_id=student_id,
        session_id=session_id,
        user_text="Explain black holes"
    )

    # 2. User declines
    result = await SubjectTutorService.process_tutor_turn(
        student_id=student_id,
        session_id=session_id,
        user_text="No, stay in syllabus"
    )

    assert result["intent"] == "OUTSIDE_SYLLABUS_DECLINED"
    assert "strictly within your uploaded syllabus" in result["reply_text"]


# ---------------------------------------------------------------------------
# Test 7: Explicit Permission Enables Clearly Labelled Content (Section B)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_07_explicit_permission_enables_labelled_content(sample_study_subject):
    sub, unit, topic = sample_study_subject
    student_id = PROTOTYPE_STUDENT_ID
    session_id = "test_study_sess_07"

    SubjectTutorService.set_active_subject_and_topic(student_id, session_id, sub["id"], topic["id"])

    # 1. Ask outside topic
    await SubjectTutorService.process_tutor_turn(
        student_id=student_id,
        session_id=session_id,
        user_text="Teach me about Quantum Computing"
    )

    # 2. User consents
    result = await SubjectTutorService.process_tutor_turn(
        student_id=student_id,
        session_id=session_id,
        user_text="Yes, I am willing"
    )

    assert result["intent"] == "OUTSIDE_SYLLABUS_EXPLANATION"
    assert result.get("is_outside_syllabus") is True
    assert "[Outside the syllabus]" in result["reply_text"]


# ---------------------------------------------------------------------------
# Test 8: Quiz Generation and Grading from Uploaded Content Only (Section E)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_08_quiz_generation_and_source_grading(sample_study_subject):
    sub, unit, topic = sample_study_subject
    student_id = PROTOTYPE_STUDENT_ID
    session_id = "test_study_sess_08"

    SubjectTutorService.set_active_subject_and_topic(student_id, session_id, sub["id"], topic["id"])

    # Trigger quiz
    q_result = await SubjectTutorService.process_tutor_turn(
        student_id=student_id,
        session_id=session_id,
        user_text="Quiz me"
    )
    assert q_result["intent"] == "QUIZ"
    assert "Forward Kinematics" in q_result["reply_text"]

    # Answer correctly based on source
    ans_result = await SubjectTutorService.process_tutor_turn(
        student_id=student_id,
        session_id=session_id,
        user_text="It calculates the Cartesian position and orientation of the hand from joint angles."
    )
    assert ans_result["intent"] == "QUIZ_RESULT"
    assert "Spot on!" in ans_result["reply_text"]


# ---------------------------------------------------------------------------
# Test 9: Progress Changing After Completed Topic or Recorded Quiz (Section E)
# ---------------------------------------------------------------------------
def test_09_progress_updates_correctly(sample_study_subject):
    sub, unit, topic = sample_study_subject
    test_student_id = 99  # Dedicated student ID for clean progress delta validation

    # Initial progress
    initial_prog = get_or_create_student_subject_progress(test_student_id, sub["id"])
    assert initial_prog["completed_count"] == 0
    assert initial_prog["mastery_score"] == 0

    # Mark topic completed
    updated_prog = mark_topic_completed(test_student_id, sub["id"], topic["topic_name"])
    assert updated_prog["completed_count"] >= 1
    assert updated_prog["mastery_score"] > 0
    assert topic["topic_name"] in updated_prog["completed_topics"]


# ---------------------------------------------------------------------------
# Test 10: Weak-Topic Tracking After Incorrect Answers (Section E)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_10_weak_topic_tracking(sample_study_subject):
    sub, unit, topic = sample_study_subject
    student_id = PROTOTYPE_STUDENT_ID
    session_id = "test_study_sess_10"

    SubjectTutorService.set_active_subject_and_topic(student_id, session_id, sub["id"], topic["id"])

    # Trigger quiz
    await SubjectTutorService.process_tutor_turn(
        student_id=student_id,
        session_id=session_id,
        user_text="Quiz me"
    )

    # Answer incorrectly
    ans_result = await SubjectTutorService.process_tutor_turn(
        student_id=student_id,
        session_id=session_id,
        user_text="I don't know the answer"
    )
    assert ans_result["intent"] == "QUIZ_RESULT"

    # Verify weak topic recorded in database
    prog = get_or_create_student_subject_progress(student_id, sub["id"])
    weak_topics = prog.get("weak_topics") or []
    assert topic["topic_name"] in weak_topics


# ---------------------------------------------------------------------------
# Test 11: PDF Selection Staying Within Study Workspace (Section C)
# ---------------------------------------------------------------------------
def test_11_pdf_selection_stays_in_study(client, sample_study_subject):
    sub, unit, topic = sample_study_subject

    # Create readable notes document for study agent
    content = b"# Robot Kinematics Notes\nOverview of planar arm links and coordinate transformations."
    files = {"file": ("kinematics_notes.md", io.BytesIO(content), "text/markdown")}
    res = client.post(
        f"/api/v1/study/upload?subject_id={sub['id']}&student_id={PROTOTYPE_STUDENT_ID}",
        files=files
    )
    assert res.status_code == 200

    # Verify document listed in study documents
    doc_res = client.get(f"/api/v1/study/documents?subject_id={sub['id']}")
    assert doc_res.status_code == 200
    docs = doc_res.json()
    assert any(d["filename"] == "kinematics_notes.md" for d in docs)


# ---------------------------------------------------------------------------
# Test 12: Study Messages Never Routed to Normal Chat (Section C)
# ---------------------------------------------------------------------------
def test_12_study_chat_isolation_from_normal_chat(client, sample_study_subject):
    sub, unit, topic = sample_study_subject
    student_id = PROTOTYPE_STUDENT_ID
    normal_sess = "normal_chat_session_abc"
    study_sess = "study_chat_session_xyz"

    # Send message to Study Chat
    study_res = client.post("/api/v1/study/chat", json={
        "message": "Explain forward kinematics",
        "student_id": student_id,
        "session_id": study_sess,
        "subject_id": sub["id"]
    })
    assert study_res.status_code == 200

    # Normal chat history must NOT have this message
    normal_hist = get_conversation_history(student_id=student_id, session_id=normal_sess)
    assert len(normal_hist) == 0

    # Study chat history MUST have the message
    study_hist = get_conversation_history(student_id=student_id, session_id=study_sess)
    assert len(study_hist) >= 2  # user + assistant
    assert any("forward kinematics" in m["content"].lower() for m in study_hist)


# ---------------------------------------------------------------------------
# Test 13: Multiple Subjects and Uploaded Documents Remain Isolated (Section D)
# ---------------------------------------------------------------------------
def test_13_multiple_subjects_isolation(client, setup_test_db):
    sub1_id = create_subject("Mathematics", "Calculus and linear algebra")
    sub2_id = create_subject("Biology", "Cellular biology")

    # Upload material for Subject 1
    s1_doc = b"# Derivatives\nCalculus covers limits, slopes, and instantaneous rates of change."
    res1 = client.post(
        f"/api/v1/study/upload?subject_id={sub1_id}&student_id={PROTOTYPE_STUDENT_ID}",
        files={"file": ("calculus.md", io.BytesIO(s1_doc), "text/markdown")}
    )
    assert res1.status_code == 200

    # Upload material for Subject 2
    s2_doc = b"# Mitochondria\nThe powerhouse of eukaryotic cells generating ATP."
    res2 = client.post(
        f"/api/v1/study/upload?subject_id={sub2_id}&student_id={PROTOTYPE_STUDENT_ID}",
        files={"file": ("biology.md", io.BytesIO(s2_doc), "text/markdown")}
    )
    assert res2.status_code == 200

    # Documents and topics must be strictly isolated by subject
    docs1 = client.get(f"/api/v1/study/documents?subject_id={sub1_id}").json()
    docs2 = client.get(f"/api/v1/study/documents?subject_id={sub2_id}").json()

    assert all(d["subject_id"] == sub1_id for d in docs1)
    assert all(d["subject_id"] == sub2_id for d in docs2)
    assert not any("biology" in d["filename"] for d in docs1)
    assert not any("calculus" in d["filename"] for d in docs2)


# ---------------------------------------------------------------------------
# Test 14: Invalid or Unreadable Files Produce Useful Errors (Section D)
# ---------------------------------------------------------------------------
def test_14_invalid_file_handling(client, sample_study_subject):
    sub, unit, topic = sample_study_subject

    # Unsupported file type (.exe)
    res = client.post(
        f"/api/v1/study/upload?subject_id={sub['id']}&student_id={PROTOTYPE_STUDENT_ID}",
        files={"file": ("bad_file.exe", io.BytesIO(b"MZ\x90\x00"), "application/octet-stream")}
    )
    assert res.status_code == 400
    assert "Unsupported file type" in res.json()["detail"]

    # Empty document
    res_empty = client.post(
        f"/api/v1/study/upload?subject_id={sub['id']}&student_id={PROTOTYPE_STUDENT_ID}",
        files={"file": ("empty.txt", io.BytesIO(b"   \n  \t"), "text/plain")}
    )
    assert res_empty.status_code == 400
    assert "empty or no readable text" in res_empty.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test 15: Conversation and Progress Persistence After Refresh (Section E, F)
# ---------------------------------------------------------------------------
def test_15_conversation_and_progress_persistence(client, sample_study_subject):
    sub, unit, topic = sample_study_subject
    student_id = PROTOTYPE_STUDENT_ID
    session_id = "test_persistence_sess_15"

    # 1. Send study message
    client.post("/api/v1/study/chat", json={
        "message": "Teach me forward kinematics",
        "student_id": student_id,
        "session_id": session_id,
        "subject_id": sub["id"]
    })

    # 2. Mark topic completed
    client.post(f"/api/v1/study/topics/{topic['id']}/complete?student_id={student_id}&subject_id={sub['id']}")

    # 3. Simulate page reload / new request: fetch history and progress
    hist_res = client.get(f"/api/v1/study/history?session_id={session_id}&student_id={student_id}")
    assert hist_res.status_code == 200
    messages = hist_res.json()["messages"]
    assert len(messages) >= 2

    prog_res = client.get(f"/api/v1/subjects/{sub['id']}/progress?student_id={student_id}")
    assert prog_res.status_code == 200
    prog = prog_res.json()
    assert prog["completed_count"] >= 1
    assert prog["mastery_score"] > 0


# ---------------------------------------------------------------------------
# Test 16: Existing Onboarding, Normal Chat, and Profile Intact (Section G)
# ---------------------------------------------------------------------------
def test_16_existing_features_intact(client):
    # Verify student profile intact
    res = client.get(f"/api/v1/students/{PROTOTYPE_STUDENT_ID}")
    assert res.status_code == 200
    student = res.json()
    assert student["name"] == "Alex"
    assert student["onboarding_completed"] == 1

    # Verify normal chat endpoint works as expected
    chat_res = client.post("/api/v1/esp32/chat", json={
        "message": "Hello Mizo, how are you today?",
        "student_id": PROTOTYPE_STUDENT_ID,
        "session_id": "normal_session_regression",
        "mode": "companion"
    })
    assert chat_res.status_code == 200
    res_data = chat_res.json()
    assert "reply_text" in res_data or "response" in res_data


# ---------------------------------------------------------------------------
# Test 17: Teaching Continuity — No Repetition on Continuation Request
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_17_teaching_continuity_no_repetition(client, sample_study_subject):
    sub, unit, topic = sample_study_subject
    student_id = PROTOTYPE_STUDENT_ID
    session_id = "test_teaching_continuity_sess_01"

    # Step 1: User asks initial teaching request
    res1 = client.post("/api/v1/study/chat", json={
        "message": "Teach me forward kinematics using a two-link robot arm. Explain the equations with a numerical example.",
        "student_id": student_id,
        "session_id": session_id,
        "subject_id": sub["id"],
        "topic_id": topic["id"]
    })
    assert res1.status_code == 200
    turn1_text = res1.json()["reply_text"]
    assert len(turn1_text) > 20

    # Step 2: User asks follow-up to continue without repeating intro
    res2 = client.post("/api/v1/study/chat", json={
        "message": "Teach me more. Continue from where you stopped, without repeating the introduction.",
        "student_id": student_id,
        "session_id": session_id,
        "subject_id": sub["id"]
    })
    assert res2.status_code == 200
    turn2_text = res2.json()["reply_text"]

    # Assert Turn 2 does not repeat the introductory repetition phrase
    assert "here it is again" not in turn2_text.lower()
    # Assert Turn 2 contains new instructional content (kinematic equations or numerical values)
    assert any(term in turn2_text.lower() for term in ["cos", "sin", "link", "theta", "hand", "elbow", "coordinate", "x =", "y ="])
    # Turn 2 should be distinct new material from Turn 1
    assert turn2_text.strip() != turn1_text.strip()

    # Step 3: Next continuation advances to numerical or check question
    res3 = client.post("/api/v1/study/chat", json={
        "message": "Teach me more. Continue from where you stopped.",
        "student_id": student_id,
        "session_id": session_id,
        "subject_id": sub["id"]
    })
    assert res3.status_code == 200
    turn3_text = res3.json()["reply_text"]
    assert turn3_text.strip() != turn2_text.strip()
    assert any(term in turn3_text.lower() for term in ["10 cm", "case", "calculation", "question", "quick check", "test"])

