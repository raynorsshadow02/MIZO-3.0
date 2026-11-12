"""
Regression and Lifecycle Test Suite for Mizo 3.0:
- Context and topic switching (One Piece topic-locking bug fix)
- Persistence and Lifecycle (Onboarding, Chat History, Persistent Memory separation)
- UI and API safety (Save to memory, New Chat, Delete Chat, Multi-student isolation)
"""
import pytest
from starlette.testclient import TestClient
from app.main import app
from app.db.database import (
    init_db,
    get_connection,
    get_student,
    create_student,
    update_student,
    create_session,
    save_durable_memory,
    get_student_memories,
    delete_durable_memory,
    delete_chat,
    get_conversation_history,
    update_session_context,
    get_session_context,
    PROTOTYPE_STUDENT_ID
)
from app.services.context_service import (
    extract_topic_and_entities,
    detect_conversation_intent,
    filter_relevant_memories
)
from app.services.llm_service import clean_llm_response, llm_service
from app.services.memory_service import MizoMemoryManager

client = TestClient(app)


@pytest.fixture(autouse=True)
def ensure_database():
    """Ensure database tables and initial student exist before running tests."""
    init_db()
    update_student(PROTOTYPE_STUDENT_ID, {
        "name": "Alex",
        "onboarding_completed": 1,
        "onboarding_step": "completed",
        "preferred_name": "Alex",
        "durable_memories": []
    })
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM memories WHERE student_id = ?", (PROTOTYPE_STUDENT_ID,))
    conn.commit()
    conn.close()



# =====================================================================
# Context and Topic Switching Tests (Scenarios 1 - 6)
# =====================================================================

def test_01_discuss_one_piece_then_ask_kinematics():
    """Scenario 1: Discuss One Piece, then ask about kinematics -> answers kinematics directly."""
    # 1. Establish One Piece topic in session context
    sess = create_session(student_id=PROTOTYPE_STUDENT_ID, mode="coach")
    sid = sess["session_id"]
    update_session_context(sid, {
        "current_topic": "Anime & Manga",
        "current_subtopic": "One Piece",
        "current_entities": ["Luffy", "Zoro"]
    })

    # 2. Next message asks about kinematics
    msg = "Tell me about kinematics."
    topic, subtopic, entities = extract_topic_and_entities(
        msg, current_topic="Anime & Manga", current_subtopic="One Piece", current_entities=["Luffy", "Zoro"]
    )
    assert topic in ["Physics & Engineering", "physics_and_engineering"]
    assert subtopic == "Kinematics"
    # Entities from previous topic must not leak into kinematics
    assert "Luffy" not in entities
    assert "Zoro" not in entities

    # 3. LLM response must answer kinematics directly without anime analogies
    res = client.post("/api/v1/esp32/chat", json={
        "message": msg,
        "student_id": PROTOTYPE_STUDENT_ID,
        "session_id": sid,
        "is_typed_text": True
    })
    assert res.status_code == 200
    resp_lower = res.json()["reply_text"].lower()
    assert "kinematics" in resp_lower or "motion" in resp_lower
    assert "one piece" not in resp_lower
    assert "luffy" not in resp_lower


def test_02_discuss_one_piece_then_ask_user_projects():
    """Scenario 2: Discuss One Piece, then ask about user projects -> uses saved project memory without returning to One Piece."""
    sess = create_session(student_id=PROTOTYPE_STUDENT_ID, mode="coach")
    sid = sess["session_id"]
    update_session_context(sid, {
        "current_topic": "Anime & Manga",
        "current_subtopic": "One Piece",
        "current_entities": ["Zoro"]
    })

    # Save a project memory
    save_durable_memory(
        student_id=PROTOTYPE_STUDENT_ID,
        memory_text="Developing an autonomous quadruped robot using ESP32-S3 and ROS2.",
        category="project",
        session_id=sid
    )

    msg = "What are my projects?"
    intent = detect_conversation_intent(msg)
    topic, subtopic, entities = extract_topic_and_entities(
        msg, current_topic="Anime & Manga", current_subtopic="One Piece"
    )
    assert intent == "PROJECTS_QUERY"

    res = client.post("/api/v1/esp32/chat", json={
        "message": msg,
        "student_id": PROTOTYPE_STUDENT_ID,
        "session_id": sid,
        "is_typed_text": True
    })
    assert res.status_code == 200
    resp_lower = res.json()["reply_text"].lower()
    assert ("project" in resp_lower or "robot" in resp_lower or "esp32" in resp_lower or "quadruped" in resp_lower)
    assert "one piece" not in resp_lower
    assert "zoro" not in resp_lower

    # Clean up test memory so it doesn't leak into subsequent tests
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM memories WHERE student_id = ?", (PROTOTYPE_STUDENT_ID,))
    c.execute("UPDATE students SET durable_memories = '[]' WHERE id = ?", (PROTOTYPE_STUDENT_ID,))
    conn.commit()
    conn.close()


def test_03_discuss_kinematics_then_teach_me():
    """Scenario 3: Discuss kinematics, then say 'Teach me' -> continues the relevant kinematics lesson."""
    sess = create_session(student_id=PROTOTYPE_STUDENT_ID, mode="coach")
    sid = sess["session_id"]
    update_session_context(sid, {
        "current_topic": "Physics & Engineering",
        "current_subtopic": "Kinematics",
        "current_entities": ["Displacement", "Velocity"]
    })

    msg = "Teach me."
    intent = detect_conversation_intent(msg)
    topic, subtopic, entities = extract_topic_and_entities(
        msg, current_topic="Physics & Engineering", current_subtopic="Kinematics", current_entities=["Velocity"]
    )
    assert intent == "TEACH_REQUEST"
    assert topic == "Physics & Engineering"
    assert subtopic == "Kinematics"

    res = client.post("/api/v1/esp32/chat", json={
        "message": msg,
        "student_id": PROTOTYPE_STUDENT_ID,
        "session_id": sid,
        "is_typed_text": True
    })
    assert res.status_code == 200
    resp_lower = res.json()["reply_text"].lower()
    assert any(term in resp_lower for term in ["motion", "velocity", "displacement", "acceleration", "kinematics", "lesson"])


def test_04_say_thank_you_does_not_revive_old_topic():
    """Scenario 4: Say 'Thank you' -> responds naturally without reviving an old topic."""
    sess = create_session(student_id=PROTOTYPE_STUDENT_ID, mode="coach")
    sid = sess["session_id"]
    update_session_context(sid, {
        "current_topic": "Anime & Manga",
        "current_subtopic": "One Piece"
    })

    msg = "Thank you!"
    intent = detect_conversation_intent(msg)
    assert intent == "COURTESY"

    topic, subtopic, entities = extract_topic_and_entities(
        msg, current_topic="Anime & Manga", current_subtopic="One Piece"
    )
    # Courtesies clear sticky topic state
    assert topic == ""

    res = client.post("/api/v1/esp32/chat", json={
        "message": msg,
        "student_id": PROTOTYPE_STUDENT_ID,
        "session_id": sid,
        "is_typed_text": True
    })
    assert res.status_code == 200
    resp_lower = res.json()["reply_text"].lower()
    assert "welcome" in resp_lower or "anytime" in resp_lower or "glad" in resp_lower
    assert "one piece" not in resp_lower
    assert "luffy" not in resp_lower


def test_05_say_move_on_with_ambiguous_context():
    """Scenario 5: Say 'move on' with ambiguous context -> asks for concise clarification rather than assuming One Piece."""
    sess = create_session(student_id=PROTOTYPE_STUDENT_ID, mode="coach")
    sid = sess["session_id"]
    update_session_context(sid, {
        "current_topic": "Anime & Manga",
        "current_subtopic": "One Piece"
    })

    msg = "I want to move on."
    intent = detect_conversation_intent(msg)
    assert intent == "MOVE_ON_QUERY"

    res = client.post("/api/v1/esp32/chat", json={
        "message": msg,
        "student_id": PROTOTYPE_STUDENT_ID,
        "session_id": sid,
        "is_typed_text": True
    })
    assert res.status_code == 200
    resp_lower = res.json()["reply_text"].lower()
    assert "what" in resp_lower or "next" in resp_lower or "like to" in resp_lower or "move" in resp_lower
    assert "one piece" not in resp_lower


def test_06_saved_interests_do_not_override_explicit_new_requests():
    """Scenario 6: Confirm saved interests do not override explicit new requests."""
    st = get_student(PROTOTYPE_STUDENT_ID)
    update_student(PROTOTYPE_STUDENT_ID, {"interests": "One Piece, Anime, Straw Hats"})

    # User explicitly asks an engineering question
    msg = "Can you explain how a PID controller works?"
    student = get_student(PROTOTYPE_STUDENT_ID)
    filtered = filter_relevant_memories(student, current_topic="Engineering", current_subtopic="Control Systems", user_message=msg)

    # Interests like One Piece should be excluded from relevant prompt memories
    assert "One Piece" not in filtered.get("relevant_interests", [])

    ctx = MizoMemoryManager.build_llm_context(
        student_id=PROTOTYPE_STUDENT_ID,
        current_message=msg
    )
    sys_prompt = ctx["system_prompt"]
    assert "CURRENT CONVERSATION TOPIC MUST TAKE PRIORITY" in sys_prompt
    assert "Address the user's latest explicit request immediately" in sys_prompt


# =====================================================================
# Persistence and Lifecycle Tests (Scenarios 7 - 16)
# =====================================================================

def test_07_refreshing_restores_current_chat():
    """Scenario 7: Session endpoint restores the current active chat."""
    res = client.get(f"/api/v1/students/{PROTOTYPE_STUDENT_ID}/sessions/active")
    assert res.status_code == 200
    data = res.json()
    assert "session_id" in data
    active_id = data["session_id"]

    # Simulating a subsequent page refresh request
    res2 = client.get(f"/api/v1/students/{PROTOTYPE_STUDENT_ID}/sessions/active")
    assert res2.status_code == 200
    assert res2.json()["session_id"] == active_id


def test_08_new_chat_starts_with_empty_visible_history():
    """Scenario 8: New Chat starts with a fresh session and empty visible history."""
    # 1. Add messages to current session
    client.post("/api/v1/esp32/chat", json={
        "message": "Hello Mizo, testing old chat turn 1",
        "student_id": PROTOTYPE_STUDENT_ID,
        "is_typed_text": True
    })

    # 2. Trigger New Chat
    new_chat_res = client.post("/api/v1/conversations/new-chat", json={
        "student_id": PROTOTYPE_STUDENT_ID,
        "mode": "coach"
    })
    assert new_chat_res.status_code == 200
    new_chat_data = new_chat_res.json()
    new_sess_id = new_chat_data["session_id"]

    # 3. Verify history filtered by this new session is completely empty
    hist_res = client.get(f"/api/v1/conversations?student_id={PROTOTYPE_STUDENT_ID}&session_id={new_sess_id}")
    assert hist_res.status_code == 200
    assert len(hist_res.json()) == 0


def test_09_new_chat_preserves_onboarding_and_saved_memories():
    """Scenario 9: New Chat preserves onboarding profile and saved durable memories."""
    # Ensure student profile has known name and completed onboarding
    update_student(PROTOTYPE_STUDENT_ID, {
        "name": "Alex Mercer",
        "onboarding_completed": 1,
        "learning_goals": "Improve technical presentation"
    })

    # Save a persistent memory
    save_res = client.post(f"/api/v1/students/{PROTOTYPE_STUDENT_ID}/memories", json={
        "memory_text": "Prefers concise technical analogies without jargon.",
        "category": "preference"
    })
    assert save_res.status_code == 200

    # Start new chat
    client.post("/api/v1/conversations/new-chat", json={"student_id": PROTOTYPE_STUDENT_ID})

    # Check student profile
    student = get_student(PROTOTYPE_STUDENT_ID)
    assert student["name"] == "Alex Mercer"
    assert student["onboarding_completed"] == 1
    assert student["learning_goals"] == "Improve technical presentation"

    # Check saved memories
    mems = get_student_memories(PROTOTYPE_STUDENT_ID)
    assert any("concise technical analogies" in m["memory_text"] for m in mems)


def test_10_deleting_chat_removes_messages_and_summaries():
    """Scenario 10: Deleting a chat removes that chat's messages and session context."""
    # Create specific chat
    sess = create_session(student_id=PROTOTYPE_STUDENT_ID, mode="coach")
    sid = sess["session_id"]

    # Add message
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversations (student_id, session_id, role, content) VALUES (?, ?, ?, ?)",
        (PROTOTYPE_STUDENT_ID, sid, "user", "Message to be deleted")
    )
    conn.commit()
    conn.close()

    assert len(get_conversation_history(student_id=PROTOTYPE_STUDENT_ID, session_id=sid)) == 1

    # Delete this chat
    del_res = client.delete(f"/api/v1/conversations/chats/{sid}?student_id={PROTOTYPE_STUDENT_ID}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted_messages"] >= 1

    # Verify messages are gone
    remaining = get_conversation_history(student_id=PROTOTYPE_STUDENT_ID, session_id=sid)
    assert len(remaining) == 0


def test_11_deleting_chat_preserves_onboarding_and_saved_memories():
    """Scenario 11: Deleting a chat preserves onboarding and independently saved memories."""
    update_student(PROTOTYPE_STUDENT_ID, {
        "name": "Elena Rostova",
        "onboarding_completed": 1
    })

    save_durable_memory(
        student_id=PROTOTYPE_STUDENT_ID,
        memory_text="Built a self-balancing robot with STM32.",
        category="project"
    )

    sess = create_session(student_id=PROTOTYPE_STUDENT_ID)
    sid = sess["session_id"]

    # Delete the chat
    client.delete(f"/api/v1/conversations/chats/{sid}")

    # Verify student profile is intact
    st = get_student(PROTOTYPE_STUDENT_ID)
    assert st["name"] == "Elena Rostova"
    assert st["onboarding_completed"] == 1

    # Verify memory is intact
    mems = get_student_memories(PROTOTYPE_STUDENT_ID)
    assert any("self-balancing robot" in m["memory_text"] for m in mems)


def test_12_saved_memory_persists_in_sqlite():
    """Scenario 12: Saved memory persists across application queries."""
    text = "Prefers bullet points for code syntax explanations."
    saved = save_durable_memory(
        student_id=PROTOTYPE_STUDENT_ID,
        memory_text=text,
        category="preference"
    )
    assert saved["id"] is not None

    mems = get_student_memories(PROTOTYPE_STUDENT_ID)
    found = [m for m in mems if m["memory_text"] == text]
    assert len(found) == 1
    assert found[0]["category"] == "preference"


def test_13_saving_same_memory_twice_prevents_uncontrolled_duplicates():
    """Scenario 13: Saving the same memory twice does not create uncontrolled duplicates."""
    text = "Interested in autonomous navigation algorithms."
    mem1 = save_durable_memory(student_id=PROTOTYPE_STUDENT_ID, memory_text=text, category="academic")
    mem2 = save_durable_memory(student_id=PROTOTYPE_STUDENT_ID, memory_text=text, category="academic")

    assert mem1["id"] == mem2["id"]

    all_mems = [m for m in get_student_memories(PROTOTYPE_STUDENT_ID) if m["memory_text"] == text]
    assert len(all_mems) == 1


def test_14_different_students_cannot_access_each_others_chats_or_memories():
    """Scenario 14: Different students cannot access each other's chats or memories."""
    # Create Student #2
    s2 = create_student({
        "name": "Student Two",
        "education": "University",
        "onboarding_completed": True
    })
    s2_id = s2["id"]

    # Save a memory for Student #2
    save_durable_memory(student_id=s2_id, memory_text="Secret Student 2 Project", category="project")

    # Fetch Student 1 memories -> must NOT contain Student 2's memory
    s1_mems = client.get(f"/api/v1/students/{PROTOTYPE_STUDENT_ID}/memories").json()
    assert not any("Secret Student 2 Project" in m["memory_text"] for m in s1_mems)

    # Fetch Student 2 memories -> contains their memory
    s2_mems = client.get(f"/api/v1/students/{s2_id}/memories").json()
    assert any("Secret Student 2 Project" in m["memory_text"] for m in s2_mems)

    # Deleting Student 2 memory using Student 1 ID must fail
    s2_mem_id = [m["id"] for m in s2_mems if "Secret Student 2 Project" in m["memory_text"]][0]
    unauthorized_del = client.delete(f"/api/v1/students/{PROTOTYPE_STUDENT_ID}/memories/{s2_mem_id}")
    assert unauthorized_del.status_code == 404


def test_15_missing_student_identity_cannot_cause_shared_global_memory_leakage():
    """Scenario 15: Missing student identity returns 404 and does not leak global memory."""
    res = client.get("/api/v1/students/999999/memories")
    assert res.status_code == 404

    post_res = client.post("/api/v1/students/999999/memories", json={"memory_text": "Test"})
    assert post_res.status_code == 404


def test_16_existing_new_session_compatible_with_study_workflow():
    """Scenario 16: Existing New Session endpoint preserves learning session behavior."""
    sess_res = client.post(f"/api/v1/students/{PROTOTYPE_STUDENT_ID}/sessions", json={"mode": "coach"})
    assert sess_res.status_code == 200
    data = sess_res.json()
    assert "session_id" in data
    assert data["metrics"]["message_count"] == 0


# =====================================================================
# UI and API Safety Tests (Scenarios 17 - 19)
# =====================================================================

def test_17_save_to_memory_candidate_review_and_cancellation():
    """Scenario 17: Candidate memory generation supports review, confirmation, cancellation, and validation."""
    sess = create_session(student_id=PROTOTYPE_STUDENT_ID, mode="coach")
    sid = sess["session_id"]

    # Candidate generation with specific text
    res = client.post(f"/api/v1/students/{PROTOTYPE_STUDENT_ID}/memories/candidate", json={
        "session_id": sid,
        "message_text": "I really enjoy building ROS2 robotics packages on Linux."
    })
    assert res.status_code == 200
    data = res.json()
    assert "candidate_memory" in data
    assert data["category"] in ["project", "academic", "general"]
    assert "ROS2" in data["candidate_memory"]

    # Empty payload validation
    bad_res = client.post(f"/api/v1/students/{PROTOTYPE_STUDENT_ID}/memories", json={"memory_text": "   "})
    assert bad_res.status_code == 400


def test_18_reasoning_leakage_and_stale_response_prevention():
    """Scenario 18: Reasoning text like 'Here's a thinking process' is cleaned and never exposed."""
    raw_output = """<think>
The user is asking about kinematics. I should explain displacement and velocity.
</think>
Here's a thinking process that leads to the suggested response:
1. Identify the core concepts.
Kinematics is the branch of classical mechanics that describes the motion of points, bodies, and systems of bodies without considering the forces that cause the motion."""

    cleaned = clean_llm_response(raw_output)
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    assert "Here's a thinking process" not in cleaned
    assert "Kinematics is the branch" in cleaned


def test_19_delete_chat_and_new_chat_api_endpoints():
    """Scenario 19: New Chat and Delete Chat endpoints respond with expected schema and status."""
    # New chat
    new_chat = client.post("/api/v1/conversations/new-chat", json={"student_id": PROTOTYPE_STUDENT_ID})
    assert new_chat.status_code == 200
    created_sid = new_chat.json()["session_id"]
    assert created_sid.startswith("ses_")

    # Delete chat
    del_chat = client.delete(f"/api/v1/conversations/chats/{created_sid}?student_id={PROTOTYPE_STUDENT_ID}")
    assert del_chat.status_code == 200
    assert del_chat.json()["status"] == "deleted"


def test_20_project_retrieval_returns_unsupported_error_when_no_records_exist_without_guessing():
    """Regression Test 1: Querying projects when no reliable project records exist must not invent projects or statuses."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM memories WHERE student_id = ? AND category = 'project'", (PROTOTYPE_STUDENT_ID,))
    c.execute("UPDATE students SET durable_memories = '[]' WHERE id = ?", (PROTOTYPE_STUDENT_ID,))
    conn.commit()
    conn.close()

    sess = create_session(student_id=PROTOTYPE_STUDENT_ID, mode="coach")
    sid = sess["session_id"]

    prompt = "Tell me about my projects. List only the projects you have information about, explain their current status, and tell me if any details are missing. Do not guess."
    res = client.post("/api/v1/esp32/chat", json={
        "message": prompt,
        "student_id": PROTOTYPE_STUDENT_ID,
        "session_id": sid,
        "is_typed_text": True
    })
    assert res.status_code == 200
    reply = res.json()["reply_text"].lower()

    # Must state that records do not contain enough information
    assert ("do not contain enough information" in reply or "no specific project records" in reply or "not enough information" in reply)
    # Must NOT hallucinate quadruped, ROS2, or Mizo robotic tutor as the user's project
    assert "quadruped" not in reply
    assert "ros2" not in reply
    assert "working on your mizo robotic tutor" not in reply


def test_21_teaching_continuity_followup_produces_new_content_without_repetition():
    """Regression Test 2: 'Teach me more' / 'continue' must advance the lesson rather than repeating the intro."""
    sess = create_session(student_id=PROTOTYPE_STUDENT_ID, mode="coach")
    sid = sess["session_id"]

    # 1. Ask initial teaching question
    turn1 = client.post("/api/v1/esp32/chat", json={
        "message": "Teach me forward kinematics in robotics. Explain it step by step with a simple robot-arm example.",
        "student_id": PROTOTYPE_STUDENT_ID,
        "session_id": sid,
        "is_typed_text": True
    })
    assert turn1.status_code == 200
    reply1 = turn1.json()["reply_text"]
    assert len(reply1) > 20

    # 2. Follow-up: "Teach me more."
    turn2 = client.post("/api/v1/esp32/chat", json={
        "message": "Teach me more.",
        "student_id": PROTOTYPE_STUDENT_ID,
        "session_id": sid,
        "is_typed_text": True
    })
    assert turn2.status_code == 200
    reply2 = turn2.json()["reply_text"]

    # Must NOT repeat the exact same response
    assert reply2.strip() != reply1.strip()
    # Reply 2 must contain concrete instructional progression (robot arm components, formulas, or angles)
    r2_lower = reply2.lower()
    assert any(term in r2_lower for term in ["shoulder", "elbow", "link", "l₁", "l1", "theta", "hand", "trigonometric", "x₁", "x1", "cos", "angle"])

    # 3. Follow-up again: "Explain the next step"
    turn3 = client.post("/api/v1/esp32/chat", json={
        "message": "Explain the next step.",
        "student_id": PROTOTYPE_STUDENT_ID,
        "session_id": sid,
        "is_typed_text": True
    })
    assert turn3.status_code == 200
    reply3 = turn3.json()["reply_text"]
    assert reply3.strip() != reply2.strip()
    assert reply3.strip() != reply1.strip()


def test_22_new_chat_clears_visible_conversation_and_persists_across_refresh():
    """Regression Test 3: New Chat starts fresh, visible history is cleared, and survives page refresh."""
    # 1. Have existing messages in a session
    sess = create_session(student_id=PROTOTYPE_STUDENT_ID, mode="coach")
    sid1 = sess["session_id"]
    client.post("/api/v1/esp32/chat", json={
        "message": "Hello from session 1",
        "student_id": PROTOTYPE_STUDENT_ID,
        "session_id": sid1,
        "is_typed_text": True
    })

    # History for sid1 has 2 messages (user + assistant)
    hist1 = client.get(f"/api/v1/conversations?student_id={PROTOTYPE_STUDENT_ID}&session_id={sid1}").json()
    assert len(hist1) >= 2

    # 2. Start New Chat
    new_res = client.post("/api/v1/conversations/new-chat", json={"student_id": PROTOTYPE_STUDENT_ID})
    assert new_res.status_code == 200
    sid2 = new_res.json()["session_id"]
    assert sid2 != sid1

    # History for sid2 must be completely empty
    hist2 = client.get(f"/api/v1/conversations?student_id={PROTOTYPE_STUDENT_ID}&session_id={sid2}").json()
    assert len(hist2) == 0

    # 3. Simulate page refresh: active session endpoint returns active session sid2
    active_res = client.get(f"/api/v1/students/{PROTOTYPE_STUDENT_ID}/sessions/active")
    assert active_res.status_code == 200
    active_data = active_res.json()
    assert active_data["session_id"] == sid2

    # Verify learner profile survived untouched
    st = get_student(PROTOTYPE_STUDENT_ID)
    assert st["onboarding_completed"] == 1
    assert st["name"] == "Alex"


def test_23_saved_memory_technical_preference_persistence_and_selective_application():
    """Regression Test 4: Save technical preference memory -> persists across new chats and applies selectively."""
    pref_text = (
        "I enjoy discussing One Piece, but I also want to learn robotics and artificial intelligence. "
        "When I ask about a technical subject, answer the technical question directly rather than forcing anime examples into it."
    )
    save_res = client.post(f"/api/v1/students/{PROTOTYPE_STUDENT_ID}/memories", json={
        "memory_text": pref_text,
        "category": "preference"
    })
    assert save_res.status_code == 200

    # 1. Verify it appears in Manage Memories list
    mems = client.get(f"/api/v1/students/{PROTOTYPE_STUDENT_ID}/memories").json()
    assert any(m["memory_text"] == pref_text for m in mems)

    # 2. Start a new chat
    new_chat = client.post("/api/v1/conversations/new-chat", json={"student_id": PROTOTYPE_STUDENT_ID}).json()
    new_sid = new_chat["session_id"]

    # 3. Memory remains present after starting new chat
    mems_after = client.get(f"/api/v1/students/{PROTOTYPE_STUDENT_ID}/memories").json()
    assert any(m["memory_text"] == pref_text for m in mems_after)

    # 4. Ask technical question: preference must be retrieved and applied (no anime forced)
    tech_turn = client.post("/api/v1/esp32/chat", json={
        "message": "Explain how PID control maintains stability in a robotic joint.",
        "student_id": PROTOTYPE_STUDENT_ID,
        "session_id": new_sid,
        "is_typed_text": True
    })
    assert tech_turn.status_code == 200
    tech_reply = tech_turn.json()["reply_text"].lower()
    assert "one piece" not in tech_reply
    assert "luffy" not in tech_reply

    # 5. Delete this chat: preference memory must NOT be deleted
    client.delete(f"/api/v1/conversations/chats/{new_sid}?student_id={PROTOTYPE_STUDENT_ID}")
    mems_final = client.get(f"/api/v1/students/{PROTOTYPE_STUDENT_ID}/memories").json()
    assert any(m["memory_text"] == pref_text for m in mems_final)

