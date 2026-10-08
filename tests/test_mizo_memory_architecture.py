import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import (
    get_student,
    create_student,
    update_student,
    delete_learner_data,
    create_session,
    add_conversation_message,
    get_conversation_history,
    init_db
)
from app.services.memory_service import memory_manager, MizoMemoryManager
from app.services.personalization_service import personalization_service
from app.services.voice_service import voice_manager, VoiceState
from app.services.llm_service import LLMResult

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db():
    init_db()
    delete_learner_data(1)
    voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "test_fixture")


# =============================================================================
# 1. TEST THE EXACT BUG (5+ Turns, 10 Turns, 20 Turns)
# =============================================================================
@pytest.mark.asyncio
async def test_exact_memory_bug_user_name_persists_across_multiple_turns():
    """
    Test the exact reported bug:
    Student Name = Shahid.
    Turn 1: Hello.
    Turn 2: I am studying robotics.
    Turn 3: I want to improve my English.
    Turn 4: What do you think about my goal?
    Turn 5: Tell me something about communication.
    Turn 6: What is my name?
    Expected: Shahid.
    Continue to 10 turns and 20 turns.
    At turn 10 and turn 20, LLM context MUST contain 'Shahid' and reply must identify Shahid.
    """
    # Initialize student with established profile name in persistent database
    update_student(1, {
        "name": "Shahid",
        "onboarding_completed": True,
        "onboarding_step": "completed",
        "education": "Robotics Engineering Student",
        "learning_goals": "Improve English communication"
    })
    st = get_student(1)
    assert st["name"] == "Shahid"
    assert st["onboarding_completed"] is True

    session = create_session(student_id=1, mode="coach")
    session_id = session["session_id"]

    script_turns = [
        "Hello.",
        "I am studying robotics.",
        "I want to improve my English.",
        "What do you think about my goal?",
        "Tell me something about communication."
    ]

    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm, \
         patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        mock_tts.return_value = ("/audio/reply.wav", "reply.wav")

        # Run 5 initial turns
        for i, user_msg in enumerate(script_turns):
            mock_llm.return_value = LLMResult(
                text=f"Response to turn {i + 1}",
                provider="groq",
                model="llama-3.3-70b-versatile",
                is_fallback=False,
                latency_ms=100.0
            )
            res = client.post("/api/v1/esp32/chat", json={
                "student_id": 1,
                "session_id": session_id,
                "message": user_msg
            })
            assert res.status_code == 200

        # Verify Turn 2 ("I am studying robotics.") DID NOT overwrite the user's name
        st_after_5 = get_student(1)
        assert st_after_5["name"] == "Shahid"

        # Turn 6: "What is my name?"
        mock_llm.return_value = LLMResult(
            text="Your name is Shahid! How can I help you today?",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False,
            latency_ms=110.0
        )
        res_t6 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "session_id": session_id,
            "message": "What is my name?"
        })
        assert res_t6.status_code == 200
        assert "Shahid" in res_t6.json()["reply_text"]

        # Inspect the context that was passed to the LLM at Turn 6
        call_args = mock_llm.call_args
        sys_prompt_used = call_args.kwargs.get("system_prompt") or call_args[1].get("system_prompt")
        assert "SHAHID" in sys_prompt_used
        assert "The student's name is SHAHID" in sys_prompt_used

        # Continue conversation past 10 turns
        for i in range(7, 11):
            mock_llm.return_value = LLMResult(
                text=f"Turn {i} reply",
                provider="groq",
                model="llama-3.3-70b-versatile",
                is_fallback=False,
                latency_ms=90.0
            )
            client.post("/api/v1/esp32/chat", json={
                "student_id": 1,
                "session_id": session_id,
                "message": f"This is conversational turn {i} about engineering topics."
            })

        # Turn 11: Ask again after 10+ turns
        mock_llm.return_value = LLMResult(
            text="You are Shahid, of course!",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False,
            latency_ms=95.0
        )
        res_t11 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "session_id": session_id,
            "message": "What is my name?"
        })
        assert res_t11.status_code == 200
        assert "Shahid" in res_t11.json()["reply_text"]
        call_args_11 = mock_llm.call_args
        sys_prompt_11 = call_args_11.kwargs.get("system_prompt") or call_args_11[1].get("system_prompt")
        assert "SHAHID" in sys_prompt_11

        # Continue all the way to 20+ turns
        for i in range(12, 22):
            mock_llm.return_value = LLMResult(
                text=f"Turn {i} reply",
                provider="groq",
                model="llama-3.3-70b-versatile",
                is_fallback=False,
                latency_ms=85.0
            )
            client.post("/api/v1/esp32/chat", json={
                "student_id": 1,
                "session_id": session_id,
                "message": f"Discussing robotics and control systems in turn {i}."
            })

        # Turn 22: Ask name after 20+ turns
        mock_llm.return_value = LLMResult(
            text="You are Shahid, an engineering student.",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False,
            latency_ms=90.0
        )
        res_t22 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "session_id": session_id,
            "message": "What is my name?"
        })
        assert res_t22.status_code == 200
        assert "Shahid" in res_t22.json()["reply_text"]

        call_args_22 = mock_llm.call_args
        sys_prompt_22 = call_args_22.kwargs.get("system_prompt") or call_args_22[1].get("system_prompt")
        messages_22 = call_args_22.kwargs.get("messages") or call_args_22[1].get("messages")

        # Verify context properties at 20+ turns
        assert "SHAHID" in sys_prompt_22
        assert "The student's name is SHAHID" in sys_prompt_22
        # History messages are bounded (sliding window) + summary
        assert len(messages_22) <= 12
        # Check that older turns were summarized
        summary_msg = [m for m in messages_22 if m.get("role") == "system" and "SUMMARY" in m.get("content", "")]
        assert len(summary_msg) >= 1
        assert "PREVIOUS SESSION CONVERSATION SUMMARY" in summary_msg[0]["content"]


# =============================================================================
# 2. TEST SERVER RESTART / PERSISTENCE
# =============================================================================
def test_name_survives_simulated_server_restart():
    """
    Proves that student identity is stored in persistent SQLite, not ephemeral memory.
    1. Create user and set Name = Shahid
    2. Have turns
    3. Simulate server restart by creating a new database connection and new session
    4. Verify build_llm_context still retrieves Shahid
    """
    update_student(1, {
        "name": "Shahid",
        "onboarding_completed": True,
        "education": "Robotics",
        "learning_goals": "Master English presentations"
    })
    session_1 = create_session(student_id=1, mode="coach")

    # Add several messages to session 1
    for i in range(5):
        add_conversation_message(
            student_id=1,
            session_id=session_1["session_id"],
            role="user",
            content=f"Restart simulation turn {i}"
        )

    # Simulated server restart: Start a completely new session
    session_restart = create_session(student_id=1, mode="coach")
    new_session_id = session_restart["session_id"]
    assert new_session_id != session_1["session_id"]

    # Build context for the restarted server in the new session
    ctx = memory_manager.build_llm_context(
        student_id=1,
        session_id=new_session_id,
        current_message="What is my name?"
    )

    assert ctx["is_name_known"] is True
    assert ctx["student_name"] == "Shahid"
    assert "The student's name is SHAHID" in ctx["system_prompt"]
    assert "- Student Name: Shahid" in ctx["system_prompt"]
    # New session starts clean without old session dialogue polluting recent window
    assert ctx["history_count"] == 0


# =============================================================================
# 3. TEST CONTEXT TRUNCATION DOES NOT LOSE IDENTITY
# =============================================================================
def test_context_truncation_preserves_identity():
    """
    Tests that when a conversation exceeds the maximum sliding window (10 messages),
    the short-term history is truncated, older turns are summarized, and the student's
    name remains anchored in the persistent profile.
    """
    update_student(1, {
        "name": "Shahid",
        "onboarding_completed": True,
        "learning_goals": "Improve fluency"
    })
    session = create_session(student_id=1, mode="coach")
    session_id = session["session_id"]

    # Insert 24 conversation messages (12 turns)
    for i in range(12):
        add_conversation_message(
            student_id=1,
            session_id=session_id,
            role="user",
            content=f"Topic discussion turn {i + 1} regarding AI and robots."
        )
        add_conversation_message(
            student_id=1,
            session_id=session_id,
            role="assistant",
            content=f"AI insight for turn {i + 1}."
        )

    ctx = memory_manager.build_llm_context(
        student_id=1,
        session_id=session_id,
        current_message="Do you know who I am?",
        max_recent_messages=10
    )

    # Long-term identity is preserved
    assert ctx["student_name"] == "Shahid"
    assert "The student's name is SHAHID" in ctx["system_prompt"]

    # History is truncated to max_recent_messages
    assert ctx["history_count"] == 10
    assert ctx["history_truncated"] is True

    # Summary exists in messages
    summary_msgs = [m for m in ctx["messages"] if m.get("role") == "system" and "SUMMARY" in m.get("content", "")]
    assert len(summary_msgs) == 1
    assert "PREVIOUS SESSION CONVERSATION SUMMARY" in summary_msgs[0]["content"]

    # Total messages in context = 1 summary + 10 recent + 1 current = 12
    assert len(ctx["messages"]) == 12


# =============================================================================
# 4. PREVENT HALLUCINATED NAMES & PROTECT UNKNOWN IDENTITIES
# =============================================================================
def test_prevent_hallucinated_names_when_known_and_unknown():
    """
    Verifies that:
    1. If student name is known: context explicitly forbids calling them Alex, John, User, Text, Unknown.
    2. If student name is unknown: context explicitly instructs Mizo not to invent or hallucinate a name.
    """
    # 1. Known name: Shahid
    update_student(1, {"name": "Shahid", "onboarding_completed": True})
    ctx_known = memory_manager.build_llm_context(
        student_id=1,
        session_id="ses_test_known",
        current_message="Hello"
    )
    assert "The student's name is SHAHID" in ctx_known["system_prompt"]
    assert "NEVER invent, assume, or call the student by any other name" in ctx_known["system_prompt"]

    # 2. Unknown name: fresh learner
    delete_learner_data(1)
    ctx_unknown = memory_manager.build_llm_context(
        student_id=1,
        session_id="ses_test_unknown",
        current_message="Hello"
    )
    assert ctx_unknown["is_name_known"] is False
    assert "The student's name is currently unknown" in ctx_unknown["system_prompt"]
    assert "Never hallucinate or invent a name" in ctx_unknown["system_prompt"]


# =============================================================================
# 5. PROFILE UPDATE BEHAVIOR (Explicit name change)
# =============================================================================
def test_explicit_name_change_updates_persistent_memory():
    """
    If the student says 'My name is Mohammed', the system updates the persistent
    profile in SQLite, and all future contexts reflect Mohammed instead of the old name.
    """
    update_student(1, {
        "name": "Shahid",
        "onboarding_completed": True
    })
    st_initial = get_student(1)
    assert st_initial["name"] == "Shahid"

    # User explicitly changes name
    updated_st = personalization_service.extract_and_update_student_memory(1, "My name is Mohammed.")
    assert updated_st.get("name") == "Mohammed"

    # Verify SQLite was updated
    st_persisted = get_student(1)
    assert st_persisted["name"] == "Mohammed"

    # Future context builder must use Mohammed
    ctx = memory_manager.build_llm_context(
        student_id=1,
        session_id="ses_name_change",
        current_message="What is my name now?"
    )
    assert ctx["student_name"] == "Mohammed"
    assert "The student's name is MOHAMMED" in ctx["system_prompt"]
    assert "Shahid" not in ctx["system_prompt"]


# =============================================================================
# 6. MULTI-USER SESSION ISOLATION
# =============================================================================
def test_multi_user_sessions_do_not_mix_memory():
    """
    Student 1 (Shahid) and Student 2 (Aisha) must have strictly isolated profiles
    and conversation histories.
    """
    # Student 1: Shahid
    update_student(1, {
        "name": "Shahid",
        "onboarding_completed": True,
        "learning_goals": "Robotics communication"
    })
    session_1 = create_session(student_id=1, mode="coach")

    # Student 2: Aisha
    st2 = create_student({
        "name": "Aisha",
        "onboarding_completed": True,
        "education": "Biomedical Engineering",
        "learning_goals": "Medical research presentations"
    })
    st2_id = st2["id"]
    session_2 = create_session(student_id=st2_id, mode="coach")

    # Populate Student 1 messages
    add_conversation_message(student_id=1, session_id=session_1["session_id"], role="user", content="Robotics hardware query")
    add_conversation_message(student_id=1, session_id=session_1["session_id"], role="assistant", content="Robotics response")

    # Populate Student 2 messages
    add_conversation_message(student_id=st2_id, session_id=session_2["session_id"], role="user", content="DNA sequencing question")
    add_conversation_message(student_id=st2_id, session_id=session_2["session_id"], role="assistant", content="Biomedical response")

    # Build context for Student 1
    ctx1 = memory_manager.build_llm_context(
        student_id=1,
        session_id=session_1["session_id"],
        current_message="Check my project"
    )
    assert ctx1["student_name"] == "Shahid"
    assert "Robotics" in ctx1["system_prompt"]
    assert "Aisha" not in ctx1["system_prompt"]
    assert "Biomedical" not in ctx1["system_prompt"]
    assert any("Robotics hardware" in m["content"] for m in ctx1["messages"])
    assert not any("DNA sequencing" in m["content"] for m in ctx1["messages"])

    # Build context for Student 2
    ctx2 = memory_manager.build_llm_context(
        student_id=st2_id,
        session_id=session_2["session_id"],
        current_message="Check my research"
    )
    assert ctx2["student_name"] == "Aisha"
    assert "Biomedical" in ctx2["system_prompt"]
    assert "Shahid" not in ctx2["system_prompt"]
    assert "Robotics hardware" not in ctx2["system_prompt"]
    assert any("DNA sequencing" in m["content"] for m in ctx2["messages"])
    assert not any("Robotics hardware" in m["content"] for m in ctx2["messages"])
