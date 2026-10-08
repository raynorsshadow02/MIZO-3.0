"""
Test Suite: Normal Communication Agent (Mizo Task 1)
=====================================================
Validates all 14 requirements for the Normal Communication Coach:
1. Normal greeting.
2. Casual conversation.
3. Current-topic continuity.
4. Topic change.
5. User correction.
6. "I don't understand" -> simplified response.
7. "Make it simpler" -> simplified response.
8. Relevant memory retrieval.
9. Irrelevant memory is NOT injected.
10. Clearly understandable English is not rejected.
11. Simple communication does not use Groq.
12. Stop remains deterministic.
13. Existing name remains available.
14. Normal conversation does not restart onboarding.
"""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import (
    get_student,
    update_student,
    create_session,
    get_session_context,
    update_session_context,
    PROTOTYPE_STUDENT_ID
)
from app.services.context_service import (
    detect_conversation_intent,
    extract_durable_facts,
    extract_topic_and_entities,
    update_conversation_topic_state,
    filter_relevant_memories
)
from app.services.personalization_service import personalization_service
from app.services.llm_service import llm_service, LLMService
from app.services.memory_service import memory_manager
from app.services.voice_service import is_stop_command
from app.api.esp32 import is_non_english_input

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_prototype_learner():
    """Ensure student #1 has an active onboarded profile before each test."""
    update_student(PROTOTYPE_STUDENT_ID, {
        "name": "Shahid",
        "onboarding_completed": True,
        "onboarding_step": "completed",
        "current_teaching_difficulty": 2,
        "interests": "Robotics, Artificial Intelligence, Anime, Storytelling",
        "favorite_things": {"favorite_character": "Zoro", "favorite_anime": "One Piece", "favorite_hobby": "Robotics"}
    })


# -----------------------------------------------------------------------------
# 1. Normal Greeting
# -----------------------------------------------------------------------------
def test_1_normal_greeting_routes_ollama_not_groq():
    """Validates that a normal greeting routes to Ollama, not Groq."""
    intent, complexity, primary, cascade = llm_service.classify_route(
        messages=[{"role": "user", "content": "Hi"}],
        system_prompt="system"
    )
    assert primary == "ollama"
    assert "groq" not in cascade


# -----------------------------------------------------------------------------
# 2. Casual Conversation
# -----------------------------------------------------------------------------
def test_2_casual_conversation_routes_simple():
    """Validates that casual conversation turns route to Ollama."""
    intent, complexity, primary, cascade = llm_service.classify_route(
        messages=[{"role": "user", "content": "I had a really busy and productive day today."}],
        system_prompt="system"
    )
    assert primary == "ollama"
    assert "groq" not in cascade


# -----------------------------------------------------------------------------
# 3. Current-Topic Continuity
# -----------------------------------------------------------------------------
def test_3_current_topic_continuity():
    """Validates that discussion stays on One Piece across turns."""
    sess = create_session(student_id=PROTOTYPE_STUDENT_ID, mode="coach")
    session_id = sess["session_id"]

    ctx1 = update_conversation_topic_state(
        session_id=session_id,
        student_id=PROTOTYPE_STUDENT_ID,
        user_text="I like One Piece."
    )
    assert ctx1["current_subtopic"] == "One Piece"

    ctx2 = update_conversation_topic_state(
        session_id=session_id,
        student_id=PROTOTYPE_STUDENT_ID,
        user_text="Who are the Straw Hat members?"
    )
    assert ctx2["current_subtopic"] == "One Piece"
    assert "Straw Hat Pirates" in ctx2["current_entities"]


# -----------------------------------------------------------------------------
# 4. Topic Change
# -----------------------------------------------------------------------------
def test_4_topic_change_clears_previous_topic():
    """Validates that 'Let's talk about something else' resets current topic."""
    sess = create_session(student_id=PROTOTYPE_STUDENT_ID, mode="coach")
    session_id = sess["session_id"]

    # Start topic
    update_conversation_topic_state(
        session_id=session_id,
        student_id=PROTOTYPE_STUDENT_ID,
        user_text="Who are the Straw Hat members?"
    )

    # Change topic
    ctx_change = update_conversation_topic_state(
        session_id=session_id,
        student_id=PROTOTYPE_STUDENT_ID,
        user_text="Let's talk about something else."
    )
    assert ctx_change["last_user_intent"] == "TOPIC_CHANGE"
    assert ctx_change["current_subtopic"] == ""
    assert ctx_change["current_topic"] == ""
    assert ctx_change["current_entities"] == []


# -----------------------------------------------------------------------------
# 5. User Correction
# -----------------------------------------------------------------------------
def test_5_user_correction_detected():
    """Validates user correction intent and context capture."""
    sess = create_session(student_id=PROTOTYPE_STUDENT_ID, mode="coach")
    session_id = sess["session_id"]

    ctx = update_conversation_topic_state(
        session_id=session_id,
        student_id=PROTOTYPE_STUDENT_ID,
        user_text="You forgot Jinbe."
    )
    assert ctx["last_user_intent"] == "USER_CORRECTION"
    assert "You forgot Jinbe." in ctx["last_user_correction"]


# -----------------------------------------------------------------------------
# 6. "I don't understand" -> Simplified Response
# -----------------------------------------------------------------------------
def test_6_i_dont_understand_triggers_simplification():
    """Validates that 'I don't understand' triggers confusion and lowers difficulty."""
    is_confused, severity = personalization_service.is_learner_confused("I don't understand what you mean. Explain simply.")
    assert is_confused is True

    diff = personalization_service.adapt_teaching_difficulty(
        student_id=PROTOTYPE_STUDENT_ID,
        user_utterance="I don't understand what you mean. Explain simply."
    )
    assert diff <= 2


# -----------------------------------------------------------------------------
# 7. "Make it simpler" -> Simplified Response
# -----------------------------------------------------------------------------
def test_7_make_it_simpler_triggers_simplification():
    """Validates that 'Make it simpler' triggers major simplification."""
    is_confused, severity = personalization_service.is_learner_confused("Make it simpler, please.")
    assert is_confused is True
    assert severity == "major"


# -----------------------------------------------------------------------------
# 8. Relevant Memory Retrieval
# -----------------------------------------------------------------------------
def test_8_relevant_memory_retrieval():
    """Validates that when topic is One Piece, favorite character Zoro is retrieved."""
    st = get_student(PROTOTYPE_STUDENT_ID)
    filtered = filter_relevant_memories(st, current_topic="anime", current_subtopic="One Piece")
    assert filtered["relevant_favorites"].get("favorite_character") == "Zoro"


# -----------------------------------------------------------------------------
# 9. Irrelevant Memory is NOT Injected
# -----------------------------------------------------------------------------
def test_9_irrelevant_memory_not_injected():
    """Validates that robotics/ESP32 is omitted from favorites when topic is One Piece."""
    st = get_student(PROTOTYPE_STUDENT_ID)
    filtered = filter_relevant_memories(st, current_topic="anime", current_subtopic="One Piece")
    assert "favorite_hobby" not in filtered["relevant_favorites"]
    assert "Robotics" not in filtered["relevant_favorites"].values()


# -----------------------------------------------------------------------------
# 10. Clearly Understandable English is Not Rejected
# -----------------------------------------------------------------------------
def test_10_clearly_understandable_english_not_rejected():
    """Validates that valid English sentences are never rejected as foreign speech."""
    for phrase in [
        "Hello everyone!",
        "1 kg of chicken breast",
        "I will prepare the vegetables for the sauce.",
        "Hello everyone, welcome to my channel."
    ]:
        is_non_eng, _ = is_non_english_input(phrase)
        assert not is_non_eng, f"English phrase was rejected: {phrase}"


# -----------------------------------------------------------------------------
# 11. Simple Communication Does Not Use Groq
# -----------------------------------------------------------------------------
def test_11_simple_communication_does_not_use_groq():
    """Validates that greetings, thank you, simple questions do not use Groq."""
    for phrase in ["thank you", "hello", "who is Luffy?", "how are you today?"]:
        _, _, primary, cascade = llm_service.classify_route(
            messages=[{"role": "user", "content": phrase}],
            system_prompt="system"
        )
        assert primary != "groq"
        assert "groq" not in cascade


# -----------------------------------------------------------------------------
# 12. Stop Remains Deterministic
# -----------------------------------------------------------------------------
def test_12_stop_remains_deterministic():
    """Validates that Stop commands are deterministic Tier 0."""
    res_en = llm_service.resolve_tier0("Stop Mizo")
    assert res_en is not None
    assert res_en["handled"] is True
    assert res_en["provider"] == "system"

    # Multilingual stop
    assert is_stop_command("میزو سٹاپ") is True


# -----------------------------------------------------------------------------
# 13. Existing Name Remains Available
# -----------------------------------------------------------------------------
def test_13_existing_name_remains_available():
    """Validates that student name is deterministically retrieved via Tier 0."""
    st = get_student(PROTOTYPE_STUDENT_ID)
    res = llm_service.resolve_tier0("What is my name?", student_data=st)
    assert res is not None
    assert res["handled"] is True
    assert "Shahid" in res["reply_text"]


# -----------------------------------------------------------------------------
# 14. Normal Conversation Does Not Restart Onboarding
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_14_normal_conversation_does_not_restart_onboarding():
    """Validates that chatting with an onboarded student preserves onboarding_completed: True."""
    st_before = get_student(PROTOTYPE_STUDENT_ID)
    assert st_before["onboarding_completed"] is True

    with patch.object(LLMService, "_call_ollama", new_callable=AsyncMock) as mock_ollama, \
         patch("app.services.tts_service.TTSService.synthesize_to_file", new_callable=AsyncMock) as mock_tts:
        mock_ollama.return_value = "One Piece is a great anime!"
        mock_tts.return_value = ("/audio/mock.wav", "dummy.wav")

        res = client.post("/api/v1/esp32/chat", json={
            "message": "I really enjoy watching One Piece.",
            "is_typed_text": True
        })
        assert res.status_code == 200
        data = res.json()
        assert data["onboarding_active"] is False
        assert data["onboarding_step"] == "completed"

    st_after = get_student(PROTOTYPE_STUDENT_ID)
    assert st_after["onboarding_completed"] is True
    assert st_after["onboarding_step"] == "completed"
