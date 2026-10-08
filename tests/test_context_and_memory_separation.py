"""
Test Suite: Long-Term Memory, Conversation Context Separation & Topic Tracking
==============================================================================
Validates Sections 21-33:
- Section 21: Separation of Long-Term Profile, Conversation Context, and Mode
- Section 22: Topic & Entity Tracking
- Section 23: User Correction Handling
- Section 24: Language Detection (No false positives on English test samples)
- Section 25: STOP Command Priority & Non-English Handling
- Section 26: Lightweight Rule-Based Intent Detection
- Section 27: Selective Memory Retrieval (Topic-relevant only)
- Section 28: Durable Memory Creation
- Section 29: Context Continuity vs Memory Retrieval
- Section 30: Zero Fabricated Context
- Section 32: Direct Answers without Unsolicited Redirection
"""

import pytest
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
from app.services.voice_service import is_stop_command
from app.api.esp32 import is_non_english_input
from app.services.llm_service import llm_service
from app.services.memory_service import memory_manager


def test_section_24_english_phrases_never_rejected():
    """Validates that all specified English phrases are recognized as English."""
    mandatory_english_phrases = [
        "Hello everyone!",
        "1 kg of chicken breast",
        "I will prepare the vegetables for the sauce.",
        "Hello everyone, welcome to my channel.",
        "Hi, are you able to understand now?"
    ]
    for phrase in mandatory_english_phrases:
        # Test without STT metadata
        is_non_eng, reason = is_non_english_input(phrase)
        assert not is_non_eng, f"Phrase incorrectly marked as non-English: '{phrase}' (reason: {reason})"

        # Test even if Whisper hallucinates non-English language metadata
        is_non_eng_stt, reason_stt = is_non_english_input(phrase, audio_metadata={"language": "es"})
        assert not is_non_eng_stt, f"Phrase rejected due to STT metadata hallucination: '{phrase}' (reason: {reason_stt})"


def test_section_25_multilingual_stop_commands_priority():
    """Validates that STOP commands (including Urdu) are recognized and not treated as foreign rejection."""
    stop_phrases = [
        "میزو سٹاپ",
        "سٹاپ",
        "ruko",
        "mizo stop",
        "stop mizo"
    ]
    for p in stop_phrases:
        assert is_stop_command(p), f"Failed to recognize stop command: '{p}'"
        is_non_eng, _ = is_non_english_input(p)
        assert not is_non_eng, f"Stop command '{p}' should not be rejected as non-English speech"


def test_section_25_genuine_non_english_detection():
    """Validates that genuine non-English is detected."""
    foreign_phrases = [
        "terima kasih banyak selamat pagi",
        "hola como estas amigo mio",
        "bonjour comment allez vous"
    ]
    for fp in foreign_phrases:
        is_non_eng, _ = is_non_english_input(fp)
        assert is_non_eng, f"Genuine foreign phrase was not detected: '{fp}'"


def test_section_26_lightweight_intent_detection():
    """Validates deterministic intent classification."""
    # Question
    assert detect_conversation_intent("Who are the Straw Hat members?") == "QUESTION"
    assert detect_conversation_intent("Can you explain how this works?") == "QUESTION"

    # Correction (Section 23)
    assert detect_conversation_intent("You forgot Jinbe.") == "USER_CORRECTION"
    assert detect_conversation_intent("That's wrong.") == "USER_CORRECTION"
    assert detect_conversation_intent("No, I meant Luffy.") == "USER_CORRECTION"

    # Context query (Section 29)
    assert detect_conversation_intent("What were we talking about?") == "CONTEXT_QUERY"
    assert detect_conversation_intent("What was our topic?") == "CONTEXT_QUERY"

    # Memory query (Section 29)
    assert detect_conversation_intent("Do you remember my favorite character?") == "MEMORY_QUERY"
    assert detect_conversation_intent("What is my favorite character?") == "MEMORY_QUERY"

    # Confusion / Clarification
    assert detect_conversation_intent("I'm confused.") == "CONFUSION"
    assert detect_conversation_intent("Could you clarify that?") == "CLARIFICATION"

    # Topic change
    assert detect_conversation_intent("Let's talk about something else.") == "TOPIC_CHANGE"


def test_section_28_durable_memory_creation():
    """Validates that only explicit durable facts are captured."""
    # Explicit durable facts
    fact1 = extract_durable_facts("My favorite character is Zoro.")
    assert fact1 is not None
    assert fact1[0] == "favorite_things"
    assert fact1[1] == "favorite_character"
    assert "Zoro" in fact1[2]

    fact2 = extract_durable_facts("My favorite anime is One Piece.")
    assert fact2 is not None
    assert fact2[1] == "favorite_anime"
    assert "One Piece" in fact2[2]

    # Casual remarks must NOT create durable memories
    assert extract_durable_facts("I think Zoro is cool.") is None
    assert extract_durable_facts("I am watching TV.") is None
    assert extract_durable_facts("It is raining outside.") is None


def test_section_22_topic_tracking_and_corrections():
    """Validates topic tracking and correction flow across multiple turns."""
    sess = create_session(student_id=PROTOTYPE_STUDENT_ID, mode="coach")
    session_id = sess["session_id"]

    # Turn 1: "My favorite character is Zoro."
    ctx1 = update_conversation_topic_state(
        session_id=session_id,
        student_id=PROTOTYPE_STUDENT_ID,
        user_text="My favorite character is Zoro."
    )
    assert ctx1["current_topic"] == "anime"
    assert ctx1["current_subtopic"] == "One Piece"
    assert "Zoro" in ctx1["current_entities"]

    # Verify durable memory was saved to student profile
    st = get_student(PROTOTYPE_STUDENT_ID)
    assert st["favorite_things"].get("favorite_character") == "Zoro"

    # Turn 2: "Who are the main crew members?"
    ctx2 = update_conversation_topic_state(
        session_id=session_id,
        student_id=PROTOTYPE_STUDENT_ID,
        user_text="Who are the main crew members?"
    )
    assert ctx2["current_topic"] == "anime"
    assert ctx2["current_subtopic"] == "One Piece"
    assert "Straw Hat Pirates" in ctx2["current_entities"]

    # Turn 3: "You forgot Jinbe."
    ctx3 = update_conversation_topic_state(
        session_id=session_id,
        student_id=PROTOTYPE_STUDENT_ID,
        user_text="You forgot Jinbe."
    )
    assert ctx3["last_user_intent"] == "USER_CORRECTION"
    assert ctx3["current_topic"] == "anime"
    assert ctx3["current_subtopic"] == "One Piece"
    assert "Jinbe" in ctx3["current_entities"]
    assert "You forgot Jinbe." in ctx3["last_user_correction"]


def test_section_21_and_27_selective_memory_retrieval():
    """
    Validates that when the topic is One Piece, unrelated profile interests
    (like robotics or ESP32) are excluded from the prompt context.
    """
    # Set student with diverse long-term interests
    update_student(PROTOTYPE_STUDENT_ID, {
        "interests": "Robotics, Artificial Intelligence, ESP32, Storytelling",
        "favorite_things": {"favorite_character": "Zoro", "favorite_anime": "One Piece", "favorite_hobby": "Robotics"}
    })
    st = get_student(PROTOTYPE_STUDENT_ID)

    # Filter for One Piece topic
    filtered = filter_relevant_memories(st, current_topic="anime", current_subtopic="One Piece")
    favs = filtered["relevant_favorites"]
    assert "favorite_character" in favs
    assert favs["favorite_character"] == "Zoro"
    assert "favorite_hobby" not in favs  # Robotics hobby excluded!

    # Filter for Robotics topic
    filtered_robotics = filter_relevant_memories(st, current_topic="robotics", current_subtopic="Mizo robot")
    favs_rob = filtered_robotics["relevant_favorites"]
    assert "favorite_hobby" in favs_rob
    assert "favorite_character" not in favs_rob  # Anime character excluded!


def test_section_29_context_continuity_vs_memory_query():
    """
    Validates Tier 0 resolution:
    - 'What were we talking about?' -> conversation context
    - 'Do you remember my favorite character?' -> long-term memory
    """
    sess = create_session(student_id=PROTOTYPE_STUDENT_ID, mode="coach")
    session_id = sess["session_id"]

    # Set topic state
    update_session_context(session_id, {
        "current_topic": "anime",
        "current_subtopic": "One Piece",
        "current_entities": ["Zoro", "Straw Hat Pirates"]
    })

    # Set student profile
    update_student(PROTOTYPE_STUDENT_ID, {
        "name": "Shahid",
        "favorite_things": {"favorite_character": "Zoro"}
    })
    st = get_student(PROTOTYPE_STUDENT_ID)

    # 1. Context query
    res_context = llm_service.resolve_tier0("What were we talking about?", student_data=st, session_id=session_id)
    assert res_context is not None
    assert res_context["handled"] is True
    assert "One Piece" in res_context["reply_text"]
    assert "Zoro" in res_context["reply_text"] or "Straw Hat Pirates" in res_context["reply_text"]

    # 2. Memory query (Favorite character)
    res_mem = llm_service.resolve_tier0("Do you remember my favorite character?", student_data=st, session_id=session_id)
    assert res_mem is not None
    assert res_mem["handled"] is True
    assert "Zoro" in res_mem["reply_text"]

    # 3. Memory query when not yet recorded (Section 30: Zero fabrication)
    st_empty = dict(st)
    st_empty["favorite_things"] = {}
    res_mem_empty = llm_service.resolve_tier0("Do you remember my favorite character?", student_data=st_empty, session_id=session_id)
    assert res_mem_empty is not None
    assert res_mem_empty["handled"] is True
    assert "don't have your favorite character" in res_mem_empty["reply_text"].lower()


def test_section_21_build_llm_context_priority_directives():
    """Validates that build_llm_context includes the topic priority and correction rules."""
    sess = create_session(student_id=PROTOTYPE_STUDENT_ID, mode="coach")
    session_id = sess["session_id"]

    update_session_context(session_id, {
        "current_topic": "anime",
        "current_subtopic": "One Piece",
        "current_entities": ["Straw Hat Pirates", "Zoro"]
    })

    ctx = memory_manager.build_llm_context(
        student_id=PROTOTYPE_STUDENT_ID,
        session_id=session_id,
        current_message="Who are the main crew members?",
        mode="coach"
    )

    sys_prompt = ctx["system_prompt"]
    assert "CURRENT CONVERSATION CONTEXT (TOPIC PRIORITY OVER PROFILE)" in sys_prompt
    assert "Active Subtopic: One Piece" in sys_prompt
    assert "CURRENT CONVERSATION TOPIC MUST TAKE PRIORITY" in sys_prompt
    assert "DIRECT FACTUAL ANSWERS" in sys_prompt
    assert "USER CORRECTION HANDLING" in sys_prompt
