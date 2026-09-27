"""
Tests verifying all 13 Final Acceptance Criteria for Mikaza Voice Interaction & Onboarding Flow
================================================================================================
Test 1: Standby + user says 'Hello.' -> Ignored, no response.
Test 2: Standby + user says 'What are you doing?' -> Ignored, no response.
Test 3: Standby + user says 'Hey Mikaza.' -> Activates to ACTIVE_LISTENING.
Test 4: Standby + user says 'Hey Megaza.' -> Activates on phonetic variant.
Test 5: After wake word: 'Teach me biology.' -> Subject Tutor selected.
Test 6: After wake word: 'Help me improve my communication.' -> Communication Coach selected.
Test 7: After wake word: 'I want to practice my speech.' -> Speech Coach selected.
Test 8: While speaking: 'Stop Mikaza.' -> Halts TTS immediately, returns to STANDBY.
Test 9: While speaking: User gives new instruction -> Barge-in interrupts TTS, transitions to ACTIVE_LISTENING.
Test 10: Standby lasts > 12 seconds -> Continuous monitoring does not terminate.
Test 11: Onboarding: 'What is your name?' -> 'Shahid' -> Natural dynamic acknowledgement + next question, waits.
Test 12: User does not answer onboarding question -> Prompts repetition without advancing state incorrectly.
Test 13: User answers onboarding question -> Stores answer, acknowledges naturally, asks next question after answer.
"""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import get_student, update_student, reset_student_to_fresh
from app.services.voice_service import voice_manager, VoiceState, is_wake_word, is_stop_command, run_continuous_standby_listener
from app.services.personalization_service import personalization_service
from app.services.llm_service import LLMResult

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test 1 & Test 2: In Standby mode, ordinary speech is ignored completely
# ---------------------------------------------------------------------------
def test_1_and_2_standby_ignores_ordinary_speech():
    update_student(1, {"onboarding_completed": True, "onboarding_step": "completed"})
    voice_manager.transition_to(VoiceState.STANDBY, "test_setup")

    # Test 1: User says "Hello."
    res1 = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Hello."
    })
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["reply_text"] == ""
    assert data1["audio_url"] is None
    assert voice_manager.state == VoiceState.STANDBY

    # Test 2: User says "What are you doing?"
    res2 = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "What are you doing?"
    })
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["reply_text"] == ""
    assert data2["audio_url"] is None
    assert voice_manager.state == VoiceState.STANDBY


# ---------------------------------------------------------------------------
# Test 3 & Test 4: Standby wake-word activation ("Hey Mikaza", "Hey Megaza")
# ---------------------------------------------------------------------------
def test_3_and_4_standby_activates_on_wake_phrase_and_phonetic_variants():
    update_student(1, {"onboarding_completed": True, "onboarding_step": "completed"})
    voice_manager.transition_to(VoiceState.STANDBY, "test_setup")

    # Test 3: Standby + "Hey Mikaza"
    res3 = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Hey Mikaza"
    })
    assert res3.status_code == 200
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING

    # Reset to Standby
    voice_manager.transition_to(VoiceState.STANDBY, "test_reset")

    # Test 4: Standby + "Hey Megaza" (Whisper phonetic variant)
    res4 = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Hey Megaza"
    })
    assert res4.status_code == 200
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING


# ---------------------------------------------------------------------------
# Test 5, 6, 7: Agent routing after activation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_5_6_7_agent_routing_after_wake_word():
    update_student(1, {"onboarding_completed": True, "onboarding_step": "completed"})

    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm, \
         patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        mock_llm.return_value = LLMResult(
            text="Let's study biology.",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False,
            latency_ms=100.0
        )
        mock_tts.return_value = ("/audio/test.wav", "test.wav")

        # Test 5: "Teach me biology." -> Subject Tutor
        voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "test_setup")
        res5 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "Teach me biology."
        })
        assert res5.status_code == 200
        assert res5.json()["learning_mode"] == "tutor"
        assert voice_manager.selected_agent == "Subject Tutor"

        # Test 6: "Help me improve my communication." -> Communication Coach
        voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "test_setup")
        res6 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "Help me improve my communication."
        })
        assert res6.status_code == 200
        assert res6.json()["learning_mode"] == "coach"
        assert voice_manager.selected_agent == "Communication Coach"

        # Test 7: "I want to practice my speech." -> Speech Coach
        voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "test_setup")
        res7 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "I want to practice my speech."
        })
        assert res7.status_code == 200
        assert res7.json()["learning_mode"] == "speech"
        assert voice_manager.selected_agent == "Speech Coach"


# ---------------------------------------------------------------------------
# Test 8: While speaking, 'Stop Mikaza' halts TTS and returns to STANDBY
# ---------------------------------------------------------------------------
def test_8_stop_mikaza_while_speaking():
    voice_manager.transition_to(VoiceState.SPEAKING, "tts_playing")
    assert voice_manager.state == VoiceState.SPEAKING

    res = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Stop Mikaza"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["is_control_command"] is True
    assert data["command"] == "stop"
    assert data["voice_state"] == "STANDBY"
    assert data["reply_text"] == ""
    assert voice_manager.state == VoiceState.STANDBY
    assert voice_manager.is_interrupted is True


# ---------------------------------------------------------------------------
# Test 9: While speaking, user barge-in halts speech and listens to user
# ---------------------------------------------------------------------------
def test_9_barge_in_interruption_during_tts():
    voice_manager.transition_to(VoiceState.SPEAKING, "tts_playing")

    # User speaks while Mikaza is speaking
    voice_manager.handle_barge_in("No, explain it more simply.")
    assert voice_manager.is_interrupted is True
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING


# ---------------------------------------------------------------------------
# Test 10: Standby lasts longer than 12 seconds without terminating
# ---------------------------------------------------------------------------
def test_10_standby_continuous_monitoring_indefinite():
    voice_manager.transition_to(VoiceState.STANDBY, "standby_start")

    # Simulate continuous listener audio callbacks over several cycles
    iterations = 0
    detected_wake = []

    def mock_audio_stream():
        nonlocal iterations
        iterations += 1
        if iterations == 1:
            return "hello"  # ordinary speech
        elif iterations == 2:
            return "what are you doing"  # ordinary speech
        elif iterations == 3:
            raise TimeoutError("Microphone silence window expired")  # should recover
        elif iterations == 4:
            return "hey mikaza"  # wake word
        return ""

    def on_wake(trailing):
        detected_wake.append(trailing)

    def should_stop():
        return len(detected_wake) > 0 or iterations >= 5

    # Run listener loop
    run_continuous_standby_listener(
        audio_transcribe_func=mock_audio_stream,
        on_wake_callback=on_wake,
        should_stop_func=should_stop
    )

    assert len(detected_wake) == 1
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING


# ---------------------------------------------------------------------------
# Test 11: Onboarding dynamic acknowledgement + next question (Shahid)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_11_onboarding_natural_acknowledgement_and_next_question():
    reset_student_to_fresh(1)
    student = get_student(1)
    assert student["onboarding_step"] == "ask_name"

    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm, \
         patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        mock_llm.return_value = LLMResult(
            text="That's a wonderful name, Shahid! What is your main goal with Mikaza?",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False,
            latency_ms=120.0
        )
        mock_tts.return_value = ("/audio/onboard1.wav", "onboard1.wav")

        res = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "Shahid"
        })
        assert res.status_code == 200
        data = res.json()

        st_updated = get_student(1)
        assert st_updated["name"] == "Shahid"
        assert st_updated["onboarding_step"] == "ask_goals"
        assert "That's a wonderful name" in data["reply_text"]
        assert "main goal" in data["reply_text"]


# ---------------------------------------------------------------------------
# Test 12: User does not answer onboarding question (silence handling)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_12_onboarding_silence_asks_to_repeat_without_advancing():
    reset_student_to_fresh(1)
    update_student(1, {"onboarding_step": "ask_goals", "name": "Shahid"})

    with patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:
        mock_tts.return_value = ("/audio/retry.wav", "retry.wav")

        # Simulate empty audio / silence from microphone
        res = client.post("/api/v1/esp32/audio?format=json", files={
            "audio": ("silence.wav", b"", "audio/wav")
        }, data={"student_id": "1"})

        assert res.status_code == 200
        data = res.json()
        assert data["onboarding_active"] is True
        assert data["onboarding_step"] == "ask_goals"  # Did NOT advance state incorrectly!
        assert "Could you say that again?" in data["reply_text"]


# ---------------------------------------------------------------------------
# Test 13: User answers onboarding question sequentially
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_13_onboarding_sequential_ask_wait_answer_pattern():
    reset_student_to_fresh(1)
    update_student(1, {"onboarding_step": "ask_goals", "name": "Shahid"})

    with patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm, \
         patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:

        mock_llm.return_value = LLMResult(
            text="Improving communication is a fantastic goal! What would you say is your biggest weakness when speaking?",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False,
            latency_ms=120.0
        )
        mock_tts.return_value = ("/audio/onboard2.wav", "onboard2.wav")

        res = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "I want to improve my communication skills."
        })
        assert res.status_code == 200
        data = res.json()

        st_updated = get_student(1)
        assert "communication" in st_updated["learning_goals"].lower()
        assert st_updated["onboarding_step"] == "ask_weaknesses"
        assert "weakness" in data["reply_text"].lower()


# ---------------------------------------------------------------------------
# Test 14: Automatic microphone readiness after every question
# ---------------------------------------------------------------------------
def test_14_automatic_mic_readiness_after_question():
    """
    Whenever Mikaza asks a question / finishes speaking, the voice state machine
    automatically transitions back to ACTIVE_LISTENING so the mic is immediately ready
    for the user's response without requiring 'Hey Mikaza'.
    """
    voice_manager.auto_listen_after_tts = True
    voice_manager.handle_speaking_start()
    assert voice_manager.state == VoiceState.SPEAKING

    # TTS playback finishes
    voice_manager.handle_speaking_ended()
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING


# ---------------------------------------------------------------------------
# Test 15: Always allow user interruption (Barge-in stops TTS immediately)
# ---------------------------------------------------------------------------
def test_15_interruption_stops_tts_and_listens():
    """
    If Mikaza is speaking and user starts speaking:
    User speech -> immediately stop TTS -> transition to ACTIVE_LISTENING.
    If user says 'Stop Mikaza', immediately halts and returns to STANDBY.
    """
    voice_manager.handle_speaking_start()
    assert voice_manager.state == VoiceState.SPEAKING

    # 1. Normal user speech barge-in
    voice_manager.handle_barge_in("I have a question about this.")
    assert voice_manager.is_interrupted is True
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING

    # 2. 'Stop Mikaza' barge-in command
    voice_manager.handle_speaking_start()
    assert voice_manager.state == VoiceState.SPEAKING
    voice_manager.handle_barge_in("Stop Mikaza")
    assert voice_manager.is_interrupted is True
    assert voice_manager.state == VoiceState.STANDBY


# ---------------------------------------------------------------------------
# Test 16: 10-second inactivity = SNOOZE, NOT mic shutdown (ignores speech until 'Hey Mikaza')
# ---------------------------------------------------------------------------
def test_16_snooze_mode_after_10s_inactivity_ignores_speech_until_wake_word():
    """
    When ~10s of inactivity occurs while waiting for user:
    - Transitions to SNOOZE/STANDBY.
    - Microphone monitoring remains active.
    - Normal speech is completely ignored (no LLM call, no reply, no state advancement).
    - Wake phrase 'Hey Mikaza' reactivates to ACTIVE_LISTENING.
    """
    reset_student_to_fresh(1)
    update_student(1, {"onboarding_step": "ask_goals", "name": "Shahid"})

    # Enter SNOOZE after 10s inactivity
    voice_manager.enter_snooze()
    assert voice_manager.state == VoiceState.STANDBY

    # User speaks normal speech during SNOOZE mode -> MUST BE IGNORED
    res_ignored = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "I want to improve communication."
    })
    assert res_ignored.status_code == 200
    data_ignored = res_ignored.json()
    assert data_ignored["reply_text"] == ""
    assert data_ignored["audio_url"] is None
    assert voice_manager.state == VoiceState.STANDBY

    # Student onboarding state is preserved (NOT modified)
    st = get_student(1)
    assert st["onboarding_step"] == "ask_goals"

    # User says wake phrase 'Hey Mikaza' -> Reactivates to ACTIVE_LISTENING
    res_wake = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "Hey Mikaza"
    })
    assert res_wake.status_code == 200
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING

    # Now that it's ACTIVE_LISTENING, user's answer is processed
    res_answer = client.post("/api/v1/esp32/chat", json={
        "student_id": 1,
        "message": "I want to improve my communication."
    })
    assert res_answer.status_code == 200
    data_ans = res_answer.json()
    assert data_ans["reply_text"] != ""
    st_after = get_student(1)
    assert st_after["onboarding_step"] == "ask_weaknesses"


# ---------------------------------------------------------------------------
# Test 17: Context-aware onboarding question answering
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_17_context_aware_onboarding_counter_question_reprompts_without_advancing():
    """
    When Mikaza asks 'What is your goal?' and user asks 'What is your name?':
    - System detects it does NOT answer the current question.
    - Mikaza does NOT store it as the goal.
    - Mikaza does NOT advance to the next question.
    - Mikaza responds naturally and asks the current question again.
    - Then when user gives a valid answer, it stores the answer and advances!
    """
    reset_student_to_fresh(1)
    update_student(1, {"onboarding_step": "ask_goals", "name": "Shahid", "learning_goals": ""})
    voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "test_setup")

    with patch("app.services.tts_service.tts_service.synthesize_to_file", new_callable=AsyncMock) as mock_tts:
        mock_tts.return_value = ("/audio/reprompt.wav", "reprompt.wav")

        # 1. User deflects / asks a counter-question: "What is your name?"
        res1 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "What is your name?"
        })
        assert res1.status_code == 200
        data1 = res1.json()

        st_check = get_student(1)
        # MUST NOT store "What is your name?" as the goal!
        assert "what is your name" not in (st_check.get("learning_goals") or "").lower()
        # MUST NOT advance state to ask_weaknesses!
        assert st_check["onboarding_step"] == "ask_goals"
        # Natural reprompt asked
        assert "goal" in data1["reply_text"].lower()

        # 2. User now answers the actual question with a valid goal
        res2 = client.post("/api/v1/esp32/chat", json={
            "student_id": 1,
            "message": "My main goal is to improve my speaking fluency and communication skills."
        })
        assert res2.status_code == 200
        data2 = res2.json()

        st_updated = get_student(1)
        # Stored the real goal!
        assert "fluency" in st_updated["learning_goals"].lower() or "communication" in st_updated["learning_goals"].lower()
        # Successfully advanced to the next question!
        assert st_updated["onboarding_step"] == "ask_weaknesses"

