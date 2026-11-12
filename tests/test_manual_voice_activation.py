import io
import wave
import struct
from pathlib import Path
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent

from app.main import app
from app.services.voice_service import voice_manager, VoiceState
from app.services.llm_service import LLMResult

client = TestClient(app)


def _create_dummy_wav():
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        frames = bytearray()
        for _ in range(8000):
            frames.extend(struct.pack('<h', 0))
        wf.writeframes(frames)
    wav_io.seek(0)
    return wav_io


@pytest.fixture(autouse=True)
def reset_voice_state():
    voice_manager.deactivate_manual_mode("test_fixture_reset")
    yield
    voice_manager.deactivate_manual_mode("test_fixture_teardown")


def test_manual_activate_endpoint_authentication_and_activation():
    """Manual activate endpoint requires valid authorization and activates state."""
    res = client.post(
        "/api/v1/esp32/manual_activate",
        json={"action": "activate", "timeout_seconds": 20.0},
        headers={"X-Device-Key": "mizo_esp32_secret_key_123"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["active"] is True
    assert data["voice_state"] == VoiceState.ACTIVE_LISTENING.value
    assert voice_manager.is_manual_activation_active() is True
    assert voice_manager.state == VoiceState.ACTIVE_LISTENING


def test_manual_activate_toggle_and_deactivate():
    """Manual activate endpoint can toggle and deactivate."""
    # Activate first
    client.post(
        "/api/v1/esp32/manual_activate",
        json={"action": "activate"},
        headers={"X-Device-Key": "mizo_esp32_secret_key_123"}
    )
    assert voice_manager.is_manual_activation_active() is True

    # Deactivate
    res = client.post(
        "/api/v1/esp32/manual_activate",
        json={"action": "deactivate"},
        headers={"X-Device-Key": "mizo_esp32_secret_key_123"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["active"] is False
    assert voice_manager.is_manual_activation_active() is False
    assert voice_manager.state == VoiceState.WAKE_LISTENING

    # Toggle back on
    res_toggle = client.post(
        "/api/v1/esp32/manual_activate",
        json={"action": "toggle"},
        headers={"X-Device-Key": "mizo_esp32_secret_key_123"}
    )
    assert res_toggle.status_code == 200
    assert res_toggle.json()["active"] is True
    assert voice_manager.is_manual_activation_active() is True


def test_manual_activate_timeout_expiry():
    """Manual activation expires safely after configured timeout."""
    voice_manager.activate_manual_mode(timeout_seconds=0.1)
    assert voice_manager.is_manual_activation_active() is True

    # Simulate time passing beyond timeout
    voice_manager._manual_activation_timestamp -= 2.0
    assert voice_manager.is_manual_activation_active() is False
    assert voice_manager.state == VoiceState.WAKE_LISTENING


def test_voice_state_endpoint_reports_manual_activation():
    """GET /voice_state reports manual_activation_active status."""
    voice_manager.activate_manual_mode(timeout_seconds=30.0)
    res = client.get("/api/v1/esp32/voice_state")
    assert res.status_code == 200
    data = res.json()
    assert data["manual_activation_active"] is True

    voice_manager.deactivate_manual_mode()
    res2 = client.get("/api/v1/esp32/voice_state")
    assert res2.status_code == 200
    assert res2.json()["manual_activation_active"] is False


@pytest.mark.asyncio
async def test_esp32_audio_processed_without_wake_word_when_manual_mode_active():
    """
    When manual activation is active, ESP32 audio without wake word is processed
    and returns a 16kHz WAV audio stream to the MAX98357A speaker.
    It then consumes the manual activation.
    """
    voice_manager.activate_manual_mode(timeout_seconds=30.0)
    assert voice_manager.is_manual_activation_active() is True

    wav_io = _create_dummy_wav()

    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:

        mock_stt.return_value = {
            "text": "Explain planetary motion",  # No "Hey Mizo" wake word!
            "duration_seconds": 1.5,
            "language": "en"
        }
        mock_llm.return_value = LLMResult(
            text="Planets orbit the sun in elliptical paths with the sun at one focus.",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False,
            latency_ms=120.0
        )

        files = {"audio": ("inmp441.wav", wav_io, "audio/wav")}
        data = {"student_id": "1", "mode": "coach"}
        res = client.post(
            "/api/v1/esp32/audio?format=audio",
            files=files,
            data=data,
            headers={"X-Device-Key": "mizo_esp32_secret_key_123"}
        )

        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/wav"

        # Verify manual activation was consumed
        assert voice_manager.is_manual_activation_active() is False


@pytest.mark.asyncio
async def test_esp32_audio_ignored_without_wake_word_when_manual_mode_inactive():
    """
    When manual activation is inactive and in standby, audio without wake word
    is ignored and returns a JSON response with ignored: True.
    """
    voice_manager.deactivate_manual_mode()
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_idle")
    assert voice_manager.is_manual_activation_active() is False

    wav_io = _create_dummy_wav()

    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.config.settings.HARDWARE_TEST_MODE", False):
        mock_stt.return_value = {
            "text": "Explain planetary motion",  # No wake word in idle state
            "duration_seconds": 1.5,
            "language": "en"
        }

        files = {"audio": ("inmp441.wav", wav_io, "audio/wav")}
        data = {"student_id": "1", "mode": "coach"}
        res = client.post(
            "/api/v1/esp32/audio?format=audio&test_mode=false",
            files=files,
            data=data,
            headers={"X-Device-Key": "mizo_esp32_secret_key_123"}
        )

        assert res.status_code == 200
        assert "application/json" in res.headers["content-type"]
        body = res.json()
        assert body["ignored"] is True
        assert body["reply_text"] == ""


def test_hardware_test_mode_status_endpoint():
    """GET /test_mode and GET /audio return hardware test mode status."""
    res = client.get("/api/v1/esp32/test_mode")
    assert res.status_code == 200
    assert "hardware_test_mode" in res.json()

    res_audio = client.get("/api/v1/esp32/audio")
    assert res_audio.status_code == 200
    assert "hardware_test_mode" in res_audio.json()


@pytest.mark.asyncio
async def test_hardware_test_mode_bypasses_wake_word_and_returns_wav(capsys):
    """
    In hardware test mode, any audio from ESP32 is processed directly without 'Mizo'.
    Clear logs for Stages 1 to 4 are output.
    Returns 16kHz WAV audio stream for MAX98357A speaker.
    """
    wav_io = _create_dummy_wav()

    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm, \
         patch("app.config.settings.HARDWARE_TEST_MODE", True):

        mock_stt.return_value = {
            "text": "Testing external microphone input directly",
            "duration_seconds": 2.0,
            "language": "en"
        }
        mock_llm.return_value = LLMResult(
            text="I hear you clearly through the external microphone. Testing speaker now.",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False,
            latency_ms=80.0
        )

        files = {"audio": ("inmp441.wav", wav_io, "audio/wav")}
        data = {"student_id": "1"}
        res = client.post(
            "/api/v1/esp32/audio?format=audio",
            files=files,
            data=data,
            headers={"X-Device-Key": "mizo_esp32_secret_key_123"}
        )

        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/wav"

        out = capsys.readouterr().out
        assert "Stage 1/4: Audio received" in out
        assert "Stage 2/4: Transcription completed" in out
        assert "Stage 3/4: AI response generated" in out
        assert "Stage 4/4: Response audio returned" in out
        assert "Persisted conversation messages" in out

        # Verify messages are retrievable by existing conversation-history API
        hist = client.get("/api/v1/conversations?student_id=1&limit=50").json()
        assert len(hist) >= 2
        user_msgs = [m for m in hist if m["role"] == "user" and "Testing external microphone" in m["content"]]
        asst_msgs = [m for m in hist if m["role"] == "assistant" and "Testing speaker now" in m["content"]]
        assert len(user_msgs) == 1
        assert len(asst_msgs) == 1
        assert user_msgs[0]["session_id"] == asst_msgs[0]["session_id"]


@pytest.mark.asyncio
async def test_hardware_test_mode_conversations_saved_and_searchable_on_dashboard():
    """
    Verifies that conversations sent via ESP32 hardware test mode:
    1. Are saved with both user transcript and assistant reply.
    2. Are associated with the active session without creating new sessions every time.
    3. Are retrievable and searchable through GET /api/v1/conversations.
    4. Do not store duplicate messages on quick retries.
    5. Do not store empty/failed recognition as conversation messages.
    """
    wav_io = _create_dummy_wav()

    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm, \
         patch("app.config.settings.HARDWARE_TEST_MODE", True):

        mock_stt.return_value = {
            "text": "What is the speed of sound?",
            "duration_seconds": 1.8,
            "language": "en"
        }
        mock_llm.return_value = LLMResult(
            text="The speed of sound in dry air at 20 degrees Celsius is approximately 343 meters per second.",
            provider="ollama",
            model="llama3:latest",
            is_fallback=False,
            latency_ms=95.0
        )

        files = {"audio": ("sound_test.wav", wav_io, "audio/wav")}
        data = {"student_id": "1"}
        res = client.post(
            "/api/v1/esp32/audio?format=audio",
            files=files,
            data=data,
            headers={"X-Device-Key": "mizo_esp32_secret_key_123"}
        )
        assert res.status_code == 200

        # Verify messages appear in dashboard history
        hist = client.get("/api/v1/conversations?student_id=1&limit=50").json()
        matching_user = [m for m in hist if m["role"] == "user" and "speed of sound" in m["content"]]
        matching_asst = [m for m in hist if m["role"] == "assistant" and "343 meters per second" in m["content"]]
        assert len(matching_user) == 1
        assert len(matching_asst) == 1
        active_sess_id = matching_user[0]["session_id"]
        assert matching_asst[0]["session_id"] == active_sess_id
        assert matching_asst[0]["provider_used"] == "ollama (llama3:latest)"

        # Search query support (used by dashboard search box)
        search_res = client.get("/api/v1/conversations?student_id=1&search=343").json()
        assert len(search_res) >= 1
        assert "343 meters per second" in search_res[0]["content"]

        # Duplicate retry test: sending the exact same transcript again should NOT duplicate
        wav_io_retry = _create_dummy_wav()
        res_retry = client.post(
            "/api/v1/esp32/audio?format=audio",
            files={"audio": ("sound_test_retry.wav", wav_io_retry, "audio/wav")},
            data={"student_id": "1"},
            headers={"X-Device-Key": "mizo_esp32_secret_key_123"}
        )
        assert res_retry.status_code == 200
        hist_after_retry = client.get("/api/v1/conversations?student_id=1&limit=50").json()
        matching_user_retry = [m for m in hist_after_retry if m["role"] == "user" and "speed of sound" in m["content"]]
        assert len(matching_user_retry) == 1  # Not duplicated!

        # Silence / missing speech test: should NOT persist empty conversation
        wav_io_silence = _create_dummy_wav()
        mock_stt.return_value = {
            "text": "",
            "duration_seconds": 0.5,
            "language": "en"
        }
        res_silence = client.post(
            "/api/v1/esp32/audio?format=audio",
            files={"audio": ("silence.wav", wav_io_silence, "audio/wav")},
            data={"student_id": "1"},
            headers={"X-Device-Key": "mizo_esp32_secret_key_123"}
        )
        assert res_silence.status_code == 200
        hist_after_silence = client.get("/api/v1/conversations?student_id=1&limit=50").json()
        assert len(hist_after_silence) == len(hist_after_retry)  # No extra messages persisted


def test_dashboard_frontend_polling_and_scroll_preservation_contract():
    """
    Verifies frontend dashboard implementation for conversation auto-refresh:
    1. index.html references cache-busted script /js/app.js?v=3.0.3.
    2. app.js contains dedicated initConversationPolling with overlap prevention (_isPollingConversations).
    3. app.js pauses unnecessary polling when inactive (document.hidden) and resumes on visibilitychange.
    4. app.js preserves scroll position (wasNearBottom check) and deduplicates message IDs.
    """
    index_path = BASE_DIR / "app" / "static" / "index.html"
    app_js_path = BASE_DIR / "app" / "static" / "js" / "app.js"

    assert index_path.exists()
    assert app_js_path.exists()

    index_html = index_path.read_text(encoding="utf-8")
    app_js = app_js_path.read_text(encoding="utf-8")

    # Script tag cache-buster
    assert '<script src="/js/app.js?v=3.0.' in index_html

    # Dedicated polling engine functions & safeguards
    assert "initConversationPolling" in app_js
    assert "_isPollingConversations" in app_js
    assert "document.hidden" in app_js
    assert "visibilitychange" in app_js

    # Scroll position preservation & ID deduplication
    assert "wasNearBottom" in app_js
    assert "chatBox.scrollHeight" in app_js
    assert "seenIds" in app_js
    assert "uniqueLogs" in app_js


@pytest.mark.asyncio
async def test_conversation_polling_api_contract_and_duplicate_safety():
    """
    Verifies that the /api/v1/conversations endpoint behaves consistently under rapid polling:
    1. Returns 200 with chronological array.
    2. When new ESP32 messages arrive, next poll reflects updated count and new latest ID.
    3. Handles consecutive rapid poll requests without errors or duplicate messages.
    """
    # 1. Initial poll
    res1 = client.get("/api/v1/conversations?student_id=1&limit=50")
    assert res1.status_code == 200
    initial_logs = res1.json()

    # 2. Add an ESP32 hardware turn
    wav_io = _create_dummy_wav()
    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm, \
         patch("app.config.settings.HARDWARE_TEST_MODE", True):

        mock_stt.return_value = {
            "text": "Tell me a fun physics fact",
            "duration_seconds": 1.5,
            "language": "en"
        }
        mock_llm.return_value = LLMResult(
            text="Light from the sun takes approximately 8 minutes and 20 seconds to reach Earth.",
            provider="ollama",
            model="llama3:latest",
            is_fallback=False,
            latency_ms=60.0
        )

        res_audio = client.post(
            "/api/v1/esp32/audio?format=audio",
            files={"audio": ("fact.wav", wav_io, "audio/wav")},
            data={"student_id": "1"},
            headers={"X-Device-Key": "mizo_esp32_secret_key_123"}
        )
        assert res_audio.status_code == 200

    # 3. Subsequent polls simulate periodic polling
    res2 = client.get("/api/v1/conversations?student_id=1&limit=50")
    assert res2.status_code == 200
    updated_logs = res2.json()
    assert len(updated_logs) >= len(initial_logs) + 2

    # Latest messages must include user query and AI response
    assert any("fun physics fact" in m["content"] for m in updated_logs)
    assert any("8 minutes and 20 seconds" in m["content"] for m in updated_logs)

    # 4. Rapid back-to-back polls return identical, deduplicated payloads
    res3 = client.get("/api/v1/conversations?student_id=1&limit=50")
    assert res3.status_code == 200
    assert len(res3.json()) == len(updated_logs)
    assert [m["id"] for m in res3.json()] == [m["id"] for m in updated_logs]

