import io
import wave
import struct
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.api.esp32 import _esp32_hardware_state
from app.services.llm_service import LLMResult

client = TestClient(app)


def test_both_audio_modes_present_in_dashboard():
    """Requirement 1: Both ESP32 Robot and Laptop Audio modes must be present in dashboard HTML."""
    index_path = Path("app/static/index.html")
    assert index_path.exists(), "app/static/index.html must exist"
    content = index_path.read_text(encoding="utf-8")

    # Verify mode selector section and cards
    assert "audio-mode-selector-section" in content
    assert "card-mode-esp32" in content
    assert "card-mode-laptop" in content
    assert "ESP32 Robot" in content
    assert "Laptop Audio" in content
    assert "esp32-mode-status" in content
    assert "laptop-mode-status" in content


def test_audio_mode_get_and_set_endpoints():
    """Requirement 2: Mode persistence and server validation via admin and audio endpoints."""
    # Test admin settings GET
    res = client.get("/api/v1/admin/settings/audio_mode")
    assert res.status_code == 200
    data = res.json()
    assert "audio_mode" in data
    assert data["audio_mode"] in ["esp32", "laptop"]

    # Test setting to laptop mode
    res = client.post("/api/v1/admin/settings/audio_mode", json={"audio_mode": "laptop"})
    assert res.status_code == 200
    assert res.json()["audio_mode"] == "laptop"

    # Verify persisted mode
    res = client.get("/api/v1/admin/settings/audio_mode")
    assert res.json()["audio_mode"] == "laptop"

    # Test dedicated /api/v1/audio/mode endpoint
    res = client.get("/api/v1/audio/mode")
    assert res.status_code == 200
    assert res.json()["audio_mode"] == "laptop"

    # Test setting back to esp32 mode
    res = client.post("/api/v1/audio/mode", json={"mode": "esp32"})
    assert res.status_code == 200
    assert res.json()["audio_mode"] == "esp32"

    # Test invalid mode rejection
    res = client.post("/api/v1/admin/settings/audio_mode", json={"audio_mode": "invalid_mode"})
    assert res.status_code == 400

    res = client.post("/api/v1/audio/mode", json={"mode": "bluetooth"})
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_laptop_audio_processing_endpoint():
    """Requirement 5 & 6: Laptop audio reaches backend processing, runs STT/LLM/TTS, and returns audio_url."""
    sample_rate = 16000
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for _ in range(8000):
            frames.extend(struct.pack('<h', 0))
        wf.writeframes(frames)
    wav_io.seek(0)

    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:

        mock_stt.return_value = {
            "text": "What is photosynthesis?",
            "duration_seconds": 1.2,
            "language": "en"
        }
        mock_llm.return_value = LLMResult(
            text="Photosynthesis is the process by which green plants synthesize nutrients from sunlight.",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False,
            latency_ms=120.0
        )

        files = {"audio": ("laptop_speech.webm", wav_io, "audio/webm")}
        data = {"student_id": "1", "mode": "coach"}
        res = client.post("/api/v1/audio/laptop?format=json&test_mode=true", files=files, data=data)

        assert res.status_code == 200
        payload = res.json()
        assert payload.get("success") is True
        assert "photosynthesis" in payload.get("transcript", "").lower()
        assert "reply_text" in payload and len(payload["reply_text"]) > 0
        assert payload.get("audio_url") is not None
        assert "audio_url" in payload


def test_laptop_audio_does_not_mark_esp32_connected():
    """Requirement 8: Laptop mode must work without ESP32 and never mark ESP32 connected."""
    # Reset ESP32 hardware state to disconnected
    _esp32_hardware_state["last_seen_timestamp"] = None
    _esp32_hardware_state["last_ip"] = None

    # Check hardware status is initially disconnected
    res = client.get("/api/v1/esp32/hardware_status")
    assert res.json()["connected"] is False

    # Simulate a laptop audio upload
    sample_rate = 16000
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for _ in range(4000):
            frames.extend(struct.pack('<h', 0))
        wf.writeframes(frames)
    wav_io.seek(0)

    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt:
        mock_stt.return_value = {"text": "Hey Mizo", "duration_seconds": 0.5, "language": "en"}
        files = {"audio": ("browser_mic.wav", wav_io, "audio/wav")}
        res = client.post("/api/v1/audio/laptop?format=json", files=files, data={"student_id": "1"})
        assert res.status_code == 200

    # ESP32 hardware status must STILL be disconnected!
    res = client.get("/api/v1/esp32/hardware_status")
    assert res.json()["connected"] is False, "Laptop audio must not falsely mark ESP32 hardware as connected"


def test_laptop_audio_payload_limit():
    """Step 3: Enforce reasonable upload limits on laptop audio endpoint (e.g. 15MB limit)."""
    # Create oversized fake payload (> 15MB)
    huge_payload = io.BytesIO(b"0" * (16 * 1024 * 1024))
    files = {"audio": ("huge.wav", huge_payload, "audio/wav")}
    res = client.post("/api/v1/audio/laptop", files=files, data={"student_id": "1"})
    assert res.status_code == 413


def test_frontend_app_js_mode_handling():
    """Requirements 2, 3, 4, 6: app.js contains functions for mode selection, persistence, and cleanup."""
    app_js = Path("app/static/js/app.js").read_text(encoding="utf-8")

    assert "function selectAudioMode" in app_js or "async function selectAudioMode" in app_js
    assert "function initAudioMode" in app_js or "async function initAudioMode" in app_js
    assert "function updateLaptopStatusIndicator" in app_js
    assert "function testLaptopMicPermission" in app_js
    assert "currentAudioMode" in app_js
    assert "isLaptopAudioDisabled" in app_js
    assert "/api/v1/audio/laptop" in app_js
    assert "new Audio(" in app_js  # for laptop audio playback


def test_frontend_laptop_mode_controls_and_isolation():
    """Verify frontend code ensures laptop mode never requires ESP32 or manual_activate endpoint."""
    app_js = Path("app/static/js/app.js").read_text(encoding="utf-8")

    # In setupAudioBench, btn-mic-record and btn-toggle-voice-mode must route to laptop recording when in laptop mode
    assert 'if (currentAudioMode === "laptop") {' in app_js
    assert 'stopVoiceRecording();' in app_js
    assert 'startVoiceRecording(false);' in app_js

    # toggleManualVoiceActivation must guard against executing in laptop mode
    assert 'if (currentAudioMode === "laptop") {' in app_js

    # Face panel tap should also be supported
    assert 'facePanel.addEventListener("click"' in app_js

    # Mode switching must cleanly tear down micMediaStream and wake word
    assert 'micMediaStream.getTracks().forEach(t => t.stop());' in app_js
    assert 'wakeWordRecognizer.abort()' in app_js


def test_robot_mode_hardware_disconnect_preservation():
    """Robot mode preserves genuine disconnect reporting when ESP32 is offline."""
    _esp32_hardware_state["last_seen_timestamp"] = None
    _esp32_hardware_state["last_ip"] = None

    # Switch mode to esp32
    res = client.post("/api/v1/audio/mode", json={"audio_mode": "esp32"})
    assert res.status_code == 200
    assert res.json()["audio_mode"] == "esp32"

    # Hardware status must accurately report disconnected
    res = client.get("/api/v1/esp32/hardware_status")
    data = res.json()
    assert data["connected"] is False
    assert "disconnected" in data["status_message"].lower()


@pytest.mark.asyncio
async def test_tts_service_generates_real_speech_not_beep():
    """Verify TTS produces actual spoken audio and not the 1.0s 440Hz fallback beep."""
    from app.services.tts_service import tts_service
    url, file_path = await tts_service.synthesize_to_file("This is a full conversational reply from Mizo for laptop audio testing.")
    path = Path(file_path)
    assert path.exists(), "Synthesized audio file must exist"
    assert path.stat().st_size > 35000, f"Spoken audio file size ({path.stat().st_size}) should exceed short beep size (32044)"

    with wave.open(str(path), 'rb') as wf:
        duration = wf.getnframes() / wf.getframerate()
        assert duration > 1.2, f"Spoken speech duration ({duration}s) must be longer than 1.0s test tone beep"

    # Verify both audio cache URL routes can serve this file
    filename = path.name
    res_esp32 = client.get(f"/api/v1/esp32/audio/cache/{filename}")
    assert res_esp32.status_code == 200
    assert res_esp32.headers["content-type"] in ["audio/wav", "audio/mpeg"]

    res_laptop = client.get(f"/api/v1/audio/cache/{filename}")
    assert res_laptop.status_code == 200
    assert res_laptop.headers["content-type"] in ["audio/wav", "audio/mpeg"]


@pytest.mark.asyncio
async def test_tts_offline_local_fallback_when_edge_tts_fails():
    """Verify that when Edge-TTS raises an error, local offline TTS generates actual spoken words."""
    from app.services.tts_service import tts_service
    with patch("edge_tts.Communicate.save", side_effect=RuntimeError("EdgeTTS network down")):
        url, file_path = await tts_service.synthesize_to_file("Testing offline local fallback speech generation.")
        path = Path(file_path)
        assert path.exists()
        assert path.stat().st_size > 35000, "Offline speech fallback must produce speech audio, not empty or corrupt file"


def test_frontend_playback_logging_and_speech_synthesis_fallback():
    """Verify app.js includes diagnostic logging and client-side SpeechSynthesis fallback."""
    app_js = Path("app/static/js/app.js").read_text(encoding="utf-8")
    assert "[Laptop Audio]" in app_js
    assert "onloadedmetadata" in app_js
    assert "window.speechSynthesis" in app_js
    assert "SpeechSynthesisUtterance" in app_js

