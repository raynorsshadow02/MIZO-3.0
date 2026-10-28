import json
import re
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.voice_service import voice_manager, VoiceState

client = TestClient(app)


def test_virtual_face_html_elements_and_preview_controls():
    """Verify presence of virtual face panel, visor elements, and developer preview controls."""
    html_file = Path("app/static/index.html")
    assert html_file.exists()
    content = html_file.read_text(encoding="utf-8")

    # Core Face Panel Elements
    assert 'id="mizo-face-panel"' in content
    assert 'class="mizo-face-visor"' in content
    assert 'class="mizo-visor-screen-grid"' in content
    assert 'class="mizo-listening-ring"' in content
    assert 'class="mizo-thinking-matrix"' in content
    assert 'id="mizo-eye-left"' in content
    assert 'id="mizo-eye-right"' in content
    assert 'id="mizo-face-mouth"' in content
    assert 'id="mizo-face-status-label"' in content
    assert 'id="mizo-face-status-hint"' in content
    assert 'id="mizo-face-status-dot"' in content

    # Developer Preview Controls
    assert 'id="mizo-face-preview-controls"' in content
    assert 'id="mizo-face-mode-badge"' in content
    assert 'id="btn-preview-idle"' in content
    assert 'id="btn-preview-listening"' in content
    assert 'id="btn-preview-thinking"' in content
    assert 'id="btn-preview-speaking"' in content
    assert 'id="btn-preview-error"' in content
    assert 'id="btn-preview-exit"' in content

    # Modular Face Script Loaded before App Script
    assert 'src="/js/mizo_face.js?v=3.0.4"' in content
    assert 'src="/js/app.js?v=3.0.4"' in content


def test_virtual_face_css_states_and_reduced_motion():
    """Verify CSS selectors for all 5 states, preview controls, gaze shifts, mouth speech aperture, and reduced-motion."""
    css_file = Path("app/static/css/style.css")
    assert css_file.exists()
    content = css_file.read_text(encoding="utf-8")

    # State selectors
    assert '.mizo-face-card[data-state="idle"]' in content
    assert '.mizo-face-card[data-state="listening"]' in content
    assert '.mizo-face-card[data-state="thinking"]' in content
    assert '.mizo-face-card[data-state="speaking"]' in content
    assert '.mizo-face-card[data-state="error"]' in content
    assert '.mizo-face-card[data-state="onboarding"]' in content

    # Preview Bar styles
    assert '.mizo-face-preview-bar' in content
    assert '.mizo-mode-pill.live' in content
    assert '.mizo-mode-pill.preview' in content
    assert '.btn-face-preview' in content

    # Gaze shifting classes
    assert '.mizo-face-card.gaze-left' in content
    assert '.mizo-face-card.gaze-right' in content
    assert '.mizo-face-card.gaze-up' in content
    assert '.mizo-face-card.gaze-down' in content
    assert '.mizo-face-card.gaze-center' in content

    # Animation keyframes
    assert '@keyframes mizo-blink' in content
    assert '@keyframes mizo-listen-ring-pulse' in content
    assert '@keyframes mizo-think-scan' in content
    assert '@keyframes mizo-think-dot' in content
    assert '@keyframes mizo-mouth-aperture' in content
    assert '@keyframes mizo-speak-wave-1' in content
    assert '@keyframes mizo-idle-float' in content
    assert '@keyframes mizo-mini-spin' in content
    assert '@keyframes mizo-sw-bar' in content

    # Dynamic status indicators
    assert '.status-mini-spinner' in content
    assert '.status-mini-wave' in content

    # Reduced motion fallback
    assert '@media (prefers-reduced-motion: reduce)' in content
    reduced_section = content[content.index('@media (prefers-reduced-motion: reduce)'):]
    assert 'animation: none !important;' in reduced_section


def test_mizo_face_js_controller_module_integrity():
    """Verify modular face controller contains exact required status labels, preview logic, blinking, and gaze tracking."""
    js_file = Path("app/static/js/mizo_face.js")
    assert js_file.exists()
    content = js_file.read_text(encoding="utf-8")

    # Controller Class & Global Exposure
    assert 'class MizoFaceController' in content
    assert 'window.mizoFace =' in content
    assert 'window.FACE_STATE_CONFIG =' in content

    # 5 required states configured with exact labels
    assert 'Ready — Say "Hey Mizo" or tap to speak' in content
    assert 'Listening...' in content
    assert 'Thinking... Let me figure that out' in content
    assert 'Mizo is speaking...' in content
    assert 'Something went wrong. Please try again.' in content

    # Continuous animation methods
    assert '_startBlinking()' in content
    assert '_performBlink(' in content
    assert '_stopBlinking()' in content
    assert '_startGazeTracking()' in content
    assert '_applyGaze(' in content
    assert '_stopGazeTracking()' in content

    # Preview Mode State Machine Methods
    assert 'setPreviewState(' in content
    assert 'exitPreview(' in content
    assert 'update(' in content

    # Status indicator icon rendering
    assert 'status-mini-spinner' in content
    assert 'status-mini-wave' in content

    # Hardware Export Interface for Physical OLED / TFT Display Adaptation
    assert 'getDisplayFrame(' in content


def test_app_js_delegates_to_mizo_face_and_guards_pipeline():
    """Verify updateMizoFace in app.js integrates cleanly and background polling does not clobber active speech/processing."""
    app_js = Path("app/static/js/app.js").read_text(encoding="utf-8")
    assert "function updateMizoFace(" in app_js
    assert "window.mizoFace" in app_js
    assert "window.mizoFace.update(" in app_js

    # Real pipeline transitions
    assert 'updateMizoFace("listening"' in app_js
    assert 'updateMizoFace("thinking"' in app_js
    assert 'updateMizoFace("speaking"' in app_js
    assert 'updateMizoFace("error"' in app_js

    # Polling protection guard
    assert "!isMizoSpeaking" in app_js
    assert "isProcessingBackend" in app_js


def test_live_voice_state_endpoint_mapping():
    """Verify GET /api/v1/esp32/voice_state supplies real states mapped to virtual face states."""
    res = client.get("/api/v1/esp32/voice_state")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "voice_state" in data

    # Simulated transitions in voice_manager
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_idle")
    res_idle = client.get("/api/v1/esp32/voice_state").json()
    assert res_idle["voice_state"] == "WAKE_LISTENING"

    voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "test_listening")
    res_listen = client.get("/api/v1/esp32/voice_state").json()
    assert res_listen["voice_state"] == "ACTIVE_LISTENING"

    voice_manager.transition_to(VoiceState.PROCESSING, "test_thinking")
    res_think = client.get("/api/v1/esp32/voice_state").json()
    assert res_think["voice_state"] == "PROCESSING"

    voice_manager.transition_to(VoiceState.SPEAKING, "test_speaking")
    res_speak = client.get("/api/v1/esp32/voice_state").json()
    assert res_speak["voice_state"] == "SPEAKING"

    voice_manager.transition_to(VoiceState.ERROR_RECOVERY, "test_error")
    res_err = client.get("/api/v1/esp32/voice_state").json()
    assert res_err["voice_state"] == "ERROR_RECOVERY"

    # Restore to standby
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "restore")
