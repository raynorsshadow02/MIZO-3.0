import pytest
import re
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.db.database import (
    get_student,
    update_student,
    get_prototype_student,
    PROTOTYPE_STUDENT_ID,
    get_conversation_history,
    add_conversation_message,
    set_pending_name_change,
    get_pending_name_change,
    clear_pending_name_change
)
from app.services.personalization_service import personalization_service

client = TestClient(app)


def test_onboarding_starts_with_first_question():
    """1. Starting onboarding triggers the first onboarding question asking for name."""
    res = client.post(f"/api/v1/students/{PROTOTYPE_STUDENT_ID}/restart-onboarding")
    assert res.status_code == 200

    student = get_student(PROTOTYPE_STUDENT_ID)
    assert student is not None
    assert student["onboarding_step"] == "ask_name"
    assert student["onboarding_completed"] in (0, False)

    # First question text
    q_text = personalization_service.get_onboarding_question_text("ask_name", student=student)
    assert "name" in q_text.lower() or "call you" in q_text.lower()


def test_onboarding_step_by_step_advancement_and_persistence():
    """2 & 3. Step-by-step interview advances, saves to DB, and persists in SQLite."""
    # Reset to clean onboarding state
    client.post(f"/api/v1/students/{PROTOTYPE_STUDENT_ID}/restart-onboarding")

    # Step 1: Provide Name
    turn1_msg = "My name is Ethan Vance"
    turn1_res = client.post(
        "/api/v1/esp32/chat",
        json={
            "student_id": PROTOTYPE_STUDENT_ID,
            "message": turn1_msg,
            "is_typed_text": True
        }
    )
    assert turn1_res.status_code == 200
    st1 = get_student(PROTOTYPE_STUDENT_ID)
    assert st1["name"] == "Ethan Vance"
    # Should advance from ask_name to education/goals
    assert st1["onboarding_step"] in ("ask_goals", "ask_weaknesses", "ask_self_level")

    # Step 2: Provide Education & Goals
    turn2_msg = "I am studying computer engineering at Stanford and my goal is to pass job interviews confidently"
    turn2_res = client.post(
        "/api/v1/esp32/chat",
        json={
            "student_id": PROTOTYPE_STUDENT_ID,
            "message": turn2_msg,
            "is_typed_text": True
        }
    )
    assert turn2_res.status_code == 200
    st2 = get_student(PROTOTYPE_STUDENT_ID)
    assert st2["name"] == "Ethan Vance"  # Name still preserved
    assert st2["education"] or st2["learning_goals"]

    # Step 3: Provide Strengths and Weaknesses
    turn3_msg = "My strength is vocabulary but my main weakness is I get nervous and hesitate when speaking"
    turn3_res = client.post(
        "/api/v1/esp32/chat",
        json={
            "student_id": PROTOTYPE_STUDENT_ID,
            "message": turn3_msg,
            "is_typed_text": True
        }
    )
    assert turn3_res.status_code == 200
    st3 = get_student(PROTOTYPE_STUDENT_ID)
    assert st3["name"] == "Ethan Vance"
    assert st3["weaknesses"] is not None

    # Verify no empty overwrite protection in DB
    update_student(PROTOTYPE_STUDENT_ID, {"name": "", "education": ""})
    st_guarded = get_student(PROTOTYPE_STUDENT_ID)
    assert st_guarded["name"] == "Ethan Vance"
    assert st_guarded["education"] != ""


def test_name_change_confirmation_guard():
    """Verify assistant asks confirmation before changing an established student name."""
    # Ensure profile has Ethan Vance
    update_student(PROTOTYPE_STUDENT_ID, {"name": "Ethan Vance"})
    st = get_student(PROTOTYPE_STUDENT_ID)
    assert st["name"] == "Ethan Vance"

    # Someone introduces another name
    turn_conflict = client.post(
        "/api/v1/esp32/chat",
        json={
            "student_id": PROTOTYPE_STUDENT_ID,
            "message": "Call me Alexander",
            "is_typed_text": True
        }
    )
    assert turn_conflict.status_code == 200
    reply = turn_conflict.json()["reply_text"]
    # Should ask if they want to update/replace saved name
    assert "Ethan" in reply or "update" in reply.lower() or "Alexander" in reply

    # Name is not immediately overwritten without confirmation
    st_after = get_student(PROTOTYPE_STUDENT_ID)
    assert st_after["name"] == "Ethan Vance"


def test_shared_identity_and_conversation_persistence():
    """5 & 6. Dashboard and ESP32 share same student record and persist messages in history."""
    st = get_prototype_student()
    assert st["id"] == PROTOTYPE_STUDENT_ID

    unique_content = "ESP32 hardware voice verification marker #4482"
    res = client.post(
        "/api/v1/esp32/chat",
        json={
            "student_id": PROTOTYPE_STUDENT_ID,
            "message": unique_content,
            "is_typed_text": True
        }
    )
    assert res.status_code == 200

    # Retrieve via dashboard history endpoint
    hist_res = client.get(f"/api/v1/conversations?student_id={PROTOTYPE_STUDENT_ID}&limit=20")
    assert hist_res.status_code == 200
    messages = hist_res.json()
    found = any(unique_content in m.get("content", "") for m in messages)
    assert found, "Message was not found in conversation history"


def test_voice_state_endpoint_reports_onboarding_and_status():
    """7. GET /api/v1/esp32/voice_state returns real application and onboarding state."""
    res = client.get("/api/v1/esp32/voice_state")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "voice_state" in data
    assert "onboarding_active" in data
    assert "onboarding_step" in data
    assert "microphone_active" in data


def test_face_panel_html_css_and_js_integrity():
    """7. Verify face panel elements, styles for all 6 states, and reduced-motion fallback."""
    index_path = Path("app/static/index.html")
    assert index_path.exists()
    html_content = index_path.read_text(encoding="utf-8")

    # Required HTML elements
    assert 'id="mizo-face-panel"' in html_content
    assert 'id="mizo-eye-left"' in html_content
    assert 'id="mizo-eye-right"' in html_content
    assert 'id="mizo-face-mouth"' in html_content
    assert 'id="mizo-face-status-label"' in html_content
    assert 'id="mizo-face-status-hint"' in html_content
    assert 'id="mizo-face-status-dot"' in html_content

    css_path = Path("app/static/css/style.css")
    assert css_path.exists()
    css_content = css_path.read_text(encoding="utf-8")

    # Required state styles
    assert '.mizo-face-card[data-state="idle"]' in css_content
    assert '.mizo-face-card[data-state="listening"]' in css_content
    assert '.mizo-face-card[data-state="thinking"]' in css_content
    assert '.mizo-face-card[data-state="speaking"]' in css_content
    assert '.mizo-face-card[data-state="onboarding"]' in css_content
    assert '.mizo-face-card[data-state="error"]' in css_content
    assert '@media (prefers-reduced-motion: reduce)' in css_content

    js_path = Path("app/static/js/app.js")
    assert js_path.exists()
    js_content = js_path.read_text(encoding="utf-8")

    # JS controller function
    assert "function updateMizoFace(" in js_content
    assert '"idle"' in js_content
    assert '"listening"' in js_content
    assert '"thinking"' in js_content
    assert '"speaking"' in js_content
    assert '"onboarding"' in js_content
    assert '"error"' in js_content
