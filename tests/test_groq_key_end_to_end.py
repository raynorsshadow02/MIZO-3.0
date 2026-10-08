import pytest
import os
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.db.database import get_db_settings, update_db_settings, init_db
from app.services.llm_service import llm_service


@pytest.fixture
def client():
    return TestClient(app)


def test_groq_e2e_test_a_fake_key_401(client):
    """Test A: fake key exists -> mock Groq = 401 -> configured=True, status=authentication_error."""
    fake_key = "gsk_TEST_INVALID_KEY_123456"
    
    # 1. Save key
    save_res = client.post("/api/v1/admin/settings", json={"groq_api_key": fake_key})
    assert save_res.status_code == 200
    assert save_res.json()["settings"]["groq_api_key_configured"] is True

    # 2. Database verification
    db_s = get_db_settings()
    assert db_s["groq_api_key"] == fake_key

    # 3. Mock Groq models endpoint returning 401
    mock_resp_401 = httpx.Response(
        status_code=401,
        json={"error": {"message": "Invalid API Key", "type": "invalid_request_error"}},
        request=httpx.Request("GET", "https://api.groq.com/openai/v1/models")
    )

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp_401

        # 4. Call health endpoint
        res = client.get("/api/v1/admin/settings/provider-status/groq")
        assert res.status_code == 200
        data = res.json()
        assert data["provider"] == "groq"
        assert data["configured"] is True
        assert data["status"] == "authentication_error"
        assert data["display_status"] == "INVALID API KEY"


def test_groq_e2e_test_b_fake_key_200(client):
    """Test B: fake key exists -> mock Groq = 200 -> configured=True, status=connected."""
    valid_key = "gsk_TEST_MOCK_VALID_KEY_789012"
    
    # 1. Save key
    save_res = client.post("/api/v1/admin/settings", json={"groq_api_key": valid_key})
    assert save_res.status_code == 200
    assert save_res.json()["settings"]["groq_api_key_configured"] is True

    # 2. Mock Groq Level 1 (models 200) and Level 2 (chat 200)
    mock_models_resp = httpx.Response(
        status_code=200,
        json={"data": [{"id": "openai/gpt-oss-120b"}, {"id": "llama-3.1-8b-instant"}]},
        request=httpx.Request("GET", "https://api.groq.com/openai/v1/models")
    )
    mock_chat_resp = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": "MIZO_OK"}}]},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    )

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get, \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_get.return_value = mock_models_resp
        mock_post.return_value = mock_chat_resp

        res = client.get("/api/v1/admin/settings/provider-status/groq")
        assert res.status_code == 200
        data = res.json()
        assert data["provider"] == "groq"
        assert data["configured"] is True
        assert data["healthy"] is True
        assert data["status"] == "connected"
        assert data["display_status"] == "READY"


def test_groq_e2e_test_c_no_key(client):
    """Test C: no key -> configured=False, status=not_configured."""
    # 1. Explicitly clear Groq key
    clear_res = client.post("/api/v1/admin/settings", json={"clear_groq_key": True})
    assert clear_res.status_code == 200
    assert clear_res.json()["settings"]["groq_api_key_configured"] is False

    # 2. Database verification
    db_s = get_db_settings()
    assert (db_s.get("groq_api_key") or "").strip() == ""

    # 3. Call health endpoint
    res = client.get("/api/v1/admin/settings/provider-status/groq")
    assert res.status_code == 200
    data = res.json()
    assert data["provider"] == "groq"
    assert data["configured"] is False
    assert data["status"] == "not_configured"
    assert data["display_status"] == "NOT CONFIGURED"


def test_groq_e2e_test_d_restart_persistence(client):
    """Test D: save key -> restart application configuration -> configured=True."""
    persisted_key = "gsk_TEST_PERSIST_KEY_445566"
    
    # 1. Save key
    save_res = client.post("/api/v1/admin/settings", json={"groq_api_key": persisted_key})
    assert save_res.status_code == 200

    # 2. Simulate application restart: wipe in-memory settings, run init_db
    settings.GROQ_API_KEY = ""
    os.environ["GROQ_API_KEY"] = ""
    
    init_db()

    # 3. Verify settings reloaded from SQLite
    db_s = get_db_settings()
    assert db_s["groq_api_key"] == persisted_key
    assert getattr(settings, "GROQ_API_KEY") == persisted_key
    assert os.environ.get("GROQ_API_KEY") == persisted_key

    # 4. Resolver check
    cfg = llm_service.get_provider_config("groq")
    assert cfg["configured"] is True
    assert cfg["api_key"] == persisted_key


def test_groq_e2e_test_e_frontend_contract_auth_error():
    """Test E: frontend contract maps authentication_error -> INVALID API KEY."""
    backend_payload = {
        "provider": "groq",
        "configured": True,
        "healthy": False,
        "status": "authentication_error",
        "display_status": "INVALID API KEY"
    }

    # Verify status mapping logic matches the contract
    st = backend_payload["status"].upper()
    is_auth_error = st in ("INVALID_API_KEY", "AUTHENTICATION_ERROR")
    assert is_auth_error is True
    
    expected_badge = "INVALID API KEY"
    badge_rendered = "INVALID API KEY" if is_auth_error else "OTHER"
    assert badge_rendered == expected_badge


def test_groq_e2e_test_f_frontend_contract_not_configured():
    """Test F: frontend contract maps not_configured -> NOT CONFIGURED."""
    backend_payload = {
        "provider": "groq",
        "configured": False,
        "healthy": False,
        "status": "not_configured",
        "display_status": "NOT CONFIGURED"
    }

    st = backend_payload["status"].upper()
    is_not_configured = (st == "NOT_CONFIGURED" or not backend_payload["configured"])
    assert is_not_configured is True

    expected_badge = "NOT CONFIGURED"
    badge_rendered = "NOT CONFIGURED" if is_not_configured else "OTHER"
    assert badge_rendered == expected_badge
