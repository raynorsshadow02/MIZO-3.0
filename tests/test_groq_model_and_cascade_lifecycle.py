import pytest
import sqlite3
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.db.database import get_db_settings, update_db_settings, init_db, get_connection
from app.services.llm_service import llm_service, LLMService, LLMResult, LLMServiceError


@pytest.fixture
def client():
    return TestClient(app)


# 1. No API key -> NOT CONFIGURED
@pytest.mark.asyncio
async def test_case_1_no_api_key_not_configured():
    conf = {"groq_api_key": "", "groq_model": "openai/gpt-oss-120b"}
    with patch.object(settings, "GROQ_API_KEY", ""):
        res = await llm_service._verify_groq_health(conf)
        assert res["configured"] is False
        assert res["status"] == "not_configured"
        assert res["display_status"] == "NOT CONFIGURED"
        assert res["generation_ready"] is False


# 2. Invalid API key -> INVALID API KEY
@pytest.mark.asyncio
async def test_case_2_invalid_api_key():
    conf = {"groq_api_key": "gsk_invalid_bogus_key", "groq_model": "openai/gpt-oss-120b"}
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = '{"error":{"message":"Invalid API Key","type":"invalid_request_error","code":"invalid_api_key"}}'

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        res = await llm_service._verify_groq_health(conf)
        assert res["configured"] is True
        assert res["healthy"] is False
        assert res["status"] == "authentication_error"
        assert res["auth_status"] == "authentication_error"
        assert res["display_status"] == "INVALID API KEY"


# 3. Valid API key + deprecated model -> MODEL UNAVAILABLE
@pytest.mark.asyncio
async def test_case_3_valid_api_key_deprecated_model():
    conf = {"groq_api_key": "gsk_valid_key_sample", "groq_model": "llama-3.3-70b-versatile"}
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"id": "openai/gpt-oss-120b"},
            {"id": "openai/gpt-oss-20b"},
            {"id": "llama-3.1-8b-instant"}
        ]
    }

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        res = await llm_service._verify_groq_health(conf)
        assert res["configured"] is True
        assert res["healthy"] is False
        assert res["status"] in ("model_error", "MODEL_UNAVAILABLE")
        assert res["auth_status"] == "AUTHENTICATED"
        assert res["display_status"] == "MODEL UNAVAILABLE"
        assert res["model"] == "llama-3.3-70b-versatile"


# 4. Valid API key + supported model -> CONNECTED
# 5. Valid API key + supported model + generation -> READY
@pytest.mark.asyncio
async def test_case_4_and_5_valid_key_supported_model_generation():
    conf = {"groq_api_key": "gsk_valid_key_sample", "groq_model": "openai/gpt-oss-120b"}
    mock_models_resp = MagicMock()
    mock_models_resp.status_code = 200
    mock_models_resp.json.return_value = {
        "data": [
            {"id": "openai/gpt-oss-120b"},
            {"id": "openai/gpt-oss-20b"},
            {"id": "llama-3.1-8b-instant"}
        ]
    }

    mock_chat_resp = MagicMock()
    mock_chat_resp.status_code = 200
    mock_chat_resp.json.return_value = {
        "choices": [{"message": {"content": "MIZO_OK"}}]
    }

    with patch("httpx.AsyncClient.get", return_value=mock_models_resp), \
         patch("httpx.AsyncClient.post", return_value=mock_chat_resp):
        res = await llm_service._verify_groq_health(conf)
        assert res["configured"] is True
        assert res["healthy"] is True
        assert res["status"] == "connected"
        assert res["auth_status"] == "AUTHENTICATED"
        assert res["generation_ready"] is True
        assert res["display_status"] == "READY"
        assert res["model"] == "openai/gpt-oss-120b"
        assert res["probe_response"] == "MIZO_OK"


# 6. Groq unavailable -> cascade continues
# 7. Groq unavailable + Ollama connected -> Ollama responds
@pytest.mark.asyncio
async def test_case_6_and_7_cascade_continues_to_ollama():
    messages = [{"role": "user", "content": "Hello Mizo!"}]
    
    # Simulate:
    # 1. Groq fails because model is unavailable/decommissioned
    # 2. OpenAI fails because API key is invalid (401)
    # 3. OpenRouter is not configured
    # 4. Ollama succeeds and returns response
    with patch.object(LLMService, "_call_groq", side_effect=RuntimeError("HTTP 404: model 'llama-3.3-70b-versatile' has been decommissioned")), \
         patch.object(LLMService, "_call_openai", side_effect=RuntimeError("HTTP 401: Invalid API Key")), \
         patch.object(LLMService, "_call_qwen", side_effect=ValueError("Qwen API key not configured")), \
         patch.object(LLMService, "_call_ollama", new_callable=AsyncMock) as mock_ollama:
        
        mock_ollama.return_value = ("Hello! I am your AI tutor powered by Ollama.", "llama3:latest")

        result = await LLMService.generate_response(messages=messages, provider_override="groq")
        assert result.text == "Hello! I am your AI tutor powered by Ollama."
        assert result.provider == "ollama"
        assert result.model == "llama3:latest"
        assert result.is_fallback is True
        assert len(result.fallback_chain) >= 2
        assert "groq" in result.fallback_chain[0]


# 8. Server restart -> selected model persists
def test_case_8_server_restart_persistence(client):
    test_key = "gsk_persist_key_7788"
    test_model = "openai/gpt-oss-20b"

    # Save via admin settings
    res = client.post("/api/v1/admin/settings", json={
        "groq_api_key": test_key,
        "groq_model": test_model,
        "active_provider": "groq"
    })
    assert res.status_code == 200

    # Simulate restart by reading DB directly
    db_s = get_db_settings()
    assert db_s.get("groq_model") == test_model

    # Canonical config resolver uses DB precedence
    cfg = llm_service.get_provider_config("groq")
    assert cfg.get("model") == test_model


# 9. Old deprecated model exists in database -> migration/default handling works
def test_case_9_database_migration_deprecated_model():
    with patch.object(settings, "GROQ_MODEL", "openai/gpt-oss-120b"):
        # Force deprecated model into system_settings
        conn = get_connection()
        c = conn.cursor()
        c.execute("UPDATE system_settings SET groq_model = 'llama-3.3-70b-versatile' WHERE id = 1")
        conn.commit()
        conn.close()

        db_before = get_db_settings()
        assert db_before.get("groq_model") == "llama-3.3-70b-versatile"

        # Run init_db() simulating server startup
        init_db()

        db_after = get_db_settings()
        assert db_after.get("groq_model") == "openai/gpt-oss-120b"


# 10. Dynamic models endpoint returns models list with active/deprecated metadata
def test_case_10_dynamic_models_endpoint(client):
    # Test GET /api/v1/admin/providers/groq/models
    res = client.get("/api/v1/admin/providers/groq/models")
    assert res.status_code == 200
    data = res.json()
    assert data["provider"] == "groq"
    assert "models" in data
    assert len(data["models"]) > 0

    # Ensure recommended production models are present and marked active
    gpt120 = next((m for m in data["models"] if m["id"] == "openai/gpt-oss-120b"), None)
    assert gpt120 is not None
    assert gpt120["active"] is True
    assert gpt120["deprecated"] is False

    # Check that deprecated model flag is respected
    old_model = next((m for m in data["models"] if m["id"] == "llama-3.3-70b-versatile"), None)
    if old_model:
        assert old_model["deprecated"] is True
        assert old_model["active"] is False
