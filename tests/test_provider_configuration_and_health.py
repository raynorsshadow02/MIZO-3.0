import pytest
import os
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import httpx

from app.main import app
from app.config import settings
from app.db.database import get_db_settings, update_db_settings, mask_api_key
from app.services.llm_service import llm_service, LLMService, LLMResult

client = TestClient(app)


def test_mask_api_key_utility():
    """Verify mask_api_key conceals keys without leaking plaintext."""
    assert mask_api_key("") == "Not configured"
    assert mask_api_key(None) == "Not configured"
    assert mask_api_key("12345") == "********"
    assert mask_api_key("gsk_1234567890abcdef") == "gsk_****cdef"
    assert mask_api_key("sk-proj-123456789xyz") == "sk-p****9xyz"


def test_api_keys_never_exposed_in_admin_endpoints():
    """Verify GET and POST /api/v1/admin/settings never return unmasked secrets."""
    secret = "gsk_super_secret_key_to_protect_9999"
    update_res = client.post("/api/v1/admin/settings", json={"groq_api_key": secret})
    assert update_res.status_code == 200
    data = update_res.json()
    assert data["success"] is True
    assert "settings" in data
    # Secret must NOT be in the response body
    assert secret not in str(data)
    assert data["settings"]["groq_api_key"] == ""
    assert data["settings"]["groq_api_key_configured"] is True
    assert "gsk_****9999" in data["settings"]["groq_api_key_masked"]

    # GET endpoint must also NEVER expose raw key
    get_res = client.get("/api/v1/admin/settings")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert secret not in str(get_data)
    assert get_data["groq_api_key"] == ""
    assert get_data["groq_api_key_configured"] is True
    assert "gsk_****9999" in get_data["groq_api_key_masked"]


def test_empty_string_preserves_existing_key():
    """Empty string or None in settings payload must NOT delete an existing key."""
    secret = "gsk_keep_me_alive_1111"
    client.post("/api/v1/admin/settings", json={"groq_api_key": secret})
    
    # Send empty string update
    client.post("/api/v1/admin/settings", json={
        "groq_api_key": "",
        "active_provider": "groq"
    })
    
    db_s = get_db_settings()
    assert db_s.get("groq_api_key") == secret
    assert settings.GROQ_API_KEY == secret


def test_explicit_clear_key():
    """Explicit clear_{prov}_key clears the key in DB and in-memory settings."""
    secret = "gsk_to_clear_2222"
    client.post("/api/v1/admin/settings", json={"groq_api_key": secret})
    assert get_db_settings().get("groq_api_key") == secret

    res = client.post("/api/v1/admin/settings", json={"clear_groq_key": True})
    assert res.status_code == 200
    db_s = get_db_settings()
    assert not db_s.get("groq_api_key")
    assert settings.GROQ_API_KEY == ""


# ==============================================================================
# SECTION 11 ACCEPTANCE TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_acceptance_test1_valid_groq_ready():
    """TEST 1: Valid Groq key + valid supported model -> Expected: READY"""
    conf = {"groq_api_key": "gsk_valid_key_123", "groq_model": "openai/gpt-oss-120b"}
    with patch("httpx.AsyncClient.get") as mock_get, patch("httpx.AsyncClient.post") as mock_post:
        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.json.return_value = {"data": [{"id": "openai/gpt-oss-120b"}]}
        mock_get.return_value = mock_get_resp

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {"choices": [{"message": {"content": "MIZO_OK"}}]}
        mock_post.return_value = mock_post_resp

        res = await llm_service._verify_groq_health(conf)
        assert res["provider"] == "groq"
        assert res["configured"] is True
        assert res["healthy"] is True
        assert res["status"] in ("READY", "ready", "connected")
        assert res["auth_status"] == "AUTHENTICATED"
        assert res["generation_ready"] is True
        assert res["error"] is None


@pytest.mark.asyncio
async def test_acceptance_test2_invalid_groq_key():
    """TEST 2: Invalid Groq key -> Expected: INVALID_API_KEY"""
    conf = {"groq_api_key": "gsk_invalid_revoked_key", "groq_model": "openai/gpt-oss-120b"}
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = '{"error": {"message": "Invalid API Key", "code": "invalid_api_key"}}'
        mock_get.return_value = mock_resp

        res = await llm_service._verify_groq_health(conf)
        assert res["provider"] == "groq"
        assert res["configured"] is True
        assert res["healthy"] is False
        assert res["status"] in ("INVALID_API_KEY", "authentication_error")
        assert res["auth_status"] in ("INVALID_API_KEY", "authentication_error")
        assert res["generation_ready"] is False
        assert "401" in res["error"] or "Invalid" in res["error"]


@pytest.mark.asyncio
async def test_acceptance_test3_valid_auth_unavailable_model():
    """TEST 3: Valid authentication + deprecated/unavailable model -> Expected: MODEL_UNAVAILABLE"""
    conf = {"groq_api_key": "gsk_valid_key_123", "groq_model": "llama-3.3-70b-versatile"}
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "openai/gpt-oss-120b"}, {"id": "llama-3.1-8b-instant"}]}
        mock_get.return_value = mock_resp

        res = await llm_service._verify_groq_health(conf)
        assert res["provider"] == "groq"
        assert res["configured"] is True
        assert res["healthy"] is False
        assert res["status"] in ("MODEL_UNAVAILABLE", "degraded", "model_error")
        assert res["auth_status"] == "AUTHENTICATED"
        assert res["generation_ready"] is False
        assert res["display_status"] == "MODEL UNAVAILABLE"


@pytest.mark.asyncio
async def test_acceptance_test4_openai_authenticated_quota_exceeded():
    """TEST 4: Valid OpenAI authentication + no quota -> Expected: AUTHENTICATED / QUOTA_EXCEEDED"""
    conf = {"openai_api_key": "sk-proj-valid-key", "openai_model": "gpt-4o-mini"}
    with patch("httpx.AsyncClient.get") as mock_get, patch("httpx.AsyncClient.post") as mock_post:
        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.json.return_value = {"data": [{"id": "gpt-4o-mini"}]}
        mock_get.return_value = mock_get_resp

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 429
        mock_post_resp.text = '{"error": {"message": "You have no credits remaining", "type": "insufficient_quota", "code": "credit_balance_exhausted"}}'
        mock_post.return_value = mock_post_resp

        res = await llm_service._verify_openai_health(conf)
        assert res["provider"] == "openai"
        assert res["configured"] is True
        assert res["healthy"] is False
        assert res["status"] in ("AUTHENTICATED", "QUOTA_EXCEEDED", "degraded")
        assert res["auth_status"] == "AUTHENTICATED"
        assert res["generation_ready"] is False
        assert res["reason"] == "QUOTA_EXCEEDED" or "quota" in res["error"].lower()
        # Must NOT call it invalid API key
        assert "invalid api key" not in res["error"].lower()


@pytest.mark.asyncio
async def test_acceptance_test5_no_openrouter_key():
    """TEST 5: No OpenRouter key -> Expected: NOT_CONFIGURED"""
    conf = {"qwen_api_key": "", "qwen_model": "qwen/qwen-2.5-72b-instruct"}
    res = await llm_service._verify_qwen_health(conf)
    assert res["provider"] == "qwen"
    assert res["configured"] is False
    assert res["healthy"] is False
    assert res["status"] in ("NOT_CONFIGURED", "not_configured")
    assert res["auth_status"] in ("NOT_CONFIGURED", "not_configured")
    assert res["generation_ready"] is False


@pytest.mark.asyncio
async def test_acceptance_test6_ollama_ready():
    """TEST 6: Ollama running -> Expected: READY"""
    conf = {"ollama_base_url": "http://localhost:11434", "ollama_model": "llama3:latest"}
    with patch("httpx.AsyncClient.get") as mock_get, patch("httpx.AsyncClient.post") as mock_post:
        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.json.return_value = {"models": [{"name": "llama3:latest"}]}
        mock_get.return_value = mock_get_resp

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {"message": {"content": "OK"}}
        mock_post.return_value = mock_post_resp

        res = await llm_service._verify_ollama_health(conf)
        assert res["provider"] == "ollama"
        assert res["configured"] is True
        assert res["healthy"] is True
        assert res["status"] in ("READY", "ready")
        assert res["auth_status"] == "AUTHENTICATED"
        assert res["generation_ready"] is True
        assert res["error"] is None


@pytest.mark.asyncio
async def test_acceptance_test7_cascade_falls_back_to_ollama():
    """TEST 7: Groq broken + OpenAI unavailable + OpenRouter unavailable + Ollama available -> Mizo automatically uses Ollama."""
    messages = [{"role": "user", "content": "Explain energy conservation."}]

    with patch.object(LLMService, "_call_groq", side_effect=RuntimeError("Groq 401 Invalid API Key")), \
         patch.object(LLMService, "_call_openai", side_effect=RuntimeError("OpenAI 429 Insufficient quota")), \
         patch.object(LLMService, "_call_qwen", side_effect=ValueError("Qwen API key not configured")), \
         patch.object(LLMService, "_call_ollama", new_callable=AsyncMock) as mock_ollama:

        mock_ollama.return_value = ("Energy is conserved in closed systems.", "llama3:latest")

        result = await LLMService.generate_response(messages=messages, provider_override="groq")
        assert result.text == "Energy is conserved in closed systems."
        assert result.provider == "ollama"
        assert result.model == "llama3:latest"
        assert result.is_fallback is True
        assert len(result.fallback_chain) == 3
        mock_ollama.assert_called_once()


def test_acceptance_test8_server_restart_persistence():
    """TEST 8: Save API key -> DB -> Load settings -> configuration persists without leakage."""
    test_key = "gsk_persisted_secret_test_key_8888"
    test_model = "llama-3.1-8b-instant"

    # Step 1: Save configuration
    save_res = client.post("/api/v1/admin/settings", json={
        "groq_api_key": test_key,
        "groq_model": test_model,
        "active_provider": "groq"
    })
    assert save_res.status_code == 200

    # Step 2: Query database directly (simulating fresh load on server restart)
    db_s = get_db_settings()
    assert db_s.get("groq_api_key") == test_key
    assert db_s.get("groq_model") == test_model
    assert db_s.get("active_provider") == "groq"

    # Step 3: Runtime resolution via get_provider_config
    cfg = llm_service.get_provider_config("groq")
    assert cfg.get("api_key") == test_key
    assert cfg.get("model") == test_model

    # Step 4: GET settings endpoint masks the secret
    get_res = client.get("/api/v1/admin/settings")
    assert get_res.status_code == 200
    data = get_res.json()
    assert test_key not in str(data)
    assert data["groq_api_key"] == ""
    assert data["groq_api_key_configured"] is True
    assert "gsk_****8888" in data["groq_api_key_masked"]
    assert data["groq_model"] == test_model


def test_get_available_models_endpoint():
    """Verify GET /api/v1/admin/settings/models/{provider} returns supported models."""
    res = client.get("/api/v1/admin/settings/models/groq")
    assert res.status_code == 200
    data = res.json()
    assert data["provider"] == "groq"
    assert "recommended_models" in data
    assert "openai/gpt-oss-120b" in data["recommended_models"]
    assert "available_models" in data
    assert "models" in data
    assert any(m["id"] == "openai/gpt-oss-120b" and m["active"] is True for m in data["models"])

    # Also verify route at /api/v1/admin/providers/groq/models
    res_direct = client.get("/api/v1/admin/providers/groq/models")
    assert res_direct.status_code == 200
    assert res_direct.json()["provider"] == "groq"
    assert "models" in res_direct.json()

    res_openai = client.get("/api/v1/admin/settings/models/openai")
    assert res_openai.status_code == 200
    assert "gpt-4o-mini" in res_openai.json()["recommended_models"]

    res_bad = client.get("/api/v1/admin/settings/models/unknown_provider")
    assert res_bad.status_code == 400


def test_single_provider_endpoint():
    """Verify GET /api/v1/admin/settings/provider-status/{provider} endpoint."""
    res = client.get("/api/v1/admin/settings/provider-status/groq")
    assert res.status_code == 200
    data = res.json()
    assert data["provider"] == "groq"
    assert "status" in data
    assert "configured" in data

    # Invalid provider returns 400
    res_bad = client.get("/api/v1/admin/settings/provider-status/invalid_provider")
    assert res_bad.status_code == 400
