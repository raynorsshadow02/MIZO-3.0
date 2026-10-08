import pytest
from unittest.mock import patch, AsyncMock
from app.services.llm_service import LLMService, LLMServiceError, LLMResult


@pytest.mark.asyncio
async def test_llm_primary_groq_success():
    """Verify that primary Llama 3.3 70B via Groq is called first and returns proper result metadata."""
    messages = [{"role": "user", "content": "Explain energy conservation."}]

    with patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq:
        mock_groq.return_value = ("Energy cannot be created or destroyed.", "llama-3.3-70b-versatile")

        result = await LLMService.generate_response(messages=messages, provider_override="groq")
        assert result.text == "Energy cannot be created or destroyed."
        assert result.provider == "groq"
        assert result.model == "llama-3.3-70b-versatile"
        assert result.is_fallback is False
        assert result.latency_ms >= 0
        mock_groq.assert_called_once()


@pytest.mark.asyncio
async def test_llm_fallback_to_openai_when_groq_fails():
    """Verify that when Groq (primary) fails or is unconfigured, it automatically cascades to OpenAI."""
    messages = [{"role": "user", "content": "What is gravity?"}]

    with patch.object(LLMService, "_call_groq", side_effect=Exception("Groq API rate limit or key missing")), \
         patch.object(LLMService, "_call_openai", new_callable=AsyncMock) as mock_openai:

        mock_openai.return_value = ("Gravity is an attractive force between masses.", "gpt-4o-mini")

        result = await LLMService.generate_response(messages=messages, provider_override="groq")
        assert result.text == "Gravity is an attractive force between masses."
        assert result.provider == "openai"
        assert result.model == "gpt-4o-mini"
        assert result.is_fallback is True
        assert len(result.fallback_chain) > 0
        assert "groq" in result.fallback_chain[0]


@pytest.mark.asyncio
async def test_llm_all_providers_fail_raises_exception():
    """Verify that when all providers fail, LLMServiceError is raised without fake mock text."""
    messages = [{"role": "user", "content": "Test all failures."}]

    with patch.object(LLMService, "_call_groq", side_effect=Exception("Groq down")), \
         patch.object(LLMService, "_call_openai", side_effect=Exception("OpenAI down")), \
         patch.object(LLMService, "_call_qwen", side_effect=Exception("Qwen down")), \
         patch.object(LLMService, "_call_ollama", side_effect=Exception("Ollama down")):

        with pytest.raises(LLMServiceError) as exc_info:
            await LLMService.generate_response(messages=messages, provider_override="groq")

        assert "All AI LLM providers failed" in str(exc_info.value)
        assert "groq" in exc_info.value.provider_errors
        assert "openai" in exc_info.value.provider_errors


@pytest.mark.asyncio
async def test_llm_simple_query_never_falls_back_to_groq():
    """Verify that when Ollama fails for a normal question, it only falls back to OpenRouter and NEVER Groq."""
    messages = [{"role": "user", "content": "Hi there, how are you?"}]

    with patch.object(LLMService, "_call_ollama", side_effect=Exception("Ollama offline")), \
         patch.object(LLMService, "_call_qwen", new_callable=AsyncMock) as mock_qwen, \
         patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq:

        mock_qwen.return_value = ("Hello! I am doing great.", "openrouter/free")

        result = await LLMService.generate_response(messages=messages)

        assert result.text == "Hello! I am doing great."
        assert result.provider == "qwen"
        assert result.is_fallback is True
        mock_groq.assert_not_called()


@pytest.mark.asyncio
async def test_provider_connectivity_verification():
    """Verify provider connectivity check safely reports status without crashing."""
    with patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq:
        mock_groq.return_value = ("ready", "llama-3.3-70b-versatile")
        status = await LLMService.verify_provider_connectivity("groq")
        assert status["status"].lower() in ("connected", "ready")
        assert status["configured"] is True
        assert status["provider"] == "groq"
