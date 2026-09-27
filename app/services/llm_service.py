import os
import time
import json
import httpx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from app.config import settings
from app.db.database import get_db_settings


class LLMServiceError(Exception):
    """Raised when LLM generation fails across all configured providers."""
    def __init__(self, message: str, provider_errors: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.provider_errors = provider_errors or {}


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    is_fallback: bool = False
    latency_ms: float = 0.0
    fallback_chain: List[str] = field(default_factory=list)


class LLMService:
    """
    Production-grade LLM Router:
    1. Primary: Llama 3.3 70B Versatile (via Groq API / configured Llama provider).
    2. Fallback cascade: OpenAI -> Qwen / OpenRouter -> Local Ollama.
    3. Strict policy: Never silently return hardcoded strings; propagate real errors with diagnostics.
    4. Startup and on-demand connectivity verification.
    """

    @classmethod
    async def generate_response(
        cls,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 350,
        provider_override: Optional[str] = None
    ) -> LLMResult:
        db_conf = get_db_settings()
        primary_provider = provider_override or db_conf.get("active_provider") or "groq"

        # Prepend system prompt if supplied
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        # Define priority cascade order based on primary provider
        # Always prioritize primary provider first, then sequentially fallback
        candidate_order = [primary_provider]
        fallbacks = ["groq", "openai", "qwen", "ollama"]
        for fb in fallbacks:
            if fb not in candidate_order:
                candidate_order.append(fb)

        provider_errors: Dict[str, str] = {}
        fallback_chain: List[str] = []

        for idx, provider in enumerate(candidate_order):
            is_fallback_attempt = (idx > 0)
            start_time = time.perf_counter()

            try:
                if provider == "groq":
                    res_text, model = await cls._call_groq(full_messages, db_conf, temperature, max_tokens)
                elif provider == "openai":
                    res_text, model = await cls._call_openai(full_messages, db_conf, temperature, max_tokens)
                elif provider == "qwen":
                    res_text, model = await cls._call_qwen(full_messages, db_conf, temperature, max_tokens)
                elif provider == "ollama":
                    res_text, model = await cls._call_ollama(full_messages, db_conf, temperature, max_tokens)
                else:
                    continue

                if res_text and res_text.strip():
                    latency = round((time.perf_counter() - start_time) * 1000, 2)
                    if is_fallback_attempt:
                        print(f"[LLM] Primary failed; successfully used fallback provider: {provider} ({model}) in {latency}ms")
                    return LLMResult(
                        text=res_text.strip(),
                        provider=provider,
                        model=model,
                        is_fallback=is_fallback_attempt,
                        latency_ms=latency,
                        fallback_chain=fallback_chain
                    )

            except Exception as exc:
                err_msg = str(exc)
                provider_errors[provider] = err_msg
                fallback_chain.append(f"{provider}: {err_msg}")
                print(f"[LLM] Provider '{provider}' failed: {err_msg}")

        # If every configured provider in the cascade failed, raise an explicit error
        error_summary = " | ".join([f"{p}: {err}" for p, err in provider_errors.items()])
        print(f"[LLM ERROR] All LLM providers failed. Summary: {error_summary}")
        raise LLMServiceError(
            f"All AI LLM providers failed to complete the request. Details: {error_summary}",
            provider_errors=provider_errors
        )

    @classmethod
    @classmethod
    async def _call_groq(cls, messages: List[Dict[str, str]], conf: Dict[str, Any], temp: float, max_tok: int, timeout: float = 15.0):
        api_key = conf.get("groq_api_key") or settings.GROQ_API_KEY
        model = conf.get("groq_model") or settings.GROQ_MODEL or "llama-3.3-70b-versatile"

        if not api_key:
            raise ValueError("Groq API key not configured")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"], model
            else:
                raise RuntimeError(f"Groq API returned HTTP {resp.status_code}: {resp.text}")

    @classmethod
    async def _call_openai(cls, messages: List[Dict[str, str]], conf: Dict[str, Any], temp: float, max_tok: int, timeout: float = 15.0):
        api_key = conf.get("openai_api_key") or settings.OPENAI_API_KEY
        model = conf.get("openai_model") or settings.OPENAI_MODEL or "gpt-4o-mini"

        if not api_key:
            raise ValueError("OpenAI API key not configured")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"], model
            else:
                raise RuntimeError(f"OpenAI API returned HTTP {resp.status_code}: {resp.text}")

    @classmethod
    async def _call_qwen(cls, messages: List[Dict[str, str]], conf: Dict[str, Any], temp: float, max_tok: int, timeout: float = 15.0):
        api_key = conf.get("qwen_api_key") or settings.QWEN_API_KEY
        model = conf.get("qwen_model") or settings.QWEN_MODEL or "qwen/qwen-2.5-72b-instruct"

        if not api_key:
            raise ValueError("Qwen / OpenRouter API key not configured")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok
        }

        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(endpoint, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"], model
            else:
                raise RuntimeError(f"Qwen API returned HTTP {resp.status_code}: {resp.text}")

    @classmethod
    async def _call_ollama(cls, messages: List[Dict[str, str]], conf: Dict[str, Any], temp: float, max_tok: int, timeout: float = 15.0):
        base_url = conf.get("ollama_base_url") or settings.OLLAMA_BASE_URL
        model = conf.get("ollama_model") or settings.OLLAMA_MODEL or "llama3:latest"

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temp, "num_predict": max_tok}
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("message", {}).get("content", ""), model
            else:
                raise RuntimeError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")

    @classmethod
    async def verify_provider_connectivity(cls, provider: str) -> Dict[str, Any]:
        """Tests connectivity for a specific provider without exposing keys in response."""
        db_conf = get_db_settings()
        test_messages = [{"role": "user", "content": "Respond with 'ready'."}]
        start = time.perf_counter()

        try:
            if provider == "groq":
                res, model = await cls._call_groq(test_messages, db_conf, temp=0.1, max_tok=10, timeout=3.0)
            elif provider == "openai":
                res, model = await cls._call_openai(test_messages, db_conf, temp=0.1, max_tok=10, timeout=3.0)
            elif provider == "qwen":
                res, model = await cls._call_qwen(test_messages, db_conf, temp=0.1, max_tok=10, timeout=3.0)
            elif provider == "ollama":
                res, model = await cls._call_ollama(test_messages, db_conf, temp=0.1, max_tok=10, timeout=3.0)
            else:
                return {"provider": provider, "configured": False, "status": "unknown", "model": "none"}

            latency = round((time.perf_counter() - start) * 1000, 1)
            return {
                "provider": provider,
                "configured": True,
                "status": "connected",
                "model": model,
                "latency_ms": latency,
                "details": f"Connected ({latency}ms)"
            }
        except Exception as e:
            err = str(e)
            is_key_missing = "not configured" in err.lower()
            return {
                "provider": provider,
                "configured": not is_key_missing,
                "status": "not_configured" if is_key_missing else "error",
                "model": db_conf.get(f"{provider}_model", "unknown"),
                "details": "API Key missing" if is_key_missing else f"Error: {err[:80]}"
            }

    @classmethod
    async def verify_all_providers(cls) -> List[Dict[str, Any]]:
        """Verifies all supported providers and returns their connectivity states."""
        results = []
        for p in ["groq", "openai", "qwen", "ollama"]:
            res = await cls.verify_provider_connectivity(p)
            results.append(res)
        return results


llm_service = LLMService()

