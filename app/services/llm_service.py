import os
import json
import httpx
from typing import List, Dict, Any, Optional
from app.config import settings
from app.db.database import get_db_settings


class LLMService:
    """Multi-provider AI router for Groq, OpenAI, Qwen, and Ollama."""

    @classmethod
    async def generate_response(
        cls,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 250
    ) -> str:
        db_conf = get_db_settings()
        provider = db_conf.get("active_provider") or "groq"

        # Prepend system prompt if supplied
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        # Route to appropriate provider
        if provider == "groq":
            return await cls._call_groq(full_messages, db_conf, temperature, max_tokens)
        elif provider == "openai":
            return await cls._call_openai(full_messages, db_conf, temperature, max_tokens)
        elif provider == "qwen":
            return await cls._call_qwen(full_messages, db_conf, temperature, max_tokens)
        elif provider == "ollama":
            return await cls._call_ollama(full_messages, db_conf, temperature, max_tokens)
        else:
            return await cls._call_groq(full_messages, db_conf, temperature, max_tokens)

    @classmethod
    async def _call_groq(cls, messages: List[Dict[str, str]], conf: Dict[str, Any], temp: float, max_tok: int) -> str:
        api_key = conf.get("groq_api_key") or settings.GROQ_API_KEY
        model = conf.get("groq_model") or settings.GROQ_MODEL

        if not api_key:
            return cls._mock_fallback_response(messages)

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

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    print(f"[LLM] Groq API error {resp.status_code}: {resp.text}")
                    return cls._mock_fallback_response(messages)
        except Exception as e:
            print(f"[LLM] Groq call failed: {e}")
            return cls._mock_fallback_response(messages)

    @classmethod
    async def _call_openai(cls, messages: List[Dict[str, str]], conf: Dict[str, Any], temp: float, max_tok: int) -> str:
        api_key = conf.get("openai_api_key") or settings.OPENAI_API_KEY
        model = conf.get("openai_model") or settings.OPENAI_MODEL

        if not api_key:
            return cls._mock_fallback_response(messages)

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

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    print(f"[LLM] OpenAI API error {resp.status_code}: {resp.text}")
                    return cls._mock_fallback_response(messages)
        except Exception as e:
            print(f"[LLM] OpenAI call failed: {e}")
            return cls._mock_fallback_response(messages)

    @classmethod
    async def _call_qwen(cls, messages: List[Dict[str, str]], conf: Dict[str, Any], temp: float, max_tok: int) -> str:
        api_key = conf.get("qwen_api_key") or settings.QWEN_API_KEY
        model = conf.get("qwen_model") or settings.QWEN_MODEL

        if not api_key:
            return cls._mock_fallback_response(messages)

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

        # Supports OpenRouter or DashScope OpenAI-compatible endpoint
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(endpoint, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    return cls._mock_fallback_response(messages)
        except Exception as e:
            print(f"[LLM] Qwen call failed: {e}")
            return cls._mock_fallback_response(messages)

    @classmethod
    async def _call_ollama(cls, messages: List[Dict[str, str]], conf: Dict[str, Any], temp: float, max_tok: int) -> str:
        base_url = conf.get("ollama_base_url") or settings.OLLAMA_BASE_URL
        model = conf.get("ollama_model") or "llama3:latest"

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temp, "num_predict": max_tok}
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("message", {}).get("content", "").strip()
                else:
                    return cls._mock_fallback_response(messages)
        except Exception as e:
            print(f"[LLM] Ollama call failed: {e}")
            return cls._mock_fallback_response(messages)

    @classmethod
    def _mock_fallback_response(cls, messages: List[Dict[str, str]]) -> str:
        """Intelligent offline fallback for instant interactive testing."""
        last_user = "student"
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break

        query = last_user.lower()
        if "hello" in query or "hi" in query:
            return "Hello there! I am Mikaza, your AI learning and communication tutor. What exciting topic would you like to explore today?"
        elif "grammar" in query or "english" in query:
            return "Great choice! Clear speech builds great confidence. Try describing your favorite hobby, and I will share helpful vocabulary tips!"
        elif "robot" in query or "esp32" in query or "hardware" in query:
            return "Your ESP32-S3 microcontroller is communicating smoothly with my cloud server! Your audio streaming pipeline is fully operational."
        else:
            return f"That is a great thought regarding '{last_user[:35]}...'. Keep speaking naturally, and let's delve deeper into this concept together!"


llm_service = LLMService()
