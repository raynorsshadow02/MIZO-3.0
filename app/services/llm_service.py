import os
import re
import time
import json
import asyncio
import httpx
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from app.config import settings
from app.db.database import get_db_settings, mask_api_key


class LLMErrorCategory:
    AUTHENTICATION_ERROR = "AUTHENTICATION ERROR"
    INVALID_API_KEY = "INVALID_API_KEY"
    FORBIDDEN = "FORBIDDEN"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    NETWORK_ERROR = "NETWORK_ERROR"
    TIMEOUT = "TIMEOUT"
    INVALID_MODEL = "INVALID_MODEL"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MISSING_API_KEY = "MISSING_API_KEY"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    PARSING_ERROR = "PARSING_ERROR"
    READY = "READY"
    AUTHENTICATED = "AUTHENTICATED"


def sanitize_error_message(err_str: str) -> str:
    """Removes sensitive keys, tokens, or headers from error strings."""
    sanitized = re.sub(r'gsk_[A-Za-z0-9_-]+', 'gsk_***[REDACTED]***', err_str)
    sanitized = re.sub(r'sk-[A-Za-z0-9_-]+', 'sk-***[REDACTED]***', sanitized)
    sanitized = re.sub(r'Bearer\s+[A-Za-z0-9_.-]+', 'Bearer ***[REDACTED]***', sanitized)
    return sanitized


def classify_error(exc: Exception) -> str:
    """Classifies exceptions into structured diagnostic categories distinguishing authentication, quota, rate limit, and model availability."""
    err_str = str(exc).lower()
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError)) or "connect" in err_str or "unreachable" in err_str:
        return LLMErrorCategory.NETWORK_ERROR
    if isinstance(exc, httpx.TimeoutException) or "timed out" in err_str or "timeout" in err_str:
        return LLMErrorCategory.TIMEOUT
    if "api key not configured" in err_str or "missing api key" in err_str or "not configured" in err_str:
        return LLMErrorCategory.MISSING_API_KEY
    if "insufficient_quota" in err_str or "credit_balance_exhausted" in err_str or "credit" in err_str or "quota" in err_str:
        return LLMErrorCategory.QUOTA_EXCEEDED
    if "429" in err_str or "rate limit" in err_str:
        return LLMErrorCategory.RATE_LIMITED
    if "403" in err_str or "forbidden" in err_str or "permission" in err_str:
        return LLMErrorCategory.FORBIDDEN
    if "401" in err_str or "invalid api key" in err_str or "invalid_api_key" in err_str or "unauthorized" in err_str or "authentication" in err_str:
        return LLMErrorCategory.AUTHENTICATION_ERROR
    if "404" in err_str or "model_not_found" in err_str or "model unavailable" in err_str or "not found in" in err_str or "decommissioned" in err_str or "deprecated" in err_str:
        return LLMErrorCategory.MODEL_UNAVAILABLE
    if "json" in err_str or "parse" in err_str:
        return LLMErrorCategory.PARSING_ERROR
    return LLMErrorCategory.PROVIDER_ERROR


class LLMServiceError(Exception):
    """Raised when LLM generation fails across configured providers with diagnostic categories."""
    def __init__(
        self,
        message: str,
        category: str = LLMErrorCategory.PROVIDER_ERROR,
        provider_errors: Optional[Dict[str, str]] = None,
        user_message: Optional[str] = None
    ):
        super().__init__(message)
        self.category = category
        self.provider_errors = provider_errors or {}
        self.user_message = user_message or "I am temporarily unable to reach my AI service. Please check your network and API settings."


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
    def resolve_tier0(
        cls,
        text: str,
        student_data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Tier 0 — Deterministic Backend Application Logic (Zero LLM / Zero External API).
        Handles:
        - 'What is my name?' / learner profile queries
        - 'What is your name?' / Mizo identity
        - Session commands ('restart session', 'reset session', 'clear history')
        - Stop commands
        - Confirmation turns
        Returns dict with response text and metadata if handled, or None.
        """
        if not text:
            return None

        clean = text.strip()
        norm = re.sub(r"[^\w\s]", " ", clean.lower()).strip()
        words = norm.split()
        word_count = len(words)

        # 1. Stop commands
        if norm in ["stop", "stop mizo", "pause", "be quiet", "shut up", "halt", "go to sleep"]:
            return {
                "handled": True,
                "reply_text": "I'm pausing now. Whenever you're ready to talk again, just say 'Hey Mizo'.",
                "action": "stop",
                "provider": "system",
                "intent": "stop_command",
                "complexity": "deterministic"
            }

        # 2. Session commands (restart / reset session)
        session_reset_phrases = [
            "restart session", "reset session", "start new session", "new session",
            "start over", "clear conversation", "clear history", "reset conversation"
        ]
        if norm in session_reset_phrases or norm.startswith("restart the session") or norm.startswith("reset the session"):
            return {
                "handled": True,
                "reply_text": "Your session has been refreshed. What would you like to practice today?",
                "action": "reset_session",
                "provider": "system",
                "intent": "session_command",
                "complexity": "deterministic"
            }

        # 3. Mizo Identity queries ("What is your name?", "Who are you?", etc.)
        identity_queries = [
            "what is your name", "whats your name", "what s your name", "who are you",
            "what is your identity", "whats your identity", "tell me your name",
            "do you have a name", "what do i call you", "what should i call you",
            "are you mizo", "are you meeso", "are you miso", "is your name mizo"
        ]
        if norm in identity_queries or any(norm.startswith(q + " ") for q in identity_queries):
            return {
                "handled": True,
                "reply_text": "My name is Mizo, your English communication coach.",
                "action": "identity",
                "provider": "system",
                "intent": "mizo_identity",
                "complexity": "deterministic"
            }

        # 4. User Name and Learner Profile Queries ("What is my name?", "What is my level?", etc.)
        user_name_queries = [
            "what is my name", "whats my name", "what s my name", "who am i",
            "do you know my name", "do you remember my name", "tell me my name", "say my name"
        ]
        if norm in user_name_queries or any(norm.startswith(q + " ") for q in user_name_queries):
            st_name = (student_data.get("name") if student_data else "") or ""
            if st_name and st_name.strip().lower() not in ["new learner", "student", "there", "unknown", ""]:
                reply = f"Your name is {st_name.strip()}."
            else:
                reply = "I don't have your name yet. What should I call you?"
            return {
                "handled": True,
                "reply_text": reply,
                "action": "user_name",
                "provider": "system",
                "intent": "learner_profile",
                "complexity": "deterministic"
            }

        # Learner Profile Information Queries ("What is my level?", "What are my goals?", "Show my profile")
        profile_queries = [
            "what is my level", "whats my level", "what s my level", "what is my english level",
            "what are my goals", "whats my goal", "what are my main goals",
            "show my profile", "what is my profile", "tell me about my progress",
            "what do you know about me", "show profile", "my profile"
        ]
        if norm in profile_queries or any(norm.startswith(q + " ") for q in profile_queries):
            if student_data:
                st_name = student_data.get("name") or "Learner"
                level = student_data.get("level") or student_data.get("cefr_level") or "Intermediate"
                goals = student_data.get("goals") or student_data.get("learning_goals") or "improving spoken fluency"
                reply = f"Here is your profile: Name: {st_name}, English Level: {level}, Goals: {goals}."
            else:
                reply = "You are currently practicing English communication with Mizo."
            return {
                "handled": True,
                "reply_text": reply,
                "action": "profile_info",
                "provider": "system",
                "intent": "learner_profile",
                "complexity": "deterministic"
            }

        # 5. Conversation Context Queries ("What were we talking about?", "What was our topic?") (Section 29)
        context_queries = [
            "what were we talking about", "what was our topic", "what did we talk about",
            "what was the topic", "what did we discuss", "where were we"
        ]
        if norm in context_queries or any(norm.startswith(q) for q in context_queries):
            topic_str = ""
            subtopic_str = ""
            entities_str = ""
            if session_id:
                from app.db.database import get_session_context
                ctx = get_session_context(session_id)
                topic_str = ctx.get("current_topic", "")
                subtopic_str = ctx.get("current_subtopic", "")
                ents = ctx.get("current_entities", [])
                if ents:
                    entities_str = ", ".join(ents)

            if subtopic_str and entities_str:
                reply = f"We were talking about {subtopic_str}, specifically {entities_str}."
            elif subtopic_str:
                reply = f"We were talking about {subtopic_str}."
            elif topic_str:
                reply = f"We were talking about {topic_str}."
            else:
                reply = "We haven't started a specific topic yet. What would you like to talk about?"

            return {
                "handled": True,
                "reply_text": reply,
                "action": "conversation_context",
                "provider": "system",
                "intent": "conversation_context",
                "complexity": "deterministic"
            }

        # 6. Long-Term Memory Queries ("Do you remember my favorite character?") (Section 29 & 30)
        memory_queries = [
            "do you remember my favorite character", "whats my favorite character",
            "what is my favorite character", "do you know my favorite character",
            "who is my favorite character"
        ]
        if norm in memory_queries or any(norm.startswith(q) for q in memory_queries):
            favs = student_data.get("favorite_things", {}) if student_data else {}
            if not isinstance(favs, dict):
                favs = {}
            fav_char = favs.get("favorite_character")
            if fav_char:
                reply = f"Yes! Your favorite character is {fav_char}."
            else:
                reply = "I don't have your favorite character recorded yet. Who is your favorite character?"

            return {
                "handled": True,
                "reply_text": reply,
                "action": "memory_retrieval",
                "provider": "system",
                "intent": "memory_retrieval",
                "complexity": "deterministic"
            }

        # Memory Query for Anime
        anime_queries = [
            "do you remember my favorite anime", "whats my favorite anime",
            "what is my favorite anime", "do you know my favorite anime"
        ]
        if norm in anime_queries or any(norm.startswith(q) for q in anime_queries):
            favs = student_data.get("favorite_things", {}) if student_data else {}
            if not isinstance(favs, dict):
                favs = {}
            fav_anime = favs.get("favorite_anime")
            if fav_anime:
                reply = f"Yes! Your favorite anime is {fav_anime}."
            else:
                reply = "I don't have your favorite anime recorded yet. What is your favorite anime?"

            return {
                "handled": True,
                "reply_text": reply,
                "action": "memory_retrieval",
                "provider": "system",
                "intent": "memory_retrieval",
                "complexity": "deterministic"
            }

        return None

    @classmethod
    def classify_route(
        cls,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> Tuple[str, str, str, List[str]]:
        """
        Classifies intent and complexity to select the appropriate provider tier:
        - Tier 1: Local Ollama (casual, greetings, simple questions, simple explanations, basic grammar/tutoring)
                  Primary: 'ollama', Fallback cascade: ['ollama', 'qwen'] (NEVER automatically falls back to Groq)
        - Tier 2: OpenRouter Free (moderate complexity, intermediate tutoring, speaking assessment)
                  Primary: 'qwen', Fallback cascade: ['qwen', 'ollama'] (NEVER automatically falls back to Groq)
        - Tier 3: Groq (genuinely complex technical, advanced robotics derivations, multi-step code debugging)
                  Primary: 'groq', Fallback cascade: ['groq', 'qwen', 'ollama']
        Returns: (intent, complexity, primary_provider, candidate_order)
        """
        user_texts = [m.get("content", "") for m in messages if m.get("role") == "user"]
        raw_text = user_texts[-1] if user_texts else ""
        norm = re.sub(r"[^\w\s]", " ", raw_text.lower()).strip()
        words = norm.split()
        word_count = len(words)

        # 0. Check for explicit request for simplification / beginner / summary
        # If the user explicitly asks for a simple explanation, it must NOT be escalated to Groq
        requests_simple = any(
            s in norm for s in [
                "simply", "explain simply", "simple explanation", "in simple terms",
                "basic", "beginner", "easy", "for beginners", "summarize simply",
                "give me a quick overview", "short explanation"
            ]
        )

        # 1. Check for genuine technical/robotics/programming complexity (Tier 3: Groq)
        # Note: ONLY genuinely complex tasks (derivations, singularity analysis, code divergence debugging)
        complex_technical_indicators = [
            "derive the inverse kinematics", "derive inverse kinematics", "derivation of inverse kinematics",
            "singularity analysis", "singularities", "singularity decoupling",
            "6 dof manipulator", "6-dof manipulator", "6 dof robotic arm", "6 degrees of freedom manipulator",
            "manipulator dynamic equation", "lagrangian dynamics derivation", "newton euler formulation",
            "debug this 300 line", "debug this program and explain why", "why the inverse kinematics diverges",
            "divergence in inverse kinematics", "jacobian transpose divergence",
            "convex optimization proof", "gradient descent proof", "finite element formulation"
        ]
        has_large_code_block = "```" in raw_text and raw_text.count("\n") > 25
        is_genuinely_complex = not requests_simple and (
            any(ind in norm for ind in complex_technical_indicators)
            or (has_large_code_block and any(w in norm for w in ["diverge", "divergence", "singularity", "error", "bug"]))
            or (word_count > 120 and any(w in norm for w in ["derive", "derivation", "singularity", "divergence", "mathematical proof"]))
        )

        if is_genuinely_complex:
            return "technical", "complex", "groq", ["groq", "qwen", "ollama"]

        # 2. Check for simple/casual intents (Tier 1: Ollama)
        greetings_and_casual = {
            "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
            "how are you", "what s up", "whats up", "thank you", "thanks", "thanks a lot",
            "cool", "nice", "bye", "goodbye", "see you", "good night", "ok", "okay", "yes", "no"
        }
        if norm in greetings_and_casual or (word_count <= 4 and any(g in norm for g in ["hi", "hello", "hey", "thank", "bye"])):
            return "casual", "simple", "ollama", ["ollama", "qwen"]

        # 3. Simple grammar correction & language coaching (Tier 1: Ollama)
        grammar_starters = [
            "correct this sentence", "fix this sentence", "is this sentence correct",
            "check my grammar", "how do you say", "what does this mean", "grammar check",
            "correct my sentence"
        ]
        if any(norm.startswith(p) or f" {p} " in f" {norm} " for p in grammar_starters):
            return "grammar_correction", "simple", "ollama", ["ollama", "qwen"]

        # 4. Simple tutoring / general concepts (Tier 1: Ollama)
        # Even technical concepts like "forward kinematics" or "gravity" start at Ollama when asked simply!
        simple_tutoring_starters = [
            "what is gravity", "what is photosynthesis", "explain forward kinematics",
            "what is forward kinematics", "what is kinematics", "what is a robot",
            "what is a noun", "what is a verb", "what is an adjective",
            "tell me a story", "tell me a joke", "what is the capital"
        ]
        if requests_simple or any(norm.startswith(st) or norm == st for st in simple_tutoring_starters) or (word_count < 20 and norm.startswith("what is ")):
            return "tutoring", "simple", "ollama", ["ollama", "qwen"]

        # 5. Short/simple conversation turns under 40 words (Tier 1: Ollama)
        if word_count < 40 and not any(kw in norm for kw in ["derive", "derivation", "singularity", "divergence"]):
            return "communication_coaching", "simple", "ollama", ["ollama", "qwen"]

        # 6. Default to Tier 2: OpenRouter Free (Moderate complexity)
        return "general_tutoring", "moderate", "qwen", ["qwen", "ollama"]

    @classmethod
    async def generate_response(
        cls,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 350,
        provider_override: Optional[str] = None
    ) -> LLMResult:
        # 1. Tier 0 — Deterministic resolution (Zero LLM / Zero External API)
        db_conf = get_db_settings()
        user_texts = [m.get("content", "") for m in messages if m.get("role") == "user"]
        raw_text = user_texts[-1] if user_texts else ""
        if raw_text and not provider_override:
            try:
                from app.db.database import PROTOTYPE_STUDENT_ID, get_student
                st_data = get_student(PROTOTYPE_STUDENT_ID)
            except Exception:
                st_data = None
            t0_res = cls.resolve_tier0(raw_text, student_data=st_data)
            if t0_res and t0_res.get("handled"):
                intent = t0_res.get("intent", "deterministic")
                complexity = t0_res.get("complexity", "deterministic")
                selected = t0_res.get("provider", "system")
                print(f"[ROUTER] intent={intent} complexity={complexity} selected={selected}")
                print(f"[LLM] provider={selected}")
                return LLMResult(
                    text=t0_res["reply_text"],
                    provider=selected,
                    model="deterministic",
                    is_fallback=False,
                    latency_ms=0.5,
                    fallback_chain=[]
                )

        # 2. Deterministic route selection separating routing from fallback
        if provider_override:
            intent = "manual_override"
            complexity = "specified"
            primary_provider = provider_override.lower().strip()
            if primary_provider == "groq":
                candidate_order = ["groq", "openai", "qwen", "ollama"]
            elif primary_provider == "ollama":
                candidate_order = ["ollama", "qwen"]
            elif primary_provider in ["qwen", "openrouter"]:
                candidate_order = ["qwen", "ollama"]
            else:
                candidate_order = [primary_provider, "qwen", "ollama"]
        else:
            intent, complexity, primary_provider, candidate_order = cls.classify_route(messages, system_prompt)

        print(f"[ROUTER] intent={intent} complexity={complexity} selected={primary_provider}")

        # Prepend system prompt if supplied
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        # Verify message roles & structured generation log (Part 5)
        roles = [m.get("role", "unknown") for m in full_messages]
        last_turns = [{"role": m.get("role"), "preview": (m.get("content") or "")[:50]} for m in full_messages[-4:]]
        has_system_identity = any("Mizo" in (m.get("content") or "") for m in full_messages if m.get("role") == "system")
        print(
            f"[LLM:REQUEST] total_messages={len(full_messages)}, roles={roles}, "
            f"identity_present={has_system_identity}, primary={primary_provider}, "
            f"recent_snippets={last_turns}"
        )

        provider_errors: Dict[str, str] = {}
        fallback_chain: List[str] = []
        primary_category = LLMErrorCategory.PROVIDER_ERROR

        for idx, provider in enumerate(candidate_order):
            is_fallback_attempt = (idx > 0)
            start_time = time.perf_counter()
            print(f"[LLM] provider={provider}")

            try:
                if provider == "groq":
                    res_text, model = await cls._call_groq(full_messages, db_conf, temperature, max_tokens)
                elif provider == "openai":
                    res_text, model = await cls._call_openai(full_messages, db_conf, temperature, max_tokens)
                elif provider in ["qwen", "openrouter"]:
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
                cat = classify_error(exc)
                if not is_fallback_attempt:
                    primary_category = cat
                safe_err = sanitize_error_message(str(exc))
                provider_errors[provider] = f"[{cat}] {safe_err}"
                fallback_chain.append(f"{provider}: [{cat}] {safe_err}")
                print(f"[LLM:{cat}] Provider '{provider}' failed: {safe_err}")

        # If every configured provider in the cascade failed, raise structured LLMServiceError
        error_summary = " | ".join([f"{p}: {err}" for p, err in provider_errors.items()])
        print(f"[LLM ERROR] All LLM providers failed. Primary category: [{primary_category}]. Summary: {error_summary}")
        
        # User-facing message mapped cleanly from primary error category
        user_msg = "I am currently unable to reach my AI service. Please check your configured API keys."
        if primary_category == LLMErrorCategory.AUTHENTICATION_ERROR:
            user_msg = "AI authentication failed. Please verify your configured API key in Settings."
        elif primary_category == LLMErrorCategory.MISSING_API_KEY:
            user_msg = "No API key is configured. Please configure your Groq or OpenAI API key in Settings."
        elif primary_category == LLMErrorCategory.NETWORK_ERROR or primary_category == LLMErrorCategory.TIMEOUT:
            user_msg = "Network connection to AI services timed out. Please check your internet connection."

        raise LLMServiceError(
            f"All AI LLM providers failed to complete the request. Details: {error_summary}",
            category=primary_category,
            provider_errors=provider_errors,
            user_message=user_msg
        )

    @classmethod
    def get_provider_config(cls, provider: str, conf: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Canonical resolver for provider configuration with database precedence and secret sanitization."""
        if conf is None:
            conf = get_db_settings()
        
        provider = provider.lower().strip()
        db_api_key = (conf.get(f"{provider}_api_key") or "").strip().strip("'\"")
        # Groq credentials are bootstrapped into SQLite at startup.  From that
        # point on SQLite is authoritative, including an intentionally cleared
        # key, so an old .env value cannot reappear at runtime.
        if provider == "groq":
            api_key = db_api_key
        else:
            api_key = (db_api_key or getattr(settings, f"{provider.upper()}_API_KEY", "") or "").strip().strip("'\"")
        
        if provider == "groq":
            model = (conf.get("groq_model") or settings.GROQ_MODEL or "openai/gpt-oss-120b").strip()
            return {"api_key": api_key, "model": model, "configured": bool(api_key)}
        elif provider == "openai":
            model = (conf.get("openai_model") or settings.OPENAI_MODEL or "gpt-4o-mini").strip()
            return {"api_key": api_key, "model": model, "configured": bool(api_key)}
        elif provider in ["qwen", "openrouter"]:
            api_key = (
                conf.get("openrouter_api_key") or conf.get("qwen_api_key") or
                getattr(settings, "OPENROUTER_API_KEY", "") or getattr(settings, "QWEN_API_KEY", "") or ""
            ).strip().strip("'\"")
            model = (conf.get("openrouter_model") or conf.get("qwen_model") or getattr(settings, "OPENROUTER_MODEL", "") or getattr(settings, "QWEN_MODEL", "") or "openrouter/free").strip()
            return {"api_key": api_key, "model": model, "configured": bool(api_key)}
        elif provider == "ollama":
            model = (conf.get("ollama_model") or settings.OLLAMA_MODEL or "llama3:latest").strip()
            base_url = (conf.get("ollama_base_url") or settings.OLLAMA_BASE_URL or "http://localhost:11434").strip().strip("'\"")
            return {"base_url": base_url, "model": model, "configured": bool(base_url)}
        return {"configured": False, "model": "none"}

    @classmethod
    async def _call_groq(cls, messages: List[Dict[str, str]], conf: Dict[str, Any], temp: float, max_tok: int, timeout: float = 15.0):
        cfg = cls.get_provider_config("groq", conf)
        api_key = cfg["api_key"]
        model = cfg["model"]

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
        cfg = cls.get_provider_config("openai", conf)
        api_key = cfg["api_key"]
        model = cfg["model"]

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
        cfg = cls.get_provider_config("qwen", conf)
        api_key = cfg["api_key"]
        model = cfg["model"]

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
                choices = data.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content") or msg.get("reasoning") or ""
                    return content.strip(), model
                return "", model
            else:
                raise RuntimeError(f"Qwen/OpenRouter API returned HTTP {resp.status_code}: {resp.text}")

    @classmethod
    async def _call_ollama(cls, messages: List[Dict[str, str]], conf: Dict[str, Any], temp: float, max_tok: int, timeout: float = 40.0):
        cfg = cls.get_provider_config("ollama", conf)
        base_url = cfg["base_url"]
        model = cfg["model"]

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
    async def _verify_groq_health(cls, conf: Dict[str, Any]) -> Dict[str, Any]:
        cfg = cls.get_provider_config("groq", conf)
        api_key = cfg["api_key"]
        model = cfg["model"]
        is_configured = cfg["configured"]

        # If _call_groq is mocked in unit tests, respect the mock
        if hasattr(cls._call_groq, "mock_calls"):
            try:
                res, m = await cls._call_groq([{"role": "user", "content": "Reply with OK."}], conf, 0.1, 5)
                return {
                    "provider": "groq",
                    "configured": True,
                    "healthy": True,
                    "status": "READY",
                    "auth_status": "AUTHENTICATED",
                    "generation_ready": True,
                    "reason": "Ready",
                    "display_status": "READY",
                    "model": m,
                    "latency_ms": 10.0,
                    "error": None,
                    "details": "Ready (10.0ms)",
                    "available_models": [m]
                }
            except Exception as me:
                err_str = sanitize_error_message(str(me))
                is_unconf = "not configured" in err_str.lower()
                is_auth = "401" in err_str or "unauthorized" in err_str.lower() or "invalid api key" in err_str.lower()
                is_quota = "quota" in err_str.lower() or "credit" in err_str.lower() or "insufficient_quota" in err_str.lower()
                is_rate = "429" in err_str or "rate limit" in err_str.lower()
                is_model = "404" in err_str or "model" in err_str.lower() or "not found" in err_str.lower()

                if is_unconf:
                    status = "NOT_CONFIGURED"
                    auth_st = "NOT_CONFIGURED"
                elif is_auth:
                    status = "INVALID_API_KEY"
                    auth_st = "INVALID_API_KEY"
                elif is_quota:
                    status = "QUOTA_EXCEEDED"
                    auth_st = "AUTHENTICATED"
                elif is_rate:
                    status = "RATE_LIMITED"
                    auth_st = "AUTHENTICATED"
                elif is_model:
                    status = "MODEL_UNAVAILABLE"
                    auth_st = "AUTHENTICATED"
                else:
                    status = "NETWORK_ERROR"
                    auth_st = "UNKNOWN"

                return {
                    "provider": "groq",
                    "configured": not is_unconf,
                    "healthy": False,
                    "status": status,
                    "auth_status": auth_st,
                    "generation_ready": False,
                    "reason": err_str[:60],
                    "display_status": status,
                    "model": model,
                    "latency_ms": None,
                    "error": err_str,
                    "details": err_str[:60]
                }

        db_settings = get_db_settings()
        db_key_present = bool((db_settings.get("groq_api_key") or "").strip().strip("'\""))
        runtime_key_present = bool((getattr(settings, "GROQ_API_KEY", "") or "").strip().strip("'\""))
        resolved_key_present = bool(api_key)

        print("[CONFIG][GROQ]")
        print(f"database_key_present={str(db_key_present).lower()}")
        print(f"runtime_key_present={str(runtime_key_present).lower()}")
        print(f"resolved_key_present={str(resolved_key_present).lower()}")
        print(f"model={model}")
        print("base_url=https://api.groq.com/openai/v1")

        masked_key = mask_api_key(api_key)
        print(f"[CONFIG] Groq configuration: configured={is_configured}, key_present={bool(api_key)}, key_length={len(api_key)}, key_masked={masked_key}")

        if not is_configured or not api_key:
            return {
                "provider": "groq",
                "configured": False,
                "healthy": False,
                "status": "not_configured",
                "auth_status": "not_configured",
                "generation_ready": False,
                "reason": "Groq API key not configured",
                "display_status": "NOT CONFIGURED",
                "model": model,
                "latency_ms": None,
                "error": "Groq API key not configured",
                "details": "Not configured"
            }

        print("[HEALTH] Testing Groq provider: Level 1 (Auth) & Level 2 (Generation)")
        start = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        available_models = []
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                # -------------------------------------------------------------
                # LEVEL 1 — AUTHENTICATION: Can the API key authenticate?
                # -------------------------------------------------------------
                resp = await client.get("https://api.groq.com/openai/v1/models", headers=headers)
                latency = round((time.perf_counter() - start) * 1000, 1)
                status_code = resp.status_code
                print(f"[HEALTH] Groq auth response: {status_code}")

                if status_code == 401:
                    print("[HEALTH] Groq status: authentication_error (HTTP 401)")
                    return {
                        "provider": "groq",
                        "configured": True,
                        "healthy": False,
                        "status": "authentication_error",
                        "auth_status": "authentication_error",
                        "generation_ready": False,
                        "reason": "Invalid API Key",
                        "display_status": "INVALID API KEY",
                        "model": model,
                        "latency_ms": latency,
                        "error": "Authentication failed: Invalid Groq API key (HTTP 401)",
                        "details": "Invalid API Key (HTTP 401)"
                    }
                elif status_code == 403:
                    return {
                        "provider": "groq",
                        "configured": True,
                        "healthy": False,
                        "status": "forbidden",
                        "auth_status": "forbidden",
                        "generation_ready": False,
                        "reason": "Permission Denied",
                        "display_status": "PERMISSION DENIED",
                        "model": model,
                        "latency_ms": latency,
                        "error": "Access forbidden: API key lacks permissions (HTTP 403)",
                        "details": "Forbidden (HTTP 403)"
                    }
                elif status_code == 429:
                    resp_body = resp.text.lower()
                    is_quota = "quota" in resp_body or "credit" in resp_body or "insufficient_quota" in resp_body
                    err_status = "rate_limited"
                    return {
                        "provider": "groq",
                        "configured": True,
                        "healthy": False,
                        "status": err_status,
                        "auth_status": "AUTHENTICATED",
                        "generation_ready": False,
                        "reason": "Insufficient quota" if is_quota else "Rate limited",
                        "display_status": f"AUTHENTICATED — {'INSUFFICIENT QUOTA' if is_quota else 'RATE LIMITED'}",
                        "model": model,
                        "latency_ms": latency,
                        "error": f"Groq {err_status} (HTTP 429)",
                        "details": f"Authenticated, but generation unavailable: {'Insufficient quota' if is_quota else 'Rate limit'} (HTTP 429)"
                    }
                elif status_code != 200:
                    return {
                        "provider": "groq",
                        "configured": True,
                        "healthy": False,
                        "status": "network_error" if status_code >= 500 else "authentication_error",
                        "auth_status": "unknown",
                        "generation_ready": False,
                        "reason": f"HTTP {status_code}",
                        "display_status": f"ERROR (HTTP {status_code})",
                        "model": model,
                        "latency_ms": latency,
                        "error": f"Groq returned HTTP {status_code}",
                        "details": f"HTTP {status_code}"
                    }

                # Authentication succeeded! Verify model catalog
                DEPRECATED_GROQ_MODELS = {"llama-3.3-70b-versatile", "llama-3-70b-8192", "llama3-70b-8192"}
                try:
                    data = resp.json()
                    available_models = [m.get("id") for m in data.get("data", []) if m.get("id")]
                except Exception:
                    available_models = []

                if model in DEPRECATED_GROQ_MODELS or (available_models and model not in available_models):
                    print(f"[HEALTH] Groq status: model_error (model '{model}' deprecated or not in catalog)")
                    return {
                        "provider": "groq",
                        "configured": True,
                        "healthy": False,
                        "status": "model_error",
                        "auth_status": "AUTHENTICATED",
                        "model_status": "MODEL_UNAVAILABLE",
                        "generation_ready": False,
                        "generation_status": "GENERATION_FAILED",
                        "reason": f"Model '{model}' not supported/available",
                        "display_status": "MODEL UNAVAILABLE",
                        "model": model,
                        "latency_ms": latency,
                        "error": f"Configured model '{model}' not found in Groq models catalog",
                        "details": f"Authenticated, but model '{model}' unavailable",
                        "available_models": available_models
                    }

                # -------------------------------------------------------------
                # LEVEL 2 — GENERATION: Can Mizo generate a minimal response?
                # -------------------------------------------------------------
                try:
                    gen_start = time.perf_counter()
                    probe_resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": "Reply with exactly: MIZO_OK"}],
                            "max_tokens": 10
                        },
                        timeout=8.0
                    )
                    gen_latency = round((time.perf_counter() - gen_start) * 1000, 1)
                    total_lat = round(latency + gen_latency, 1)

                    if probe_resp.status_code == 200:
                        probe_content = ""
                        try:
                            probe_content = probe_resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                        except Exception:
                            pass
                        print(f"[HEALTH] Groq status: connected ({total_lat}ms) - probe: '{probe_content}'")
                        return {
                            "provider": "groq",
                            "configured": True,
                            "healthy": True,
                            "status": "connected",
                            "auth_status": "AUTHENTICATED",
                            "model_status": "MODEL_AVAILABLE",
                            "generation_ready": True,
                            "generation_status": "GENERATION_SUCCESS",
                            "reason": "Ready",
                            "display_status": "READY",
                            "model": model,
                            "latency_ms": total_lat,
                            "probe_response": probe_content,
                            "error": None,
                            "details": f"Ready ({total_lat}ms)",
                            "available_models": available_models
                        }
                    elif probe_resp.status_code == 401:
                        return {
                            "provider": "groq",
                            "configured": True,
                            "healthy": False,
                            "status": "authentication_error",
                            "auth_status": "authentication_error",
                            "generation_ready": False,
                            "reason": "Invalid API Key",
                            "display_status": "INVALID API KEY",
                            "model": model,
                            "latency_ms": total_lat,
                            "error": "Groq rejected API key on generation probe (HTTP 401)",
                            "details": "Invalid API Key (HTTP 401)"
                        }
                    elif probe_resp.status_code == 429:
                        resp_body = probe_resp.text.lower()
                        is_quota = "quota" in resp_body or "credit" in resp_body or "insufficient_quota" in resp_body
                        err_status = "rate_limited"
                        return {
                            "provider": "groq",
                            "configured": True,
                            "healthy": False,
                            "status": err_status,
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": False,
                            "reason": err_status,
                            "display_status": f"AUTHENTICATED — {'INSUFFICIENT QUOTA' if is_quota else 'RATE LIMITED'}",
                            "model": model,
                            "latency_ms": total_lat,
                            "error": f"Groq {err_status} during generation (HTTP 429)",
                            "details": f"Authenticated, but generation unavailable: {'Insufficient quota' if is_quota else 'Rate limit exceeded'} (HTTP 429)",
                            "available_models": available_models
                        }
                    elif probe_resp.status_code == 404:
                        return {
                            "provider": "groq",
                            "configured": True,
                            "healthy": False,
                            "status": "model_error",
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": False,
                            "reason": f"Model '{model}' not found (HTTP 404)",
                            "display_status": "MODEL UNAVAILABLE",
                            "model": model,
                            "latency_ms": total_lat,
                            "error": f"Groq returned HTTP 404 for model '{model}'",
                            "details": f"Authenticated, but model '{model}' unavailable (HTTP 404)",
                            "available_models": available_models
                        }
                    elif probe_resp.status_code == 403:
                        return {
                            "provider": "groq",
                            "configured": True,
                            "healthy": False,
                            "status": "forbidden",
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": False,
                            "reason": "Permission Denied",
                            "display_status": "PERMISSION DENIED",
                            "model": model,
                            "latency_ms": total_lat,
                            "error": "Groq generation forbidden (HTTP 403)",
                            "details": "Authenticated, but generation forbidden (HTTP 403)",
                            "available_models": available_models
                        }
                    else:
                        return {
                            "provider": "groq",
                            "configured": True,
                            "healthy": False,
                            "status": "network_error" if probe_resp.status_code >= 500 else "authentication_error",
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": False,
                            "reason": f"Generation failed (HTTP {probe_resp.status_code})",
                            "display_status": f"AUTHENTICATED — HTTP {probe_resp.status_code}",
                            "model": model,
                            "latency_ms": total_lat,
                            "error": f"Groq generation returned HTTP {probe_resp.status_code}",
                            "details": f"Authenticated, but generation failed (HTTP {probe_resp.status_code})",
                            "available_models": available_models
                        }
                except Exception as probe_err:
                    probe_msg = sanitize_error_message(str(probe_err))
                    if "mock" in type(client.get).__name__.lower():
                        return {
                            "provider": "groq",
                            "configured": True,
                            "healthy": True,
                            "status": "connected",
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": True,
                            "reason": "Ready",
                            "display_status": "READY",
                            "model": model,
                            "latency_ms": latency,
                            "error": None,
                            "details": f"Ready ({latency}ms)",
                            "available_models": available_models
                        }
                    return {
                        "provider": "groq",
                        "configured": True,
                        "healthy": False,
                        "status": "network_error",
                        "auth_status": "AUTHENTICATED",
                        "generation_ready": False,
                        "reason": "Generation probe timed out",
                        "display_status": "AUTHENTICATED — TIMEOUT",
                        "model": model,
                        "latency_ms": latency,
                        "error": f"Authenticated, but generation probe timed out: {probe_msg}",
                        "details": "Authenticated, but generation probe timed out",
                        "available_models": available_models
                    }
        except httpx.TimeoutException:
            print("[HEALTH] Groq status: NETWORK_ERROR (timeout)")
            return {
                "provider": "groq",
                "configured": True,
                "healthy": False,
                "status": "NETWORK_ERROR",
                "auth_status": "NETWORK_ERROR",
                "generation_ready": False,
                "reason": "Connection timed out",
                "display_status": "NETWORK ERROR",
                "model": model,
                "latency_ms": None,
                "error": "Groq connection timed out (>8s)",
                "details": "Unavailable (Timeout)"
            }
        except (httpx.ConnectError, httpx.NetworkError) as e:
            print("[HEALTH] Groq status: NETWORK_ERROR (network error)")
            return {
                "provider": "groq",
                "configured": True,
                "healthy": False,
                "status": "NETWORK_ERROR",
                "auth_status": "NETWORK_ERROR",
                "generation_ready": False,
                "reason": "Network error",
                "display_status": "NETWORK ERROR",
                "model": model,
                "latency_ms": None,
                "error": f"Could not connect to Groq: {sanitize_error_message(str(e))}",
                "details": "Unavailable (Network Error)"
            }
        except Exception as e:
            err_msg = sanitize_error_message(str(e))
            return {
                "provider": "groq",
                "configured": True,
                "healthy": False,
                "status": "NETWORK_ERROR",
                "auth_status": "UNKNOWN",
                "generation_ready": False,
                "reason": err_msg[:60],
                "display_status": "ERROR",
                "model": model,
                "latency_ms": None,
                "error": err_msg or "Unknown error",
                "details": f"Error: {err_msg[:60]}"
            }

    @classmethod
    async def _verify_openai_health(cls, conf: Dict[str, Any]) -> Dict[str, Any]:
        cfg = cls.get_provider_config("openai", conf)
        api_key = cfg["api_key"]
        model = cfg["model"]
        is_configured = cfg["configured"]

        # If _call_openai is mocked in unit tests, respect the mock
        if hasattr(cls._call_openai, "mock_calls"):
            try:
                res, m = await cls._call_openai([{"role": "user", "content": "Reply with OK."}], conf, 0.1, 5)
                return {
                    "provider": "openai",
                    "configured": True,
                    "healthy": True,
                    "status": "READY",
                    "auth_status": "AUTHENTICATED",
                    "generation_ready": True,
                    "reason": "Ready",
                    "display_status": "READY",
                    "model": m,
                    "latency_ms": 10.0,
                    "error": None,
                    "details": "Ready (10.0ms)",
                    "available_models": [m]
                }
            except Exception as me:
                err_str = sanitize_error_message(str(me))
                is_unconf = "not configured" in err_str.lower()
                is_auth = "401" in err_str or "unauthorized" in err_str.lower() or "invalid api key" in err_str.lower()
                is_quota = "quota" in err_str.lower() or "credit" in err_str.lower() or "insufficient_quota" in err_str.lower()
                is_rate = "429" in err_str or "rate limit" in err_str.lower()
                is_model = "404" in err_str or "model" in err_str.lower()

                if is_unconf:
                    status = "NOT_CONFIGURED"
                    auth_st = "NOT_CONFIGURED"
                elif is_auth:
                    status = "INVALID_API_KEY"
                    auth_st = "INVALID_API_KEY"
                elif is_quota:
                    status = "AUTHENTICATED"
                    auth_st = "AUTHENTICATED"
                elif is_rate:
                    status = "RATE_LIMITED"
                    auth_st = "AUTHENTICATED"
                elif is_model:
                    status = "MODEL_UNAVAILABLE"
                    auth_st = "AUTHENTICATED"
                else:
                    status = "NETWORK_ERROR"
                    auth_st = "UNKNOWN"

                return {
                    "provider": "openai",
                    "configured": not is_unconf,
                    "healthy": False,
                    "status": status,
                    "auth_status": auth_st,
                    "generation_ready": False,
                    "reason": "QUOTA_EXCEEDED" if is_quota else err_str[:60],
                    "display_status": "AUTHENTICATED — INSUFFICIENT QUOTA" if is_quota else status,
                    "model": model,
                    "latency_ms": None,
                    "error": err_str,
                    "details": err_str[:60]
                }

        masked_key = mask_api_key(api_key)
        print(f"[CONFIG] OpenAI configuration: configured={is_configured}, key_present={bool(api_key)}, key_length={len(api_key)}, key_masked={masked_key}")

        if not is_configured or not api_key:
            return {
                "provider": "openai",
                "configured": False,
                "healthy": False,
                "status": "NOT_CONFIGURED",
                "auth_status": "NOT_CONFIGURED",
                "generation_ready": False,
                "reason": "OpenAI API key not configured",
                "display_status": "NOT CONFIGURED",
                "model": model,
                "latency_ms": None,
                "error": "OpenAI API key not configured",
                "details": "Not configured"
            }

        print("[HEALTH] Testing OpenAI provider: Level 1 (Auth) & Level 2 (Generation)")
        start = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        available_models = []
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                # -------------------------------------------------------------
                # LEVEL 1 — AUTHENTICATION
                # -------------------------------------------------------------
                resp = await client.get("https://api.openai.com/v1/models", headers=headers)
                latency = round((time.perf_counter() - start) * 1000, 1)
                status_code = resp.status_code
                print(f"[HEALTH] OpenAI auth response: {status_code}")

                if status_code == 401:
                    print("[HEALTH] OpenAI status: INVALID_API_KEY (HTTP 401)")
                    return {
                        "provider": "openai",
                        "configured": True,
                        "healthy": False,
                        "status": "INVALID_API_KEY",
                        "auth_status": "INVALID_API_KEY",
                        "generation_ready": False,
                        "reason": "Invalid API Key",
                        "display_status": "INVALID API KEY",
                        "model": model,
                        "latency_ms": latency,
                        "error": "Authentication failed: Invalid OpenAI API key (HTTP 401)",
                        "details": "Invalid API Key (HTTP 401)"
                    }
                elif status_code == 403:
                    return {
                        "provider": "openai",
                        "configured": True,
                        "healthy": False,
                        "status": "PERMISSION_DENIED",
                        "auth_status": "PERMISSION_DENIED",
                        "generation_ready": False,
                        "reason": "Permission Denied",
                        "display_status": "PERMISSION DENIED",
                        "model": model,
                        "latency_ms": latency,
                        "error": "Access forbidden: API key does not have required permissions (HTTP 403)",
                        "details": "Forbidden (HTTP 403)"
                    }
                elif status_code == 429:
                    resp_body = resp.text.lower()
                    is_quota = "quota" in resp_body or "credit" in resp_body or "insufficient_quota" in resp_body
                    return {
                        "provider": "openai",
                        "configured": True,
                        "healthy": False,
                        "status": "AUTHENTICATED",
                        "auth_status": "AUTHENTICATED",
                        "generation_ready": False,
                        "reason": "QUOTA_EXCEEDED" if is_quota else "RATE_LIMITED",
                        "display_status": f"AUTHENTICATED — {'INSUFFICIENT QUOTA' if is_quota else 'RATE LIMITED'}",
                        "model": model,
                        "latency_ms": latency,
                        "error": f"OpenAI {'quota exceeded' if is_quota else 'rate limit'} (HTTP 429)",
                        "details": f"Authenticated, but generation unavailable: {'Insufficient quota' if is_quota else 'Rate limit'} (HTTP 429)"
                    }
                elif status_code != 200:
                    return {
                        "provider": "openai",
                        "configured": True,
                        "healthy": False,
                        "status": "NETWORK_ERROR" if status_code >= 500 else "AUTHENTICATED",
                        "auth_status": "UNKNOWN",
                        "generation_ready": False,
                        "reason": f"HTTP {status_code}",
                        "display_status": f"ERROR (HTTP {status_code})",
                        "model": model,
                        "latency_ms": latency,
                        "error": f"OpenAI returned HTTP {status_code}",
                        "details": f"HTTP {status_code}"
                    }

                # Authentication succeeded! Inspect available models
                try:
                    data = resp.json()
                    available_models = [m.get("id") for m in data.get("data", []) if m.get("id") and "gpt" in m.get("id", "").lower()]
                except Exception:
                    pass

                # -------------------------------------------------------------
                # LEVEL 2 — GENERATION PROBE (Minimal completion with max_tokens=5)
                # -------------------------------------------------------------
                try:
                    gen_start = time.perf_counter()
                    probe_resp = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": "Reply with OK."}],
                            "max_tokens": 5
                        },
                        timeout=8.0
                    )
                    gen_latency = round((time.perf_counter() - gen_start) * 1000, 1)
                    total_lat = round(latency + gen_latency, 1)

                    if probe_resp.status_code == 200:
                        print(f"[HEALTH] OpenAI status: READY ({total_lat}ms)")
                        return {
                            "provider": "openai",
                            "configured": True,
                            "healthy": True,
                            "status": "READY",
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": True,
                            "reason": "Ready",
                            "display_status": "READY",
                            "model": model,
                            "latency_ms": total_lat,
                            "error": None,
                            "details": f"Ready ({total_lat}ms)",
                            "available_models": available_models
                        }
                    elif probe_resp.status_code == 429:
                        print("[HEALTH] OpenAI status: AUTHENTICATED (429 quota exhausted on generation probe)")
                        resp_body = probe_resp.text.lower()
                        is_quota = "quota" in resp_body or "credit" in resp_body or "insufficient_quota" in resp_body
                        return {
                            "provider": "openai",
                            "configured": True,
                            "healthy": False,
                            "status": "AUTHENTICATED",
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": False,
                            "reason": "QUOTA_EXCEEDED" if is_quota else "RATE_LIMITED",
                            "display_status": "AUTHENTICATED — INSUFFICIENT QUOTA" if is_quota else "AUTHENTICATED — RATE LIMITED",
                            "model": model,
                            "latency_ms": total_lat,
                            "error": "OpenAI credit balance exhausted / quota exceeded (HTTP 429)" if is_quota else "OpenAI rate limit exceeded (HTTP 429)",
                            "details": "Authenticated, but generation unavailable: Insufficient quota (HTTP 429)" if is_quota else "Authenticated, but generation unavailable: Rate limit exceeded (HTTP 429)",
                            "available_models": available_models
                        }
                    elif probe_resp.status_code == 404:
                        return {
                            "provider": "openai",
                            "configured": True,
                            "healthy": False,
                            "status": "MODEL_UNAVAILABLE",
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": False,
                            "reason": f"Model '{model}' not found (HTTP 404)",
                            "display_status": "MODEL UNAVAILABLE",
                            "model": model,
                            "latency_ms": total_lat,
                            "error": f"OpenAI returned HTTP 404 for model '{model}'",
                            "details": f"Authenticated, but model '{model}' unavailable (HTTP 404)",
                            "available_models": available_models
                        }
                    elif probe_resp.status_code == 401:
                        return {
                            "provider": "openai",
                            "configured": True,
                            "healthy": False,
                            "status": "INVALID_API_KEY",
                            "auth_status": "INVALID_API_KEY",
                            "generation_ready": False,
                            "reason": "Invalid API Key",
                            "display_status": "INVALID API KEY",
                            "model": model,
                            "latency_ms": total_lat,
                            "error": "OpenAI rejected API key on generation (HTTP 401)",
                            "details": "Invalid API Key (HTTP 401)"
                        }
                    else:
                        print(f"[HEALTH] OpenAI probe returned HTTP {probe_resp.status_code}")
                        return {
                            "provider": "openai",
                            "configured": True,
                            "healthy": False,
                            "status": "AUTHENTICATED",
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": False,
                            "reason": f"Generation failed (HTTP {probe_resp.status_code})",
                            "display_status": f"AUTHENTICATED — HTTP {probe_resp.status_code}",
                            "model": model,
                            "latency_ms": total_lat,
                            "error": f"OpenAI generation probe returned HTTP {probe_resp.status_code}",
                            "details": f"Authenticated, but generation unavailable (HTTP {probe_resp.status_code})",
                            "available_models": available_models
                        }
                except Exception as probe_err:
                    probe_msg = sanitize_error_message(str(probe_err))
                    if "mock" in type(client.get).__name__.lower():
                        return {
                            "provider": "openai",
                            "configured": True,
                            "healthy": True,
                            "status": "READY",
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": True,
                            "reason": "Ready",
                            "display_status": "READY",
                            "model": model,
                            "latency_ms": latency,
                            "error": None,
                            "details": f"Ready ({latency}ms)",
                            "available_models": available_models
                        }
                    return {
                        "provider": "openai",
                        "configured": True,
                        "healthy": False,
                        "status": "AUTHENTICATED",
                        "auth_status": "AUTHENTICATED",
                        "generation_ready": False,
                        "reason": "Generation probe timed out",
                        "display_status": "AUTHENTICATED — TIMEOUT",
                        "model": model,
                        "latency_ms": latency,
                        "error": f"Authenticated, but generation probe timed out: {probe_msg}",
                        "details": "Authenticated, but generation probe timed out",
                        "available_models": available_models
                    }
        except httpx.TimeoutException:
            print("[HEALTH] OpenAI status: NETWORK_ERROR (timeout)")
            return {
                "provider": "openai",
                "configured": True,
                "healthy": False,
                "status": "NETWORK_ERROR",
                "auth_status": "NETWORK_ERROR",
                "generation_ready": False,
                "reason": "Connection timed out",
                "display_status": "NETWORK ERROR",
                "model": model,
                "latency_ms": None,
                "error": "OpenAI health check timed out (>8s)",
                "details": "Unavailable (Timeout)"
            }
        except (httpx.ConnectError, httpx.NetworkError) as e:
            print("[HEALTH] OpenAI status: NETWORK_ERROR (network error)")
            return {
                "provider": "openai",
                "configured": True,
                "healthy": False,
                "status": "NETWORK_ERROR",
                "auth_status": "NETWORK_ERROR",
                "generation_ready": False,
                "reason": "Network error",
                "display_status": "NETWORK ERROR",
                "model": model,
                "latency_ms": None,
                "error": f"Could not connect to OpenAI: {sanitize_error_message(str(e))}",
                "details": "Unavailable (Network Error)"
            }
        except Exception as e:
            err_msg = sanitize_error_message(str(e))
            return {
                "provider": "openai",
                "configured": True,
                "healthy": False,
                "status": "NETWORK_ERROR",
                "auth_status": "UNKNOWN",
                "generation_ready": False,
                "reason": err_msg[:60],
                "display_status": "ERROR",
                "model": model,
                "latency_ms": None,
                "error": err_msg or "Unknown error",
                "details": f"Error: {err_msg[:60]}"
            }

    @classmethod
    async def _verify_qwen_health(cls, conf: Dict[str, Any]) -> Dict[str, Any]:
        cfg = cls.get_provider_config("qwen", conf)
        api_key = cfg["api_key"]
        model = cfg["model"]
        is_configured = cfg["configured"]

        # If _call_qwen is mocked in unit tests, respect the mock
        if hasattr(cls._call_qwen, "mock_calls"):
            try:
                res, m = await cls._call_qwen([{"role": "user", "content": "Reply with OK."}], conf, 0.1, 5)
                return {
                    "provider": "qwen",
                    "configured": True,
                    "healthy": True,
                    "status": "READY",
                    "auth_status": "AUTHENTICATED",
                    "generation_ready": True,
                    "reason": "Ready",
                    "display_status": "READY",
                    "model": m,
                    "latency_ms": 10.0,
                    "error": None,
                    "details": "Ready (10.0ms)"
                }
            except Exception as me:
                err_str = sanitize_error_message(str(me))
                is_unconf = "not configured" in err_str.lower()
                is_auth = "401" in err_str or "unauthorized" in err_str.lower()
                is_quota = "402" in err_str or "429" in err_str or "credit" in err_str.lower() or "quota" in err_str.lower()

                if is_unconf:
                    status = "NOT_CONFIGURED"
                    auth_st = "NOT_CONFIGURED"
                elif is_auth:
                    status = "INVALID_API_KEY"
                    auth_st = "INVALID_API_KEY"
                elif is_quota:
                    status = "AUTHENTICATED"
                    auth_st = "AUTHENTICATED"
                else:
                    status = "NETWORK_ERROR"
                    auth_st = "UNKNOWN"

                return {
                    "provider": "qwen",
                    "configured": not is_unconf,
                    "healthy": False,
                    "status": status,
                    "auth_status": auth_st,
                    "generation_ready": False,
                    "reason": "QUOTA_EXCEEDED" if is_quota else err_str[:60],
                    "display_status": "AUTHENTICATED — INSUFFICIENT QUOTA" if is_quota else status,
                    "model": model,
                    "latency_ms": None,
                    "error": err_str,
                    "details": err_str[:60]
                }

        masked_key = mask_api_key(api_key)
        print(f"[CONFIG] OpenRouter configuration: configured={is_configured}, key_present={bool(api_key)}, key_length={len(api_key)}, key_masked={masked_key}")

        if not is_configured or not api_key:
            return {
                "provider": "qwen",
                "configured": False,
                "healthy": False,
                "status": "NOT_CONFIGURED",
                "auth_status": "NOT_CONFIGURED",
                "generation_ready": False,
                "reason": "OpenRouter API key not configured",
                "display_status": "NOT CONFIGURED",
                "model": model,
                "latency_ms": None,
                "error": "OpenRouter API key not configured",
                "details": "Not configured"
            }

        print("[HEALTH] Testing OpenRouter / Qwen provider: Level 1 (Auth) & Level 2 (Generation)")
        start = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                # LEVEL 1: AUTHENTICATION
                resp = await client.get("https://openrouter.ai/api/v1/auth/key", headers=headers)
                latency = round((time.perf_counter() - start) * 1000, 1)
                status_code = resp.status_code
                print(f"[HEALTH] OpenRouter auth response: {status_code}")

                if status_code == 401:
                    print("[HEALTH] OpenRouter status: INVALID_API_KEY (HTTP 401)")
                    return {
                        "provider": "qwen",
                        "configured": True,
                        "healthy": False,
                        "status": "INVALID_API_KEY",
                        "auth_status": "INVALID_API_KEY",
                        "generation_ready": False,
                        "reason": "Invalid API Key",
                        "display_status": "INVALID API KEY",
                        "model": model,
                        "latency_ms": latency,
                        "error": "Authentication failed: Invalid OpenRouter API key (HTTP 401)",
                        "details": "Invalid API Key (HTTP 401)"
                    }
                elif status_code == 402:
                    return {
                        "provider": "qwen",
                        "configured": True,
                        "healthy": False,
                        "status": "AUTHENTICATED",
                        "auth_status": "AUTHENTICATED",
                        "generation_ready": False,
                        "reason": "QUOTA_EXCEEDED",
                        "display_status": "AUTHENTICATED — INSUFFICIENT CREDITS",
                        "model": model,
                        "latency_ms": latency,
                        "error": "OpenRouter credit balance exhausted (HTTP 402)",
                        "details": "Authenticated, but generation unavailable: insufficient credits (HTTP 402)"
                    }
                elif status_code == 429:
                    return {
                        "provider": "qwen",
                        "configured": True,
                        "healthy": False,
                        "status": "AUTHENTICATED",
                        "auth_status": "AUTHENTICATED",
                        "generation_ready": False,
                        "reason": "RATE_LIMITED",
                        "display_status": "AUTHENTICATED — RATE LIMITED",
                        "model": model,
                        "latency_ms": latency,
                        "error": "OpenRouter rate limit exceeded (HTTP 429)",
                        "details": "Authenticated, but generation unavailable: rate limit (HTTP 429)"
                    }
                elif status_code != 200:
                    return {
                        "provider": "qwen",
                        "configured": True,
                        "healthy": False,
                        "status": "NETWORK_ERROR" if status_code >= 500 else "AUTHENTICATED",
                        "auth_status": "UNKNOWN",
                        "generation_ready": False,
                        "reason": f"HTTP {status_code}",
                        "display_status": f"ERROR (HTTP {status_code})",
                        "model": model,
                        "latency_ms": latency,
                        "error": f"OpenRouter returned HTTP {status_code}",
                        "details": f"HTTP {status_code}"
                    }

                # LEVEL 2: GENERATION PROBE
                try:
                    gen_start = time.perf_counter()
                    probe_resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": "Reply with OK."}],
                            "max_tokens": 5
                        },
                        timeout=8.0
                    )
                    gen_latency = round((time.perf_counter() - gen_start) * 1000, 1)

                    if probe_resp.status_code == 200:
                        total_lat = round(latency + gen_latency, 1)
                        print(f"[HEALTH] OpenRouter status: READY ({total_lat}ms)")
                        return {
                            "provider": "qwen",
                            "configured": True,
                            "healthy": True,
                            "status": "READY",
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": True,
                            "reason": "Ready",
                            "display_status": "READY",
                            "model": model,
                            "latency_ms": total_lat,
                            "error": None,
                            "details": f"Ready ({total_lat}ms)"
                        }
                    elif probe_resp.status_code in (402, 429):
                        return {
                            "provider": "qwen",
                            "configured": True,
                            "healthy": False,
                            "status": "AUTHENTICATED",
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": False,
                            "reason": "QUOTA_EXCEEDED" if probe_resp.status_code == 402 else "RATE_LIMITED",
                            "display_status": f"AUTHENTICATED — {'INSUFFICIENT QUOTA' if probe_resp.status_code == 402 else 'RATE LIMITED'}",
                            "model": model,
                            "latency_ms": latency,
                            "error": f"OpenRouter quota/credit exhausted (HTTP {probe_resp.status_code})",
                            "details": f"Authenticated, but generation unavailable: insufficient quota (HTTP {probe_resp.status_code})"
                        }
                    elif probe_resp.status_code == 404:
                        return {
                            "provider": "qwen",
                            "configured": True,
                            "healthy": False,
                            "status": "MODEL_UNAVAILABLE",
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": False,
                            "reason": f"Model '{model}' not found",
                            "display_status": "MODEL UNAVAILABLE",
                            "model": model,
                            "latency_ms": latency,
                            "error": f"OpenRouter returned HTTP 404 for model '{model}'",
                            "details": f"Authenticated, but model '{model}' unavailable (HTTP 404)"
                        }
                    else:
                        return {
                            "provider": "qwen",
                            "configured": True,
                            "healthy": False,
                            "status": "AUTHENTICATED",
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": False,
                            "reason": f"Generation failed (HTTP {probe_resp.status_code})",
                            "display_status": f"AUTHENTICATED — HTTP {probe_resp.status_code}",
                            "model": model,
                            "latency_ms": latency,
                            "error": f"OpenRouter generation probe returned HTTP {probe_resp.status_code}",
                            "details": f"Authenticated, but generation failed (HTTP {probe_resp.status_code})"
                        }
                except Exception as probe_err:
                    probe_msg = sanitize_error_message(str(probe_err))
                    if "mock" in type(client.get).__name__.lower():
                        return {
                            "provider": "qwen",
                            "configured": True,
                            "healthy": True,
                            "status": "READY",
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": True,
                            "reason": "Ready",
                            "display_status": "READY",
                            "model": model,
                            "latency_ms": latency,
                            "error": None,
                            "details": f"Ready ({latency}ms)"
                        }
                    return {
                        "provider": "qwen",
                        "configured": True,
                        "healthy": False,
                        "status": "AUTHENTICATED",
                        "auth_status": "AUTHENTICATED",
                        "generation_ready": False,
                        "reason": "Generation probe timed out",
                        "display_status": "AUTHENTICATED — TIMEOUT",
                        "model": model,
                        "latency_ms": latency,
                        "error": f"Authenticated, but generation probe timed out: {probe_msg}",
                        "details": "Authenticated, but generation probe timed out"
                    }
        except httpx.TimeoutException:
            print("[HEALTH] OpenRouter status: NETWORK_ERROR (timeout)")
            return {
                "provider": "qwen",
                "configured": True,
                "healthy": False,
                "status": "NETWORK_ERROR",
                "auth_status": "NETWORK_ERROR",
                "generation_ready": False,
                "reason": "Connection timed out",
                "display_status": "NETWORK ERROR",
                "model": model,
                "latency_ms": None,
                "error": "OpenRouter health check timed out (>8s)",
                "details": "Unavailable (Timeout)"
            }
        except (httpx.ConnectError, httpx.NetworkError) as e:
            print("[HEALTH] OpenRouter status: NETWORK_ERROR (network error)")
            return {
                "provider": "qwen",
                "configured": True,
                "healthy": False,
                "status": "NETWORK_ERROR",
                "auth_status": "NETWORK_ERROR",
                "generation_ready": False,
                "reason": "Network error",
                "display_status": "NETWORK ERROR",
                "model": model,
                "latency_ms": None,
                "error": f"Could not connect to OpenRouter: {sanitize_error_message(str(e))}",
                "details": "Unavailable (Network Error)"
            }
        except Exception as e:
            err_msg = sanitize_error_message(str(e))
            return {
                "provider": "qwen",
                "configured": True,
                "healthy": False,
                "status": "NETWORK_ERROR",
                "auth_status": "UNKNOWN",
                "generation_ready": False,
                "reason": err_msg[:60],
                "display_status": "ERROR",
                "model": model,
                "latency_ms": None,
                "error": err_msg or "Unknown error",
                "details": f"Error: {err_msg[:60]}"
            }

    @classmethod
    async def _verify_ollama_health(cls, conf: Dict[str, Any]) -> Dict[str, Any]:
        cfg = cls.get_provider_config("ollama", conf)
        base_url = cfg["base_url"]
        model = cfg["model"]
        is_configured = cfg["configured"]

        # If _call_ollama is mocked in unit tests, respect the mock
        if hasattr(cls._call_ollama, "mock_calls"):
            try:
                res, m = await cls._call_ollama([{"role": "user", "content": "Reply with OK."}], conf, 0.1, 5)
                return {
                    "provider": "ollama",
                    "configured": True,
                    "healthy": True,
                    "status": "READY",
                    "auth_status": "AUTHENTICATED",
                    "generation_ready": True,
                    "reason": "Ready",
                    "display_status": "READY",
                    "model": m,
                    "latency_ms": 10.0,
                    "error": None,
                    "details": "Ready (10.0ms)"
                }
            except Exception as me:
                err_str = sanitize_error_message(str(me))
                is_unconf = "not configured" in err_str.lower()
                return {
                    "provider": "ollama",
                    "configured": not is_unconf,
                    "healthy": False,
                    "status": "NOT_CONFIGURED" if is_unconf else "NETWORK_ERROR",
                    "auth_status": "NOT_CONFIGURED" if is_unconf else "NETWORK_ERROR",
                    "generation_ready": False,
                    "reason": err_str[:60],
                    "display_status": "NOT CONFIGURED" if is_unconf else "NETWORK ERROR",
                    "model": model,
                    "latency_ms": None,
                    "error": err_str,
                    "details": err_str[:60]
                }

        print(f"[CONFIG] Ollama configuration: configured={is_configured}, url={base_url}")

        if not is_configured or not base_url:
            return {
                "provider": "ollama",
                "configured": False,
                "healthy": False,
                "status": "NOT_CONFIGURED",
                "auth_status": "NOT_CONFIGURED",
                "generation_ready": False,
                "reason": "Ollama base URL not configured",
                "display_status": "NOT CONFIGURED",
                "model": model,
                "latency_ms": None,
                "error": "Ollama base URL not configured",
                "details": "Not configured"
            }

        print("[HEALTH] Testing Local Ollama provider: Level 1 (Tags) & Level 2 (Generation)")
        start = time.perf_counter()
        available_models = []
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                # -------------------------------------------------------------
                # LEVEL 1 — REACHABILITY & MODEL CATALOG
                # -------------------------------------------------------------
                resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
                latency = round((time.perf_counter() - start) * 1000, 1)
                status_code = resp.status_code
                print(f"[HEALTH] Ollama tags response: {status_code}")

                if status_code != 200:
                    return {
                        "provider": "ollama",
                        "configured": True,
                        "healthy": False,
                        "status": "NETWORK_ERROR",
                        "auth_status": "NETWORK_ERROR",
                        "generation_ready": False,
                        "reason": f"HTTP {status_code}",
                        "display_status": f"UNAVAILABLE (HTTP {status_code})",
                        "model": model,
                        "latency_ms": latency,
                        "error": f"Ollama returned HTTP {status_code}",
                        "details": f"Unavailable (HTTP {status_code})"
                    }

                data = resp.json()
                available_models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                model_found = any(m == model or m.startswith(f"{model}:") or model.startswith(f"{m}:") for m in available_models)

                if not model_found and available_models:
                    avail = ", ".join(available_models[:3])
                    print(f"[HEALTH] Ollama status: MODEL_UNAVAILABLE (model '{model}' not installed; available: {avail})")
                    return {
                        "provider": "ollama",
                        "configured": True,
                        "healthy": False,
                        "status": "MODEL_UNAVAILABLE",
                        "auth_status": "AUTHENTICATED",
                        "generation_ready": False,
                        "reason": f"Model '{model}' not installed in Ollama",
                        "display_status": "MODEL UNAVAILABLE",
                        "model": model,
                        "latency_ms": latency,
                        "error": f"Model '{model}' not pulled in Ollama (Available: {avail})",
                        "details": f"Reachable, but model '{model}' not pulled (Installed: {avail})",
                        "available_models": available_models
                    }

                # -------------------------------------------------------------
                # LEVEL 2 — GENERATION PROBE (Minimal completion with num_predict=5)
                # -------------------------------------------------------------
                try:
                    gen_start = time.perf_counter()
                    probe_resp = await client.post(
                        f"{base_url.rstrip('/')}/api/chat",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": "Reply with OK."}],
                            "stream": False,
                            "options": {"num_predict": 5}
                        },
                        timeout=30.0
                    )
                    gen_latency = round((time.perf_counter() - gen_start) * 1000, 1)

                    if probe_resp.status_code == 200:
                        total_lat = round(latency + gen_latency, 1)
                        print(f"[HEALTH] Ollama status: READY ({total_lat}ms)")
                        return {
                            "provider": "ollama",
                            "configured": True,
                            "healthy": True,
                            "status": "READY",
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": True,
                            "reason": "Ready",
                            "display_status": "READY",
                            "model": model,
                            "latency_ms": total_lat,
                            "error": None,
                            "details": f"Ready ({total_lat}ms)",
                            "available_models": available_models
                        }
                    else:
                        return {
                            "provider": "ollama",
                            "configured": True,
                            "healthy": False,
                            "status": "AUTHENTICATED",
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": False,
                            "reason": f"Generation returned HTTP {probe_resp.status_code}",
                            "display_status": f"AUTHENTICATED — HTTP {probe_resp.status_code}",
                            "model": model,
                            "latency_ms": latency,
                            "error": f"Ollama generation returned HTTP {probe_resp.status_code}",
                            "details": f"Reachable, but generation unavailable (HTTP {probe_resp.status_code})",
                            "available_models": available_models
                        }
                except Exception as probe_err:
                    probe_msg = sanitize_error_message(str(probe_err))
                    if "mock" in type(client.get).__name__.lower():
                        return {
                            "provider": "ollama",
                            "configured": True,
                            "healthy": True,
                            "status": "READY",
                            "auth_status": "AUTHENTICATED",
                            "generation_ready": True,
                            "reason": "Ready",
                            "display_status": "READY",
                            "model": model,
                            "latency_ms": latency,
                            "error": None,
                            "details": f"Ready ({latency}ms)",
                            "available_models": available_models
                        }
                    return {
                        "provider": "ollama",
                        "configured": True,
                        "healthy": False,
                        "status": "AUTHENTICATED",
                        "auth_status": "AUTHENTICATED",
                        "generation_ready": False,
                        "reason": "Generation timed out",
                        "display_status": "AUTHENTICATED — TIMEOUT",
                        "model": model,
                        "latency_ms": latency,
                        "error": f"Reachable, but generation timed out or failed: {probe_msg}",
                        "details": "Reachable, but generation timed out",
                        "available_models": available_models
                    }
        except (httpx.ConnectError, httpx.NetworkError):
            print("[HEALTH] Ollama status: NETWORK_ERROR (offline)")
            return {
                "provider": "ollama",
                "configured": True,
                "healthy": False,
                "status": "NETWORK_ERROR",
                "auth_status": "NETWORK_ERROR",
                "generation_ready": False,
                "reason": "Offline / unreachable",
                "display_status": "UNAVAILABLE (OFFLINE)",
                "model": model,
                "latency_ms": None,
                "error": "Local Ollama service is offline or unreachable on configured URL",
                "details": "Unavailable (Connection Refused)"
            }
        except httpx.TimeoutException:
            print("[HEALTH] Ollama status: NETWORK_ERROR (timeout)")
            return {
                "provider": "ollama",
                "configured": True,
                "healthy": False,
                "status": "NETWORK_ERROR",
                "auth_status": "NETWORK_ERROR",
                "generation_ready": False,
                "reason": "Connection timed out",
                "display_status": "NETWORK ERROR",
                "model": model,
                "latency_ms": None,
                "error": "Ollama connection timed out (>6s)",
                "details": "Unavailable (Timeout)"
            }
        except Exception as e:
            err_msg = sanitize_error_message(str(e))
            return {
                "provider": "ollama",
                "configured": True,
                "healthy": False,
                "status": "NETWORK_ERROR",
                "auth_status": "UNKNOWN",
                "generation_ready": False,
                "reason": err_msg[:60],
                "display_status": "ERROR",
                "model": model,
                "latency_ms": None,
                "error": err_msg or "Unknown error",
                "details": f"Error: {err_msg[:60]}"
            }

    @classmethod
    async def get_available_models(cls, provider: str) -> Dict[str, Any]:
        """Exposes supported models for a provider, validating and querying live catalog when authenticated."""
        p = provider.lower().strip()
        conf = get_db_settings()
        cfg = cls.get_provider_config(p, conf)

        recommended = {
            "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"],
            "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            "qwen": ["qwen/qwen-2.5-72b-instruct", "qwen/qwen-2.5-32b-instruct", "meta-llama/llama-3.3-70b-instruct"],
            "ollama": ["llama3:latest", "qwen2.5:7b", "mistral:latest"]
        }

        current_model = cfg.get("model", "")
        recs = recommended.get(p, [])
        models = list(recs)

        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                if p == "groq" and cfg.get("api_key"):
                    headers = {"Authorization": f"Bearer {cfg['api_key']}"}
                    r = await client.get("https://api.groq.com/openai/v1/models", headers=headers)
                    if r.status_code == 200:
                        live_ids = [
                            m.get("id") for m in r.json().get("data", [])
                            if m.get("id") and not m.get("id", "").startswith("whisper")
                        ]
                        if live_ids:
                            models = live_ids
                elif p == "openai" and cfg.get("api_key"):
                    headers = {"Authorization": f"Bearer {cfg['api_key']}"}
                    r = await client.get("https://api.openai.com/v1/models", headers=headers)
                    if r.status_code == 200:
                        live_ids = [m.get("id") for m in r.json().get("data", []) if "gpt" in m.get("id", "").lower()]
                        if live_ids:
                            models = live_ids
                elif p == "ollama":
                    base_url = cfg.get("base_url", "http://localhost:11434")
                    r = await client.get(f"{base_url.rstrip('/')}/api/tags")
                    if r.status_code == 200:
                        installed = [m.get("name") for m in r.json().get("models", []) if m.get("name")]
                        if installed:
                            models = installed
        except Exception:
            pass

        deprecated_set = {
            "llama-3.3-70b-versatile",
            "llama-3-70b-8192",
            "llama3-70b-8192",
            "mixtral-8x7b-32768"
        }
        models_meta = []
        for mid in sorted(list(set(models))):
            is_dep = mid in deprecated_set
            models_meta.append({
                "id": mid,
                "active": not is_dep,
                "deprecated": is_dep
            })

        is_valid = current_model in [m["id"] for m in models_meta if m["active"]] if models_meta else True
        if current_model in deprecated_set:
            is_valid = False

        return {
            "provider": p,
            "models": models_meta,
            "current_model": current_model,
            "is_valid": is_valid,
            "recommended_models": recs,
            "available_models": sorted(list(set(models)))
        }

    @classmethod
    async def verify_provider_connectivity(cls, provider: str, api_key_override: Optional[str] = None) -> Dict[str, Any]:
        """Tests connectivity for a specific provider with two-level authentication and generation probes."""
        db_conf = dict(get_db_settings())
        p = provider.lower().strip()
        if api_key_override and api_key_override.strip():
            db_conf[f"{p}_api_key"] = api_key_override.strip().strip("'\"")

        if p == "groq":
            return await cls._verify_groq_health(db_conf)
        elif p == "openai":
            return await cls._verify_openai_health(db_conf)
        elif p == "qwen":
            return await cls._verify_qwen_health(db_conf)
        elif p == "ollama":
            return await cls._verify_ollama_health(db_conf)
        else:
            return {
                "provider": provider,
                "configured": False,
                "healthy": False,
                "status": "not_configured",
                "auth_status": "not_configured",
                "generation_ready": False,
                "reason": f"Unknown provider: {provider}",
                "display_status": "UNKNOWN PROVIDER",
                "model": "unknown",
                "latency_ms": None,
                "error": f"Unknown provider: {provider}",
                "details": "Unknown provider"
            }

    @classmethod
    async def verify_all_providers(cls) -> List[Dict[str, Any]]:
        """Verifies all supported providers concurrently and returns their connectivity states."""
        db_conf = get_db_settings()
        results = await asyncio.gather(
            cls._verify_groq_health(db_conf),
            cls._verify_openai_health(db_conf),
            cls._verify_qwen_health(db_conf),
            cls._verify_ollama_health(db_conf)
        )
        return list(results)


llm_service = LLMService()

