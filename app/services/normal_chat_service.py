"""
Mizo Normal Chat Service
========================
Dedicated agent for General Conversation & English Communication Coaching.

Purpose:
- General conversation
- English communication practice
- Grammar correction when useful
- General questions
- Natural conversation

Strict Isolation Boundaries:
- Must NOT automatically become a tutor simply because the user mentions studying.
- Must NOT load syllabus material.
- Must NOT run speech analysis.
- Keeps working memory isolated from Study and Speech agents.
- Old study topics (e.g., Forward Kinematics) do NOT appear in Normal Chat responses
  unless explicitly requested by the user.
"""

import re
from typing import Dict, Any, List, Optional, Tuple

from app.db.database import (
    get_student,
    update_student,
    PROTOTYPE_STUDENT_ID,
    get_session_by_id,
    update_session_context
)
from app.services.context_service import update_conversation_topic_state
from app.services.personalization_service import personalization_service
from app.services.memory_service import memory_manager
from app.services.llm_service import llm_service, LLMResult, LLMServiceError


class NormalChatService:
    """
    Dedicated Normal Communication & English Coaching Agent.
    """

    @classmethod
    def get_isolated_topic_state(cls, session_id: str, student_id: int, user_text: str) -> Dict[str, Any]:
        """
        Updates and retrieves normal conversation topic state without contaminating from tutor or speech sessions.
        """
        return update_conversation_topic_state(
            session_id=session_id,
            student_id=student_id,
            user_text=user_text
        )

    @classmethod
    def clear_working_context(cls, student_id: int, session_id: Optional[str] = None):
        """Clears working context for the learner."""
        from app.db.database import get_active_session
        sess = get_session_by_id(session_id) if session_id else get_active_session(student_id)
        if sess:
            update_session_context(sess["id"], {"current_topic": None, "subtopic": None, "recent_user_turns": []})

    @classmethod
    def get_working_context(cls, student_id: int, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Returns the current working context for normal chat."""
        from app.db.database import get_active_session, get_session_context
        sess = get_session_by_id(session_id) if session_id else get_active_session(student_id)
        if not sess:
            return {"recent_user_turns": [], "current_topic": None}
        ctx = get_session_context(sess["id"])
        return {
            "recent_user_turns": ctx.get("recent_user_turns", []),
            "current_topic": ctx.get("current_topic")
        }

    @classmethod
    async def process_chat_turn(
        cls,
        student_id: int,
        session_id: str,
        user_text: str,
        is_typed_text: bool = False
    ) -> Dict[str, Any]:
        """
        Processes a conversation turn strictly within the Normal Chat workspace.
        """
        student = get_student(student_id) or {}
        st_name = student.get("name") or "there"

        # 1. Update isolated topic state
        conv_context = cls.get_isolated_topic_state(session_id, student_id, user_text)

        # 2. Linguistic evaluation (grammar corrections, vocabulary suggestions)
        g_errs, v_suggs, flu, pac, conf = personalization_service.evaluate_utterance(
            text=user_text,
            student_id=student_id,
            session_id=session_id
        )

        # 3. Build LLM Context strictly with mode="coach" (no syllabus or RAG injection)
        context = memory_manager.build_llm_context(
            student_id=student_id,
            session_id=session_id,
            current_message=user_text,
            mode="coach",
            rag_context="",  # strictly empty in Normal Chat
            max_recent_messages=8
        )

        # Ensure no accidental syllabus leak in prompt
        system_prompt = context.get("system_prompt", "")
        # Add explicit guard ensuring general communication persona
        chat_guard = (
            f"\nYou are Mizo, a friendly English communication and conversational coach for {st_name}. "
            "Engage naturally on the user's chosen topic. Do not teach syllabus or lecture unless requested. "
            "Keep your tone warm, concise, and conversational."
        )
        system_prompt += chat_guard

        reply_text = ""
        provider_used = "ollama"
        is_fallback = False
        latency_ms = 0.0

        try:
            llm_res = await llm_service.generate_response(
                messages=context["messages"],
                system_prompt=system_prompt
            )
            reply_text = llm_res.text
            provider_used = f"{llm_res.provider} ({llm_res.model})"
            is_fallback = llm_res.is_fallback
            latency_ms = llm_res.latency_ms
        except LLMServiceError as e:
            reply_text = e.user_message
            provider_used = f"error ({e.category})"
        except Exception as e:
            # Fallback natural responses
            norm = user_text.lower()
            if "one piece" in norm or "anime" in norm:
                reply_text = "I love talking about anime! Who is your favorite character in One Piece?"
            elif "how are you" in norm:
                reply_text = f"I'm doing well, {st_name}! How has your day been going?"
            else:
                reply_text = f"That's interesting, {st_name}! Tell me more about that."
            provider_used = "system"

        # Record metrics for conversation practice
        personalization_service.record_interaction_metrics(
            student_id=student_id,
            session_id=session_id,
            grammar_errors=g_errs,
            vocab_suggestions=v_suggs,
            fluency_score=flu,
            pacing_score=pac,
            confidence_score=conf
        )

        return {
            "reply_text": reply_text,
            "response": reply_text,
            "provider_used": provider_used,
            "is_fallback": is_fallback,
            "latency_ms": latency_ms,
            "grammar_errors": g_errs,
            "vocab_suggestions": v_suggs,
            "fluency_score": flu,
            "pacing_score": pac,
            "confidence_score": conf,
            "current_topic": conv_context.get("current_topic") or ""
        }


normal_chat_service = NormalChatService()
