"""
Mizo Agent Router Service
=========================
Dedicated router responsible strictly for deciding which specialized agent workspace
the user needs based on current intent:
- NORMAL_CHAT: General conversation, English communication practice, grammar corrections, casual dialogue.
- STUDY: Material-based teaching, syllabus reading, chapter study, exam prep from notes.
- SPEECH: Presentation rehearsal, seminar practice, speech timing & analysis.
- AMBIGUOUS: When intent is genuinely ambiguous (e.g. "I want to practice"), prompts the user with clarifying options.

CRITICAL RULES:
1. Current intent has the HIGHEST priority. Never allow stale context to improperly override current intent.
2. Deterministic, ultra-fast pattern detection is preferred for obvious intents (zero LLM overhead).
3. Router does NOT perform actual teaching, conversation, or speech analysis.
"""

import re
from typing import Dict, Any, Optional, Tuple
from enum import Enum


class AgentRoute(str, Enum):
    NORMAL_CHAT = "NORMAL_CHAT"
    STUDY = "STUDY"
    SPEECH = "SPEECH"
    AMBIGUOUS = "AMBIGUOUS"


class AgentRouter:
    """
    Automated Intent Router for Mizo 3.0 Multi-Agent Architecture.
    """

    AMBIGUOUS_CLARIFICATION_PROMPT = (
        "Sure. Do you want to practice a conversation, study from your notes, or practice a speech/presentation?"
    )

    # 1. Genuinely ambiguous patterns (when user only asks to "practice" without specifying what)
    AMBIGUOUS_PATTERNS = [
        r"^\s*(?:i want to |can we |let'?s |help me )?practice\.?\s*$",
        r"^\s*(?:i need to |i would like to )?practice(?:\s+something)?\.?\s*$",
        r"^\s*let'?s do some practice\.?\s*$"
    ]

    # 2. Speech & Seminar Practice Patterns
    SPEECH_PATTERNS = [
        r"\bpractice (?:my |a |the )?(?:speech|seminar|presentation|pitch|talk|keynote)\b",
        r"\b(?:speech|seminar|presentation|public speaking)\s+practice\b",
        r"\b(?:rehearse|rehearsal)\b",
        r"\blisten to my speech\b",
        r"\bseminar tomorrow\b",
        r"\bpresentation tomorrow\b",
        r"\bspeech coach\b",
        r"\bted[- ]?talk\b",
        r"\bprepare (?:for )?(?:my |a )?(?:seminar|presentation|speech)\b",
        r"\bgive (?:a |my )?(?:speech|presentation|seminar|talk)\b",
        r"\bpresentation rehearsal\b",
        r"\bcheck my speech\b",
        r"\bevaluate my speech\b",
        r"\bspeech feedback\b",
        r"\bfeedback on (?:my |the )?(?:speech|seminar|presentation|talk)\b",
        r"\b(?:seminar|presentation) feedback\b",
        r"\bseminar\b",
        r"\bpractice speaking for a presentation\b"
    ]

    # 3. Study & Material-Based Teaching Patterns
    STUDY_PATTERNS = [
        r"\b(?:my |the |this )?(?:syllabus|notes|pdf|textbook|study material|chapter|course material)\b",
        r"\bstudy for (?:my |the )?(?:exam|test|quiz|midterm|finals)\b",
        r"\bexam prep(?:aration)?\b",
        r"\bteach me (?:from |about )?(?:my |this |the )?(?:notes|pdf|syllabus|chapter|material)\b",
        r"\bteach me\b",
        r"\bi (?:have|uploaded|have uploaded) (?:my )?(?:notes|syllabus|pdf|material)\b",
        r"\bi want to study\b",
        r"\bwant to study for\b",
        r"\blearn from (?:my |these )?(?:notes|pdf|syllabus)\b",
        r"\bprepare for (?:my |the )?exam from (?:these |my )?notes\b",
        r"\bteach me (?:forward kinematics|inverse kinematics|dh parameters|robotics|dynamics|kinematics)\b",
        r"\bteach me (?:chapter|unit|topic)\b",
        r"\bexplain (?:forward kinematics|inverse kinematics|dh parameters)\b",
        r"\bexplain (?:it )?(?:very )?simply\b",
        r"\bexplain simply\b",
        r"\bgive (?:me )?(?:an? )?example\b",
        r"\bquiz me\b",
        r"\brevise\b",
        r"\btest me\b",
        r"\bnext topic\b",
        r"\bmove to (?:the )?next topic\b",
        r"\bselect\b",
        r"\bchoose topic\b",
        r"\bchoose subject\b",
        r"\bstudy section\b",
        r"\bacademic tutor\b",
        r"\bsubject tutor\b"
    ]

    # 4. Explicit Normal Chat Patterns
    NORMAL_CHAT_PATTERNS = [
        r"\b(?:talk|discuss|chat) (?:about )?(?:anime|one piece|naruto|movies|series|my day|something else|weather|life)\b",
        r"\b(?:let'?s|can we) (?:talk|chat|discuss)\b",
        r"\bimprove my english\b",
        r"\bcasual chat\b",
        r"\bconversation practice\b",
        r"\bpractice (?:a |my |english )?conversation\b",
        r"\benglish communication\b",
        r"\bcommunication coach\b",
        r"\bhow are you\b",
        r"\bwhat is your name\b",
        r"\bwhat'?s your name\b",
        r"\bwho are you\b",
        r"\bhello\b",
        r"\bhi mizo\b",
        r"\bhey mizo\b"
    ]

    @classmethod
    def route_intent(
        cls,
        text: str,
        current_route: Optional[str] = None,
        session_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Determines the destination agent for the user's message.
        Always prioritizes CURRENT user intent over any previous session topic.
        """
        if not text or not text.strip():
            return {
                "route": AgentRoute.NORMAL_CHAT.value,
                "confidence": 1.0,
                "is_ambiguous": False,
                "clarification_prompt": None,
                "reason": "Empty input defaulted to normal chat"
            }

        norm = text.lower().strip()
        # Clean punctuation for exact word matching
        clean = re.sub(r"[?!.,'\"]", " ", norm).strip()
        clean = re.sub(r"\s+", " ", clean)

        # Step 1: Check Genuine Ambiguity
        # If user only said "I want to practice" without specifying conversation, study, or speech:
        for p in cls.AMBIGUOUS_PATTERNS:
            if re.search(p, norm, re.IGNORECASE) or norm in ["practice", "i want to practice", "let's practice", "can we practice"]:
                return {
                    "route": AgentRoute.AMBIGUOUS.value,
                    "confidence": 0.5,
                    "is_ambiguous": True,
                    "clarification_prompt": cls.AMBIGUOUS_CLARIFICATION_PROMPT,
                    "reason": "Genuinely ambiguous practice request"
                }

        # Step 2: Speech & Seminar Practice (High Priority deterministic match)
        for pattern in cls.SPEECH_PATTERNS:
            if re.search(pattern, norm, re.IGNORECASE):
                return {
                    "route": AgentRoute.SPEECH.value,
                    "confidence": 0.98,
                    "is_ambiguous": False,
                    "clarification_prompt": None,
                    "reason": f"Matched speech pattern: {pattern}"
                }

        # Step 3: Study / Material-Based Teaching (High Priority deterministic match)
        for pattern in cls.STUDY_PATTERNS:
            if re.search(pattern, norm, re.IGNORECASE):
                return {
                    "route": AgentRoute.STUDY.value,
                    "confidence": 0.98,
                    "is_ambiguous": False,
                    "clarification_prompt": None,
                    "reason": f"Matched study pattern: {pattern}"
                }

        # Step 4: Explicit Normal Chat
        for pattern in cls.NORMAL_CHAT_PATTERNS:
            if re.search(pattern, norm, re.IGNORECASE):
                return {
                    "route": AgentRoute.NORMAL_CHAT.value,
                    "confidence": 0.95,
                    "is_ambiguous": False,
                    "clarification_prompt": None,
                    "reason": f"Matched normal chat pattern: {pattern}"
                }

        # Step 5: Default Fallback -> NORMAL_CHAT
        # General queries, questions, or greetings without study/speech triggers route to Normal Chat.
        return {
            "route": AgentRoute.NORMAL_CHAT.value,
            "confidence": 0.85,
            "is_ambiguous": False,
            "clarification_prompt": None,
            "reason": "Default conversational fallback to normal chat"
        }


agent_router = AgentRouter()
