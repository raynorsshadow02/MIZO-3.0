import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from app.db.database import (
    get_student,
    update_student,
    get_conversation_history,
    get_recent_mistakes,
    get_db_settings,
    get_active_session
)

logger = logging.getLogger("mizo.memory")

# Blacklist of words that must NEVER be extracted or accepted as a student's name
NAME_EXCLUDE_WORDS = {
    # Pronouns & Articles
    "i", "im", "i'm", "me", "my", "myself", "you", "your", "yourself",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "a", "an", "the", "this", "that", "these", "those",
    # Verbs, Gerunds & Action Participles
    "studying", "learning", "working", "trying", "hoping", "practicing", "doing",
    "going", "goes", "go", "gone", "went", "feeling", "struggling", "interested", "looking", "reading", "writing",
    "speaking", "talking", "thinking", "playing", "living", "staying", "watching",
    "hearing", "listening", "preparing", "improving", "aiming", "starting", "asking",
    "wanting", "needing", "liking", "loving", "helping", "teaching", "coaching",
    "am", "is", "are", "was", "were", "be", "being", "been",
    "have", "has", "had", "do", "does", "did", "can", "could", "will", "would",
    "shall", "should", "may", "might", "must",
    # Stop & Control Words
    "stop", "stops", "stopping", "stopped", "snooze", "snoozed", "sleep", "wake", "waking", "awake",
    "pause", "halt", "quit", "exit", "cancel", "reset", "clear",
    # Assessment & Submission Words
    "done", "finish", "finished", "submit", "submitted", "complete", "completed",
    # Adjectives, Adverbs, Prepositions & Physical / Mental States
    "good", "bad", "tired", "confident", "from", "in", "at", "here", "there",
    "to", "too", "into", "onto", "upon", "about", "above", "across", "after",
    "fine", "happy", "sad", "not", "ready", "okay", "sure", "busy", "late",
    "early", "sick", "well", "exhausted", "bored", "excited", "nervous", "anxious",
    "proud", "curious", "very", "so", "really", "quite", "also", "just",
    "always", "never", "sometimes", "often", "now", "today", "tomorrow", "yesterday",
    "beginner", "intermediate", "advanced", "elementary", "fluent", "native",
    # Deflections, Negations & Question Words
    "what", "why", "how", "who", "whom", "whose", "when", "where", "which",
    "nothing", "anything", "everything", "something", "none", "no", "yes", "yeah", "nope",
    "hello", "hi", "hey", "pass", "skip", "unclear", "unknown", "text", "user", "name",
    "please", "thanks", "thank",
    # Conjunctions & Connectors
    "and", "or", "but", "so", "because", "as", "if", "though", "although", "than", "then",
    # Roles, Generic Subjects & Technical Artifacts
    "student", "learner", "engineer", "developer", "teacher", "doctor", "scientist",
    "analyst", "manager", "robot", "robotics", "gravity", "machine", "system", "model",
    "ai", "mizo", "mikaza", "assistant", "coach", "tutor", "device", "data", "record",
    "records", "update", "change", "profile", "lesson", "topic",
    "english", "communication", "grammar", "vocabulary", "fluency", "accent"
}


class MizoMemoryManager:
    """
    Centralized Memory & Context Architecture for Mizo:
    1. Long-Term User Memory: Authoritative database student profile (Name, Goals, Level, Strengths, Weaknesses).
    2. Short-Term Conversation Memory: Session-isolated recent turns with controlled sliding-window truncation.
    3. Persistent Session Summaries: Compact consolidation of older conversation turns beyond the active window.
    4. Protected Identity Guardrails: Prevents casual statements ('I am studying...') from corrupting user identity.
    5. Central Context Builder: Unified build_llm_context() used by all endpoints and future agents.
    """

    @classmethod
    def is_valid_name(cls, candidate: str) -> bool:
        """
        Validates that a string represents a genuine human name.
        - Must contain only letters, hyphens, or spaces.
        - Length between 2 and 35 characters.
        - None of its component words can be in the exclusion blacklist.
        """
        if not candidate or not isinstance(candidate, str):
            return False

        clean = candidate.strip()
        # Strip leading and trailing punctuation (e.g. commas, periods, exclamation points)
        clean = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", clean).strip()
        if len(clean) < 2 or len(clean) > 35:
            return False

        # Must consist only of alphabetic characters, hyphens, and spaces
        if not re.match(r"^[A-Za-z]+(?:[\s\-][A-Za-z]+)*$", clean):
            return False

        # Check each component word against the exclusion set
        words = [w.lower() for w in re.split(r"[\s\-]+", clean) if w]
        if not words:
            return False

        for w in words:
            if w in NAME_EXCLUDE_WORDS:
                return False

        return True

    @classmethod
    def extract_name_safely(
        cls,
        text: str,
        current_student: Optional[Dict[str, Any]] = None,
        is_name_step: bool = False
    ) -> Optional[str]:
        """
        Safely extracts user's name from an utterance with strict protection against false positives.
        - If student already has a known name: ONLY explicit name-change phrases are recognized.
        - If student has no known name or is in the initial 'ask_name' step: permits introductions and standalone names.
        """
        if not text or not text.strip():
            return None

        clean = text.strip()

        # Check if student already has an established non-placeholder name
        current_name = None
        if current_student:
            raw_curr = (current_student.get("name") or "").strip()
            if raw_curr and raw_curr.lower() not in ["student", "new learner", "there", "unknown"]:
                if cls.is_valid_name(raw_curr):
                    current_name = raw_curr

        # Case 1: Student already has an established name
        # Identity is an anchored invariant: only an EXPLICIT name change is allowed
        if current_name and not is_name_step:
            explicit_change_patterns = [
                r"(?:my name is actually|actually,? my name is|my real name is)\s+([A-Za-z\-]+(?:\s+[A-Za-z\-]+)?)(?:[,\.]|$|\s+(?:a|an|and)\b)",
                r"(?:want you to|want to|please\s+)?(?:change|update|set|replace)\s+my name\s+(?:to|as)\s+([A-Za-z\-]+(?:\s+[A-Za-z\-]+)?)(?:[,\.]|$|\s+(?:a|an|and)\b)",
                r"(?:change my name to|update my name to)\s+([A-Za-z\-]+(?:\s+[A-Za-z\-]+)?)(?:[,\.]|$|\s+(?:a|an|and)\b)",
                r"(?:please call me|you can call me|call me)\s+([A-Za-z\-]+(?:\s+[A-Za-z\-]+)?)(?:[,\.]|$|\s+(?:a|an|and)\b)",
                r"^my name is\s+([A-Za-z\-]+(?:\s+[A-Za-z\-]+)?)(?:[,\.]|$|\s+(?:a|an|and)\b)",
                r"^(?:i am|i'm|im)\s+([A-Za-z\-]+(?:\s+[A-Za-z\-]+)?)(?:[,\.]|$|\s+(?:a|an|and)\b)"
            ]
            for pat in explicit_change_patterns:
                m = re.search(pat, clean, re.IGNORECASE)
                if m:
                    cand = m.group(1).strip()
                    cand = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", cand).strip()
                    if cls.is_valid_name(cand):
                        return " ".join(p.capitalize() for p in cand.split())
            # Casual conversation (e.g. "I am studying robotics" or "I am going to work") must NEVER change an established name
            return None

        # Case 2: Student has no established name yet (or is currently in onboarding ask_name step)
        introduction_patterns = [
            r"^(?:i am|i'm|im)\s+([A-Za-z\-]+(?:\s+[A-Za-z\-]+)?)(?:[,\.]|$|\s+(?:a|an|and|who|from|studying|working)\b)",
            r"(?:my name is|i am called|this is|you can call me)\s+([A-Za-z\-]+(?:\s+[A-Za-z\-]+)?)(?:[,\.]|$|\s+(?:a|an|and|who|from|studying|working)\b)",
            r"(?:call me)\s+([A-Za-z\-]+(?:\s+[A-Za-z\-]+)?)(?:[,\.]|$|\s+(?:a|an|and|who|from|studying|working)\b)"
        ]
        for pat in introduction_patterns:
            m = re.search(pat, clean, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                cand = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", cand).strip()
                if cls.is_valid_name(cand):
                    return " ".join(p.capitalize() for p in cand.split())

        # If in initial ask_name step OR student has no established name yet: check standalone name
        if is_name_step or not current_name:
            # Check standalone 1-2 words (e.g. "Shahid" or "Mohamed Shahid")
            standalone_match = re.match(r"^([A-Za-z\-]+(?:\s+[A-Za-z\-]+)?)[.!?]?$", clean)
            if standalone_match:
                cand = standalone_match.group(1).strip()
                cand = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", cand).strip()
                if cls.is_valid_name(cand):
                    return " ".join(p.capitalize() for p in cand.split())

        return None

    @classmethod
    def get_student_profile(cls, student_id: int) -> Dict[str, Any]:
        """
        Retrieves the authoritative student profile from SQLite.
        Guarantees well-formed fields for long-term memory injection.
        """
        student = get_student(student_id)
        if not student:
            return {
                "id": student_id,
                "name": "New Learner",
                "is_name_known": False,
                "education": "Student",
                "target_level": "Intermediate",
                "learning_goals": "Improve English communication",
                "weaknesses": ["Foundational grammar", "Speaking confidence"],
                "strengths": ["Active participation"],
                "interests": "General Topics",
                "onboarding_completed": False
            }

        name = (student.get("name") or "").strip()
        is_name_known = bool(name and name.lower() not in ["student", "new learner", "there", "unknown", ""])

        return {
            "id": student.get("id", student_id),
            "name": name if is_name_known else "New Learner",
            "is_name_known": is_name_known,
            "education": student.get("education") or "Student",
            "grade": student.get("grade") or "Grade 8",
            "target_level": student.get("target_level") or "Intermediate",
            "self_reported_level": student.get("self_reported_level") or "Not sure",
            "learning_goals": student.get("learning_goals") or "Improve speaking fluency and communication",
            "learning_topics": student.get("learning_topics") or "",
            "interests": student.get("interests") or "General Topics",
            "weaknesses": list(student.get("weaknesses") or ["Grammar consistency", "Speaking confidence"]),
            "strengths": list(student.get("strengths") or ["Active participation"]),
            "baseline_grammar": student.get("baseline_grammar"),
            "baseline_vocabulary": student.get("baseline_vocabulary"),
            "baseline_fluency": student.get("baseline_fluency"),
            "baseline_confidence": student.get("baseline_confidence"),
            "baseline_pronunciation": student.get("baseline_pronunciation"),
            "grammar_score": student.get("grammar_score"),
            "vocabulary_score": student.get("vocabulary_score"),
            "fluency_score": student.get("fluency_score"),
            "confidence_score": student.get("confidence_score"),
            "pronunciation_score": student.get("pronunciation_score"),
            "onboarding_completed": bool(student.get("onboarding_completed", False)),
            "onboarding_step": student.get("onboarding_step", "completed"),
            "speaking_assessment_completed": bool(student.get("speaking_assessment_completed", False)),
            "assessed_level": student.get("assessed_level"),
            "overall_level": student.get("assessed_level") or student.get("target_level") or "Intermediate",
            "grammar_level": student.get("grammar_level") or "Beginner",
            "vocabulary_level": student.get("vocabulary_level") or "Beginner",
            "fluency_level": student.get("fluency_level") or "Beginner",
            "sentence_formation_level": student.get("sentence_formation_level") or "Beginner",
            "communication_confidence": student.get("communication_confidence") or "Low",
            "speaking_hesitation": student.get("speaking_hesitation") or "Frequent",
            "presentation_confidence": student.get("presentation_confidence") or "Low",
            "preferred_explanation_difficulty": int(student.get("preferred_explanation_difficulty") or 1),
            "current_teaching_difficulty": int(student.get("current_teaching_difficulty") or 1),
            "known_words": list(student.get("known_words") or []),
            "difficult_words": list(student.get("difficult_words") or []),
            "recurring_grammar_errors": list(student.get("recurring_grammar_errors") or []),
            "mastered_topics": list(student.get("mastered_topics") or []),
            "struggling_topics": list(student.get("struggling_topics") or []),
            "cumulative_assessment_words": int(student.get("cumulative_assessment_words") or 0),
            "assessment_samples_count": int(student.get("assessment_samples_count") or 0)
        }

    @classmethod
    def get_session_dialogue(
        cls,
        student_id: int,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Retrieves recent conversation messages strictly isolated to this student and session.
        Returns canonical chronological list of {'role': 'user'|'assistant', 'content': '...'}.
        """
        if session_id:
            raw_msgs = get_conversation_history(
                student_id=student_id,
                session_id=session_id,
                limit=limit
            )
        else:
            raw_msgs = get_conversation_history(
                student_id=student_id,
                limit=limit
            )

        dialogue = []
        for msg in raw_msgs:
            # Enforce strict student and session isolation
            if int(msg.get("student_id", 0)) != int(student_id):
                continue
            if session_id and str(msg.get("session_id")) != str(session_id):
                continue
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in ["user", "assistant"] and content:
                dialogue.append({"role": role, "content": content})
        return dialogue

    @classmethod
    def get_session_summary(
        cls,
        student_id: int,
        session_id: Optional[str] = None,
        max_recent: int = 10
    ) -> Optional[str]:
        """
        Condenses older conversation turns beyond the active sliding window into a compact summary.
        Prevents conversational context from growing unbounded while retaining discussion topics.
        """
        if not session_id:
            return None

        # Fetch up to 100 conversation history messages for this session
        all_msgs = get_conversation_history(
            student_id=student_id,
            session_id=session_id,
            limit=100
        )
        # Filter strictly by session and student
        session_msgs = [
            m for m in all_msgs
            if str(m.get("session_id")) == str(session_id) and int(m.get("student_id", 0)) == int(student_id)
            and m.get("role") in ["user", "assistant"] and (m.get("content") or "").strip()
        ]

        if len(session_msgs) <= max_recent:
            return None

        # Summarize older messages that will be evicted from the active sliding window
        older_msgs = session_msgs[:-max_recent]
        user_topics = []
        feedback_points = []

        for m in older_msgs:
            content = m.get("content", "").strip()
            if m.get("role") == "user":
                # Extract first sentence or brief fragment
                first_sentence = content.split(".")[0].strip()
                if first_sentence and len(first_sentence) > 5 and first_sentence not in user_topics:
                    user_topics.append(first_sentence[:60])
            elif m.get("role") == "assistant":
                vocab = m.get("vocabulary_suggestions") or []
                if isinstance(vocab, list) and vocab:
                    for v in vocab:
                        if v not in feedback_points:
                            feedback_points.append(str(v))

        summary_parts = []
        if user_topics:
            summary_parts.append(f"Earlier topics discussed: {'; '.join(user_topics[-4:])}")
        if feedback_points:
            summary_parts.append(f"Vocabulary/grammar points introduced: {', '.join(feedback_points[-4:])}")

        return " | ".join(summary_parts) if summary_parts else f"Session began with {len(older_msgs)} prior conversation turns."

    @classmethod
    def build_llm_context(
        cls,
        student_id: int,
        session_id: Optional[str] = None,
        current_message: str = "",
        mode: str = "coach",
        rag_context: Optional[str] = None,
        system_instruction_override: Optional[str] = None,
        max_recent_messages: int = 10
    ) -> Dict[str, Any]:
        """
        Canonical context constructor for all Mizo conversational interactions:
        1. Loads authoritative student profile from SQLite database.
        2. Formats persistent memory & identity directives into system_prompt.
        3. Formats recent conversation history strictly isolated to active session.
        4. Injects older-session summary if history was truncated.
        5. Appends current user utterance.
        """
        # Ensure session_id exists
        if not session_id:
            act_s = get_active_session(student_id=student_id, default_mode=mode)
            session_id = act_s["session_id"] if act_s else "ses_default"

        # 1. Authoritative Long-Term Profile
        student = cls.get_student_profile(student_id)
        student_name = student["name"]
        is_name_known = student["is_name_known"]

        # 2. Base System Instructions
        db_conf = get_db_settings()
        base_prompt = system_instruction_override or db_conf.get("system_prompt") or (
            "You are Mizo, an empathetic, encouraging, and highly intelligent AI learning tutor. "
            "Keep spoken responses concise and conversational (2-4 sentences max) suitable for clear voice output."
        )
        # Check if user requested simplification or is confused
        from app.services.personalization_service import personalization_service
        is_confused, _ = personalization_service.is_learner_confused(current_message)
        simplification_requested = is_confused or any(
            p in current_message.lower() for p in [
                "explain simply", "make it easier", "make it simpler", "too complicated",
                "i don't understand", "i dont understand", "what do you mean", "explain again"
            ]
        )

        simplification_rules = (
            "• IMMEDIATE SIMPLIFICATION REQUIRED (Learner asked for simpler explanation / confused):\n"
            "  - Immediately simplify your response! Use 1 to 3 short sentences with very familiar everyday words.\n"
            "  - Explain one simple idea at a time with a direct, everyday example.\n"
            "  - Do not introduce complex words or abstract grammar terminology."
        ) if simplification_requested else (
            "• Keep responses short: 1 to 4 short, natural sentences suitable for spoken conversation.\n"
            "• Use clear, accessible vocabulary matching the student's level."
        )

        # Mode Instructions
        mode_instructions = {
            "coach": (
                "MODE: PERSONALIZED COMMUNICATION COACH (NORMAL CONVERSATION).\n"
                "• Natural Conversationalist: Chat naturally and warmly about whatever topic the learner brings up (anime, cooking, hobbies, daily life).\n"
                "• Do NOT act like a constant English examiner. Do NOT force grammar exercises, English lessons, or robotics unless the learner explicitly asks for them.\n"
                "• Selective Grammar Guidance: Do NOT correct every sentence automatically. Only correct when a mistake meaningfully harms understanding or when the user asks. When correcting, keep it to one short sentence (e.g. \"Nice! A more natural way to say that is: '...' \") and then continue the conversation naturally without giving a grammar lecture.\n"
                "• Never constantly ask 'What would you like to talk about?'—respond directly to what the student actually said.\n"
                f"{simplification_rules}"
            ),
            "tutor": (
                "MODE: ACADEMIC SUBJECT TUTOR.\n"
                "• STRICT SYLLABUS BOUNDARY: Teach ONLY information contained in the provided syllabus, chapter, notes, or uploaded study material. Do NOT invent missing syllabus content or hallucinate outside topics.\n"
                "• If the user asks about a topic not in the provided syllabus/material, state clearly: 'I don't have that topic in the current syllabus/material. Please provide that chapter or add it to the subject.'\n"
                "• Do NOT inject unrelated personal interests (such as anime, cooking, or gaming) into academic lessons.\n"
                "• Teaching Style: Extremely beginner-friendly, 3 to 8 short sentences max (2 to 5 for simple questions), one concept at a time, clear practical examples, and avoid unnecessary jargon.\n"
                "• Loop: EXPLAIN -> SIMPLE EXAMPLE -> CHECK UNDERSTANDING. Conclude with 1 simple check-for-understanding question."
            ),
            "speech": (
                "MODE: SPEECH & PRESENTATION COACH.\n"
                "• Guide the student in structuring speeches (Hook -> Core Points -> Impactful Conclusion).\n"
                "• Give constructive feedback on clarity, pacing, eliminating filler words, and vocal confidence.\n"
                "• Encourage vocal projection and expressive cadence."
            )
        }.get(mode, "MODE: COMMUNICATION COACH")

        # Past mistake reinforcements
        recent_mistakes = get_recent_mistakes(student_id, limit=3)
        mistakes_block = ""
        if recent_mistakes:
            mistakes_block = "PAST MISTAKES TO GENTLY REINFORCE:\n" + "\n".join([
                f"- Used '{m['error_text']}' -> preferred: '{m['correction']}'"
                for m in recent_mistakes
            ])

        # Strict assistant and user identity directives (Part 6 & Part 14)
        assistant_identity_directive = (
            "CRITICAL ASSISTANT IDENTITY:\n"
            "1. Your permanent name is Mizo. You are Mizo, an AI-powered personalized learning and communication tutor.\n"
            "2. If the user asks 'What is your name?' or 'Who are you?', you MUST answer: 'My name is Mizo.'\n"
            "3. You must NEVER say 'I don't have a name' or 'I'm just your communication coach'. Your name is always Mizo."
        )

        if is_name_known:
            identity_directive = (
                f"CRITICAL USER IDENTITY & MEMORY DIRECTIVES:\n"
                f"1. The student's name is {student_name.upper()}. This name is authoritative from persistent database memory.\n"
                f"2. NEVER invent, assume, or call the student by any other name (such as Alex, John, User, Text, Unknown, Stop, Going To, Gravity, Robotics, or any word from their speech).\n"
                f"3. If the student asks 'What is my name?', 'Do you remember my name?', 'Who am I?', or 'What do you know about me?', answer directly and accurately that their name is {student_name}.\n"
                f"4. This identity persists across all conversation turns regardless of recent dialogue content."
            )
        else:
            identity_directive = (
                "CRITICAL USER IDENTITY & MEMORY DIRECTIVES:\n"
                "1. The student's name is currently unknown.\n"
                "2. If the student asks for their name, state politely that you don't know their name yet and ask what they would like to be called.\n"
                "3. Never hallucinate or invent a name."
            )

        strengths_str = ", ".join(student.get("strengths", [])) or "Active participation"
        weaknesses_str = ", ".join(student.get("weaknesses", [])) or "Grammar foundations"

        # Conversation Context & Selective Memory Retrieval (Sections 21, 22, 27)
        from app.db.database import get_session_context
        from app.services.context_service import filter_relevant_memories

        session_ctx = get_session_context(session_id)
        current_topic = session_ctx.get("current_topic", "")
        current_subtopic = session_ctx.get("current_subtopic", "")
        current_entities = session_ctx.get("current_entities", [])
        last_intent = session_ctx.get("last_user_intent", "")
        last_correction = session_ctx.get("last_user_correction", "")

        filtered_memory = filter_relevant_memories(student, current_topic=current_topic, current_subtopic=current_subtopic)
        rel_interests = ", ".join(filtered_memory.get("relevant_interests", [])) or (student.get("interests") if not current_topic else "Current discussion topic")
        fav_items = filtered_memory.get("relevant_favorites", {})
        fav_items_str = ", ".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in fav_items.items()]) if fav_items else "None recorded"

        # Assemble full System Prompt
        system_prompt = f"""{base_prompt}

{assistant_identity_directive}

{mode_instructions}

==================================================
CURRENT CONVERSATION CONTEXT (TOPIC PRIORITY OVER PROFILE):
==================================================
- Active Topic: {current_topic or "General Conversation"}
- Active Subtopic: {current_subtopic or "None"}
- Recent Entities: {", ".join(current_entities) if current_entities else "None"}
- Last User Intent: {last_intent or "None"}
- Last User Correction: {last_correction or "None"}

CRITICAL TOPIC & RESPONSE RULES:
1. CURRENT CONVERSATION TOPIC MUST TAKE PRIORITY: Always prioritize the active conversation topic ({current_subtopic or current_topic or "current topic"}) over unrelated profile interests. Do NOT suddenly introduce robotics, ESP32, or engineering unless the user explicitly asks about them!
2. DIRECT FACTUAL ANSWERS: When the user asks a simple question (e.g. 'Who are the main crew members?'), answer the question directly and factually. Do NOT say 'Let's practice English by talking about...' unless the user specifically requested an English lesson.
3. USER CORRECTION HANDLING: If the user corrects you (e.g. 'You forgot Jinbe', 'That's wrong'), reconsider the previous answer using current context and available knowledge. Acknowledge and integrate the correction if accurate.
4. ZERO FABRICATED CONTEXT: Never say 'I remember that...' unless the relevant information is explicitly in the verified context or student record below.

==================================================
LONG-TERM STUDENT PROFILE (RELEVANT DURABLE RECORD):
==================================================
- Student ID: {student_id}
- Student Name: {student_name}
- Current Education / Background: {student.get('education')}
- Target English Level: {student.get('target_level')} (Self-Reported: {student.get('self_reported_level')})
- Primary Learning Goals: {student.get('learning_goals')}
- Focus Weaknesses / Challenges: {weaknesses_str}
- Identified Strengths: {strengths_str}
- Topic-Relevant Interests: {rel_interests}
- Topic-Relevant Favorites: {fav_items_str}
- Baseline Scores: Grammar: {student.get('baseline_grammar')}% | Fluency: {student.get('baseline_fluency')}% | Vocab: {student.get('baseline_vocabulary')}% | Confidence: {student.get('baseline_confidence')}%
- Current Performance: Grammar: {student.get('grammar_score')}% | Fluency: {student.get('fluency_score')}% | Vocab: {student.get('vocabulary_score')}% | Confidence: {student.get('confidence_score')}%

{identity_directive}

{mistakes_block}

COACHING RULES:
- If the student asks about their name, weaknesses, goals, or previous discussions, answer accurately using their profile above.
- Strictness: {student.get('correction_strictness', 'gentle')}.
- Output spoken text directly: 2 to 4 concise spoken sentences for clear audio playback on the robotic tutor speaker.
- Do not use markdown headers, asterisks, or bullet points in the spoken text.
"""
        if rag_context:
            system_prompt += f"\nGROUNDING KNOWLEDGE BASE CONTEXT:\n{rag_context}\n"

        # 3. Short-Term Conversation History & Truncation Handling
        dialogue = cls.get_session_dialogue(
            student_id=student_id,
            session_id=session_id,
            limit=max_recent_messages
        )
        session_summary = cls.get_session_summary(
            student_id=student_id,
            session_id=session_id,
            max_recent=max_recent_messages
        )

        messages: List[Dict[str, str]] = []
        if session_summary:
            messages.append({
                "role": "system",
                "content": f"PREVIOUS SESSION CONVERSATION SUMMARY (Older turns truncated):\n{session_summary}"
            })

        messages.extend(dialogue)
        if current_message and current_message.strip():
            # Avoid duplicate appending if current_message is already the last message in dialogue
            if not dialogue or dialogue[-1].get("content") != current_message.strip():
                messages.append({"role": "user", "content": current_message.strip()})

        # Structured diagnostic log (never prints complete secrets or raw transcripts)
        print(
            f"[MEMORY] Context built: student_id={student_id}, "
            f"profile_loaded=True, name_present={is_name_known}, "
            f"history_messages={len(dialogue)}, truncated={bool(session_summary)}"
        )

        return {
            "system_prompt": system_prompt,
            "messages": messages,
            "student_profile": student,
            "student_name": student_name,
            "is_name_known": is_name_known,
            "session_id": session_id,
            "history_count": len(dialogue),
            "history_truncated": bool(session_summary)
        }


# Singleton instance
memory_manager = MizoMemoryManager()
