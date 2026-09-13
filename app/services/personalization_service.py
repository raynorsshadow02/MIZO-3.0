import re
import json
from typing import Dict, Any, List, Tuple
from app.db.database import get_student, update_student_scores, get_db_settings
from app.services.rag_service import rag_service


class PersonalizationService:
    """Manages student profile adaptation, learning modes, grammar analysis, and memory updates."""

    @classmethod
    def build_system_prompt(cls, student_id: int, mode_override: str = None) -> str:
        """Constructs an adaptive system prompt combining student profile, mode, and RAG knowledge."""
        student = get_student(student_id)
        if not student:
            student = {
                "name": "Student",
                "grade": "Grade 8",
                "target_level": "Intermediate",
                "interests": "Science and Technology",
                "learning_goals": "Improve speaking and academic concepts",
                "grammar_score": 75.0,
                "fluency_score": 80.0
            }

        db_conf = get_db_settings()
        base_prompt = db_conf.get("system_prompt") or "You are Mikaza, an AI learning and communication tutor."
        learning_mode = mode_override or db_conf.get("coaching_mode") or "coach"
        strictness = db_conf.get("grammar_strictness") or "balanced"

        mode_instructions = {
            "coach": (
                "MODE: COMMUNICATION & CONVERSATION COACH.\n"
                "• Your primary goal is to help the student speak naturally and fluently in English.\n"
                "• If the student makes a grammatical mistake, first respond warmly to their idea, "
                "then gently mention how to say it more naturally.\n"
                "• Suggest 1 interesting vocabulary word that fits the conversation."
            ),
            "tutor": (
                "MODE: ACADEMIC SUBJECT TUTOR.\n"
                "• Break down complex concepts into engaging, intuitive explanations suitable for their grade level.\n"
                "• Use analogies related to their interests.\n"
                "• Always end with 1 thought-provoking follow-up question to check understanding."
            ),
            "speech": (
                "MODE: SPEECH & PRESENTATION TRAINER.\n"
                "• Help the student practice public speaking, structuring ideas, and eliminating filler words.\n"
                "• Provide constructive feedback on clarity, structure, and speaking confidence."
            )
        }.get(learning_mode, "")

        prompt = f"""{base_prompt}

{mode_instructions}

STUDENT PROFILE:
- Name: {student.get('name')}
- Grade Level: {student.get('grade')}
- Proficiency Level: {student.get('target_level')}
- Personal Interests: {student.get('interests')}
- Learning Goals: {student.get('learning_goals')}
- Current Fluency: {student.get('fluency_score')}/100 | Grammar: {student.get('grammar_score')}/100

COACHING GUIDELINES:
- Keep spoken replies concise (2 to 4 spoken sentences) so audio playback on the robotic speaker is crisp and natural.
- Be supportive, warm, and encourage continuous voice interaction.
- Strictness level: {strictness}.
"""
        return prompt

    @classmethod
    def evaluate_utterance(cls, text: str) -> Tuple[List[Dict[str, str]], List[str], float]:
        """
        Analyzes student utterance for basic grammar flaws, vocabulary enhancements, and fluency estimation.
        Returns: (grammar_errors, vocab_suggestions, fluency_score)
        """
        grammar_errors = []
        vocab_suggestions = []
        words = text.strip().split()
        word_count = len(words)

        # 1. Fluency heuristic (based on sentence length, absence of excessive fillers)
        fillers = len(re.findall(r'\b(um|uh|er|like|you know)\b', text, re.IGNORECASE))
        base_fluency = min(95.0, max(50.0, 70.0 + (word_count * 1.5) - (fillers * 4.0)))

        # 2. Common grammatical pattern heuristics
        lower = text.lower()
        if re.search(r'\bhe don\'t\b', lower):
            grammar_errors.append({"error": "he don't", "correction": "he doesn't", "rule": "Subject-verb agreement"})
        if re.search(r'\bshe don\'t\b', lower):
            grammar_errors.append({"error": "she don't", "correction": "she doesn't", "rule": "Subject-verb agreement"})
        if re.search(r'\bi seen\b', lower):
            grammar_errors.append({"error": "I seen", "correction": "I saw / I have seen", "rule": "Past tense verb form"})
        if re.search(r'\bmore better\b', lower):
            grammar_errors.append({"error": "more better", "correction": "better", "rule": "Double comparative"})
        if re.search(r'\bain\'t\b', lower):
            grammar_errors.append({"error": "ain't", "correction": "am not / is not / are not", "rule": "Formal communication"})

        # 3. Vocabulary enrichment suggestions
        vocab_map = {
            "good": "exceptional / beneficial",
            "bad": "unfavorable / adverse",
            "big": "substantial / immense",
            "happy": "delighted / thrilled",
            "fast": "rapid / swift",
            "hard": "challenging / rigorous"
        }
        for w in words:
            clean_w = w.lower().strip(",.!?")
            if clean_w in vocab_map and vocab_map[clean_w] not in vocab_suggestions:
                vocab_suggestions.append(f"Instead of '{clean_w}', try '{vocab_map[clean_w]}'")
                if len(vocab_suggestions) >= 2:
                    break

        return grammar_errors, vocab_suggestions, round(base_fluency, 1)

    @classmethod
    def update_student_progress(cls, student_id: int, grammar_errors: List[Any], vocab_suggestions: List[str], fluency: float):
        """Updates student statistics based on performance in the conversation."""
        grammar_delta = -1.5 if grammar_errors else +0.8
        vocab_delta = +0.5 if len(vocab_suggestions) > 0 else +0.2
        update_student_scores(student_id, grammar_delta, vocab_delta, fluency)


personalization_service = PersonalizationService()
