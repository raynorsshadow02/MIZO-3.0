import re
import json
import time
from typing import Dict, Any, List, Tuple, Optional
from app.db.database import (
    get_student,
    update_student,
    record_onboarding_baseline,
    save_assessment,
    update_student_historical_scores,
    update_session_metrics,
    log_student_mistake,
    get_student_memory_context,
    get_recent_mistakes,
    get_db_settings
)
from app.services.rag_service import rag_service


class PersonalizationService:
    """
    Core Personalization, Onboarding State Machine & Assessment Engine for Mizo:
    1. Dynamic Intent & Mode Switching (Coach, Tutor, Speech).
    2. Robust Onboarding State Machine with multi-field extraction from natural speech.
    3. Dynamic topic generation for 1-2 minute speaking assessment based on student profile.
    4. Comprehensive linguistic evaluation (Grammar, Vocabulary, Fluency, Pronunciation, Confidence, Communication).
    5. Baseline calibration vs Current session metrics separation.
    6. Non-destructive returning user recognition & memory assembly.
    """

    # Valid Onboarding States
    STATE_NEW = "ask_name"
    STATE_WELCOME = "ask_name"
    STATE_ASK_NAME = "ask_name"
    STATE_ASK_GOALS = "ask_goals"
    STATE_ASK_WEAKNESSES = "ask_weaknesses"
    STATE_ASK_SELF_LEVEL = "ask_self_level"
    STATE_ASSESSMENT_INTRO = "speech_test_prompt"
    STATE_ASSESSMENT = "speech_test_prompt"
    STATE_ASSESSMENT_ANALYSIS = "speech_evaluation"
    STATE_COMPLETED = "completed"

    @classmethod
    def detect_intent_and_mode(cls, utterance: str, current_mode: str = "coach") -> str:
        """
        Dynamically infers user intent to transition between learning modes:
        - 'coach': Natural conversational English, communication coaching (Communication Coach).
        - 'tutor': Academic subjects, STEM, conceptual questions (Subject Tutor).
        - 'speech': Presentations, public speaking, speech rehearsal (Speech Coach).
        """
        text = utterance.lower().strip()

        # Academic tutor triggers (Subject Tutor)
        if any(phrase in text for phrase in [
            "teach me", "explain to me", "what is the concept", "how does", "why does",
            "solve this", "physics", "chemistry", "biology", "mathematics", "history of",
            "lesson on", "definition of", "how do plants", "photosynthesis", "gravitation",
            "subject tutor", "tutor me", "science", "stem"
        ]):
            return "tutor"

        # Speech coaching triggers (Speech Coach)
        if any(phrase in text for phrase in [
            "practice presentation", "speech practice", "public speaking", "rehearse my speech",
            "give a presentation", "debate practice", "pitch my idea", "speech training",
            "practice my speech", "practice speech", "practice my presentation", "speech coach",
            "want to practice my speech", "presentation skills", "speech rehearsal"
        ]):
            return "speech"

        # Conversation coach triggers (Communication Coach)
        if any(phrase in text for phrase in [
            "chat with me", "conversation practice", "improve my english", "talk about my day",
            "practice speaking", "casual chat", "grammar help", "improve my communication",
            "communication coach", "help me improve my communication", "communication skills",
            "help with communication", "improve communication"
        ]):
            return "coach"

        return current_mode or "coach"

    @classmethod
    def get_agent_name(cls, mode: str) -> str:
        """Returns the formal agent display name for logging and UI display."""
        mapping = {
            "coach": "Communication Coach",
            "tutor": "Subject Tutor",
            "speech": "Speech Coach"
        }
        return mapping.get(mode, "Communication Coach")

    @classmethod
    def extract_name(cls, text: str) -> Optional[str]:
        """Extracts person's name from user utterance with safety checks."""
        t = text.strip()
        patterns = [
            r"(?:my name is|i am|i'm|call me|myself|this is)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
            r"^([A-Za-z]+(?:\s+[A-Za-z]+)?)$"
        ]
        for pat in patterns:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                extracted = m.group(1).strip()
                # Exclude common non-name words
                if extracted.lower() not in [
                    "hello", "hi", "hey", "yes", "no", "okay", "ready", "sure",
                    "good", "fine", "what", "how", "student", "learner", "robotic", "robotics",
                    "beginner", "intermediate", "advanced", "fluent", "unsure", "not sure"
                ]:
                    return " ".join(part.capitalize() for part in extracted.split())
        return None

    @classmethod
    def extract_student_info(cls, text: str, current_student: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extracts multiple pieces of student information from a single natural utterance:
        - name
        - education / background / work status
        - learning goals / communication goals
        - learning topics / subjects
        - strengths
        - weaknesses / struggles
        - self-reported English level
        """
        extracted: Dict[str, Any] = {}
        lower = text.lower().strip()

        # 1. Name extraction
        name_cand = cls.extract_name(text)
        if name_cand:
            extracted["name"] = name_cand

        # 2. Education / background status
        edu_patterns = [
            r"(?:i work as a|i work as an|work as a|work as an|working as a|working as an|i am a|i'm a|i am an|i'm an|currently a|studying as a)\s+([^.,;]+?(?:student|engineer|developer|scientist|analyst|manager|teacher|professional|graduate|undergrad|worker|designer|specialist|consultant|in grade \d+|in class \d+))",
            r"(?:work as a|work as an|working as a|working as an)\s+([^.,;]+)",
            r"(?:a|an)\s+([^.,;]+?(?:student|engineer|developer|scientist|scholar|major))",
            r"(?:study|studying|major in|field is|work in)\s+([^.,;]+)"
        ]
        for pat in edu_patterns:
            m = re.search(pat, lower)
            if m:
                edu_text = m.group(1).strip()
                if len(edu_text) > 2 and edu_text not in ["good", "fine", "ready", "not sure"]:
                    extracted["education"] = edu_text.capitalize()
                    break

        # 3. Strengths
        strength_patterns = [
            r"(?:i am good at|i'm good at|good at|skilled in|confident in)\s+([^.,;]+)",
            r"(?:my strength is|my strengths are|strength is|strong point is)\s+([^.,;]+)",
            r"(?:i excel at|i'm great at|great at)\s+([^.,;]+)"
        ]
        found_strengths = []
        for pat in strength_patterns:
            for m in re.finditer(pat, lower):
                s_str = m.group(1).strip().capitalize()
                if s_str and len(s_str) > 2 and s_str not in found_strengths:
                    found_strengths.append(s_str)
        if found_strengths:
            extracted["strengths"] = found_strengths

        # 4. Weaknesses / struggles
        weakness_patterns = [
            r"(?:i struggle with|i struggle to|struggling with|struggle in)\s+([^.,;]+)",
            r"(?:i am weak in|i'm weak in|my weakness is|my weaknesses are|weak at)\s+([^.,;]+)",
            r"(?:i have trouble with|trouble in|difficulty with|difficult for me to|hard for me to)\s+([^.,;]+)",
            r"(?:bad at|problem with|struggling to|challenge is|challenges are)\s+([^.,;]+)",
            r"(?:hesitate when|often hesitate|tend to hesitate|hesitation with|stumble when)\s+([^.,;]+)",
            r"\b(hesitate when speaking|hesitating|hesitation)\b"
        ]
        found_weaknesses = []
        for pat in weakness_patterns:
            for m in re.finditer(pat, lower):
                w_str = m.group(1).strip().capitalize()
                if w_str and len(w_str) > 2 and w_str not in found_weaknesses:
                    found_weaknesses.append(w_str)
        if found_weaknesses:
            extracted["weaknesses"] = found_weaknesses

        # 5. Learning goals / communication goals
        goals_patterns = [
            r"(?:i want to improve|want to improve|goal is to|looking to improve)\s+([^.,;]+)",
            r"(?:i want to learn|wish to learn|my goal is|aiming to)\s+([^.,;]+)",
            r"(?:need help with|like to get better at|hope to master|want to master)\s+([^.,;]+)",
            r"(?:want to|looking to|hoping to|aiming for|focus on|plan to)\s+([^.,;]+?(?:interview|speaking|presentation|conversation|fluency|english|communication|career|exam|ielts|toefl|job))",
            r"(?:master english for|learn english for|study english for)\s+([^.,;]+)",
            r"(?:improve my|improve our|master my)\s+([^.,;]+)"
        ]
        found_goals = []
        for pat in goals_patterns:
            for m in re.finditer(pat, lower):
                g_str = m.group(1).strip().capitalize()
                if g_str and len(g_str) > 2 and g_str not in found_goals:
                    found_goals.append(g_str)
        if found_goals:
            extracted["learning_goals"] = "; ".join(found_goals)

        # 6. Learning topics / subjects
        topic_patterns = [
            r"(?:learn about|interested in learning|topics like|subject of|subjects like)\s+([^.,;]+)",
            r"(?:interested in|passionate about|curious about)\s+([^.,;]+)"
        ]
        found_topics = []
        for pat in topic_patterns:
            for m in re.finditer(pat, lower):
                top_str = m.group(1).strip().capitalize()
                if top_str and len(top_str) > 2 and top_str not in found_topics:
                    found_topics.append(top_str)
        if found_topics:
            extracted["learning_topics"] = ", ".join(found_topics)
            extracted["interests"] = ", ".join(found_topics)

        # 7. Self-reported English level
        if any(w in lower for w in ["not sure", "not-sure", "unsure", "don't know", "dont know", "no idea", "hard to say"]):
            extracted["self_reported_level"] = "Not sure"
        elif any(w in lower for w in ["upper intermediate", "upper-intermediate", "b2"]):
            extracted["self_reported_level"] = "Upper Intermediate"
        elif any(w in lower for w in ["intermediate", "b1", "medium", "moderate"]):
            extracted["self_reported_level"] = "Intermediate"
        elif any(w in lower for w in ["beginner", "a1", "starter", "basic", "novice"]):
            extracted["self_reported_level"] = "Beginner"
        elif any(w in lower for w in ["elementary", "a2"]):
            extracted["self_reported_level"] = "Elementary"
        elif any(w in lower for w in ["advanced", "c1", "proficient"]):
            extracted["self_reported_level"] = "Advanced"
        elif any(w in lower for w in ["fluent", "native", "c2"]):
            extracted["self_reported_level"] = "Fluent"

        return extracted

    @classmethod
    def get_missing_onboarding_fields(cls, student: Dict[str, Any]) -> List[str]:
        """
        Calculates which structured learner profile information is still missing.
        Required onboarding coverage goals:
        1. Name (must be obtained first)
        2. Current studies or work (if relevant)
        3. English / communication goals
        4. Hobbies, interests, or subjects/topics to learn
        5. Strengths in English
        6. Weaknesses or challenges
        7. Self-reported English level (including 'not sure')
        """
        missing: List[str] = []
        name = (student.get("name") or "").strip()
        if not name or name in ["New Learner", "Student", "there"]:
            return ["name"]

        edu = (student.get("education") or "").strip()
        if not edu or edu in ["Student", ""]:
            missing.append("current studies or work")

        goals = (student.get("learning_goals") or "").strip()
        if not goals or goals in ["Improve communication skills", "Improve speaking fluency and communication", ""]:
            missing.append("English or communication goals")

        interests = (student.get("interests") or "").strip()
        topics = (student.get("learning_topics") or "").strip()
        if (not interests or interests in ["General Topics", "Science, Robotics, Storytelling", ""]) and (not topics or topics in ["General Topics", ""]):
            missing.append("hobbies, interests, or topics you want to learn")

        strengths = student.get("strengths") or []
        if not strengths or (len(strengths) == 1 and strengths[0] in ["Active participation", "Good pronunciation", ""]):
            missing.append("strengths in English")

        weaknesses = student.get("weaknesses") or []
        if not weaknesses or (len(weaknesses) == 1 and weaknesses[0] in ["Subject-verb agreement", "Filler words", ""]):
            missing.append("weaknesses or challenges in English")

        self_level = (student.get("self_reported_level") or "").strip()
        if not self_level or self_level in ["Pending", "Not calibrated", ""]:
            missing.append("self-reported English level (or whether you are not sure)")

        return missing

    @classmethod
    def extract_and_update_student_memory(cls, student_id: int, text: str) -> Dict[str, Any]:
        """Extracts personal profile items and persists them immediately to SQLite."""
        student = get_student(student_id)
        if not student:
            return {}

        extracted = cls.extract_student_info(text, current_student=student)
        updates: Dict[str, Any] = {}
        clean_text = text.strip()
        lower = clean_text.lower()

        # Check for skip / unsure indicators
        is_skip_or_unsure = any(w in lower for w in [
            "not sure", "unsure", "don't know", "dont know", "no idea", "skip", "pass", "not really sure", "hard to say"
        ])

        # 1. Update name if new
        if "name" in extracted:
            updates["name"] = extracted["name"]

        # 2. Update education if new
        if "education" in extracted:
            updates["education"] = extracted["education"]
            updates["grade"] = extracted["education"]
        elif is_skip_or_unsure and (not student.get("education") or student.get("education") == "Student"):
            updates["education"] = "General Learner"

        # 3. Update learning goals
        if "learning_goals" in extracted:
            curr_g = student.get("learning_goals") or ""
            new_g = extracted["learning_goals"]
            if not curr_g or curr_g in ["Improve communication skills", "Improve speaking fluency and communication"]:
                updates["learning_goals"] = new_g
            elif new_g.lower() not in curr_g.lower():
                updates["learning_goals"] = f"{curr_g}; {new_g}"
        elif is_skip_or_unsure and (not student.get("learning_goals") or student.get("learning_goals") == "Improve communication skills"):
            updates["learning_goals"] = "General communication improvement"

        # 4. Update learning topics & interests
        if "learning_topics" in extracted:
            curr_t = student.get("learning_topics") or ""
            new_t = extracted["learning_topics"]
            if not curr_t or curr_t in ["General Topics", ""]:
                updates["learning_topics"] = new_t
            elif new_t.lower() not in curr_t.lower():
                updates["learning_topics"] = f"{curr_t}, {new_t}"

        if "interests" in extracted:
            curr_i = student.get("interests") or ""
            new_i = extracted["interests"]
            if not curr_i or curr_i in ["General Topics", "Science, Robotics, Storytelling", ""]:
                updates["interests"] = new_i
            elif new_i.lower() not in curr_i.lower():
                updates["interests"] = f"{curr_i}, {new_i}"
        elif is_skip_or_unsure and (not student.get("interests") or student.get("interests") in ["General Topics", "Science, Robotics, Storytelling"]):
            updates["interests"] = "General Topics"
            updates["learning_topics"] = "General Topics"

        # 5. Update strengths
        if "strengths" in extracted:
            curr_s = list(student.get("strengths") or [])
            for s in extracted["strengths"]:
                if s not in curr_s:
                    curr_s.append(s)
            updates["strengths"] = curr_s[:6]
        elif is_skip_or_unsure and not student.get("strengths"):
            updates["strengths"] = ["Curious learner"]

        # 6. Update weaknesses
        if "weaknesses" in extracted:
            curr_w = list(student.get("weaknesses") or [])
            for w in extracted["weaknesses"]:
                if w not in curr_w:
                    curr_w.append(w)
            updates["weaknesses"] = curr_w[:6]
        elif is_skip_or_unsure and not student.get("weaknesses"):
            updates["weaknesses"] = ["General speaking practice"]

        # 7. Update self-reported level
        if "self_reported_level" in extracted:
            updates["self_reported_level"] = extracted["self_reported_level"]
        elif is_skip_or_unsure and (not student.get("self_reported_level") or student.get("self_reported_level") in ["Pending", "Not calibrated"]):
            updates["self_reported_level"] = "Not sure"

        # Context-aware fallback if student provided a substantial direct answer
        curr_step = (student.get("onboarding_step") or "").lower()
        is_valid_resp, _ = cls.is_valid_onboarding_response(student, curr_step, clean_text)
        if is_valid_resp:
            if curr_step in ["ask_problems", "ask_about_user", "ask_goals"] and "learning_goals" not in updates and len(clean_text) > 3:
                if not student.get("learning_goals") or student.get("learning_goals") in ["Improve communication skills", "Improve speaking fluency and communication"]:
                    updates["learning_goals"] = clean_text
            elif curr_step in ["ask_weaknesses"] and "weaknesses" not in updates and not student.get("weaknesses") and len(clean_text) > 3:
                updates["weaknesses"] = [clean_text]
            elif curr_step in ["ask_self_level"] and "self_reported_level" not in updates and len(clean_text) > 1:
                updates["self_reported_level"] = "Not sure" if is_skip_or_unsure else clean_text

        if updates:
            update_student(student_id, updates)
            return get_student(student_id) or {}
        return student

    @classmethod
    def is_counter_question_or_deflection(cls, text: str) -> bool:
        """
        Determines whether user utterance is a counter-question or off-topic deflection
        rather than answering a direct question.
        Examples:
        - "What is your name?"
        - "Who are you?"
        - "What is your goal?"
        - "Can you speak Spanish?"
        - "Why do you ask?"
        - "What are you doing?"
        """
        if not text:
            return True
        t = text.lower().strip()
        t_no_punct = re.sub(r"[^\w\s]", "", t).strip()

        # Direct question marks with interrogative starter
        if "?" in text:
            question_words = ["what", "who", "where", "why", "how", "when", "can you", "could you", "do you", "are you", "tell me"]
            if any(t_no_punct.startswith(qw) or f" {qw} " in f" {t_no_punct} " for qw in question_words):
                return True

        # Common counter-questions directed at Mizo
        counter_questions = [
            "what is your name", "whats your name", "who are you", "what are you",
            "what is your goal", "whats your goal", "what can you do", "what do you do",
            "how old are you", "where are you from", "where do you live",
            "what is your age", "what is mizo", "who is mizo", "what is mikaza", "who is mikaza", "who made you",
            "who created you", "what are your weaknesses", "can you hear me",
            "are you a robot", "are you ai", "what time is it", "tell me about yourself",
            "tell me a joke", "what is this", "why are you asking", "why do you want to know"
        ]
        for cq in counter_questions:
            if t_no_punct == cq or t_no_punct.startswith(cq):
                return True

        # Questions starting with question starters
        for starter in ["what is your", "whats your", "what are your", "who is", "who are", "can you", "could you", "how do you", "why do you", "tell me about"]:
            if t_no_punct.startswith(starter):
                return True

        return False

    @classmethod
    def is_valid_onboarding_response(cls, student: Dict[str, Any], current_step: str, text: str) -> Tuple[bool, str]:
        """
        Context-aware validation of user response to current onboarding question.
        Returns: (is_valid: bool, reason: str)
        Ensures Mikaza waits until it receives an answer relevant to the current question
        before moving to the next question.
        """
        if not text or not text.strip():
            return False, "empty_or_silence"

        clean = text.strip()
        lower = clean.lower()
        step = (current_step or "").lower()

        # 1. Any counter-question directed at Mikaza is not an answer to Mikaza's question
        if cls.is_counter_question_or_deflection(clean):
            return False, "counter_question"

        # 2. Check per onboarding step
        if step in ["ask_name", "new_student", "welcome"]:
            name = cls.extract_name(clean)
            if name:
                return True, "valid_name"
            # Non-name words or short greetings
            if lower in ["hello", "hi", "hey", "good morning", "good evening", "yes", "no", "okay", "sure", "ready"]:
                return False, "greeting_not_name"
            # If user spoke something like "I want to improve my communication", it's a goal, not a name!
            if any(w in lower for w in ["goal", "improve", "english", "speak", "practice", "learn", "study"]):
                return False, "goal_given_instead_of_name"
            return False, "unclear_name"

        elif step in ["ask_goals", "ask_problems"]:
            # Check if user provided an actual goal or intention
            goal_keywords = [
                "improve", "learn", "goal", "practice", "speak", "speaking", "communication",
                "skills", "fluent", "fluency", "english", "interview", "confidence", "accent",
                "study", "job", "career", "exam", "ielts", "toefl", "master", "better", "coach",
                "tutor", "presentation", "conversation", "talk", "vocabulary", "grammar",
                "hesitat", "public speaking", "work", "school", "business"
            ]
            if any(kw in lower for kw in goal_keywords):
                return True, "valid_goal"

            # Check if user gave an answer with enough descriptive context (>= 3 words) without being deflection
            words = lower.split()
            if len(words) >= 3 and not any(w in lower for w in ["name is", "call me", "i am called", "my name"]):
                return True, "descriptive_goal"

            return False, "unrelated_to_goal"

        elif step in ["ask_weaknesses"]:
            weakness_keywords = [
                "hesitat", "grammar", "vocab", "word", "pronunc", "struggle", "trouble",
                "difficult", "challenge", "bad", "weak", "stumble", "fear", "shy", "nervous",
                "confidence", "fluency", "pace", "speed", "accent", "speaking", "sentence",
                "listen", "not sure", "dont know", "don't know", "no idea", "none", "nothing",
                "hard to say", "everything", "all of it"
            ]
            if any(kw in lower for kw in weakness_keywords):
                return True, "valid_weakness"

            words = lower.split()
            if len(words) >= 3:
                return True, "descriptive_weakness"

            return False, "unrelated_to_weakness"

        elif step in ["ask_self_level"]:
            level_keywords = [
                "beginner", "intermediate", "advanced", "elementary", "basic", "fluent",
                "native", "unsure", "not sure", "dont know", "don't know", "medium", "moderate",
                "average", "c1", "c2", "b1", "b2", "a1", "a2", "starter", "novice", "proficient",
                "hard to say"
            ]
            if any(kw in lower for kw in level_keywords):
                return True, "valid_level"

            return False, "unrelated_to_level"

        return True, "default_accept"

    @classmethod
    def determine_next_onboarding_state(cls, student: Dict[str, Any], current_step: str) -> str:
        """
        Dynamically calculates the next onboarding state based on missing profile fields.
        No fixed question sequence; transitions to speaking assessment when all goals are covered.
        """
        step = (current_step or "").lower()

        if student.get("onboarding_completed", False):
            return "completed"

        if step in ["speech_test_prompt", "speaking_assessment", "speech_evaluation", "speaking_assessment_intro"]:
            return "speech_test_prompt"

        missing = cls.get_missing_onboarding_fields(student)
        if not missing:
            return "speech_test_prompt"

        if "name" in missing:
            return "ask_name"

        if step in ["ask_name", "new_student", "welcome"]:
            if "English or communication goals" in missing:
                return "ask_goals"
            elif "weaknesses or challenges in English" in missing:
                return "ask_weaknesses"
            elif "self-reported English level (or whether you are not sure)" in missing:
                return "ask_self_level"
            return "speech_test_prompt"

        if step in ["ask_goals", "ask_problems"]:
            return "ask_weaknesses"

        if step in ["ask_weaknesses"]:
            if "weaknesses or challenges in English" in missing:
                return "ask_weaknesses"
            elif "self-reported English level (or whether you are not sure)" in missing:
                return "ask_self_level"
            return "speech_test_prompt"

        if step in ["ask_self_level"]:
            if "self-reported English level (or whether you are not sure)" in missing:
                return "ask_self_level"
            return "speech_test_prompt"

        if "English or communication goals" in missing:
            return "ask_goals"
        if "weaknesses or challenges in English" in missing:
            return "ask_weaknesses"
        if "self-reported English level (or whether you are not sure)" in missing:
            return "ask_self_level"

        return "speech_test_prompt"

    @classmethod
    def generate_dynamic_assessment_topic(cls, student: Dict[str, Any]) -> str:
        """Dynamically picks an engaging 1-2 minute speaking topic tailored to the student."""
        education = student.get("education") or student.get("grade") or ""
        topics = student.get("learning_topics") or student.get("interests") or ""
        goals = student.get("learning_goals") or ""

        lower_context = f"{education} {topics} {goals}".lower()

        if "robot" in lower_context or "ai" in lower_context or "engineering" in lower_context:
            return "Tell me about a project or technology you've worked on or find fascinating, and what you found interesting or challenging about it."
        elif "science" in lower_context or "physics" in lower_context or "space" in lower_context:
            return "Tell me about a scientific discovery, concept, or curiosity that excites you and why you find it fascinating."
        elif "business" in lower_context or "marketing" in lower_context or "interview" in lower_context or "presentation" in lower_context:
            return "Describe a recent accomplishment, professional goal, or project you are working on, and how you plan to achieve it."
        elif topics and topics not in ["General Topics", "Science, Robotics, Storytelling", ""]:
            return f"Tell me about your interest in {topics} and why you want to learn more about it."
        else:
            return "Tell me a little about yourself, what you are currently studying or working on, and what you hope to achieve in English communication."

    @classmethod
    def build_system_prompt(
        cls,
        student_id: int,
        mode: str = "coach",
        session_id: Optional[str] = None,
        rag_context: Optional[str] = None
    ) -> str:
        """
        Builds a personalized, context-rich system prompt with:
        - Learner profile & Onboarding state
        - Historical weaknesses & past mistake patterns from persistent database
        - Specific pedagogical guidelines per mode
        - RAG knowledge excerpts (if subject tutor)
        """
        db_conf = get_db_settings()
        base_prompt = db_conf.get("system_prompt") or (
            "You are Mizo, an empathetic, encouraging, and highly intelligent AI learning tutor robot. "
            "Keep spoken responses concise and conversational (2-4 sentences max) suitable for clear voice output."
        )
        strictness = db_conf.get("grammar_strictness") or "balanced"

        mem = get_student_memory_context(student_id)
        student = mem.get("student") or {
            "name": "New Learner",
            "education": "Student",
            "grade": "Grade 8",
            "target_level": "Intermediate",
            "interests": "",
            "learning_goals": "",
            "onboarding_completed": False,
            "onboarding_step": cls.STATE_ASK_NAME,
            "grammar_score": 0.0,
            "vocabulary_score": 0.0,
            "fluency_score": 0.0,
            "confidence_score": 0.0,
            "strengths": [],
            "weaknesses": []
        }

        # Check if student is in onboarding phase
        if not student.get("onboarding_completed", False):
            onboarding_step = student.get("onboarding_step", cls.STATE_ASK_NAME)
            return cls._build_onboarding_prompt(student, onboarding_step, base_prompt=base_prompt)

        # Mode specific instructions for regular sessions
        mode_instructions = {
            "coach": (
                "MODE: COMMUNICATION & CONVERSATION COACH.\n"
                "• Your goal is to help the student speak naturally, fluidly, and confidently in English.\n"
                "• When the student makes a grammatical mistake, respond warmly to their thought first, "
                "then gently offer the natural phrasing.\n"
                "• Naturally introduce 1 rich vocabulary word suitable for their level.\n"
                "• End with 1 engaging open-ended question to keep the voice dialogue flowing."
            ),
            "tutor": (
                "MODE: ACADEMIC SUBJECT TUTOR.\n"
                "• Actively teach academic concepts, STEM topics, and requested subjects.\n"
                "• Break down complex ideas simply, using creative analogies tied to their personal interests.\n"
                "• If knowledge base context is provided below, prioritize it for accurate instruction.\n"
                "• Always conclude with 1 check-for-understanding question."
            ),
            "speech": (
                "MODE: SPEECH & PRESENTATION COACH.\n"
                "• Guide the student in structuring speeches (Hook -> Core Points -> Impactful Conclusion).\n"
                "• Give constructive feedback on clarity, pacing, eliminating filler words, and vocal confidence.\n"
                "• Encourage vocal projection and expressive cadence."
            )
        }.get(mode, "MODE: COMMUNICATION COACH")

        recent_mistakes = mem.get("recent_mistakes", [])
        mistakes_context = ""
        if recent_mistakes:
            mistakes_context = "PAST MISTAKES TO GENTLY REINFORCE:\n" + "\n".join([
                f"- Used '{m['error_text']}' -> preferred: '{m['correction']}' ({m.get('mistake_type', 'grammar')})"
                for m in recent_mistakes[:3]
            ])

        strengths_str = ", ".join(student.get("strengths", [])) or "Active participation"
        weaknesses_str = ", ".join(student.get("weaknesses", [])) or "Working on foundations"

        prompt = f"""{base_prompt}

{mode_instructions}

PERSISTENT LEARNER PROFILE & MEMORY:
- Student Name: {student.get('name')} (Always address them by name when appropriate)
- Background / Education: {student.get('education') or student.get('grade')}
- Target Level: {student.get('target_level')}
- Known Goals: {student.get('learning_goals') or 'Improve English communication'}
- Identified Weaknesses: {weaknesses_str}
- Personal Interests: {student.get('interests') or 'Curious learner'}
- Baseline Performance: Grammar: {student.get('baseline_grammar', 0)}% | Fluency: {student.get('baseline_fluency', 0)}% | Vocab: {student.get('baseline_vocabulary', 0)}%
- Current Scores: Grammar: {student.get('grammar_score', 0)}% | Fluency: {student.get('fluency_score', 0)}% | Vocab: {student.get('vocabulary_score', 0)}% | Confidence: {student.get('confidence_score', 0)}%
- Strengths: {strengths_str}

{mistakes_context}

COACHING RULES:
- If the student asks about their name, weaknesses, goals, or previous discussions, answer accurately using their profile above.
- Strictness: {strictness}.
- Output spoken text directly: 2 to 4 concise spoken sentences for clear audio playback on the robotic tutor speaker.
- Do not use markdown headers, asterisks, or bullet points in the spoken text.
"""
        if rag_context:
            prompt += f"\nGROUNDING KNOWLEDGE BASE CONTEXT:\n{rag_context}\n"

        return prompt

    @classmethod
    def _build_onboarding_prompt(cls, student: Dict[str, Any], step: str, base_prompt: Optional[str] = None) -> str:
        """
        Constructs dynamic onboarding system prompt based on active step.
        The state machine defines WHAT information is required; the LLM dynamically controls HOW Mikaza speaks naturally.
        Proactively guides the student, asks questions one by one, and asks again if unclear.
        """
        step_upper = step.upper()
        student_name = student.get("name", "there")
        education = student.get("education", "")
        goals = student.get("learning_goals", "")
        weaknesses = ", ".join(student.get("weaknesses", []))
        interests = student.get("interests", "")
        dynamic_topic = cls.generate_dynamic_assessment_topic(student)

        tutor_persona = base_prompt or "You are Mizo, an intelligent, empathetic, and proactive AI tutor robot."

        missing_fields = cls.get_missing_onboarding_fields(student)
        missing_desc = "\n".join([f"- {f}" for f in missing_fields]) if missing_fields else "- None (All profile goals covered)"
        strengths_str = ", ".join(student.get("strengths", [])) or "Not yet specified"
        weaknesses_str = ", ".join(student.get("weaknesses", [])) or "Not yet specified"
        self_level_str = student.get("self_reported_level") or "Pending"

        if step_upper in [cls.STATE_NEW, cls.STATE_WELCOME, cls.STATE_ASK_NAME, "ASK_NAME"]:
            instruction = (
                "PURPOSE: Welcome the student warmly and ask for their name as your first question.\n"
                "• Introduce yourself warmly as Mizo, their AI English learning tutor.\n"
                "• The wording may be generated naturally, but asking for their name MUST be the first onboarding question.\n"
                "• Time of day can inform a greeting naturally if useful (e.g., 'Hello! Good morning/afternoon...').\n"
                "• If what the student said was unclear, politely clarify and ask for their name again."
            )
        elif step_upper in [cls.STATE_ASSESSMENT_INTRO, "SPEECH_TEST_PROMPT"]:
            instruction = (
                f"PURPOSE: Introduce the 1-minute speaking skills assessment naturally.\n"
                f"• Student: {student_name} | Background: {education or 'General Learner'} | Interests/Topics: {interests or 'General Topics'}.\n"
                "• Explain that to calibrate their baseline, you'd love to hear them speak freely for about one minute on a tailored topic.\n"
                f"• Present this tailored topic: '{dynamic_topic}'.\n"
                "• Encourage them warmly to speak at their own pace for about 60 seconds."
            )
        elif step_upper in [cls.STATE_ASSESSMENT, cls.STATE_ASSESSMENT_ANALYSIS, "SPEECH_EVALUATION"]:
            instruction = (
                f"PURPOSE: Conclude calibration and give an encouraging assessment summary.\n"
                f"• Student: {student_name}.\n"
                "• Provide a warm, concise spoken summary of their strengths and areas to develop.\n"
                "• Enthusiastically announce that calibration is complete and ask what topic they would like to practice first."
            )
        else:
            instruction = (
                f"PURPOSE: Dynamically acknowledge what the student answered, then ask the next onboarding question.\n"
                f"• Student: {student_name}\n"
                f"• Known Studies/Work: {education or 'Not yet specified'}\n"
                f"• Known Goals: {goals or 'Not yet specified'}\n"
                f"• Known Topics/Interests: {interests or 'Not yet specified'}\n"
                f"• Known Strengths: {strengths_str}\n"
                f"• Known Weaknesses: {weaknesses_str}\n"
                f"• Self-Reported Level: {self_level_str}\n"
                f"INFORMATION STILL MISSING TO COVER:\n{missing_desc}\n\n"
                "MANDATORY INTERACTION PATTERN (ASK -> WAIT -> ANSWER -> ACKNOWLEDGE & NEXT QUESTION -> WAIT):\n"
                "1. Dynamically acknowledge what the student just said in 1 warm, natural sentence (e.g. 'That is a wonderful name, Shahid!', 'Great goal to work towards!', 'That sounds like a fascinating field!'). Do NOT use fixed or canned strings.\n"
                "2. Do NOT use a fixed question sequence or canned questions.\n"
                "3. In the EXACT SAME turn, ask ONE clear, natural question about what is still missing from the list above.\n"
                "4. Then STOP speaking immediately and wait for the student's reply. Do NOT ask multiple future questions. Do NOT answer on behalf of the student.\n"
                "5. If the student gave no understandable answer or silence occurred, do NOT invent or assume facts. Politely ask: 'Sorry, I didn't quite catch that. Could you say that again?' and wait.\n"
                "6. STRICT RULE: NEVER use repetitive filler phrases like 'Take your time', 'Whenever you're ready', or 'Okay, go ahead'.\n"
                "7. Keep your entire turn concise (1 to 3 spoken sentences total)."
            )

        return f"""{tutor_persona}
You are leading a one-on-one voice onboarding conversation with a new student.

CONVERSATION DIRECTIVE:
- YOU must lead the conversation. Ask questions one by one. Do not wait for the student to ask you questions.
- If the student's answer is unclear, incomplete, or doesn't answer what was asked, gently clarify and ASK AGAIN until you get a clear answer.
- Always be encouraging, patient, and conversational.
- Do NOT use a fixed question sequence or canned questions.

CURRENT STEP INSTRUCTION:
{instruction}

CRITICAL VOICE DELIVERY RULES:
- Speak naturally and concisely (1 to 3 sentences max for clear voice delivery).
- Do not use markdown headers, asterisks, bullet points, or robotic numbered lists.
- Avoid repetitive canned phrases or filler like 'Take your time'.
"""

    @classmethod
    def get_onboarding_question_text(cls, step: str, student: Optional[Dict[str, Any]] = None) -> str:
        """Returns natural question text corresponding to an onboarding step."""
        s = (step or "").lower()
        name = student.get("name") if student else None
        has_name = name and name not in ["New Learner", "Student", "there", ""]

        if s in [cls.STATE_NEW, cls.STATE_WELCOME, cls.STATE_ASK_NAME, "ask_name"]:
            return "What is your name?"
        elif s in [cls.STATE_ASK_GOALS, "ask_goals", "ask_problems"]:
            return f"What is your main goal with Mizo{' ' + name if has_name else ''}?"
        elif s in [cls.STATE_ASK_WEAKNESSES, "ask_weaknesses"]:
            return "What would you say is your biggest weakness or challenge in English right now?"
        elif s in [cls.STATE_ASK_SELF_LEVEL, "ask_self_level"]:
            return "How would you describe your current English level—beginner, intermediate, advanced, or are you not sure?"
        elif s in [cls.STATE_ASSESSMENT_INTRO, "speech_test_prompt", "speaking_assessment"]:
            topic = cls.generate_dynamic_assessment_topic(student or {})
            return f"To calibrate your speaking baseline, please speak freely for about one minute on this topic: {topic}"
        return "What would you like to learn or practice today?"

    @classmethod
    def build_onboarding_turn_prompt(
        cls,
        student: Dict[str, Any],
        current_step: str,
        next_step: str,
        user_answer: str,
        last_question_asked: Optional[str] = None
    ) -> str:
        """
        Builds structured context for the LLM to generate a natural dynamic acknowledgment
        plus the next onboarding question in the same conversational turn.
        """
        curr_q = last_question_asked or cls.get_onboarding_question_text(current_step, student)
        next_q = cls.get_onboarding_question_text(next_step, student)
        student_name = student.get("name", "there")

        return f"""You are Mizo, an empathetic, encouraging AI learning tutor robot leading a voice onboarding conversation.

Current onboarding question:
{curr_q}

User answer:
{user_answer}

Next onboarding question:
{next_q}

Generate a short, natural conversational response.

Requirements:
- Acknowledge the user's answer naturally and warmly (e.g., "That's a wonderful name, {student_name}!", "That's a great goal to work towards!").
- Do not repeat the user's answer unnecessarily.
- Do NOT use a fixed question sequence or canned questions.
- Ask the next question in the exact same turn.
- Keep it concise (1 to 2 spoken sentences total).
- Do not ask multiple future questions.
- Do not answer on behalf of the user.
- Strictly avoid repetitive filler phrases like 'Take your time', 'Whenever you're ready', or 'Okay, go ahead'.
- Output spoken text directly with no markdown, asterisks, or bullet points.
"""

    @classmethod
    def build_onboarding_clarification_prompt(
        cls,
        student: Dict[str, Any],
        current_step: str,
        user_answer: str,
        reason: str = ""
    ) -> str:
        """
        Builds structured context for the LLM when user's response did NOT answer the current question.
        Instructs the LLM to address what the user said naturally and ask the current question again.
        """
        curr_q = cls.get_onboarding_question_text(current_step, student)
        student_name = student.get("name") or "there"

        return f"""You are Mizo, an empathetic, encouraging AI learning tutor robot leading a voice onboarding conversation.

CURRENT ONBOARDING QUESTION YOU ASKED:
"{curr_q}"

WHAT THE USER SAID:
"{user_answer}"

SITUATION:
The user's response did NOT answer the current question (they may have asked you a question back like "What is your name?", changed the subject, or gave an unrelated response).

YOUR TASK:
1. Politely and naturally address what they said in 1 brief sentence.
   - For example, if they asked about your name/identity: "That's something you can ask me later!" or "I'm Mizo, your AI English learning tutor!"
   - For example, if they asked another question: "We can definitely explore that later!"
   - For example, if they said something unrelated: "That's good to know!"
2. Then politely ask the CURRENT question again: "{curr_q}"
3. Keep the total response concise, warm, and conversational (1 to 2 spoken sentences total).
4. Do NOT move to the next topic until you receive an answer to your current question.
5. Strictly avoid repetitive filler phrases like 'Take your time', 'Whenever you're ready', or 'Okay, go ahead'.
6. Output spoken text directly with no markdown, asterisks, or bullet points.
"""

    @classmethod
    def get_onboarding_clarification_fallback(
        cls,
        student: Dict[str, Any],
        current_step: str,
        user_answer: str
    ) -> str:
        """
        Provides a dynamic, natural spoken fallback when the LLM is offline or in fallback mode
        and the user's answer did not address the current onboarding question.
        """
        curr_q = cls.get_onboarding_question_text(current_step, student)
        lower = (user_answer or "").lower().strip()
        step = (current_step or "").lower()

        if "name" in lower and ("what" in lower or "who" in lower):
            if step in ["ask_goals", "ask_problems"]:
                return "That's something you can ask me later. First, what is your main goal?"
            return "I'm Mizo, your AI learning tutor! To get started, what should I call you?"
        elif "who are you" in lower or "what are you" in lower:
            if step in ["ask_goals", "ask_problems"]:
                return "That's something you can ask me later. First, what is your main goal?"
            return f"I'm Mizo, your robotic English tutor! First, {curr_q}"
        elif step in ["ask_goals", "ask_problems"]:
            return "That's something you can ask me later. First, what is your main goal?"
        elif step in ["ask_name", "new_student", "welcome"]:
            return "I'm Mizo, your AI English learning tutor! To get started, what should I call you?"
        elif step in ["ask_weaknesses"]:
            return "We can discuss that later. First, what would you say is your biggest weakness or challenge in English right now?"
        elif step in ["ask_self_level"]:
            return "To help me tailor our lessons, how would you describe your current English level—beginner, intermediate, advanced, or are you not sure?"
        return f"We can discuss that later. First, {curr_q}"

    @classmethod
    def build_returning_user_prompt(cls, student: Dict[str, Any]) -> str:
        """Builds a warm, personalized greeting prompt for a returning student upon system restart."""
        name = student.get("name", "there")
        goals = student.get("learning_goals", "improving communication")
        weaknesses = ", ".join(student.get("weaknesses", [])) or "grammar and fluency"
        strengths = ", ".join(student.get("strengths", [])) or "great curiosity"

        return f"""You are Mizo, the AI robotic tutor robot.
The student '{name}' has returned for a new learning session!

STUDENT MEMORY:
- Name: {name}
- Goals: {goals}
- Focus Weaknesses: {weaknesses}
- Key Strengths: {strengths}

INSTRUCTION:
- Give a warm, genuine 2-sentence spoken welcome back to {name}.
- Mention that you remember what you are working on together ({weaknesses} or {goals}).
- Ask an open-ended question to kick off today's session.
- Keep it natural, conversational, and direct (no markdown or asterisks).
"""

    @classmethod
    def evaluate_utterance(
        cls,
        text: str,
        student_id: int = 1,
        session_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, str]], List[str], float, float, float]:
        """
        Linguistic evaluation engine for standard conversational turns:
        - Grammar error detection with rules and corrections.
        - Vocabulary enhancement suggestions.
        - Fluency calculation (word volume, filler word penalty).
        - Pacing calculation.
        - Confidence score.
        Returns: (grammar_errors, vocab_suggestions, fluency_score, pacing_score, confidence_score)
        """
        grammar_errors = []
        vocab_suggestions = []
        clean_text = text.strip()
        words = clean_text.split()
        word_count = len(words)

        if word_count == 0:
            return [], [], 50.0, 50.0, 50.0

        # Grammar & Syntax Analysis
        lower = clean_text.lower()
        patterns = [
            (r"\b(he|she|it) don\'t\b", r"\1 doesn't", "Subject-verb agreement with singular third person"),
            (r"\bi seen\b", "I saw / I have seen", "Incorrect past participle without auxiliary verb"),
            (r"\bmore better\b", "better / much better", "Double comparative error"),
            (r"\bmore faster\b", "faster", "Double comparative error"),
            (r"\bain\'t\b", "am not / is not / are not", "Informal contraction; use standard phrasing"),
            (r"\bthey is\b", "they are", "Subject-verb agreement with plural pronoun"),
            (r"\bwe was\b", "we were", "Subject-verb agreement in past tense"),
            (r"\bme and (my|him|her|\w+)\b", r"My friend and I / \1 and I", "Subject pronoun order and case"),
            (r"\beverybody are\b", "everybody is", "Indefinite pronoun agreement"),
            (r"\bi didn\'t knew\b", "I didn't know", "Base verb form required after auxiliary didn't"),
            (r"\bhe go to\b", "he goes to", "Subject-verb agreement in present simple"),
            (r"\bshe go to\b", "she goes to", "Subject-verb agreement in present simple"),
            (r"\bhe work on\b", "he works on", "Subject-verb agreement in present simple"),
            (r"\bi have went\b", "I have gone", "Incorrect past participle form"),
        ]

        for pat, corr, rule in patterns:
            match = re.search(pat, lower)
            if match:
                matched_str = match.group(0)
                grammar_errors.append({
                    "error": matched_str,
                    "correction": corr,
                    "rule": rule
                })
                if session_id:
                    log_student_mistake(
                        student_id=student_id,
                        session_id=session_id,
                        utterance=clean_text,
                        mistake_type="grammar",
                        error_text=matched_str,
                        correction=corr,
                        explanation=rule
                    )

        # Vocabulary Enrichment
        vocab_lookup = {
            "good": ["exceptional", "beneficial", "valuable"],
            "bad": ["unfavorable", "suboptimal", "adverse"],
            "big": ["substantial", "prominent", "immense"],
            "happy": ["delighted", "enthusiastic", "thrilled"],
            "hard": ["challenging", "rigorous", "intricate"],
            "fast": ["rapid", "swift", "efficient"],
            "small": ["compact", "minimal", "minute"],
            "important": ["crucial", "essential", "pivotal"],
            "easy": ["straightforward", "accessible", "manageable"]
        }

        for w in words:
            cw = w.lower().strip(",.!?\"'")
            if cw in vocab_lookup and len(vocab_suggestions) < 2:
                synonyms = " / ".join(vocab_lookup[cw][:2])
                suggestion = f"Instead of '{cw}', consider using '{synonyms}'"
                if suggestion not in vocab_suggestions:
                    vocab_suggestions.append(suggestion)

        # Fluency & Filler Words
        filler_matches = re.findall(r"\b(um|uh|er|ah|like|you know|basically|actually)\b", lower)
        filler_count = len(filler_matches)

        length_bonus = min(20.0, word_count * 2.0)
        filler_penalty = filler_count * 6.0
        grammar_penalty = len(grammar_errors) * 5.0
        raw_fluency = 72.0 + length_bonus - filler_penalty - grammar_penalty
        fluency_score = round(min(98.0, max(40.0, raw_fluency)), 1)

        # Pacing Score
        if 8 <= word_count <= 30:
            pacing_score = 90.0 - (filler_count * 5.0)
        elif word_count < 8:
            pacing_score = 65.0 + (word_count * 3.0)
        else:
            pacing_score = max(70.0, 95.0 - (word_count - 30) * 0.8)
        pacing_score = round(min(98.0, max(45.0, pacing_score)), 1)

        # Confidence Score
        grammar_score = max(40.0, 95.0 - (len(grammar_errors) * 12.0))
        vocab_score = 75.0 + (5.0 if len(vocab_suggestions) > 0 else 0.0) + min(15.0, word_count * 1.0)
        vocab_score = min(98.0, vocab_score)

        confidence_score = round((grammar_score * 0.25) + (vocab_score * 0.25) + (fluency_score * 0.3) + (pacing_score * 0.2), 1)

        return grammar_errors, vocab_suggestions, fluency_score, pacing_score, confidence_score

    @classmethod
    async def run_speaking_assessment(
        cls,
        transcript: str,
        student_id: int = 1,
        session_id: str = "",
        topic: str = "Self Introduction & Experience",
        duration_seconds: float = 60.0
    ) -> Dict[str, Any]:
        """
        Executes a deep, comprehensive speaking assessment on the 1-2 minute voice transcript:
        1. Evaluates Grammar, Vocabulary, Fluency, Pronunciation/Intelligibility, Confidence, Communication.
        2. Assigns 0-100 scores and overall level (Beginner, Elementary, Intermediate, Upper Intermediate, Advanced).
        3. Extracts evidence and detailed reasoning for every score.
        4. Extracts concrete grammatical and vocabulary mistakes into student_mistakes.
        5. Saves permanent assessment record into SQLite assessments table.
        6. Sets permanent baseline scores on students table.
        7. Marks onboarding_completed = 1 and onboarding_step = 'completed'.
        """
        from app.services.llm_service import llm_service

        student = get_student(student_id) or {}
        words = transcript.strip().split()
        word_count = len(words)

        # Build structured evaluation prompt for Llama 3.3
        analysis_system_prompt = (
            "You are Mizo's Senior Linguistic Assessment Engine. "
            "Analyze the student's 1-2 minute spoken English transcript with precision and objective evidence.\n"
            "Evaluate:\n"
            "A. Grammar (0-100 score + observed issues & error patterns)\n"
            "B. Vocabulary (0-100 score + range & richness)\n"
            "C. Fluency (0-100 score + continuity & filler word analysis)\n"
            "D. Pronunciation & Intelligibility (0-100 score based on readable word clarity and flow)\n"
            "E. Speaking Confidence (0-100 score from delivery continuity and sentence completeness)\n"
            "F. Communication Effectiveness (0-100 score on explaining ideas and organization)\n"
            "G. Overall Level: 'Beginner' | 'Elementary' | 'Intermediate' | 'Upper Intermediate' | 'Advanced'\n"
            "H. Strengths (2-3 items supported by speech)\n"
            "I. Focus Weaknesses (2-3 items supported by speech)\n"
            "J. Mistakes: List of individual errors with error_text, correction, mistake_type, explanation.\n"
            "K. Summary: A warm, encouraging 2-3 sentence spoken summary directly to the student.\n\n"
            "CRITICAL: Output strictly valid JSON with no markdown formatting or commentary."
        )

        user_content = f"""Student Profile:
Name: {student.get('name')}
Education: {student.get('education')}
Goals: {student.get('learning_goals')}
Assessment Topic: {topic}
Speaking Duration: ~{duration_seconds} seconds

Spoken Transcript:
"{transcript}"

Output JSON format:
{{
  "grammar_score": 65.0,
  "vocabulary_score": 70.0,
  "fluency_score": 60.0,
  "pronunciation_score": 75.0,
  "confidence_score": 62.0,
  "communication_score": 68.0,
  "overall_level": "Intermediate",
  "grammar_feedback": "Observed issues...",
  "vocabulary_feedback": "Observed issues...",
  "fluency_feedback": "Observed issues...",
  "pronunciation_feedback": "Observed issues...",
  "confidence_feedback": "Observed issues...",
  "communication_feedback": "Observed issues...",
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "mistakes": [
    {{
      "utterance": "sentence snippet",
      "error_text": "error snippet",
      "correction": "corrected snippet",
      "mistake_type": "grammar",
      "explanation": "why this correction is needed"
    }}
  ],
  "spoken_summary": "Warm 2-3 sentence spoken summary of assessment..."
}}"""

        assessment_result: Dict[str, Any] = {}
        try:
            llm_res = await llm_service.generate_response(
                messages=[{"role": "user", "content": user_content}],
                system_prompt=analysis_system_prompt,
                temperature=0.2,
                max_tokens=800
            )
            # Parse JSON from response
            cleaned_json = llm_res.text.strip()
            if cleaned_json.startswith("```"):
                cleaned_json = re.sub(r"^```(?:json)?\n?", "", cleaned_json)
                cleaned_json = re.sub(r"\n?```$", "", cleaned_json)
            parsed = json.loads(cleaned_json)
            assessment_result = parsed
        except Exception as e:
            print(f"[Assessment] LLM structured analysis fallback: {e}")
            # Robust Rule-Based Fallback Evaluator
            g_errs, v_suggs, flu, pac, conf = cls.evaluate_utterance(transcript, student_id=student_id)
            g_score = max(40.0, 95.0 - (len(g_errs) * 12.0))
            v_score = min(95.0, 70.0 + min(15.0, word_count * 0.8))
            comm_score = round((g_score * 0.4) + (flu * 0.3) + (v_score * 0.3), 1)
            level = "Intermediate" if comm_score >= 65 else ("Elementary" if comm_score >= 50 else "Beginner")

            fallback_mistakes = [
                {
                    "utterance": transcript,
                    "error_text": ge["error"],
                    "correction": ge["correction"],
                    "mistake_type": "grammar",
                    "explanation": ge["rule"]
                }
                for ge in g_errs
            ]

            assessment_result = {
                "grammar_score": round(g_score, 1),
                "vocabulary_score": round(v_score, 1),
                "fluency_score": round(flu, 1),
                "pronunciation_score": 75.0,
                "confidence_score": round(conf, 1),
                "communication_score": round(comm_score, 1),
                "overall_level": level,
                "grammar_feedback": f"Detected {len(g_errs)} grammatical structures to refine." if g_errs else "Good grammatical control.",
                "vocabulary_feedback": "Good vocabulary coverage with opportunity for descriptive expansion.",
                "fluency_feedback": f"Fluency calibrated with {word_count} spoken words.",
                "pronunciation_feedback": "Speech transcript successfully processed with clear speech characteristics.",
                "confidence_feedback": "Natural speaking continuity demonstrated.",
                "communication_feedback": "Effectively conveyed thoughts on the topic.",
                "strengths": ["Clear expression of ideas", "Active participation"],
                "weaknesses": [ge["rule"] for ge in g_errs[:2]] if g_errs else ["Complex sentence linking"],
                "mistakes": fallback_mistakes,
                "spoken_summary": f"Great job on your assessment! Your communication score is {comm_score}%, and your level is calibrated to {level}. Let's begin our personalized lessons!"
            }

        # 1. Log all extracted mistakes to persistent student_mistakes table
        mistakes = assessment_result.get("mistakes", [])
        for m in mistakes:
            if isinstance(m, dict):
                log_student_mistake(
                    student_id=student_id,
                    session_id=session_id,
                    utterance=m.get("utterance", transcript[:100]),
                    mistake_type=m.get("mistake_type", "grammar"),
                    error_text=m.get("error_text", ""),
                    correction=m.get("correction", ""),
                    explanation=m.get("explanation", "")
                )

        # 2. Save full detailed assessment record into assessments table
        assessment_record = {
            "student_id": student_id,
            "session_id": session_id,
            "topic": topic,
            "transcript": transcript,
            "duration_seconds": duration_seconds,
            "grammar_score": float(assessment_result.get("grammar_score", 70.0)),
            "vocabulary_score": float(assessment_result.get("vocabulary_score", 70.0)),
            "fluency_score": float(assessment_result.get("fluency_score", 70.0)),
            "pronunciation_score": float(assessment_result.get("pronunciation_score", 75.0)),
            "confidence_score": float(assessment_result.get("confidence_score", 70.0)),
            "communication_score": float(assessment_result.get("communication_score", 70.0)),
            "overall_level": assessment_result.get("overall_level", "Intermediate"),
            "grammar_feedback": assessment_result.get("grammar_feedback", ""),
            "vocabulary_feedback": assessment_result.get("vocabulary_feedback", ""),
            "fluency_feedback": assessment_result.get("fluency_feedback", ""),
            "pronunciation_feedback": assessment_result.get("pronunciation_feedback", ""),
            "confidence_feedback": assessment_result.get("confidence_feedback", ""),
            "communication_feedback": assessment_result.get("communication_feedback", ""),
            "strengths": assessment_result.get("strengths", []),
            "weaknesses": assessment_result.get("weaknesses", []),
            "raw_analysis_json": assessment_result
        }
        saved_rec = save_assessment(assessment_record)

        # 3. Establish permanent baseline scores and mark onboarding complete
        record_onboarding_baseline(
            student_id=student_id,
            grammar=assessment_record["grammar_score"],
            vocab=assessment_record["vocabulary_score"],
            fluency=assessment_record["fluency_score"],
            pronunciation=assessment_record["pronunciation_score"],
            confidence=assessment_record["confidence_score"],
            communication=assessment_record["communication_score"],
            overall_level=assessment_record["overall_level"],
            strengths=assessment_record["strengths"],
            weaknesses=assessment_record["weaknesses"]
        )

        assessment_result["assessment_id"] = saved_rec.get("id")
        return assessment_result

    @classmethod
    def record_interaction_metrics(
        cls,
        student_id: int,
        session_id: str,
        grammar_errors: List[Any],
        vocab_suggestions: List[str],
        fluency_score: float,
        pacing_score: float,
        confidence_score: float
    ):
        """
        Updates:
        1. Current active session metrics (running session average).
        2. Cumulative historical student metrics in database.
        """
        grammar_accuracy = max(40.0, 100.0 - (len(grammar_errors) * 15.0))
        vocab_richness = 75.0 + (10.0 if len(vocab_suggestions) > 0 else 0.0)

        # Update Session Metrics (Current active session)
        update_session_metrics(
            session_id=session_id,
            grammar_score=grammar_accuracy,
            fluency_score=fluency_score,
            vocab_score=vocab_richness,
            pacing_score=pacing_score
        )

        # Update Historical Student Metrics (Long-term profile evolution)
        update_student_historical_scores(
            student_id=student_id,
            grammar_val=grammar_accuracy,
            vocab_val=vocab_richness,
            fluency_val=fluency_score,
            pacing_val=pacing_score
        )

    @classmethod
    def complete_text_onboarding_without_audio(cls, student_id: int) -> Dict[str, Any]:
        """
        Completes the onboarding profile based on text answers when audio/microphone is unavailable.
        Leaves the speaking assessment pending and does NOT generate a fake baseline.
        """
        student = get_student(student_id)
        if not student:
            return {}

        update_student(student_id, {
            "onboarding_completed": True,
            "onboarding_step": "completed",
            "speaking_assessment_completed": False
        })
        return get_student(student_id) or {}


personalization_service = PersonalizationService()
