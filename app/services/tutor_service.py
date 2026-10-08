import re
import json
from typing import Dict, Any, List, Optional, Tuple

from app.db.database import (
    get_connection,
    get_subject,
    get_subject_by_name,
    list_all_subjects,
    get_syllabus_units,
    get_syllabus_topics,
    get_topic_by_id,
    get_topic_by_name,
    get_all_topics_for_subject,
    get_entire_subject_hierarchy,
    get_or_create_student_subject_progress,
    update_student_subject_progress,
    record_topic_completed,
    record_topic_struggle,
    record_topic_understood,
    get_session_by_id,
    update_session_context,
    get_student,
    update_student
)


class SubjectTutorService:
    """
    Academic Subject-Based Tutor Agent.
    Strictly teaches ONLY from provided syllabus, chapter, notes, or uploaded study material.
    Adheres to:
    1. Syllabus Boundary Enforcement (rejection of un-provided topics).
    2. Adaptive Simplicity (simple sentences, practical examples, 1 concept at a time).
    3. Teaching Loop (EXPLAIN -> SIMPLE EXAMPLE -> CHECK UNDERSTANDING -> USER ANSWERS -> CORRECT -> NEXT CONCEPT).
    4. User lesson control (simplify, example, quiz, revise, summarize, next topic).
    5. Exam preparation mode strictly grounded in syllabus.
    6. Learning progress tracking (completed, weak, questions asked, revision topics).
    """

    # In-memory session tracking for active quizzes
    _active_quizzes: Dict[str, Dict[str, Any]] = {}
    # In-memory session tracking for active subject/topic: {session_id: (subject_id, topic_id)}
    _session_active_topics: Dict[str, Tuple[Optional[int], Optional[int]]] = {}

    @classmethod
    def reset_study_session(cls, student_id: Optional[int] = None, session_id: Optional[str] = None):
        """Clears old tutor state when user starts a fresh study session."""
        if session_id:
            cls._session_active_topics.pop(session_id, None)
            keys_to_remove = [k for k in cls._active_quizzes if k.startswith(f"{session_id}_") or f"_{session_id}_" in k]
            for k in keys_to_remove:
                cls._active_quizzes.pop(k, None)
        else:
            cls._session_active_topics.clear()
            cls._active_quizzes.clear()

        conn = get_connection()
        cursor = conn.cursor()
        if student_id:
            cursor.execute("UPDATE student_subject_progress SET current_topic_id = NULL WHERE student_id = ?", (student_id,))
        else:
            cursor.execute("UPDATE student_subject_progress SET current_topic_id = NULL")
        conn.commit()
        conn.close()


    @classmethod
    def has_active_quiz(cls, session_id: Optional[str] = None) -> bool:
        """Returns True if there is an unresolved quiz question for this session."""
        if not session_id:
            return bool(cls._active_quizzes)
        return any(k.startswith(f"{session_id}_") or f"_{session_id}_" in k or k == session_id for k in cls._active_quizzes)

    @classmethod
    def get_tutor_session_state(cls, student_id: int, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Returns the active subject and topic in the tutor session."""
        sub, top = cls.get_current_subject_and_topic(student_id, session_id)
        return {
            "active_subject": sub["name"] if sub else None,
            "active_topic": top["topic_name"] if top else None
        }


    @classmethod
    def set_active_subject_and_topic(
        cls,
        student_id: int,
        session_id: Optional[str],
        subject_name_or_id: Any,
        topic_name_or_id: Optional[Any] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Sets the active subject and optional topic in session and student progress."""
        subject = None
        if isinstance(subject_name_or_id, int):
            subject = get_subject(subject_name_or_id)
        elif isinstance(subject_name_or_id, str):
            subject = get_subject_by_name(subject_name_or_id)

        if not subject:
            return None, None

        sub_id = subject["id"]
        sub_name = subject["name"]

        topic = None
        unit_id = None
        topic_id = None
        topic_name = ""

        if topic_name_or_id:
            if isinstance(topic_name_or_id, int):
                topic = get_topic_by_id(topic_name_or_id)
            elif isinstance(topic_name_or_id, str):
                topic = get_topic_by_name(sub_id, topic_name_or_id)

        if not topic:
            # Pick first available topic in first unit if not specified
            units = get_syllabus_units(sub_id)
            if units:
                unit_id = units[0]["id"]
                topics = get_syllabus_topics(unit_id)
                if topics:
                    topic = topics[0]

        if topic:
            topic_id = topic["id"]
            unit_id = topic["unit_id"]
            topic_name = topic["topic_name"]

        # Track in memory for instantaneous session resolution
        sess_key = session_id or f"default_{student_id}"
        cls._session_active_topics[sess_key] = (sub_id, topic_id)

        # Update student_subject_progress
        get_or_create_student_subject_progress(student_id, sub_id)
        prog_updates: Dict[str, Any] = {
            "current_unit_id": unit_id,
            "current_topic_id": topic_id
        }
        update_student_subject_progress(student_id, sub_id, prog_updates)

        # Update learning session if session_id provided
        if session_id:
            update_session_context(session_id, {
                "current_subject_id": sub_id,
                "current_unit_id": unit_id,
                "current_topic_id": topic_id,
                "current_subject": sub_name,
                "current_topic": topic_name
            })

        return subject, topic

    @classmethod
    def get_current_subject_and_topic(
        cls,
        student_id: int,
        session_id: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Resolves current subject and topic from session or progress."""
        subject = None
        topic = None

        sess_key = session_id or f"default_{student_id}"
        if sess_key in cls._session_active_topics:
            sub_id, top_id = cls._session_active_topics[sess_key]
            if sub_id:
                subject = get_subject(sub_id)
            if top_id:
                topic = get_topic_by_id(top_id)
            if subject:
                return subject, topic

        if session_id:
            sess = get_session_by_id(session_id)
            if sess:
                sub_id = sess.get("current_subject_id")
                top_id = sess.get("current_topic_id")
                if sub_id:
                    subject = get_subject(sub_id)
                if top_id:
                    topic = get_topic_by_id(top_id)
                if subject:
                    return subject, topic

        if not subject:
            # Check any existing subject with progress
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT subject_id, current_unit_id, current_topic_id 
                FROM student_subject_progress 
                WHERE student_id = ? 
                ORDER BY updated_at DESC LIMIT 1
            """, (student_id,))
            row = cursor.fetchone()
            conn.close()
            if row:
                subject = get_subject(row["subject_id"])
                if row["current_topic_id"]:
                    topic = get_topic_by_id(row["current_topic_id"])

        if not subject:
            # Fallback to first subject in DB if any
            subjects = list_all_subjects()
            if subjects:
                subject = subjects[0]
                topics = get_all_topics_for_subject(subject["id"])
                if topics:
                    topic = topics[0]

        return subject, topic

    @classmethod
    def detect_subject_selection(cls, text: str) -> Optional[Dict[str, Any]]:
        """Detects if user is asking to select or switch to a subject."""
        norm = text.lower().strip()
        subjects = list_all_subjects()
        for sub in subjects:
            name = sub["name"].lower()
            # check patterns like "select robotics", "teach me robotics", "subject: robotics", "robotics subject"
            patterns = [
                f"select {name}", f"choose {name}", f"switch to {name}",
                f"subject {name}", f"subject: {name}", f"learn {name}",
                f"study {name}", f"teach me {name}"
            ]
            if any(p in norm for p in patterns) or norm == name:
                return sub
        return None

    @classmethod
    def detect_topic_selection(cls, subject_id: int, text: str) -> Optional[Dict[str, Any]]:
        """Detects if user is selecting a specific topic within the subject."""
        norm = text.lower().strip()
        topics = get_all_topics_for_subject(subject_id)
        for top in topics:
            name = top["topic_name"].lower()
            patterns = [
                f"select {name}", f"topic {name}", f"topic: {name}",
                f"teach me {name}", f"explain {name}", f"learn {name}",
                f"start with {name}", f"about {name}"
            ]
            if any(p in norm for p in patterns) or norm == name:
                return top
        return None

    @classmethod
    def extract_requested_topic_phrase(cls, text: str) -> Optional[str]:
        """Extracts candidate topic/concept from user query."""
        norm = text.lower().strip()
        # Strip punctuation
        norm_clean = re.sub(r"[?!.,'\"]", " ", norm).strip()

        prefixes = [
            "teach me about ", "teach me ", "explain to me ", "explain ",
            "what is ", "what are ", "tell me about ", "how does ", "why does ",
            "can you teach me ", "can you explain ", "i want to learn ", "i want to study ",
            "let's learn ", "start teaching "
        ]
        for p in prefixes:
            if norm_clean.startswith(p):
                candidate = norm_clean[len(p):].strip()
                # Remove filler words
                candidate = re.sub(r"^(the|a|an)\s+", "", candidate).strip()
                non_topics = [
                    "it", "this", "that", "it simply", "it very simply", "very simply",
                    "simply", "it again", "again", "more", "now", "here", "to me", "me",
                    "this concept", "it for me", "it in simple words", "in simple words",
                    "easier", "it easier", "it briefly", "briefly", "what this means"
                ]
                if candidate.lower() in non_topics:
                    return None
                if candidate and len(candidate) > 2:
                    return candidate
        return None

    @classmethod
    def check_syllabus_boundary(
        cls,
        subject: Optional[Dict[str, Any]],
        text: str
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Enforces strict syllabus boundaries.
        Returns: (is_within_syllabus, matched_topic, out_of_boundary_topic_name)
        """
        if not subject:
            return True, None, None

        sub_id = subject["id"]
        topics = get_all_topics_for_subject(sub_id)
        if not topics:
            return True, None, None

        norm = text.lower().strip()
        candidate = cls.extract_requested_topic_phrase(text)

        # Non-topic intent queries (like "give me an example", "quiz me", "i don't understand") are within syllabus
        control_phrases = [
            "quiz me", "test me", "give me a question", "ask me a question",
            "explain simply", "explain it simply", "explain it very simply", "explain very simply",
            "make it easier", "too complicated", "i don't understand", "i don't get it",
            "give me an example", "another example", "show an example", "summarize",
            "revise", "repeat", "next topic", "go back", "yes", "no", "what does that mean"
        ]
        if any(cp in norm for cp in control_phrases):
            return True, None, None

        if candidate:
            # Check if candidate matches any topic or key term in current subject
            for t in topics:
                t_name = t["topic_name"].lower()
                if candidate in t_name or t_name in candidate:
                    return True, t, None
                # Check key terms
                for kt in t.get("key_terms", []):
                    term = kt.get("term", "").lower() if isinstance(kt, dict) else str(kt).lower()
                    if term and (candidate in term or term in candidate):
                        return True, t, None
                # Check content keywords
                if candidate in t.get("content", "").lower():
                    return True, t, None

            # Candidate topic was explicitly requested but does NOT exist in syllabus
            # Format candidate nicely (capitalize words)
            cand_display = candidate.title()
            return False, None, cand_display

        # Check if entire query matches or contains any topic
        for t in topics:
            t_name = t["topic_name"].lower()
            if t_name in norm:
                return True, t, None

        return True, None, None

    @classmethod
    def clear_active_quizzes(cls):
        """Clears all in-memory quiz states (useful for test resets)."""
        cls._active_quizzes.clear()

    @classmethod
    def classify_tutor_intent(cls, text: str, has_active_quiz: bool = False) -> str:
        """Classifies tutor control intent deterministically."""
        norm = text.lower().strip()
        # Normalization with and without apostrophes
        clean = re.sub(r"[?!.,'\"]", "", norm).strip()
        clean_with_apos = re.sub(r"[?!.,\"]", "", norm).strip()

        # Simplification triggers (Section 5)
        simplification_triggers = [
            "explain simply", "explain it simply", "explain it very simply", "explain very simply",
            "make it easier", "too complicated", "i dont understand", "i don't understand",
            "i dont get it", "i don't get it", "what does that mean", "explain again",
            "simpler way", "make it simple", "can you make it easier", "confused", "still complicated"
        ]
        if any(trig in clean for trig in simplification_triggers) or any(trig in clean_with_apos for trig in simplification_triggers):
            return "EXPLAIN_SIMPLY"

        # Example triggers
        if any(kw in clean for kw in ["give me an example", "give an example", "show an example", "another example", "practical example"]):
            return "GIVE_EXAMPLE"

        # Quiz & test triggers
        if any(kw in clean for kw in ["quiz me", "test me", "give me a question", "ask me a question", "test my knowledge"]):
            return "QUIZ"

        # Revision triggers
        if any(kw in clean for kw in ["revise this", "revise", "revision", "let's revise", "revision mode"]):
            return "REVISE"

        # Summarize triggers
        if any(kw in clean for kw in ["summarize this", "summarize", "summary", "give me a summary"]):
            return "SUMMARIZE"

        # Navigation triggers
        if any(kw in clean for kw in ["move to the next topic", "next topic", "next concept", "next chapter"]):
            return "NEXT_TOPIC"
        if any(kw in clean for kw in ["go back", "previous topic", "prior topic"]):
            return "PREV_TOPIC"
        if any(kw in clean for kw in ["repeat", "say that again", "repeat this"]):
            return "REPEAT"

        # Exam mode triggers (Section 10)
        if any(kw in clean for kw in ["exam mode", "exam prep", "prepare for exam", "important questions", "mock test", "exam questions"]):
            return "EXAM_PREP"

        # Study session initiation triggers (e.g. entering study section or announcing study material)
        init_triggers = [
            "i want to study", "want to study", "i have notes", "have notes", "uploaded my notes",
            "study for my", "i have my syllabus", "ready to study", "let's study", "study section"
        ]
        if any(trig in clean for trig in init_triggers) and not any(q in clean for q in ["what is", "teach me", "explain", "how does", "why does"]):
            return "STUDY_INIT"

        # If an active quiz question was waiting for an answer
        if has_active_quiz:
            return "ANSWER_QUIZ"

        # Direct questions or explanations
        if clean.startswith("what is ") or clean.startswith("what are ") or clean.startswith("how does ") or clean.startswith("why does "):
            return "ASK_QUESTION"

        if clean.startswith("teach me") or clean.startswith("explain "):
            return "TEACH"

        return "GENERAL"

        return "GENERAL"

    @classmethod
    def get_simple_explanation(cls, topic: Dict[str, Any], ultra_simple: bool = False) -> str:
        """
        Generates beginner-friendly explanation following Section 4.
        Default: 3-8 short sentences, 1 concept at a time, clear meaning, simple vocabulary.
        If ultra_simple: 2-4 very short sentences.
        """
        name = topic["topic_name"]
        simple_exp = topic.get("simple_explanation") or ""
        summary = topic.get("summary") or ""
        content = topic.get("content") or ""

        # Check for key terms
        terms = topic.get("key_terms") or []
        term_meaning = ""
        if terms and isinstance(terms, list) and len(terms) > 0:
            first_term = terms[0]
            if isinstance(first_term, dict):
                term_meaning = f"Term:\n\"{first_term.get('term', name)}\"\n\nSimple meaning:\n\"{first_term.get('meaning', '')}\""
            elif isinstance(first_term, str):
                term_meaning = f"Term:\n\"{name}\"\n\nSimple meaning:\n\"{first_term}\""

        if ultra_simple:
            if simple_exp:
                return f"{simple_exp}\n\nThink of it simply: one step at a time."
            if summary:
                return f"{summary}\n\nLet's keep it very simple."
            return f"{name} is a basic concept in this subject. Let's take it step by step."

        # Standard explanation (3-8 short sentences)
        lines = []
        if term_meaning:
            lines.append(term_meaning)
        elif simple_exp:
            lines.append(simple_exp)
        elif summary:
            lines.append(summary)
        else:
            # Fallback concise sentences from content
            sentences = [s.strip() for s in content.split(".") if s.strip()]
            short_content = ". ".join(sentences[:3]) + "."
            lines.append(short_content)

        # Simple example (Section 4 & 6)
        examples = topic.get("examples") or []
        if examples:
            lines.append(f"Simple example:\n{examples[0]}")

        # Check understanding (Section 6)
        lines.append("Want me to explain it with a simple robot example, or give you a quick check question?")

        return "\n\n".join(lines)

    @classmethod
    def get_example_for_topic(cls, topic: Dict[str, Any]) -> str:
        """Returns a practical real-world example from the topic."""
        examples = topic.get("examples") or []
        name = topic["topic_name"]
        if examples:
            ex_text = examples[0] if len(examples) == 1 else "\n".join([f"• {e}" for e in examples[:2]])
            return f"Here is a simple example for {name}:\n\n{ex_text}\n\nDoes this example make sense to you?"
        return (
            f"Here is a simple practical example for {name}:\n\n"
            f"Imagine moving a robot arm. If we know the angles of its joints, we know exactly where its hand touches.\n\n"
            f"Does this help you picture it?"
        )

    @classmethod
    def get_quiz_question(cls, topic: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a sample quiz question from the topic content."""
        questions = topic.get("sample_questions") or []
        name = topic["topic_name"]
        if questions and isinstance(questions, list) and len(questions) > 0:
            q_data = questions[0]
            if isinstance(q_data, dict):
                return {
                    "question": q_data.get("question", f"What is the main goal of {name}?"),
                    "expected_keywords": q_data.get("expected_keywords", [name.lower()]),
                    "correct_answer": q_data.get("answer", f"{name} describes the position using joint parameters.")
                }
        # Fallback question based on topic name & summary
        summary = topic.get("summary") or topic.get("simple_explanation") or name
        return {
            "question": f"Quick check on {name}: In your own words, what does {name} calculate?",
            "expected_keywords": ["position", "hand", "angles", "joints", "end", "effector"],
            "correct_answer": f"{name} finds the position of the robot hand from joint angles."
        }

    @classmethod
    def evaluate_quiz_answer(
        cls,
        topic: Dict[str, Any],
        user_answer: str,
        quiz_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Evaluates learner's answer against topic knowledge.
        Returns: (is_correct, feedback_message)
        """
        clean = user_answer.lower().strip()
        expected = quiz_data.get("expected_keywords") or []
        correct_ans = quiz_data.get("correct_answer", "")
        name = topic["topic_name"]

        # Check for non-answers or confusion
        if any(p in clean for p in ["i don't know", "not sure", "don't know", "no idea"]):
            feedback = (
                f"No worries at all! Here is the answer:\n\n"
                f"{correct_ans}\n\n"
                f"We'll add {name} to your revision list so we can practice it again later."
            )
            return False, feedback

        # Check keyword matches
        matches = [kw for kw in expected if kw.lower() in clean]
        if len(matches) >= 1 or any(w in clean for w in ["position", "angles", "hand", "joints", "where", "calculat"]):
            feedback = (
                f"Spot on! That's correct. 👍\n\n"
                f"You understood that {name} connects joint angles to the hand's position.\n\n"
                f"Ready to move to the next topic, or do you want another question?"
            )
            return True, feedback

        # Incorrect answer
        feedback = (
            f"Not quite. Here is the correct idea:\n\n"
            f"{correct_ans}\n\n"
            f"Don't worry—I've noted {name} for revision so we can review it together."
        )
        return False, feedback

    @classmethod
    def get_revision_summary(cls, student_id: int, subject: Dict[str, Any], current_topic: Optional[Dict[str, Any]]) -> str:
        """Returns structured revision based on progress and weak topics."""
        prog = get_or_create_student_subject_progress(student_id, subject["id"])
        weak = prog.get("weak_topics") or []
        completed = prog.get("completed_topics") or []

        lines = [f"Let's do a quick revision for {subject['name']}! 📝"]

        if current_topic:
            name = current_topic["topic_name"]
            summary = current_topic.get("summary") or current_topic.get("simple_explanation") or ""
            lines.append(f"• Current Topic: {name}\n  Key point: {summary}")

        if weak:
            lines.append(f"• Topics to practice: {', '.join(weak)}")
        if completed:
            lines.append(f"• Mastered topics: {', '.join(completed)}")

        lines.append("\nWould you like a quick mock question on one of these?")
        return "\n\n".join(lines)

    @classmethod
    def get_exam_prep(cls, subject: Dict[str, Any], current_topic: Optional[Dict[str, Any]]) -> str:
        """Generates exam-oriented revision strictly from syllabus content."""
        sub_id = subject["id"]
        topics = get_all_topics_for_subject(sub_id)
        if not topics:
            return f"No syllabus topics found for {subject['name']} to prepare for exams."

        lines = [f"Here is your Exam Preparation Guide for {subject['name']}: 🎓\n"]
        lines.append("Important Definitions & Concepts:")
        for t in topics[:3]:
            t_name = t["topic_name"]
            t_sum = t.get("summary") or t.get("simple_explanation") or "Core syllabus concept."
            lines.append(f"• {t_name}: {t_sum}")

        lines.append("\nSample Exam Question:")
        if topics:
            q_data = cls.get_quiz_question(topics[0])
            lines.append(f"Q: {q_data['question']}")

        lines.append("\nWould you like me to test you with a mock exam question?")
        return "\n".join(lines)

    @classmethod
    def move_to_next_topic(cls, student_id: int, session_id: Optional[str], subject: Dict[str, Any], current_topic: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str]:
        """Advances to the next topic in the subject."""
        sub_id = subject["id"]
        topics = get_all_topics_for_subject(sub_id)
        if not topics:
            return None, "No topics available in this syllabus."

        next_t = None
        if current_topic:
            curr_id = current_topic.get("id")
            curr_name = (current_topic.get("topic_name") or "").lower().strip()
            for idx, t in enumerate(topics):
                if t["id"] == curr_id or t.get("topic_name", "").lower().strip() == curr_name:
                    for future_t in topics[idx + 1:]:
                        if future_t.get("topic_name", "").lower().strip() != curr_name:
                            next_t = future_t
                            break
                    if next_t:
                        break

        if not next_t:
            next_t = topics[0]

        # Update session & progress
        cls.set_active_subject_and_topic(student_id, session_id, sub_id, next_t["id"])
        reply = (
            f"Moving to the next topic: {next_t['topic_name']}.\n\n"
            f"{cls.get_simple_explanation(next_t)}"
        )
        return next_t, reply

    @classmethod
    def move_to_prev_topic(cls, student_id: int, session_id: Optional[str], subject: Dict[str, Any], current_topic: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str]:
        """Moves to the previous topic in the subject."""
        sub_id = subject["id"]
        topics = get_all_topics_for_subject(sub_id)
        if not topics:
            return None, "No topics available in this syllabus."

        prev_t = None
        if current_topic:
            curr_id = current_topic["id"]
            for idx, t in enumerate(topics):
                if t["id"] == curr_id and idx > 0:
                    prev_t = topics[idx - 1]
                    break

        if not prev_t:
            prev_t = topics[0]

        cls.set_active_subject_and_topic(student_id, session_id, sub_id, prev_t["id"])
        reply = (
            f"Going back to: {prev_t['topic_name']}.\n\n"
            f"{cls.get_simple_explanation(prev_t)}"
        )
        return prev_t, reply

    @classmethod
    async def process_tutor_turn(
        cls,
        student_id: int,
        session_id: str,
        user_text: str,
        current_difficulty: int = 1
    ) -> Dict[str, Any]:
        """
        Main entry point for Subject Tutor Agent.
        Guarantees strict syllabus adherence, adaptive simplicity, and zero outside hallucinations.
        """
        # 1. Resolve subject and current topic
        subject, topic = cls.get_current_subject_and_topic(student_id, session_id)

        # 2. Check for explicit subject selection
        selected_sub = cls.detect_subject_selection(user_text)
        if selected_sub:
            subject, topic = cls.set_active_subject_and_topic(student_id, session_id, selected_sub["id"])
            topic_str = f" We'll begin with '{topic['topic_name']}'." if topic else ""
            reply = f"Great! Let's study {subject['name']}.{topic_str} What would you like to know first?"
            return {
                "reply_text": reply,
                "provider": "system",
                "intent": "SELECT_SUBJECT",
                "new_difficulty": current_difficulty,
                "current_subject": subject["name"],
                "current_topic": topic["topic_name"] if topic else ""
            }

        if not subject:
            # No subject loaded yet
            return {
                "reply_text": "I don't have a syllabus loaded yet. Please select or add a subject first.",
                "provider": "system",
                "intent": "NO_SUBJECT",
                "new_difficulty": 1,
                "current_subject": "",
                "current_topic": ""
            }

        # 3. Check for explicit topic selection within the active subject
        selected_topic = cls.detect_topic_selection(subject["id"], user_text)
        if selected_topic and (not topic or selected_topic["id"] != topic["id"]):
            topic = selected_topic
            cls.set_active_subject_and_topic(student_id, session_id, subject["id"], topic["id"])
            explanation = cls.get_simple_explanation(topic, ultra_simple=False)
            reply = f"Now focusing on {topic['topic_name']}.\n\n{explanation}"
            return {
                "reply_text": reply,
                "provider": "system",
                "intent": "SELECT_TOPIC",
                "new_difficulty": current_difficulty,
                "current_subject": subject["name"],
                "current_topic": topic["topic_name"]
            }

        if not topic:
            topics = get_all_topics_for_subject(subject["id"])
            if topics:
                topic = topics[0]
                cls.set_active_subject_and_topic(student_id, session_id, subject["id"], topic["id"])
            else:
                return {
                    "reply_text": f"The subject {subject['name']} has no topics registered in its syllabus.",
                    "provider": "system",
                    "intent": "EMPTY_SYLLABUS",
                    "new_difficulty": 1,
                    "current_subject": subject["name"],
                    "current_topic": ""
                }

        # 4. Check active quiz status & Classify Intent
        active_quiz_key = f"{student_id}_{session_id or 'default'}_{subject['id']}"
        has_active_quiz = active_quiz_key in cls._active_quizzes
        intent = cls.classify_tutor_intent(user_text, has_active_quiz=has_active_quiz)

        # 5. Strict Syllabus Boundary Check (Sections 1, 8, 14)
        if intent in ["TEACH", "ASK_QUESTION", "GENERAL"]:
            is_in_bounds, matched_top, out_of_bounds_topic = cls.check_syllabus_boundary(subject, user_text)
            if not is_in_bounds and out_of_bounds_topic:
                reply = f"{out_of_bounds_topic} isn't in the current syllabus/material. Would you like to add that topic?"
                return {
                    "reply_text": reply,
                    "provider": "system",
                    "intent": "OUT_OF_SYLLABUS",
                    "new_difficulty": current_difficulty,
                    "current_subject": subject["name"],
                    "current_topic": topic["topic_name"] if topic else ""
                }

            if matched_top and (not topic or matched_top["id"] != topic["id"]):
                topic = matched_top
                cls.set_active_subject_and_topic(student_id, session_id, subject["id"], topic["id"])

        # 7. Execute Intent Action
        if intent == "ANSWER_QUIZ" and has_active_quiz:
            q_data = cls._active_quizzes.pop(active_quiz_key)
            is_correct, feedback = cls.evaluate_quiz_answer(topic, user_text, q_data)
            if is_correct:
                record_topic_understood(student_id, subject["id"], topic["topic_name"])
                new_diff = min(5, current_difficulty + 1)
            else:
                record_topic_struggle(student_id, subject["id"], topic["topic_name"])
                new_diff = 1
            return {
                "reply_text": feedback,
                "provider": "system",
                "intent": "QUIZ_RESULT",
                "new_difficulty": new_diff,
                "current_subject": subject["name"],
                "current_topic": topic["topic_name"]
            }

        if intent == "STUDY_INIT":
            sub_name = subject["name"] if subject else "your subject"
            topics = get_all_topics_for_subject(subject["id"]) if subject else []
            topic_list = f" Topics in your syllabus: {', '.join([t['topic_name'] for t in topics])}." if topics else " You can upload your syllabus or study notes right here."
            reply = f"Welcome to your {sub_name} study section!{topic_list} What would you like to learn or revise first?"
            return {
                "reply_text": reply,
                "provider": "system",
                "intent": "STUDY_INIT",
                "new_difficulty": current_difficulty,
                "current_subject": subject["name"] if subject else "",
                "current_topic": topic["topic_name"] if topic else ""
            }

        elif intent == "EXPLAIN_SIMPLY":
            # Adaptive difficulty down (Section 5)
            new_diff = 1
            ultra_simple = True
            simple_exp = cls.get_simple_explanation(topic, ultra_simple=ultra_simple)
            reply = f"Let's make it super simple:\n\n{simple_exp}"
            return {
                "reply_text": reply,
                "provider": "system",
                "intent": "EXPLAIN_SIMPLY",
                "new_difficulty": new_diff,
                "current_subject": subject["name"],
                "current_topic": topic["topic_name"]
            }

        elif intent == "GIVE_EXAMPLE":
            reply = cls.get_example_for_topic(topic)
            return {
                "reply_text": reply,
                "provider": "system",
                "intent": "GIVE_EXAMPLE",
                "new_difficulty": current_difficulty,
                "current_subject": subject["name"],
                "current_topic": topic["topic_name"]
            }

        elif intent == "QUIZ":
            q_data = cls.get_quiz_question(topic)
            cls._active_quizzes[active_quiz_key] = q_data
            reply = f"Here is a quick question on {topic['topic_name']}:\n\n{q_data['question']}"
            return {
                "reply_text": reply,
                "provider": "system",
                "intent": "QUIZ",
                "new_difficulty": current_difficulty,
                "current_subject": subject["name"],
                "current_topic": topic["topic_name"]
            }

        elif intent == "REVISE":
            reply = cls.get_revision_summary(student_id, subject, topic)
            return {
                "reply_text": reply,
                "provider": "system",
                "intent": "REVISE",
                "new_difficulty": current_difficulty,
                "current_subject": subject["name"],
                "current_topic": topic["topic_name"]
            }

        elif intent == "SUMMARIZE":
            summary = topic.get("summary") or topic.get("simple_explanation") or "Core syllabus topic."
            reply = f"Summary of {topic['topic_name']}:\n\n{summary}"
            return {
                "reply_text": reply,
                "provider": "system",
                "intent": "SUMMARIZE",
                "new_difficulty": current_difficulty,
                "current_subject": subject["name"],
                "current_topic": topic["topic_name"]
            }

        elif intent == "NEXT_TOPIC":
            next_t, reply = cls.move_to_next_topic(student_id, session_id, subject, topic)
            return {
                "reply_text": reply,
                "provider": "system",
                "intent": "NEXT_TOPIC",
                "new_difficulty": current_difficulty,
                "current_subject": subject["name"],
                "current_topic": next_t["topic_name"] if next_t else topic["topic_name"]
            }

        elif intent == "PREV_TOPIC":
            prev_t, reply = cls.move_to_prev_topic(student_id, session_id, subject, topic)
            return {
                "reply_text": reply,
                "provider": "system",
                "intent": "PREV_TOPIC",
                "new_difficulty": current_difficulty,
                "current_subject": subject["name"],
                "current_topic": prev_t["topic_name"] if prev_t else topic["topic_name"]
            }

        elif intent == "REPEAT":
            reply = cls.get_simple_explanation(topic, ultra_simple=True)
            return {
                "reply_text": f"Here it is again:\n\n{reply}",
                "provider": "system",
                "intent": "REPEAT",
                "new_difficulty": current_difficulty,
                "current_subject": subject["name"],
                "current_topic": topic["topic_name"]
            }

        elif intent == "EXAM_PREP":
            reply = cls.get_exam_prep(subject, topic)
            return {
                "reply_text": reply,
                "provider": "system",
                "intent": "EXAM_PREP",
                "new_difficulty": current_difficulty,
                "current_subject": subject["name"],
                "current_topic": topic["topic_name"]
            }

        # Direct explanation / teaching turn (Section 4 & 9)
        # e.g., "What is forward kinematics?"
        explanation = cls.get_simple_explanation(topic, ultra_simple=False)
        return {
            "reply_text": explanation,
            "provider": "system",
            "intent": "TEACH",
            "new_difficulty": current_difficulty,
            "current_subject": subject["name"],
            "current_topic": topic["topic_name"]
        }


tutor_service = SubjectTutorService()
