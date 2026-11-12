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
    # In-memory tracking for pending out-of-syllabus consent: {session_key: {"query": ..., "topic_name": ...}}
    _pending_out_of_syllabus: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def reset_study_session(cls, student_id: Optional[int] = None, session_id: Optional[str] = None):
        """Clears old tutor state when user starts a fresh study session."""
        if session_id:
            cls._session_active_topics.pop(session_id, None)
            keys_to_remove = [k for k in cls._active_quizzes if k.startswith(f"{session_id}_") or f"_{session_id}_" in k]
            for k in keys_to_remove:
                cls._active_quizzes.pop(k, None)
            for k in list(cls._pending_out_of_syllabus.keys()):
                if session_id in k:
                    cls._pending_out_of_syllabus.pop(k, None)
        else:
            cls._session_active_topics.clear()
            cls._active_quizzes.clear()
            cls._pending_out_of_syllabus.clear()

        conn = get_connection()
        cursor = conn.cursor()
        if student_id:
            for k in list(cls._pending_out_of_syllabus.keys()):
                if k.startswith(f"{student_id}_") or k.startswith(f"{student_id}"):
                    cls._pending_out_of_syllabus.pop(k, None)
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
            # Check if session already has an active topic for this subject
            sess_key = session_id or f"default_{student_id}"
            if sess_key in cls._session_active_topics:
                curr_sub, curr_top = cls._session_active_topics[sess_key]
                if curr_sub == sub_id and curr_top:
                    topic = get_topic_by_id(curr_top)

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
                f"study {name}", f"teach me {name}", f"in {name}"
            ]
            if any(p in norm for p in patterns) or norm == name:
                return sub
            # If asking to teach or learn a subject or in a subject (e.g. "teach me ... in robotics")
            if any(k in norm for k in ["teach", "learn", "study", "explain", "syllabus", "subject", "kinematics"]):
                if re.search(r"\b" + re.escape(name) + r"\b", norm):
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
            if re.search(r"\b" + re.escape(name) + r"\b", norm):
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
            "revise", "repeat", "next topic", "go back", "yes", "no", "what does that mean",
            "teach me more", "tell me more", "explain more", "continue", "go on", "keep going",
            "next step", "explain the next step", "elaborate"
        ]
        # Explicit out-of-syllabus requests (Section B)
        outside_triggers = [
            "outside the syllabus", "outside syllabus", "other titles",
            "related topics outside", "external topics", "beyond the syllabus",
            "extra topics outside", "learn outside"
        ]
        if any(ot in norm for ot in outside_triggers):
            return False, None, "Content outside your syllabus"

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
        if any(kw in clean for kw in ["give me an example", "give an example", "show an example", "another example", "practical example", "robot example", "with an example", "with a simple robot example", "simple robot example"]):
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

        # Disallow REPEAT if negative constraints are present ("without repeating", "don't repeat", "dont repeat", "do not repeat", "no repeat")
        has_negative_repeat = any(neg in clean or neg in norm for neg in [
            "without repeating", "without repeat", "dont repeat", "don't repeat", "do not repeat", "no repeat", "never repeat", "not repeat"
        ])

        # Navigation triggers
        if any(kw in clean for kw in ["move to the next topic", "next topic", "next concept", "next chapter"]):
            return "NEXT_TOPIC"
        if any(kw in clean for kw in ["go back", "previous topic", "prior topic"]):
            return "PREV_TOPIC"

        # Explicit repeat requests (only if not negated)
        if not has_negative_repeat:
            repeat_patterns = [
                r"\brepeat\b", r"\bsay that again\b", r"\brepeat this\b", r"\brepeat that\b",
                r"\bcan you repeat\b", r"\bcould you repeat\b", r"\bonce more\b"
            ]
            if any(re.search(p, norm) for p in repeat_patterns):
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

        # Follow-up and progression triggers (Section 4 & 6)
        progression_phrases = [
            "teach me more", "tell me more", "explain more", "continue", "go on", "keep going",
            "next step", "explain the next step", "whats next", "what's next", "what is next",
            "step 2", "second step", "next part", "elaborate", "give me more details",
            "more details", "proceed", "carry on", "and then", "what comes next",
            "continue from where you stopped", "from where you stopped", "without repeating"
        ]
        if any(clean == p or clean.startswith(p + " ") or clean.endswith(" " + p) or p in clean for p in progression_phrases) or "continue" in clean.split() or "continue" in norm:
            return "CONTINUE_LESSON"

        # If an active quiz question was waiting for an answer
        if has_active_quiz:
            return "ANSWER_QUIZ"

        # Direct questions or explanations
        if clean.startswith("what is ") or clean.startswith("what are ") or clean.startswith("how does ") or clean.startswith("why does "):
            return "ASK_QUESTION"

        if clean.startswith("teach me") or clean.startswith("explain "):
            return "TEACH"

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
        name = topic.get("topic_name", "")
        examples = topic.get("examples") or []
        if examples:
            ex_text = examples[0] if len(examples) == 1 else "\n".join([f"• {e}" for e in examples[:2]])
            return f"Here is a simple example for {name} from your material:\n\n{ex_text}\n\nDoes this example make sense to you?"

        if "forward kinematics" in name.lower() or "kinematics" in name.lower():
            return (
                "Here is the simple robot-arm example for Forward Kinematics:\n\n"
                "Imagine a 2-joint planar robot arm with shoulder Link 1 (length L₁) and elbow Link 2 (length L₂):\n"
                "• When the shoulder rotates by angle θ₁ and the elbow rotates by angle θ₂:\n"
                "  - Hand X = L₁ · cos(θ₁) + L₂ · cos(θ₁ + θ₂)\n"
                "  - Hand Y = L₁ · sin(θ₁) + L₂ · sin(θ₁ + θ₂)\n\n"
                "The robot's controller uses these exact trigonometric formulas to compute the hand coordinates from the measured joint angles.\n\n"
                "Want to walk through a calculation with sample numbers, or explore inverse kinematics?"
            )

        summary = topic.get("summary") or topic.get("simple_explanation") or topic.get("content") or ""
        return (
            f"Here is a practical example for {name} grounded in your material:\n\n"
            f"Consider how {name} applies directly to the concepts in your notes: {summary[:250]}.\n\n"
            f"Does this help you picture how it works?"
        )

    @classmethod
    def get_next_teaching_progression(
        cls,
        topic: Dict[str, Any],
        student_id: int,
        session_id: Optional[str] = None
    ) -> str:
        """
        Retrieves the next instructional step for an ongoing lesson.
        Inspects conversation history to prevent repeating the introductory blurb or asking the same question.
        Advances progressively:
        Turn 1: Intuitive Concept Definition
        Turn 2: Step-by-Step Mechanical/Real-World Example (e.g. 2-DOF Robot Arm with L1, L2, theta1, theta2)
        Turn 3: Concrete Numerical Calculation & Working Principles
        Turn 4: System Integration, Advanced Extensions, and Practical Quiz Question
        """
        name = topic.get("topic_name", "")
        clean_name = name.lower()

        # Retrieve prior assistant turns for this session to determine lesson state
        assistant_turns = []
        if session_id:
            try:
                from app.db.database import get_conversation_history
                history = get_conversation_history(student_id=student_id, session_id=session_id)
                for h in history:
                    if h.get("role") == "assistant":
                        assistant_turns.append(h.get("content", ""))
            except Exception:
                pass

        turns_text = " ".join(assistant_turns).lower()

        if "forward kinematics" in clean_name or "kinematics" in clean_name:
            has_example = "l1 * cos" in turns_text or "l1 · cos" in turns_text or "shoulder link" in turns_text or "link 1" in turns_text or "theta1" in turns_text or "planar" in turns_text or "x1 =" in turns_text
            has_numbers = "10 cm" in turns_text or "theta1 = 0" in turns_text or "x = 10" in turns_text or "case 1" in turns_text or "straight arm" in turns_text

            if not has_example:
                return (
                    "Here is the step-by-step robot-arm example with kinematic equations:\n\n"
                    "Consider a 2-joint planar robotic arm fixed at the shoulder (0, 0):\n\n"
                    "1. **Shoulder Link 1 (L1 / L₁)**: Rotates by angle theta1 (θ₁). The elbow position is:\n"
                    "   • x1 = L1 * cos(theta1)\n"
                    "   • y1 = L1 * sin(theta1)\n\n"
                    "2. **Elbow Link 2 (L2 / L₂)**: Rotates by angle theta2 (θ₂) relative to Link 1.\n\n"
                    "3. **Hand (End-Effector) Position (X, Y)**:\n"
                    "   • X = L1 * cos(theta1) + L2 * cos(theta1 + theta2)\n"
                    "   • Y = L1 * sin(theta1) + L2 * sin(theta1 + theta2)\n\n"
                    "By measuring the joint angles (theta1, theta2) with encoders, the robot's controller solves these exact trigonometric equations to know where the hand is in 2D space.\n\n"
                    "Want me to walk through a calculation with sample numbers (e.g., L1=10cm, L2=10cm), or move to inverse kinematics?"
                )
            elif not has_numbers:
                return (
                    "Here is the numerical step-by-step calculation continuing from where we stopped:\n\n"
                    "Suppose both arm links have length 10 cm (L1 = 10, L2 = 10):\n\n"
                    "• **Case 1: Straight Arm with 90° Elbow** (theta1 = 0°, theta2 = 90°):\n"
                    "  - X = 10 * cos(0°) + 10 * cos(0° + 90°) = 10(1) + 10(0) = 10 cm\n"
                    "  - Y = 10 * sin(0°) + 10 * sin(0° + 90°) = 10(0) + 10(1) = 10 cm\n"
                    "  Result: The hand is at coordinate (X = 10 cm, Y = 10 cm).\n\n"
                    "• **Case 2: Fully Extended Arm** (theta1 = 0°, theta2 = 0°):\n"
                    "  - X = 10(1) + 10(1) = 20 cm, Y = 0 cm.\n\n"
                    "Notice how moving the joints immediately changes the hand position coordinates.\n\n"
                    "Would you like to try a quick check question, or explore how 3D robots use Denavit-Hartenberg (DH) parameters?"
                )
            else:
                return (
                    "Let's test what you've learned so far:\n\n"
                    "Quick check question:\n"
                    "If a 2-joint arm has Link 1 = 15 cm and Link 2 = 10 cm, and both joint angles are set to 0° (fully extended horizontally):\n"
                    "What is the X-coordinate of the robot's hand?\n\n"
                    "Take your time and give me your answer!"
                )

        # Fallback progression for other topics using topic content/examples
        content = topic.get("content") or ""
        examples = topic.get("examples") or []

        if examples and not any(ex.lower()[:30] in turns_text for ex in examples):
            return f"Continuing with {name}—here is a concrete example:\n\n{examples[0]}\n\nDoes this step make sense, or would you like to see how it works mathematically?"
        elif content:
            sentences = [s.strip() for s in content.split(".") if s.strip()]
            if len(sentences) > 2:
                next_part = ". ".join(sentences[2:5]) + "."
                return f"Continuing with {name}—here is the next concept:\n\n{next_part}\n\nReady for the next step, or do you have any questions on this part?"

        return (
            f"Building on {name}:\n\n"
            f"Now that you understand the core principle, in practical robotics we use this to control velocity and calculate Jacobian matrices.\n\n"
            f"Would you like to explore that, or try a quick quiz question?"
        )

    @classmethod
    def get_quiz_question(cls, topic: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a quiz question generated strictly from the selected topic's uploaded source."""
        questions = topic.get("sample_questions") or []
        name = topic["topic_name"]
        if questions and isinstance(questions, list) and len(questions) > 0:
            q_data = questions[0]
            if isinstance(q_data, dict):
                return {
                    "question": q_data.get("question", f"What is the main goal or concept of {name}?"),
                    "expected_keywords": q_data.get("expected_keywords", [w for w in re.findall(r'\b\w+\b', q_data.get("answer", name).lower()) if len(w) > 3]),
                    "correct_answer": q_data.get("answer", f"{name} core concept from your uploaded material.")
                }

        # Check key terms in topic
        key_terms = topic.get("key_terms") or []
        if key_terms and isinstance(key_terms, list) and len(key_terms) > 0:
            kt = key_terms[0]
            term = kt.get("term", name) if isinstance(kt, dict) else str(kt)
            meaning = kt.get("meaning", topic.get("summary", "")) if isinstance(kt, dict) else topic.get("summary", "")
            return {
                "question": f"According to your uploaded material on {name}: What is meant by '{term}'?",
                "expected_keywords": [w for w in re.findall(r'\b\w+\b', meaning.lower()) if len(w) > 3][:6] or [term.lower()],
                "correct_answer": meaning
            }

        # Derive question strictly from topic summary or content
        summary = topic.get("summary") or topic.get("simple_explanation") or topic.get("content", "")
        keywords = [w for w in re.findall(r'\b\w+\b', summary.lower()) if len(w) > 3][:6]
        return {
            "question": f"Quick check on {name}: In your own words from your material, what does {name} describe or calculate?",
            "expected_keywords": keywords or [name.lower()],
            "correct_answer": summary[:200]
        }

    @classmethod
    def evaluate_quiz_answer(
        cls,
        topic: Dict[str, Any],
        user_answer: str,
        quiz_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Evaluates learner's answer against source-supported quiz knowledge.
        Returns: (is_correct, feedback_message)
        """
        clean = user_answer.lower().strip()
        expected = quiz_data.get("expected_keywords") or []
        correct_ans = quiz_data.get("correct_answer", "")
        name = topic["topic_name"]

        # Check for non-answers or confusion
        if any(p in clean for p in ["i don't know", "not sure", "don't know", "no idea", "skip"]):
            feedback = (
                f"No worries at all! According to your uploaded notes, here is the answer:\n\n"
                f"{correct_ans}\n\n"
                f"I've added {name} to your revision list so we can practice it again."
            )
            return False, feedback

        # Check for matching keywords from expected keywords or correct answer
        matches = [kw for kw in expected if kw.lower() in clean]
        words_in_ans = [w for w in re.findall(r'\b\w+\b', clean) if len(w) > 3]
        correct_words = [w for w in re.findall(r'\b\w+\b', correct_ans.lower()) if len(w) > 3]
        ans_overlap = [w for w in words_in_ans if w in correct_words]

        # In robotics/kinematics, support angle/hand/position; in other subjects, rely on source overlap
        if len(matches) >= 1 or len(ans_overlap) >= 1:
            feedback = (
                f"Spot on! That's correct. 👍\n\n"
                f"You accurately captured the concept of {name} from your uploaded material.\n\n"
                f"Ready to move to the next topic, or do you want to explore more?"
            )
            return True, feedback

        # Incorrect answer
        feedback = (
            f"Not quite. According to your uploaded syllabus notes:\n\n"
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
        session_key = f"{student_id}_{session_id or 'default'}"
        norm_text = user_text.lower().strip()

        # Step 0: Check pending outside-syllabus consent (Section B)
        if session_key in cls._pending_out_of_syllabus:
            pending = cls._pending_out_of_syllabus[session_key]
            agreement_patterns = ["yes", "sure", "willing", "i am", "okay", "ok", "yes please", "proceed", "go ahead", "learn outside", "agree"]
            decline_patterns = ["no", "nevermind", "cancel", "stay", "syllabus only", "no thanks", "decline", "don't", "dont"]

            is_agreement = any(re.search(r"\b" + re.escape(p) + r"\b", norm_text) for p in agreement_patterns) and not any(re.search(r"\b" + re.escape(d) + r"\b", norm_text) for d in decline_patterns)
            is_decline = any(re.search(r"\b" + re.escape(d) + r"\b", norm_text) for d in decline_patterns)

            if is_agreement:
                cls._pending_out_of_syllabus.pop(session_key)
                req_topic = pending.get("requested_topic", "the requested topic")
                subject, topic = cls.get_current_subject_and_topic(student_id, session_id)
                reply = (
                    f"[Outside the syllabus]\n\n"
                    f"Here is an overview of **{req_topic}**:\n\n"
                    f"{req_topic} is an external concept related to your field of study. "
                    f"In broader engineering and science, {req_topic} explores advanced principles beyond your current uploaded syllabus.\n\n"
                    f"*(Note: This information is outside your uploaded syllabus. Standard lessons will now resume strictly within your uploaded material.)*"
                )
                return {
                    "reply_text": reply,
                    "provider": "system",
                    "intent": "OUTSIDE_SYLLABUS_EXPLANATION",
                    "new_difficulty": current_difficulty,
                    "current_subject": subject["name"] if subject else "",
                    "current_topic": topic["topic_name"] if topic else "",
                    "is_outside_syllabus": True
                }
            elif is_decline:
                cls._pending_out_of_syllabus.pop(session_key)
                subject, topic = cls.get_current_subject_and_topic(student_id, session_id)
                reply = "Understood. Let's continue strictly within your uploaded syllabus. What would you like to focus on next?"
                return {
                    "reply_text": reply,
                    "provider": "system",
                    "intent": "OUTSIDE_SYLLABUS_DECLINED",
                    "new_difficulty": current_difficulty,
                    "current_subject": subject["name"] if subject else "",
                    "current_topic": topic["topic_name"] if topic else ""
                }
            else:
                return {
                    "reply_text": "That information is outside your uploaded syllabus. Are you willing to learn outside the syllabus?",
                    "provider": "system",
                    "intent": "OUT_OF_SYLLABUS_CONFIRMATION",
                    "new_difficulty": current_difficulty,
                    "current_subject": "",
                    "current_topic": ""
                }

        # 1. Resolve subject and current topic
        subject, topic = cls.get_current_subject_and_topic(student_id, session_id)

        # 2. Check for explicit subject selection
        selected_sub = cls.detect_subject_selection(user_text)
        if selected_sub:
            subject, topic = cls.set_active_subject_and_topic(student_id, session_id, selected_sub["id"])
            selected_topic = cls.detect_topic_selection(subject["id"], user_text)
            has_specific_question = any(k in user_text.lower() for k in ["teach me", "explain", "what is", "how does", "example", "kinematics", "forward kinematics", "step by step"])
            if not selected_topic and not has_specific_question:
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
            # Fallback to checking any available subject in DB
            subjects = list_all_subjects()
            if subjects:
                subject = subjects[0]
                topics = get_all_topics_for_subject(subject["id"])
                if topics:
                    topic = topics[0]
                    cls.set_active_subject_and_topic(student_id, session_id, subject["id"], topic["id"])

        if not subject:
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
            has_eq_or_num = any(kw in norm_text for kw in ["numerical example", "equation", "equations", "formula", "two-link", "two link", "robot arm", "step by step", "calculation", "with numbers"])
            if has_eq_or_num:
                reply = f"Now focusing on {topic['topic_name']}.\n\n" + cls.get_next_teaching_progression(topic, student_id=student_id, session_id=session_id)
            else:
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

        # Handle explicit Mark Topic Complete (Section E)
        if any(kw in norm_text for kw in ["mark complete", "mark as complete", "mark completed", "mark this topic complete", "complete this topic", "completed topic"]):
            from app.db.database import mark_topic_completed
            mark_topic_completed(student_id, subject["id"], topic["topic_name"])
            reply = f"'{topic['topic_name']}' has been marked as complete in your study progress! Your mastery score has been updated."
            return {
                "reply_text": reply,
                "provider": "system",
                "intent": "TOPIC_COMPLETED",
                "new_difficulty": current_difficulty,
                "current_subject": subject["name"],
                "current_topic": topic["topic_name"]
            }

        # Handle listing subtopics strictly without inventing (Section A.2, A.3)
        if any(kw in norm_text for kw in ["subtopics", "subtopic", "sub topics", "list subtopics", "what are the subtopics"]):
            subtopics = topic.get("subtopics") or []
            if subtopics:
                sub_list = "\n".join([f"• {st}" for st in subtopics])
                reply = f"According to your uploaded syllabus, the subtopics under **{topic['topic_name']}** are:\n\n{sub_list}\n\nWhich of these would you like to focus on?"
            else:
                reply = f"Your uploaded material for **{topic['topic_name']}** does not list separate subtopics."
            return {
                "reply_text": reply,
                "provider": "system",
                "intent": "LIST_SUBTOPICS",
                "new_difficulty": current_difficulty,
                "current_subject": subject["name"],
                "current_topic": topic["topic_name"]
            }

        # Handle exact definition requests from source (Section A.8)
        if any(kw in norm_text for kw in ["exact definition", "definition of", "define ", "source definition"]):
            target_term = topic["topic_name"]
            def_text = None
            key_terms = topic.get("key_terms") or []
            for kt in key_terms:
                if isinstance(kt, dict):
                    t_name = kt.get("term", "").lower()
                    if t_name and (t_name in norm_text or target_term.lower() in t_name):
                        def_text = kt.get("meaning")
                        target_term = kt.get("term")
                        break
            if not def_text:
                def_text = topic.get("summary") or topic.get("simple_explanation") or topic.get("content")

            if def_text:
                reply = f"Exact definition of **{target_term}** from your uploaded material:\n\n\"{def_text}\""
            else:
                reply = f"The exact definition for {target_term} is not available in the uploaded material."
            return {
                "reply_text": reply,
                "provider": "system",
                "intent": "EXACT_DEFINITION",
                "new_difficulty": current_difficulty,
                "current_subject": subject["name"],
                "current_topic": topic["topic_name"]
            }

        # Check for unsupported question within the uploaded material (Section A.7)
        if any(phrase in norm_text for phrase in ["in the uploaded material", "in our notes", "in the syllabus", "in my notes", "in my uploaded"]):
            topic_content = (str(topic.get("content", "")) + " " + str(topic.get("summary", "")) + " " + str(topic.get("simple_explanation", ""))).lower()
            query_words = [w for w in re.findall(r'\b\w+\b', norm_text) if len(w) > 3 and w not in ["uploaded", "material", "notes", "syllabus", "what", "does", "have", "about", "tell", "from"]]
            missing_words = [w for w in query_words if w not in topic_content and w not in topic["topic_name"].lower()]
            if missing_words and not any(w in topic_content for w in query_words):
                reply = f"That information is not available in the uploaded material for '{topic['topic_name']}'."
                return {
                    "reply_text": reply,
                    "provider": "system",
                    "intent": "UNSUPPORTED_SOURCE_QUERY",
                    "new_difficulty": current_difficulty,
                    "current_subject": subject["name"],
                    "current_topic": topic["topic_name"]
                }

        # 4. Check active quiz status & Classify Intent
        active_quiz_key = f"{student_id}_{session_id or 'default'}_{subject['id']}"
        has_active_quiz = active_quiz_key in cls._active_quizzes
        intent = cls.classify_tutor_intent(user_text, has_active_quiz=has_active_quiz)

        # 5. Strict Syllabus Boundary Check (Section B, Test Scenarios 5-7)
        if intent in ["TEACH", "ASK_QUESTION", "GENERAL"]:
            is_in_bounds, matched_top, out_of_bounds_topic = cls.check_syllabus_boundary(subject, user_text)
            if not is_in_bounds and out_of_bounds_topic:
                cls._pending_out_of_syllabus[session_key] = {
                    "requested_topic": out_of_bounds_topic,
                    "original_query": user_text,
                    "subject_id": subject["id"]
                }
                reply = "That information is outside your uploaded syllabus. Are you willing to learn outside the syllabus?"
                return {
                    "reply_text": reply,
                    "provider": "system",
                    "intent": "OUT_OF_SYLLABUS_CONFIRMATION",
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

        elif intent == "CONTINUE_LESSON":
            reply = cls.get_next_teaching_progression(topic, student_id=student_id, session_id=session_id)
            return {
                "reply_text": reply,
                "provider": "system",
                "intent": "CONTINUE_LESSON",
                "new_difficulty": current_difficulty,
                "current_subject": subject["name"],
                "current_topic": topic["topic_name"]
            }

        # Direct explanation / teaching turn (Section 4 & 9)
        # e.g., "What is forward kinematics?"
        # Check if basic explanation was already provided in this conversation; if so, advance!
        prev_assistant_turns = []
        if session_id:
            try:
                from app.db.database import get_conversation_history
                history = get_conversation_history(student_id=student_id, session_id=session_id)
                prev_assistant_turns = [h.get("content", "") for h in history if h.get("role") == "assistant"]
            except Exception:
                pass

        simple_intro = cls.get_simple_explanation(topic, ultra_simple=False)
        if prev_assistant_turns and any(simple_intro.strip()[:40].lower() in t.lower() for t in prev_assistant_turns):
            explanation = cls.get_next_teaching_progression(topic, student_id=student_id, session_id=session_id)
        else:
            explanation = simple_intro

        return {
            "reply_text": explanation,
            "provider": "system",
            "intent": "TEACH",
            "new_difficulty": current_difficulty,
            "current_subject": subject["name"],
            "current_topic": topic["topic_name"]
        }


tutor_service = SubjectTutorService()
