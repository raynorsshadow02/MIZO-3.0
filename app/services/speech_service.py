import re
import json
from typing import Dict, Any, List, Optional, Tuple

from app.db.database import (
    create_or_get_speech_session,
    get_speech_session_by_id,
    get_active_speech_session,
    update_speech_session,
    get_previous_speech_attempts,
    PROTOTYPE_STUDENT_ID
)


from enum import Enum


class SpeechState(str, Enum):
    SETUP = "SETUP"
    READY = "READY"
    RECORDING = "RECORDING"
    TRANSCRIBING = "TRANSCRIBING"
    ANALYZING = "ANALYZING"
    FEEDBACK = "FEEDBACK"
    OPTIONAL_RETRY = "OPTIONAL_RETRY"


class TimingResult(dict):
    """Custom dictionary supporting tuple unpacking (status, desc) and dict-key lookups."""
    def __iter__(self):
        return iter((self.get("status_text", self.get("status", "")), self.get("desc", "")))


class SpeechPracticeService:
    """
    Dedicated Speech & Seminar Practice Agent.
    Operates like a TED-talk / seminar rehearsal coach.
    State Machine:
    SETUP -> READY -> RECORDING -> TRANSCRIBING -> ANALYZING -> FEEDBACK -> OPTIONAL_RETRY
    """

    STATE_SETUP = SpeechState.SETUP.value
    STATE_READY = SpeechState.READY.value
    STATE_RECORDING = SpeechState.RECORDING.value
    STATE_TRANSCRIBING = SpeechState.TRANSCRIBING.value
    STATE_ANALYZING = SpeechState.ANALYZING.value
    STATE_FEEDBACK = SpeechState.FEEDBACK.value
    STATE_OPTIONAL_RETRY = SpeechState.OPTIONAL_RETRY.value

    COMMON_FILLERS = [
        "um", "uh", "er", "ah", "like", "you know", "basically", "actually",
        "literally", "sort of", "kind of", "i mean"
    ]

    @classmethod
    def get_or_create_session(cls, student_id: int = PROTOTYPE_STUDENT_ID, force_new: bool = False) -> Dict[str, Any]:
        return create_or_get_speech_session(student_id=student_id, force_new=force_new)

    @classmethod
    def extract_setup_information(cls, text: str) -> Dict[str, Any]:
        topic, time_limit, points = cls.parse_full_setup(text)
        return {
            "topic": topic,
            "time_limit_seconds": time_limit,
            "required_points": points
        }

    @classmethod
    def format_time(cls, seconds: int) -> str:
        """Formats seconds into MM:SS format."""
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{mins}:{secs:02d}"

    @classmethod
    def extract_time_limit(cls, text: str) -> Optional[int]:
        """Extracts allowed speaking duration in seconds from natural language."""
        clean = text.lower().strip()

        # Word numbers
        word_numbers = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "half": 0.5
        }

        # Check for phrase like "half an hour"
        if "half an hour" in clean:
            return 1800
        if "an hour" in clean or "one hour" in clean:
            return 3600
        if "a minute" in clean or "one minute" in clean:
            return 60

        # Pattern: digits + minutes/min/mins
        m = re.search(r'(\d+(?:\.\d+)?)\s*(?:minute|minutes|min|mins)\b', clean)
        if m:
            return int(float(m.group(1)) * 60)

        # Pattern: word numbers + minutes/min/mins
        for w, num in word_numbers.items():
            if f"{w} minute" in clean or f"{w} min" in clean:
                return int(num * 60)

        # Pattern: digits + seconds/sec/secs
        m = re.search(r'(\d+)\s*(?:second|seconds|sec|secs)\b', clean)
        if m:
            return int(m.group(1))

        # Pattern: MM:SS like "3:30"
        m = re.search(r'(\d+):(\d{2})', clean)
        if m:
            return int(m.group(1)) * 60 + int(m.group(2))

        # Plain number like "3" or "5" if asked about time
        if clean.isdigit():
            val = int(clean)
            # Typically 1-30 minutes if single small digit, else seconds
            return val * 60 if val <= 30 else val

        return None

    @classmethod
    def extract_topic(cls, text: str) -> Optional[str]:
        """Extracts presentation/speech topic from user statement."""
        clean = text.strip()
        norm = clean.lower()

        # Explicit topic prefixes
        patterns = [
            r'(?:topic is|topic:|about|present on|speech on|seminar on|talk about|discuss)\s+["\']?([^"\'.,;\n]+)["\']?',
            r'topic\s*=\s*["\']?([^"\'.,;\n]+)["\']?',
        ]
        for p in patterns:
            m = re.search(p, clean, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                # Stop before user says "I want to" or "in 3 minutes"
                cand = re.split(r'\b(i want|and i|in \d|with \d|for \d|time limit)\b', cand, flags=re.IGNORECASE)[0].strip()
                if cand and len(cand) > 2 and cand.lower() not in ["presentation", "seminar", "speech"]:
                    return cand

        # If user provides short direct title (e.g. "Artificial Intelligence in Healthcare" or "Robotics")
        words = clean.split()
        if 1 <= len(words) <= 7 and not any(w in norm for w in ["minute", "minutes", "points", "cover", "want to", "hello", "start"]):
            # Filter punctuation
            cand = re.sub(r'^[Tt]opic:\s*', '', clean).strip()
            return cand

        return None

    @classmethod
    def extract_required_points(cls, text: str) -> List[str]:
        """Extracts list of key seminar points from statement."""
        clean = text.strip()

        # Check for numbered or bullet points: 1. ... 2. ...
        numbered = re.findall(r'(?:\d+\.|\-|\•)\s*([^\n\d\.\-•]+)', clean)
        if len(numbered) >= 2:
            return [p.strip().rstrip(",;.") for p in numbered if p.strip()]

        # Pattern: "cover/explain/points: X, Y, Z and W"
        m = re.search(r'(?:points|cover|explain|discuss|include|talking about|touch upon)\s*(?:are|:)?\s*(.+)', clean, re.IGNORECASE)
        target_text = m.group(1) if m else clean

        # Stop target text before time expressions like "in 3 minutes"
        target_text = re.split(r'\b(?:in \d|for \d|time limit|allowed time)\b', target_text, flags=re.IGNORECASE)[0].strip()

        # Split on commas, semicolons, or " and "
        parts = re.split(r'[,;]|\band\b', target_text)
        points = []
        for p in parts:
            p_clean = p.strip().rstrip(".").strip()
            # Remove leading introductory words
            p_clean = re.sub(r'^(what|how|the|a|an|to|i want to|my points are|point is)\s+', '', p_clean, flags=re.IGNORECASE).strip()
            if p_clean and len(p_clean) >= 2 and p_clean.lower() not in ["presentation", "speech", "seminar", "minute", "minutes"]:
                points.append(p_clean)

        return points if len(points) >= 2 else []

    @classmethod
    def parse_full_setup(cls, text: str) -> Tuple[Optional[str], Optional[int], List[str]]:
        """Parses combined natural language setup statement."""
        topic = cls.extract_topic(text)
        time_limit = cls.extract_time_limit(text)
        points = cls.extract_required_points(text)
        return topic, time_limit, points

    @classmethod
    def check_point_coverage(cls, transcript: str, points: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Evidence-based point coverage determination.
        Returns: (covered, partially_covered, missing)
        """
        if not transcript or not points:
            return [], [], points

        raw_lower = transcript.lower()
        sentences = [s.strip().lower() for s in re.split(r'[.?!]', raw_lower) if s.strip()]

        covered = []
        partial = []
        missing = []

        for pt in points:
            pt_clean = pt.lower().strip()
            # Tokenize key words (ignoring basic stop words)
            keywords = [w for w in re.findall(r'\w+', pt_clean) if len(w) > 2 and w not in ["what", "how", "the", "and", "for", "with", "about"]]
            if not keywords:
                keywords = [pt_clean]

            # Count occurrences and sentence contexts
            match_sentences = []
            for s in sentences:
                if any(kw in s for kw in keywords):
                    match_sentences.append(s)

            if len(match_sentences) >= 2 or (len(match_sentences) == 1 and len(match_sentences[0].split()) >= 5):
                covered.append(pt)
            elif len(match_sentences) == 1:
                partial.append(pt)
            else:
                missing.append(pt)

        return covered, partial, missing

    @classmethod
    def evaluate_point_coverage(cls, required_points: Any, transcript: Any) -> List[Dict[str, str]]:
        """
        Evidence-based point coverage determination returning structured results:
        [{"point": str, "status": "covered"|"partial"|"missing", "evidence": str}]
        """
        if isinstance(required_points, str) and isinstance(transcript, list):
            required_points, transcript = transcript, required_points

        covered, partial, missing = cls.check_point_coverage(transcript, required_points)
        sentences = [s.strip() for s in re.split(r'[.?!]', transcript or "") if s.strip()]

        result = []
        for pt in (required_points or []):
            if pt in covered:
                status = "covered"
            elif pt in partial:
                status = "partial"
            else:
                status = "missing"

            keywords = [w.lower() for w in re.findall(r'\w+', pt) if len(w) > 2 and w.lower() not in ["what", "how", "the", "and", "for", "with", "about"]]
            evidence = ""
            if status != "missing":
                for s in sentences:
                    if any(kw in s.lower() for kw in keywords):
                        evidence = s.strip()
                        break
                if not evidence and sentences:
                    evidence = sentences[0]
            result.append({
                "point": pt,
                "status": status,
                "evidence": evidence
            })
        return result

    @classmethod
    def detect_grammar_issues(cls, transcript: str) -> List[Dict[str, str]]:
        issues, _ = cls.analyze_grammar(transcript)
        return issues

    @classmethod
    def analyze_structure(cls, transcript: str) -> Dict[str, str]:
        """Analyzes speech structure (Introduction, Flow, Conclusion)."""
        words = transcript.lower().split()
        total_words = len(words)

        if total_words < 15:
            return {
                "introduction": "Incomplete or too brief",
                "flow": "Insufficient speech to assess flow",
                "conclusion": "Missing"
            }

        first_third = " ".join(words[:max(10, total_words // 3)])
        last_third = " ".join(words[max(0, (2 * total_words) // 3):])

        # Introduction check
        has_intro_phrases = any(p in first_third for p in [
            "hello", "hi", "good morning", "good afternoon", "today i", "talk about",
            "discuss", "topic is", "welcome", "my name", "let's look at"
        ])
        intro = "Good opening hook and clear topic statement" if has_intro_phrases else "Could be stronger (add a greeting or topic statement)"

        # Conclusion check
        has_conclusion = any(p in last_third for p in [
            "in conclusion", "to sum up", "to conclude", "finally", "thank you",
            "thanks for listening", "in summary", "wrap up", "lastly"
        ])
        conclusion = "Clear and decisive conclusion" if has_conclusion else "Missing or abrupt finish (add a summary or closing thought)"

        return {
            "introduction": intro,
            "flow": "Logical progression across points",
            "conclusion": conclusion
        }

    @classmethod
    def analyze_fluency_and_clarity(cls, transcript: str) -> Tuple[List[str], List[str]]:
        """Analyzes fillers, repetitions, broken sentences, and clarity."""
        norm = transcript.lower()
        words = re.findall(r'\b\w+\b', norm)

        # Filler detection
        fillers_found = []
        for filler in cls.COMMON_FILLERS:
            count = len(re.findall(r'\b' + re.escape(filler) + r'\b', norm))
            if count > 0:
                fillers_found.append((filler, count))

        fluency_notes = []
        if fillers_found:
            top_fillers = [f"'{f}' ({cnt}x)" for f, cnt in sorted(fillers_found, key=lambda x: x[1], reverse=True)[:3]]
            fluency_notes.append(f"Filler words noted: {', '.join(top_fillers)}")
        else:
            fluency_notes.append("Good pacing with few to no noticeable fillers")

        # Repetition detection (e.g. "I I", "the the")
        repeated_words = []
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) > 1:
                repeated_words.append(words[i])
        if repeated_words:
            unique_reps = list(set(repeated_words))[:2]
            fluency_notes.append(f"Repeated words detected: {', '.join(unique_reps)}")

        # Clarity assessment
        clarity_notes = []
        sentences = [s.strip() for s in re.split(r'[.?!]', transcript) if s.strip()]
        long_sentences = [s for s in sentences if len(s.split()) > 25]
        if long_sentences:
            clarity_notes.append("A few sentences were overly long—try splitting complex thoughts into shorter sentences")
        else:
            clarity_notes.append("Sentences were concise and clear to follow")

        return fluency_notes, clarity_notes

    @classmethod
    def analyze_grammar(cls, transcript: str) -> Tuple[List[Dict[str, str]], int]:
        """Detects identifiable grammar issues and suggests corrections."""
        issues = []
        # Common grammar patterns
        grammar_rules = [
            (r'\b(ai|technology|robotics|healthcare)\s+have\b', "AI have", "AI has", "Singular subject takes 'has'."),
            (r'\bi\s+am\s+agree\b', "I am agree", "I agree", "Use 'I agree' instead of 'I am agree'."),
            (r'\bpeople\s+is\b', "people is", "people are", "'People' is plural and takes 'are'."),
            (r'\bmany\s+informations\b', "many informations", "a lot of information", "'Information' is uncountable."),
            (r'\bit\s+give\b', "it give", "it gives", "Third-person singular takes 'gives'."),
            (r'\bwe\s+was\b', "we was", "we were", "'We' takes 'were' in past tense."),
            (r'\bhe\s+do\b', "he do", "he does", "Third-person singular takes 'does'."),
            (r'\bmuch\s+people\b', "much people", "many people", "'People' is countable and takes 'many'."),
        ]

        for pat, err, corr, expl in grammar_rules:
            if re.search(pat, transcript, re.IGNORECASE):
                issues.append({
                    "original": err,
                    "correction": corr,
                    "explanation": expl
                })

        return issues, len(issues)

    @classmethod
    def analyze_vocabulary(cls, transcript: str) -> List[str]:
        """Detects repetitive vocabulary and offers alternatives."""
        words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', transcript)]
        overused = ["good", "bad", "thing", "things", "stuff", "very", "big"]
        notes = []
        for word in overused:
            count = words.count(word)
            if count >= 3:
                alts = {
                    "good": "beneficial, effective, valuable, or impactful",
                    "thing": "aspect, element, factor, or component",
                    "things": "features, elements, or concepts",
                    "very": "substantially, highly, or exceptionally"
                }.get(word, "more precise vocabulary")
                notes.append(f"'{word}' was repeated {count} times — consider using {alts}")

        if not notes:
            notes.append("Good vocabulary variety across topic areas")
        return notes

    @classmethod
    def evaluate_timing(cls, actual_seconds: float, allowed_seconds: float, finished_intentionally: bool = False) -> TimingResult:
        """Classifies delivery time adherence."""
        ratio = actual_seconds / allowed_seconds if allowed_seconds > 0 else 1.0

        if ratio < 0.4:
            status = "too_short"
            status_text = "too short"
            desc = "Speech was quite brief for the allotted time"
        elif ratio <= 1.05:
            status = "within_time"
            status_text = "within time"
            desc = "Excellent time management—well within the allotted window"
        elif ratio <= 1.25:
            status = "slightly_over"
            status_text = "slightly over"
            desc = "Slightly over the time limit—aim to trim 15–30 seconds"
        else:
            status = "significantly_over"
            status_text = "significantly over"
            desc = "Significantly over time—prioritize core points to finish on schedule"

        return TimingResult({
            "status": status,
            "status_text": status_text,
            "desc": desc,
            "actual_seconds": actual_seconds,
            "allowed_seconds": allowed_seconds,
            "finished_intentionally": finished_intentionally
        })

    @classmethod
    def generate_feedback_report(
        cls,
        topic: str,
        actual_seconds: float,
        allowed_seconds: int,
        required_points: List[str],
        transcript: str
    ) -> Dict[str, Any]:
        """Constructs structured, practical seminar review feedback."""
        # 1. Point coverage
        covered, partial, missing = cls.check_point_coverage(transcript, required_points)

        # 2. Structure
        structure = cls.analyze_structure(transcript)

        # 3. Fluency & clarity
        fluency_notes, clarity_notes = cls.analyze_fluency_and_clarity(transcript)

        # 4. Grammar
        grammar_issues, issue_count = cls.analyze_grammar(transcript)

        # 5. Vocabulary
        vocab_notes = cls.analyze_vocabulary(transcript)

        # 6. Timing
        timing_status, timing_desc = cls.evaluate_timing(actual_seconds, allowed_seconds)

        # 7. Delivery indicators (Observable facts only, zero mind-reading)
        fillers_count = sum(len(re.findall(r'\b' + re.escape(f) + r'\b', transcript.lower())) for f in cls.COMMON_FILLERS)
        delivery_indicators = []
        if fillers_count > 4:
            delivery_indicators.append(f"Several filler words were noted ({fillers_count} fillers), which can make delivery sound less steady.")
        if structure.get("conclusion") == "Missing or abrupt finish (add a summary or closing thought)":
            delivery_indicators.append("Ending abruptly without a formal concluding sentence can leave the audience hanging.")
        if not delivery_indicators:
            delivery_indicators.append("Delivery appeared steady and focused throughout.")

        # 8. Actionable improvement points
        improvements = []
        if missing:
            improvements.append(f"Include the missing required points: {', '.join(missing[:2])}.")
        if partial:
            improvements.append(f"Expand briefly on partially covered points: {', '.join(partial[:2])}.")
        if structure.get("conclusion") != "Clear and decisive conclusion":
            improvements.append("Add a 1-sentence conclusion summarizing your main message.")
        if issue_count > 0:
            improvements.append(f"Address {issue_count} grammar patterns noted above.")
        if fillers_count > 3:
            improvements.append("Pause silently instead of using filler words when gathering your thoughts.")
        if not improvements:
            improvements.append("Great job! Try delivering the same speech with even more vocal variety and projection.")

        # Format full spoken/written text
        act_str = cls.format_time(int(actual_seconds))
        allow_str = cls.format_time(allowed_seconds)

        lines = [
            "SPEECH REVIEW",
            f"\nTopic:\n{topic}",
            f"\nTime:\n{act_str} / {allow_str} ({timing_status})",
            "\nCONTENT"
        ]

        for pt in covered:
            lines.append(f"✓ {pt}")
        for pt in partial:
            lines.append(f"⚠ {pt} — partially covered")
        for pt in missing:
            lines.append(f"✗ {pt} — missing")

        lines.append(f"\nSTRUCTURE\n- Introduction: {structure['introduction']}\n- Flow: {structure['flow']}\n- Conclusion: {structure['conclusion']}")
        lines.append(f"\nFLUENCY\n" + "\n".join([f"- {n}" for n in fluency_notes]))

        lines.append(f"\nGRAMMAR\n- {issue_count} identifiable issues")
        if grammar_issues:
            first_err = grammar_issues[0]
            lines.append(f"  You said: \"{first_err['original']}\"\n  Better: \"{first_err['correction']}\"")

        lines.append(f"\nCLARITY\n" + "\n".join([f"- {n}" for n in clarity_notes]))
        lines.append(f"\nVOCABULARY\n" + "\n".join([f"- {n}" for n in vocab_notes]))

        lines.append("\nPRONUNCIATION\nPronunciation was not scored because this analysis used the transcript.")

        lines.append("\nNEXT IMPROVEMENTS\n" + "\n".join([f"{i+1}. {imp}" for i, imp in enumerate(improvements[:4])]))
        lines.append("\nWould you like to try the same speech again?")

        formatted_report = "\n".join(lines)

        coverage_eval = cls.evaluate_point_coverage(required_points, transcript)

        analysis_dict = {
            "topic": topic,
            "actual_seconds": actual_seconds,
            "allowed_seconds": allowed_seconds,
            "timing_status": timing_status,
            "timing": {
                "status": timing_status,
                "actual_seconds": actual_seconds,
                "allowed_seconds": allowed_seconds
            },
            "coverage": coverage_eval,
            "covered_points": covered,
            "partial_points": partial,
            "missing_points": missing,
            "structure": structure,
            "clarity": clarity_notes,
            "clarity_notes": clarity_notes,
            "fluency": fluency_notes,
            "fluency_notes": fluency_notes,
            "grammar": grammar_issues,
            "grammar_issues": grammar_issues,
            "grammar_issue_count": issue_count,
            "vocabulary": vocab_notes,
            "vocab_notes": vocab_notes,
            "delivery_indicators": delivery_indicators,
            "improvements": improvements,
            "improvement_points": improvements,
            "formatted_report": formatted_report
        }

        return analysis_dict

    @classmethod
    async def analyze_speech_attempt(
        cls,
        topic: str,
        required_points: List[str],
        time_limit_seconds: int,
        actual_duration_seconds: float,
        transcript: str
    ) -> Dict[str, Any]:
        """Analyzes speech attempt and returns structured analysis (Section 13)."""
        return cls.generate_feedback_report(
            topic=topic,
            actual_seconds=actual_duration_seconds,
            allowed_seconds=time_limit_seconds,
            required_points=required_points,
            transcript=transcript
        )

    @classmethod
    async def process_speech_turn(
        cls,
        student_id: int,
        session_id: str,
        user_text: str,
        duration_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for Speech/Seminar Agent interaction turns.
        Follows state machine:
        SETUP -> READY -> RECORDING -> TRANSCRIBING -> ANALYZING -> FEEDBACK -> OPTIONAL_RETRY
        """
        clean = (user_text or "").strip()
        norm = clean.lower()

        # 1. Fetch or create speech session
        sess = create_or_get_speech_session(student_id, session_id)
        current_state = sess.get("state") or cls.STATE_SETUP
        topic = sess.get("topic") or ""
        time_limit = int(sess.get("time_limit_seconds") or 180)
        required_points = sess.get("required_points") or []
        attempt_num = int(sess.get("attempt_number") or 1)

        # 2. Check for stop/cancel command
        if norm in ["cancel speech", "cancel session", "stop seminar", "exit speech"]:
            update_speech_session(sess["id"], {"state": cls.STATE_SETUP})
            return {
                "reply_text": "Speech practice session cancelled. Let me know when you'd like to try again.",
                "state": cls.STATE_SETUP,
                "provider": "system",
                "is_control": True
            }

        # 3. Handle OPTIONAL_RETRY state commands
        if current_state in [cls.STATE_FEEDBACK, cls.STATE_OPTIONAL_RETRY]:
            if any(p in norm for p in ["try again", "practice again", "same topic", "retry", "one more time"]):
                new_attempt = attempt_num + 1
                update_speech_session(sess["id"], {
                    "state": cls.STATE_READY,
                    "attempt_number": new_attempt,
                    "transcript": "",
                    "actual_duration_seconds": 0.0
                })
                time_str = cls.format_time(time_limit)
                return {
                    "reply_text": f"Awesome! Let's do Attempt #{new_attempt} on '{topic}'. You have {time_str} to cover your {len(required_points)} points. Say 'start' when you're ready.",
                    "state": cls.STATE_READY,
                    "provider": "system"
                }

            if any(p in norm for p in ["new topic", "different topic", "change topic"]):
                update_speech_session(sess["id"], {
                    "state": cls.STATE_SETUP,
                    "topic": "",
                    "required_points": [],
                    "time_limit_seconds": 180,
                    "transcript": "",
                    "attempt_number": 1
                })
                return {
                    "reply_text": "Sure! Let's set up a new speech. What is your topic?",
                    "state": cls.STATE_SETUP,
                    "provider": "system"
                }

            if "show previous feedback" in norm or "previous feedback" in norm:
                analysis = sess.get("analysis_json") or {}
                rep = analysis.get("formatted_report") or "No previous feedback recorded for this attempt."
                return {
                    "reply_text": rep,
                    "state": cls.STATE_FEEDBACK,
                    "provider": "system"
                }

        # 4. State: RECORDING (User is delivering speech)
        if current_state == cls.STATE_RECORDING:
            # Check for early finish command
            finish_commands = ["finish", "i am done", "i'm done", "stop", "submit", "that is all", "that's all", "done"]
            is_finish = any(clean.lower() == cmd or clean.lower().startswith(cmd) for cmd in finish_commands)

            actual_dur = duration_seconds if (duration_seconds and duration_seconds > 0) else float(time_limit)
            transcript_text = clean
            if is_finish:
                # Strip finish word if trailing
                for fc in finish_commands:
                    if transcript_text.lower().endswith(fc):
                        transcript_text = transcript_text[:len(transcript_text) - len(fc)].strip()

            # Transcript quality check (Section 4)
            words = transcript_text.split()
            if len(words) < 4:
                # Do NOT generate fake feedback!
                update_speech_session(sess["id"], {"state": cls.STATE_READY})
                return {
                    "reply_text": "I couldn't get enough speech to analyze this attempt. Please try again.",
                    "state": cls.STATE_READY,
                    "provider": "system"
                }

            # Generate Analysis & Feedback (Section 6 & 8)
            analysis = cls.generate_feedback_report(
                topic=topic,
                actual_seconds=actual_dur,
                allowed_seconds=time_limit,
                required_points=required_points,
                transcript=transcript_text
            )

            # Persist into DB
            update_speech_session(sess["id"], {
                "state": cls.STATE_FEEDBACK,
                "transcript": transcript_text,
                "actual_duration_seconds": actual_dur,
                "analysis_json": analysis,
                "covered_points": analysis["covered_points"],
                "partial_points": analysis["partial_points"],
                "missing_points": analysis["missing_points"],
                "grammar_issue_count": analysis["grammar_issue_count"]
            })

            return {
                "reply_text": analysis["formatted_report"],
                "state": cls.STATE_FEEDBACK,
                "analysis": analysis,
                "provider": "system"
            }

        # 5. State: READY (Waiting for explicit "start" command)
        if current_state == cls.STATE_READY:
            if any(p in norm for p in ["start", "begin", "ready", "i am ready", "i'm ready", "let's go"]):
                update_speech_session(sess["id"], {"state": cls.STATE_RECORDING})
                time_str = cls.format_time(time_limit)
                return {
                    "reply_text": f"Listening now! You have {time_str}. Speak clearly, and say 'finish' or 'I'm done' when you're finished.",
                    "state": cls.STATE_RECORDING,
                    "provider": "system"
                }
            else:
                return {
                    "reply_text": f"All set for '{topic}' ({cls.format_time(time_limit)}). Say 'start' when you're ready to begin speaking.",
                    "state": cls.STATE_READY,
                    "provider": "system"
                }

        # 6. State: SETUP (Collecting topic, required points, time limit)
        # Parse inputs incrementally or all at once
        extracted_topic, extracted_time, extracted_pts = cls.parse_full_setup(clean)

        new_topic = extracted_topic or topic
        new_time = extracted_time or time_limit
        new_pts = extracted_pts or required_points

        # Direct answer if coach just asked for a specific field
        if not new_topic:
            # If input is short and no topic is set, treat whole input as topic
            if len(clean.split()) <= 6 and not extracted_time and not extracted_pts:
                new_topic = clean

        update_speech_session(sess["id"], {
            "topic": new_topic,
            "time_limit_seconds": new_time,
            "required_points": new_pts
        })

        # Check what is still missing
        if not new_topic:
            return {
                "reply_text": "Sure! What is your speech or seminar topic?",
                "state": cls.STATE_SETUP,
                "provider": "system"
            }

        if not new_pts:
            return {
                "reply_text": f"Got it, topic is '{new_topic}'. What important points do you want to cover?",
                "state": cls.STATE_SETUP,
                "provider": "system"
            }

        if not extracted_time and time_limit == 180 and not any(w in norm for w in ["minute", "min", "second", "sec"]):
            # If user hasn't explicitly specified time
            return {
                "reply_text": "How much time do you have for this speech?",
                "state": cls.STATE_SETUP,
                "provider": "system"
            }

        # All 3 pieces collected! Transition to READY
        update_speech_session(sess["id"], {"state": cls.STATE_READY})
        time_str = cls.format_time(new_time)
        pts_list = "\n".join([f"{i+1}. {p}" for i, p in enumerate(new_pts)])

        ready_msg = (
            f"Got it. Topic: {new_topic}.\n"
            f"You have {time_str} and need to cover {len(new_pts)} points:\n"
            f"{pts_list}\n\n"
            f"Say 'start' when you're ready."
        )
        return {
            "reply_text": ready_msg,
            "state": cls.STATE_READY,
            "provider": "system"
        }

    @classmethod
    def setup_speech_session(
        cls,
        student_id: int,
        topic: str,
        required_points: List[str],
        allowed_duration_seconds: int = 180
    ) -> Dict[str, Any]:
        """Sets up a speech rehearsal session."""
        sess = cls.get_or_create_session(student_id=student_id, force_new=True)
        pts_json = json.dumps(required_points) if required_points else "[]"
        update_speech_session(
            sess["id"],
            {
                "topic": topic,
                "required_points": pts_json,
                "time_limit_seconds": allowed_duration_seconds,
                "state": cls.STATE_READY
            }
        )
        return get_speech_session_by_id(sess["id"])

    @classmethod
    def get_active_session_state(cls, student_id: int) -> Dict[str, Any]:
        """Returns the active speech session parameters."""
        sess = get_active_speech_session(student_id=student_id)
        if not sess:
            return {"topic": None, "required_points": [], "time_limit_seconds": 180, "state": cls.STATE_SETUP}
        pts = json.loads(sess.get("required_points") or "[]") if isinstance(sess.get("required_points"), str) else (sess.get("required_points") or [])
        return {
            "topic": sess.get("topic"),
            "required_points": pts,
            "time_limit_seconds": sess.get("time_limit_seconds") or 180,
            "state": sess.get("state") or cls.STATE_SETUP
        }

    @classmethod
    def clear_speech_session(cls, student_id: int) -> bool:
        """Clears/finishes the active speech session so no stale context lingers."""
        sess = get_active_speech_session(student_id=student_id)
        if sess:
            update_speech_session(sess["id"], {"state": "COMPLETED", "topic": None, "required_points": "[]"})
            return True
        return False


speech_service = SpeechPracticeService()
