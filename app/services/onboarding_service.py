"""
Mizo Onboarding Service
======================
Handles the student onboarding flow and shared learner profile lifecycle.
Requirements:
1. Onboarding happens only before the main system becomes active.
2. After onboarding is complete: ONBOARDING_COMPLETE = true.
3. Do not restart onboarding during normal conversations.
4. Shared learner profile memory (Name, goals, level) is accessible by all agents.
5. Profile / Name changes require explicit confirmation:
   e.g. Current: Shahid, User: "Change my name to Migaza."
   Response: "Your current name is Shahid. You asked me to change it to Migaza. Should I replace your name and call you Migaza?"
   Only updates after user confirms.
"""

import re
from typing import Dict, Any, Optional, Tuple
from app.db.database import (
    get_student,
    update_student,
    PROTOTYPE_STUDENT_ID,
    get_prototype_student
)
from app.services.personalization_service import personalization_service


class OnboardingService:
    """
    Dedicated Onboarding & Learner Profile Service.
    """

    # Track pending name changes per session: {session_id: new_candidate_name}
    _pending_name_changes: Dict[str, str] = {}

    @classmethod
    def is_onboarded(cls, student_id: int = PROTOTYPE_STUDENT_ID) -> bool:
        """Checks if the learner has completed initial onboarding."""
        st = get_student(student_id) or get_prototype_student()
        if not st:
            return False
        return bool(st.get("onboarding_completed", False))

    @classmethod
    def get_learner_profile(cls, student_id: int = PROTOTYPE_STUDENT_ID) -> Dict[str, Any]:
        """Returns shared profile information (accessible across all agent sections)."""
        st = get_student(student_id) or get_prototype_student() or {}
        return {
            "id": st.get("id", student_id),
            "name": st.get("name") or "Learner",
            "grade": st.get("grade") or "",
            "target_level": st.get("target_level") or "Intermediate",
            "goals": st.get("goals") or "",
            "interests": st.get("interests") or "",
            "onboarding_completed": bool(st.get("onboarding_completed", False)),
            "onboarding_step": st.get("onboarding_step") or "completed"
        }

    @classmethod
    def detect_name_change_intent(cls, user_text: str) -> Optional[str]:
        """Detects if user is asking to change their existing profile name."""
        norm = user_text.strip()
        # Patterns like: "change my name to Migaza", "update my name to Migaza", "call me Migaza from now on", "my new name is Migaza"
        patterns = [
            r"\bchange my name to\s+([A-Za-z0-9_-]+)",
            r"\bupdate my name to\s+([A-Za-z0-9_-]+)",
            r"\bcall me\s+([A-Za-z0-9_-]+)\s*(?:from now on|instead)?",
            r"\bmy new name is\s+([A-Za-z0-9_-]+)",
            r"\bplease change my name to\s+([A-Za-z0-9_-]+)"
        ]
        for pat in patterns:
            m = re.search(pat, norm, re.IGNORECASE)
            if m:
                cand = m.group(1).strip().capitalize()
                if cand.lower() not in ["mizo", "robot", "tutor", "student", "learner", "stop", "cancel"]:
                    return cand
        return None

    @classmethod
    def handle_name_change_flow(
        cls,
        student_id: int,
        session_id: str,
        user_text: str
    ) -> Optional[str]:
        """
        Handles explicit name change confirmation flow.
        Returns response string if a confirmation turn or proposal occurred, else None.
        """
        student = get_student(student_id) or get_prototype_student() or {}
        curr_name = student.get("name") or "Learner"
        norm = user_text.lower().strip()

        # 1. Check if there's an ongoing pending confirmation
        pending_name = cls._pending_name_changes.get(session_id)
        if pending_name:
            # Check for confirmation
            is_confirm = any(re.search(p, norm) for p in [r"\byes\b", r"\byep\b", r"\byeah\b", r"\bconfirm\b", r"\bsure\b", r"\bcorrect\b", r"\bplease do\b"])
            is_reject = any(re.search(p, norm) for p in [r"\bno\b", r"\bnever\s*mind\b", r"\bcancel\b", r"\bdon'?t\b", r"\bkeep\b"])

            if is_confirm:
                cls._pending_name_changes.pop(session_id, None)
                update_student(student_id, {"name": pending_name})
                return f"Got it! I have updated your name to {pending_name}. What would you like to focus on today?"
            elif is_reject:
                cls._pending_name_changes.pop(session_id, None)
                return f"Alright, I'll keep calling you {curr_name}."

        # 2. Check for fresh name change request
        new_name_candidate = cls.detect_name_change_intent(user_text)
        if new_name_candidate and new_name_candidate.lower() != curr_name.lower():
            cls._pending_name_changes[session_id] = new_name_candidate
            return f"Your current name is {curr_name}. You asked me to change it to {new_name_candidate}. Should I replace your name and call you {new_name_candidate}?"

        return None


onboarding_service = OnboardingService()
