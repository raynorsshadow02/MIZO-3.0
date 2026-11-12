"""
Mizo Context & Topic Tracking Service
=====================================
Implements:
1. Strict separation of Long-Term Learner Profile, Current Conversation Context, and Current Mode.
2. Lightweight, deterministic conversation intent detection (zero LLM overhead).
3. Topic, subtopic, and entity tracking across conversation turns.
4. User correction detection and context reconsiderations.
5. Selective memory retrieval (only inject memories relevant to current topic).
6. Durable memory creation (only persist explicitly stated personal facts).
7. Context continuity ('What were we talking about?' vs 'Do you remember my favorite character?').
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from app.db.database import (
    get_student,
    update_student,
    get_session_context,
    update_session_context,
    get_prototype_student,
    PROTOTYPE_STUDENT_ID
)

logger = logging.getLogger("mizo.context")


# ---------------------------------------------------------------------------
# 1. Lightweight Rule-Based Intent Detection (Zero LLM)
# ---------------------------------------------------------------------------
CORRECTION_PATTERNS = [
    r"\byou forgot\b",
    r"\bthat'?s wrong\b",
    r"\byou(?:'?re| are) wrong\b",
    r"\bno[,\s]+(?:i meant|i was asking|not that|actually)\b",
    r"\byou missed\b",
    r"\bthat'?s not (?:correct|right|what i meant|what i said)\b",
    r"\bi was asking\b",
    r"\bno\s+no\b",
    r"\bi didn'?t say that\b",
    r"\byou left out\b",
    r"\bactually[,\s]+(?:it'?s|i meant|he is|she is|they are)\b"
]

CONFUSION_PATTERNS = [
    r"\bi'?m confused\b",
    r"\bi don'?t (?:get|understand|follow) (?:it|that|this)\b",
    r"\bwhat do you mean\b",
    r"\byou lost me\b",
    r"\bi am lost\b",
    r"^\s*(?:huh|what\?|pardon\?)\s*$"
]

CLARIFICATION_PATTERNS = [
    r"\bcould you clarify\b",
    r"\bcan you explain (?:that|again)\b",
    r"\bwhat does that mean\b",
    r"\bexplain more\b",
    r"\bwhat exactly\b"
]

TOPIC_CHANGE_PATTERNS = [
    r"\blet'?s talk about something else\b",
    r"\bchange (?:the )?topic\b",
    r"\bnew topic\b",
    r"\bswitch topic\b",
    r"\benough about (?:that|this)\b",
    r"\bby the way[,\s]+can we talk about\b",
    r"^\s*(?:i want to |let'?s |can we )?(?:move on|proceed|move ahead)\s*[.!?]?$",
    r"\bmove on from\b",
    r"\bdifferent topic\b"
]

MOVE_ON_PATTERNS = [
    r"^\s*(?:i want to |let'?s |can we )?(?:move on|proceed|move ahead)\s*[.!?]?$"
]

PROJECT_QUERY_PATTERNS = [
    r"\b(?:what are my projects|what is my project|tell me about my projects|my projects|what projects am i working on)\b"
]

TEACH_REQUEST_PATTERNS = [
    r"^\s*(?:teach me|can you teach me|please teach me|teach me more|start teaching)\b"
]

COURTESY_PATTERNS = [
    r"^\s*(?:thank you|thanks|thanks a lot|thank you so much|thx)[.!?]?$",
    r"^\s*(?:goodbye|bye|see you|good night)[.!?]?$"
]

GREETING_PATTERNS = [
    r"^\s*(?:hello|hi|hey|good morning|good afternoon|good evening)[.!?]?$"
]

CONTEXT_QUERY_PATTERNS = [
    r"\bwhat were we talking about\b",
    r"\bwhat was (?:our|the) topic\b",
    r"\bwhat did we (?:just )?(?:talk|speak|discuss) about\b",
    r"\bwhere were we\b",
    r"\bwhat was i saying\b"
]

MEMORY_QUERY_PATTERNS = [
    r"\bdo you remember my\b",
    r"\bwhat is my favorite\b",
    r"\bwhats my favorite\b",
    r"\bdo you know my favorite\b",
    r"\bdo you remember (?:what|who) i (?:like|love)\b",
    r"\bwhat did i say my favorite\b"
]

AGREEMENT_PATTERNS = [
    r"^\s*(?:yes|yeah|yep|yup|exactly|agreed|i agree|right|correct|sure|definitely|absolutely)\b"
]

DISAGREEMENT_PATTERNS = [
    r"^\s*(?:no|nope|nah|i disagree|not really|disagree|never)\b"
]

QUESTION_STARTERS = [
    r"^\s*(?:who|what|where|when|why|how|which|whose|whom)\b",
    r"^\s*(?:can|could|would|will|do|does|did|is|are|was|were|have|has|should)\s+(?:you|i|we|they|there|it)\b"
]


def detect_conversation_intent(text: str, prev_mizo_question: Optional[str] = None) -> str:
    """
    Lightweight rule-based intent detection:
    - USER_CORRECTION
    - CONTEXT_QUERY
    - MEMORY_QUERY
    - PROJECTS_QUERY
    - MOVE_ON_QUERY
    - TEACH_REQUEST
    - COURTESY
    - GREETING
    - CONFUSION
    - CLARIFICATION
    - TOPIC_CHANGE
    - QUESTION
    - AGREEMENT
    - DISAGREEMENT
    - ANSWER
    - CASUAL_CONVERSATION
    """
    if not text or not text.strip():
        return "UNKNOWN"

    norm = text.strip().lower()

    # 1. Corrections (Section 23)
    for pat in CORRECTION_PATTERNS:
        if re.search(pat, norm):
            return "USER_CORRECTION"

    # 2. Context query ("What were we talking about?") (Section 29)
    for pat in CONTEXT_QUERY_PATTERNS:
        if re.search(pat, norm):
            return "CONTEXT_QUERY"

    # 3. Memory query ("Do you remember my favorite character?") (Section 29)
    for pat in MEMORY_QUERY_PATTERNS:
        if re.search(pat, norm):
            return "MEMORY_QUERY"

    # 4. Project queries ("What are my projects?")
    for pat in PROJECT_QUERY_PATTERNS:
        if re.search(pat, norm):
            return "PROJECTS_QUERY"

    # 5. Ambiguous move on ("move on")
    for pat in MOVE_ON_PATTERNS:
        if re.search(pat, norm):
            return "MOVE_ON_QUERY"

    # 6. Teach request ("teach me")
    for pat in TEACH_REQUEST_PATTERNS:
        if re.search(pat, norm):
            return "TEACH_REQUEST"

    # 7. Courtesy (Thanks, Bye)
    for pat in COURTESY_PATTERNS:
        if re.search(pat, norm):
            return "COURTESY"

    # 8. Greetings (Hello, Hi)
    for pat in GREETING_PATTERNS:
        if re.search(pat, norm):
            return "GREETING"

    # 9. Confusion
    for pat in CONFUSION_PATTERNS:
        if re.search(pat, norm):
            return "CONFUSION"

    # 10. Clarification
    for pat in CLARIFICATION_PATTERNS:
        if re.search(pat, norm):
            return "CLARIFICATION"

    # 11. Topic change
    for pat in TOPIC_CHANGE_PATTERNS:
        if re.search(pat, norm):
            return "TOPIC_CHANGE"

    # 12. Questions
    if norm.endswith("?"):
        return "QUESTION"
    for pat in QUESTION_STARTERS:
        if re.search(pat, norm):
            return "QUESTION"

    # 13. Agreement / Disagreement
    for pat in AGREEMENT_PATTERNS:
        if re.search(pat, norm):
            return "AGREEMENT"
    for pat in DISAGREEMENT_PATTERNS:
        if re.search(pat, norm):
            return "DISAGREEMENT"

    # 14. Answer to previous Mizo question
    if prev_mizo_question and prev_mizo_question.strip():
        return "ANSWER"

    return "CASUAL_CONVERSATION"



# ---------------------------------------------------------------------------
# 2. Durable Memory Creation (Section 28)
# ---------------------------------------------------------------------------
DURABLE_PATTERNS = [
    # "My favorite character is Zoro"
    (r"\bmy favorite character is\s+([A-Za-z0-9\s\-]+)", "favorite_things", "favorite_character"),
    # "My favorite anime is One Piece"
    (r"\bmy favorite anime is\s+([A-Za-z0-9\s\-]+)", "favorite_things", "favorite_anime"),
    # "My favorite food is chicken"
    (r"\bmy favorite food is\s+([A-Za-z0-9\s\-]+)", "favorite_things", "favorite_food"),
    # "My favorite color is blue"
    (r"\bmy favorite color is\s+([A-Za-z0-9\s\-]+)", "favorite_things", "favorite_color"),
    # "My favorite book is Harry Potter"
    (r"\bmy favorite book is\s+([A-Za-z0-9\s\-]+)", "favorite_things", "favorite_book"),
    # "My favorite hobby is robotics"
    (r"\bmy favorite hobby is\s+([A-Za-z0-9\s\-]+)", "favorite_things", "favorite_hobby"),
    # "Please remember that I like ..."
    (r"\b(?:please )?remember that i (?:like|love|study|work as|am a)\s+([A-Za-z0-9\s\-]+)", "durable_memories", "general_fact")
]


def extract_durable_facts(text: str) -> Optional[Tuple[str, str, str]]:
    """
    Extracts durable personal facts when explicitly stated by user.
    Returns (category, key, value) or None.
    Does NOT store casual remarks like 'I think Zoro is cool'.
    """
    if not text or not text.strip():
        return None

    clean = text.strip()
    clean_lower = clean.lower()

    for pattern, cat, key in DURABLE_PATTERNS:
        m = re.search(pattern, clean_lower, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            # Clean trailing punctuation
            val = re.sub(r"[.,!?;]+$", "", val).strip()
            if len(val) >= 2:
                # Capitalize words appropriately
                val_cap = " ".join([w.capitalize() for w in val.split()])
                return (cat, key, val_cap)

    return None


# ---------------------------------------------------------------------------
# 3. Topic & Entity Tracking (Section 22)
# ---------------------------------------------------------------------------
KNOWN_TOPIC_DOMAINS = {
    "anime": {
        "subtopics": {
            "One Piece": [
                "one piece", "luffy", "zoro", "nami", "sanji", "usopp", "chopper",
                "robin", "franky", "brook", "jinbe", "straw hat", "straw hats",
                "straw hat pirates", "going merry", "thousand sunny", "grand line", "devil fruit"
            ],
            "Naruto": ["naruto", "sasuke", "sakura", "kakashi", "hokage", "leaf village", "ninja"],
            "Dragon Ball": ["dragon ball", "goku", "vegeta", "saiyan", "kamehameha"],
            "Attack on Titan": ["attack on titan", "eren", "mikasa", "armin", "titan", "scout"]
        },
        "general_keywords": ["anime", "manga", "character", "otaku", "favorite character", "crew members", "crew"]
    },
    "cooking": {
        "subtopics": {
            "recipe preparation": ["recipe", "prepare", "preparing", "vegetables", "sauce", "cook", "cooking"],
            "ingredients": ["chicken breast", "chicken", "meat", "kg", "grams", "garlic", "onion", "spices"]
        },
        "general_keywords": ["food", "meal", "dinner", "lunch", "kitchen", "cook"]
    },
    "robotics": {
        "subtopics": {
            "Mizo robot": ["mizo", "robot", "esp32", "hardware", "microcontroller"],
            "Artificial Intelligence": ["ai", "machine learning", "neural network", "deep learning", "llm"]
        },
        "general_keywords": ["robotics", "sensors", "electronics", "motor"]
    },
    "physics_and_engineering": {
        "subtopics": {
            "Kinematics": [
                "kinematics", "forward kinematics", "inverse kinematics", "degrees of freedom",
                "end-effector", "joint angles", "jacobian", "motion planning", "singularity", "rigid body motion"
            ],
            "Dynamics": [
                "dynamics", "newton euler", "lagrangian dynamics", "torque", "forces", "inertia"
            ],
            "Mechanics": [
                "mechanics", "velocity", "acceleration", "gravity", "momentum", "friction"
            ]
        },
        "general_keywords": ["kinematics", "physics", "mechanics", "engineering", "motion", "joints"]
    },
    "learner_projects": {
        "subtopics": {
            "Projects": [
                "project", "projects", "my project", "my projects", "what are my projects", "esp32 robot"
            ]
        },
        "general_keywords": ["project", "projects", "portfolio", "coursework"]
    },
    "english_learning": {
        "subtopics": {
            "grammar practice": ["grammar", "tense", "past tense", "preposition"],
            "speaking skills": ["presentation", "fluency", "pronunciation", "accent"]
        },
        "general_keywords": ["english", "lesson", "practice", "speak english"]
    }
}


def extract_topic_and_entities(
    text: str,
    current_topic: str = "",
    current_subtopic: str = "",
    current_entities: Optional[List[str]] = None
) -> Tuple[str, str, List[str]]:
    """
    Extracts or maintains topic, subtopic, and entities from user utterance.
    - Explicit requests (e.g. kinematics, projects) switch topics immediately.
    - Previous topics (e.g. One Piece) do not persist into unrelated new questions.
    - Ambiguous 'move on' or courtesy phrases ('thank you') do not keep the old topic locked.
    """
    clean_lower = text.lower().strip()
    entities = list(current_entities or [])

    # Check for ambiguous 'move on' or explicit topic change request
    is_move_on = any(re.search(pat, clean_lower) for pat in MOVE_ON_PATTERNS)
    is_topic_change = is_move_on or any(re.search(pat, clean_lower) for pat in TOPIC_CHANGE_PATTERNS)
    if is_topic_change:
        return "", "", []

    # Check for standalone courtesy (e.g. "Thank you", "Thanks")
    is_courtesy = any(re.search(pat, clean_lower) for pat in COURTESY_PATTERNS)
    if is_courtesy:
        # Courtesies do not force or revive an old topic
        return "", "", []

    # Check for "teach me", "continue", "go on", "tell me more"
    is_teach = any(re.search(pat, clean_lower) for pat in TEACH_REQUEST_PATTERNS)
    is_continue = any(p in clean_lower for p in ["continue", "go on", "keep going", "next step", "whats next", "what's next", "step 2", "more details", "tell me more", "explain more", "teach me more"])
    if is_teach or is_continue:
        # If the preceding subtopic was an academic/educational topic, preserve it for continued teaching!
        if current_subtopic and any(k in current_subtopic.lower() for k in ["kinematics", "dynamics", "mechanics", "robot", "physics", "engineering", "ai", "artificial intelligence", "grammar", "speaking"]):
            return current_topic, current_subtopic, entities
        if is_teach:
            # If previous topic was anime or entertainment, clear it so tutor can ask for a subject
            return "", "", []

    # Check for explicit match in known domains
    matched_domain = ""
    matched_subtopic = ""
    matched_entity = ""

    for top, data in KNOWN_TOPIC_DOMAINS.items():
        subtopics = data.get("subtopics", {})
        for sub, kws in subtopics.items():
            for kw in kws:
                if re.search(r"\b" + re.escape(kw) + r"\b", clean_lower):
                    matched_domain = top
                    matched_subtopic = sub
                    matched_entity = " ".join(w.capitalize() for w in kw.split())
                    break
            if matched_domain:
                break
        if matched_domain:
            break

    if matched_domain:
        # Switching domain or establishing domain
        if matched_domain != current_topic:
            # Complete domain switch: discard old entities!
            entities = [matched_entity] if matched_entity else []
        else:
            if matched_entity and matched_entity not in entities:
                entities.append(matched_entity)
        detected_topic = matched_domain
        detected_subtopic = matched_subtopic
    else:
        detected_topic = current_topic
        detected_subtopic = current_subtopic

    # If within One Piece discussion, extract characters/crew
    if detected_subtopic == "One Piece":
        one_piece_entities = {
            "zoro": "Zoro", "luffy": "Luffy", "jinbe": "Jinbe", "nami": "Nami",
            "sanji": "Sanji", "chopper": "Chopper", "straw hat pirates": "Straw Hat Pirates",
            "straw hats": "Straw Hat Pirates", "straw hat": "Straw Hat Pirates",
            "crew": "Straw Hat Pirates", "crew members": "Straw Hat Pirates"
        }
        for k, v in one_piece_entities.items():
            if re.search(r"\b" + re.escape(k) + r"\b", clean_lower):
                if v not in entities:
                    entities.append(v)
        if "Straw Hat" in entities and "Straw Hat Pirates" in entities:
            entities.remove("Straw Hat")

    if detected_topic == "cooking":
        cooking_entities = {
            "chicken breast": "Chicken Breast", "vegetables": "Vegetables",
            "sauce": "Sauce", "chicken": "Chicken"
        }
        for k, v in cooking_entities.items():
            if re.search(r"\b" + re.escape(k) + r"\b", clean_lower):
                if v not in entities:
                    entities.append(v)

    # If the user asks a completely unrelated question without matching keywords
    # and the previous topic was anime, don't force One Piece if it's an explicit general/technical question
    if current_subtopic == "One Piece" and not matched_domain:
        unrelated_question = any(q in clean_lower for q in ["what are my", "tell me about", "explain", "how do", "what is", "why is"])
        if unrelated_question:
            return "", "", []

    return detected_topic, detected_subtopic, entities[-6:]



def update_conversation_topic_state(
    session_id: str,
    student_id: int,
    user_text: str,
    last_mizo_question: str = ""
) -> Dict[str, Any]:
    """
    Updates the conversation topic state after every meaningful user message:
    - topic
    - subtopic
    - entities
    - last user intent
    - last Mizo question
    - last user correction
    - conversation goal
    Also persists durable facts if detected.
    """
    prev_context = get_session_context(session_id)
    curr_topic = prev_context.get("current_topic", "")
    curr_subtopic = prev_context.get("current_subtopic", "")
    curr_entities = prev_context.get("current_entities", [])

    intent = detect_conversation_intent(user_text, prev_mizo_question=last_mizo_question)

    new_topic, new_subtopic, new_entities = extract_topic_and_entities(
        user_text,
        current_topic=curr_topic,
        current_subtopic=curr_subtopic,
        current_entities=curr_entities
    )

    last_correction = prev_context.get("last_user_correction", "")
    if intent == "USER_CORRECTION":
        last_correction = user_text

    # Extract durable fact if present
    durable_fact = extract_durable_facts(user_text)
    if durable_fact:
        cat, key, val = durable_fact
        student = get_student(student_id) or {}
        if cat == "favorite_things":
            favs = dict(student.get("favorite_things", {}) or {})
            favs[key] = val
            update_student(student_id, {"favorite_things": favs})
            logger.info(f"[DURABLE_MEMORY] Saved favorite: {key} = {val} for student {student_id}")
        elif cat == "durable_memories":
            durs = list(student.get("durable_memories", []) or [])
            if val not in durs:
                durs.append(val)
                update_student(student_id, {"durable_memories": durs})
                logger.info(f"[DURABLE_MEMORY] Saved durable fact: {val} for student {student_id}")

        # Also persist to memories table
        try:
            from app.db.database import save_durable_memory
            save_durable_memory(
                student_id=student_id,
                memory_text=f"{key}: {val}" if cat == "favorite_things" else val,
                category=cat,
                session_id=session_id
            )
        except Exception as me:
            logger.debug(f"Failed to persist to memories table: {me}")

    context_updates = {
        "current_topic": new_topic,
        "current_subtopic": new_subtopic,
        "current_entities": new_entities,
        "last_user_intent": intent,
        "last_mizo_question": last_mizo_question,
        "last_user_correction": last_correction,
    }

    update_session_context(session_id, context_updates)

    return {
        "current_topic": new_topic,
        "current_subtopic": new_subtopic,
        "current_entities": new_entities,
        "last_user_intent": intent,
        "last_mizo_question": last_mizo_question,
        "last_user_correction": last_correction,
        "conversation_goal": prev_context.get("conversation_goal", "")
    }


# ---------------------------------------------------------------------------
# 4. Selective Memory Retrieval (Section 27 & 21)
# ---------------------------------------------------------------------------
def filter_relevant_memories(
    student: Dict[str, Any],
    current_topic: str,
    current_subtopic: str = "",
    user_message: str = ""
) -> Dict[str, Any]:
    """
    Retrieves only durable memories relevant to the current conversation topic and user request.
    CRITICAL: Current conversation topic takes absolute priority over unrelated
    long-term profile interests (e.g. if topic is One Piece, robotics/ESP32 is excluded;
    if user asks about kinematics or projects, One Piece anime is excluded).
    """
    topic_clean = (current_topic or "").lower()
    subtopic_clean = (current_subtopic or "").lower()
    msg_clean = (user_message or "").lower()

    fav_things = student.get("favorite_things", {}) or {}
    all_durable_mems = student.get("durable_memories", []) or []

    relevant_favorites: Dict[str, Any] = {}
    relevant_interests: List[str] = []
    relevant_durable_memories: List[str] = []

    # Check if request or current topic is technical / robotics / kinematics / projects / engineering
    is_technical = (
        "robotics" in topic_clean or "ai" in topic_clean or "physics" in topic_clean or
        "engineering" in topic_clean or "control" in topic_clean or "science" in topic_clean or
        "kinematics" in subtopic_clean or "kinematics" in topic_clean or "learner_projects" in topic_clean or
        "projects" in subtopic_clean or "control" in subtopic_clean or
        any(k in msg_clean for k in ["kinematics", "robot", "robotics", "project", "projects", "physics", "engineering", "esp32", "course", "pid", "controller", "dynamics", "mechanics"])
    )

    # Check if request or current topic is anime
    is_anime = not is_technical and (
        "anime" in topic_clean or "one piece" in subtopic_clean or "one piece" in topic_clean or
        any(k in msg_clean for k in ["anime", "one piece", "luffy", "zoro", "manga", "straw hat", "straw hats"])
    )

    # Check if cooking
    is_cooking = not is_technical and not is_anime and (
        "cooking" in topic_clean or "food" in topic_clean or
        any(k in msg_clean for k in ["cook", "cooking", "recipe", "ingredients", "dinner", "lunch"])
    )

    if is_technical:
        # Technical topics: include robotics hobby, exclude anime character favorites!
        if "favorite_hobby" in fav_things:
            relevant_favorites["favorite_hobby"] = fav_things["favorite_hobby"]
        all_interests = student.get("interests", "") or ""
        relevant_interests = [i.strip() for i in all_interests.split(",") if any(k in i.lower() for k in ["robot", "ai", "tech", "science", "engineering"])]
        if not relevant_interests:
            relevant_interests = ["Robotics", "Engineering"]

        # Only inject relevant technical / project memories
        for m in all_durable_mems:
            m_low = str(m).lower()
            if any(k in m_low for k in ["robot", "esp32", "project", "engineering", "kinematics", "study"]):
                relevant_durable_memories.append(m)

    elif is_anime:
        # Anime topics: include anime character/anime favorites, exclude robotics/ESP32!
        if "favorite_character" in fav_things:
            relevant_favorites["favorite_character"] = fav_things["favorite_character"]
        if "favorite_anime" in fav_things:
            relevant_favorites["favorite_anime"] = fav_things["favorite_anime"]
        relevant_interests = ["Anime", "Storytelling"]

        for m in all_durable_mems:
            m_low = str(m).lower()
            if any(k in m_low for k in ["anime", "one piece", "manga", "zoro", "luffy"]):
                relevant_durable_memories.append(m)

    elif is_cooking:
        if "favorite_food" in fav_things:
            relevant_favorites["favorite_food"] = fav_things["favorite_food"]
        relevant_interests = ["Cooking", "Food"]

        for m in all_durable_mems:
            m_low = str(m).lower()
            if any(k in m_low for k in ["cook", "food", "recipe", "kitchen"]):
                relevant_durable_memories.append(m)

    else:
        # General/unspecified topic: pass general favorites if any
        relevant_favorites = fav_things
        relevant_interests = [i.strip() for i in (student.get("interests") or "").split(",") if i.strip()]
        relevant_durable_memories = all_durable_mems[:5]

    return {
        "student_name": student.get("name", "Student"),
        "education": student.get("education", "Student"),
        "target_level": student.get("target_level", "Intermediate"),
        "current_teaching_difficulty": student.get("current_teaching_difficulty", 1),
        "relevant_favorites": relevant_favorites,
        "relevant_interests": relevant_interests,
        "durable_memories": relevant_durable_memories
    }

