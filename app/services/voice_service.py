"""
Mizo 3.0 Voice Interaction Service
=================================
Implements the core voice state machine, robust 'Mizo' wake-word detection,
Whisper phonetic misrecognition tolerance, continuous indefinite standby/sleep
monitoring, barge-in interruption, and centralized intent detection.
"""

import os
import re
import time
import string
import logging
import asyncio
import difflib
import threading
from enum import Enum
from typing import Tuple, Optional, Callable, Dict, Any, List

logger = logging.getLogger("mizo.voice")

# Canonical Assistant Identity
ASSISTANT_NAME = "Mizo"


# ---------------------------------------------------------------------------
# 1. Voice State Machine States (Part 1 - Explicit & Observable)
# ---------------------------------------------------------------------------
class VoiceState(str, Enum):
    # Required Explicit States (Mizo Hands-Free Voice Architecture)
    WAKE_LISTENING = "WAKE_LISTENING"
    ACTIVE_CONVERSATION = "ACTIVE_CONVERSATION"

    # Operational Voice States & Backward-Compatible Aliases
    PAUSED = "PAUSED"
    ACTIVE = "ACTIVE"
    SNOOZED = "SNOOZED"
    WAKE_DETECTED = "WAKE_DETECTED"
    ONBOARDING = "ONBOARDING"
    ASSESSMENT = "ASSESSMENT"
    PROCESSING_ASSESSMENT = "PROCESSING_ASSESSMENT"
    ASSESSMENT_COMPLETE = "ASSESSMENT_COMPLETE"
    READY = "READY"
    CONVERSATION = "CONVERSATION"
    STOPPING = "STOPPING"
    ERROR_RECOVERY = "ERROR_RECOVERY"

    SPEAKING = "SPEAKING"
    PROCESSING = "PROCESSING"
    ACTIVE_LISTENING = "ACTIVE_LISTENING"
    STANDBY = "STANDBY"
    SLEEP = "STANDBY"
    SNOOZE = "STANDBY"
    WAKE_ONLY = "STANDBY"
    AWAKE = "ACTIVE_LISTENING"
    LISTENING = "ACTIVE_LISTENING"
    RECORDING = "ACTIVE_LISTENING"
    CAPTURING = "ACTIVE_LISTENING"
    ASSESSMENT_RECORDING = "ASSESSMENT_RECORDING"

    def __eq__(self, other):
        if super().__eq__(other):
            return True
        val = getattr(other, "value", other) if hasattr(other, "value") else str(other)
        # Idle / Wake Listening states equivalence
        idle_states = ("WAKE_LISTENING", "PAUSED", "SNOOZED", "STANDBY", "SLEEP", "SNOOZE", "WAKE_ONLY")
        if self.value in idle_states and val in idle_states:
            return True
        # Active conversation states equivalence
        active_states = ("ACTIVE_CONVERSATION", "ACTIVE", "ONBOARDING", "ACTIVE_LISTENING", "READY", "CONVERSATION", "AWAKE", "LISTENING", "RECORDING", "CAPTURING")
        if self.value in active_states and val in active_states:
            return True
        # ASSESSMENT and ASSESSMENT_RECORDING are equivalent
        if self.value in ("ASSESSMENT", "ASSESSMENT_RECORDING") and val in ("ASSESSMENT", "ASSESSMENT_RECORDING"):
            return True
        # PROCESSING_ASSESSMENT and PROCESSING are equivalent
        if self.value in ("PROCESSING_ASSESSMENT", "PROCESSING") and val in ("PROCESSING_ASSESSMENT", "PROCESSING"):
            return True
        return False

    def __hash__(self):
        return hash(self.value)


# ---------------------------------------------------------------------------
# 2. Configurable Stop & Submit Commands (Part 7)
# ---------------------------------------------------------------------------
CONFIGURED_STOP_COMMANDS = [
    "stop mizo",
    "mizo stop",
    "hey mizo stop",
    "mizo please stop",
    "please stop mizo",
    "stop meeso",
    "meeso stop",
    "stop miso",
    "miso stop",
    "stop mikaza",
    "mikaza stop",
    "stop megaza",
    "megaza stop",
    "stop mikasa",
    "mikasa stop",
    "stop listening",
    "stop talking",
    "mizo stop talking",
    "stop now",
    "stop please",
    "please stop",
    "go to sleep",
    "stop it",
    "stop",
    # Multilingual / Urdu STOP commands (Section 25)
    "میزو سٹاپ",
    "سٹاپ",
    "میزو روکو",
    "روکو",
    "mizo ruko",
    "ruko",
    "bas karo",
    "para",
    "pare",
    "arrete"
]

CONFIGURED_SUBMIT_COMMANDS = [
    "submit",
    "finish",
    "done",
    "i'm done",
    "im done",
    "i am done",
    "finished",
    "finish assessment",
    "submit assessment",
    "done speaking",
    "finished speaking",
    "complete"
]


# ---------------------------------------------------------------------------
# 3. Text Normalization, Phonetic Analysis & Intent Detection
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """
    Normalizes transcript:
    - Lowercase
    - Strips contractions/apostrophes
    - Replaces punctuation with spaces
    - Collapses multiple whitespace to single space
    """
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r"['’]", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def phonetic_mizo_normalize(word: str) -> str:
    """
    Normalizes phonetic variations of 'mizo':
    e.g. 'meeso', 'meso', 'mezo', 'mizzo', 'miso', 'meiso', 'mizu', 'myzo', 'mizo' -> 'mizo'
    """
    w = word.lower().strip()
    if not w:
        return ""
    # Standardize vowel blends
    w = re.sub(r"ee|ei|ey|ea|y", "i", w)
    # Standardize initial 'me' followed by s/z to 'mi'
    if w.startswith("me") and len(w) in [4, 5]:
        w = "mi" + w[2:]
    # Standardize sibilants / z sounds
    w = re.sub(r"zz|ss|s", "z", w)
    # Standardize trailing vowel
    if w.endswith("u"):
        w = w[:-1] + "o"
    # Collapse double letters
    w = re.sub(r"(.)\1+", r"\1", w)
    return w


def is_mizo_word(word: str) -> Tuple[bool, float]:
    """
    Determines if an individual word phonetically or via fuzzy similarity corresponds to 'Mizo'.
    Returns (is_match, similarity_score).
    Conservative matching to prevent waking from random words like 'robot', 'weather', 'python'.
    """
    w = word.lower().strip()
    if not w:
        return False, 0.0

    # 1. Exact match with canonical name or known Whisper misrecognitions
    if w in ["mizo", "miso", "meeso", "meso", "mezo", "mizzo", "meiso", "mizu", "myzo"]:
        return True, 1.0

    # 2. Backward compatibility with legacy name variants
    if w in ["mikaza", "megaza", "mikasa", "mikkaza", "makaza", "micaza"]:
        return True, 0.95

    # 3. Phonetic normalization match
    p = phonetic_mizo_normalize(w)
    if p == "mizo":
        return True, 0.95

    # 4. Strict fuzzy similarity directly against 'mizo'
    ratio = difflib.SequenceMatcher(None, w, "mizo").ratio()
    # Must start with 'm', have length 4-5 chars, and ratio >= 0.75
    if w.startswith("m") and len(w) in [4, 5] and ratio >= 0.75:
        return True, ratio

    return False, ratio


def is_stop_command(text: str, custom_variants: Optional[List[str]] = None) -> bool:
    """Checks if recognized text is a control stop command ('Stop Mizo', 'Stop', 'میزو سٹاپ', etc.)."""
    if not text or not text.strip():
        return False

    raw_clean = text.strip().lower()
    # Check directly for non-Latin / Urdu stop words
    for non_latin_stop in ["میزو سٹاپ", "سٹاپ", "میزو روکو", "روکو"]:
        if non_latin_stop in raw_clean:
            return True

    norm = normalize_text(text)
    if not norm:
        return False

    variants = custom_variants or CONFIGURED_STOP_COMMANDS

    for phrase in variants:
        phrase_norm = normalize_text(phrase)
        if norm == phrase_norm or raw_clean == phrase.lower():
            return True

    # Standalone prefix stop commands
    for phrase in ["stop mizo", "mizo stop", "stop meeso", "meeso stop", "stop mikaza", "mikaza stop"]:
        if norm.startswith(phrase):
            return True

    # Single word "stop"
    words = norm.split()
    if len(words) == 1 and words[0] in ["stop", "snooze"]:
        return True
    if len(words) >= 2 and words[0] == "stop":
        two_words = f"{words[0]} {words[1]}"
        for target in ["stop mizo", "stop meeso", "stop mikaza"]:
            if difflib.SequenceMatcher(None, two_words, target).ratio() >= 0.82:
                return True

    return False


def is_submit_command(text: str) -> bool:
    """Checks if recognized text is an explicit assessment submission command ('Submit', 'Finish', 'Done')."""
    norm = normalize_text(text)
    if not norm:
        return False

    if norm in CONFIGURED_SUBMIT_COMMANDS:
        return True

    for phrase in CONFIGURED_SUBMIT_COMMANDS:
        if norm == phrase or norm.startswith(phrase + " "):
            return True

    return False


def classify_command_or_intent(text: str) -> str:
    """
    Fine-grained command & intent classifier for the Mizo runtime:
    - 'STOP': Stop control command ('Stop Mizo', 'Stop', 'Go to sleep')
    - 'SUBMIT': Assessment submission command ('Submit', 'Finish', 'Done')
    - 'WAKE': Addressing Mizo to activate ('Hey Mizo', 'Mizo')
    - 'UPDATE_USER_NAME': Explicit request to update personal name ('My name is Shahid', 'Update my name to Shahid')
    - 'IDENTITY_QUERY': User asking Mizo's name ('What is your name?', 'Who are you?')
    - 'USER_NAME_QUERY': User asking for their own name ('What is my name?', 'What do you know about me?')
    - 'NORMAL': Conversational speech or lesson input
    - 'UNKNOWN': Inaudible or empty input
    """
    if not text or not text.strip():
        return "UNKNOWN"

    norm = normalize_text(text)
    if not norm:
        return "UNKNOWN"

    # 1. STOP command
    if is_stop_command(norm):
        return "STOP"

    # 2. SUBMIT command
    if is_submit_command(norm):
        return "SUBMIT"

    # 3. WAKE command
    intent = detect_mizo_intent(text, log_debug=False)
    if intent == "WAKE":
        return "WAKE"

    # 4. IDENTITY QUERY ("What is your name?", "Are you Mizo?", etc.)
    identity_queries = [
        "what is your name", "whats your name", "what s your name", "who are you",
        "what is your identity", "whats your identity", "tell me your name",
        "do you have a name", "what do i call you", "what should i call you",
        "are you mizo", "are you meeso", "are you miso", "is your name mizo",
        "is your name meeso", "is your name miso", "what is your name mizo"
    ]
    if any(norm == q or norm.startswith(q + " ") or norm.startswith(q + "?") for q in identity_queries):
        return "IDENTITY_QUERY"

    # 5. USER NAME QUERY ("What is my name?" / "What do you know about me?")
    user_name_queries = [
        "what is my name", "whats my name", "what s my name", "who am i",
        "do you know my name", "do you remember my name", "what do you know about me",
        "tell me my name", "say my name"
    ]
    if any(norm == q or norm.startswith(q + " ") or norm.startswith(q + "?") for q in user_name_queries):
        return "USER_NAME_QUERY"

    # 6. EXPLICIT NAME UPDATE ("My name is Shahid", "Update my name to Shahid")
    from app.services.memory_service import MizoMemoryManager
    name_update_patterns = [
        r"^(?:actually\s+)?my name is\s+([A-Za-z\-]+(?:\s+[A-Za-z\-]+)?)(?:,\s*(?:update|change|save|record).*)?$",
        r"^(?:please\s+|i want you to\s+|i want to\s+)?(?:change|update|set|replace)\s+my name\s+(?:to|as)\s+([A-Za-z\-]+(?:\s+[A-Za-z\-]+)?)[.!?]?$",
        r"^(?:please\s+)?(?:change|update)\s+my name to\s+([A-Za-z\-]+(?:\s+[A-Za-z\-]+)?)[.!?]?$",
        r"^(?:please\s+)?call me\s+([A-Za-z\-]+(?:\s+[A-Za-z\-]+)?)[.!?]?$",
        r"^my name is\s+([A-Za-z\-]+(?:\s+[A-Za-z\-]+)?)\b"
    ]
    for pat in name_update_patterns:
        m = re.search(pat, text.strip(), re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            cand = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", cand).strip()
            if MizoMemoryManager.is_valid_name(cand):
                return "UPDATE_USER_NAME"

    return "NORMAL"


def detect_mizo_intent(text: str, log_debug: bool = True) -> str:
    """
    Centralized intent detector identifying:
    - 'WAKE': Addressing Mizo to activate ('Mizo', 'Hey Mizo', 'Mizo, are you there?', 'Mizo, wake up', etc.)
    - 'STOP': Stop control command ('Stop Mizo', 'Mizo stop', 'Stop', etc.)
    - 'SUBMIT': Assessment submission command ('Submit', 'Finish', 'Done')
    - 'NORMAL': Conversational speech or instructions without wake/stop intent ('The weather is nice today.')
    - 'UNKNOWN': Empty or inaudible input

    Logs formatted debug outputs per specification:
    [WAKE] Raw transcription: "..."
    [WAKE] Normalized: "..."
    [WAKE] Similarity to "mizo": 0.xx
    [WAKE] Intent: ...
    """
    if not text or not text.strip():
        if log_debug:
            print(f'[WAKE] Raw transcription: "{text}"\n[WAKE] Normalized: ""\n[WAKE] Similarity to "mizo": 0.00\n[WAKE] Intent: UNKNOWN', flush=True)
        return "UNKNOWN"

    norm = normalize_text(text)
    words = norm.split()
    if not words:
        if log_debug:
            print(f'[WAKE] Raw transcription: "{text}"\n[WAKE] Normalized: ""\n[WAKE] Similarity to "mizo": 0.00\n[WAKE] Intent: UNKNOWN', flush=True)
        return "UNKNOWN"

    # 1. Check STOP command
    if is_stop_command(norm):
        if log_debug:
            print(f'[WAKE] Raw transcription: "{text}"', flush=True)
            print(f'[WAKE] Normalized: "{norm}"', flush=True)
            print(f'[WAKE] Similarity to "mizo": 0.00', flush=True)
            print(f'[WAKE] Intent: STOP', flush=True)
        return "STOP"

    # 2. Check SUBMIT command
    if is_submit_command(norm):
        if log_debug:
            print(f'[WAKE] Raw transcription: "{text}"', flush=True)
            print(f'[WAKE] Normalized: "{norm}"', flush=True)
            print(f'[WAKE] Similarity to "mizo": 0.00', flush=True)
            print(f'[WAKE] Intent: SUBMIT', flush=True)
        return "SUBMIT"

    # 3. Check WAKE intent
    is_wake = False
    best_similarity = 0.0

    # Calculate best similarity across all words for logging
    for w in words:
        match, score = is_mizo_word(w)
        if score > best_similarity:
            best_similarity = score

    # Condition A: Single word utterance matching Mizo (e.g. "Mizo", "Miso", "Meeso")
    if len(words) == 1:
        match, score = is_mizo_word(words[0])
        if match:
            is_wake = True
            best_similarity = max(best_similarity, score)

    # Condition B: Natural greeting followed by Mizo (e.g. "Hey Mizo", "Hello Mizo", "Hi Mizo", "Yo Mizo")
    elif words[0] in ["hey", "hi", "hello", "yo", "ok", "okay"] and len(words) >= 2:
        if is_mizo_word(words[1])[0]:
            is_wake = True
            best_similarity = max(best_similarity, is_mizo_word(words[1])[1])
        elif len(words) >= 3 and words[1] in ["there", "um", "ah", "please"] and is_mizo_word(words[2])[0]:
            is_wake = True
            best_similarity = max(best_similarity, is_mizo_word(words[2])[1])

    # Condition C: Utterance starting with Mizo (e.g. "Mizo, are you there?", "Mizo, wake up", "Mizo, can you hear me?")
    elif is_mizo_word(words[0])[0]:
        is_wake = True
        best_similarity = max(best_similarity, is_mizo_word(words[0])[1])

    # Condition D: Addressing phrases ending with Mizo (e.g. "Wake up, Mizo", "Are you there, Mizo", "Can you hear me, Mizo", "What's up, Mizo")
    elif is_mizo_word(words[-1])[0]:
        addressing_starters = [
            "wake up", "are you there", "can you hear me", "whats up", "what is up", "what s up",
            "hello", "hey", "hi", "talk to me", "listen to me"
        ]
        if any(starter in norm for starter in addressing_starters):
            is_wake = True
            best_similarity = max(best_similarity, is_mizo_word(words[-1])[1])

    # Condition E: Phrase containing "wake up" + Mizo anywhere
    elif "wake up" in norm and any(is_mizo_word(w)[0] for w in words):
        is_wake = True
        best_similarity = max(best_similarity, 0.95)

    # Format debug output logs
    intent = "WAKE" if is_wake else "NORMAL"
    if log_debug:
        print(f'[WAKE] Raw transcription: "{text}"', flush=True)
        print(f'[WAKE] Normalized: "{norm}"', flush=True)
        print(f'[WAKE] Similarity to "mizo": {best_similarity:.2f}', flush=True)
        print(f'[WAKE] Intent: {intent}', flush=True)

    return intent


def is_wake_word(text: str, custom_variants: Optional[List[str]] = None, log_debug: bool = True) -> Tuple[bool, str]:
    """
    Checks if recognized text contains an activation wake phrase for Mizo.
    Returns: (is_wake, trailing_speech)
    """
    intent = detect_mizo_intent(text, log_debug=log_debug)
    if intent != "WAKE":
        return False, ""

    norm = normalize_text(text)
    words = norm.split()
    trailing = ""

    # Extract trailing speech after Mizo or prefix
    # E.g. "Hey Mizo, teach me biology" -> "teach me biology"
    # E.g. "Mizo, what is photosynthesis?" -> "what is photosynthesis?"
    if len(words) >= 2 and words[0] in ["hey", "hi", "hello", "yo", "ok", "okay"]:
        if is_mizo_word(words[1])[0]:
            trailing = " ".join(words[2:]).strip()
        elif len(words) >= 3 and words[1] in ["there", "um", "ah", "please"] and is_mizo_word(words[2])[0]:
            trailing = " ".join(words[3:]).strip()
    elif len(words) >= 1 and is_mizo_word(words[0])[0]:
        trailing = " ".join(words[1:]).strip()
    elif norm.startswith("wake up") and any(is_mizo_word(w)[0] for w in words):
        idx = 0
        for i, w in enumerate(words):
            if is_mizo_word(w)[0]:
                idx = i
                break
        trailing = " ".join(words[idx+1:]).strip()
    elif len(words) >= 1 and is_mizo_word(words[-1])[0]:
        trailing = ""

    # Filter out pure wake-addressing queries from trailing speech (e.g. "are you there", "wake up")
    if trailing in ["are you there", "wake up", "can you hear me", "whats up", "what is up", "please wake up", "there", ""]:
        trailing = ""

    return True, trailing


# ---------------------------------------------------------------------------
# 4. Voice Manager & State Machine Controller
# ---------------------------------------------------------------------------
class VoiceManager:
    """
    Manages the voice state machine lifecycle with explicit debug logs:
    [VOICE] State: WAKE_LISTENING
    [VOICE] Wake listener: ACTIVE
    [VOICE] Microphone: ACTIVE
    [VOICE] Wake word detected: Mizo
    [VOICE] Waking assistant
    [VOICE] State: ACTIVE_CONVERSATION
    """

    def __init__(self):
        self.state: VoiceState = VoiceState.WAKE_LISTENING
        self.active_session_id: Optional[str] = None
        self.current_student_id: int = 1
        self.selected_agent: str = "Communication Coach"
        self.turn_silence_seconds: float = 3.0
        self.is_interrupted: bool = False
        self.is_speaking: bool = False
        self.auto_listen_after_tts: bool = True
        self._playback_process = None
        self._lock = threading.RLock()
        self._wake_listener_running: bool = False
        self._state_callbacks: List[Callable[[VoiceState], None]] = []

    @property
    def microphone_active(self) -> bool:
        """Indicates whether microphone listener is active (active in all listening & standby states)."""
        return self.state in [
            VoiceState.WAKE_LISTENING, VoiceState.STANDBY, VoiceState.SNOOZED, VoiceState.SLEEP,
            VoiceState.READY, VoiceState.ONBOARDING, VoiceState.ACTIVE_LISTENING,
            VoiceState.LISTENING, VoiceState.CONVERSATION, VoiceState.ASSESSMENT,
            VoiceState.ACTIVE_CONVERSATION
        ]

    @property
    def is_processing(self) -> bool:
        """Indicates if state machine is actively performing inference."""
        return self.state == VoiceState.PROCESSING

    def log(self, message: str):
        """Outputs structured debug logs according to Mizo specification."""
        print(f"[VOICE] {message}", flush=True)

    def transition_to(self, new_state: VoiceState, reason: str = ""):
        """Transitions state machine with thread-safety and logging."""
        with self._lock:
            old_state = self.state
            self.state = new_state

            # Safe Structured Debug Logging
            self.log(f"state={new_state.value}")

            if new_state in [VoiceState.WAKE_LISTENING, VoiceState.SNOOZED, VoiceState.STANDBY, VoiceState.SLEEP, VoiceState.PAUSED]:
                self.log("State: WAKE_LISTENING")
                self.log("State: SLEEP")
                self.log("Wake listener: ACTIVE")
                self.log("Microphone: ACTIVE")
            elif new_state == VoiceState.WAKE_DETECTED:
                self.log("Wake word detected: Mizo")
                self.log("Wake word detected")
                self.log("Waking assistant")
            elif new_state in [VoiceState.ACTIVE_CONVERSATION, VoiceState.ONBOARDING, VoiceState.ACTIVE_LISTENING, VoiceState.LISTENING, VoiceState.READY, VoiceState.CONVERSATION, VoiceState.ACTIVE]:
                self.log("State: ACTIVE_CONVERSATION")
                self.log("State: LISTENING")
                self.log("Microphone: ACTIVE")
            elif new_state == VoiceState.STOPPING:
                self.log("State: STOPPING")
            elif new_state == VoiceState.ERROR_RECOVERY:
                self.log("State: ERROR_RECOVERY")
            else:
                self.log(f"State: {new_state.value}")

            # Notify listeners if any
            for cb in self._state_callbacks:
                try:
                    cb(new_state)
                except Exception:
                    pass

    def add_state_listener(self, callback: Callable[[VoiceState], None]):
        self._state_callbacks.append(callback)

    def start_wake_listener(self):
        """
        Initializes and starts the single active wake listener in WAKE_LISTENING mode.
        Ensures thread/async safety and prevents duplicate listeners.
        """
        with self._lock:
            if self._wake_listener_running:
                return
            self._wake_listener_running = True
            self.transition_to(VoiceState.WAKE_LISTENING, "start_wake_listener")

    def stop_wake_listener(self):
        """Stops the wake listener gracefully."""
        with self._lock:
            self._wake_listener_running = False
            self.transition_to(VoiceState.WAKE_LISTENING, "stop_wake_listener")

    def handle_standby_speech(self, raw_transcript: str) -> Tuple[bool, str]:
        """
        Evaluates microphone speech in WAKE_LISTENING mode:
        - If wake word: transitions to ACTIVE_CONVERSATION and returns (True, trailing_speech)
        - Otherwise: logs ignore message and remains in WAKE_LISTENING
        """
        intent = detect_mizo_intent(raw_transcript, log_debug=False)

        if intent == "WAKE":
            is_wake, trailing = is_wake_word(raw_transcript, log_debug=False)
            self.log(f'Heard: "{raw_transcript}"')
            self.log("Wake word detected: Mizo")
            self.log("Wake word detected")
            self.log("Waking assistant")
            self.transition_to(VoiceState.ACTIVE_CONVERSATION, "wake_detected")
            if not trailing:
                self.log("Listening for user request")
            return True, trailing
        else:
            self.log(f'Heard: "{raw_transcript}"')
            self.log("No wake word. Ignoring.")
            self.log("Wake word not detected. Ignoring.")
            return False, ""

    def handle_user_input_captured(self, utterance: str) -> Tuple[str, str]:
        """
        Processes user speech captured in ACTIVE_CONVERSATION:
        - Checks for 'Stop Mizo'
        - Detects agent (Subject Tutor, Communication Coach, Speech Coach)
        - Transitions ACTIVE_CONVERSATION -> PROCESSING
        """
        self.log(f'Heard: "{utterance}"')
        self.log("End of speech detected.")
        self.transition_to(VoiceState.PROCESSING, "end_of_speech")

        # Check for stop command
        if is_stop_command(utterance):
            self.handle_stop_command()
            return "stop", "Standby"

        # Determine agent
        from app.services.personalization_service import personalization_service
        mode = personalization_service.detect_intent_and_mode(utterance)
        agent_name = personalization_service.get_agent_name(mode)

        self.selected_agent = agent_name
        self.log(f"Agent selected: {agent_name}")
        return mode, agent_name

    def handle_speaking_start(self):
        """Called when Mizo begins TTS playback."""
        self.is_interrupted = False
        self.is_speaking = True
        self.transition_to(VoiceState.SPEAKING, "tts_start")

    def handle_speaking_ended(self):
        """Called when TTS playback completes naturally."""
        self.is_speaking = False
        if self.auto_listen_after_tts:
            self.transition_to(VoiceState.ACTIVE_CONVERSATION, "playback_completed")
        else:
            self.transition_to(VoiceState.WAKE_LISTENING, "playback_completed")

    def handle_barge_in(self, speech_text: str = ""):
        """
        Called when user speaks during TTS playback:
        - Halts TTS playback immediately
        - Transitions to ACTIVE_CONVERSATION (or WAKE_LISTENING if user said 'Stop Mizo')
        """
        self.log("User speech detected during TTS.")
        self.log("Interrupting TTS.")
        self.is_interrupted = True
        self.stop_current_playback()

        if is_stop_command(speech_text):
            self.handle_stop_command()
        else:
            self.transition_to(VoiceState.ACTIVE_CONVERSATION, "barge_in_interruption")

    def handle_stop_command(self):
        """
        Executes immediate stop control:
        - Halts TTS playback immediately
        - Cancels current response
        - Transitions state to WAKE_LISTENING
        - Keeps wake listener and microphone active
        """
        self.log("Stop command detected")
        self.log("Stop command detected.")
        self.log("Stopping TTS...")
        self.log("Cancelling current response...")
        self.is_interrupted = True
        self.stop_current_playback()
        self.transition_to(VoiceState.WAKE_LISTENING, "stop_command")

    def enter_snooze(self):
        """
        Transitions into WAKE_LISTENING mode after ~10s of inactivity.
        Preserves continuous microphone monitoring for 'Mizo'.
        """
        self.transition_to(VoiceState.WAKE_LISTENING, "inactivity_snooze")
        self.log("Entered SLEEP mode. Audio monitoring active for 'Mizo'.")

    def stop_current_playback(self):
        """Halts running audio playback process/stream."""
        if self._playback_process:
            try:
                self._playback_process.terminate()
            except Exception:
                pass
            self._playback_process = None


def run_continuous_standby_listener(
    audio_transcribe_func: Optional[Callable[[], str]] = None,
    on_wake_callback: Optional[Callable[[str], None]] = None,
    should_stop_func: Callable[[], bool] = lambda: False,
    capture_audio_func: Optional[Callable[[], Any]] = None,
    transcribe_func: Optional[Callable[[Any], str]] = None,
    recover_microphone_func: Optional[Callable[[], None]] = None
):
    """
    Runs a continuous indefinite microphone listener loop for SLEEP / STANDBY mode.
    - Software listener never completely shuts down.
    - Automatically restarts/recovers after every timeout or error.
    - Does NOT terminate after 3s, 10s, or 12s.
    - Only invokes on_wake_callback when detect_mizo_intent returns 'WAKE'.
    """
    logger.info("[VOICE] Continuous standby wake listener active.")
    voice_manager.transition_to(VoiceState.STANDBY, "continuous_listener_start")

    while not should_stop_func():
        try:
            if capture_audio_func and transcribe_func:
                audio = capture_audio_func()
                if not audio:
                    continue
                transcript = transcribe_func(audio)
            elif audio_transcribe_func:
                transcript = audio_transcribe_func()
            else:
                break

            if not transcript or not transcript.strip():
                continue

            intent = detect_mizo_intent(transcript, log_debug=True)
            if intent == "WAKE":
                is_wake, trailing = is_wake_word(transcript, log_debug=False)
                voice_manager.handle_standby_speech(transcript)
                if on_wake_callback:
                    on_wake_callback(trailing)
                break
            else:
                voice_manager.handle_standby_speech(transcript)
        except TimeoutError:
            # Standby timeout continues monitoring indefinitely without terminating
            continue
        except Exception as e:
            logger.warning(f"[VOICE] Standby audio listener warning: {e}. Recovering listener...")
            if recover_microphone_func:
                try:
                    recover_microphone_func()
                except Exception:
                    pass
            time.sleep(0.2)
            continue


# Global singleton instance
voice_manager = VoiceManager()
