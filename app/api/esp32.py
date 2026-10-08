import os
import re
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from fastapi import APIRouter, UploadFile, File, Form, Header, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings
from app.db.database import (
    validate_device_key,
    add_conversation_message,
    get_student,
    update_student,
    record_onboarding_baseline,
    PROTOTYPE_STUDENT_ID,
    get_prototype_student,
    resolve_prototype_session,
    get_session_by_id,
    get_pending_name_change,
    set_pending_name_change,
    clear_pending_name_change,
    handle_device_start_or_reset,
    get_db_settings,
    get_conversation_history
)
from app.db.models import ESP32ChatRequest, ESP32ChatResponse, ESP32StartRequest, ESP32StartResponse
from app.services.stt_service import stt_service
from app.services.llm_service import llm_service, LLMServiceError
from app.services.tts_service import tts_service
from app.services.personalization_service import personalization_service
from app.services.memory_service import memory_manager
from app.services.rag_service import rag_service
from app.services.tutor_service import tutor_service
from app.services.speech_service import speech_service
from app.services.agent_router import agent_router, AgentRoute, AgentRouter
from app.services.normal_chat_service import normal_chat_service
from app.services.onboarding_service import onboarding_service
from app.services.voice_service import is_wake_word, is_stop_command, is_submit_command, classify_command_or_intent, voice_manager, VoiceState, detect_mizo_intent, ASSISTANT_NAME

router = APIRouter(prefix="/api/v1/esp32", tags=["ESP32 Hardware"])


def _require_primary_student() -> Dict[str, Any]:
    student = get_prototype_student()
    if not student:
        raise HTTPException(status_code=503, detail="Primary student profile (Student #1) is missing")
    return student


def _name_change_decision(text: str) -> str:
    normalized = " ".join((text or "").lower().strip().replace(".", "").replace(",", "").split())
    confirm_keywords = ["yes", "yeah", "yep", "change it", "update it", "replace it", "please do", "confirm", "correct", "sure"]
    if any(k in normalized for k in confirm_keywords) and not any(k in normalized for k in ["no", "don't", "dont", "keep"]):
        return "confirm"
    if any(k in normalized for k in ["no", "keep it", "dont change it", "don't change it", "cancel"]):
        return "reject"
    return "unknown"


def _handle_name_change(student: Dict[str, Any], session_id: str, text: str, command: str) -> Optional[str]:
    """Keeps proposed profile-name changes session-scoped until explicit confirmation."""
    current_name = (student.get("name") or "").strip()
    is_known_name = current_name and current_name.lower() not in ["new learner", "student", "there", "unknown", ""]

    pending_name = get_pending_name_change(session_id)
    if pending_name:
        decision = _name_change_decision(text)
        clean_lower = text.strip().lower().replace(",", "")
        if decision == "confirm" or clean_lower in {f"call me {pending_name.lower()}", f"call me {pending_name.lower()}.", "yes", "yes change it", "yes, change it"}:
            update_student(PROTOTYPE_STUDENT_ID, {"name": pending_name})
            clear_pending_name_change(session_id)
            return f"Got it! I have updated your name to {pending_name}."
        if decision == "reject":
            clear_pending_name_change(session_id)
            return f"Okay, I will keep your name as {current_name if is_known_name else 'your current profile name'}."
        return f"Your current name is {current_name if is_known_name else 'your current profile name'}. You asked me to change it to {pending_name}. Should I replace your name and call you {pending_name}?"

    proposed_name = memory_manager.extract_name_safely(text, current_student=student)
    if not proposed_name and command == "UPDATE_USER_NAME":
        from app.services.memory_service import MizoMemoryManager
        m = re.search(r"(?:name\s+(?:to|as)|my name is|call me)\s+([A-Za-z\-]+)", text, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            if MizoMemoryManager.is_valid_name(cand):
                proposed_name = cand.capitalize()

    if not proposed_name:
        return None

    if is_known_name:
        if proposed_name.lower() == current_name.lower():
            return f"I already have your name saved as {current_name}."
        set_pending_name_change(session_id, proposed_name)
        return f"Your current name is {current_name}. You asked me to change it to {proposed_name}. Should I replace your name and call you {proposed_name}?"
    else:
        update_student(PROTOTYPE_STUDENT_ID, {"name": proposed_name})
        return None


def is_non_english_input(text: str, audio_metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """
    Detects genuinely non-English input speech or text.
    Strictly prevents false positive rejection of valid English input:
    - Never reject 'Hello everyone!'
    - Never reject '1 kg of chicken breast'
    - Never reject 'I will prepare the vegetables for the sauce.'
    - Never reject 'Hello everyone, welcome to my channel.'
    - Never reject 'Hi, are you able to understand now?'
    Never reject if input is a STOP command.
    """
    if not text or not text.strip():
        return False, ""

    from app.services.voice_service import is_stop_command
    if is_stop_command(text):
        return False, ""

    clean_text = text.strip()
    lower_words = re.sub(r"[^\w\s]", " ", clean_text.lower()).split()
    if not lower_words:
        return False, ""

    english_common = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
        "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by",
        "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all",
        "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him",
        "know", "take", "people", "into", "year", "your", "good", "some", "could", "them",
        "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
        "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us", "am",
        "is", "are", "was", "were", "been", "being", "studying", "study", "engineering", "robotics",
        # Food & cooking
        "chicken", "breast", "prepare", "vegetables", "vegetable", "sauce", "kg", "grams", "meat", "food",
        "cooking", "cook", "recipe", "water", "tea", "coffee", "rice",
        # Greetings & channel / conversation
        "hello", "hi", "hey", "everyone", "everybody", "welcome", "channel", "understand", "able", "speaking", "speak",
        "listen", "please", "thanks", "thank", "yes", "yeah", "okay", "ok", "talk", "talking", "about",
        # Anime & favorites
        "character", "favorite", "anime", "crew", "members", "pirates", "straw", "hat", "forgot", "wrong",
        "piece", "zoro", "luffy", "jinbe"
    }

    english_matches = [w for w in lower_words if w in english_common]
    # If standard English words are found, NEVER classify as non-English!
    if english_matches:
        return False, ""

    # Non-Latin script detection (Urdu, Arabic, Devanagari, Chinese, Cyrillic, etc.)
    non_latin_count = sum(1 for c in clean_text if ord(c) > 0x024F and c.isalnum())
    total_alpha = sum(1 for c in clean_text if c.isalpha())
    if total_alpha >= 3 and (non_latin_count / total_alpha) > 0.40:
        return True, "Non-Latin script detected"

    foreign_indicators = {
        "terima", "kasih", "telah", "menonton", "selamat", "pagi", "siang", "malam",
        "apa", "kabar", "saya", "kamu", "bisa", "tidak", "dengan", "untuk", "dari",
        "yang", "dan", "ini", "itu", "sudah", "belum", "bagaimana", "kenapa", "apakah",
        "gracias", "hola", "buenos", "dias", "tardes", "noches", "por", "favor",
        "como", "estas", "amigo", "amiga", "adios", "hasta", "luego", "mucho", "gusto",
        "merci", "bonjour", "bonsoir", "s'il", "vous", "plait", "comment", "allez",
        "danke", "bitte", "guten", "morgen", "wie", "geht", "tschuss", "auf", "wiedersehen"
    }
    foreign_matches = [w for w in lower_words if w in foreign_indicators]
    if len(foreign_matches) >= 2 and len(foreign_matches) > len(english_matches):
        return True, "Foreign language words detected"

    if audio_metadata:
        lang = audio_metadata.get("language")
        if lang and isinstance(lang, str):
            lang_clean = lang.lower().strip()
            # Only trigger on STT metadata if zero English words AND foreign indicators were observed
            if lang_clean not in ["en", "english", ""] and len(foreign_matches) >= 1:
                return True, f"STT detected non-English language ({lang_clean})"

    return False, ""


def verify_device(x_device_key: Optional[str] = Header(None), key: Optional[str] = Query(None)):
    """Verifies ESP32 device authorization key."""
    token = x_device_key or key
    if not token or not validate_device_key(token):
        if token == settings.DEFAULT_DEVICE_KEY:
            return True
        if settings.ENVIRONMENT == "development" and not token:
            return True
        raise HTTPException(status_code=401, detail="Invalid or missing ESP32 Device API Key")
    return True


@router.get("/ping")
async def ping():
    """Healthcheck endpoint for ESP32 Wi-Fi heartbeat."""
    return {"status": "online", "server": settings.APP_NAME, "version": "3.0.0"}


@router.get("/audio/cache/{filename}")
async def get_cached_audio(filename: str):
    """Serves synthesized speech WAV files for ESP32 MAX98357A I2S playback."""
    file_path = settings.AUDIO_CACHE_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path=str(file_path), media_type="audio/wav", filename=filename)


@router.get("/voice_state")
async def get_voice_state():
    """Returns current voice state and microphone active status."""
    return {
        "success": True,
        "voice_state": voice_manager.state.value,
        "microphone_active": voice_manager.microphone_active,
        "assistant_name": ASSISTANT_NAME
    }


@router.post("/start", response_model=ESP32StartResponse)
@router.post("/reset", response_model=ESP32StartResponse)
async def handle_esp32_start_button(req: Optional[ESP32StartRequest] = None):
    """
    Physical ESP32 Start / Reset button handler (and browser simulator button):
    - If student exists & onboarding is complete:
      -> Returning user! Loads profile, creates a NEW session (current metrics = 0), preserves all historical data.
      -> Generates personalized AI welcome acknowledging their name, goals, and focus weaknesses.
    - If student exists & onboarding is incomplete:
      -> Resumes onboarding from the last incomplete state.
    - If student does not exist:
      -> Starts fresh onboarding from ASK_NAME.
    """
    student_id = PROTOTYPE_STUDENT_ID
    action = req.action if req and req.action else "start"

    start_info = handle_device_start_or_reset(student_id=student_id)
    if start_info["status"] == "primary_student_missing":
        raise HTTPException(status_code=503, detail="Primary student profile (Student #1) is missing")
    student = start_info["student"]
    session = start_info["session"]
    session_id = session["session_id"]
    is_returning = start_info["status"] == "returning_user"
    onboarding_active = start_info["onboarding_active"]
    current_step = start_info["onboarding_step"]

    if is_returning:
        # Returning student: build personalized welcome prompt
        system_prompt = personalization_service.build_returning_user_prompt(student)
        messages = [{"role": "user", "content": "Hello Mizo, I am back!"}]
    else:
        # Onboarding student: build onboarding prompt for the current step
        system_prompt = personalization_service.build_system_prompt(
            student_id=student_id,
            mode="coach",
            session_id=session_id
        )
        messages = [{"role": "user", "content": "Start onboarding"}]

    try:
        llm_res = await llm_service.generate_response(messages=messages, system_prompt=system_prompt)
        reply_text = llm_res.text
    except Exception as e:
        if is_returning:
            reply_text = f"Welcome back, {student.get('name', 'learner')}! I'm glad to see you again. Let's continue working on your communication goals."
        else:
            reply_text = "Hi! I'm Mizo. Before we begin, I'd like to get to know you. What's your name?"

    # Synthesize speech to 16kHz WAV
    voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "start_session")
    voice_manager.handle_speaking_start()
    audio_url, _ = await tts_service.synthesize_to_file(reply_text)
    voice_manager.handle_speaking_ended()

    # Log initial interaction to conversation history
    add_conversation_message(
        student_id=student_id,
        session_id=session_id,
        role="assistant",
        content=reply_text,
        audio_path=audio_url,
        learning_mode="coach",
        onboarding_state=current_step if onboarding_active else "returning_welcome",
        intent="start_session"
    )

    return ESP32StartResponse(
        success=True,
        action=action,
        is_returning_user=is_returning,
        student_id=student_id,
        student_name=student.get("name", "New Learner"),
        reply_text=reply_text,
        audio_url=audio_url,
        onboarding_active=onboarding_active,
        onboarding_step=current_step if onboarding_active else "completed",
        session_id=session_id,
        topic=personalization_service.generate_dynamic_assessment_topic(student) if current_step in ["speech_test_prompt", "speaking_assessment"] else None
    )


@router.get("/audio")
async def esp32_audio_status():
    """Status endpoint for ESP32 hardware voice pipeline."""
    return {
        "status": "ready",
        "service": "Mizo 3.0 ESP32 Hardware Audio Pipeline",
        "voice_state": voice_manager.state.value,
        "instructions": "Send POST multipart/form-data with 'audio' (.wav) to interact with Mizo.",
        "supported_formats": ["wav", "pcm", "mp3"]
    }


@router.post("/audio")
async def process_esp32_audio(
    audio: UploadFile = File(...),
    student_id: int = Form(1),
    session_id: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
    format: str = Query("json", description="Response format: 'json' or 'audio'")
):
    """
    Core Hardware Voice Pipeline for ESP32-S3 + INMP441 Microphone + MAX98357A Speaker.
    1. Receives recorded audio from ESP32.
    2. Transcribes via STT (Groq Whisper / OpenAI Whisper / Faster-Whisper). Real errors, no mock text.
    3. Handles Onboarding State Machine & multi-field information extraction.
    4. Handles 1-2 minute speaking assessment evaluation & baseline establishment.
    5. Dispatches to LLM Router with dynamic personalized context.
    6. Synthesizes 16kHz 16-bit Mono WAV audio for MAX98357A I2S DAC.
    7. Returns either direct WAV audio stream (format=audio) or JSON metadata (format=json).
    """
    # 1. Save uploaded audio temporarily
    file_ext = Path(audio.filename or "recording.wav").suffix or ".wav"
    temp_in_path = settings.AUDIO_CACHE_DIR / f"esp32_in_{uuid.uuid4().hex[:8]}{file_ext}"
    with open(temp_in_path, "wb") as f:
        f.write(await audio.read())

    transcript = ""
    measured_duration_seconds = 0.0
    stt_metadata = {}
    try:
        # 2. Transcribe Audio using real STT with metadata
        stt_res = await stt_service.transcribe_audio_with_metadata(temp_in_path)
        transcript = stt_res.get("text", "")
        measured_duration_seconds = float(stt_res.get("duration_seconds") or 0.0)
        stt_metadata = {
            "duration_seconds": measured_duration_seconds,
            "segments": stt_res.get("segments", []),
            "words": stt_res.get("words", []),
            "language": stt_res.get("language")
        }
    except Exception as e:
        print(f"[STT Error] Transcription failed: {e}")
        transcript = ""
    finally:
        if temp_in_path.exists():
            try:
                temp_in_path.unlink()
            except Exception:
                pass

    # 3. Classify Command or Intent BEFORE standard LLM/dialogue processing
    clean_transcript = transcript.strip() if transcript else ""
    cmd_intent = classify_command_or_intent(clean_transcript)

    # 4. Check for "Stop Mizo" control command (Deterministic Tier 0)
    if clean_transcript and (cmd_intent == "STOP" or is_stop_command(clean_transcript)):
        voice_manager.handle_stop_command()
        reply_msg = "I'm pausing now. Whenever you're ready to talk again, just say 'Hey Mizo'."
        audio_url, wav_path = await tts_service.synthesize_to_file(reply_msg)
        if format == "audio":
            return FileResponse(path=wav_path, media_type="audio/wav", filename="reply.wav")
        return {
            "success": True,
            "is_control_command": True,
            "command": "stop",
            "transcript": transcript,
            "reply_text": reply_msg,
            "audio_url": audio_url,
            "voice_state": "PAUSED"
        }

    # 5. Check student & onboarding status
    student_id = PROTOTYPE_STUDENT_ID
    student = _require_primary_student()

    st_name = (student.get("name") or "").strip()
    is_name_known = bool(st_name and st_name.lower() not in ["new learner", "student", "there", "unknown", ""])
    onboarding_active = not bool(student.get("onboarding_completed", False))
    current_step = (student.get("onboarding_step") or ("completed" if not onboarding_active else "ask_name")).lower()
    if onboarding_active and is_name_known and current_step in ["ask_name", "new_student", "welcome"]:
        current_step = "ask_goals"

    # Handle silence or missing speech
    if not clean_transcript or clean_transcript in ["[Speech Recognition Error]", ""]:
        if onboarding_active:
            retry_msg = "I couldn't hear that clearly. Could you say that again?"
            audio_url, wav_path = await tts_service.synthesize_to_file(retry_msg)
            if format == "audio":
                return FileResponse(path=wav_path, media_type="audio/wav", filename="reply.wav")
            return {
                "success": True,
                "transcript": clean_transcript or "[Speech Recognition Error]",
                "reply_text": retry_msg,
                "audio_url": audio_url,
                "onboarding_active": True,
                "onboarding_step": current_step,
                "detail": "No audible speech detected; repeating prompt."
            }
        else:
            if voice_manager.state in [VoiceState.STANDBY, VoiceState.SLEEP, VoiceState.SNOOZED, VoiceState.PAUSED]:
                return {
                    "success": True,
                    "ignored": True,
                    "transcript": "",
                    "reply_text": "",
                    "audio_url": None,
                    "voice_state": voice_manager.state.value
                }
            silence_msg = "I had trouble hearing your audio clearly. Could you please try speaking again or check your microphone?"
            audio_url, wav_path = await tts_service.synthesize_to_file(silence_msg)
            if format == "audio":
                return FileResponse(path=wav_path, media_type="audio/wav", filename="reply.wav")
            return {
                "success": False,
                "transcript": "[Speech Recognition Error]",
                "reply_text": silence_msg,
                "audio_url": audio_url,
                "detail": "No audible speech or STT failure detected"
            }

    # 6. WAKE_LISTENING / SNOOZED / STANDBY / PAUSED State Machine
    if voice_manager.state in [VoiceState.WAKE_LISTENING, VoiceState.STANDBY, VoiceState.SLEEP, VoiceState.SNOOZED, VoiceState.PAUSED, VoiceState.WAKE_ONLY]:
        is_wake, trailing_speech = is_wake_word(clean_transcript, log_debug=False)
        is_explicit_resume = clean_transcript.strip().lower() in ["resume", "continue", "start"]
        if is_wake or is_explicit_resume:
            voice_manager.log(f'Heard: "{clean_transcript}"')
            voice_manager.log("Wake word detected: Mizo")
            voice_manager.log("Wake word detected")
            voice_manager.log("Waking assistant")
            voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "wake_word_detected")
            if trailing_speech:
                clean_transcript = trailing_speech
                voice_manager.transition_to(VoiceState.ONBOARDING if onboarding_active else VoiceState.ACTIVE_CONVERSATION, "wake_with_speech")
            else:
                voice_manager.log("Listening for user request")
                if is_name_known or student.get("onboarding_completed", False):
                    reply_msg = f"Yes, {st_name}? I'm listening." if is_name_known else "Yes, I'm listening."
                    voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "wake_ready")
                elif onboarding_active and current_step in ["ask_name", "welcome"]:
                    reply_msg = "Hi! I'm Mizo. Before we begin, I'd like to get to know you. What's your name?"
                    voice_manager.transition_to(VoiceState.ONBOARDING, "start_onboarding")
                    active_session = resolve_prototype_session(None, default_mode=mode or "coach")
                    curr_session_id = active_session["session_id"]
                    add_conversation_message(
                        student_id=student_id,
                        session_id=curr_session_id,
                        role="assistant",
                        content=reply_msg,
                        learning_mode="coach",
                        onboarding_state="ask_name"
                    )
                else:
                    reply_msg = "Yes, I'm listening."
                    voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "wake_ready")

                audio_url, wav_path = await tts_service.synthesize_to_file(reply_msg)
                voice_manager.handle_speaking_start()
                voice_manager.handle_speaking_ended()
                if format == "audio":
                    return FileResponse(path=wav_path, media_type="audio/wav", filename="reply.wav")
                return {
                    "success": True,
                    "activated": True,
                    "transcript": transcript,
                    "reply_text": reply_msg,
                    "audio_url": audio_url,
                    "onboarding_active": onboarding_active,
                    "onboarding_step": current_step,
                    "voice_state": "ACTIVE"
                }
        else:
            voice_manager.log(f'Heard: "{clean_transcript}"')
            voice_manager.log("No wake word. Ignoring.")
            voice_manager.log("Wake word not detected. Ignoring.")
            return {
                "success": True,
                "ignored": True,
                "transcript": transcript,
                "reply_text": "",
                "audio_url": None,
                "voice_state": "STANDBY"
            }
    elif voice_manager.state == VoiceState.SPEAKING:
        voice_manager.handle_barge_in(clean_transcript)
        if cmd_intent == "STOP" or is_stop_command(clean_transcript):
            return {
                "success": True,
                "is_control_command": True,
                "command": "stop",
                "transcript": transcript,
                "reply_text": "",
                "audio_url": None,
                "voice_state": "STANDBY"
            }
    else:
        is_wake, trailing_speech = is_wake_word(clean_transcript, log_debug=False)
        if is_wake and trailing_speech:
            clean_transcript = trailing_speech

    # 7. Identity & Name Commands (Interception before regular dialogue)
    active_session = resolve_prototype_session(session_id, default_mode=mode or "coach")
    if not active_session:
        raise HTTPException(status_code=503, detail="No active session is available for Student #1")
    curr_session_id = active_session["session_id"]
    active_mode = mode or active_session.get("mode") or "coach"

    # 7-0. Non-English Detection (Tier 0 deterministic response)
    is_non_eng, non_eng_reason = is_non_english_input(clean_transcript, audio_metadata=stt_metadata)
    if is_non_eng:
        reply_text = "I didn't understand that. Could you say it in English?"
        voice_manager.handle_speaking_start()
        audio_url, wav_path = await tts_service.synthesize_to_file(reply_text)
        voice_manager.handle_speaking_ended()
        voice_manager.transition_to(VoiceState.ONBOARDING if onboarding_active else VoiceState.READY, "non_english_rejected")
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="user", content=clean_transcript, learning_mode=active_mode)
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="assistant", content=reply_text, audio_path=audio_url, learning_mode=active_mode)
        if format == "audio":
            return FileResponse(path=wav_path, media_type="audio/wav", filename="reply.wav")
        return {
            "success": True,
            "transcript": transcript,
            "reply_text": reply_text,
            "audio_url": audio_url,
            "learning_mode": active_mode,
            "session_id": curr_session_id,
            "provider_used": "system",
            "is_fallback": False,
            "onboarding_active": onboarding_active,
            "onboarding_step": current_step,
            "voice_state": voice_manager.state.value
        }

    is_known_name = bool(student.get("name") and student.get("name").strip().lower() not in ["new learner", "student", "there", "unknown", ""])
    pending_name = get_pending_name_change(curr_session_id)
    name_change_reply = None
    if is_known_name or pending_name or not onboarding_active:
        name_change_reply = _handle_name_change(student, curr_session_id, clean_transcript, cmd_intent)

    if name_change_reply:
        voice_manager.handle_speaking_start()
        audio_url, wav_path = await tts_service.synthesize_to_file(name_change_reply)
        voice_manager.handle_speaking_ended()
        voice_manager.transition_to(VoiceState.ONBOARDING if onboarding_active else VoiceState.READY, "name_change_pending")
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="user", content=clean_transcript, learning_mode=active_mode)
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="assistant", content=name_change_reply, audio_path=audio_url, learning_mode=active_mode)
        if format == "audio":
            return FileResponse(path=wav_path, media_type="audio/wav", filename="reply.wav")
        return {"success": True, "transcript": transcript, "reply_text": name_change_reply, "audio_url": audio_url,
                "learning_mode": active_mode, "session_id": curr_session_id, "provider_used": "system",
                "is_fallback": False, "onboarding_active": onboarding_active, "onboarding_step": current_step,
                "voice_state": voice_manager.state.value}

    # 7A. Assistant Identity Query ("What is your name?")
    if cmd_intent == "IDENTITY_QUERY":
        reply_text = "My name is Mizo, your English communication coach."
        if onboarding_active:
            if current_step in ["ask_name", "welcome"]:
                reply_text += " What's your name?"
            elif current_step in ["ask_goals", "ask_problems"]:
                reply_text += " Tell me a bit about yourself—what are your main goals for improving your English?"
            elif current_step in ["speech_test_prompt", "speaking_assessment"]:
                reply_text += " Please speak freely for about one minute on your favorite project or hobby."
        voice_manager.handle_speaking_start()
        audio_url, wav_path = await tts_service.synthesize_to_file(reply_text)
        voice_manager.handle_speaking_ended()
        voice_manager.transition_to(VoiceState.ONBOARDING if onboarding_active else VoiceState.READY, "identity_answered")
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="user", content=clean_transcript, learning_mode=active_mode)
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="assistant", content=reply_text, audio_path=audio_url, learning_mode=active_mode)
        if format == "audio":
            return FileResponse(path=wav_path, media_type="audio/wav", filename="reply.wav")
        return {
            "success": True,
            "transcript": transcript,
            "reply_text": reply_text,
            "audio_url": audio_url,
            "learning_mode": active_mode,
            "session_id": curr_session_id,
            "provider_used": "system",
            "is_fallback": False,
            "onboarding_active": onboarding_active,
            "onboarding_step": current_step,
            "voice_state": voice_manager.state.value
        }

    # 7B. Student Name Query ("What is my name?" / "Who am I?")
    if cmd_intent == "USER_NAME_QUERY":
        st_name = student.get("name")
        if st_name and st_name not in ["New Learner", "Student", "there", "Unknown", ""]:
            reply_text = f"Your name is {st_name}."
        else:
            reply_text = "I don't have your name yet. What should I call you?"
        if onboarding_active and current_step in ["ask_goals", "ask_problems"]:
            reply_text += " Tell me a bit about yourself—what are your main goals for improving your English?"
        voice_manager.handle_speaking_start()
        audio_url, wav_path = await tts_service.synthesize_to_file(reply_text)
        voice_manager.handle_speaking_ended()
        voice_manager.transition_to(VoiceState.ONBOARDING if onboarding_active else VoiceState.READY, "user_name_answered")
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="user", content=clean_transcript, learning_mode=active_mode)
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="assistant", content=reply_text, audio_path=audio_url, learning_mode=active_mode)
        if format == "audio":
            return FileResponse(path=wav_path, media_type="audio/wav", filename="reply.wav")
        return {
            "success": True,
            "transcript": transcript,
            "reply_text": reply_text,
            "audio_url": audio_url,
            "learning_mode": active_mode,
            "session_id": curr_session_id,
            "provider_used": "system",
            "is_fallback": False,
            "onboarding_active": onboarding_active,
            "onboarding_step": current_step,
            "voice_state": voice_manager.state.value
        }

    # 7C. Tier 0 Centralized Deterministic Resolution (Profile queries, Session commands, etc.)
    tier0_res = llm_service.resolve_tier0(clean_transcript, student_data=student, session_id=curr_session_id)
    if tier0_res and tier0_res.get("handled"):
        reply_text = tier0_res["reply_text"]
        action = tier0_res.get("action")
        if action == "reset_session":
            from app.db.database import get_connection
            conn = get_connection()
            conn.cursor().execute("DELETE FROM conversation_messages WHERE session_id = ?", (curr_session_id,))
            conn.commit()
            conn.close()
        elif action == "stop":
            voice_manager.handle_stop_command()

        voice_manager.handle_speaking_start()
        audio_url, wav_path = await tts_service.synthesize_to_file(reply_text)
        voice_manager.handle_speaking_ended()
        voice_manager.transition_to(VoiceState.ONBOARDING if onboarding_active else VoiceState.READY, "tier0_handled")
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="user", content=clean_transcript, learning_mode=active_mode)
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="assistant", content=reply_text, audio_path=audio_url, learning_mode=active_mode)
        if format == "audio":
            return FileResponse(path=wav_path, media_type="audio/wav", filename="reply.wav")
        return {
            "success": True,
            "transcript": transcript,
            "reply_text": reply_text,
            "audio_url": audio_url,
            "learning_mode": active_mode,
            "session_id": curr_session_id,
            "provider_used": "system",
            "is_fallback": False,
            "onboarding_active": onboarding_active,
            "onboarding_step": current_step,
            "voice_state": voice_manager.state.value
        }

    # 8. Processing user input
    voice_manager.log(f'Heard: "{clean_transcript}"')
    voice_manager.log("End of speech detected.")
    voice_manager.transition_to(VoiceState.PROCESSING, "end_of_speech")

    # Track conversation topic, subtopic, entities & lightweight intent (Sections 21-23, 26, 28)
    from app.services.context_service import update_conversation_topic_state
    conv_context = update_conversation_topic_state(
        session_id=curr_session_id,
        student_id=student_id,
        user_text=clean_transcript
    )

    detected_mode = personalization_service.detect_intent_and_mode(clean_transcript, current_mode=active_mode)
    agent_name = personalization_service.get_agent_name(detected_mode)
    voice_manager.selected_agent = agent_name
    voice_manager.log(f"Agent selected: {agent_name}")

    grammar_errors = []
    vocab_suggestions = []
    fluency_score = 75.0
    pacing_score = 80.0
    confidence_score = 75.0
    reply_text = ""
    provider_used = "ollama"
    is_fallback = False
    latency_ms = 0.0

    if onboarding_active:
        voice_manager.transition_to(VoiceState.ONBOARDING, "onboarding_active")
        is_assessment_step = current_step in [personalization_service.STATE_ASSESSMENT, "SPEECH_TEST_PROMPT", "SPEECH_EVALUATION", "speech_test_prompt", "speaking_assessment"] or cmd_intent == "SUBMIT"

        if is_assessment_step:
            words_in_transcript = len(clean_transcript.split())
            prev_cum_words = int(student.get("cumulative_assessment_words") or 0)
            prev_samples_cnt = int(student.get("assessment_samples_count") or 0)
            cum_words = prev_cum_words + words_in_transcript
            samples_cnt = prev_samples_cnt + 1

            update_student(student_id, {
                "cumulative_assessment_words": cum_words,
                "assessment_samples_count": samples_cnt
            })

            # Check if this single sample is very brief and cumulative words is under 40 and first sample
            if words_in_transcript < 12 and cum_words < 40 and samples_cnt <= 1:
                retry_msg = "Please share a little more about yourself or your projects so I can calibrate our lessons accurately."
                audio_url, wav_path = await tts_service.synthesize_to_file(retry_msg)
                voice_manager.transition_to(VoiceState.ASSESSMENT, "retry_assessment")
                if format == "audio":
                    return FileResponse(path=wav_path, media_type="audio/wav", filename="reply.wav")
                return {
                    "success": False,
                    "transcript": clean_transcript,
                    "reply_text": retry_msg,
                    "audio_url": audio_url,
                    "onboarding_active": True,
                    "onboarding_step": "speech_test_prompt",
                    "topic": personalization_service.generate_dynamic_assessment_topic(student),
                    "detail": "Preliminary sample recorded. Additional evidence requested.",
                    "voice_state": voice_manager.state.value
                }

            # Multiple samples or sufficient evidence reached! Finalize preliminary assessment.
            voice_manager.transition_to(VoiceState.PROCESSING_ASSESSMENT, "evaluate_assessment")
            assessment_data = await personalization_service.run_speaking_assessment(
                transcript=clean_transcript,
                student_id=student_id,
                session_id=curr_session_id,
                topic=personalization_service.generate_dynamic_assessment_topic(student),
                duration_seconds=measured_duration_seconds if measured_duration_seconds > 0 else None,
                audio_metadata=stt_metadata
            )

            if assessment_data.get("invalid_input"):
                retry_msg = assessment_data.get("spoken_summary") or "Speech recognition quality was insufficient. Please speak clearly for about one minute."
                audio_url, wav_path = await tts_service.synthesize_to_file(retry_msg)
                voice_manager.transition_to(VoiceState.ASSESSMENT, "retry_invalid_assessment")
                if format == "audio":
                    return FileResponse(path=wav_path, media_type="audio/wav", filename="reply.wav")
                return {
                    "success": False,
                    "transcript": clean_transcript,
                    "reply_text": retry_msg,
                    "audio_url": audio_url,
                    "onboarding_active": True,
                    "onboarding_step": "speech_test_prompt",
                    "topic": personalization_service.generate_dynamic_assessment_topic(student),
                    "detail": assessment_data.get("quality_reason", "Low quality or inaudible input"),
                    "voice_state": voice_manager.state.value
                }

            reply_text = assessment_data.get("spoken_summary", "Calibration complete! Your speaking assessment is saved.")
            onboarding_active = False
            next_step = "completed"
            update_student(student_id, {"onboarding_completed": True, "onboarding_step": "completed"})
            voice_manager.transition_to(VoiceState.ASSESSMENT_COMPLETE, "assessment_done")
            voice_manager.transition_to(VoiceState.READY, "ready_post_assessment")
        else:
            is_valid_ans, val_reason = personalization_service.is_valid_onboarding_response(student, current_step, clean_transcript)
            if not is_valid_ans:
                next_step = current_step
                system_prompt = personalization_service.build_onboarding_clarification_prompt(
                    student=student,
                    current_step=current_step,
                    user_answer=clean_transcript,
                    reason=val_reason
                )
                messages = [{"role": "user", "content": clean_transcript}]
                try:
                    llm_res = await llm_service.generate_response(messages=messages, system_prompt=system_prompt)
                    reply_text = llm_res.text
                    provider_used = f"{llm_res.provider} ({llm_res.model})"
                    is_fallback = llm_res.is_fallback
                    latency_ms = llm_res.latency_ms
                except Exception:
                    reply_text = personalization_service.get_onboarding_clarification_fallback(
                        student=student,
                        current_step=current_step,
                        user_answer=clean_transcript
                    )
            else:
                updated_student = personalization_service.extract_and_update_student_memory(student_id, clean_transcript)
                g_errs, v_suggs, flu, pac, conf = personalization_service.evaluate_utterance(
                    text=clean_transcript,
                    student_id=student_id,
                    session_id=curr_session_id
                )
                grammar_errors = g_errs
                vocab_suggestions = v_suggs
                fluency_score = flu
                pacing_score = pac
                confidence_score = conf

                next_step = personalization_service.determine_next_onboarding_state(updated_student, current_step)
                update_student(student_id, {"onboarding_step": next_step})

                system_prompt = personalization_service.build_onboarding_turn_prompt(
                    student=updated_student,
                    current_step=current_step,
                    next_step=next_step,
                    user_answer=clean_transcript
                )
                messages = [{"role": "user", "content": clean_transcript}]
                try:
                    llm_res = await llm_service.generate_response(messages=messages, system_prompt=system_prompt)
                    reply_text = llm_res.text
                    provider_used = f"{llm_res.provider} ({llm_res.model})"
                    is_fallback = llm_res.is_fallback
                    latency_ms = llm_res.latency_ms
                except Exception:
                    st_name = updated_student.get("name") or "there"
                    if current_step in ["ask_name", "welcome"]:
                        reply_text = f"Nice to meet you, {st_name}! Tell me a bit about yourself—what do you do, and what are your main goals for improving your English?"
                    else:
                        topic = personalization_service.generate_dynamic_assessment_topic(updated_student)
                        reply_text = f"Thanks for sharing that! Now, let's do a quick speaking assessment to calibrate your baseline. Please speak freely for about one minute on this topic: {topic}. Whenever you're finished, just say 'submit', 'done', or pause for a few seconds."
    else:
        # Regular active coaching session
        voice_manager.transition_to(VoiceState.CONVERSATION, "conversation_active")
        detected_mode = personalization_service.detect_intent_and_mode(clean_transcript, current_mode=active_mode)
        g_errs, v_suggs, flu, pac, conf = personalization_service.evaluate_utterance(
            text=clean_transcript,
            student_id=student_id,
            session_id=curr_session_id
        )
        grammar_errors = g_errs
        vocab_suggestions = v_suggs
        fluency_score = flu
        pacing_score = pac
        confidence_score = conf

        personalization_service.extract_and_update_student_memory(student_id, clean_transcript)

        rag_context = ""
        if detected_mode == "tutor" or "teach" in clean_transcript.lower() or "explain" in clean_transcript.lower():
            rag_chunks = rag_service.query_knowledge(clean_transcript, top_k=2)
            if rag_chunks:
                rag_context = "\n---\n".join([c["content"] for c in rag_chunks])

        context = memory_manager.build_llm_context(
            student_id=student_id,
            session_id=curr_session_id,
            current_message=clean_transcript,
            mode=detected_mode,
            rag_context=rag_context,
            max_recent_messages=10
        )

        # Tutor Agent turn or Speech Practice turn or Progressive simplified teaching turn
        if detected_mode == "tutor":
            tutor_turn = await tutor_service.process_tutor_turn(
                student_id=student_id,
                session_id=curr_session_id,
                user_text=clean_transcript,
                current_difficulty=int(student.get("current_teaching_difficulty") or 1)
            )
            reply_text = tutor_turn["reply_text"]
            new_diff = tutor_turn.get("new_difficulty", 1)
            update_student(student_id, {"current_teaching_difficulty": new_diff})
            provider_used = tutor_turn.get("provider", "system")
            is_fallback = False
            latency_ms = 1.0
            print(f"[TUTOR] intent={tutor_turn.get('intent')} topic={tutor_turn.get('current_topic')}", flush=True)
            print(f"[ROUTER] intent=tutor complexity=simple selected={provider_used}", flush=True)
        elif detected_mode == "speech":
            speech_turn = await speech_service.process_speech_turn(
                student_id=student_id,
                session_id=curr_session_id,
                user_text=clean_transcript,
                duration_seconds=measured_duration_seconds if measured_duration_seconds > 0 else None
            )
            reply_text = speech_turn["reply_text"]
            provider_used = speech_turn.get("provider", "system")
            is_fallback = False
            latency_ms = 1.0
            print(f"[SPEECH] state={speech_turn.get('state')} topic={speech_turn.get('topic')}", flush=True)
            print(f"[ROUTER] intent=speech complexity=tier1 selected={provider_used}", flush=True)
        else:
            teaching_turn = personalization_service.get_simplified_teaching_turn(
                student=student,
                user_text=clean_transcript,
                current_difficulty=int(student.get("current_teaching_difficulty") or 1)
            )
            if teaching_turn:
                reply_text = teaching_turn["reply_text"]
                new_diff = teaching_turn.get("new_difficulty", 1)
                update_student(student_id, {"current_teaching_difficulty": new_diff})
                provider_used = "system"
                is_fallback = False
                latency_ms = 1.0
                print(f"[TEACHING] mode={teaching_turn.get('mode')} difficulty={new_diff}", flush=True)
                print(f"[ROUTER] intent=teaching complexity=simple selected=system", flush=True)
            else:
                personalization_service.adapt_teaching_difficulty(
                    student_id=student_id,
                    session_id=curr_session_id,
                    user_utterance=clean_transcript,
                    success=False,
                    struggle=bool(grammar_errors)
                )
                try:
                    llm_res = await llm_service.generate_response(
                        messages=context["messages"],
                        system_prompt=context["system_prompt"]
                    )
                    reply_text = llm_res.text
                    provider_used = f"{llm_res.provider} ({llm_res.model})"
                    is_fallback = llm_res.is_fallback
                    latency_ms = llm_res.latency_ms
                except LLMServiceError as e:
                    voice_manager.transition_to(VoiceState.ERROR_RECOVERY, "llm_error")
                    voice_manager.transition_to(VoiceState.READY, "error_recovered")
                    reply_text = e.user_message
                    provider_used = f"error ({e.category})"
                except Exception:
                    voice_manager.transition_to(VoiceState.ERROR_RECOVERY, "unhandled_error")
                    voice_manager.transition_to(VoiceState.READY, "error_recovered")
                    reply_text = "I am currently unable to reach my AI service. Please check your network and API settings."
                    provider_used = "error"

        personalization_service.record_interaction_metrics(
            student_id=student_id,
            session_id=curr_session_id,
            grammar_errors=grammar_errors,
            vocab_suggestions=vocab_suggestions,
            fluency_score=fluency_score,
            pacing_score=pacing_score,
            confidence_score=confidence_score
        )
        next_step = "completed"

    # Synthesize Speech (16kHz Mono WAV for MAX98357A)
    voice_manager.handle_speaking_start()
    try:
        audio_url, wav_path = await tts_service.synthesize_to_file(reply_text)
    except Exception as tts_err:
        print(f"[TTS Error] Synthesis failed: {tts_err}")
        audio_url, wav_path = None, None
    voice_manager.handle_speaking_ended()

    if not onboarding_active and voice_manager.state != VoiceState.READY:
        voice_manager.transition_to(VoiceState.READY, "turn_complete")

    # Log to conversations database
    add_conversation_message(
        student_id=student_id,
        session_id=curr_session_id,
        role="user",
        content=clean_transcript,
        grammar_errors=grammar_errors,
        vocabulary_suggestions=vocab_suggestions,
        fluency_score=fluency_score,
        pacing_score=pacing_score,
        confidence_score=confidence_score,
        learning_mode="coach" if onboarding_active else detected_mode,
        onboarding_state=current_step if onboarding_active else None,
        provider_used=provider_used,
        is_fallback=is_fallback,
        latency_ms=latency_ms
    )
    add_conversation_message(
        student_id=student_id,
        session_id=curr_session_id,
        role="assistant",
        content=reply_text,
        audio_path=audio_url,
        learning_mode="coach" if onboarding_active else detected_mode,
        onboarding_state=next_step if onboarding_active else None,
        provider_used=provider_used,
        is_fallback=is_fallback,
        latency_ms=latency_ms
    )

    if format == "audio" and wav_path:
        return FileResponse(path=wav_path, media_type="audio/wav", filename="reply.wav")

    latest_student = get_student(student_id) or {}
    updated_session = get_session_by_id(curr_session_id) or {}

    res_dict = {
        "success": True,
        "transcript": transcript,
        "reply_text": reply_text,
        "audio_url": audio_url,
        "learning_mode": "coach" if onboarding_active else (detected_mode if 'detected_mode' in locals() else "coach"),
        "session_id": curr_session_id,
        "provider_used": provider_used,
        "is_fallback": is_fallback,
        "latency_ms": latency_ms,
        "onboarding_active": onboarding_active,
        "onboarding_step": next_step,
        "grammar_errors": grammar_errors,
        "vocabulary_suggestions": vocab_suggestions,
        "session_metrics": updated_session.get("metrics", {}),
        "historical_metrics": {
            "grammar_score": float(latest_student.get("grammar_score") or 0.0),
            "vocabulary_score": float(latest_student.get("vocabulary_score") or 0.0),
            "fluency_score": float(latest_student.get("fluency_score") or 0.0),
            "confidence_score": float(latest_student.get("confidence_score") or 0.0),
        },
        "voice_state": voice_manager.state.value
    }
    if 'assessment_data' in locals() and assessment_data:
        res_dict["assessment"] = assessment_data
    if next_step in ["speech_test_prompt", "speaking_assessment"]:
        res_dict["topic"] = personalization_service.generate_dynamic_assessment_topic(latest_student)
    return res_dict


@router.post("/chat", response_model=ESP32ChatResponse)
async def process_esp32_chat(req: ESP32ChatRequest):
    """
    Lightweight JSON text endpoint for ESP32 or client simulator test:
    Takes text string -> Generates real AI response via Llama 3.3/fallbacks + TTS WAV audio.
    """
    transcript = req.message.strip()
    student_id = PROTOTYPE_STUDENT_ID
    session_id = req.session_id

    # 1. Classify Command or Intent BEFORE standard LLM/dialogue processing
    clean_transcript = transcript.strip() if transcript else ""
    cmd_intent = classify_command_or_intent(clean_transcript)

    # 2. Check for "Stop Mizo" control command
    if clean_transcript and (cmd_intent == "STOP" or is_stop_command(clean_transcript)):
        voice_manager.handle_stop_command()
        return ESP32ChatResponse(
            transcript=transcript,
            reply_text="",
            audio_url=None,
            learning_mode="coach",
            session_id=session_id or "ses_default",
            provider_used="system",
            is_fallback=False,
            onboarding_active=False,
            onboarding_step="completed",
            is_control_command=True,
            command="stop",
            voice_state="STANDBY"
        )

    # 3. Check student & onboarding state
    student = _require_primary_student()

    st_name = (student.get("name") or "").strip()
    is_name_known = bool(st_name and st_name.lower() not in ["new learner", "student", "there", "unknown", ""])
    onboarding_active = not bool(student.get("onboarding_completed", False))
    current_step = (student.get("onboarding_step") or ("completed" if not onboarding_active else "ask_name")).lower()
    if onboarding_active and is_name_known and current_step in ["ask_name", "new_student", "welcome"]:
        current_step = "ask_goals"

    # 4. Check WAKE_LISTENING / SNOOZED / STANDBY / PAUSED State Machine
    is_typed = bool(getattr(req, "is_typed_text", False))
    is_idle_state = voice_manager.state in [
        VoiceState.WAKE_LISTENING, VoiceState.STANDBY,
        VoiceState.SLEEP, VoiceState.SNOOZED, VoiceState.PAUSED, VoiceState.WAKE_ONLY
    ]

    if is_idle_state:
        is_wake, trailing_speech = is_wake_word(clean_transcript, log_debug=False)
        is_explicit_resume = clean_transcript.strip().lower() in ["resume", "continue", "start"]
        if is_wake or is_explicit_resume:
            voice_manager.log(f'Heard: "{clean_transcript}"')
            voice_manager.log("Wake word detected: Mizo")
            voice_manager.log("Wake word detected")
            voice_manager.log("Waking assistant")
            voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "wake_word_detected")
            if trailing_speech:
                clean_transcript = trailing_speech
                voice_manager.transition_to(VoiceState.ONBOARDING if onboarding_active else VoiceState.ACTIVE_CONVERSATION, "wake_with_speech")
            else:
                voice_manager.log("Listening for user request")
                if is_name_known or student.get("onboarding_completed", False):
                    reply_msg = f"Yes, {st_name}? I'm listening." if is_name_known else "Yes, I'm listening."
                    voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "wake_ready")
                elif onboarding_active and current_step in ["ask_name", "welcome"]:
                    reply_msg = "Hi! I'm Mizo. Before we begin, I'd like to get to know you. What's your name?"
                    voice_manager.transition_to(VoiceState.ONBOARDING, "start_onboarding")
                    active_session = resolve_prototype_session(None, default_mode=req.mode or "coach")
                    curr_session_id = active_session["session_id"]
                    add_conversation_message(
                        student_id=student_id,
                        session_id=curr_session_id,
                        role="assistant",
                        content=reply_msg,
                        learning_mode="coach",
                        onboarding_state="ask_name"
                    )
                else:
                    reply_msg = "Yes, I'm listening."
                    voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "wake_ready")

                audio_url, _ = await tts_service.synthesize_to_file(reply_msg)
                return ESP32ChatResponse(
                    transcript=transcript,
                    reply_text=reply_msg,
                    audio_url=audio_url,
                    learning_mode="coach",
                    session_id=session_id or "ses_default",
                    provider_used="system",
                    is_fallback=False,
                    onboarding_active=onboarding_active,
                    onboarding_step=current_step,
                    is_control_command=False,
                    voice_state="ACTIVE",
                    active_section="normal_chat",
                    route="NORMAL_CHAT"
                )
        elif is_typed:
            voice_manager.log(f"Typed message received in {voice_manager.state.value}. Auto-waking assistant.")
            voice_manager.transition_to(VoiceState.ONBOARDING if onboarding_active else VoiceState.ACTIVE_LISTENING, "typed_text_wake")
        else:
            voice_manager.log(f'Heard: "{clean_transcript}"')
            voice_manager.log("No wake word. Ignoring.")
            voice_manager.log("Wake word not detected. Ignoring.")
            return ESP32ChatResponse(
                transcript=transcript,
                reply_text="",
                audio_url=None,
                learning_mode="coach",
                session_id=session_id or "ses_default",
                provider_used="system",
                is_fallback=False,
                onboarding_active=onboarding_active,
                onboarding_step=current_step,
                is_control_command=False,
                voice_state="STANDBY"
            )
    elif voice_manager.state == VoiceState.SPEAKING:
        voice_manager.handle_barge_in(clean_transcript)
        if cmd_intent == "STOP" or is_stop_command(clean_transcript):
            voice_manager.handle_stop_command()
            return ESP32ChatResponse(
                transcript=transcript,
                reply_text="",
                audio_url=None,
                learning_mode="coach",
                session_id=session_id or "ses_default",
                provider_used="system",
                is_fallback=False,
                onboarding_active=False,
                onboarding_step="completed",
                is_control_command=True,
                command="stop",
                voice_state="STANDBY"
            )
    else:
        is_wake, trailing_speech = is_wake_word(clean_transcript, log_debug=False)
        if is_wake and trailing_speech:
            clean_transcript = trailing_speech

    # 5. Retrieve or create active session
    active_session = resolve_prototype_session(session_id, default_mode=req.mode or "coach")
    if not active_session:
        raise HTTPException(status_code=503, detail="No active session is available for Student #1")
    curr_session_id = active_session["session_id"]
    active_mode = req.mode or active_session.get("mode") or "coach"

    # 5-0. Non-English Detection (Tier 0 deterministic response)
    is_non_eng, non_eng_reason = is_non_english_input(clean_transcript)
    if is_non_eng:
        reply_text = "I didn't understand that. Could you say it in English?"
        voice_manager.handle_speaking_start()
        audio_url, _ = await tts_service.synthesize_to_file(reply_text)
        voice_manager.handle_speaking_ended()
        voice_manager.transition_to(VoiceState.ONBOARDING if onboarding_active else VoiceState.READY, "non_english_rejected")
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="user", content=clean_transcript, learning_mode=active_mode)
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="assistant", content=reply_text, audio_path=audio_url, learning_mode=active_mode)
        return ESP32ChatResponse(
            transcript=transcript,
            reply_text=reply_text,
            audio_url=audio_url,
            learning_mode=active_mode,
            session_id=curr_session_id,
            provider_used="system",
            is_fallback=False,
            onboarding_active=onboarding_active,
            onboarding_step=current_step,
            voice_state=voice_manager.state.value
        )

    is_known_name = bool(student.get("name") and student.get("name").strip().lower() not in ["new learner", "student", "there", "unknown", ""])
    pending_name = get_pending_name_change(curr_session_id)
    name_change_reply = None
    if is_known_name or pending_name or not onboarding_active:
        name_change_reply = _handle_name_change(student, curr_session_id, clean_transcript, cmd_intent)

    if name_change_reply:
        voice_manager.handle_speaking_start()
        audio_url, _ = await tts_service.synthesize_to_file(name_change_reply)
        voice_manager.handle_speaking_ended()
        voice_manager.transition_to(VoiceState.ONBOARDING if onboarding_active else VoiceState.READY, "name_change_pending")
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="user", content=clean_transcript, learning_mode=active_mode)
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="assistant", content=name_change_reply, audio_path=audio_url, learning_mode=active_mode)
        return ESP32ChatResponse(transcript=transcript, reply_text=name_change_reply, audio_url=audio_url,
                                 learning_mode=active_mode, session_id=curr_session_id, provider_used="system",
                                 is_fallback=False, onboarding_active=onboarding_active, onboarding_step=current_step,
                                 voice_state=voice_manager.state.value)

    # 6. Identity & Name Commands (Interception before regular dialogue)
    # 6A. Assistant Identity Query ("What is your name?")
    if cmd_intent == "IDENTITY_QUERY":
        reply_text = "My name is Mizo, your English communication coach."
        if onboarding_active:
            if current_step in ["ask_name", "welcome"]:
                reply_text += " What's your name?"
            elif current_step in ["ask_goals", "ask_problems"]:
                reply_text += " Tell me a bit about yourself—what are your main goals for improving your English?"
            elif current_step in ["speech_test_prompt", "speaking_assessment"]:
                reply_text += " Please speak freely for about one minute on your favorite project or hobby."
        voice_manager.handle_speaking_start()
        audio_url, _ = await tts_service.synthesize_to_file(reply_text)
        voice_manager.handle_speaking_ended()
        voice_manager.transition_to(VoiceState.ONBOARDING if onboarding_active else VoiceState.READY, "identity_answered")
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="user", content=clean_transcript, learning_mode=active_mode)
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="assistant", content=reply_text, audio_path=audio_url, learning_mode=active_mode)
        return ESP32ChatResponse(
            transcript=transcript,
            reply_text=reply_text,
            audio_url=audio_url,
            learning_mode=active_mode,
            session_id=curr_session_id,
            provider_used="system",
            is_fallback=False,
            onboarding_active=onboarding_active,
            onboarding_step=current_step,
            voice_state=voice_manager.state.value
        )

    # 6B. Student Name Query ("What is my name?" / "Who am I?")
    if cmd_intent == "USER_NAME_QUERY":
        st_name = student.get("name")
        if st_name and st_name not in ["New Learner", "Student", "there", "Unknown", ""]:
            reply_text = f"Your name is {st_name}."
        else:
            reply_text = "I don't have your name yet. What should I call you?"
        if onboarding_active and current_step in ["ask_goals", "ask_problems"]:
            reply_text += " Tell me a bit about yourself—what are your main goals for improving your English?"
        voice_manager.handle_speaking_start()
        audio_url, _ = await tts_service.synthesize_to_file(reply_text)
        voice_manager.handle_speaking_ended()
        voice_manager.transition_to(VoiceState.ONBOARDING if onboarding_active else VoiceState.READY, "user_name_answered")
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="user", content=clean_transcript, learning_mode=active_mode)
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="assistant", content=reply_text, audio_path=audio_url, learning_mode=active_mode)
        return ESP32ChatResponse(
            transcript=transcript,
            reply_text=reply_text,
            audio_url=audio_url,
            learning_mode=active_mode,
            session_id=curr_session_id,
            provider_used="system",
            is_fallback=False,
            onboarding_active=onboarding_active,
            onboarding_step=current_step,
            voice_state=voice_manager.state.value
        )

    # 6C. Tier 0 Centralized Deterministic Resolution (Profile queries, Session commands, etc.)
    # In tutor/study mode, lesson control commands (simplify, example, quiz) belong to the Study Agent.
    tier0_res = None
    if active_mode != "tutor" and getattr(req, "mode", None) != "tutor":
        tier0_res = llm_service.resolve_tier0(clean_transcript, student_data=student, session_id=curr_session_id)
    if tier0_res and tier0_res.get("handled"):
        reply_text = tier0_res["reply_text"]
        action = tier0_res.get("action")
        if action == "reset_session":
            from app.db.database import get_connection
            conn = get_connection()
            conn.cursor().execute("DELETE FROM conversation_messages WHERE session_id = ?", (curr_session_id,))
            conn.commit()
            conn.close()
        elif action == "stop":
            voice_manager.handle_stop_command()

        voice_manager.handle_speaking_start()
        audio_url, _ = await tts_service.synthesize_to_file(reply_text)
        voice_manager.handle_speaking_ended()
        voice_manager.transition_to(VoiceState.ONBOARDING if onboarding_active else VoiceState.READY, "tier0_handled")
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="user", content=clean_transcript, learning_mode=active_mode)
        add_conversation_message(student_id=student_id, session_id=curr_session_id, role="assistant", content=reply_text, audio_path=audio_url, learning_mode=active_mode)
        return ESP32ChatResponse(
            transcript=transcript,
            reply_text=reply_text,
            audio_url=audio_url,
            learning_mode=active_mode,
            session_id=curr_session_id,
            provider_used="system",
            is_fallback=False,
            onboarding_active=onboarding_active,
            onboarding_step=current_step,
            voice_state=voice_manager.state.value
        )

    # 7. Processing user input
    voice_manager.log(f'Heard: "{clean_transcript}"')
    voice_manager.log("End of speech detected.")
    voice_manager.transition_to(VoiceState.PROCESSING, "end_of_speech")

    # Track conversation topic, subtopic, entities & lightweight intent (Sections 21-23, 26, 28)
    from app.services.context_service import update_conversation_topic_state
    conv_context = update_conversation_topic_state(
        session_id=curr_session_id,
        student_id=student_id,
        user_text=clean_transcript
    )

    detected_mode = personalization_service.detect_intent_and_mode(clean_transcript, current_mode=active_mode)
    agent_name = personalization_service.get_agent_name(detected_mode)
    voice_manager.selected_agent = agent_name
    voice_manager.log(f"Agent selected: {agent_name}")

    grammar_errors = []
    vocab_suggestions = []
    fluency_score = 75.0
    pacing_score = 80.0
    confidence_score = 75.0
    reply_text = ""
    provider_used = "ollama"
    is_fallback = False
    latency_ms = 0.0

    if onboarding_active:
        voice_manager.transition_to(VoiceState.ONBOARDING, "onboarding_active")
        is_assessment_step = current_step in [personalization_service.STATE_ASSESSMENT, "SPEECH_TEST_PROMPT", "SPEECH_EVALUATION", "speech_test_prompt", "speaking_assessment"] or cmd_intent == "SUBMIT"

        if is_assessment_step:
            words_in_transcript = len(clean_transcript.split())
            prev_cum_words = int(student.get("cumulative_assessment_words") or 0)
            prev_samples_cnt = int(student.get("assessment_samples_count") or 0)
            cum_words = prev_cum_words + words_in_transcript
            samples_cnt = prev_samples_cnt + 1

            update_student(student_id, {
                "cumulative_assessment_words": cum_words,
                "assessment_samples_count": samples_cnt
            })

            # Check if this single sample is very brief and cumulative words is under 40 and first sample
            if words_in_transcript < 12 and cum_words < 40 and samples_cnt <= 1:
                retry_msg = "Please share a little more about yourself or your projects so I can calibrate our lessons accurately."
                audio_url, _ = await tts_service.synthesize_to_file(retry_msg)
                voice_manager.transition_to(VoiceState.ASSESSMENT, "retry_assessment")
                return ESP32ChatResponse(
                    transcript=transcript,
                    reply_text=retry_msg,
                    audio_url=audio_url,
                    learning_mode="coach",
                    session_id=curr_session_id,
                    provider_used="system",
                    onboarding_active=True,
                    onboarding_step="speech_test_prompt",
                    topic=personalization_service.generate_dynamic_assessment_topic(student),
                    detail="Preliminary sample recorded. Additional evidence requested.",
                    voice_state=voice_manager.state.value
                )

            # Multiple samples or sufficient evidence reached! Finalize preliminary assessment.
            voice_manager.transition_to(VoiceState.PROCESSING_ASSESSMENT, "evaluate_assessment")
            assessment_data = await personalization_service.run_speaking_assessment(
                transcript=clean_transcript,
                student_id=student_id,
                session_id=curr_session_id,
                topic=personalization_service.generate_dynamic_assessment_topic(student),
                duration_seconds=float(getattr(req, "duration_seconds", 0.0) or 0.0) if getattr(req, "duration_seconds", None) else None,
                audio_metadata=getattr(req, "audio_metadata", None) or getattr(req, "stt_metadata", None)
            )

            if assessment_data.get("invalid_input"):
                retry_msg = assessment_data.get("spoken_summary") or "Speech recognition quality was insufficient. Please speak clearly for about one minute."
                audio_url, _ = await tts_service.synthesize_to_file(retry_msg)
                voice_manager.transition_to(VoiceState.ASSESSMENT, "retry_invalid_assessment")
                return ESP32ChatResponse(
                    transcript=transcript,
                    reply_text=retry_msg,
                    audio_url=audio_url,
                    learning_mode="coach",
                    session_id=curr_session_id,
                    provider_used="system",
                    onboarding_active=True,
                    onboarding_step="speech_test_prompt",
                    topic=personalization_service.generate_dynamic_assessment_topic(student),
                    detail=assessment_data.get("quality_reason", "Low quality or inaudible input"),
                    voice_state=voice_manager.state.value
                )

            reply_text = assessment_data.get("spoken_summary", "Calibration complete! Your speaking assessment is saved.")
            onboarding_active = False
            next_step = "completed"
            update_student(student_id, {"onboarding_completed": True, "onboarding_step": "completed"})
            voice_manager.transition_to(VoiceState.ASSESSMENT_COMPLETE, "assessment_done")
            voice_manager.transition_to(VoiceState.READY, "ready_post_assessment")
        else:
            is_valid_ans, val_reason = personalization_service.is_valid_onboarding_response(student, current_step, clean_transcript)
            if not is_valid_ans:
                next_step = current_step
                system_prompt = personalization_service.build_onboarding_clarification_prompt(
                    student=student,
                    current_step=current_step,
                    user_answer=clean_transcript,
                    reason=val_reason
                )
                messages = [{"role": "user", "content": clean_transcript}]
                try:
                    llm_res = await llm_service.generate_response(messages=messages, system_prompt=system_prompt)
                    reply_text = llm_res.text
                    provider_used = f"{llm_res.provider} ({llm_res.model})"
                    is_fallback = llm_res.is_fallback
                    latency_ms = llm_res.latency_ms
                except Exception:
                    reply_text = personalization_service.get_onboarding_clarification_fallback(
                        student=student,
                        current_step=current_step,
                        user_answer=clean_transcript
                    )
            else:
                updated_student = personalization_service.extract_and_update_student_memory(student_id, clean_transcript)
                g_errs, v_suggs, flu, pac, conf = personalization_service.evaluate_utterance(
                    text=clean_transcript,
                    student_id=student_id,
                    session_id=curr_session_id
                )
                grammar_errors = g_errs
                vocab_suggestions = v_suggs
                fluency_score = flu
                pacing_score = pac
                confidence_score = conf

                next_step = personalization_service.determine_next_onboarding_state(updated_student, current_step)
                update_student(student_id, {"onboarding_step": next_step})

                system_prompt = personalization_service.build_onboarding_turn_prompt(
                    student=updated_student,
                    current_step=current_step,
                    next_step=next_step,
                    user_answer=clean_transcript
                )
                messages = [{"role": "user", "content": clean_transcript}]
                try:
                    llm_res = await llm_service.generate_response(messages=messages, system_prompt=system_prompt)
                    reply_text = llm_res.text
                    provider_used = f"{llm_res.provider} ({llm_res.model})"
                    is_fallback = llm_res.is_fallback
                    latency_ms = llm_res.latency_ms
                except Exception:
                    st_name = updated_student.get("name") or "there"
                    if current_step in ["ask_name", "welcome"]:
                        reply_text = f"Nice to meet you, {st_name}! Tell me a bit about yourself—what do you do, and what are your main goals for improving your English?"
                    else:
                        topic = personalization_service.generate_dynamic_assessment_topic(updated_student)
                        reply_text = f"Thanks for sharing that! Now, let's do a quick speaking assessment to calibrate your baseline. Please speak freely for about one minute on this topic: {topic}. Whenever you're finished, just say 'submit', 'done', or pause for a few seconds."
    else:
        # Regular active session
        voice_manager.transition_to(VoiceState.CONVERSATION, "conversation_active")
        
        # Determine destination agent via AgentRouter (Sections 1-4, 10, 11)
        route_decision = agent_router.route_intent(clean_transcript, current_route=active_mode)
        selected_route = route_decision["route"]

        active_section = "normal_chat"
        detected_mode = "coach"

        is_explicit_normal_chat = any(re.search(p, clean_transcript.lower()) for p in AgentRouter.NORMAL_CHAT_PATTERNS)
        is_explicit_speech = any(re.search(p, clean_transcript.lower()) for p in AgentRouter.SPEECH_PATTERNS)

        if selected_route == AgentRoute.AMBIGUOUS.value:
            reply_text = route_decision["clarification_prompt"]
            provider_used = "system"
            is_fallback = False
            latency_ms = 1.0
            active_section = "main_chat"
            detected_mode = "coach"
        elif (
            selected_route == AgentRoute.STUDY.value 
            or (req.mode == "tutor" and not is_explicit_normal_chat and not is_explicit_speech)
            or (tutor_service.has_active_quiz(curr_session_id) and not is_explicit_normal_chat and not is_explicit_speech)
        ):
            selected_route = AgentRoute.STUDY.value
            active_section = "study"
            detected_mode = "tutor"
            tutor_turn = await tutor_service.process_tutor_turn(
                student_id=student_id,
                session_id=curr_session_id,
                user_text=clean_transcript,
                current_difficulty=int(student.get("current_teaching_difficulty") or 1)
            )
            reply_text = tutor_turn["reply_text"]
            new_diff = tutor_turn.get("new_difficulty", 1)
            update_student(student_id, {"current_teaching_difficulty": new_diff})
            provider_used = tutor_turn.get("provider", "system")
            is_fallback = False
            latency_ms = 1.0
            print(f"[TUTOR] intent={tutor_turn.get('intent')} topic={tutor_turn.get('current_topic')}", flush=True)
            print(f"[ROUTER] intent=tutor complexity=simple selected={provider_used}", flush=True)
        elif selected_route == AgentRoute.SPEECH.value or (req.mode == "speech" and not is_explicit_normal_chat and selected_route != AgentRoute.STUDY.value):
            selected_route = AgentRoute.SPEECH.value
            active_section = "speech"
            detected_mode = "speech"
            speech_turn = await speech_service.process_speech_turn(
                student_id=student_id,
                session_id=curr_session_id,
                user_text=clean_transcript,
                duration_seconds=req.duration_seconds
            )
            reply_text = speech_turn["reply_text"]
            provider_used = speech_turn.get("provider", "system")
            is_fallback = False
            latency_ms = 1.0
            print(f"[SPEECH] state={speech_turn.get('state')} topic={speech_turn.get('topic')}", flush=True)
            print(f"[ROUTER] intent=speech complexity=tier1 selected={provider_used}", flush=True)
        else:
            # NORMAL_CHAT Agent (Sections 2A, 6, 7)
            selected_route = AgentRoute.NORMAL_CHAT.value
            active_section = "normal_chat"
            detected_mode = "coach"
            
            # Check for simplified teaching turn if previously explicitly requested in legacy context
            teaching_turn = personalization_service.get_simplified_teaching_turn(
                student=student,
                user_text=clean_transcript,
                current_difficulty=int(student.get("current_teaching_difficulty") or 1)
            )
            if teaching_turn:
                reply_text = teaching_turn["reply_text"]
                new_diff = teaching_turn.get("new_difficulty", 1)
                update_student(student_id, {"current_teaching_difficulty": new_diff})
                provider_used = "system"
                is_fallback = False
                latency_ms = 1.0
            else:
                chat_turn = await normal_chat_service.process_chat_turn(
                    student_id=student_id,
                    session_id=curr_session_id,
                    user_text=clean_transcript,
                    is_typed_text=is_typed
                )
                reply_text = chat_turn["reply_text"]
                provider_used = chat_turn.get("provider_used", "ollama")
                is_fallback = chat_turn.get("is_fallback", False)
                latency_ms = chat_turn.get("latency_ms", 0.0)
                grammar_errors = chat_turn.get("grammar_errors", [])
                vocab_suggestions = chat_turn.get("vocab_suggestions", [])
                fluency_score = chat_turn.get("fluency_score", 75.0)
                pacing_score = chat_turn.get("pacing_score", 80.0)
                confidence_score = chat_turn.get("confidence_score", 75.0)

        personalization_service.record_interaction_metrics(
            student_id=student_id,
            session_id=curr_session_id,
            grammar_errors=grammar_errors,
            vocab_suggestions=vocab_suggestions,
            fluency_score=fluency_score,
            pacing_score=pacing_score,
            confidence_score=confidence_score
        )
        next_step = "completed"

    # Synthesize Speech Audio (16kHz WAV)
    voice_manager.handle_speaking_start()
    try:
        audio_url, _ = await tts_service.synthesize_to_file(reply_text)
    except Exception as tts_err:
        print(f"[TTS Error] Synthesis failed: {tts_err}")
        audio_url = None
    voice_manager.handle_speaking_ended()

    if not onboarding_active and voice_manager.state != VoiceState.READY:
        voice_manager.transition_to(VoiceState.READY, "chat_turn_complete")

    # Log to conversation history
    add_conversation_message(
        student_id=student_id,
        session_id=curr_session_id,
        role="user",
        content=transcript,
        grammar_errors=grammar_errors,
        vocabulary_suggestions=vocab_suggestions,
        fluency_score=fluency_score,
        pacing_score=pacing_score,
        confidence_score=confidence_score,
        learning_mode="coach" if onboarding_active else detected_mode,
        onboarding_state=current_step if onboarding_active else None,
        provider_used=provider_used,
        is_fallback=is_fallback,
        latency_ms=latency_ms
    )
    add_conversation_message(
        student_id=student_id,
        session_id=curr_session_id,
        role="assistant",
        content=reply_text,
        audio_path=audio_url,
        learning_mode="coach" if onboarding_active else detected_mode,
        onboarding_state=next_step if onboarding_active else None,
        provider_used=provider_used,
        is_fallback=is_fallback,
        latency_ms=latency_ms
    )

    latest_student = get_student(student_id) or {}
    updated_session = get_session_by_id(curr_session_id) or {}
    corrections = [f"{e['error']} ➔ {e['correction']}" for e in grammar_errors]

    resp_kwargs = {
        "transcript": transcript,
        "reply_text": reply_text,
        "audio_url": audio_url,
        "learning_mode": "coach" if onboarding_active else (detected_mode if 'detected_mode' in locals() else "coach"),
        "session_id": curr_session_id,
        "provider_used": provider_used,
        "is_fallback": is_fallback,
        "onboarding_active": onboarding_active,
        "onboarding_step": next_step,
        "corrections": corrections,
        "vocab_suggestions": vocab_suggestions,
        "session_metrics": updated_session.get("metrics", {}),
        "historical_metrics": {
            "grammar_score": float(latest_student.get("grammar_score") or 0.0),
            "vocabulary_score": float(latest_student.get("vocabulary_score") or 0.0),
            "fluency_score": float(latest_student.get("fluency_score") or 0.0),
            "confidence_score": float(latest_student.get("confidence_score") or 0.0),
        },
        "voice_state": voice_manager.state.value,
        "route": selected_route if 'selected_route' in locals() else ("ONBOARDING" if onboarding_active else "NORMAL_CHAT"),
        "active_section": active_section if 'active_section' in locals() else ("onboarding" if onboarding_active else "normal_chat")
    }
    if 'assessment_data' in locals() and assessment_data:
        resp_kwargs["assessment"] = assessment_data
    if next_step in ["speech_test_prompt", "speaking_assessment"]:
        resp_kwargs["topic"] = personalization_service.generate_dynamic_assessment_topic(latest_student)

    return ESP32ChatResponse(**resp_kwargs)
