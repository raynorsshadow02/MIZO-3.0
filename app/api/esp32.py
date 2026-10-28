import os
import re
import uuid
import time
import io
import wave
import struct
import math
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from contextlib import contextmanager
from fastapi import APIRouter, UploadFile, File, Form, Header, HTTPException, Query, Response, Request, Depends
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger("mizo.esp32")

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
from app.db.models import (
    ESP32ChatRequest,
    ESP32ChatResponse,
    ESP32StartRequest,
    ESP32StartResponse,
    ESP32ManualActivateRequest,
    ESP32ManualActivateResponse
)
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
from app.services.voice_service import (
    is_wake_word,
    is_stop_command,
    is_submit_command,
    is_assessment_skip_command,
    is_unclear_or_low_confidence_speech,
    classify_command_or_intent,
    voice_manager,
    VoiceState,
    detect_mizo_intent,
    ASSISTANT_NAME
)

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


_esp32_hardware_state = {
    "last_seen_timestamp": None,
    "last_ip": None,
    "last_activity": None
}


def record_esp32_activity(client_ip: Optional[str] = None, activity: str = "ping"):
    """Records the most recent communication from the physical ESP32-S3 hardware."""
    _esp32_hardware_state["last_seen_timestamp"] = time.time()
    _esp32_hardware_state["last_activity"] = activity
    if client_ip:
        _esp32_hardware_state["last_ip"] = client_ip


@router.get("/ping")
async def ping(request: Request):
    """Healthcheck endpoint for ESP32 Wi-Fi heartbeat."""
    client_ip = request.client.host if request.client else None
    record_esp32_activity(client_ip, "heartbeat")
    return {
        "status": "online",
        "server": settings.APP_NAME,
        "version": "3.0.0",
        "hardware_input": "INMP441 Microphone (I2S)",
        "hardware_output": "MAX98357A Speaker DAC (I2S)",
        "laptop_fallback": False
    }


@router.get("/hardware_status")
async def get_hardware_status():
    """
    Returns real-time connection status of the ESP32-S3 hardware.
    Mizo exclusively uses the ESP32-S3 INMP441 mic and MAX98357A speaker for audio.
    Laptop audio fallback is disabled.
    """
    now = time.time()
    last_seen = _esp32_hardware_state.get("last_seen_timestamp")
    seconds_ago = round(now - last_seen, 1) if last_seen else None
    # Connected if seen within the last 45 seconds
    is_connected = bool(seconds_ago is not None and seconds_ago < 45.0)

    return {
        "connected": is_connected,
        "exclusive_mode": True,
        "input_device": "ESP32-S3 INMP441 Microphone (I2S)",
        "output_device": "ESP32-S3 MAX98357A Speaker DAC (I2S)",
        "laptop_microphone_enabled": False,
        "laptop_speaker_enabled": False,
        "laptop_fallback_enabled": False,
        "last_seen_seconds_ago": seconds_ago,
        "last_ip": _esp32_hardware_state.get("last_ip"),
        "last_activity": _esp32_hardware_state.get("last_activity"),
        "status_message": (
            "ESP32-S3 hardware connected. INMP441 microphone and MAX98357A speaker active."
            if is_connected
            else "ESP32-S3 hardware is disconnected or unavailable. Audio input and output are exclusive to the ESP32-S3. Laptop audio fallback is disabled."
        )
    }


@router.get("/speaker_test")
@router.get("/test_tone")
async def get_speaker_test_audio():
    """
    Returns an enhanced 16kHz 16-bit Mono WAV test tone (1-second 440Hz sine wave)
    with smooth anti-pop envelopes and clean amplitude for testing MAX98357A speaker clarity.
    """
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    freq = 440.0
    fade_samples = int(sample_rate * 0.05)  # 50ms anti-pop ramp

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(num_samples):
            # Smooth envelope at start and finish
            env = 1.0
            if i < fade_samples:
                env = i / fade_samples
            elif i > num_samples - fade_samples:
                env = (num_samples - i) / fade_samples
            val = int(math.sin(2 * math.pi * freq * (i / sample_rate)) * 22000 * env)
            frames.extend(struct.pack('<h', val))
        wf.writeframes(frames)
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speaker_test_440hz.wav"'}
    )


@router.get("/audio/cache/{filename}")
async def get_cached_audio(filename: str):
    """Serves synthesized speech audio files for ESP32 and browser playback."""
    clean_filename = os.path.basename(filename)
    file_path = settings.AUDIO_CACHE_DIR / clean_filename
    if not file_path.exists():
        logger.warning(f"[Audio Serve] Requested audio file not found: {clean_filename}")
        raise HTTPException(status_code=404, detail="Audio file not found")

    media_type = "audio/wav"
    try:
        with open(file_path, "rb") as f:
            header = f.read(12)
            if header.startswith(b"ID3") or (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
                media_type = "audio/mpeg"
            elif header.startswith(b"RIFF") and b"WAVE" in header:
                media_type = "audio/wav"
    except Exception:
        pass

    logger.info(f"[Audio Serve] Serving audio file: {clean_filename} (size={file_path.stat().st_size} bytes, type={media_type})")
    return FileResponse(path=str(file_path), media_type=media_type, filename=clean_filename)


@router.get("/voice_state")
async def get_voice_state():
    """Returns current voice state, microphone active status, and onboarding progress."""
    voice_manager.check_inactivity_timeout()
    onboarding_active = False
    onboarding_step = "completed"
    try:
        student = get_prototype_student()
        if student:
            onboarding_active = not bool(student.get("onboarding_completed", False))
            onboarding_step = student.get("onboarding_step") or "ask_name"
    except Exception:
        pass

    return {
        "success": True,
        "voice_state": voice_manager.state.value,
        "microphone_active": voice_manager.microphone_active,
        "manual_activation_active": voice_manager.is_manual_activation_active(),
        "hardware_test_mode": getattr(settings, "HARDWARE_TEST_MODE", False),
        "assistant_name": ASSISTANT_NAME,
        "onboarding_active": onboarding_active,
        "onboarding_step": onboarding_step
    }


@router.get("/test_mode")
@router.get("/hardware_test_mode")
async def get_hardware_test_mode_status():
    """Returns the current hardware test mode configuration status."""
    return {
        "hardware_test_mode": getattr(settings, "HARDWARE_TEST_MODE", False),
        "description": "When active, POST /api/v1/esp32/audio processes speech directly without requiring a wake word."
    }


@router.post("/manual_activate", response_model=ESP32ManualActivateResponse)
async def manual_voice_activate(
    request: Request,
    req: Optional[ESP32ManualActivateRequest] = None,
    authenticated: bool = Depends(verify_device)
):
    """
    Manually activates voice interaction override from dashboard button.
    Bypasses wake-word requirement for the next valid speech capture from ESP32-S3 hardware.
    Consumes manual activation once audio is processed, or times out safely.
    """
    client_ip = request.client.host if (request and request.client) else None
    record_esp32_activity(client_ip, "manual_activate")

    action = (req.action if req and req.action else "activate").lower()
    timeout = float(req.timeout_seconds if req and req.timeout_seconds else 30.0)

    if action == "toggle":
        action = "deactivate" if voice_manager.is_manual_activation_active() else "activate"

    if action == "deactivate":
        voice_manager.deactivate_manual_mode("user_manual_deactivate")
        return ESP32ManualActivateResponse(
            success=True,
            active=False,
            voice_state=voice_manager.state.value,
            message="Manual voice activation cancelled. Standby wake listening restored.",
            timeout_seconds=0.0
        )

    voice_manager.activate_manual_mode(timeout_seconds=timeout)
    return ESP32ManualActivateResponse(
        success=True,
        active=True,
        voice_state=voice_manager.state.value,
        message="Manual voice activation enabled. Speak into the INMP441 microphone (wake word not required).",
        timeout_seconds=timeout
    )


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
    now = time.time()
    last_seen = _esp32_hardware_state.get("last_seen_timestamp")
    is_connected = bool(last_seen and (now - last_seen) < 45.0)
    return {
        "status": "ready",
        "service": "Mizo 3.0 ESP32 Hardware Audio Pipeline",
        "voice_state": voice_manager.state.value,
        "exclusive_hardware": True,
        "hardware_connected": is_connected,
        "hardware_test_mode": getattr(settings, "HARDWARE_TEST_MODE", False),
        "input_device": "INMP441 I2S Microphone (ESP32-S3)",
        "output_device": "MAX98357A I2S Speaker (ESP32-S3)",
        "laptop_fallback_enabled": False,
        "instructions": "Send POST multipart/form-data with 'audio' (.wav) to interact with Mizo.",
        "supported_formats": ["wav", "pcm", "mp3"]
    }


def _analyze_audio_signal(audio_bytes: bytes) -> Dict[str, Any]:
    """
    Non-destructive diagnostic analyzer for incoming audio from ESP32 INMP441:
    Measures duration, sample rate, bit depth, channel count, peak amplitude, and RMS.
    Ensures safe logging without leaking raw audio or conversation content.
    """
    info = {
        "format": "unknown",
        "size_bytes": len(audio_bytes) if audio_bytes else 0,
        "channels": 0,
        "sample_rate": 0,
        "bit_depth": 0,
        "duration_seconds": 0.0,
        "peak": 0,
        "rms": 0.0,
        "dbfs": -99.0,
        "is_silent": True,
    }
    if not audio_bytes:
        return info

    if audio_bytes.startswith(b"RIFF") and len(audio_bytes) >= 44:
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                info["format"] = "wav"
                info["channels"] = wf.getnchannels()
                sample_width = wf.getsampwidth()
                info["bit_depth"] = sample_width * 8
                info["sample_rate"] = wf.getframerate()
                nframes = wf.getnframes()
                info["duration_seconds"] = nframes / float(info["sample_rate"]) if info["sample_rate"] > 0 else 0.0

                raw_frames = wf.readframes(nframes)
                if sample_width == 2 and raw_frames:
                    sample_count = len(raw_frames) // 2
                    step = max(1, sample_count // 16000)
                    samples = struct.unpack(f"<{sample_count}h", raw_frames)[::step]
                    if samples:
                        peak = max(abs(s) for s in samples)
                        sum_sq = sum(s * s for s in samples)
                        rms = math.sqrt(sum_sq / len(samples))
                        dbfs = 20 * math.log10(rms / 32768.0) if rms > 0 else -99.0
                        info["peak"] = peak
                        info["rms"] = round(rms, 2)
                        info["dbfs"] = round(dbfs, 1)
                        info["is_silent"] = (rms < 120.0 and peak < 350)
        except Exception as e:
            info["format"] = f"wav_parse_error: {e}"
    else:
        if len(audio_bytes) % 2 == 0:
            sample_count = len(audio_bytes) // 2
            step = max(1, sample_count // 16000)
            try:
                samples = struct.unpack(f"<{sample_count}h", audio_bytes)[::step]
                if samples:
                    peak = max(abs(s) for s in samples)
                    sum_sq = sum(s * s for s in samples)
                    rms = math.sqrt(sum_sq / len(samples))
                    dbfs = 20 * math.log10(rms / 32768.0) if rms > 0 else -99.0
                    info["format"] = "raw_pcm_16bit"
                    info["channels"] = 1
                    info["sample_rate"] = 16000
                    info["bit_depth"] = 16
                    info["duration_seconds"] = sample_count / 16000.0
                    info["peak"] = peak
                    info["rms"] = round(rms, 2)
                    info["dbfs"] = round(dbfs, 1)
                    info["is_silent"] = (rms < 120.0 and peak < 350)
            except Exception:
                pass

    return info


@router.post("/audio")
async def process_esp32_audio(
    request: Request,
    audio: UploadFile = File(...),
    student_id: int = Form(1),
    session_id: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
    format: str = Query("json", description="Response format: 'json' or 'audio'"),
    test_mode: Optional[bool] = Query(None, description="Hardware test mode: bypass wake word")
):
    """
    Core Hardware Voice Pipeline for ESP32-S3 + INMP441 Microphone + MAX98357A Speaker.
    Protected by active turn scope to prevent premature timeouts during STT, LLM, or TTS.
    """
    voice_manager.check_inactivity_timeout()
    voice_manager.begin_turn()
    try:
        return await _process_esp32_audio_core(
            request=request,
            audio=audio,
            student_id=student_id,
            session_id=session_id,
            mode=mode,
            format=format,
            test_mode=test_mode,
            is_laptop=False
        )
    finally:
        voice_manager.end_turn()


@router.post("/audio/laptop")
async def process_esp32_laptop_audio(
    request: Request,
    audio: UploadFile = File(...),
    student_id: int = Form(1),
    session_id: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
    format: str = Query("json", description="Response format: 'json' or 'audio'"),
    test_mode: Optional[bool] = Query(None, description="Test mode: bypass wake word")
):
    """
    Laptop Audio Processing Endpoint on ESP32 Router.
    Uses browser microphone and laptop speakers without affecting ESP32 connection state.
    """
    voice_manager.check_inactivity_timeout()
    voice_manager.begin_turn()
    try:
        return await _process_esp32_audio_core(
            request=request,
            audio=audio,
            student_id=student_id,
            session_id=session_id,
            mode=mode,
            format=format,
            test_mode=test_mode,
            is_laptop=True
        )
    finally:
        voice_manager.end_turn()


async def _process_esp32_audio_core(
    request: Request,
    audio: UploadFile,
    student_id: int = 1,
    session_id: Optional[str] = None,
    mode: Optional[str] = None,
    format: str = "json",
    test_mode: Optional[bool] = None,
    is_laptop: bool = False
):
    client_ip = request.client.host if (request and request.client) else None
    if not is_laptop:
        record_esp32_activity(client_ip, "audio_upload")

    # Read uploaded audio bytes
    audio_bytes = await audio.read()
    if is_laptop and len(audio_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file exceeds 15MB limit")

    if not is_laptop:
        # Diagnostic analysis of physical audio signal from INMP441
        diag = _analyze_audio_signal(audio_bytes)
        print(
            f"[AUDIO_DIAG] INMP441 audio received: {diag['size_bytes']} bytes, format={diag['format']}, "
            f"{diag['duration_seconds']:.2f}s, {diag['sample_rate']}Hz, {diag['channels']}ch, {diag['bit_depth']}bit, "
            f"Peak={diag['peak']}/32768, RMS={diag['rms']} ({diag['dbfs']} dBFS), "
            f"silent={diag['is_silent']}",
            flush=True
        )
        now = time.time()
        last_upload = _esp32_hardware_state.get("last_audio_upload_timestamp")
        if last_upload and (now - last_upload) < 1.0:
            print(f"[AUDIO_DIAG] Warning: Rapid consecutive/overlapping audio chunk received ({now - last_upload:.2f}s since previous chunk)", flush=True)
        _esp32_hardware_state["last_audio_upload_timestamp"] = now
    else:
        print(f"[LAPTOP_AUDIO] Received audio chunk: {len(audio_bytes)} bytes, filename='{audio.filename}', format='{format}'", flush=True)

    # Determine whether hardware test mode is active
    is_hw_test = (test_mode is True) or (test_mode is None and getattr(settings, "HARDWARE_TEST_MODE", False))
    if is_hw_test:
        print(f"\n[HARDWARE_TEST] Stage 1/4: Audio received ({len(audio_bytes)} bytes, filename='{audio.filename}', format='{format}', client={client_ip})", flush=True)

    # 1. Save uploaded audio temporarily
    file_ext = Path(audio.filename or "recording.wav").suffix or ".wav"
    prefix = "laptop_in_" if is_laptop else "esp32_in_"
    temp_in_path = settings.AUDIO_CACHE_DIR / f"{prefix}{uuid.uuid4().hex[:8]}{file_ext}"
    with open(temp_in_path, "wb") as f:
        f.write(audio_bytes)

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

    clean_transcript = transcript.strip() if transcript else ""
    print(f"[AUDIO_DIAG] STT transcription: \"{clean_transcript}\" (duration={measured_duration_seconds:.2f}s, lang={stt_metadata.get('language')})", flush=True)

    # Hardware Test Mode Stage 2 logging
    if is_hw_test:
        voice_manager.consume_manual_activation()
        print(f"[HARDWARE_TEST] Stage 2/4: Transcription completed: \"{clean_transcript}\" (duration={measured_duration_seconds:.2f}s, lang={stt_metadata.get('language')})", flush=True)

    # 3. Classify Command or Intent BEFORE standard LLM/dialogue processing
    cmd_intent = classify_command_or_intent(clean_transcript)

    # 4. Check for "Stop Mizo" control command (Deterministic Tier 0)
    # Stop command must interrupt Mizo promptly, halt playback, and NEVER return audio.
    if clean_transcript and (cmd_intent == "STOP" or is_stop_command(clean_transcript)):
        voice_manager.handle_stop_command()
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "is_control_command": True,
                "command": "stop",
                "transcript": transcript,
                "reply_text": "",
                "audio_url": None,
                "voice_state": "STANDBY"
            }
        )

    # 5. Check student & onboarding status
    resolved_student = (get_student(int(student_id)) if student_id else None) or _require_primary_student()
    student_id = resolved_student["id"]
    student = resolved_student

    st_name = (student.get("name") or "").strip()
    is_name_known = bool(st_name and st_name.lower() not in ["new learner", "student", "there", "unknown", ""])
    onboarding_active = not bool(student.get("onboarding_completed", False))
    current_step = (student.get("onboarding_step") or ("completed" if not onboarding_active else "ask_name")).lower()
    if onboarding_active and is_name_known and current_step in ["ask_name", "new_student", "welcome"]:
        current_step = "ask_goals"

    manual_active = voice_manager.is_manual_activation_active()
    is_standby = voice_manager.state in [
        VoiceState.WAKE_LISTENING, VoiceState.STANDBY, VoiceState.SLEEP,
        VoiceState.SNOOZED, VoiceState.PAUSED, VoiceState.WAKE_ONLY
    ]

    # 6. STANDBY / WAKE_LISTENING GATING
    # In standby mode, ONLY wake phrases ("Hey Mizo", etc.) or explicit test mode / manual activation can activate Mizo.
    # Non-wake speech, background noise, or silence MUST be ignored cleanly without producing audio.
    if is_standby and not manual_active and not is_hw_test:
        if not clean_transcript or clean_transcript in ["[Speech Recognition Error]", ""]:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "ignored": True,
                    "transcript": "",
                    "reply_text": "",
                    "audio_url": None,
                    "voice_state": "STANDBY"
                }
            )

        state_before = voice_manager.state.value
        is_wake, trailing_speech = is_wake_word(clean_transcript, log_debug=False)
        is_explicit_resume = clean_transcript.strip().lower() in ["resume", "continue", "start"]
        if not (is_wake or is_explicit_resume):
            voice_manager.log(f'Heard: "{clean_transcript}"')
            voice_manager.log("No wake word. Ignoring.")
            voice_manager.log("Wake word not detected. Ignoring.")
            print(f"[WAKE_DIAG] State before: {state_before} | Transcript: '{clean_transcript}' | Wake match: False | Reason: No wake greeting or Mizo name variant detected | State after: {voice_manager.state.value}", flush=True)
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "ignored": True,
                    "transcript": transcript,
                    "reply_text": "",
                    "audio_url": None,
                    "voice_state": "STANDBY"
                }
            )

        # Wake phrase detected!
        voice_manager.touch_interaction()
        voice_manager.log(f'Heard: "{clean_transcript}"')
        voice_manager.log("Wake word detected: Mizo")
        voice_manager.log("Wake word detected")
        voice_manager.log("Waking assistant")
        voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "wake_word_detected")

        print(f"[WAKE_DIAG] State before: {state_before} | Transcript: '{transcript.strip()}' | Wake match: True | Reason: Valid wake phrase recognized | State after: {voice_manager.state.value}", flush=True)

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
                return FileResponse(
                    path=wav_path,
                    media_type="audio/wav",
                    filename="reply.wav",
                    headers={
                        "X-Voice-State": "ACTIVE_LISTENING",
                        "X-Transcript": clean_transcript[:100],
                        "X-Wake-Detected": "true"
                    }
                )
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

    # Handle silence or missing speech in active mode (do NOT reset inactivity timer on silence/empty input)
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

    # Unreliable speech recognition & low confidence handling in active mode
    is_unclear, unclear_reason = is_unclear_or_low_confidence_speech(
        clean_transcript,
        stt_metadata=stt_metadata,
        duration_seconds=measured_duration_seconds
    )
    if is_unclear:
        voice_manager.log(f"Unclear speech rejected ({unclear_reason}): '{clean_transcript}'")
        unclear_msg = "Sorry, I didn't catch that. Could you repeat it?"
        audio_url, wav_path = await tts_service.synthesize_to_file(unclear_msg)
        if format == "audio":
            return FileResponse(path=wav_path, media_type="audio/wav", filename="reply.wav")
        return {
            "success": True,
            "transcript": clean_transcript or "[Unclear Speech]",
            "reply_text": unclear_msg,
            "audio_url": audio_url,
            "onboarding_active": onboarding_active,
            "onboarding_step": current_step,
            "voice_state": voice_manager.state.value,
            "detail": f"Unclear speech detected ({unclear_reason})"
        }

    # Meaningful user speech verified: reset conversation inactivity timer
    voice_manager.touch_interaction()

    # Manual Active or HW Test Turn Processing
    if manual_active or is_hw_test:
        if manual_active:
            voice_manager.consume_manual_activation()
            voice_manager.log(f'Manual voice activation turn active: "{clean_transcript}"')
        if is_hw_test:
            voice_manager.log(f'Hardware test mode turn active: "{clean_transcript}"')
        is_wake, trailing_speech = is_wake_word(clean_transcript, log_debug=False)
        if is_wake and not trailing_speech:
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

            if is_hw_test:
                print(f"[HARDWARE_TEST] Stage 3/4: AI response generated [system]: \"{reply_msg}\"", flush=True)
            audio_url, wav_path = await tts_service.synthesize_to_file(reply_msg)
            if is_hw_test:
                print(f"[HARDWARE_TEST] Stage 4/4: Response audio returned ({wav_path}, format={format})\n", flush=True)
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
            if is_wake and trailing_speech:
                clean_transcript = trailing_speech
            voice_manager.transition_to(VoiceState.ONBOARDING if onboarding_active else VoiceState.ACTIVE_CONVERSATION, "test_or_manual_activation_processed")
    elif voice_manager.state == VoiceState.SPEAKING:
        voice_manager.handle_barge_in(clean_transcript)
        if cmd_intent == "STOP" or is_stop_command(clean_transcript):
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "is_control_command": True,
                    "command": "stop",
                    "transcript": transcript,
                    "reply_text": "",
                    "audio_url": None,
                    "voice_state": "STANDBY"
                }
            )
    else:
        is_wake, trailing_speech = is_wake_word(clean_transcript, log_debug=False)
        if is_wake:
            if trailing_speech:
                clean_transcript = trailing_speech
            else:
                reply_msg = f"Yes, {st_name}? I'm listening." if is_name_known else "Yes, I'm listening."
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

        # Check for assessment skip / deferral commands
        is_skip_requested = is_assessment_skip_command(clean_transcript) or cmd_intent == "SKIP_ASSESSMENT"
        if is_skip_requested:
            update_student(student_id, {
                "assessment_deferred": 1,
                "onboarding_completed": True,
                "onboarding_step": "completed"
            })
            onboarding_active = False
            reply_text = "No problem at all! We can skip the assessment for now and just talk normally. What would you like to discuss today?"
            voice_manager.transition_to(VoiceState.READY, "assessment_skipped")
            audio_url, wav_path = await tts_service.synthesize_to_file(reply_text)
            add_conversation_message(
                student_id=student_id,
                session_id=curr_session_id,
                role="user",
                content=clean_transcript,
                learning_mode=active_mode
            )
            add_conversation_message(
                student_id=student_id,
                session_id=curr_session_id,
                role="assistant",
                content=reply_text,
                audio_path=audio_url,
                learning_mode=active_mode,
                onboarding_state="completed",
                intent="skip_assessment"
            )
            if format == "audio":
                return FileResponse(path=wav_path, media_type="audio/wav", filename="reply.wav")
            return {
                "success": True,
                "transcript": clean_transcript,
                "reply_text": reply_text,
                "audio_url": audio_url,
                "learning_mode": active_mode,
                "session_id": curr_session_id,
                "provider_used": "system",
                "onboarding_active": False,
                "onboarding_step": "completed",
                "voice_state": voice_manager.state.value,
                "detail": "Assessment deferred by user request."
            }

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
                retry_msg = "That sample was a bit too short. Please share a little more about yourself or your projects so I can calibrate our lessons accurately."
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

                recent_turns = get_conversation_history(student_id=student_id, session_id=curr_session_id, limit=6)
                assessment_status = (
                    "deferred" if updated_student.get("assessment_deferred")
                    else ("completed" if updated_student.get("onboarding_completed")
                          else "active" if next_step in ["speech_test_prompt", "speaking_assessment"]
                          else "pending")
                )

                system_prompt = personalization_service.build_onboarding_turn_prompt(
                    student=updated_student,
                    current_step=current_step,
                    next_step=next_step,
                    user_answer=clean_transcript,
                    recent_history=recent_turns,
                    assessment_status=assessment_status
                )
                dialogue_msgs = []
                for turn in (recent_turns or [])[-4:]:
                    r = "assistant" if turn.get("role") == "assistant" else "user"
                    c = (turn.get("content") or "").strip()
                    if c:
                        dialogue_msgs.append({"role": r, "content": c})
                if not dialogue_msgs or dialogue_msgs[-1].get("content") != clean_transcript:
                    dialogue_msgs.append({"role": "user", "content": clean_transcript})

                try:
                    llm_res = await llm_service.generate_response(messages=dialogue_msgs, system_prompt=system_prompt)
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

    if is_hw_test:
        print(f"[HARDWARE_TEST] Stage 3/4: AI response generated [{provider_used}]: \"{reply_text}\"", flush=True)

    # Synthesize Speech (16kHz Mono WAV for MAX98357A)
    voice_manager.handle_speaking_start()
    try:
        audio_url, wav_path = await tts_service.synthesize_to_file(reply_text)
    except Exception as tts_err:
        print(f"[TTS Error] Synthesis failed: {tts_err}")
        audio_url, wav_path = None, None
    voice_manager.handle_speaking_ended()

    if is_hw_test:
        print(f"[HARDWARE_TEST] Stage 4/4: Response audio returned ({wav_path}, format={format})\n", flush=True)

    if not onboarding_active and voice_manager.state != VoiceState.READY:
        voice_manager.transition_to(VoiceState.READY, "turn_complete")

    # Log to conversations database with rapid retry deduplication
    recent_msgs = get_conversation_history(student_id=student_id, session_id=curr_session_id, limit=2)
    is_dup = False
    if recent_msgs:
        for prev in reversed(recent_msgs):
            if prev.get("role") == "user" and prev.get("content", "").strip() == clean_transcript:
                is_dup = True
                break

    if not is_dup:
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
        if is_hw_test:
            print(f"[HARDWARE_TEST] Persisted conversation messages for student_id={student_id}, session_id='{curr_session_id}'", flush=True)
    else:
        if is_hw_test:
            print(f"[HARDWARE_TEST] Duplicate turn detected for student_id={student_id}, session_id='{curr_session_id}'. Skipped saving duplicate.", flush=True)

    if format == "audio":
        headers = {
            "X-Voice-State": voice_manager.state.value if hasattr(voice_manager.state, 'value') else str(voice_manager.state),
            "X-Transcript": clean_transcript[:100]
        }
        if wav_path and Path(wav_path).exists():
            return FileResponse(path=wav_path, media_type="audio/wav", filename="reply.wav", headers=headers)
        else:
            fallback_filename = f"mizo_fallback_{uuid.uuid4().hex[:8]}.wav"
            fallback_path = settings.AUDIO_CACHE_DIR / fallback_filename
            tts_service._generate_fallback_wav(fallback_path, reply_text)
            return FileResponse(path=str(fallback_path), media_type="audio/wav", filename="reply.wav", headers=headers)

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
    Guarded by active turn scope to prevent premature timeouts during inference/TTS.
    """
    voice_manager.check_inactivity_timeout()
    voice_manager.begin_turn()
    try:
        return await _process_esp32_chat_core(req)
    finally:
        voice_manager.end_turn()


async def _process_esp32_chat_core(req: ESP32ChatRequest):
    """Core logic for text chat request."""
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
    student = (get_student(req.student_id) if req.student_id else None) or _require_primary_student()
    student_id = student["id"]

    st_name = (student.get("name") or "").strip()
    is_name_known = bool(st_name and st_name.lower() not in ["new learner", "student", "there", "unknown", ""])
    onboarding_active = not bool(student.get("onboarding_completed", False))
    current_step = (student.get("onboarding_step") or ("completed" if not onboarding_active else "ask_name")).lower()
    if onboarding_active and is_name_known and current_step in ["ask_name", "new_student", "welcome"]:
        current_step = "ask_goals"

    # Empty or inaudible input check
    if not clean_transcript or clean_transcript in ["[Speech Recognition Error]", ""]:
        unclear_msg = "Sorry, I didn't catch that. Could you repeat it?"
        audio_url, _ = await tts_service.synthesize_to_file(unclear_msg)
        return ESP32ChatResponse(
            transcript=clean_transcript or "[Unclear]",
            reply_text=unclear_msg,
            audio_url=audio_url,
            learning_mode=req.mode or "coach",
            session_id=session_id or "ses_default",
            provider_used="system",
            is_fallback=False,
            onboarding_active=onboarding_active,
            onboarding_step=current_step,
            voice_state=voice_manager.state.value,
            detail="Empty or missing message input"
        )

    # 4. Check WAKE_LISTENING / SNOOZED / STANDBY / PAUSED State Machine
    is_typed = bool(getattr(req, "is_typed_text", False))
    is_idle_state = voice_manager.state in [
        VoiceState.WAKE_LISTENING, VoiceState.STANDBY,
        VoiceState.SLEEP, VoiceState.SNOOZED, VoiceState.PAUSED, VoiceState.WAKE_ONLY
    ]

    manual_active = voice_manager.is_manual_activation_active()
    if manual_active:
        voice_manager.consume_manual_activation()
        voice_manager.log(f'Manual voice activation chat active. Auto-waking assistant for: "{clean_transcript}"')
        voice_manager.transition_to(VoiceState.ONBOARDING if onboarding_active else VoiceState.ACTIVE_LISTENING, "manual_chat_wake")
    elif is_idle_state:
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
                    voice_state="ACTIVE_LISTENING",
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

    # Reset inactivity timer on meaningful learner interaction
    voice_manager.touch_interaction()

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

        # Check for assessment skip / deferral commands
        is_skip_requested = is_assessment_skip_command(clean_transcript) or cmd_intent == "SKIP_ASSESSMENT"
        if is_skip_requested:
            update_student(student_id, {
                "assessment_deferred": 1,
                "onboarding_completed": True,
                "onboarding_step": "completed"
            })
            onboarding_active = False
            reply_text = "No problem at all! We can skip the assessment for now and just talk normally. What would you like to discuss today?"
            voice_manager.transition_to(VoiceState.READY, "assessment_skipped")
            audio_url, _ = await tts_service.synthesize_to_file(reply_text)
            add_conversation_message(
                student_id=student_id,
                session_id=curr_session_id,
                role="user",
                content=clean_transcript,
                learning_mode=active_mode
            )
            add_conversation_message(
                student_id=student_id,
                session_id=curr_session_id,
                role="assistant",
                content=reply_text,
                audio_path=audio_url,
                learning_mode=active_mode,
                onboarding_state="completed",
                intent="skip_assessment"
            )
            return ESP32ChatResponse(
                transcript=transcript,
                reply_text=reply_text,
                audio_url=audio_url,
                learning_mode=active_mode,
                session_id=curr_session_id,
                provider_used="system",
                is_fallback=False,
                onboarding_active=False,
                onboarding_step="completed",
                topic=None,
                detail="Assessment deferred by user request.",
                voice_state=voice_manager.state.value
            )

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
                retry_msg = "That sample was a bit too short. Please share a little more about yourself or your projects so I can calibrate our lessons accurately."
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

                recent_turns = get_conversation_history(student_id=student_id, session_id=curr_session_id, limit=6)
                assessment_status = (
                    "deferred" if updated_student.get("assessment_deferred")
                    else ("completed" if updated_student.get("onboarding_completed")
                          else "active" if next_step in ["speech_test_prompt", "speaking_assessment"]
                          else "pending")
                )

                system_prompt = personalization_service.build_onboarding_turn_prompt(
                    student=updated_student,
                    current_step=current_step,
                    next_step=next_step,
                    user_answer=clean_transcript,
                    recent_history=recent_turns,
                    assessment_status=assessment_status
                )
                dialogue_msgs = []
                for turn in (recent_turns or [])[-4:]:
                    r = "assistant" if turn.get("role") == "assistant" else "user"
                    c = (turn.get("content") or "").strip()
                    if c:
                        dialogue_msgs.append({"role": r, "content": c})
                if not dialogue_msgs or dialogue_msgs[-1].get("content") != clean_transcript:
                    dialogue_msgs.append({"role": "user", "content": clean_transcript})

                try:
                    llm_res = await llm_service.generate_response(messages=dialogue_msgs, system_prompt=system_prompt)
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
