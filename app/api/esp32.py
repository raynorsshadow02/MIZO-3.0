import os
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, Header, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings
from app.db.database import (
    validate_device_key,
    add_conversation_message,
    get_student,
    create_student,
    update_student,
    record_onboarding_baseline,
    get_active_session,
    get_session_by_id,
    handle_device_start_or_reset,
    get_db_settings,
    get_conversation_history
)
from app.db.models import ESP32ChatRequest, ESP32ChatResponse, ESP32StartRequest, ESP32StartResponse
from app.services.stt_service import stt_service
from app.services.llm_service import llm_service, LLMServiceError
from app.services.tts_service import tts_service
from app.services.personalization_service import personalization_service
from app.services.rag_service import rag_service
from app.services.voice_service import is_wake_word, is_stop_command, voice_manager, VoiceState, detect_mizo_intent, ASSISTANT_NAME

router = APIRouter(prefix="/api/v1/esp32", tags=["ESP32 Hardware"])


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
    student_id = req.student_id if req and req.student_id else 1
    action = req.action if req and req.action else "start"

    start_info = handle_device_start_or_reset(student_id=student_id)
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
    try:
        # 2. Transcribe Audio using real STT
        transcript = await stt_service.transcribe_audio_file(temp_in_path)
    except Exception as e:
        print(f"[STT Error] Transcription failed: {e}")
        transcript = ""
    finally:
        if temp_in_path.exists():
            try:
                temp_in_path.unlink()
            except Exception:
                pass

    # 3. Check for "Stop Mizo" control command
    clean_transcript = transcript.strip() if transcript else ""
    if clean_transcript and is_stop_command(clean_transcript):
        voice_manager.handle_stop_command()
        return {
            "success": True,
            "is_control_command": True,
            "command": "stop",
            "transcript": transcript,
            "reply_text": "",
            "audio_url": None,
            "voice_state": "STANDBY"
        }

    # 4. Check student & onboarding status
    student = get_student(student_id)
    if not student:
        student = create_student({"name": "New Learner", "onboarding_completed": False, "onboarding_step": "ask_name"})
        student_id = student["id"]

    onboarding_active = not student.get("onboarding_completed", False)
    current_step = (student.get("onboarding_step") or "ask_name").lower()

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
            if voice_manager.state == VoiceState.STANDBY:
                # In STANDBY mode, ignore silence/noise completely without speaking
                return {
                    "success": True,
                    "ignored": True,
                    "transcript": "",
                    "reply_text": "",
                    "audio_url": None,
                    "voice_state": "STANDBY"
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

    # 5. Check Standby vs Active Listening State Machine
    if voice_manager.state in [VoiceState.STANDBY, VoiceState.SLEEP]:
        intent = detect_mizo_intent(clean_transcript, log_debug=True)
        if intent == "WAKE":
            is_wake, trailing_speech = is_wake_word(clean_transcript, log_debug=False)
            voice_manager.log(f"Wake word detected: {ASSISTANT_NAME}")
            voice_manager.log("Waking assistant")
            voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "wake_word_detected")
            if trailing_speech:
                clean_transcript = trailing_speech
            else:
                # Wake word alone activates Mizo with spoken acknowledgement 'Yes?' and listens for user command
                reply_msg = "Yes?"
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
                    "voice_state": "ACTIVE_LISTENING"
                }
        else:
            # In STANDBY / SNOOZE mode, ordinary speech is ignored completely
            voice_manager.log(f'Heard: "{clean_transcript}"')
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
        # User speech detected during TTS -> barge-in interruption!
        voice_manager.handle_barge_in(clean_transcript)
        if is_stop_command(clean_transcript):
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
        # In ACTIVE_LISTENING mode: check if utterance begins with wake prefix
        is_wake, trailing_speech = is_wake_word(clean_transcript, log_debug=False)
        if is_wake and trailing_speech:
            clean_transcript = trailing_speech

    # 6. Retrieve or create active session
    active_session = get_session_by_id(session_id) if session_id else None
    if not active_session:
        active_session = get_active_session(student_id=student_id, default_mode=mode or "coach")
    curr_session_id = active_session["session_id"]
    active_mode = mode or active_session.get("mode") or "coach"

    # User input is accepted for processing
    voice_manager.log(f'Heard: "{clean_transcript}"')
    voice_manager.log("End of speech detected.")
    voice_manager.transition_to(VoiceState.PROCESSING, "end_of_speech")

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
    provider_used = "groq"
    is_fallback = False
    latency_ms = 0.0

    if onboarding_active:
        if current_step in [personalization_service.STATE_ASSESSMENT, "SPEECH_TEST_PROMPT", "SPEECH_EVALUATION", "speech_test_prompt"]:
            # Guard: too short audio / transcript must not create fake scores
            words_in_transcript = len(clean_transcript.split())
            if words_in_transcript < 12:
                retry_msg = "Your speaking sample was too short or could not be clearly heard. Please speak freely for about one minute on the topic so I can evaluate your English communication accurately."
                audio_url, wav_path = await tts_service.synthesize_to_file(retry_msg)
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
                    "detail": "Speaking sample was under 12 words. Please retry."
                }

            # Student delivered their 1-2 minute speaking assessment!
            assessment_data = await personalization_service.run_speaking_assessment(
                transcript=clean_transcript,
                student_id=student_id,
                session_id=curr_session_id,
                topic=personalization_service.generate_dynamic_assessment_topic(student),
                duration_seconds=60.0
            )
            reply_text = assessment_data.get("spoken_summary", "Calibration complete! Your speaking assessment is saved.")
            onboarding_active = False
            next_step = "completed"
            update_student(student_id, {"onboarding_completed": True, "onboarding_step": "completed"})
        else:
            is_valid_ans, val_reason = personalization_service.is_valid_onboarding_response(student, current_step, clean_transcript)
            if not is_valid_ans:
                # Answer is unrelated, a counter-question, or deflection: keep current question!
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
                # Valid answer received: extract memory and advance onboarding state
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

                # Determine next required onboarding state
                next_step = personalization_service.determine_next_onboarding_state(updated_student, current_step)
                update_student(student_id, {"onboarding_step": next_step})

                # Build structured turn prompt for LLM
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
                except Exception as e:
                    # Dynamic natural fallback acknowledgment + next question
                    st_name = updated_student.get("name") or "there"
                    if current_step in ["ask_name", "welcome"]:
                        reply_text = f"That's a wonderful name, {st_name}! What is your main goal with Mizo?"
                    elif current_step in ["ask_goals", "ask_problems"]:
                        reply_text = "That's a great goal to work towards! What would you say is your biggest weakness or challenge in English right now?"
                    elif current_step in ["ask_weaknesses"]:
                        reply_text = "I understand completely. How would you describe your current English level—beginner, intermediate, advanced, or are you not sure?"
                    elif current_step in ["ask_self_level"]:
                        topic = personalization_service.generate_dynamic_assessment_topic(updated_student)
                        reply_text = f"To calibrate your speaking baseline, please speak freely for about one minute on this topic: {topic}"
                    else:
                        reply_text = "Great! What would you like to practice today?"
    else:
        # Regular active coaching session
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

        # Extract any newly mentioned facts into long-term memory
        personalization_service.extract_and_update_student_memory(student_id, clean_transcript)

        rag_context = ""
        if detected_mode == "tutor" or "teach" in clean_transcript.lower() or "explain" in clean_transcript.lower():
            rag_chunks = rag_service.query_knowledge(clean_transcript, top_k=2)
            if rag_chunks:
                rag_context = "\n---\n".join([c["content"] for c in rag_chunks])

        system_prompt = personalization_service.build_system_prompt(
            student_id=student_id,
            mode=detected_mode,
            session_id=curr_session_id,
            rag_context=rag_context
        )
        recent_history = get_conversation_history(student_id=student_id, session_id=curr_session_id, limit=6)
        messages = []
        for msg in recent_history:
            if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": clean_transcript})
        try:
            llm_res = await llm_service.generate_response(messages=messages, system_prompt=system_prompt)
            reply_text = llm_res.text
            provider_used = f"{llm_res.provider} ({llm_res.model})"
            is_fallback = llm_res.is_fallback
            latency_ms = llm_res.latency_ms
        except Exception as e:
            reply_text = "I am currently unable to reach my AI brain. Please check your configured API keys."

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
    audio_url, wav_path = await tts_service.synthesize_to_file(reply_text)
    voice_manager.handle_speaking_ended()

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

    if format == "audio":
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
            "grammar_score": latest_student.get("grammar_score", 75.0),
            "vocabulary_score": latest_student.get("vocabulary_score", 70.0),
            "fluency_score": latest_student.get("fluency_score", 80.0),
            "confidence_score": latest_student.get("confidence_score", 78.0),
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
    student_id = req.student_id or 1
    session_id = req.session_id

    # Check for "Stop Mizo" control command
    clean_transcript = transcript.strip() if transcript else ""
    if clean_transcript and is_stop_command(clean_transcript):
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

    # 1. Check student & onboarding state
    student = get_student(student_id)
    if not student:
        student = create_student({"name": "New Learner", "onboarding_completed": False, "onboarding_step": "ask_name"})
        student_id = student["id"]

    onboarding_active = not student.get("onboarding_completed", False)
    current_step = (student.get("onboarding_step") or "ask_name").lower()

    # 2. Check Standby vs Active Listening State Machine
    if voice_manager.state in [VoiceState.STANDBY, VoiceState.SLEEP]:
        intent = detect_mizo_intent(clean_transcript, log_debug=True)
        if intent == "WAKE":
            is_wake, trailing_speech = is_wake_word(clean_transcript, log_debug=False)
            voice_manager.log(f"Wake word detected: {ASSISTANT_NAME}")
            voice_manager.log("Waking assistant")
            voice_manager.transition_to(VoiceState.ACTIVE_LISTENING, "wake_word_detected")
            if trailing_speech:
                clean_transcript = trailing_speech
            else:
                reply_msg = "Yes?"
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
                    voice_state="ACTIVE_LISTENING"
                )
        else:
            # In STANDBY / SNOOZE mode, ordinary speech is ignored completely
            voice_manager.log(f'Heard: "{clean_transcript}"')
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
        # Interruption during TTS
        voice_manager.handle_barge_in(clean_transcript)
        if is_stop_command(clean_transcript):
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
                is_control_command=True,
                command="stop",
                voice_state="STANDBY"
            )
    else:
        # In ACTIVE_LISTENING mode: check if user prefixed utterance with wake word
        is_wake, trailing_speech = is_wake_word(clean_transcript, log_debug=False)
        if is_wake and trailing_speech:
            clean_transcript = trailing_speech

    # 3. Retrieve or create active session
    active_session = get_session_by_id(session_id) if session_id else None
    if not active_session:
        active_session = get_active_session(student_id=student_id, default_mode=req.mode or "coach")
    curr_session_id = active_session["session_id"]
    active_mode = req.mode or active_session.get("mode") or "coach"

    # User input is accepted for processing
    voice_manager.log(f'Heard: "{clean_transcript}"')
    voice_manager.log("End of speech detected.")
    voice_manager.transition_to(VoiceState.PROCESSING, "end_of_speech")

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
    provider_used = "groq"
    is_fallback = False
    latency_ms = 0.0

    if onboarding_active:
        if current_step in [personalization_service.STATE_ASSESSMENT, "SPEECH_TEST_PROMPT", "SPEECH_EVALUATION", "speech_test_prompt"]:
            words_in_transcript = len(clean_transcript.split())
            if words_in_transcript < 12:
                retry_msg = "Your speaking sample was too short or could not be clearly heard. Please speak freely for about one minute on the topic so I can evaluate your English communication accurately."
                audio_url, _ = await tts_service.synthesize_to_file(retry_msg)
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
                    detail="Sample too short; please retry."
                )

            # Evaluate speaking assessment
            assessment_data = await personalization_service.run_speaking_assessment(
                transcript=clean_transcript,
                student_id=student_id,
                session_id=curr_session_id,
                topic=personalization_service.generate_dynamic_assessment_topic(student),
                duration_seconds=60.0
            )
            reply_text = assessment_data.get("spoken_summary", "Calibration complete! Your speaking assessment is saved.")
            onboarding_active = False
            next_step = "completed"
            update_student(student_id, {"onboarding_completed": True, "onboarding_step": "completed"})
        else:
            is_valid_ans, val_reason = personalization_service.is_valid_onboarding_response(student, current_step, clean_transcript)
            if not is_valid_ans:
                # Answer is unrelated, a counter-question, or deflection: keep current question!
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
                # Multi-field extraction from valid user utterance
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

                # Determine next required state in state machine
                next_step = personalization_service.determine_next_onboarding_state(updated_student, current_step)
                update_student(student_id, {"onboarding_step": next_step})

                # Build structured turn prompt for LLM
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
                except LLMServiceError as e:
                    # Dynamic natural fallback acknowledgment + next question
                    st_name = updated_student.get("name") or "there"
                    if current_step in ["ask_name", "welcome"]:
                        reply_text = f"That's a wonderful name, {st_name}! What is your main goal with Mizo?"
                    elif current_step in ["ask_goals", "ask_problems"]:
                        reply_text = "That's a great goal to work towards! What would you say is your biggest weakness or challenge in English right now?"
                    elif current_step in ["ask_weaknesses"]:
                        reply_text = "I understand completely. How would you describe your current English level—beginner, intermediate, advanced, or are you not sure?"
                    elif current_step in ["ask_self_level"]:
                        topic = personalization_service.generate_dynamic_assessment_topic(updated_student)
                        reply_text = f"To calibrate your speaking baseline, please speak freely for about one minute on this topic: {topic}"
                    else:
                        reply_text = "Great! What would you like to practice today?"
    else:
        # Regular active session
        detected_mode = personalization_service.detect_intent_and_mode(clean_transcript, current_mode=active_mode)
        g_errs, v_suggs, flu, pac, conf = personalization_service.evaluate_utterance(
            text=transcript,
            student_id=student_id,
            session_id=curr_session_id
        )
        grammar_errors = g_errs
        vocab_suggestions = v_suggs
        fluency_score = flu
        pacing_score = pac
        confidence_score = conf

        personalization_service.extract_and_update_student_memory(student_id, transcript)

        rag_context = ""
        if detected_mode == "tutor" or "teach" in transcript.lower() or "explain" in transcript.lower():
            rag_chunks = rag_service.query_knowledge(transcript, top_k=2)
            if rag_chunks:
                rag_context = "\n---\n".join([c["content"] for c in rag_chunks])

        system_prompt = personalization_service.build_system_prompt(
            student_id=student_id,
            mode=detected_mode,
            session_id=curr_session_id,
            rag_context=rag_context
        )
        # Build multi-turn context from recent conversation history
        recent_history = get_conversation_history(student_id=student_id, session_id=curr_session_id, limit=6)
        messages = []
        for msg in recent_history:
            if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": transcript})
        try:
            llm_res = await llm_service.generate_response(messages=messages, system_prompt=system_prompt)
            reply_text = llm_res.text
            provider_used = f"{llm_res.provider} ({llm_res.model})"
            is_fallback = llm_res.is_fallback
            latency_ms = llm_res.latency_ms
        except LLMServiceError as e:
            reply_text = "I am currently unable to reach my AI brain. Please check your configured API keys."

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
    audio_url, _ = await tts_service.synthesize_to_file(reply_text)
    voice_manager.handle_speaking_ended()

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
            "grammar_score": latest_student.get("grammar_score", 75.0),
            "vocabulary_score": latest_student.get("vocabulary_score", 70.0),
            "fluency_score": latest_student.get("fluency_score", 80.0),
            "confidence_score": latest_student.get("confidence_score", 78.0),
        },
        "voice_state": voice_manager.state.value
    }
    if 'assessment_data' in locals() and assessment_data:
        resp_kwargs["assessment"] = assessment_data
    if next_step in ["speech_test_prompt", "speaking_assessment"]:
        resp_kwargs["topic"] = personalization_service.generate_dynamic_assessment_topic(latest_student)

    return ESP32ChatResponse(**resp_kwargs)
