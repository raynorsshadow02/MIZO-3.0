import os
import uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Header, HTTPException, Query, Response
from fastapi.responses import FileResponse

from app.config import settings
from app.db.database import (
    validate_device_key,
    add_conversation_message,
    get_student,
    get_db_settings
)
from app.db.models import ESP32ChatRequest, ESP32ChatResponse
from app.services.stt_service import stt_service
from app.services.llm_service import llm_service
from app.services.tts_service import tts_service
from app.services.personalization_service import personalization_service
from app.services.rag_service import rag_service

router = APIRouter(prefix="/api/v1/esp32", tags=["ESP32 Hardware"])


def verify_device(x_device_key: Optional[str] = Header(None), key: Optional[str] = Query(None)):
    """Verifies ESP32 device authorization key."""
    token = x_device_key or key
    # If no keys are registered or token matches default, allow
    if not token or not validate_device_key(token):
        # Allow development bypass if token matches default configured key
        if token == settings.DEFAULT_DEVICE_KEY:
            return True
        # If in development and token is omitted, log warning but allow local test
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


@router.post("/audio")
async def process_esp32_audio(
    audio: UploadFile = File(...),
    student_id: int = Form(1),
    session_id: str = Form("esp32_session"),
    mode: Optional[str] = Form(None),
    format: str = Query("json", description="Response format: 'json' or 'audio'")
):
    """
    Core Hardware Pipeline for ESP32-S3 + INMP441 Microphone + MAX98357A Speaker.
    1. Receives recorded audio from ESP32.
    2. Transcribes via STT (Groq Whisper / Faster-Whisper).
    3. Runs RAG context retrieval and personalized LLM response.
    4. Synthesizes 16kHz 16-bit Mono WAV audio for MAX98357A I2S DAC.
    5. Returns either direct WAV audio stream (format=audio) or JSON metadata (format=json).
    """
    # 1. Save uploaded audio temporarily
    file_ext = Path(audio.filename or "recording.wav").suffix or ".wav"
    temp_in_path = settings.AUDIO_CACHE_DIR / f"esp32_in_{uuid.uuid4().hex[:8]}{file_ext}"
    with open(temp_in_path, "wb") as f:
        f.write(await audio.read())

    try:
        # 2. Transcribe Audio
        transcript = await stt_service.transcribe_audio_file(temp_in_path)
    finally:
        if temp_in_path.exists():
            try:
                temp_in_path.unlink()
            except Exception:
                pass

    # 3. Evaluate Utterance (Grammar, Fluency, Vocabulary)
    grammar_errors, vocab_suggestions, fluency_score = personalization_service.evaluate_utterance(transcript)

    # 4. RAG Knowledge Search
    rag_chunks = rag_service.query_knowledge(transcript, top_k=2)
    rag_context = ""
    if rag_chunks:
        rag_context = "\nRELEVANT KNOWLEDGE CONTEXT:\n" + "\n---\n".join([c["content"] for c in rag_chunks])

    # 5. Build Personalized System Prompt & Generate LLM Reply
    system_prompt = personalization_service.build_system_prompt(student_id, mode)
    if rag_context:
        system_prompt += f"\n{rag_context}"

    messages = [{"role": "user", "content": transcript}]
    reply_text = await llm_service.generate_response(messages=messages, system_prompt=system_prompt)

    # 6. Synthesize TTS (16kHz WAV for MAX98357A I2S)
    audio_url, wav_path = await tts_service.synthesize_to_file(reply_text)

    # 7. Update Student Progress & Log Conversation
    personalization_service.update_student_progress(student_id, grammar_errors, vocab_suggestions, fluency_score)

    add_conversation_message(
        student_id=student_id,
        session_id=session_id,
        role="user",
        content=transcript,
        grammar_errors=grammar_errors,
        vocabulary_suggestions=vocab_suggestions,
        fluency_score=fluency_score,
        learning_mode=mode or "coach"
    )
    add_conversation_message(
        student_id=student_id,
        session_id=session_id,
        role="assistant",
        content=reply_text,
        audio_path=audio_url,
        learning_mode=mode or "coach"
    )

    # 8. Return response
    if format == "audio":
        # Stream WAV binary directly to ESP32 for instant speaker playback
        return FileResponse(path=wav_path, media_type="audio/wav", filename="reply.wav")

    student = get_student(student_id) or {}
    return {
        "success": True,
        "transcript": transcript,
        "reply_text": reply_text,
        "audio_url": audio_url,
        "learning_mode": mode or "coach",
        "grammar_errors": grammar_errors,
        "vocabulary_suggestions": vocab_suggestions,
        "fluency_score": fluency_score,
        "student_stats": {
            "grammar_score": student.get("grammar_score", 75.0),
            "vocabulary_score": student.get("vocabulary_score", 70.0),
            "fluency_score": student.get("fluency_score", 80.0),
            "confidence_score": student.get("confidence_score", 78.0),
        }
    }


@router.post("/chat", response_model=ESP32ChatResponse)
async def process_esp32_chat(req: ESP32ChatRequest):
    """
    Lightweight JSON endpoint for ESP32 or client test:
    Takes text string -> Generates AI response + TTS WAV audio.
    """
    transcript = req.message.strip()
    student_id = req.student_id or 1
    session_id = req.session_id or "esp32_text"
    mode = req.mode or "coach"

    # Evaluate
    grammar_errors, vocab_suggestions, fluency_score = personalization_service.evaluate_utterance(transcript)

    # RAG
    rag_chunks = rag_service.query_knowledge(transcript, top_k=2)
    rag_context = ""
    if rag_chunks:
        rag_context = "\nRELEVANT KNOWLEDGE CONTEXT:\n" + "\n---\n".join([c["content"] for c in rag_chunks])

    # LLM
    system_prompt = personalization_service.build_system_prompt(student_id, mode) + (f"\n{rag_context}" if rag_context else "")
    messages = [{"role": "user", "content": transcript}]
    reply_text = await llm_service.generate_response(messages=messages, system_prompt=system_prompt)

    # TTS
    audio_url, _ = await tts_service.synthesize_to_file(reply_text)

    # Update DB
    personalization_service.update_student_progress(student_id, grammar_errors, vocab_suggestions, fluency_score)
    add_conversation_message(
        student_id=student_id, session_id=session_id, role="user", content=transcript,
        grammar_errors=grammar_errors, vocabulary_suggestions=vocab_suggestions, fluency_score=fluency_score, learning_mode=mode
    )
    add_conversation_message(
        student_id=student_id, session_id=session_id, role="assistant", content=reply_text,
        audio_path=audio_url, learning_mode=mode
    )

    student = get_student(student_id) or {}
    corrections = [f"{e['error']} ➔ {e['correction']}" for e in grammar_errors]

    return ESP32ChatResponse(
        transcript=transcript,
        reply_text=reply_text,
        audio_url=audio_url,
        learning_mode=mode,
        corrections=corrections,
        student_stats={
            "grammar_score": student.get("grammar_score", 75.0),
            "vocabulary_score": student.get("vocabulary_score", 70.0),
            "fluency_score": student.get("fluency_score", 80.0),
            "confidence_score": student.get("confidence_score", 78.0),
        }
    )
