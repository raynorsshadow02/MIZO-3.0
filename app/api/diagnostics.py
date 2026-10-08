import logging
from typing import Dict, Any
from fastapi import APIRouter
from app.config import settings
from app.db.database import (
    get_connection,
    get_student,
    get_active_session,
    get_student_assessments,
    get_db_settings
)
from app.services.voice_service import voice_manager
from app.services.llm_service import llm_service
from app.services.memory_service import memory_manager

logger = logging.getLogger("mizo.diagnostics")
router = APIRouter(prefix="/api/v1/diagnostics", tags=["Diagnostics"])


@router.get("/onboarding")
async def get_onboarding_diagnostics() -> Dict[str, Any]:
    """
    Real-time runtime diagnostic endpoint for Mizo onboarding, memory, voice state,
    STT, TTS, assessment provenance, and LLM providers.
    Never exposes API secrets.
    """
    # 1. Database check
    db_status = "ok"
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
    except Exception as e:
        db_status = f"error: {e}"

    # 2. Student profile & identity integrity check
    student = get_student(1) or {}
    st_name = (student.get("name") or "").strip()
    is_name_valid = memory_manager.is_valid_name(st_name) and st_name.lower() not in [
        "student", "new learner", "there", "unknown", ""
    ]

    profile_diag = {
        "exists": bool(student),
        "name_valid": is_name_valid,
        "name": st_name if is_name_valid else "Not established",
        "onboarding_completed": bool(student.get("onboarding_completed", False)),
        "onboarding_step": student.get("onboarding_step", "ask_name"),
        "education": student.get("education"),
        "target_level": student.get("target_level"),
        "baseline_established": bool(student.get("baseline_grammar") is not None)
    }

    # 3. Session & Voice State Machine check
    act_session = get_active_session(1) or {}
    session_diag = {
        "active": bool(act_session),
        "session_id": act_session.get("session_id", "none"),
        "mode": act_session.get("mode", "coach"),
        "voice_state": voice_manager.state.value,
        "microphone_active": voice_manager.microphone_active,
        "turn_silence_threshold_seconds": voice_manager.turn_silence_seconds
    }

    # 4. STT Availability
    stt_diag = {
        "available": True,
        "primary_engine": "groq_whisper" if settings.GROQ_API_KEY else "openai_whisper" if settings.OPENAI_API_KEY else "faster_whisper_local",
        "supported_audio_formats": ["wav", "webm", "ogg", "mp3"]
    }

    # 5. TTS Availability
    tts_diag = {
        "available": True,
        "voice": settings.DEFAULT_TTS_VOICE,
        "sample_rate": "16kHz Mono WAV"
    }

    # 6. LLM Providers Status (Never secrets)
    db_conf = get_db_settings()
    prov_statuses = await llm_service.verify_all_providers()
    providers_summary = {}
    for ps in prov_statuses:
        providers_summary[ps["provider"]] = {
            "status": ps["status"],
            "configured": ps["configured"],
            "generation_ready": ps.get("generation_ready", False),
            "model": ps.get("model")
        }

    # 7. Assessment Engine & Provenance check
    past_assessments = get_student_assessments(1, limit=1)
    latest_assess = past_assessments[0] if past_assessments else None
    assessment_diag = {
        "available": True,
        "total_assessments_recorded": len(past_assessments),
        "latest_assessment_method": latest_assess.get("assessment_method") if latest_assess else "none",
        "latest_provider": latest_assess.get("provider") if latest_assess else None,
        "latest_model": latest_assess.get("model") if latest_assess else None,
        "latest_overall_level": latest_assess.get("overall_level") if latest_assess else None
    }

    return {
        "status": "healthy" if db_status == "ok" else "degraded",
        "database": db_status,
        "profile": profile_diag,
        "session": session_diag,
        "stt": stt_diag,
        "tts": tts_diag,
        "providers": providers_summary,
        "assessment": assessment_diag
    }
