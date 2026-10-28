from fastapi import APIRouter, Request, UploadFile, File, Form, Query, HTTPException
from typing import Optional, Dict, Any
from app.services.voice_service import voice_manager
from app.api.esp32 import _process_esp32_audio_core
from app.db.database import get_db_settings, update_db_settings

router = APIRouter(prefix="/api/v1/audio", tags=["Audio System"])


@router.get("/mode")
async def get_audio_mode():
    """Retrieve current authoritative audio mode ('esp32' | 'laptop')."""
    settings_data = get_db_settings()
    mode = str(settings_data.get("audio_mode") or "esp32").strip().lower()
    if mode not in {"esp32", "laptop"}:
        mode = "esp32"
    return {"audio_mode": mode}


@router.post("/mode")
async def set_audio_mode(payload: Dict[str, Any]):
    """Update authoritative audio mode ('esp32' | 'laptop')."""
    mode = payload.get("audio_mode") or payload.get("mode")
    if not mode or str(mode).strip().lower() not in {"esp32", "laptop"}:
        raise HTTPException(status_code=400, detail=f"Invalid audio mode: '{mode}'. Must be 'esp32' or 'laptop'.")
    clean_mode = str(mode).strip().lower()
    update_db_settings({"audio_mode": clean_mode})
    return {"success": True, "audio_mode": clean_mode, "message": f"Audio mode set to {clean_mode}"}


@router.post("/laptop")
async def process_laptop_audio(
    request: Request,
    audio: UploadFile = File(...),
    student_id: int = Form(1),
    session_id: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
    format: str = Query("json", description="Response format: 'json' or 'audio'"),
    test_mode: Optional[bool] = Query(None, description="Test mode: bypass wake word")
):
    """
    Core Laptop Audio Voice Pipeline.
    Accepts browser microphone recording (WebM, WAV, OGG), transcribes via Whisper STT,
    executes tutor/coach dialog pipeline, synthesizes speech via TTS, and returns
    audio response URL for laptop speaker playback.
    Does NOT require ESP32 connection or report ESP32 hardware activity.
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


@router.get("/cache/{filename}")
async def get_laptop_cached_audio(filename: str):
    """Serves synthesized speech audio files directly from the laptop audio router."""
    from app.api.esp32 import get_cached_audio
    return await get_cached_audio(filename)
