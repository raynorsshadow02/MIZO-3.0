from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from app.db.database import get_db_settings, update_db_settings
from app.db.models import SystemSettingsUpdate, SystemStatusResponse, ProviderStatusItem
from app.services.llm_service import llm_service

router = APIRouter(prefix="/api/v1/admin/settings", tags=["Admin Settings"])


@router.get("")
async def get_settings():
    """Retrieve current system settings and provider configurations with masked keys."""
    settings_data = get_db_settings()
    masked = dict(settings_data)
    for key in ["groq_api_key", "openai_api_key", "qwen_api_key"]:
        val = masked.get(key) or ""
        if len(val) > 8:
            masked[f"{key}_masked"] = f"{val[:4]}...{val[-4:]}"
        else:
            masked[f"{key}_masked"] = "Not configured" if not val else "••••••••"
    return masked


@router.post("")
async def save_settings(payload: SystemSettingsUpdate):
    """Update system settings (API keys, active provider, system prompts, TTS settings)."""
    updates = payload.model_dump(exclude_unset=True)
    updated = update_db_settings(updates)
    return {"success": True, "message": "Settings updated successfully", "settings": updated}


@router.get("/provider-status")
async def check_provider_status():
    """Performs live connectivity check on all configured LLM providers."""
    db_conf = get_db_settings()
    active = db_conf.get("active_provider") or "groq"
    results = await llm_service.verify_all_providers()
    return {
        "active_provider": active,
        "primary_provider": "groq (Llama 3.3 70B)",
        "providers": results
    }

