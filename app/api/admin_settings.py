from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from app.db.database import get_db_settings, update_db_settings, mask_api_key
from app.db.models import SystemSettingsUpdate, SystemStatusResponse, ProviderStatusItem
from app.services.llm_service import llm_service

router = APIRouter(prefix="/api/v1/admin/settings", tags=["Admin Settings"])


@router.get("")
async def get_settings():
    """Retrieve current system settings and provider configurations with masked keys."""
    settings_data = get_db_settings()
    safe_data = dict(settings_data)
    for key in ["groq_api_key", "openai_api_key", "qwen_api_key"]:
        raw_val = (safe_data.get(key) or "").strip().strip("'\"")
        safe_data[f"{key}_configured"] = bool(raw_val)
        safe_data[f"{key}_masked"] = mask_api_key(raw_val)
        # Never expose raw plaintext secret to client
        safe_data[key] = ""
    return safe_data


@router.post("")
async def save_settings(payload: SystemSettingsUpdate):
    """Update system settings (API keys, active provider, system prompts, TTS settings)."""
    updates = payload.model_dump(exclude_unset=True)
    active_provider = updates.get("active_provider")
    if active_provider is not None and active_provider.strip().lower() not in {"groq", "openai", "qwen", "ollama"}:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {active_provider}")
    if active_provider is not None:
        updates["active_provider"] = active_provider.strip().lower()
    updated = update_db_settings(updates)

    # Mask secrets in returned response
    safe_updated = dict(updated)
    for key in ["groq_api_key", "openai_api_key", "qwen_api_key"]:
        raw_val = (safe_updated.get(key) or "").strip().strip("'\"")
        safe_updated[f"{key}_configured"] = bool(raw_val)
        safe_updated[f"{key}_masked"] = mask_api_key(raw_val)
        safe_updated[key] = ""

    return {"success": True, "message": "Settings updated successfully", "settings": safe_updated}


@router.get("/provider-status")
async def check_provider_status():
    """Performs live connectivity check on all configured LLM providers."""
    db_conf = get_db_settings()
    active = db_conf.get("active_provider") or "groq"
    results = await llm_service.verify_all_providers()
    return {
        "active_provider": active,
        "primary_provider": "groq (GPT OSS 120B)",
        "providers": results
    }


@router.get("/provider-status/{provider}")
async def check_single_provider_status(provider: str):
    """Performs live connectivity check on a single LLM provider using canonical configuration."""
    provider_clean = provider.strip().lower()
    if provider_clean not in ["groq", "openai", "qwen", "ollama"]:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    return await llm_service.verify_provider_connectivity(provider_clean)


@router.get("/models/{provider}")
@router.get("/providers/{provider}/models")
async def get_provider_models(provider: str):
    """Exposes available and recommended models for a specific LLM provider."""
    provider_clean = provider.strip().lower()
    if provider_clean not in ["groq", "openai", "qwen", "ollama"]:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    return await llm_service.get_available_models(provider_clean)


# Secondary router to support exact route: GET /api/v1/admin/providers/{provider}/models
providers_router = APIRouter(prefix="/api/v1/admin/providers", tags=["Admin Providers"])

@providers_router.get("/{provider}/models")
async def get_admin_provider_models_alt(provider: str):
    """Exposes available and recommended models for a specific LLM provider at /api/v1/admin/providers/{provider}/models."""
    provider_clean = provider.strip().lower()
    if provider_clean not in ["groq", "openai", "qwen", "ollama"]:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    return await llm_service.get_available_models(provider_clean)



