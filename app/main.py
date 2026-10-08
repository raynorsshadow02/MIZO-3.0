import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from app.config import settings
from app.db.database import init_db
from app.api.esp32 import router as esp32_router
from app.api.admin_settings import router as admin_settings_router, providers_router
from app.api.students import router as students_router
from app.api.conversations import router as conversations_router
from app.api.knowledge import router as knowledge_router
from app.api.subjects import router as subjects_router
from app.api.speech import router as speech_router
from app.api.devices import router as devices_router
from app.api.diagnostics import router as diagnostics_router
from app.services.llm_service import llm_service

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: Initialize DB, setup paths, enter WAKE_LISTENING state on startup."""
    print("=" * 65)
    print(f"[Mizo 3.0] Starting {settings.APP_NAME} in [{settings.ENVIRONMENT}] mode...")
    init_db()
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize Voice System and enter WAKE_LISTENING state on startup
    from app.services.voice_service import voice_manager
    voice_manager.start_wake_listener()

    # Perform background connectivity check at startup
    async def run_startup_check():
        try:
            print("[Mizo 3.0] Verifying configured AI providers...")
            statuses = await llm_service.verify_all_providers()
            for s in statuses:
                status_icon = "✓" if s["status"] in ("connected", "ready") else ("-" if s["status"] == "not_configured" else "⚠️" if s["status"] == "degraded" else "✗")
                print(f"  [{status_icon}] Provider: {s['provider']:<8} | Status: {s['status']:<14} | Ready: {str(s.get('generation_ready', False)):<5} | Model: {s['model']}")
            print("[Mizo 3.0] Primary LLM: Groq (GPT OSS 120B)")
        except Exception as e:
            print(f"[Mizo 3.0] Startup provider verification note: {e}")

    asyncio.create_task(run_startup_check())

    yield
    voice_manager.stop_wake_listener()
    print(f"[Mizo 3.0] Shutting down {settings.APP_NAME}...")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="Mizo 3.0 — AI-Powered Cloud Server for Personalized Learning and ESP32 Robotic Tutor",
        version="3.0.0",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        print(f"[ERROR] Unhandled exception on {request.url}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)}
        )

    # Health check
    @app.get("/api/v1/health", tags=["Health"])
    async def health():
        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": "3.0.0",
            "environment": settings.ENVIRONMENT,
            "primary_llm": "GPT OSS 120B (Groq)"
        }

    # Mount API Routers
    app.include_router(esp32_router)
    app.include_router(admin_settings_router)
    app.include_router(providers_router)
    app.include_router(students_router)
    app.include_router(conversations_router)
    app.include_router(knowledge_router)
    app.include_router(subjects_router)
    app.include_router(speech_router)
    app.include_router(devices_router)
    app.include_router(diagnostics_router)

    # Mount Static Admin Dashboard
    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


app = create_app()

