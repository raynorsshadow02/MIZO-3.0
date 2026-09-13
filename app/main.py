from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from app.config import settings
from app.db.database import init_db
from app.api.esp32 import router as esp32_router
from app.api.admin_settings import router as admin_settings_router
from app.api.students import router as students_router
from app.api.conversations import router as conversations_router
from app.api.knowledge import router as knowledge_router
from app.api.devices import router as devices_router

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: Initialize DB, setup paths on startup."""
    print(f"[Mizo 3.0] Starting {settings.APP_NAME} in [{settings.ENVIRONMENT}] mode...")
    init_db()
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    yield
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
            "environment": settings.ENVIRONMENT
        }

    # Mount API Routers
    app.include_router(esp32_router)
    app.include_router(admin_settings_router)
    app.include_router(students_router)
    app.include_router(conversations_router)
    app.include_router(knowledge_router)
    app.include_router(devices_router)

    # Mount Static Admin Dashboard
    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


app = create_app()
