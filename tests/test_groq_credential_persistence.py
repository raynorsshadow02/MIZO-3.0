import os

from app.config import settings
from app.db.database import get_db_settings, init_db, update_db_settings
from app.services.llm_service import LLMService


def test_groq_key_bootstraps_once_then_database_wins_after_restart(tmp_path, monkeypatch):
    """A bootstrap key must never replace an Admin-saved SQLite credential."""
    database_path = tmp_path / "mizo.db"
    bootstrap_key = "gsk_bootstrap_test_key_a"
    admin_key_b = "gsk_admin_saved_test_key_b"
    admin_key_c = "gsk_admin_saved_test_key_c"

    monkeypatch.setattr(settings, "DATABASE_PATH", database_path)
    monkeypatch.setattr(settings, "GROQ_API_KEY", bootstrap_key)
    monkeypatch.setattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("GROQ_API_KEY", bootstrap_key)

    # First run: seed an empty database from the bootstrap environment value.
    init_db()
    assert get_db_settings()["groq_api_key"] == bootstrap_key

    # Admin save: SQLite and runtime immediately use the newly saved credential.
    update_db_settings({
        "groq_api_key": admin_key_b,
        "groq_model": "openai/gpt-oss-120b",
    })
    assert get_db_settings()["groq_api_key"] == admin_key_b
    assert settings.GROQ_API_KEY == admin_key_b
    assert LLMService.get_provider_config("groq")["api_key"] == admin_key_b

    # Simulate restart with the old .env/bootstrap value still present.
    monkeypatch.setattr(settings, "GROQ_API_KEY", bootstrap_key)
    monkeypatch.setenv("GROQ_API_KEY", bootstrap_key)
    init_db()
    assert get_db_settings()["groq_api_key"] == admin_key_b
    assert settings.GROQ_API_KEY == admin_key_b
    assert os.environ["GROQ_API_KEY"] == admin_key_b
    assert LLMService.get_provider_config("groq")["api_key"] == admin_key_b

    # A subsequent Admin save continues to replace the persisted/runtime value.
    update_db_settings({"groq_api_key": admin_key_c})
    assert get_db_settings()["groq_api_key"] == admin_key_c
    assert settings.GROQ_API_KEY == admin_key_c
    assert LLMService.get_provider_config("groq")["api_key"] == admin_key_c
