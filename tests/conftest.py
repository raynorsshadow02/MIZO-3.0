import sys
import os
from pathlib import Path
import pytest

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Use isolated test database for all tests
TEST_DB_PATH = BASE_DIR / "data" / "test_mizo.db"
os.environ["DATABASE_PATH"] = str(TEST_DB_PATH)

from app.config import settings
settings.DATABASE_PATH = TEST_DB_PATH

from app.db.database import init_db, update_db_settings
from app.services.voice_service import voice_manager, VoiceState

@pytest.fixture(autouse=True)
def setup_test_db():
    # Freshly initialize test database
    if TEST_DB_PATH.exists():
        try:
            TEST_DB_PATH.unlink()
        except Exception:
            pass
    init_db()
    update_db_settings({"active_provider": "groq"})
    voice_manager.state = VoiceState.ACTIVE_LISTENING
    voice_manager.is_interrupted = False
    yield
    # Cleanup after test if needed

