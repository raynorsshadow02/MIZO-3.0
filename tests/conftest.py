import sys
from pathlib import Path
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.database import init_db, update_db_settings

@pytest.fixture(autouse=True)
def setup_test_db():
    init_db()
    update_db_settings({"active_provider": "groq"})
    from app.services.voice_service import voice_manager, VoiceState
    voice_manager.state = VoiceState.ACTIVE_LISTENING
    voice_manager.is_interrupted = False
