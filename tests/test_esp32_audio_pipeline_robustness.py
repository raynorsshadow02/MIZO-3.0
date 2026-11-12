import io
import re
import wave
import struct
from pathlib import Path
from unittest.mock import patch, AsyncMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.voice_service import voice_manager, VoiceState
from app.services.llm_service import LLMResult

client = TestClient(app)


def convert_slot32_to_pcm16(slot32: int) -> int:
    """Python reference implementation of the firmware's convertSlot32ToPcm16() logic."""
    # Simulate 32-bit signed integer arithmetic
    if slot32 > 0x7FFFFFFF:
        slot32 -= 0x100000000

    sample = slot32 >> 14
    if sample > 32767:
        sample = 32767
    elif sample < -32768:
        sample = -32768
    return sample


def test_inmp441_32bit_to_16bit_conversion_math():
    """Validates 32-bit I2S slot to 16-bit PCM conversion mathematics and saturation limits."""
    # Zero input -> zero output
    assert convert_slot32_to_pcm16(0) == 0

    # Normal 24-bit speech range inside 32-bit slot
    # Value: +3276800 in 32-bit slot -> shifted by 14 gives 200
    assert convert_slot32_to_pcm16(3276800) == 200

    # Negative speech sample
    assert convert_slot32_to_pcm16(-3276800) == -200

    # Maximum 16-bit positive sample: 32767 * 16384 = 536854528
    assert convert_slot32_to_pcm16(32767 << 14) == 32767

    # Positive saturation clipping test: value exceeding 32767 must clamp to 32767 without overflow
    extreme_positive = 0x7FFFFFFF  # Max positive 32-bit
    assert convert_slot32_to_pcm16(extreme_positive) == 32767

    # Negative saturation clipping test: value exceeding -32768 must clamp to -32768 without underflow
    extreme_negative = -0x80000000  # Max negative 32-bit
    assert convert_slot32_to_pcm16(extreme_negative) == -32768


def test_wav_header_structure_and_validity():
    """Validates 44-byte RIFF WAVE header conformity for 16kHz 16-bit Mono PCM."""
    sample_rate = 16000
    channels = 1
    bits_per_sample = 16
    pcm_data = bytearray(sample_rate * 2)  # 1 second of silence

    total_audio_len = len(pcm_data)
    total_data_len_plus_header = total_audio_len + 36
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        total_data_len_plus_header,
        b'WAVE',
        b'fmt ',
        16,               # Subchunk1Size for PCM
        1,                # AudioFormat 1 = PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        total_audio_len
    )

    assert len(header) == 44
    assert header[0:4] == b'RIFF'
    assert header[8:12] == b'WAVE'
    assert header[12:16] == b'fmt '
    assert header[36:40] == b'data'

    # Open with Python standard wave library to ensure 100% specification compliance
    complete_wav = io.BytesIO(header + pcm_data)
    with wave.open(complete_wav, 'rb') as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 16000


@pytest.mark.asyncio
async def test_post_esp32_audio_format_audio_success():
    """Verifies format=audio returns valid 16kHz mono WAV audio stream with correct headers and content."""
    sample_rate = 16000
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for _ in range(8000):
            frames.extend(struct.pack('<h', 0))
        wf.writeframes(frames)
    wav_io.seek(0)

    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt, \
         patch("app.services.llm_service.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:

        mock_stt.return_value = {
            "text": "Hey Mizo, what is reinforcement learning?",
            "duration_seconds": 0.5,
            "language": "en"
        }
        mock_llm.return_value = LLMResult(
            text="Reinforcement learning is a machine learning paradigm where an agent learns through trial and error.",
            provider="groq",
            model="llama-3.3-70b-versatile",
            is_fallback=False,
            latency_ms=120.0
        )

        files = {"audio": ("inmp441.wav", wav_io, "audio/wav")}
        data = {"student_id": "1", "mode": "coach"}
        res = client.post("/api/v1/esp32/audio?format=audio", files=files, data=data)

        assert res.status_code == 200
        assert "audio/wav" in res.headers["content-type"].lower()
        assert len(res.content) > 44

        # Validate WAV stream format
        stream_io = io.BytesIO(res.content)
        with wave.open(stream_io, 'rb') as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() > 0


def test_post_esp32_audio_standby_silence_returns_json_safe_for_esp32():
    """Verifies that silence in STANDBY returns JSON with application/json, preventing I2S DAC noise."""
    voice_manager.transition_to(VoiceState.STANDBY, "test_setup")

    sample_rate = 16000
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for _ in range(4000):
            frames.extend(struct.pack('<h', 0))
        wf.writeframes(frames)
    wav_io.seek(0)

    with patch("app.api.esp32._require_primary_student") as mock_student, \
         patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt:
        mock_student.return_value = {
            "id": 1,
            "name": "Shahid",
            "onboarding_completed": True,
            "onboarding_step": "completed"
        }
        mock_stt.return_value = {
            "text": "",
            "duration_seconds": 0.25,
            "language": "en"
        }

        files = {"audio": ("inmp441.wav", wav_io, "audio/wav")}
        data = {"student_id": "1", "mode": "coach"}
        res = client.post("/api/v1/esp32/audio?format=audio", files=files, data=data)

        assert res.status_code == 200
        # Must be application/json so ESP32 knows NOT to feed it to I2S speaker
        assert "application/json" in res.headers["content-type"].lower()
        payload = res.json()
        assert payload.get("ignored") is True


@pytest.mark.asyncio
async def test_post_esp32_audio_wake_listening_no_wake_word_returns_json():
    """Verifies speech without wake word in WAKE_LISTENING returns JSON with ignored=True."""
    voice_manager.transition_to(VoiceState.WAKE_LISTENING, "test_setup")

    sample_rate = 16000
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for _ in range(4000):
            frames.extend(struct.pack('<h', 0))
        wf.writeframes(frames)
    wav_io.seek(0)

    with patch("app.services.stt_service.stt_service.transcribe_audio_with_metadata", new_callable=AsyncMock) as mock_stt:
        mock_stt.return_value = {
            "text": "The weather outside is cloudy today.",
            "duration_seconds": 1.0,
            "language": "en"
        }

        files = {"audio": ("inmp441.wav", wav_io, "audio/wav")}
        data = {"student_id": "1", "mode": "coach"}
        res = client.post("/api/v1/esp32/audio?format=audio", files=files, data=data)

        assert res.status_code == 200
        assert "application/json" in res.headers["content-type"].lower()
        payload = res.json()
        assert payload.get("ignored") is True


def test_dashboard_browser_audio_strictly_hardware_only():
    """Verifies that app.js properly guards hardware mode, supports audio modes, and startVoiceRecording checks mode before getUserMedia."""
    app_js_path = Path("app/static/js/app.js")
    assert app_js_path.exists(), "app.js must exist"

    content = app_js_path.read_text(encoding="utf-8")

    # 1. Authoritative currentAudioMode and isLaptopAudioDisabled must be defined
    assert "currentAudioMode" in content, "currentAudioMode must be defined"
    assert "isLaptopAudioDisabled" in content, "isLaptopAudioDisabled must be defined"

    # 2. playAudio must exist and support audio modes
    play_audio_match = re.search(r"function\s+playAudio\s*\([^)]*\)\s*\{([\s\S]+?)\n\}", content)
    assert play_audio_match, "playAudio function must exist"

    # 3. startVoiceRecording must check isLaptopAudioDisabled and return early before getUserMedia
    record_fn_match = re.search(r"async\s+function\s+startVoiceRecording\s*\([^)]*\)\s*\{([\s\S]+?)try\s*\{", content)
    assert record_fn_match, "startVoiceRecording function must be present"
    record_prefix = record_fn_match.group(1)
    assert "isLaptopAudioDisabled" in record_prefix, "startVoiceRecording must check isLaptopAudioDisabled before getUserMedia"
    assert "return;" in record_prefix, "startVoiceRecording must return early when isLaptopAudioDisabled is true"

    # 4. index.html must contain both audio mode options
    index_html = Path("app/static/index.html").read_text(encoding="utf-8")
    assert "card-mode-esp32" in index_html, "ESP32 mode option must be present in index.html"
    assert "card-mode-laptop" in index_html, "Laptop audio mode option must be present in index.html"
    assert "Audio Input/Output Mode" in index_html, "Audio mode selector section header must be present in index.html"
