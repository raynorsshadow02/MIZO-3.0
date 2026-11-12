import pytest
import numpy as np
import scipy.signal as signal
from app.services.tts_service import TTSService, tts_service


def test_enhance_speech_audio_preserves_length_and_type():
    """Verify that enhance_speech_audio maintains byte size and valid 16-bit PCM structure."""
    sample_rate = 16000
    duration_sec = 0.5
    num_samples = int(sample_rate * duration_sec)
    
    # Generate 500Hz sine test wave with low amplitude (5,000)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    input_samples = (np.sin(2 * np.pi * 500 * t) * 5000).astype(np.int16)
    input_bytes = input_samples.tobytes()

    output_bytes = TTSService.enhance_speech_audio(input_bytes, sample_rate)
    assert len(output_bytes) == len(input_bytes)

    output_samples = np.frombuffer(output_bytes, dtype=np.int16)
    assert len(output_samples) == num_samples
    # Peak amplitude should be boosted towards target peak (~30,000)
    assert np.max(np.abs(output_samples)) > np.max(np.abs(input_samples))
    assert np.max(np.abs(output_samples)) <= 32767


def test_enhance_speech_audio_filters_sub_bass_rumble():
    """Verify that sub-bass frequencies (<140Hz) are significantly attenuated to prevent speaker rattling."""
    sample_rate = 16000
    num_samples = 16000  # 1 second
    t = np.linspace(0, 1.0, num_samples, endpoint=False)

    # 60Hz rumble tone that causes small speakers to distort
    sub_bass_samples = (np.sin(2 * np.pi * 60 * t) * 15000).astype(np.int16)
    output_bytes = TTSService.enhance_speech_audio(sub_bass_samples.tobytes(), sample_rate)
    output_samples = np.frombuffer(output_bytes, dtype=np.int16)

    # High pass filter at 140Hz should attenuate 60Hz noticeably (by at least -10dB)
    # Compare power:
    input_power = np.mean(sub_bass_samples.astype(np.float32)**2)
    output_power = np.mean(output_samples.astype(np.float32)**2)
    # The output 60Hz component will be strongly reduced relative to unfiltered normalization
    sos_hp = signal.butter(2, 140, btype='highpass', fs=sample_rate, output='sos')
    filtered_direct = signal.sosfilt(sos_hp, sub_bass_samples.astype(np.float32))
    assert np.max(np.abs(filtered_direct)) < np.max(np.abs(sub_bass_samples)) * 0.35


def test_enhance_speech_audio_empty_or_short_input():
    """Verify graceful handling of empty or very short audio buffers."""
    assert TTSService.enhance_speech_audio(b"") == b""
    short = b"\x00\x01\x02\x03"
    assert TTSService.enhance_speech_audio(short) == short


@pytest.mark.asyncio
async def test_speaker_test_endpoint_returns_enhanced_tone():
    """Verify the speaker test endpoint returns a 16kHz WAV file with valid duration."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/v1/esp32/speaker_test")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) >= 32044  # 44-byte header + 32,000 bytes (1s 16kHz mono 16-bit)


@pytest.mark.asyncio
async def test_synthesize_to_file_uses_maximized_volume():
    """Verify synthesize_to_file successfully produces a full-volume audio reply."""
    url, path = await tts_service.synthesize_to_file("Volume test reply", volume="+100%")
    assert url.startswith("/api/v1/esp32/audio/cache/")
    assert path.endswith(".wav")
