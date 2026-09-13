import io
import wave
import uuid
import struct
import asyncio
import edge_tts
from pathlib import Path
from typing import Optional, Tuple
from app.config import settings
from app.db.database import get_db_settings


class TTSService:
    """Text-to-Speech service using Edge-TTS with 16kHz 16-bit Mono WAV output for ESP32."""

    @staticmethod
    def _create_wav_header(sample_rate: int = 16000, bits_per_sample: int = 16, channels: int = 1, data_size: int = 0) -> bytes:
        """Create standard 44-byte RIFF WAV header."""
        byte_rate = sample_rate * channels * (bits_per_sample // 8)
        block_align = channels * (bits_per_sample // 8)
        chunk_size = 36 + data_size

        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            chunk_size,
            b'WAVE',
            b'fmt ',
            16,              # Subchunk1Size for PCM
            1,               # AudioFormat 1 = PCM
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b'data',
            data_size
        )
        return header

    @classmethod
    async def synthesize_to_file(cls, text: str, voice: Optional[str] = None, rate: Optional[str] = None, pitch: Optional[str] = None) -> Tuple[str, str]:
        """
        Synthesizes text into a WAV file.
        Returns: (relative_audio_url, absolute_file_path)
        """
        db_conf = get_db_settings()
        selected_voice = voice or db_conf.get("tts_voice") or settings.DEFAULT_TTS_VOICE
        selected_rate = rate or db_conf.get("tts_rate") or settings.DEFAULT_TTS_RATE
        selected_pitch = pitch or db_conf.get("tts_pitch") or settings.DEFAULT_TTS_PITCH

        filename = f"mizo_reply_{uuid.uuid4().hex[:10]}.wav"
        output_path = settings.AUDIO_CACHE_DIR / filename
        mp3_temp_path = settings.AUDIO_CACHE_DIR / f"temp_{uuid.uuid4().hex[:8]}.mp3"

        try:
            # 1. Synthesize via Edge-TTS
            communicate = edge_tts.Communicate(
                text=text,
                voice=selected_voice,
                rate=selected_rate,
                pitch=selected_pitch
            )
            await communicate.save(str(mp3_temp_path))

            # 2. Convert or package as WAV for ESP32 MAX98357A
            # Try to decode MP3 into PCM if av or soundfile available, or provide clean WAV
            cls._convert_mp3_to_wav(mp3_temp_path, output_path)

            if mp3_temp_path.exists():
                try:
                    mp3_temp_path.unlink()
                except Exception:
                    pass

            audio_url = f"/api/v1/esp32/audio/cache/{filename}"
            return audio_url, str(output_path)

        except Exception as e:
            # Fallback: Generate a clean PCM tone/WAV if network is down
            cls._generate_fallback_wav(output_path, text)
            if mp3_temp_path.exists():
                try:
                    mp3_temp_path.unlink()
                except Exception:
                    pass
            audio_url = f"/api/v1/esp32/audio/cache/{filename}"
            return audio_url, str(output_path)

    @classmethod
    def _convert_mp3_to_wav(cls, mp3_path: Path, wav_path: Path):
        """Convert MP3 to 16kHz Mono 16-bit PCM WAV using av/scipy or save directly."""
        converted = False
        container = None
        try:
            import av
            container = av.open(str(mp3_path))
            resampler = av.AudioResampler(format='s16', layout='mono', rate=settings.AUDIO_SAMPLE_RATE)
            pcm_bytes = bytearray()
            for frame in container.decode(audio=0):
                resampled_frames = resampler.resample(frame)
                for rf in resampled_frames:
                    pcm_bytes.extend(rf.to_ndarray().tobytes())

            with wave.open(str(wav_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(settings.AUDIO_SAMPLE_RATE)
                wf.writeframes(pcm_bytes)
            converted = True
        except Exception:
            converted = False
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass

        if not converted:
            # If pyav/ffmpeg is unavailable, copy audio directly so it can still be served
            import shutil
            shutil.copyfile(mp3_path, wav_path)

    @classmethod
    def _generate_fallback_wav(cls, wav_path: Path, text: str):
        """Generates simple silence / notification beep WAV when offline."""
        import math
        sample_rate = 16000
        duration = 1.0  # seconds
        num_samples = int(sample_rate * duration)
        with wave.open(str(wav_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            frames = bytearray()
            for i in range(num_samples):
                val = int(32767.0 * 0.1 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
                frames.extend(struct.pack('<h', val))
            wf.writeframes(frames)


tts_service = TTSService()
