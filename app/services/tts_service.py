import io
import re
import wave
import uuid
import struct
import asyncio
import edge_tts
from pathlib import Path
from typing import Optional, Tuple
from app.config import settings
from app.db.database import get_db_settings


def map_tts_pronunciation(text: str) -> str:
    """
    Substitutes pronunciation-friendly phonetic representation strictly for the TTS audio generator.
    The canonical project name 'Mizo' is preserved in all user-facing UI, databases, transcripts, and logs.
    Edge-TTS pronounces 'Meezo' consistently and naturally as /ˈmiːzoʊ/ ('Mee-zo').
    """
    if not text:
        return ""

    def _replace_mizo(match):
        w = match.group(0)
        if w.isupper():
            return "MEEZO"
        if w[0].isupper():
            return "Meezo"
        return "meezo"

    return re.sub(r'\b[Mm][Ii][Zz][Oo]\b', _replace_mizo, text)


import logging

logger = logging.getLogger("mizo.tts")


class TTSService:
    """Text-to-Speech service using Edge-TTS with 16kHz 16-bit Mono WAV output for ESP32 and browser."""

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
    async def synthesize_to_file(cls, text: str, voice: Optional[str] = None, rate: Optional[str] = None, pitch: Optional[str] = None, volume: Optional[str] = None) -> Tuple[str, str]:
        """
        Synthesizes text into a WAV file with maximized acoustic volume and clarity.
        Returns: (relative_audio_url, absolute_file_path)
        """
        db_conf = get_db_settings()
        selected_voice = voice or db_conf.get("tts_voice") or settings.DEFAULT_TTS_VOICE
        selected_rate = rate or db_conf.get("tts_rate") or settings.DEFAULT_TTS_RATE
        selected_pitch = pitch or db_conf.get("tts_pitch") or settings.DEFAULT_TTS_PITCH
        selected_volume = volume or db_conf.get("tts_volume") or getattr(settings, "DEFAULT_TTS_VOLUME", "+100%")

        filename = f"mizo_reply_{uuid.uuid4().hex[:10]}.wav"
        output_path = settings.AUDIO_CACHE_DIR / filename
        mp3_temp_path = settings.AUDIO_CACHE_DIR / f"temp_{uuid.uuid4().hex[:8]}.mp3"

        spoken_text = map_tts_pronunciation(text)

        try:
            # 1. Synthesize via Edge-TTS with maximum output volume
            communicate = edge_tts.Communicate(
                text=spoken_text,
                voice=selected_voice,
                rate=selected_rate,
                volume=selected_volume,
                pitch=selected_pitch
            )
            await asyncio.wait_for(communicate.save(str(mp3_temp_path)), timeout=12.0)

            # 2. Convert or package as WAV for ESP32 MAX98357A and browser
            cls._convert_mp3_to_wav(mp3_temp_path, output_path)

            if mp3_temp_path.exists():
                try:
                    mp3_temp_path.unlink()
                except Exception:
                    pass

            size = output_path.stat().st_size if output_path.exists() else 0
            logger.info(f"[TTS] Edge-TTS successfully synthesized speech ({len(spoken_text)} chars) -> {output_path} ({size} bytes)")
            audio_url = f"/api/v1/esp32/audio/cache/{filename}"
            return audio_url, str(output_path)

        except Exception as e:
            logger.warning(f"[TTS] Edge-TTS synthesis failed: {e}. Attempting local offline speech synthesis...")
            if mp3_temp_path.exists():
                try:
                    mp3_temp_path.unlink()
                except Exception:
                    pass

            # 3. Attempt local offline speech synthesis (pyttsx3)
            local_success = await cls._synthesize_local_pyttsx3(spoken_text, output_path)
            if local_success:
                size = output_path.stat().st_size if output_path.exists() else 0
                logger.info(f"[TTS] Local offline speech synthesis (pyttsx3) succeeded for '{spoken_text[:35]}...' -> {output_path} ({size} bytes)")
                audio_url = f"/api/v1/esp32/audio/cache/{filename}"
                return audio_url, str(output_path)

            # 4. Fallback: Generate a clean PCM tone/WAV if all synthesis engines fail
            logger.error("[TTS] All speech synthesis methods failed. Generating diagnostic fallback tone.")
            cls._generate_fallback_wav(output_path, text)
            audio_url = f"/api/v1/esp32/audio/cache/{filename}"
            return audio_url, str(output_path)

    @classmethod
    async def _synthesize_local_pyttsx3(cls, text: str, wav_path: Path) -> bool:
        """Fallback local offline TTS using pyttsx3 when Edge-TTS is unavailable."""
        temp_raw_path = wav_path.parent / f"pyttsx3_raw_{uuid.uuid4().hex[:8]}.wav"
        try:
            def _run_pyttsx3():
                import pyttsx3
                engine = pyttsx3.init()
                engine.save_to_file(text, str(temp_raw_path))
                engine.runAndWait()

            await asyncio.to_thread(_run_pyttsx3)

            if not temp_raw_path.exists() or temp_raw_path.stat().st_size == 0:
                return False

            cls._convert_mp3_to_wav(temp_raw_path, wav_path)
            return wav_path.exists() and wav_path.stat().st_size > 44
        except Exception as err:
            logger.warning(f"[TTS] pyttsx3 local fallback failed: {err}")
            return False
        finally:
            if temp_raw_path.exists():
                try:
                    temp_raw_path.unlink()
                except Exception:
                    pass

    @staticmethod
    def enhance_speech_audio(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
        """
        Enhances raw 16-bit Mono PCM speech audio for maximized loudness, clarity,
        and diction on speakers (MAX98357A I2S DAC amplifier and laptop speakers).

        Pipeline:
        1. 2nd-order High-Pass Filter (HPF @ 140Hz) - cuts low-frequency mud and cone rattling
           that small speakers cannot physically reproduce cleanly.
        2. Presence Peaking EQ (+4.0 dB @ 3200Hz, Q=1.2) - clarifies consonant intelligibility
           and speech articulation.
        3. Upward Dynamic Range Compression & Peak Normalization - brings average speech RMS
           energy up to maximum broadcast acoustic levels with smooth soft-knee tanh limiting
           to completely prevent digital flat-top clipping.
        """
        if not pcm_bytes or len(pcm_bytes) < 64:
            return pcm_bytes

        try:
            import numpy as np
            import scipy.signal as signal

            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
            if len(samples) < 64:
                return pcm_bytes

            # 1. High-Pass Filter: Cuts muddiness & sub-bass rumble below 140Hz
            sos_hp = signal.butter(2, 140, btype='highpass', fs=sample_rate, output='sos')
            filtered = signal.sosfilt(sos_hp, samples)

            # 2. Presence Peaking Equalizer: +4.0 dB boost at 3200Hz (speech intelligibility band)
            gain_db = 4.0
            A = 10.0 ** (gain_db / 40.0)
            w0 = 2.0 * np.pi * 3200.0 / sample_rate
            alpha = np.sin(w0) / (2.0 * 1.2)
            b0 = 1.0 + alpha * A
            b1 = -2.0 * np.cos(w0)
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * np.cos(w0)
            a2 = 1.0 - alpha / A
            b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float32)
            a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float32)
            clarified = signal.lfilter(b, a, filtered)

            # 3. Upward Dynamic Range Compression & Soft Limiting
            peak = float(np.max(np.abs(clarified)))
            if peak > 10.0:
                # Dynamic upward gain: lifts quieter syllables to full presence
                boost_factor = max(1.5, 31500.0 / peak) * 1.5
                amplified = clarified * boost_factor

                # Soft-knee tanh compression above threshold to guarantee zero distortion
                thresh = 22000.0
                max_val = 32000.0
                mask = np.abs(amplified) > thresh
                compressed = np.copy(amplified)
                compressed[mask] = np.sign(amplified[mask]) * (
                    thresh + (max_val - thresh) * np.tanh((np.abs(amplified[mask]) - thresh) / (max_val - thresh))
                )
                out_samples = np.clip(compressed, -32768, 32767).astype(np.int16)
            else:
                out_samples = np.clip(clarified, -32768, 32767).astype(np.int16)

            return out_samples.tobytes()
        except Exception as e:
            logger.warning(f"[TTS] Audio enhancement filter bypassed: {e}")
            return pcm_bytes

    @classmethod
    def _convert_mp3_to_wav(cls, mp3_path: Path, wav_path: Path):
        """Convert MP3/WAV to 16kHz Mono 16-bit PCM WAV using av or save directly with clarity DSP."""
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

            # Apply DSP speech clarity filter & peak volume normalization
            enhanced_bytes = cls.enhance_speech_audio(bytes(pcm_bytes), settings.AUDIO_SAMPLE_RATE)

            with wave.open(str(wav_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(settings.AUDIO_SAMPLE_RATE)
                wf.writeframes(enhanced_bytes)
            converted = True
        except Exception as e:
            logger.warning(f"[TTS] Audio resampling conversion failed: {e}. Copying directly.")
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
