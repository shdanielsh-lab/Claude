"""Microphone recording to a WAV file."""

import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16_000


def record_to_file(output_path: str, seconds: float, sample_rate: int = SAMPLE_RATE) -> str:
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    sf.write(output_path, audio, sample_rate)
    return output_path
