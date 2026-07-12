"""Shared audio file helpers."""

import subprocess
import tempfile


def to_wav(audio_path: str) -> str:
    """Convert audio to a 16kHz mono WAV via ffmpeg, tolerating malformed containers
    (e.g. raw AAC streams) that trip up stricter decoders like pyannote's torchcodec backend."""
    wav_path = tempfile.mktemp(suffix=".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
        check=True,
        capture_output=True,
    )
    return wav_path
