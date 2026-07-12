"""Speech-to-text transcription backed by a Hugging Face Whisper model."""

from functools import lru_cache

from transformers import pipeline

MODEL_ID = "openai/whisper-tiny"


@lru_cache(maxsize=1)
def _asr_pipeline():
    return pipeline("automatic-speech-recognition", model=MODEL_ID)


def transcribe_file(audio_path: str) -> str:
    result = _asr_pipeline()(audio_path)
    return result["text"].strip()
