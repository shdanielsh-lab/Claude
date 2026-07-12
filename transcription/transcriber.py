"""Speech-to-text transcription backed by a Hugging Face Whisper model."""

from functools import lru_cache

from transformers import pipeline

MODEL_ID = "openai/whisper-medium"


@lru_cache(maxsize=1)
def _asr_pipeline():
    # chunk_length_s splits audio longer than Whisper's ~30s window into
    # overlapping windows so recordings of arbitrary length can be transcribed.
    return pipeline("automatic-speech-recognition", model=MODEL_ID, chunk_length_s=30, stride_length_s=5)


def transcribe_file(audio_path: str) -> str:
    result = _asr_pipeline()(audio_path)
    return result["text"].strip()


def translate_file(audio_path: str) -> str:
    """Transcribe non-English speech directly into an English transcript."""
    result = _asr_pipeline()(audio_path, generate_kwargs={"task": "translate"})
    return result["text"].strip()
