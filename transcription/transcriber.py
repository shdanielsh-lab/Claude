"""Speech-to-text transcription backed by a Hugging Face Whisper model."""

from functools import lru_cache

from transformers import pipeline

MODEL_ID = "openai/whisper-medium"


@lru_cache(maxsize=1)
def _asr_pipeline():
    return pipeline("automatic-speech-recognition", model=MODEL_ID)


# Noisy/emotional audio can push Whisper into runaway repetition loops even
# with long-form generation. no_repeat_ngram_size hard-blocks repeating a
# 3-gram, which stops the loops without needing the model to lose confidence.
_REPETITION_GUARD = {"no_repeat_ngram_size": 3}


def _transcribe(audio_path: str, task: str) -> dict:
    # return_timestamps=True triggers Whisper's built-in long-form generation
    # (condition_on_prev_tokens + repetition/no-speech guards) for audio over
    # 30s, instead of the naive chunk_length_s split which hallucinates loops.
    generate_kwargs = dict(_REPETITION_GUARD)
    if task == "translate":
        generate_kwargs["task"] = "translate"
    return _asr_pipeline()(audio_path, return_timestamps=True, generate_kwargs=generate_kwargs)


def transcribe_file(audio_path: str) -> str:
    return _transcribe(audio_path, task="transcribe")["text"].strip()


def translate_file(audio_path: str) -> str:
    """Transcribe non-English speech directly into an English transcript."""
    return _transcribe(audio_path, task="translate")["text"].strip()


def transcribe_segments(audio_path: str, task: str = "transcribe") -> list[tuple[float, float, str]]:
    """Return (start_seconds, end_seconds, text) segments for speaker alignment."""
    result = _transcribe(audio_path, task=task)
    segments = []
    for chunk in result["chunks"]:
        start, end = chunk["timestamp"]
        segments.append((start, end if end is not None else start, chunk["text"].strip()))
    return segments
