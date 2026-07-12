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


def transcribe_file(audio_path: str) -> str:
    # return_timestamps=True triggers Whisper's built-in long-form generation
    # (condition_on_prev_tokens + repetition/no-speech guards) for audio over
    # 30s, instead of the naive chunk_length_s split which hallucinates loops.
    result = _asr_pipeline()(audio_path, return_timestamps=True, generate_kwargs=_REPETITION_GUARD)
    return result["text"].strip()


def translate_file(audio_path: str) -> str:
    """Transcribe non-English speech directly into an English transcript."""
    result = _asr_pipeline()(
        audio_path, return_timestamps=True, generate_kwargs={**_REPETITION_GUARD, "task": "translate"}
    )
    return result["text"].strip()
