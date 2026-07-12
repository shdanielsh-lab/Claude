"""Speaker diarization backed by pyannote.audio.

Requires an HF_TOKEN environment variable from an account that has accepted
the gated model terms for pyannote/speaker-diarization-3.1 and
pyannote/segmentation-3.0 on huggingface.co.
"""

import os
from functools import lru_cache

from pyannote.audio import Pipeline

MODEL_ID = "pyannote/speaker-diarization-3.1"


@lru_cache(maxsize=1)
def _diarization_pipeline():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is required for speaker diarization. "
            "Accept the model terms at huggingface.co/pyannote/speaker-diarization-3.1 "
            "and huggingface.co/pyannote/segmentation-3.0, then set HF_TOKEN to an access token."
        )
    return Pipeline.from_pretrained(MODEL_ID, token=token)


def diarize(audio_path: str) -> list[tuple[float, float, str]]:
    """Return a list of (start_seconds, end_seconds, speaker_label) turns."""
    output = _diarization_pipeline()(audio_path)
    # exclusive_speaker_diarization has no overlapping turns, which makes
    # aligning it to non-overlapping Whisper segments straightforward.
    annotation = output.exclusive_speaker_diarization
    return [(turn.start, turn.end, speaker) for turn, _, speaker in annotation.itertracks(yield_label=True)]
