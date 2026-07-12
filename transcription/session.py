"""CLI: record audio (or use an existing file) and transcribe it."""

import argparse
import sys
import tempfile

from .diarizer import diarize
from .recorder import record_to_file
from .speakers import assign_speakers
from .transcriber import transcribe_file, transcribe_segments, translate_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Recording transcript session")
    parser.add_argument("--file", help="Transcribe an existing WAV file instead of recording")
    parser.add_argument("--seconds", type=float, default=5.0, help="Recording duration in seconds")
    parser.add_argument(
        "--translate", action="store_true", help="Translate non-English speech into an English transcript"
    )
    parser.add_argument("--diarize", action="store_true", help="Label each line with the speaker who said it")
    args = parser.parse_args()

    if args.file:
        audio_path = args.file
    else:
        audio_path = tempfile.mktemp(suffix=".wav")
        print(f"Recording {args.seconds}s of audio...", file=sys.stderr)
        record_to_file(audio_path, args.seconds)

    print(f"Transcribing {audio_path}...", file=sys.stderr)
    task = "translate" if args.translate else "transcribe"

    if args.diarize:
        print("Diarizing speakers...", file=sys.stderr)
        turns = diarize(audio_path)
        segments = transcribe_segments(audio_path, task=task)
        for speaker, text in assign_speakers(segments, turns):
            print(f"[{speaker}] {text}")
    else:
        text = translate_file(audio_path) if args.translate else transcribe_file(audio_path)
        print(text)


if __name__ == "__main__":
    main()
