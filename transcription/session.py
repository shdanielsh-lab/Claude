"""CLI: record audio (or use an existing file) and transcribe it."""

import argparse
import sys
import tempfile

from .recorder import record_to_file
from .transcriber import transcribe_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Recording transcript session")
    parser.add_argument("--file", help="Transcribe an existing WAV file instead of recording")
    parser.add_argument("--seconds", type=float, default=5.0, help="Recording duration in seconds")
    args = parser.parse_args()

    if args.file:
        audio_path = args.file
    else:
        audio_path = tempfile.mktemp(suffix=".wav")
        print(f"Recording {args.seconds}s of audio...", file=sys.stderr)
        record_to_file(audio_path, args.seconds)

    print(f"Transcribing {audio_path}...", file=sys.stderr)
    text = transcribe_file(audio_path)
    print(text)


if __name__ == "__main__":
    main()
