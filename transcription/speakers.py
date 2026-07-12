"""Align transcript segments with diarization speaker turns."""


def assign_speakers(
    segments: list[tuple[float, float, str]], turns: list[tuple[float, float, str]]
) -> list[tuple[str, str]]:
    """Label each (start, end, text) segment with the speaker turn it overlaps most,
    then merge consecutive segments from the same speaker into one line each."""
    labeled = []
    for seg_start, seg_end, text in segments:
        if not text:
            continue
        best_speaker, best_overlap = "UNKNOWN", 0.0
        for turn_start, turn_end, speaker in turns:
            overlap = min(seg_end, turn_end) - max(seg_start, turn_start)
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, speaker
        labeled.append((best_speaker, text))

    merged: list[tuple[str, str]] = []
    for speaker, text in labeled:
        if merged and merged[-1][0] == speaker:
            merged[-1] = (speaker, merged[-1][1] + " " + text)
        else:
            merged.append((speaker, text))
    return merged
