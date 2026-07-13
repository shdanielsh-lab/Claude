"""Align transcript segments with diarization speaker turns."""


def assign_speakers(
    segments: list[tuple[float, float, str]], turns: list[tuple[float, float, str]]
) -> list[tuple[str, float, float, str]]:
    """Label each (start, end, text) segment with the speaker turn it overlaps most,
    then merge consecutive segments from the same speaker into one (speaker, start, end, text) line."""
    labeled = []
    for seg_start, seg_end, text in segments:
        if not text:
            continue
        best_speaker, best_overlap = "UNKNOWN", 0.0
        for turn_start, turn_end, speaker in turns:
            overlap = min(seg_end, turn_end) - max(seg_start, turn_start)
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, speaker
        labeled.append((best_speaker, seg_start, seg_end, text))

    merged: list[list] = []
    for speaker, start, end, text in labeled:
        if merged and merged[-1][0] == speaker:
            merged[-1][2] = end
            merged[-1][3] += " " + text
        else:
            merged.append([speaker, start, end, text])
    return [tuple(line) for line in merged]
