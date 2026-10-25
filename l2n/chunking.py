"""Deterministic transcript chunking (no Whisper or Ollama dependency)."""
from __future__ import annotations

from typing import Iterable

from .errors import ConfigError
from .models import Chunk, Segment


def _split_oversized(seg: Segment, size: int, overlap: int) -> list[Segment]:
    """Split one very long segment into word windows.

    Windows keep the parent's timestamps (we never invent timestamps from
    word counts) and overlap by `overlap` words so no text is lost.
    """
    words = seg.text.split()
    if len(words) <= size:
        return [seg]
    step = size - overlap
    pieces: list[Segment] = []
    i = 0
    while True:
        pieces.append(Segment(seg.start, seg.end, " ".join(words[i : i + size])))
        if i + size >= len(words):
            break
        i += step
    return pieces


def chunk_segments(
    segments: Iterable[Segment], chunk_size: int = 1200, overlap: int = 100
) -> list[Chunk]:
    """Group segments into chunks of <= chunk_size words, repeating roughly
    `overlap` words between neighbours.

    Segment boundaries are respected: overlap consists of whole trailing
    segments totalling <= `overlap` words. Forward progress is guaranteed
    because each chunk starts at least one unit after the previous start.
    """
    if chunk_size < 1:
        raise ConfigError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ConfigError("overlap must be >= 0 and smaller than chunk_size")

    units: list[Segment] = []
    for seg in segments:
        if seg.text.strip():
            units.extend(_split_oversized(seg, chunk_size, overlap))
    if not units:
        return []

    counts = [len(u.text.split()) for u in units]
    n = len(units)
    chunks: list[Chunk] = []
    i = 0
    while i < n:
        j, total = i, 0
        while j < n and (j == i or total + counts[j] <= chunk_size):
            total += counts[j]
            j += 1
        chunks.append(Chunk(index=len(chunks), segments=units[i:j]))
        if j >= n:
            break
        k, carried = j, 0
        while k - 1 > i and carried + counts[k - 1] <= overlap:
            k -= 1
            carried += counts[k]
        i = max(k, i + 1)
    return chunks
