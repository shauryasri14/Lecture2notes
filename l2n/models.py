"""Plain data structures shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .timeutil import format_timestamp


@dataclass(frozen=True)
class Segment:
    """One transcription segment with the timestamps Whisper reported."""

    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Segment":
        return cls(float(d["start"]), float(d["end"]), str(d["text"]))


@dataclass
class TranscriptResult:
    segments: list[Segment]
    language: str | None = None
    duration: float | None = None

    @property
    def has_text(self) -> bool:
        return any(s.text.strip() for s in self.segments)


@dataclass
class Chunk:
    """A contiguous run of segments sent to the LLM together."""

    index: int
    segments: list[Segment] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return max(s.end for s in self.segments)

    @property
    def word_count(self) -> int:
        return sum(len(s.text.split()) for s in self.segments)

    @property
    def plain_text(self) -> str:
        return " ".join(s.text for s in self.segments)

    @property
    def timestamped_text(self) -> str:
        return "\n".join(
            f"[{format_timestamp(s.start)} - {format_timestamp(s.end)}] {s.text}"
            for s in self.segments
        )

    @property
    def time_range(self) -> str:
        return f"{format_timestamp(self.start)} - {format_timestamp(self.end)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "word_count": self.word_count,
            "segments": [s.to_dict() for s in self.segments],
        }
