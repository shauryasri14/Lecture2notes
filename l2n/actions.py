"""Extraction, validation and deduplication of academic action items."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from .errors import Lecture2NotesError
from .models import Chunk
from .summarize import LLM
from .timeutil import format_timestamp, parse_timestamp

log = logging.getLogger("l2n.actions")

ALLOWED_TYPES = (
    "assignment", "lab_submission", "quiz", "exam", "project",
    "presentation", "attendance", "reading", "other",
)
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_RELATIVE_RE = re.compile(
    r"\b(next|this|last|tomorrow|today|tonight|yesterday|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday|week|weekend|month|end of|after|before|within|day|days)\b", re.I)
_STOP = {"the", "a", "an", "of", "to", "on", "in", "for", "and", "by", "is", "be", "your", "their", "that", "this"}

EXTRACT_SYSTEM = (
    "You extract academic obligations from lecture transcripts. You only report items that are "
    "explicitly stated. You never guess dates. You answer with JSON only."
)

EXTRACT_PROMPT = """TASK: EXTRACT_ACTIONS
Find academic action items in the transcript excerpt below (time range {rng}):
assignment announcements, homework/lab submissions, quiz and exam dates, project deadlines,
presentation dates, attendance requirements explicitly stated, required readings or preparation tasks.

Rules:
- Only include an item if the transcript explicitly supports it. If there is none, return {{"action_items": []}}.
- "deadline_text" must be the lecturer's ORIGINAL wording (e.g. "next Friday"), or null.
- "deadline_date" must ALWAYS be null unless the transcript itself states an unambiguous calendar date (YYYY-MM-DD). Never compute dates from "next Friday" etc.
- "source_timestamp" is the HH:MM:SS of the segment where it was said (use the [start - end] stamps).
- "evidence" is a short quote or close paraphrase from the transcript (max 2 sentences).
- "type" is one of: {types}.
- "confidence" is one of: high, medium, low.

Return exactly this JSON shape:
{{"action_items": [{{"type": "assignment", "description": "...", "deadline_text": null, "deadline_date": null,
"source_timestamp": "00:00:00", "evidence": "...", "confidence": "high", "needs_verification": true}}]}}

TRANSCRIPT:
{text}
"""

RETRY_SUFFIX = (
    "\n\nYour previous answer was not valid JSON in the required shape. "
    'Reply with ONLY one JSON object: {"action_items": [...]}. No prose, no code fences.'
)


class ActionValidationError(ValueError):
    """A single model-produced item is unusable."""


@dataclass
class ActionItem:
    type: str
    description: str
    deadline_text: str | None
    deadline_date: str | None
    source_timestamp: str
    evidence: str
    confidence: str
    needs_verification: bool
    notes: list[str] = field(default_factory=list)
    additional_mentions: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "description": self.description,
            "deadline_text": self.deadline_text,
            "deadline_date": self.deadline_date,
            "source_timestamp": self.source_timestamp,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "needs_verification": self.needs_verification,
            "notes": list(self.notes),
            "additional_mentions": list(self.additional_mentions),
        }


# ---------------------------------------------------------------- parsing
def parse_json_payload(text: str) -> Any:
    """Parse model output as JSON, tolerating code fences and surrounding prose."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty model output")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        a, b = cleaned.find(open_c), cleaned.rfind(close_c)
        if a != -1 and b > a:
            try:
                return json.loads(cleaned[a : b + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no valid JSON found in model output")


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def validate_action_item(raw: Any, *, chunk: Chunk, lecture_date: date | None = None) -> ActionItem:
    """Validate one raw dict. Raises ActionValidationError if it must be dropped;
    otherwise returns a sanitised item with notes explaining any downgrades."""
    if not isinstance(raw, dict):
        raise ActionValidationError("item is not an object")
    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ActionValidationError("missing description")
    evidence = raw.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ActionValidationError("no supporting evidence")

    notes: list[str] = []
    verify = bool(raw.get("needs_verification", False))

    typ = str(raw.get("type", "other")).strip().lower().replace(" ", "_").replace("-", "_")
    if typ not in ALLOWED_TYPES:
        typ = "other"

    conf = str(raw.get("confidence", "low")).strip().lower()
    if conf not in CONFIDENCE_RANK:
        conf = "low"
        notes.append("confidence missing/invalid; set to low")

    dtext = raw.get("deadline_text")
    dtext = dtext.strip() if isinstance(dtext, str) and dtext.strip() else None
    if dtext and _norm_ws(dtext) not in _norm_ws(chunk.plain_text):
        conf = "low"
        verify = True
        notes.append("deadline wording not found verbatim in the transcript; check the recording")

    ddate: str | None = None
    raw_date = raw.get("deadline_date")
    if raw_date not in (None, ""):
        try:
            parsed = date.fromisoformat(str(raw_date))
        except ValueError:
            notes.append(f"invalid deadline_date {raw_date!r} discarded")
        else:
            if lecture_date is None:
                notes.append("calendar date discarded: no reliable lecture date was provided")
            elif dtext is None:
                notes.append("calendar date discarded: no deadline wording to support it")
            else:
                ddate = parsed.isoformat()
                if _RELATIVE_RE.search(dtext):
                    verify = True
                    notes.append("date derived from relative wording; verify")
    if ddate is None:
        verify = True

    ts = raw.get("source_timestamp")
    secs = parse_timestamp(ts) if isinstance(ts, str) else None
    if secs is None or not (int(chunk.start) - 1 <= secs <= int(chunk.end) + 1):
        ts = format_timestamp(chunk.start)
        verify = True
        notes.append("source timestamp missing or outside the chunk; set to chunk start")
    else:
        ts = format_timestamp(secs)

    return ActionItem(
        type=typ,
        description=description.strip()[:300],
        deadline_text=dtext,
        deadline_date=ddate,
        source_timestamp=str(ts),
        evidence=evidence.strip()[:400],
        confidence=conf,
        needs_verification=verify,
        notes=notes,
    )


# ---------------------------------------------------------------- extraction
@dataclass
class ExtractionResult:
    items: list[ActionItem] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    error: str | None = None


def extract_chunk_actions(
    client: LLM, chunk: Chunk, *, max_attempts: int = 3, lecture_date: date | None = None
) -> ExtractionResult:
    """Ask the model for action items in one chunk. Malformed JSON triggers a
    bounded number of retries; afterwards the chunk is reported as failed and
    no items are fabricated."""
    base = EXTRACT_PROMPT.format(
        rng=chunk.time_range, types=", ".join(ALLOWED_TYPES), text=chunk.timestamped_text
    )
    last_error = "unknown error"
    for attempt in range(max_attempts):
        prompt = base if attempt == 0 else base + RETRY_SUFFIX
        try:
            raw_text = client.generate(prompt, system=EXTRACT_SYSTEM, json_mode=True)
        except Lecture2NotesError as exc:
            return ExtractionResult(error=f"LLM request failed: {exc}")
        try:
            payload = parse_json_payload(raw_text)
            if isinstance(payload, list):
                payload = {"action_items": payload}
            if not isinstance(payload, dict) or not isinstance(payload.get("action_items"), list):
                raise ValueError("JSON lacks an 'action_items' list")
        except ValueError as exc:
            last_error = f"malformed JSON ({exc})"
            log.warning("Chunk %d: %s (attempt %d/%d)", chunk.index + 1, last_error, attempt + 1, max_attempts)
            continue
        result = ExtractionResult()
        for entry in payload["action_items"]:
            try:
                result.items.append(validate_action_item(entry, chunk=chunk, lecture_date=lecture_date))
            except ActionValidationError as exc:
                result.rejected.append(str(exc))
                log.info("Chunk %d: rejected action item: %s", chunk.index + 1, exc)
        return result
    return ExtractionResult(error=f"{last_error} after {max_attempts} attempts")


# ---------------------------------------------------------------- dedupe
def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP}


def _similar(a: ActionItem, b: ActionItem) -> bool:
    if a.type != b.type:
        return False
    if a.deadline_text and b.deadline_text and _norm_ws(a.deadline_text) != _norm_ws(b.deadline_text):
        return False
    ta, tb = _tokens(a.description), _tokens(b.description)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.6


def _ts(item: ActionItem) -> float:
    return parse_timestamp(item.source_timestamp) or 0.0


def deduplicate_items(items: list[ActionItem]) -> list[ActionItem]:
    """Merge repeated announcements, keeping every piece of evidence."""
    merged: list[ActionItem] = []
    for item in sorted(items, key=_ts):
        for existing in merged:
            if _similar(existing, item):
                if (item.source_timestamp, item.evidence) != (existing.source_timestamp, existing.evidence):
                    mention = {"source_timestamp": item.source_timestamp, "evidence": item.evidence}
                    if mention not in existing.additional_mentions:
                        existing.additional_mentions.append(mention)
                existing.additional_mentions.extend(
                    m for m in item.additional_mentions if m not in existing.additional_mentions)
                if CONFIDENCE_RANK[item.confidence] > CONFIDENCE_RANK[existing.confidence]:
                    existing.confidence = item.confidence
                existing.needs_verification = existing.needs_verification or item.needs_verification
                existing.deadline_text = existing.deadline_text or item.deadline_text
                existing.deadline_date = existing.deadline_date or item.deadline_date
                existing.notes = list(dict.fromkeys(existing.notes + item.notes))
                break
        else:
            merged.append(item)
    return merged


def extract_all_actions(
    client: LLM,
    chunks: list[Chunk],
    *,
    max_attempts: int = 3,
    lecture_date: date | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[ActionItem], list[str]]:
    """Run extraction over every chunk. Returns (deduplicated items, failures)."""
    all_items: list[ActionItem] = []
    failures: list[str] = []
    for chunk in chunks:
        if progress:
            progress(f"  Extracting action items from chunk {chunk.index + 1}/{len(chunks)} ...")
        res = extract_chunk_actions(client, chunk, max_attempts=max_attempts, lecture_date=lecture_date)
        if res.error:
            failures.append(f"chunk {chunk.index + 1} ({chunk.time_range}): {res.error}")
            if progress:
                progress(f"  ! Action extraction FAILED for chunk {chunk.index + 1}: {res.error}")
        all_items.extend(res.items)
    return deduplicate_items(all_items), failures
