"""Output directory handling and file generation."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .actions import ActionItem
from .errors import OutputError
from .models import Segment
from .transcribe import format_transcript_txt

_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def safe_filename(name: str, max_len: int = 80) -> str:
    """Turn arbitrary text into a safe single path component."""
    name = Path(str(name).replace("\\", "/")).name  # drop any directory parts
    name = re.sub(r"[^\w.\- ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"[\s_]+", "_", name).strip("._ -")
    name = name[:max_len].strip("._ -")
    if not name or name.lower() in _RESERVED:
        name = f"lecture_{name}" if name else "lecture"
    return name


def unique_output_dir(base: Path, name: str) -> Path:
    """Create base/<safe name>, or base/<safe name>_2 etc. if it exists.
    mkdir(exist_ok=False) keeps this race-free."""
    safe = safe_filename(name)
    try:
        base.mkdir(parents=True, exist_ok=True)
        base_resolved = base.resolve()
        for n in range(1, 10_000):
            candidate = base / (safe if n == 1 else f"{safe}_{n}")
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            if base_resolved not in candidate.resolve().parents:
                raise OutputError(f"Refusing to write outside {base}")
            return candidate
    except OSError as exc:
        raise OutputError(f"Cannot create output directory under {base}: {exc}") from exc
    raise OutputError(f"Could not find a free output folder name for '{safe}'")


def write_text(path: Path, content: str) -> None:
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise OutputError(f"Cannot write {path}: {exc}") from exc


def write_json(path: Path, data: Any) -> None:
    write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_transcript_files(out_dir: Path, segments: list[Segment], language: str | None, duration: float | None) -> None:
    write_text(out_dir / "transcript.txt", format_transcript_txt(segments))
    write_json(out_dir / "transcript.json", {
        "language": language,
        "duration": duration,
        "segments": [s.to_dict() for s in segments],
    })


def write_action_items(out_dir: Path, items: list[ActionItem], *, complete: bool, failures: list[str], lecture_date: str | None) -> None:
    write_json(out_dir / "action_items.json", {
        "action_items": [i.to_dict() for i in items],
        "extraction_complete": complete,
        "extraction_failures": failures,
        "lecture_date": lecture_date,
        "note": "Deadlines are the lecturer's wording; deadline_date is null unless a reliable date was available.",
    })


def render_action_items_markdown(items: list[ActionItem], extraction_complete: bool = True) -> str:
    lines = ["## Action Items and Deadlines", ""]
    if not items:
        lines.append("_No assignments, exams or deadlines were detected in the processed transcript._"
                     if extraction_complete else
                     "_None detected, but extraction failed for some parts of the lecture; check the recording._")
        return "\n".join(lines) + "\n"
    lines.append("> Wording is quoted from the lecturer. Verify every deadline against the recording or course page.")
    lines.append("")
    for it in items:
        said = f'"{it.deadline_text}"' if it.deadline_text else "not stated"
        date_txt = it.deadline_date or "not determined"
        flag = " ⚠ verify" if it.needs_verification else ""
        lines.append(f"- **[{it.type.replace('_', ' ').title()}]** {it.description}{flag}")
        lines.append(f"  - Deadline as said: {said} · Calendar date: {date_txt} · Heard at {it.source_timestamp} · Confidence: {it.confidence}")
        lines.append(f"  - Evidence: {it.evidence}")
        for m in it.additional_mentions:
            lines.append(f"  - Also mentioned at {m['source_timestamp']}: {m['evidence']}")
        for n in it.notes:
            lines.append(f"  - Note: {n}")
    return "\n".join(lines) + "\n"


def build_notes_markdown(
    body: str, items: list[ActionItem], *, title_fallback: str,
    warnings: list[str], extraction_complete: bool,
) -> str:
    body = body.strip()
    if body and not re.match(r"^#\s+\S", body):
        body = f"# {title_fallback}\n\n{body}"
    elif not body:
        body = f"# {title_fallback}\n\n_No study notes could be generated._"
    # Insert warnings right after the title line.
    first_nl = body.find("\n")
    head, rest = (body, "") if first_nl == -1 else (body[:first_nl], body[first_nl:])
    banner = ""
    if warnings:
        banner = "\n\n> **⚠ These notes are INCOMPLETE or partially unverified.**\n" + "\n".join(f"> - {w}" for w in warnings)
    rest = rest.rstrip()
    parts = [head + banner, rest.lstrip("\n") if rest else "", render_action_items_markdown(items, extraction_complete)]
    return "\n\n".join(p.strip("\n") for p in parts if p.strip()) + "\n"


def write_status(out_dir: Path, status: str, stages: list[str], error: str | None = None) -> None:
    try:
        write_json(out_dir / "status.json", {
            "status": status,
            "completed_stages": stages,
            "error": error,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    except OutputError:
        pass  # status is best-effort


def setup_run_logger(out_dir: Path, verbose: bool) -> logging.Logger:
    """Attach processing.log (and optionally console) to the 'l2n' logger tree."""
    logger = logging.getLogger("l2n")
    logger.setLevel(logging.DEBUG)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    try:
        fh = logging.FileHandler(out_dir / "processing.log", encoding="utf-8")
    except OSError as exc:
        raise OutputError(f"Cannot create processing.log: {exc}") from exc
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    fh.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(fh)
    if verbose:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)
    return logger


def close_run_logger() -> None:
    logger = logging.getLogger("l2n")
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
