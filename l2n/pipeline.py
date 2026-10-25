"""Pipeline orchestration: transcribe -> chunk -> summarise -> merge -> actions -> write."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .actions import extract_all_actions
from .chunking import chunk_segments
from .config import Config
from .errors import Lecture2NotesError, TranscriptionError
from .models import Segment, TranscriptResult
from .outputs import (
    build_notes_markdown, close_run_logger, setup_run_logger, unique_output_dir,
    write_action_items, write_json, write_status, write_text, write_transcript_files,
)
from .summarize import LLM, merge_summaries, summarize_chunks
from .transcribe import ffmpeg_available, validate_audio_file

log = logging.getLogger("l2n.pipeline")

COMPLETE, PARTIAL, FAILED = "complete", "partial", "failed"


class Transcribes(Protocol):
    def transcribe(self, path: Path, language: str | None = None,
                   progress: Callable[[str], None] | None = None) -> TranscriptResult: ...


class LLMClient(LLM, Protocol):
    def check_available(self) -> None: ...


@dataclass
class PipelineResult:
    status: str
    output_dir: Path | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    stages: list[str] = field(default_factory=list)


def load_transcript_json(path: Path) -> TranscriptResult:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        segs = [Segment.from_dict(s) for s in data["segments"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TranscriptionError(f"Cannot read transcript JSON {path}: {exc}") from exc
    return TranscriptResult(segs, data.get("language"), data.get("duration"))


def run_pipeline(
    source: Path,
    cfg: Config,
    *,
    client: LLMClient,
    transcriber: Transcribes | None = None,
    transcript: TranscriptResult | None = None,
    progress: Callable[[str], None] = lambda m: print(m, flush=True),
) -> PipelineResult:
    """Process one lecture. Expected failures are returned as a PipelineResult
    (status 'failed'); only KeyboardInterrupt propagates, after status is saved."""
    source = Path(source)
    stages: list[str] = []
    warnings: list[str] = []
    out_dir: Path | None = None

    def done(stage: str) -> None:
        stages.append(stage)
        if out_dir:
            write_status(out_dir, "running", stages)
        progress(f"[ok] {stage}")

    try:
        # --- preflight: cheap checks first so no expensive work is wasted
        if transcript is None:
            if transcriber is None:
                raise TranscriptionError("No transcriber configured")
            validate_audio_file(source)
        client.check_available()
        if not cfg.host_is_local:
            progress(f"WARNING: Ollama host {cfg.ollama_host} is not localhost; lecture text will leave this machine.")

        out_dir = unique_output_dir(cfg.output_dir, source.stem)
        setup_run_logger(out_dir, cfg.verbose)
        log.info("Processing %s (whisper=%s, llm=%s)", source.name, cfg.whisper_model, cfg.llm_model)
        if not cfg.host_is_local:
            log.warning("Ollama host %s is not local", cfg.ollama_host)
        progress(f"Output folder: {out_dir}")
        done("preflight checks")

        # --- transcription
        if transcript is None:
            if not ffmpeg_available():
                log.warning("ffmpeg not found on PATH (faster-whisper can still decode many formats via PyAV)")
            transcript = transcriber.transcribe(source, cfg.language, progress)  # type: ignore[union-attr]
        if not transcript.has_text:
            raise TranscriptionError("Transcript contains no usable text.")
        write_transcript_files(out_dir, transcript.segments, transcript.language, transcript.duration)
        log.info("Transcript saved: %d segments", len(transcript.segments))
        done("transcription (transcript.txt / transcript.json saved)")

        # --- chunking
        chunks = chunk_segments(transcript.segments, cfg.chunk_size_words, cfg.chunk_overlap_words)
        log.info("Created %d chunks", len(chunks))
        done(f"chunking ({len(chunks)} chunk(s))")

        # --- chunk summaries
        summaries = summarize_chunks(client, chunks, progress)
        write_json(out_dir / "chunk_summaries.json", [
            {"index": s.index, "time_range": s.time_range, "summary": s.text, "error": s.error}
            for s in summaries])
        failed = [s for s in summaries if not s.ok]
        for s in failed:
            warnings.append(f"Chunk {s.index + 1} ({s.time_range}) could not be summarised ({s.error}); its content is missing from these notes.")
        if len(failed) == len(summaries):
            raise Lecture2NotesError("All chunk summaries failed; no notes could be generated.")
        done(f"chunk summaries ({len(summaries) - len(failed)}/{len(summaries)} ok)")

        # --- merge
        merge = merge_summaries(client, summaries, max_chars=cfg.merge_max_chars, progress=progress)
        warnings.extend(merge.warnings)
        done("merge" if merge.merged_by_llm else "merge (fallback: unmerged notes)")

        # --- action items (from transcript chunks, not from the summary)
        items, failures = extract_all_actions(
            client, chunks, max_attempts=cfg.json_attempts,
            lecture_date=cfg.lecture_date, progress=progress)
        for f in failures:
            warnings.append(f"Action-item extraction failed for {f}. Deadlines in that part may be missing.")
        write_action_items(out_dir, items, complete=not failures, failures=failures,
                           lecture_date=cfg.lecture_date.isoformat() if cfg.lecture_date else None)
        done(f"action items ({len(items)} found)")

        # --- notes.md
        notes = build_notes_markdown(
            merge.text, items, title_fallback=f"Lecture Notes: {source.stem}",
            warnings=warnings, extraction_complete=not failures)
        write_text(out_dir / "notes.md", notes)
        stages.append("notes.md written")

        status = PARTIAL if warnings else COMPLETE
        write_status(out_dir, status, stages)
        log.info("Finished with status=%s warnings=%d", status, len(warnings))
        return PipelineResult(status, out_dir, warnings, None, stages)

    except KeyboardInterrupt:
        if out_dir:
            write_status(out_dir, "interrupted", stages, "interrupted by user")
            logging.getLogger("l2n").error("Interrupted by user; partial results kept in %s", out_dir)
        raise
    except Lecture2NotesError as exc:
        log.error("Pipeline failed: %s", exc)
        if out_dir:
            write_status(out_dir, FAILED, stages, str(exc))
        return PipelineResult(FAILED, out_dir, warnings, str(exc), stages)
    finally:
        close_run_logger()
