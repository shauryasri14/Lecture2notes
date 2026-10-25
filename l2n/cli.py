"""Command-line interface."""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from . import __version__
from .config import Config, config_from_env
from .errors import ConfigError, Lecture2NotesError
from .ollama_client import OllamaClient
from .pipeline import COMPLETE, FAILED, PARTIAL, load_transcript_json, run_pipeline
from .transcribe import Transcriber

EXIT = {COMPLETE: 0, PARTIAL: 2, FAILED: 1}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lecture2notes",
        description="Turn lecture recordings into study notes, revision questions and deadlines, entirely locally "
                    "(faster-whisper + Ollama).",
    )
    p.add_argument("recordings", nargs="*", type=Path, help="audio/video file(s) to process")
    p.add_argument("--watch", metavar="DIR", type=Path, help="watch DIR and process new recordings (needs watchdog)")
    p.add_argument("--from-transcript", metavar="JSON", type=Path,
                   help="skip transcription and use an existing transcript.json")
    p.add_argument("--model", help="Ollama text model (default: env GEMMA_MODEL or gemma3:4b)")
    p.add_argument("--whisper-model", help="Whisper size, e.g. tiny, base, small, medium (default: small)")
    p.add_argument("--language", help="language code such as en (default: auto-detect)")
    p.add_argument("--output-dir", type=Path, help="where to write results (default: notes_out)")
    p.add_argument("--chunk-size", type=int, help="words per chunk (default 1200)")
    p.add_argument("--chunk-overlap", type=int, help="overlapping words between chunks (default 100)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Whisper device (default: auto)")
    p.add_argument("--compute-type", help="Whisper compute type, e.g. int8, float16, float32 (default: int8 on CPU)")
    p.add_argument("--lecture-date", metavar="YYYY-MM-DD",
                   help="actual lecture date; only then may relative deadlines yield calendar dates")
    p.add_argument("--verbose", action="store_true", help="show detailed logs")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def resolve_config(args: argparse.Namespace) -> Config:
    cfg = config_from_env()
    lecture_date = None
    if args.lecture_date:
        try:
            lecture_date = date.fromisoformat(args.lecture_date)
        except ValueError:
            raise ConfigError(f"--lecture-date must be YYYY-MM-DD, got {args.lecture_date!r}")
    cfg = cfg.with_overrides(
        llm_model=args.model, whisper_model=args.whisper_model, language=args.language,
        output_dir=args.output_dir, chunk_size_words=args.chunk_size,
        chunk_overlap_words=args.chunk_overlap, device=args.device,
        compute_type=args.compute_type, lecture_date=lecture_date,
        verbose=True if args.verbose else None,
    )
    return cfg.validate()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.recordings or args.watch or args.from_transcript):
        parser.error("provide a recording, --from-transcript JSON, or --watch DIR")
    if args.watch and (args.recordings or args.from_transcript):
        parser.error("--watch cannot be combined with explicit recordings or --from-transcript")
    if args.from_transcript and args.recordings:
        parser.error("--from-transcript replaces the recording argument; do not give both")

    try:
        cfg = resolve_config(args)
    except Lecture2NotesError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    client = OllamaClient(
        cfg.ollama_host, cfg.llm_model, timeout=cfg.request_timeout, max_retries=cfg.max_retries,
        temperature=cfg.temperature, num_ctx=cfg.num_ctx, num_predict=cfg.num_predict,
    )
    transcriber = None if args.from_transcript else Transcriber(
        cfg.whisper_model, cfg.device, cfg.compute_type, cfg.vad_filter)

    try:
        if args.watch:
            from .watcher import watch_directory

            def process(path: Path) -> str:
                res = run_pipeline(path, cfg, client=client, transcriber=transcriber)
                _report(res)
                return res.status

            watch_directory(args.watch, process)
            return 0

        if args.from_transcript:
            transcript = load_transcript_json(args.from_transcript)
            res = run_pipeline(args.from_transcript, cfg, client=client, transcript=transcript)
            _report(res)
            return EXIT[res.status]

        worst = 0
        for rec in args.recordings:
            print(f"\n=== {rec} ===")
            res = run_pipeline(rec, cfg, client=client, transcriber=transcriber)
            _report(res)
            worst = max(worst, EXIT[res.status])
        return worst
    except Lecture2NotesError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Partial results (if any) were kept in the output folder.", file=sys.stderr)
        return 130


def _report(res) -> None:  # noqa: ANN001
    if res.status == FAILED:
        print(f"\nFAILED: {res.error}", file=sys.stderr)
        if res.output_dir:
            print(f"Partial results (if any): {res.output_dir}", file=sys.stderr)
        return
    label = "Done" if res.status == COMPLETE else "Finished WITH WARNINGS (notes are incomplete)"
    print(f"\n{label}: {res.output_dir}")
    for w in res.warnings:
        print(f"  - {w}")


if __name__ == "__main__":
    sys.exit(main())
