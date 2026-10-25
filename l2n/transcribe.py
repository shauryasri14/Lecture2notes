"""Local speech-to-text with faster-whisper (lazy import, model reuse)."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from .errors import AudioError, TranscriptionError
from .models import Segment, TranscriptResult
from .timeutil import format_timestamp

log = logging.getLogger("l2n.transcribe")

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg", ".opus", ".aac",
    ".wma", ".webm", ".mkv", ".mov",
}


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def validate_audio_file(path: Path) -> Path:
    """Cheap checks before loading any model."""
    path = Path(path)
    if not path.exists():
        raise AudioError(f"Audio file not found: {path}")
    if not path.is_file():
        raise AudioError(f"Not a file: {path}")
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise AudioError(
            f"Unsupported file type '{path.suffix}'. Supported: {', '.join(sorted(AUDIO_EXTENSIONS))}"
        )
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            fh.read(1)
    except OSError as exc:
        raise AudioError(f"Cannot read audio file {path}: {exc}") from exc
    if size == 0:
        raise AudioError(f"Audio file is empty: {path}")
    return path


def decode_with_ffmpeg(path: Path) -> Any:
    """Decode to 16 kHz mono float32 with the ffmpeg CLI (fallback that bypasses
    PyAV). Loads the whole waveform in memory (~230 MB per hour of audio)."""
    import numpy as np  # installed together with faster-whisper

    exe = shutil.which("ffmpeg")
    if exe is None:
        raise AudioError("FFmpeg is not installed or not on PATH (needed as a fallback decoder; see README).")
    proc = subprocess.run(
        [exe, "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", "replace").strip()[:300]
        raise AudioError(f"FFmpeg could not decode {path.name}: {err or 'no audio stream found'}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _is_pyav_mismatch(exc: BaseException) -> bool:
    return "metadata_errors" in str(exc)


def format_transcript_txt(segments: list[Segment]) -> str:
    """Timestamped transcript, one blank line between segments."""
    return "\n\n".join(
        f"[{format_timestamp(s.start)} - {format_timestamp(s.end)}] {s.text}" for s in segments
    ) + ("\n" if segments else "")


class Transcriber:
    """Wraps faster-whisper. The model is loaded once and reused across files."""

    def __init__(
        self,
        model_name: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        vad_filter: bool = True,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.vad_filter = vad_filter
        self._factory = model_factory
        self._model: Any = None

    def _resolved_compute_type(self) -> str:
        if self.compute_type != "auto":
            return self.compute_type
        return "float16" if self.device == "cuda" else "int8"

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        factory = self._factory
        if factory is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore
            except ImportError as exc:
                raise TranscriptionError(
                    "faster-whisper is not installed. Run: pip install -r requirements.txt"
                ) from exc
            factory = WhisperModel
        log.info("Loading Whisper model '%s' (device=%s, compute_type=%s)",
                 self.model_name, self.device, self._resolved_compute_type())
        try:
            self._model = factory(self.model_name, device=self.device,
                                  compute_type=self._resolved_compute_type())
        except Exception as exc:  # download/initialisation problems vary widely
            raise TranscriptionError(
                f"Could not load Whisper model '{self.model_name}': {exc}. "
                "The first run downloads the model, so internet access is required once; "
                "otherwise try a smaller model (--whisper-model base) or --device cpu."
            ) from exc
        return self._model

    def transcribe(
        self, path: Path, language: str | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> TranscriptResult:
        path = validate_audio_file(path)
        model = self._load()
        say = progress or (lambda _m: None)
        say(f"Transcribing {path.name} with Whisper '{self.model_name}' ...")
        segments: list[Segment] = []
        try:
            # faster-whisper decodes lazily in chunks; the waveform is streamed
            # by the generator rather than all text being held before returning.
            try:
                seg_iter, info = model.transcribe(
                    str(path), language=language, vad_filter=self.vad_filter, beam_size=5
                )
            except TypeError as exc:
                if not _is_pyav_mismatch(exc):
                    raise
                log.warning("PyAV/faster-whisper version mismatch (%s); decoding with ffmpeg instead", exc)
                say("  PyAV is incompatible with this faster-whisper; decoding with FFmpeg instead ...")
                seg_iter, info = model.transcribe(
                    decode_with_ffmpeg(path), language=language, vad_filter=self.vad_filter, beam_size=5
                )
            duration = float(getattr(info, "duration", 0) or 0)
            lang = getattr(info, "language", None) or language
            last_pct = -1
            for seg in seg_iter:
                text = (seg.text or "").strip()
                if text:
                    segments.append(Segment(float(seg.start), float(seg.end), text))
                if duration > 0:
                    pct = min(100, int(seg.end / duration * 100) // 10 * 10)
                    if pct != last_pct:
                        last_pct = pct
                        say(f"  Transcription progress: ~{pct}% ({format_timestamp(seg.end)} / {format_timestamp(duration)})")
        except TranscriptionError:
            raise
        except Exception as exc:
            msg = str(exc)
            if self.device != "cpu" and any(k in msg.lower() for k in ("cublas", "cudnn", "cuda")):
                log.warning("GPU libraries unavailable (%s); retrying on CPU", msg)
                say("  GPU libraries missing; switching to CPU (int8) ...")
                self.device, self._model = "cpu", None
                if self.compute_type in ("float16", "int8_float16"):
                    self.compute_type = "auto"
                return self.transcribe(path, language, progress)
            if "ffmpeg" in msg.lower() and not ffmpeg_available():
                raise AudioError("FFmpeg appears to be missing. Install it and make sure it is on PATH (see README).") from exc
            raise AudioError(f"Could not decode/transcribe {path.name}: {msg}") from exc
        result = TranscriptResult(segments=segments, language=lang, duration=duration or None)
        if not result.has_text:
            raise TranscriptionError(
                "Transcription produced no usable text (silent audio, wrong language setting, or VAD removed everything)."
            )
        say(f"  Transcription finished: {len(segments)} segments, language={lang}")
        return result
