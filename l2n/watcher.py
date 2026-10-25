"""Optional directory watching (requires `watchdog`)."""
from __future__ import annotations

import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from .errors import Lecture2NotesError
from .outputs import safe_filename
from .transcribe import AUDIO_EXTENSIONS

PROCESSED_DIR = "processed"
FAILED_DIR = "failed"


def is_candidate(path: Path, inbox: Path) -> bool:
    """Audio file directly in the inbox (not in processed/failed, not hidden/temp)."""
    path = Path(path)
    return (
        path.suffix.lower() in AUDIO_EXTENSIONS
        and path.parent.resolve() == Path(inbox).resolve()
        and not path.name.startswith(".")
    )


def wait_until_stable(
    path: Path, *, interval: float = 2.0, required_checks: int = 3, timeout: float = 900.0,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    """True once size+mtime are unchanged for `required_checks` consecutive polls
    and the file can be opened; False on timeout or if the file disappears."""
    path = Path(path)
    deadline = monotonic() + timeout
    last: tuple[int, float] | None = None
    stable = 0
    while monotonic() < deadline:
        try:
            st = path.stat()
        except OSError:
            return False
        sig = (st.st_size, st.st_mtime)
        if sig == last and st.st_size > 0:
            stable += 1
            if stable >= required_checks:
                try:
                    with path.open("rb"):
                        return True
                except OSError:
                    stable = 0
        else:
            stable = 0
        last = sig
        sleep(interval)
    return False


def move_to(path: Path, inbox: Path, folder: str) -> Path:
    dest_dir = Path(inbox) / folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    if dest.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = dest_dir / f"{safe_filename(path.stem)}_{stamp}{path.suffix}"
    shutil.move(str(path), str(dest))
    return dest


def watch_directory(
    inbox: Path,
    process: Callable[[Path], str],
    *,
    log: Callable[[str], None] = print,
    stable_interval: float = 2.0,
    stop_event: threading.Event | None = None,
) -> None:
    """Process audio files as they appear in `inbox`. `process` returns a
    status string ('complete'/'partial'/'failed'). Finished files move to
    inbox/processed or inbox/failed so they are never handled twice."""
    try:
        from watchdog.events import FileSystemEventHandler  # type: ignore
        from watchdog.observers import Observer  # type: ignore
    except ImportError as exc:
        raise Lecture2NotesError(
            "Directory watching needs the optional 'watchdog' package: pip install watchdog"
        ) from exc

    inbox = Path(inbox)
    inbox.mkdir(parents=True, exist_ok=True)
    q: "queue.Queue[Path]" = queue.Queue()
    pending: set[Path] = set()
    lock = threading.Lock()

    def enqueue(p: str | Path) -> None:
        p = Path(p)
        if not is_candidate(p, inbox):
            return
        key = p.resolve()
        with lock:
            if key in pending:
                return
            pending.add(key)
        q.put(p)

    class Handler(FileSystemEventHandler):  # type: ignore[misc]
        def on_created(self, event):  # noqa: ANN001
            if not event.is_directory:
                enqueue(event.src_path)

        def on_moved(self, event):  # noqa: ANN001
            if not event.is_directory:
                enqueue(event.dest_path)

    for existing in sorted(inbox.iterdir()):
        if existing.is_file():
            enqueue(existing)

    observer = Observer()
    observer.schedule(Handler(), str(inbox), recursive=False)
    observer.start()
    log(f"Watching {inbox.resolve()} for new recordings (Ctrl+C to stop) ...")
    stop = stop_event or threading.Event()
    try:
        while not stop.is_set():
            try:
                path = q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                log(f"Detected {path.name}; waiting for the copy to finish ...")
                if not wait_until_stable(path, interval=stable_interval):
                    log(f"Skipping {path.name}: file vanished or never stopped changing.")
                    continue
                try:
                    status = process(path)
                except Lecture2NotesError as exc:
                    log(f"Failed: {exc}")
                    status = "failed"
                except Exception as exc:  # keep the watcher alive on bugs
                    log(f"Unexpected error while processing {path.name}: {exc}")
                    status = "failed"
                if path.exists():
                    dest = move_to(path, inbox, FAILED_DIR if status == "failed" else PROCESSED_DIR)
                    log(f"Moved {path.name} -> {dest.parent.name}/ ({status})")
            finally:
                with lock:
                    pending.discard(path.resolve())
    except KeyboardInterrupt:
        log("Stopping watcher.")
    finally:
        observer.stop()
        observer.join(timeout=5)
