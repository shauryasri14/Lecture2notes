import threading
import time

import pytest

pytest.importorskip("watchdog")
from l2n.watcher import watch_directory  # noqa: E402


def test_watch_processes_each_file_once_and_moves_it(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "existing.mp3").write_bytes(b"abc")
    seen, stop = [], threading.Event()

    def process(path):
        seen.append(path.name)
        return "failed" if path.name == "bad.mp3" else "complete"

    t = threading.Thread(target=lambda: watch_directory(inbox, process, log=lambda m: None,
                                                        stable_interval=0.05, stop_event=stop))
    t.start()
    time.sleep(0.5)
    (inbox / "new.m4a").write_bytes(b"abc")
    (inbox / "bad.mp3").write_bytes(b"abc")
    (inbox / "readme.txt").write_text("ignored")
    deadline = time.time() + 15
    while time.time() < deadline and len(seen) < 3:
        time.sleep(0.2)
    time.sleep(1)
    stop.set()
    t.join(10)
    assert sorted(seen) == ["bad.mp3", "existing.mp3", "new.m4a"]
    assert (inbox / "failed" / "bad.mp3").exists()
    assert (inbox / "processed" / "new.m4a").exists()
    assert (inbox / "readme.txt").exists()
