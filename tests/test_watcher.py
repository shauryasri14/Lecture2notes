from l2n.watcher import is_candidate, move_to, wait_until_stable


def test_wait_until_stable_waits_for_growth(tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"1")
    ticks = {"n": 0, "t": 0.0}

    def sleep(_):
        ticks["n"] += 1
        ticks["t"] += 1
        if ticks["n"] <= 3:                    # file still being copied
            f.write_bytes(b"1" * (ticks["n"] + 1))

    assert wait_until_stable(f, required_checks=3, sleep=sleep, monotonic=lambda: ticks["t"])
    assert ticks["n"] >= 6


def test_wait_until_stable_timeout_and_missing(tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"1")
    t = {"t": 0.0}

    def sleep(_):
        t["t"] += 1
        f.write_bytes(b"1" * int(t["t"] + 1))  # never stabilises

    assert not wait_until_stable(f, timeout=5, sleep=sleep, monotonic=lambda: t["t"])
    assert not wait_until_stable(tmp_path / "gone.mp3", sleep=lambda s: None)


def test_candidates_and_collision_safe_move(tmp_path):
    inbox = tmp_path
    a = inbox / "lec.mp3"
    a.write_bytes(b"x")
    assert is_candidate(a, inbox)
    assert not is_candidate(inbox / "notes.txt", inbox)
    assert not is_candidate(inbox / "processed" / "lec.mp3", inbox)
    d1 = move_to(a, inbox, "processed")
    a.write_bytes(b"y")
    d2 = move_to(a, inbox, "processed")
    assert d1 != d2 and d1.exists() and d2.exists() and not a.exists()
    assert not is_candidate(d1, inbox)
