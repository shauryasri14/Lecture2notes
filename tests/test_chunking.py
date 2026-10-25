import pytest

from l2n.chunking import chunk_segments
from l2n.errors import ConfigError
from l2n.models import Segment
from l2n.timeutil import format_timestamp, parse_timestamp


def make_segments(n_segments, words_each=20):
    segs, w = [], 0
    for i in range(n_segments):
        text = " ".join(f"w{w + k}" for k in range(words_each))
        w += words_each
        segs.append(Segment(i * 10.0, i * 10.0 + 9.0, text))
    return segs


def test_short_transcript_single_chunk():
    chunks = chunk_segments(make_segments(3), 1200, 100)
    assert len(chunks) == 1
    assert chunks[0].start == 0.0 and chunks[0].end == 29.0


def test_empty_transcript():
    assert chunk_segments([], 1200, 100) == []
    assert chunk_segments([Segment(0, 1, "   ")], 1200, 100) == []


def test_long_transcript_overlaps_and_covers_everything():
    segs = make_segments(200)  # 4000 words
    chunks = chunk_segments(segs, 1200, 100)
    assert len(chunks) >= 3
    assert all(c.word_count <= 1200 for c in chunks)
    seen = [s for c in chunks for s in c.segments]
    assert {s.text for s in seen} == {s.text for s in segs}  # nothing skipped
    for a, b in zip(chunks, chunks[1:]):
        shared = {s.text for s in a.segments} & {s.text for s in b.segments}
        overlap_words = sum(len(t.split()) for t in shared)
        assert 0 < overlap_words <= 100
        assert b.segments[0].start > a.segments[0].start  # forward progress
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_order_preserved_and_no_large_duplication():
    segs = make_segments(100)
    chunks = chunk_segments(segs, 300, 40)
    flat = [s.start for c in chunks for s in c.segments]
    total = sum(len(s.text.split()) for s in segs)
    assert sum(c.word_count for c in chunks) < total * 1.4
    assert flat[0] == 0.0 and max(flat) == segs[-1].start


def test_oversized_single_segment_makes_progress_and_keeps_text():
    words = [f"x{i}" for i in range(500)]
    seg = Segment(5.0, 95.0, " ".join(words))
    chunks = chunk_segments([seg], 200, 20)
    assert len(chunks) >= 3
    assert all(c.start == 5.0 and c.end == 95.0 for c in chunks)  # no invented timestamps
    got = set()
    for c in chunks:
        got |= set(c.plain_text.split())
    assert got == set(words)


def test_invalid_parameters():
    with pytest.raises(ConfigError):
        chunk_segments(make_segments(2), 100, 100)
    with pytest.raises(ConfigError):
        chunk_segments(make_segments(2), 100, -1)


def test_timestamp_formatting():
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(12.9) == "00:00:12"
    assert format_timestamp(3725) == "01:02:05"
    assert format_timestamp(-5) == "00:00:00"
    assert format_timestamp(None) == "00:00:00"
    assert format_timestamp(float("nan")) == "00:00:00"
    assert parse_timestamp("01:02:05") == 3725
    assert parse_timestamp("1:75:00") is None
    assert parse_timestamp("nonsense") is None
