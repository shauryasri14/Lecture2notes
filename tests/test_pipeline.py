import json

import pytest
import requests

from conftest import FakeLLM
from l2n.chunking import chunk_segments
from l2n.errors import (AudioError, OllamaModelMissingError, OllamaTimeoutError,
                        OllamaUnavailableError, TranscriptionError)
from l2n.models import Segment, TranscriptResult
from l2n.ollama_client import OllamaClient
from l2n.pipeline import COMPLETE, FAILED, PARTIAL, run_pipeline
from l2n.summarize import ChunkSummary, merge_summaries
from l2n.transcribe import Transcriber, validate_audio_file

quiet = lambda m: None  # noqa: E731


def read(out, name):
    return (out / name).read_text(encoding="utf-8")


def test_full_pipeline_success_writes_all_outputs(cfg, transcript):
    res = run_pipeline("sample.mp3", cfg, client=FakeLLM(), transcript=transcript, progress=quiet)
    assert res.status == COMPLETE, res.warnings
    out = res.output_dir
    for name in ["notes.md", "transcript.txt", "transcript.json", "action_items.json", "processing.log"]:
        assert (out / name).is_file(), name
    notes = read(out, "notes.md")
    assert notes.startswith("# Binary Search Trees")
    assert "hallucinated item" not in notes  # LLM-written action section stripped
    assert "next Friday" in notes and "00:01:20" in notes
    items = json.loads(read(out, "action_items.json"))["action_items"]
    assert len(items) == 1                      # dedup across overlapping chunks
    assert items[0]["deadline_date"] is None    # no lecture date -> never computed
    assert items[0]["needs_verification"] is True


def test_partial_failure_is_reported_not_hidden(cfg, transcript):
    llm = FakeLLM(fail_when=lambda p: "TASK: SUMMARIZE_CHUNK" in p and "chunk 2 of" in p)
    res = run_pipeline("sample.mp3", cfg, client=llm, transcript=transcript, progress=quiet)
    assert res.status == PARTIAL
    notes = read(res.output_dir, "notes.md")
    assert "INCOMPLETE" in notes and "Chunk 2" in notes
    assert json.loads(read(res.output_dir, "chunk_summaries.json"))[1]["summary"] is None
    assert json.loads(read(res.output_dir, "status.json"))["status"] == PARTIAL


def test_all_summaries_failing_gives_failed_status_but_keeps_transcript(cfg, transcript):
    llm = FakeLLM(fail_when=lambda p: "TASK: SUMMARIZE_CHUNK" in p)
    res = run_pipeline("sample.mp3", cfg, client=llm, transcript=transcript, progress=quiet)
    assert res.status == FAILED and "All chunk summaries failed" in res.error
    assert (res.output_dir / "transcript.txt").is_file()
    assert not (res.output_dir / "notes.md").exists()


def test_ollama_unavailable_fails_fast_before_work(cfg, transcript):
    llm = FakeLLM(available_error=OllamaUnavailableError("down"))
    res = run_pipeline("sample.mp3", cfg, client=llm, transcript=transcript, progress=quiet)
    assert res.status == FAILED and "down" in res.error and res.output_dir is None
    assert llm.calls == []


def test_missing_model_reported(cfg, transcript):
    llm = FakeLLM(available_error=OllamaModelMissingError("ollama pull gemma3:4b"))
    res = run_pipeline("sample.mp3", cfg, client=llm, transcript=transcript, progress=quiet)
    assert res.status == FAILED and "ollama pull" in res.error


def test_action_extraction_failure_marks_partial(cfg, transcript):
    llm = FakeLLM(json_override="garbage")
    res = run_pipeline("sample.mp3", cfg, client=llm, transcript=transcript, progress=quiet)
    assert res.status == PARTIAL
    data = json.loads(read(res.output_dir, "action_items.json"))
    assert data["action_items"] == [] and data["extraction_complete"] is False


def test_two_runs_do_not_overwrite(cfg, transcript):
    a = run_pipeline("same.mp3", cfg, client=FakeLLM(), transcript=transcript, progress=quiet)
    b = run_pipeline("same.mp3", cfg, client=FakeLLM(), transcript=transcript, progress=quiet)
    assert a.output_dir != b.output_dir


def test_keyboard_interrupt_saves_status(cfg, transcript):
    class Boom(FakeLLM):
        def generate(self, *a, **k):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pipeline("x.mp3", cfg, client=Boom(), transcript=transcript, progress=quiet)
    out = next(cfg.output_dir.iterdir())
    assert json.loads(read(out, "status.json"))["status"] == "interrupted"


def test_hierarchical_merge_used_for_large_input():
    llm = FakeLLM()
    sums = [ChunkSummary(i, "00:00:00 - 00:01:00", "x" * 400) for i in range(10)]
    res = merge_summaries(llm, sums, max_chars=1000)
    assert res.merged_by_llm and res.text.startswith("# Binary Search Trees")
    assert sum("TASK: MERGE_INTERMEDIATE" in c for c in llm.calls) >= 2
    assert sum("TASK: MERGE_FINAL" in c for c in llm.calls) == 1


def test_final_merge_failure_falls_back_to_unmerged():
    llm = FakeLLM(fail_when=lambda p: "TASK: MERGE_FINAL" in p)
    res = merge_summaries(llm, [ChunkSummary(0, "r", "- point")], max_chars=1000)
    assert not res.merged_by_llm and "- point" in res.text and res.warnings


# ---------------------------------------------------------------- Ollama client
class FakeResp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body if body is not None else {}

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, post_effects=(), get_effect=None):
        self.post_effects, self.get_effect, self.posts = list(post_effects), get_effect, 0

    def post(self, url, json=None, timeout=None):
        self.posts += 1
        eff = self.post_effects.pop(0)
        if isinstance(eff, Exception):
            raise eff
        return eff

    def get(self, url, timeout=None):
        if isinstance(self.get_effect, Exception):
            raise self.get_effect
        return self.get_effect


def client(session, retries=2):
    return OllamaClient("http://localhost:11434", "gemma3:4b", max_retries=retries,
                        session=session, sleep=lambda s: None)


def test_unreachable_server_message():
    c = client(FakeSession(get_effect=requests.ConnectionError()))
    with pytest.raises(OllamaUnavailableError, match="ollama serve"):
        c.check_available()


def test_missing_model_detected():
    c = client(FakeSession(get_effect=FakeResp(200, {"models": [{"name": "llama3:8b"}]})))
    with pytest.raises(OllamaModelMissingError, match="ollama pull gemma3:4b"):
        c.check_available()
    ok = client(FakeSession(get_effect=FakeResp(200, {"models": [{"name": "gemma3:4b"}]})))
    ok.check_available()


def test_retry_limit_is_bounded_on_timeouts():
    s = FakeSession([requests.Timeout()] * 10)
    with pytest.raises(OllamaTimeoutError):
        client(s, retries=2).generate("hi")
    assert s.posts == 3  # first try + 2 retries


def test_recovers_after_transient_failure():
    s = FakeSession([requests.ConnectionError(), FakeResp(500), FakeResp(200, {"response": " ok "})])
    assert client(s, retries=3).generate("hi") == "ok"
    assert s.posts == 3


def test_no_retry_for_missing_model_404():
    s = FakeSession([FakeResp(404)] * 5)
    with pytest.raises(OllamaModelMissingError):
        client(s).generate("hi")
    assert s.posts == 1


def test_empty_response_retried_then_error():
    s = FakeSession([FakeResp(200, {"response": ""})] * 5)
    with pytest.raises(Exception, match="empty"):
        client(s, retries=1).generate("hi")
    assert s.posts == 2


# ---------------------------------------------------------------- audio / transcriber
def test_audio_validation_errors(tmp_path):
    with pytest.raises(AudioError, match="not found"):
        validate_audio_file(tmp_path / "nope.mp3")
    empty = tmp_path / "e.mp3"
    empty.write_bytes(b"")
    with pytest.raises(AudioError, match="empty"):
        validate_audio_file(empty)
    bad = tmp_path / "x.txt"
    bad.write_text("hi")
    with pytest.raises(AudioError, match="Unsupported"):
        validate_audio_file(bad)


class FakeWhisper:
    def __init__(self, segs, boom=None):
        self.segs, self.boom = segs, boom

    def transcribe(self, path, **kw):
        if self.boom:
            raise self.boom

        class Info:
            duration, language = 10.0, "en"

        return iter(self.segs), Info()


class Seg:
    def __init__(self, s, e, t):
        self.start, self.end, self.text = s, e, t


def test_transcriber_keeps_timestamps_and_loads_model_once(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"RIFFxxxx")
    loads = []

    def factory(*a, **k):
        loads.append(1)
        return FakeWhisper([Seg(0.5, 4.0, " Hello "), Seg(4.0, 6.0, "  ")])

    t = Transcriber("base", model_factory=factory)
    r1 = t.transcribe(f)
    t._model.segs = [Seg(1, 2, "again")]
    t.transcribe(f)
    assert len(loads) == 1
    assert [(s.start, s.end, s.text) for s in r1.segments] == [(0.5, 4.0, "Hello")]


def test_transcriber_error_paths(tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"junk")
    with pytest.raises(AudioError, match="Could not decode"):
        Transcriber("base", model_factory=lambda *a, **k: FakeWhisper([], boom=RuntimeError("bad data"))).transcribe(f)
    with pytest.raises(TranscriptionError, match="no usable text"):
        Transcriber("base", model_factory=lambda *a, **k: FakeWhisper([Seg(0, 1, " ")])).transcribe(f)

    def broken(*a, **k):
        raise OSError("download failed")

    with pytest.raises(TranscriptionError, match="Could not load Whisper model"):
        Transcriber("base", model_factory=broken).transcribe(f)


def test_pyav_mismatch_falls_back_to_ffmpeg_decoder(tmp_path, monkeypatch):
    import numpy as np
    import l2n.transcribe as tr

    f = tmp_path / "a.mp4"
    f.write_bytes(b"junk")
    calls = []

    class Mismatch(FakeWhisper):
        def transcribe(self, path, **kw):
            calls.append(type(path).__name__)
            if isinstance(path, str):
                raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")
            return super().transcribe(path, **kw)

    monkeypatch.setattr(tr, "decode_with_ffmpeg", lambda p: np.zeros(16000, dtype=np.float32))
    t = Transcriber("base", model_factory=lambda *a, **k: Mismatch([Seg(0, 2, "hello")]))
    assert t.transcribe(f).segments[0].text == "hello"
    assert calls == ["str", "ndarray"]


def test_missing_cuda_libraries_fall_back_to_cpu(tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"junk")
    devices = []

    def factory(name, device, compute_type):
        devices.append((device, compute_type))
        if device != "cpu":
            return FakeWhisper([], boom=RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"))
        return FakeWhisper([Seg(0, 2, "hi")])

    t = Transcriber("base", device="auto", model_factory=factory)
    assert t.transcribe(f).segments[0].text == "hi"
    assert devices == [("auto", "int8"), ("cpu", "int8")]
