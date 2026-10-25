import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from l2n.config import Config  # noqa: E402
from l2n.errors import OllamaTimeoutError  # noqa: E402
from l2n.models import Segment, TranscriptResult  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "sample_transcript.json"

ACTION_JSON = {"action_items": [{
    "type": "assignment", "description": "Submit the lab report on binary search trees",
    "deadline_text": "next Friday", "deadline_date": "2026-10-30", "source_timestamp": "00:01:20",
    "evidence": "The lecturer says the lab report must be submitted next Friday.",
    "confidence": "high", "needs_verification": False}]}


def fake_response(prompt: str, json_mode: bool) -> str:
    """Deterministic stand-in for the LLM, keyed on the TASK marker in prompts."""
    if "TASK: EXTRACT_ACTIONS" in prompt:
        return json.dumps(ACTION_JSON if "lab report" in prompt else {"action_items": []})
    if "TASK: SUMMARIZE_CHUNK" in prompt:
        return "## Concepts\n- Binary search tree: smaller left, larger right"
    if "TASK: MERGE_INTERMEDIATE" in prompt:
        return "## Merged\n- Binary search tree notes"
    if "TASK: MERGE_FINAL" in prompt:
        return ("# Binary Search Trees\n\n## Lecture Overview\nBST basics.\n\n"
                "## Important Formulas\n- Search cost is O(h)\n\n"
                "## Action Items and Deadlines\n- hallucinated item that must be stripped\n")
    return "unexpected prompt"


class FakeLLM:
    def __init__(self, fail_when=None, json_override=None, available_error=None):
        self.calls = []
        self.fail_when = fail_when
        self.json_override = json_override
        self.available_error = available_error

    def check_available(self):
        if self.available_error:
            raise self.available_error

    def generate(self, prompt, system=None, json_mode=False):
        self.calls.append(prompt)
        if self.fail_when and self.fail_when(prompt):
            raise OllamaTimeoutError("simulated timeout")
        if json_mode and self.json_override is not None:
            return self.json_override
        return fake_response(prompt, json_mode)


@pytest.fixture
def transcript():
    d = json.loads(FIXTURE.read_text())
    return TranscriptResult([Segment.from_dict(s) for s in d["segments"]], "en", d["duration"])


@pytest.fixture
def cfg(tmp_path):
    return Config(output_dir=tmp_path / "out", chunk_size_words=50, chunk_overlap_words=5,
                  max_retries=1, json_attempts=2)
