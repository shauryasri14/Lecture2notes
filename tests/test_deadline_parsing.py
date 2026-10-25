import json
from datetime import date

from conftest import FakeLLM
from l2n.actions import (ActionValidationError, deduplicate_items, extract_chunk_actions,
                         parse_json_payload, validate_action_item)
from l2n.models import Chunk, Segment
import pytest

CHUNK = Chunk(0, [Segment(1400.0, 1460.0, "The lab report must be submitted next Friday. Also read chapter 4.")])


def raw(**kw):
    base = {"type": "assignment", "description": "Submit the lab report", "deadline_text": "next Friday",
            "deadline_date": None, "source_timestamp": "00:23:30",
            "evidence": "lab report must be submitted next Friday", "confidence": "high",
            "needs_verification": False}
    base.update(kw)
    return base


def test_valid_item_without_lecture_date_has_null_date_and_flag():
    item = validate_action_item(raw(deadline_date="2026-10-30"), chunk=CHUNK, lecture_date=None)
    assert item.deadline_date is None and item.needs_verification
    assert item.deadline_text == "next Friday"
    assert any("no reliable lecture date" in n for n in item.notes)


def test_date_kept_only_with_lecture_date_and_relative_wording_flagged():
    item = validate_action_item(raw(deadline_date="2026-10-30"), chunk=CHUNK, lecture_date=date(2026, 10, 22))
    assert item.deadline_date == "2026-10-30" and item.needs_verification


def test_invalid_date_discarded():
    item = validate_action_item(raw(deadline_date="31/02/2026"), chunk=CHUNK, lecture_date=date(2026, 1, 1))
    assert item.deadline_date is None and item.needs_verification


def test_missing_evidence_or_description_rejected():
    with pytest.raises(ActionValidationError):
        validate_action_item(raw(evidence=""), chunk=CHUNK)
    with pytest.raises(ActionValidationError):
        validate_action_item(raw(description=" "), chunk=CHUNK)
    with pytest.raises(ActionValidationError):
        validate_action_item("not a dict", chunk=CHUNK)


def test_bad_timestamp_and_unknown_type_and_confidence_sanitised():
    item = validate_action_item(raw(source_timestamp="soon", type="Pop Quiz!", confidence="certain"), chunk=CHUNK)
    assert item.source_timestamp == "00:23:20" and item.type == "other"
    assert item.confidence == "low" and item.needs_verification


def test_deadline_wording_not_in_transcript_downgrades():
    item = validate_action_item(raw(deadline_text="on March 3rd"), chunk=CHUNK)
    assert item.confidence == "low" and item.needs_verification


def test_deduplication_keeps_evidence():
    a = validate_action_item(raw(), chunk=CHUNK)
    b = validate_action_item(raw(source_timestamp="00:24:00", evidence="again: lab report next Friday"), chunk=CHUNK)
    c = validate_action_item(raw(type="exam", description="Midterm exam", deadline_text=None), chunk=CHUNK)
    merged = deduplicate_items([a, b, c])
    assert len(merged) == 2
    lab = next(m for m in merged if m.type == "assignment")
    assert lab.additional_mentions[0]["source_timestamp"] == "00:24:00"


def test_identical_overlap_duplicates_collapse_without_extra_mentions():
    a = validate_action_item(raw(), chunk=CHUNK)
    b = validate_action_item(raw(), chunk=CHUNK)
    merged = deduplicate_items([a, b])
    assert len(merged) == 1 and merged[0].additional_mentions == []


def test_parse_json_payload_variants():
    assert parse_json_payload('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_payload('Sure! {"a": 2} hope that helps') == {"a": 2}
    for bad in ["", "no json here", "{broken"]:
        with pytest.raises(ValueError):
            parse_json_payload(bad)


def test_malformed_json_retries_then_fails_safely():
    llm = FakeLLM(json_override="this is not json")
    res = extract_chunk_actions(llm, CHUNK, max_attempts=3)
    assert res.items == [] and "malformed JSON" in res.error
    assert len(llm.calls) == 3  # bounded


def test_malformed_then_valid_recovers():
    outputs = iter(["garbage", json.dumps({"action_items": [raw()]})])

    class Flaky(FakeLLM):
        def generate(self, prompt, system=None, json_mode=False):
            return next(outputs)

    res = extract_chunk_actions(Flaky(), CHUNK, max_attempts=3)
    assert res.error is None and len(res.items) == 1


def test_no_actions_returns_empty_list():
    res = extract_chunk_actions(FakeLLM(json_override='{"action_items": []}'), CHUNK)
    assert res.items == [] and res.error is None
