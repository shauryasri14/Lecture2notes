import json

from l2n.actions import ActionItem
from l2n.outputs import (build_notes_markdown, render_action_items_markdown, safe_filename,
                         unique_output_dir, write_action_items, write_transcript_files)
from l2n.models import Segment
from l2n.summarize import strip_section


def item(**kw):
    d = dict(type="assignment", description="Submit lab report", deadline_text="next Friday",
             deadline_date=None, source_timestamp="00:01:20", evidence="must be submitted next Friday",
             confidence="high", needs_verification=True)
    d.update(kw)
    return ActionItem(**d)


def test_safe_filename():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("My Lecture: Week 3?.mp3") == "My_Lecture_Week_3_.mp3"
    assert safe_filename("CON") == "lecture_CON"
    assert safe_filename("...") == "lecture"
    assert len(safe_filename("a" * 500)) <= 80


def test_unique_output_dir_never_overwrites(tmp_path):
    a = unique_output_dir(tmp_path, "lec")
    b = unique_output_dir(tmp_path, "lec")
    assert a != b and a.is_dir() and b.is_dir() and b.name == "lec_2"
    assert a.parent == tmp_path


def test_transcript_and_action_files(tmp_path):
    segs = [Segment(12, 25, "Hello there."), Segment(25, 42, "Second line.")]
    write_transcript_files(tmp_path, segs, "en", 42.0)
    txt = (tmp_path / "transcript.txt").read_text()
    assert txt.startswith("[00:00:12 - 00:00:25] Hello there.\n\n[00:00:25 - 00:00:42] Second line.")
    assert json.loads((tmp_path / "transcript.json").read_text())["segments"][0]["start"] == 12
    write_action_items(tmp_path, [item()], complete=True, failures=[], lecture_date=None)
    data = json.loads((tmp_path / "action_items.json").read_text())
    assert data["action_items"][0]["deadline_date"] is None


def test_notes_markdown_has_title_actions_and_warnings():
    md = build_notes_markdown("## Overview\ntext", [item()], title_fallback="Lecture Notes: x",
                              warnings=["Chunk 2 failed"], extraction_complete=True)
    assert md.startswith("# Lecture Notes: x")
    assert "INCOMPLETE" in md and "Chunk 2 failed" in md
    assert "## Action Items and Deadlines" in md and "next Friday" in md and "verify" in md


def test_empty_action_list_and_failed_extraction_wording():
    assert "No assignments" in render_action_items_markdown([])
    assert "extraction failed" in render_action_items_markdown([], extraction_complete=False)


def test_strip_section_removes_llm_action_items_only():
    md = "# T\n\n## A\nx\n\n## Action Items and Deadlines\n- fake\n\n## B\ny\n"
    out = strip_section(md, "Action Items and Deadlines")
    assert "fake" not in out and "## A" in out and "## B" in out
