"""Chunk summarisation and (hierarchical) merging of summaries."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .errors import Lecture2NotesError
from .models import Chunk

log = logging.getLogger("l2n.summarize")


class LLM(Protocol):
    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str: ...


SYSTEM_PROMPT = (
    "You are a careful teaching assistant who writes exam-oriented study notes from lecture "
    "transcripts. Use ONLY information stated in the transcript. Never invent or import "
    "definitions, formulas, examples, assignments or deadlines. If the transcript does not "
    "contain enough information, say so explicitly. Speech recognition errors are possible; "
    "flag unclear terms instead of guessing."
)

CHUNK_PROMPT = """TASK: SUMMARIZE_CHUNK
This is chunk {n} of {total} of a lecture transcript (time range {rng}).

Write concise study notes in Markdown (headings and nested bullet points) covering, when present:
- Main concepts and definitions
- Important explanations and reasoning
- Formulas and equations (copy them exactly as stated)
- Examples given by the lecturer
- Algorithms or procedures, with their steps in order
- Technical terminology
- Important distinctions between related concepts
- Warnings, exceptions and common mistakes
- Assignments, announcements or deadlines mentioned (quote the lecturer's wording)

Rules:
- Every statement must be supported by the transcript below. Do not add outside knowledge.
- Keep important technical detail; do not oversimplify.
- Mark statements that are ambiguous or uncertain with "(unclear in recording)".
- Omit categories that do not appear. If the chunk has too little content, say so.

TRANSCRIPT:
{text}
"""

STRUCTURE = """# <Lecture Title>
## Lecture Overview
## Learning Objectives
## Detailed Notes
## Key Definitions
## Important Formulas
## Algorithms and Procedures
## Examples Explained
## Important Distinctions
## Common Mistakes and Exam Traps
## Summary for Quick Revision
## Revision Questions
## Uncertainties and Items to Verify"""

MERGE_RULES = """- Remove redundancy but keep every important detail, example, formula and algorithm step from every part.
- Order concepts logically and group related topics under meaningful headings.
- Keep contradictions or ambiguities from the parts; do not silently pick one interpretation.
- Add NO knowledge that is not in the parts below. Never invent anything.
- Overlapping chunks were used, so the same content may appear twice; merge it once.
- Do not write an action-items/deadlines section; deadlines are handled separately."""

MERGE_FINAL_PROMPT = """TASK: MERGE_FINAL
Merge the partial lecture notes below into ONE coherent Markdown study document.

Use this structure, but INCLUDE ONLY sections that have real content (no empty sections):
{structure}

Title: infer a descriptive lecture title from the content; if impossible use a generic descriptive title.
Revision Questions: write questions based only on what the lecture actually covered.
{rules}

PARTIAL NOTES:
{parts}
"""

MERGE_INTERMEDIATE_PROMPT = """TASK: MERGE_INTERMEDIATE
Combine the partial lecture notes below into one set of notes (Markdown headings and bullets).
This is an intermediate step: be COMPREHENSIVE and keep all technical detail.
{rules}

PARTIAL NOTES:
{parts}
"""


@dataclass
class ChunkSummary:
    index: int
    time_range: str
    text: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.text is not None


def summarize_chunk(client: LLM, chunk: Chunk, total: int) -> str:
    prompt = CHUNK_PROMPT.format(
        n=chunk.index + 1, total=total, rng=chunk.time_range, text=chunk.timestamped_text
    )
    return client.generate(prompt, system=SYSTEM_PROMPT)


def summarize_chunks(
    client: LLM, chunks: list[Chunk], progress: Callable[[str], None] | None = None
) -> list[ChunkSummary]:
    """Summarise sequentially. A failing chunk is recorded and skipped (never
    fabricated); the caller decides how to flag the result as partial."""
    results: list[ChunkSummary] = []
    for chunk in chunks:
        label = f"chunk {chunk.index + 1}/{len(chunks)} ({chunk.time_range})"
        if progress:
            progress(f"  Summarising {label} ...")
        try:
            text = summarize_chunk(client, chunk, len(chunks))
            results.append(ChunkSummary(chunk.index, chunk.time_range, text))
        except Lecture2NotesError as exc:
            log.error("Summary failed for %s: %s", label, exc)
            if progress:
                progress(f"  ! Summary FAILED for {label}: {exc}")
            results.append(ChunkSummary(chunk.index, chunk.time_range, None, str(exc)))
    return results


@dataclass
class MergeResult:
    text: str
    merged_by_llm: bool
    warnings: list[str] = field(default_factory=list)


def _label(i: int, rng: str, text: str) -> str:
    return f"### PART {i} (lecture time {rng})\n{text.strip()}"


def _join(parts: list[str]) -> str:
    return "\n\n".join(parts)


def _group_by_size(parts: list[str], max_chars: int) -> list[list[str]]:
    groups: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for p in parts:
        if cur and size + len(p) > max_chars:
            groups.append(cur)
            cur, size = [], 0
        cur.append(p)
        size += len(p)
    if cur:
        groups.append(cur)
    if len(groups) >= len(parts) and len(parts) > 1:  # no reduction: force pairs
        groups = [parts[i : i + 2] for i in range(0, len(parts), 2)]
    return groups


def strip_section(markdown: str, heading: str) -> str:
    """Remove a '## heading' section (used to drop LLM-written action items)."""
    pattern = re.compile(rf"^##\s+{re.escape(heading)}\s*$.*?(?=^##\s|\Z)", re.S | re.M | re.I)
    return pattern.sub("", markdown).rstrip() + "\n"


def merge_summaries(
    client: LLM,
    summaries: list[ChunkSummary],
    *,
    max_chars: int = 14000,
    max_levels: int = 6,
    progress: Callable[[str], None] | None = None,
) -> MergeResult:
    """Merge successful chunk summaries. If everything fits in `max_chars`
    one request is made; otherwise summaries are merged in groups,
    level by level (hierarchical merge), then a final merge writes the
    structured document. Failures fall back to the unmerged summaries."""
    good = [s for s in summaries if s.ok]
    if not good:
        return MergeResult("", False, ["No chunk summaries available to merge."])
    parts = [_label(i + 1, s.time_range, s.text or "") for i, s in enumerate(good)]
    warnings: list[str] = []
    say = progress or (lambda _m: None)

    level = 0
    while len(_join(parts)) > max_chars and len(parts) > 1 and level < max_levels:
        level += 1
        groups = _group_by_size(parts, max_chars)
        say(f"  Merge level {level}: {len(parts)} parts -> {len(groups)} groups")
        new_parts: list[str] = []
        for gi, group in enumerate(groups, 1):
            if len(group) == 1:
                new_parts.append(group[0])
                continue
            prompt = MERGE_INTERMEDIATE_PROMPT.format(rules=MERGE_RULES, parts=_join(group))
            try:
                merged = client.generate(prompt, system=SYSTEM_PROMPT)
                new_parts.append(_label(gi, f"merge level {level}", merged))
            except Lecture2NotesError as exc:
                log.error("Intermediate merge failed (level %d group %d): %s", level, gi, exc)
                warnings.append(f"Intermediate merge failed (level {level}, group {gi}); its notes were kept unmerged.")
                new_parts.extend(group)
        parts = new_parts
    if len(_join(parts)) > max_chars and len(parts) > 1:
        warnings.append(
            "Summaries still exceed the merge size limit after hierarchical merging; the final request may be truncated by the model's context window."
        )

    say("  Final merge ...")
    prompt = MERGE_FINAL_PROMPT.format(structure=STRUCTURE, rules=MERGE_RULES, parts=_join(parts))
    try:
        text = client.generate(prompt, system=SYSTEM_PROMPT)
        return MergeResult(strip_section(text, "Action Items and Deadlines"), True, warnings)
    except Lecture2NotesError as exc:
        log.error("Final merge failed: %s", exc)
        warnings.append(f"Final merge failed ({exc}); showing unmerged partial notes instead.")
        return MergeResult("## Detailed Notes (unmerged)\n\n" + _join(parts) + "\n", False, warnings)
