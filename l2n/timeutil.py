"""Timestamp helpers."""
from __future__ import annotations

import math
import re

_TS_RE = re.compile(r"^(\d+):([0-5]\d):([0-5]\d)$")


def format_timestamp(seconds: float | int | None) -> str:
    """Format seconds as HH:MM:SS (floored; negatives/NaN/None become 0)."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        seconds = 0
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_timestamp(text: str | None) -> float | None:
    """Parse HH:MM:SS into seconds. Returns None when the format is invalid."""
    if not isinstance(text, str):
        return None
    m = _TS_RE.match(text.strip())
    if not m:
        return None
    h, mi, s = (int(x) for x in m.groups())
    return float(h * 3600 + mi * 60 + s)
