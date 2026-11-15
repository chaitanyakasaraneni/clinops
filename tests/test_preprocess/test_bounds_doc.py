"""The bounds table in the preprocess guide must match the bounds in the code."""

from __future__ import annotations

import re
from pathlib import Path

from clinops.preprocess.outliers import LAB_BOUNDS, VITAL_BOUNDS

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "guide" / "preprocess.md"
ROW = re.compile(
    r"^\| `(\w+)` \| (vital|lab) \| ([^|]+) \| ([^|]+) \| ([^|]*) \| ([^|]*) \|$", re.M
)


def test_guide_lists_every_bound_exactly():
    rows = [
        (name, group, float(low), float(high), unit.strip(), note.strip())
        for name, group, low, high, unit, note in ROW.findall(GUIDE.read_text())
    ]
    expected = [
        (k, group, v.low, v.high, v.unit, v.description)
        for group, bounds in (("vital", VITAL_BOUNDS), ("lab", LAB_BOUNDS))
        for k, v in bounds.items()
    ]
    # Same rows, in the same order: no stale values, units, or notes; no duplicates.
    assert rows == expected
