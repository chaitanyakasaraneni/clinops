"""The bounds table in the preprocess guide must match the bounds in the code."""

from __future__ import annotations

import re
from pathlib import Path

from clinops.preprocess.outliers import LAB_BOUNDS, VITAL_BOUNDS

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "guide" / "preprocess.md"


def test_guide_lists_every_bound_exactly():
    rows = re.findall(
        r"^\| `(\w+)` \| (vital|lab) \| ([\d.eE+-]+) \| ([\d.eE+-]+) \|", GUIDE.read_text(), re.M
    )
    documented = {name: (group, float(low), float(high)) for name, group, low, high in rows}
    expected = {
        **{k: ("vital", v.low, v.high) for k, v in VITAL_BOUNDS.items()},
        **{k: ("lab", v.low, v.high) for k, v in LAB_BOUNDS.items()},
    }
    assert documented == expected
