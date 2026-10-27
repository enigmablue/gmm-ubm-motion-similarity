"""Path configuration for the analysis scripts.

The scripts in ``code/`` were written on a single workstation and most of them
contain an absolute path constant near the top (run ``python check_paths.py``
for the current list). This module is the intended replacement for those
constants; the scripts have not yet been refactored to use it.

Set either of these environment variables before running anything:

    GMMUBM_DATA_ROOT     root of the extracted landmark / feature outputs
                         default: ./output
    GMMUBM_RESULT_ROOT   where result JSON files are written
                         default: ./output/paper_review/final

Example
-------
    set GMMUBM_DATA_ROOT=D:\\analysis\\output
    python code/pipeline.py
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get("GMMUBM_DATA_ROOT", "output")
).expanduser().resolve()

RESULT_ROOT = Path(
    os.environ.get("GMMUBM_RESULT_ROOT", "output/paper_review/final")
).expanduser().resolve()

# Subdirectories the pipeline expects inside DATA_ROOT.
EXPECTED_SUBDIRS = (
    "maincam",              # per-video head features, side view
    "maincam_hands_std",    # re-extracted side-view hand landmarks
    "solo_head_std",        # frontal head features
    "fullbody_std",         # frontal hand landmarks and full-body features
)


def report() -> list[str]:
    """Return one line per expected path, saying whether it exists."""
    lines = [f"DATA_ROOT   = {DATA_ROOT}", f"RESULT_ROOT = {RESULT_ROOT}", ""]
    for name in EXPECTED_SUBDIRS:
        p = DATA_ROOT / name
        lines.append(f"  [{'ok' if p.is_dir() else 'missing'}] {p}")
    return lines


if __name__ == "__main__":
    print("\n".join(report()))
