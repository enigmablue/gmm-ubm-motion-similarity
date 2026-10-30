"""Two-channel hand features: remove cross-hand contamination from the pipeline.

THE DEFECT
----------
`handview_2x2._one_hand_per_frame` keeps one hand per frame, preferring "right" and
falling back to "left". Fullbody recordings contain both hands in most frames, so the
selected hand flips 62-194 times per recording. When it flips, the velocity and
acceleration features are computed between two *different physical hands*, so the
displacement is the distance between left and right rather than one hand's motion.

This contaminates exactly the dynamic features the framework models, and the
fullbody result is hand-driven (hands-only rank 1/15, head-only 5/15).

THE FIX
-------
Treat left and right as separate channels and derive features per channel, so a
derivative is only ever taken within one hand's own track. Frame-level features are
then pooled across the two hands for the distributional comparison.

This keeps every other feature definition identical (same 12 dimensions, same
face-width normalisation, same subsampling), so any change in the result is
attributable to the contamination fix.

Two variants:
    span   derivatives continue across gaps within the same hand (the default)
    strict derivatives are zero unless the previous sample was the immediately
           preceding analysed frame for that same hand

Scope: GMM-UBM only. The aggregate baseline uses curl percentiles with no
derivatives, so it is unaffected by hand switching. DTW compares sequences, and
pooling two interleaved hands into one sequence would be wrong, so it needs a
separate design and is not reported here.

Usage:
    python hand_two_channel.py
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from cell_extractor import run_gmm_domain  # noqa: E402
from handview_2x2 import (SUBSAMPLE, hand_features, load_hand_raw,  # noqa: E402
                          _one_hand_per_frame)
from gmm_pure_fullbody import maincam_head, solo_head, fullbody_hands  # noqa: E402
from table_all_methods import (MAINCAM_STD, solo_head_new,  # noqa: E402
                               _strkeys, rank_consensus)

vids = ["1", "2", "3", "4", "5", "6"]
PEOPLE = [("stella", "Stella"), ("aru", "Aru")]


def two_channel(rows, fr, mode="span"):
    """Pool per-hand features, with derivatives taken within each hand only."""
    by_hand = {"left": [], "right": []}
    for r in rows:
        h = (r.get("handedness") or "").strip().lower()
        if h in by_hand:
            by_hand[h].append(r)

    parts = []
    for h in ["left", "right"]:
        side = sorted(by_hand[h], key=lambda r: int(r["frame"]))
        if len(side) < 3:
            continue
        f = hand_features(side, fr, min_speed=0)
        if f is None:
            continue
        if mode == "strict":
            # zero the derivative columns unless the previous analysed frame for
            # this hand was the immediately preceding one
            keep_rows = [int(r["frame"]) for r in side if int(r["frame"]) % SUBSAMPLE == 0
                         and int(r["frame"]) in fr]
            # hand_features may skip frames without a face, so reconstruct from
            # the returned array length only when it matches
            if len(keep_rows) == len(f):
                bad = np.array([False] + [keep_rows[i] - keep_rows[i - 1] != SUBSAMPLE
                                          for i in range(1, len(keep_rows))])
                f[bad, 2:6] = 0.0          # vx, vy, speed, accel
        parts.append(f)

    if not parts:
        return None
    out = np.vstack(parts)
    return out if len(out) > 10 else None


def load_maincam_std_raw(person):
    """Maincam hands at the matched stride, WITHOUT collapsing to one hand."""
    import csv as _csv
    out = {}
    fp = "face1" if person == "stella" else "face2"
    feat = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output\maincam")
    for v in vids:
        lm = MAINCAM_STD / f"{v}_{person}_hand_landmarks.csv"
        if not lm.exists():
            continue
        fr = {}
        for r in _csv.DictReader(open(feat / f"{v}_features.csv")):
            if r.get(f"{fp}_detected") == "1":
                fr[int(r["frame"])] = (float(r[f"{fp}_x"]) + float(r[f"{fp}_w"]) / 2,
                                        float(r[f"{fp}_y"]) + float(r[f"{fp}_h"]) / 2,
                                        float(r[f"{fp}_w"]))
        rows = sorted(_csv.DictReader(open(lm)), key=lambda r: int(r["frame"]))
        out[v] = (rows, fr)
    return out


def cache_for(view, person):
    if view == "maincam":
        return load_maincam_std_raw(person)
    return load_hand_raw("fullbody", person, one_hand=False)


def build(cache_view, person, mode):
    """{vid: feature array} using two-channel extraction."""
    cache = cache_for(cache_view, person)
    out = {}
    for v in vids:
        if v not in cache:
            continue
        rows, fr = cache[v]
        f = two_channel(rows, fr, mode=mode)
        if f is not None:
            out[v] = f
    return out


def build_single(cache_view, person):
    """Baseline: the existing one-hand-per-frame rule."""
    cache = cache_for(cache_view, person)
    out = {}
    for v in vids:
        if v not in cache:
            continue
        rows, fr = cache[v]
        f = hand_features(_one_hand_per_frame(rows), fr, min_speed=0)
        if f is not None:
            out[v] = f
    return out


def cell(head, hand):
    """Run all four domains; return (domain_ranks, consensus rank, C)."""
    dr = {}
    for person, name in PEOPLE:
        dr[f"Hand_{name}"] = _strkeys(run_gmm_domain(hand[person]))
        dr[f"Head_{name}"] = _strkeys(run_gmm_domain(head[person]))
    rank, C, _, _ = rank_consensus(dr)
    return dr, rank, C


def frames_report(label, hands):
    print(f"\n{label}")
    print(f"   {'speaker':<9}{'frames per recording (1-6)':<40}{'total':>8}")
    for person, name in PEOPLE:
        counts = [len(hands[person][v]) for v in vids if v in hands[person]]
        print(f"   {name:<9}" + " ".join(f"{c:>5}" for c in counts)
              + f"{sum(counts):>8}")


def main():
    print("Building features...")
    head_mc = {p: maincam_head(p) for p, _ in PEOPLE}
    head_fb = {p: solo_head_new(p) for p, _ in PEOPLE}

    variants = {}
    for view in ["maincam", "fullbody"]:
        variants[(view, "single")] = {p: build_single(view, p) for p, _ in PEOPLE}
        variants[(view, "span")] = {p: build(view, p, "span") for p, _ in PEOPLE}
        variants[(view, "strict")] = {p: build(view, p, "strict") for p, _ in PEOPLE}

    print()
    print("=" * 78)
    print("HAND FRAME COUNTS")
    print("=" * 78)
    frames_report("maincam, one hand per frame (baseline)",
                  variants[("maincam", "single")])
    frames_report("maincam, two channels", variants[("maincam", "span")])
    frames_report("fullbody, one hand per frame (baseline)",
                  variants[("fullbody", "single")])
    frames_report("fullbody, two channels", variants[("fullbody", "span")])

    print()
    print("=" * 78)
    print("GMM-UBM CONSENSUS RANK OF (4,5), OUT OF 15")
    print("=" * 78)
    print(f"{'configuration':<34}{'Hand_St':>9}{'Hand_Ar':>9}{'rank':>8}{'C':>8}")
    print("-" * 78)

    for view, heads in [("maincam", head_mc), ("fullbody", head_fb)]:
        for mode, label in [("single", "one hand per frame (baseline)"),
                            ("span", "two channels, gaps spanned"),
                            ("strict", "two channels, gaps broken")]:
            hands = variants[(view, mode)]
            dr, rank, C = cell(heads, hands)
            hs = {n: dr[f"Hand_{n}"].get("(4,5)", 15) for n in ["Stella", "Aru"]}
            print(f"{view + ' / ' + label:<34}{hs['Stella']:>9}{hs['Aru']:>9}"
                  f"{rank:>6}/15{C:>8.3f}")
        print()

    print("Notes")
    print("  The baseline reproduces the numbers in review2 sections 4-5.")
    print("  If two-channel extraction changes the fullbody rank, the earlier result")
    print("  depended on derivatives that spanned two different hands.")


if __name__ == "__main__":
    main()
