"""Comparator A: sequence alignment with the literature feature representation.

The manuscript's Comparator A subsection still reports numbers computed from the
old stride-3 hand source. This rebuilds them.

Definition, now stated explicitly so it can be reproduced:

  hands   five fingertips in 3D relative to the wrist, 15 dimensions, one
          channel per hand, each hand warped against the corresponding hand of
          the other recording and the two channel distances averaged per pair
  head    yaw, pitch, roll, 3 dimensions

Normalisation and subsampling are identical to Comparator B (pooled robust
scaling over the domain, stride-5 frames), so the two comparators differ only in
which features they are given. That is what the comparison is meant to isolate.

WRITES  output/paper_review/final/comparator_a_{units}.json
Usage:  python pipeline_comparator_a.py [--units fw|px]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from gmm_pure_fullbody import maincam_head  # noqa: E402
from pipeline import (DEST, PKEYS, SUBSAMPLE, VIEWS, average_rank,  # noqa: E402
                      consensus, face_ref, hand_rows, vids)
from review2_raw_scores import dtw_scores_single  # noqa: E402
from table_all_methods import solo_head_new  # noqa: E402

TIPS = [4, 8, 12, 16, 20]
PERSON = [("speakerB", "SpeakerB"), ("speakerA", "SpeakerA")]
TARGET = "(4,5)"


def tips15(view, vid, person, hand):
    """Five fingertips in 3-D relative to the wrist. No face normalisation."""
    lm_path = VIEWS[view]["lm"] / f"{vid}_{person}_hand_landmarks.csv"
    if not lm_path.exists():
        return None
    rows = list(csv.DictReader(open(lm_path)))
    if len(rows) < 30:
        return None
    ref = face_ref(view, vid, person)
    side = hand_rows(rows, hand)
    if len(side) < 3:
        return None
    out = []
    for r in side:
        if int(r["frame"]) not in ref:
            continue
        w = np.array([float(r["lm0_x"]), float(r["lm0_y"]),
                      float(r["lm0_z"])])
        out.append(np.concatenate([
            np.array([float(r[f"lm{t}_x"]), float(r[f"lm{t}_y"]),
                      float(r[f"lm{t}_z"])]) - w for t in TIPS]))
    return np.array(out) if out else None


def angles3(heads, person):
    return {v: f[:, 0:3] for v, f in heads(person).items() if len(f) > 50}


def domain_scores(view, heads):
    """Per-domain distances, hands averaged across the two hand channels."""
    out = {}
    for person, name in PERSON:
        acc = {}
        for hand in ("left", "right"):
            seqs = {v: f for v in vids
                    if (f := tips15(view, v, person, hand)) is not None
                    and len(f) > 50}
            s = dtw_scores_single(seqs)
            if s is None:
                continue
            for k, val in s.items():
                acc.setdefault(k, []).append(val)
        out[f"Hand_{name}"] = {k: float(np.mean(v)) for k, v in acc.items()}
        out[f"Head_{name}"] = dtw_scores_single(angles3(heads, person))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", choices=["fw", "px"], default="fw")
    args = ap.parse_args()
    out = {"units": args.units, "comparator": "A-literature-features"}
    print(f"Comparator A, units={args.units}")

    for view, heads, label in [("maincam", maincam_head, "side"),
                               ("fullbody", solo_head_new, "frontal")]:
        dom = domain_scores(view, heads)
        # distances: lower is better, so invert before consensus ranking
        inv = {d: {k: -v for k, v in s.items()} for d, s in dom.items()}
        _, C = consensus({d: (s, 0, 0, None) for d, s in inv.items()})
        ranks = {d: average_rank(inv[d], TARGET) for d in inv}
        r = 1 + sum(1 for v in C.values() if v > C[TARGET] + 1e-12)
        out[label] = {"ranks": {k: float(v) for k, v in ranks.items()},
                      "C": {k: float(v) for k, v in C.items()},
                      "rank_of_target": int(r)}
        print(f"  {label:<8} ranks " + " ".join(
            f"{d.split('_')[0][0]}{d.split('_')[1][0]}={ranks[d]:.0f}"
            for d in sorted(ranks))
            + f"   C(4,5)={C[TARGET]:.3f}  consensus rank {r}/15")

    p = DEST / f"comparator_a_{args.units}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
