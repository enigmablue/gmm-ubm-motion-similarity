"""Sequence-alignment comparator on the merged pipeline.

WHY
---
Table 5's DTW row and Table 6's DTW cells still carry September values computed
from the canonical hand builder. This module runs the representation-matched
comparator (Comparator B) on exactly the features the merged framework uses, so
the two rows differ in method and nothing else.

The DTW unit is the whole recording: one sequence per domain per recording,
cumulative distance divided by path length. Hands are the merged pooled 12D
vector, heads the same 18D vector, both already at the documented 6 Hz.

READS   same feature sources as pipeline.py
WRITES  output/paper_review/final/pipeline_dtw_{units}.json

Usage:  python pipeline_comparators.py [--units fw|px]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from gmm_pure_fullbody import maincam_head  # noqa: E402
from pipeline import (PKEYS, TARGET, average_rank, bh, build_hand_frames,  # noqa: E402
                      exact_rho, vids)
from review2_raw_scores import dtw_scores_single  # noqa: E402
from table_all_methods import solo_head_new  # noqa: E402

DEST = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output\paper_review\final")
PEOPLE = [("speakerB", "SpeakerB"), ("speakerA", "SpeakerA")]


def domain_dtw(view, units):
    out = {}
    for person, name in PEOPLE:
        hands = {}
        for v in vids:
            f = build_hand_frames(view, v, person, units)
            if f is not None and len(f) > 50:
                hands[v] = f
        out[f"Hand_{name}"] = dtw_scores_single(hands)
    heads = maincam_head if view == "maincam" else solo_head_new
    for person, name in PEOPLE:
        out[f"Head_{name}"] = dtw_scores_single(heads(person))
    return {d: s for d, s in out.items() if s}


def consensus(scores):
    """Lower DTW distance = more similar, so ranks run upwards from the smallest."""
    R = {}
    for d, s in scores.items():
        flipped = {p: -s[p] for p in PKEYS}
        R[d] = {p: 1.0 - (average_rank(flipped, p) - 1) / 14.0 for p in PKEYS}
    C = {p: float(np.mean([R[d][p] for d in R])) for p in PKEYS}
    return R, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", choices=["fw", "px"], default="fw")
    args = ap.parse_args()
    units = args.units

    print(f"sequence-alignment comparator, merged pipeline, units = {units}")
    dom = {v: domain_dtw(v, units) for v in ("maincam", "fullbody")}
    R, C = {}, {}
    for view in dom:
        R[view], C[view] = consensus(dom[view])

    print("\nside view, sequence alignment")
    for d in sorted(dom["maincam"]):
        rk = average_rank(R["maincam"][d], TARGET)
        print(f"  {d:<14} rank of {TARGET} {rk:>5.1f}/15")

    order = sorted(C["maincam"].items(), key=lambda t: -t[1])
    pos = [i + 1 for i, (p, _) in enumerate(order) if p == TARGET][0]
    print(f"\n  side-view consensus: {TARGET} rank {pos}/15, "
          f"C = {C['maincam'][TARGET]:.3f}")

    labels = ["consensus"] + sorted(dom["maincam"])
    pv, rows = [], []
    for lab in labels:
        side = C["maincam"] if lab == "consensus" else R["maincam"][lab]
        front = C["fullbody"] if lab == "consensus" else R["fullbody"][lab]
        rho, p = exact_rho(side, front)
        pv.append(p)
        rows.append((lab, rho, p))
    adj = bh(pv)

    print("\ncross-view consistency, sequence alignment")
    for (lab, rho, p), a in zip(rows, adj):
        print(f"  {lab:<14} rho = {rho:+.3f}  p = {p:.3f}  p_BH = {a:.3f}")

    payload = {"units": units,
               "side_consensus_rank_of_target": pos,
               "side_C": C["maincam"],
               "crossview": [{"test": l, "rho": r, "p": p, "p_bh": float(a)}
                             for (l, r, p), a in zip(rows, adj)]}
    out = DEST / f"pipeline_dtw_{units}.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
