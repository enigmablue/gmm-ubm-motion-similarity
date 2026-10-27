"""Three numbers Table 5 and Table 6 need that no stored result file contains.

1. Frontal per-domain ranks. `pipeline.py` stores the side-view domain table but
   only the frontal consensus and its first pair, so Table 6's frontal columns
   were never written down.

2. Retrieval for the sequence-alignment row. `pipeline_comparators.py` computes
   the cross-view correlation and the side-view ordering for DTW but not the
   matched-session retrieval column.

3. Retrieval for the aggregate row. Its sources are untouched, so this is a
   reproduction check: the numbers should land on the manuscript's.

Sequence-alignment retrieval reuses the original definitions in
`review2_method_tests.py` (`dtw_cell`, `gallery_ranks`, `consensus_table`) so
that only the feature source changes, not the protocol. Hand channels are built
per hand from the corrected stride-5 source, mirroring the pooled builder.

WRITES  output/paper_review/final/pipeline_extras_{units}.json
Usage:  python pipeline_extras.py [--units fw|px]
"""
import argparse
import csv
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from gmm_pure_fullbody import maincam_head  # noqa: E402
from pipeline import (DEST, FPS, HANDS, SUBSAMPLE, VIEWS, average_rank,  # noqa: E402
                      domain_scores, face_ref, hand_rows, vids)
from review2_method_tests import (agg_cell, consensus_table,  # noqa: E402
                                  dtw_cell, gallery_ranks, mrr, run_aggregate)
from table_all_methods import solo_head_new  # noqa: E402

PEOPLE = [("stella", "Stella"), ("aru", "Aru")]
TARGET = "(4,5)"


def per_hand_frames(view, vid, person, units, hand):
    """One hand's 12D vector, same rows and gap rule as the pooled builder."""
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
    prev_wrist = prev_frame = prev_speed = None
    prev_ok = False
    for r in side:
        fi = int(r["frame"])
        if fi not in ref:
            prev_frame = None
            prev_ok = False
            continue
        wrist = np.array([float(r["lm0_x"]), float(r["lm0_y"])])
        cx, cy, fw = ref[fi]
        scale = fw if (units == "fw" and fw > 0) else 1.0
        ok = prev_frame is not None and (fi - prev_frame) == SUBSAMPLE
        if ok:
            dt = SUBSAMPLE / FPS
            v = (wrist - prev_wrist) / dt / scale
            speed = float(np.linalg.norm(v))
            if speed > 1e-6:
                ds, dc = v[1] / speed, v[0] / speed
            else:
                ds = dc = 0.0
        else:
            v = np.zeros(2)
            speed = ds = dc = 0.0
        accel = ((speed - prev_speed) / (SUBSAMPLE / FPS)
                 if ok and prev_ok else 0.0)
        tips = np.array([np.linalg.norm([float(r[f"lm{t}_x"]) - wrist[0],
                                         float(r[f"lm{t}_y"]) - wrist[1]])
                         for t in [4, 8, 12, 16, 20]]) / scale
        aperture = float(np.linalg.norm(
            [float(r["lm4_x"]) - float(r["lm8_x"]),
             float(r["lm4_y"]) - float(r["lm8_y"])])) / scale
        tip_xy = np.array([[float(r[f"lm{t}_x"]), float(r[f"lm{t}_y"])]
                           for t in [4, 8, 12, 16, 20]])
        spread = float(np.mean([np.linalg.norm(tip_xy[i] - tip_xy[j])
                                for i in range(5) for j in range(i + 1, 5)])) / scale
        out.append(np.array([(wrist[0] - cx) / scale, (wrist[1] - cy) / scale,
                             v[0], v[1], speed, accel, ds, dc,
                             float(tips.mean()), float(tips.std()),
                             aperture, spread]))
        prev_wrist, prev_speed, prev_frame, prev_ok = wrist, speed, fi, ok
    return np.array(out) if out else None


def dtw_tables(units):
    tabs = {}
    for person, name in PEOPLE:
        acc = {}
        for hand in HANDS:
            q = {v: f for v in vids
                 if (f := per_hand_frames("maincam", v, person, units, hand))
                 is not None and len(f) > 50}
            g = {v: f for v in vids
                 if (f := per_hand_frames("fullbody", v, person, units, hand))
                 is not None and len(f) > 50}
            s = dtw_cell(q, g)
            if s is None:
                continue
            for k, v in s.items():
                acc.setdefault(k, []).append(v)
        tabs[f"Hand_{name}"] = gallery_ranks(
            {k: float(np.mean(v)) for k, v in acc.items()},
            lower_is_better=True)
        hq = {v: f for v in vids
              if (f := maincam_head(person).get(v)) is not None and len(f) > 50}
        hg = {v: f for v in vids
              if (f := solo_head_new(person).get(v)) is not None and len(f) > 50}
        tabs[f"Head_{name}"] = gallery_ranks(dtw_cell(hq, hg),
                                             lower_is_better=True)
    return tabs


def summarise(tabs):
    cons = {name: consensus_table([tabs[f"Hand_{name}"], tabs[f"Head_{name}"]])
            for _, name in PEOPLE}
    perms = list(permutations(range(len(vids))))
    null_dom, null_con = [], []
    for p in perms:
        pd = {vids[i]: vids[p[i]] for i in range(len(vids))}
        null_dom.append(sum(1 for t in tabs.values() for q in vids
                            if t[q][pd[q]] == 1))
        null_con.append(sum(1 for n in cons for q in vids
                            if cons[n][q][pd[q]] == 1))
    dom = sum(1 for t in tabs.values() for q in vids if t[q][q] == 1)
    con = sum(1 for n in cons for q in vids if cons[n][q][q] == 1)
    return {
        "domain_hits": dom, "domain_trials": 24,
        "domain_p": float(np.mean(np.array(null_dom) >= dom)),
        "domain_mrr": float(np.mean([mrr(t) for t in tabs.values()])),
        "speaker_hits": con, "speaker_trials": 12,
        "speaker_p": float(np.mean(np.array(null_con) >= con)),
        "speaker_mrr": float(np.mean([mrr(c) for c in cons.values()])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", choices=["fw", "px"], default="fw")
    args = ap.parse_args()
    units = args.units
    out = {"units": units}

    # 1. frontal per-domain ranks
    front = domain_scores("fullbody", units)
    out["frontal_ranks"] = {
        d: dict(K=v[1], r=v[2], rank=average_rank(v[0], TARGET),
                llr=v[0][TARGET])
        for d, v in front.items()}
    print(f"frontal per-domain ranks, units={units}")
    for d, v in out["frontal_ranks"].items():
        print(f"  {d:<14} K={v['K']} r={v['r']:<4} rank(4,5)={v['rank']:.1f}")

    # 2. sequence-alignment retrieval on corrected features
    out["dtw_retrieval"] = summarise(dtw_tables(units))
    d = out["dtw_retrieval"]
    print(f"\nsequence-alignment retrieval: domain {d['domain_hits']}/24 "
          f"p={d['domain_p']:.3f} MRR={d['domain_mrr']:.3f}; "
          f"speaker {d['speaker_hits']}/12 p={d['speaker_p']:.3f} "
          f"MRR={d['speaker_mrr']:.3f}")

    # 3. aggregate retrieval, unchanged sources, reproduction check
    out["aggregate_retrieval"] = summarise(run_aggregate())
    a = out["aggregate_retrieval"]
    print(f"aggregate retrieval (reproduction): domain {a['domain_hits']}/24 "
          f"p={a['domain_p']:.3f} MRR={a['domain_mrr']:.3f}; "
          f"speaker {a['speaker_hits']}/12 p={a['speaker_p']:.3f} "
          f"MRR={a['speaker_mrr']:.3f}")

    p = DEST / f"pipeline_extras_{units}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
