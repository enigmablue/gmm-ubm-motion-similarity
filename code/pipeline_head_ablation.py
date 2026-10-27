"""Table 8 rebuilt: head temporal-context ablation on the corrected backbone.

Why this exists
---------------
`head_ablations.py` holds the hand features to a copy of the old 12D builder,
which reads the stride-3 landmark file, filters it with SUBSAMPLE=5, and forms
derivatives across hand switches. The head side of that table was always on the
right source, but the hand backdrop it was combined with was not, so the table
could not sit next to a corrected Table 7.

This script changes the backdrop and nothing else. The 18D head base comes from
`maincam_head`, the same builder the merged pipeline uses, including its global
stillness median. The eight configurations are column subsets of that one base,
so they differ only in which temporal classes they can see. Hands are fixed at
the corrected 12D vector for both speakers.

Column layout of the 18D base:
  0-2 angles | 3-5 velocity | 6-8 |velocity| | 9-11 axis accel
  12 angular speed | 13 angular accel | 14 stillness | 15-17 excursion

WRITES  output/paper_review/final/head_ablation_{units}.json
Usage:  python pipeline_head_ablation.py [--units fw|px]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from gmm_pure_fullbody import maincam_head  # noqa: E402
from pipeline import (DEST, average_rank, build_hand_frames, consensus,  # noqa: E402
                      key, vids)
from review2_raw_scores import gmm_scores  # noqa: E402

PERSON = [("speakerB", "SpeakerB"), ("speakerA", "SpeakerA")]
PKEYS = [f"({a},{b})" for i, a in enumerate(vids) for b in vids[i + 1:]]
ORDER = ["9D", "9D+S", "9D+E", "9D+A", "9D+E+S", "9D+A+S", "9D+A+E", "18D"]

HEAD_CONFIGS = {
    "9D":     [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "9D+A":   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13],
    "9D+E":   [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17],
    "9D+S":   [0, 1, 2, 3, 4, 5, 6, 7, 8, 14],
    "9D+A+E": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17],
    "9D+A+S": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14],
    "9D+E+S": [0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17],
    "18D":    list(range(18)),
}

# Manuscript Table 8: dims, Hd-St, Hd-Ar, C45, z
MANUSCRIPT = {
    "9D":     (9, 5, 6, 0.750, 1.78),
    "9D+S":   (10, 4, 3, 0.821, 2.33),
    "9D+E":   (12, 3, 4, 0.821, 2.38),
    "9D+A":   (13, 4, 4, 0.804, 2.15),
    "9D+E+S": (13, 2, 2, 0.875, 2.79),
    "9D+A+S": (14, 2, 6, 0.804, 2.02),
    "9D+A+E": (16, 3, 3, 0.839, 2.50),
    "18D":    (18, 3, 2, 0.857, 2.48),
}


def scored(frames_by_vid):
    s, k, r = gmm_scores(frames_by_vid)
    if s is None:
        return None, None, None
    return {key(p): x for p, x in s.items()}, k, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", choices=["fw", "px"], default="fw")
    args = ap.parse_args()
    units = args.units

    # Hand backdrop, held constant across all eight rows.
    hands = {}
    for person, name in PERSON:
        raw = {v: f for v in vids
               if (f := build_hand_frames("maincam", v, person, units))
               is not None and len(f) > 50}
        s, k, r = scored(raw)
        hands[f"Hand_{name}"] = (s, k, r, None)
        print(f"hand {name}: K={k}, r={r}  (held fixed across all rows)")

    # 18D head base, built once.
    base = {person: maincam_head(person) for person, _ in PERSON}

    print(f"\nhead temporal-context ablation, {units}, hands fixed at 12D")
    print(f"{'config':<10}{'dims':>5}{'Hd-St':>7}{'Hd-Ar':>7}{'C45':>8}"
          f"{'cons':>8}{'z':>7}   manuscript")
    payload = {}
    for cfg in ORDER:
        cols = HEAD_CONFIGS[cfg]
        dom = dict(hands)
        for person, name in PERSON:
            raw = {v: f[:, cols] for v, f in base[person].items()
                   if len(f) > 50}
            s, k, r = scored(raw)
            if s is None:
                print(f"  {cfg}: no data for {person}")
                return
            dom[f"Head_{name}"] = (s, k, r, None)

        _, cons = consensus(dom)
        c45 = cons["(4,5)"]
        others = [cons[p] for p in PKEYS if p != "(4,5)"]
        z = (c45 - float(np.mean(others))) / float(np.std(others))
        rank = 1 + sum(1 for p in PKEYS if p != "(4,5)"
                       and cons[p] > c45 + 1e-12)
        r_st = average_rank(dom["Head_SpeakerB"][0], "(4,5)")
        r_ar = average_rank(dom["Head_SpeakerA"][0], "(4,5)")

        old = MANUSCRIPT[cfg]
        print(f"{cfg:<10}{len(cols):>5}{r_st:>7.0f}{r_ar:>7.0f}{c45:>8.3f}"
              f"{rank:>6}/15{z:>7.2f}   "
              f"{old[1]}/{old[2]}/{old[3]:.3f}/z{old[4]:.2f}")
        payload[cfg] = dict(dims=len(cols), Hd_St=float(r_st),
                            Hd_Ar=float(r_ar), C45=float(c45),
                            cons_rank=int(rank), z=float(z))

    out = DEST / f"head_ablation_{units}.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}")
    print("The 18D row must match the headline head ranks and C for these "
          "units; if it does not, the table is not on the same footing.")


if __name__ == "__main__":
    main()
