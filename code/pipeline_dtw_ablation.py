"""Sequence alignment across the same feature configurations, both views.

Section "Sequence alignment with matched representations" quotes DTW rankings
for eight head configurations and for alternative hand representations, in both
views. Those numbers came from the old hand source and were never recomputed
with the corrected one.

Method is unchanged from `pipeline_comparators.py`: one pooled hand channel and
one head channel per domain, per-dimension robust scaling fitted on the pooled
frames, fastdtw with Euclidean distance, cumulative cost divided by path length,
and the identical rank-inversion consensus.

WRITES  output/paper_review/final/dtw_ablation_{units}.json
Usage:  python pipeline_dtw_ablation.py [--units fw|px]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from gmm_pure_fullbody import maincam_head  # noqa: E402
from pipeline import DEST, PKEYS, build_hand_frames, vids  # noqa: E402
from pipeline_comparators import consensus, domain_dtw  # noqa: E402
from review2_raw_scores import dtw_scores_single  # noqa: E402
from table_all_methods import solo_head_new  # noqa: E402

PERSON = [("speakerB", "SpeakerB"), ("speakerA", "SpeakerA")]
HEAD_CONFIGS = {
    "9D": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "9D+S": [0, 1, 2, 3, 4, 5, 6, 7, 8, 14],
    "9D+E": [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17],
    "9D+A": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13],
    "9D+E+S": [0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17],
    "9D+A+S": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14],
    "9D+A+E": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17],
    "18D": list(range(18)),
}
HAND_CONFIGS = ["12D", "8D wrist", "4D finger"]


def hand_frames(view, vid, person, units, cfg):
    f = build_hand_frames(view, vid, person, units)
    if f is None:
        return None
    if cfg == "12D":
        return f
    return f[:, 0:8] if cfg == "8D wrist" else f[:, 8:12]


def run_heads(units, cols):
    dom = {}
    for view, heads in [("maincam", maincam_head), ("fullbody", solo_head_new)]:
        for person, name in PERSON:
            raw = {v: f[:, cols] for v, f in heads(person).items() if len(f) > 50}
            s = dtw_scores_single(raw)
            if s is None:
                return None
            dom[(view, f"Head_{name}")] = s
    return dom


def run_hands(units, cfg):
    dom = {}
    for view in ("maincam", "fullbody"):
        for person, name in PERSON:
            raw = {v: f for v in vids
                   if (f := hand_frames(view, v, person, units, cfg))
                   is not None and len(f) > 50}
            s = dtw_scores_single(raw)
            if s is None:
                return None
            dom[(view, f"Hand_{name}")] = s
    return dom


def cons_for(dom, subset):
    """Consensus using only the domains in `subset`, per view."""
    out = {}
    for view in ("maincam", "fullbody"):
        sub = {k[1]: v for k, v in dom.items() if k[0] == view and k[1] in subset}
        if len(sub) < 2:
            out[view] = None
            continue
        _, C = consensus(sub)
        out[view] = C
    return out


def rank_of(C):
    if C is None:
        return None
    t = C["(4,5)"]
    above = sum(1 for v in C.values() if v > t + 1e-12)
    tied = sum(1 for v in C.values() if abs(v - t) <= 1e-12)
    return 1 + above + (tied - 1) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", choices=["fw", "px"], default="fw")
    args = ap.parse_args()
    units = args.units
    out = {"units": units, "head": {}, "hand": {}}

    print(f"sequence alignment with matched representations, units={units}")
    print("\nhead configurations (hands pooled 12D)")
    for cfg, cols in HEAD_CONFIGS.items():
        dom = run_heads(units, cols)
        hand = run_hands(units, "12D")
        merged = dict(dom)
        merged.update({k: v for k, v in hand.items()})
        sub = {f"Head_{n}" for _, n in PERSON} | {f"Hand_{n}" for _, n in PERSON}
        C = cons_for(merged, sub)
        # per-head-domain ranks
        ranks = {}
        for person, name in PERSON:
            s = dom[("maincam", f"Head_{name}")]
            r = {p for p in PKEYS}
            t = s["(4,5)"]
            ranks[name] = (1 + sum(1 for p in r if s[p] < t - 1e-12)
                           + (sum(1 for p in r if abs(s[p] - t) <= 1e-12) - 1) / 2)
        row = dict(head_ranks=ranks,
                   side_cons=rank_of(C["maincam"]),
                   front_cons=rank_of(C["fullbody"]))
        out["head"][cfg] = row
        print(f"  {cfg:<8} head ranks St {ranks['SpeakerB']:.1f} Ar {ranks['SpeakerA']:.1f}"
              f"   consensus side {row['side_cons']:.1f}/15"
              f"  front {row['front_cons']:.1f}/15")

    print("\nhand configurations (heads 18D)")
    heads = run_heads(units, list(range(18)))
    for cfg in HAND_CONFIGS:
        dom = run_hands(units, cfg)
        merged = dict(heads)
        merged.update({k: v for k, v in dom.items()})
        sub = {f"Head_{n}" for _, n in PERSON} | {f"Hand_{n}" for _, n in PERSON}
        C = cons_for(merged, sub)
        ranks = {}
        for person, name in PERSON:
            s = dom[("maincam", f"Hand_{name}")]
            t = s["(4,5)"]
            ranks[name] = (1 + sum(1 for p in PKEYS if s[p] < t - 1e-12)
                           + (sum(1 for p in PKEYS if abs(s[p] - t) <= 1e-12) - 1) / 2)
        row = dict(hand_ranks=ranks, side_cons=rank_of(C["maincam"]),
                   front_cons=rank_of(C["fullbody"]))
        out["hand"][cfg] = row
        print(f"  {cfg:<10} hand ranks St {ranks['SpeakerB']:.1f} Ar {ranks['SpeakerA']:.1f}"
              f"   consensus side {row['side_cons']:.1f}/15"
              f"  front {row['front_cons']:.1f}/15")

    p = DEST / f"dtw_ablation_{units}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
