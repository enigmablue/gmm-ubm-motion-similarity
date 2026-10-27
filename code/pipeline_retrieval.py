"""Matched-session retrieval on the merged pipeline.

WHY
---
The manuscript's retrieval figures (2/12 speaker-level, 5/24 domain-level) were
produced in September against a hand definition and sampling stride that the
merged pipeline no longer uses. This module recomputes them from
`pipeline.build_hand_frames` plus the head builders already in the method track,
so that Table 5's retrieval columns come from the same features as its
correlation column.

WHAT IT DOES
------------
For each speaker and each feature kind (Hand, Head):
  - build the side-view and frontal-view feature dicts with the merged builder
  - score every (side query x frontal gallery) cell with gmm_cell
  - rank the six frontal recordings against each side query

Speaker-level consensus averages the hand and head normalised ranks, which is
how the manuscript defines its 12 speaker-level trials.

Inference: the observed hit count against an exact session-label permutation
null over all 6! = 720 relabellings, applied jointly to both speakers in each
domain set.

READS   same feature sources as pipeline.py
WRITES  output/paper_review/final/pipeline_retrieval_{units}.json

Usage:  python pipeline_retrieval.py [--units fw|px]
"""
import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from gmm_pure_fullbody import maincam_head  # noqa: E402
from pipeline import PAIRS, build_hand_frames, vids  # noqa: E402
from session_retrieval import gmm_cell  # noqa: E402
from table_all_methods import solo_head_new  # noqa: E402

DEST = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output\paper_review\final")
PEOPLE = [("speakerB", "SpeakerB"), ("speakerA", "SpeakerA")]
KINDS = ["Hand", "Head"]


def rank_table(scores):
    """{query: {gallery: 1-based rank}} from {(mc q, solo g): score}."""
    tab = {}
    for qi in vids:
        order = sorted(vids,
                       key=lambda g: scores.get((f"mc{qi}", f"solo{g}"), -999.0),
                       reverse=True)
        tab[qi] = {g: i + 1 for i, g in enumerate(order)}
    return tab


def mrr(tab):
    return float(np.mean([1.0 / tab[q][q] for q in vids]))


def tab_top(tab, q):
    return min(tab[q], key=lambda g: tab[q][g])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", choices=["fw", "px"], default="fw")
    args = ap.parse_args()
    units = args.units

    print(f"matched-session retrieval, merged pipeline, units = {units}")

    tables = {}
    for person, name in PEOPLE:
        side = {v: build_hand_frames("maincam", v, person, units) for v in vids}
        front = {v: build_hand_frames("fullbody", v, person, units) for v in vids}
        side = {k: a for k, a in side.items() if a is not None}
        front = {k: a for k, a in front.items() if a is not None}
        s, _, k, r = gmm_cell(side, front)
        tables[f"Hand_{name}"] = rank_table(s)
        print(f"  {name:<7} Hand  K={k} r={r:<4} self-ranks "
              + " ".join(str(tables[f'Hand_{name}'][q][q]) for q in vids))

        hs, _, k, r = gmm_cell(maincam_head(person), solo_head_new(person))
        tables[f"Head_{name}"] = rank_table(hs)
        print(f"  {name:<7} Head  K={k} r={r:<4} self-ranks "
              + " ".join(str(tables[f'Head_{name}'][q][q]) for q in vids))

    # speaker-level consensus: average normalised ranks across the two kinds
    cons = {}
    for _, name in PEOPLE:
        cons[name] = {}
        for q in vids:
            agg = {}
            for g in vids:
                agg[g] = float(np.mean([
                    1.0 - (tables[f"{k}_{name}"][q][g] - 1) / 5.0 for k in KINDS]))
            cons[name][q] = {g: i + 1 for i, (g, _) in enumerate(
                sorted(agg.items(), key=lambda t: -t[1]))}

    perms = list(permutations(vids))
    dom = {f"{k}_{n}": t for k in KINDS for _, n in PEOPLE for t in [tables[f"{k}_{n}"]]}

    def count(tabs, perm):
        return sum(1 for t in tabs.values() for q in vids
                   if perm[tab_top(t, q)] == q)

    obs_dom = count(dom, {v: v for v in vids})
    obs_con = sum(1 for n in cons for q in vids
                  if tab_top(cons[n], q) == q)

    null_dom = np.array([count(dom, dict(zip(vids, p))) for p in perms])
    null_con = np.array([
        sum(1 for n in cons for q in vids
            if dict(zip(vids, p))[tab_top(cons[n], q)] == q) for p in perms])

    p_dom = float(np.mean(null_dom >= obs_dom))
    p_con = float(np.mean(null_con >= obs_con))
    n_dom, n_con = len(dom) * len(vids), len(cons) * len(vids)

    print()
    print(f"  domain-level   {obs_dom}/{n_dom} hits   "
          f"chance {n_dom/6:.1f}   p_exact = {p_dom:.3f}   "
          f"MRR {np.mean([mrr(t) for t in dom.values()]):.3f}")
    print(f"  speaker-level  {obs_con}/{n_con} hits   "
          f"chance {n_con/6:.1f}   p_exact = {p_con:.3f}   "
          f"MRR {np.mean([mrr(cons[n]) for n in cons]):.3f}")

    payload = {"units": units,
               "self_ranks": {k: {q: t[q][q] for q in vids} for k, t in tables.items()},
               "consensus_self_ranks": {n: {q: cons[n][q][q] for q in vids}
                                        for n in cons},
               "domain_hits": obs_dom, "domain_trials": n_dom,
               "domain_p_exact": p_dom,
               "domain_mrr": float(np.mean([mrr(t) for t in dom.values()])),
               "speaker_hits": obs_con, "speaker_trials": n_con,
               "speaker_p_exact": p_con,
               "speaker_mrr": float(np.mean([mrr(cons[n]) for n in cons]))}
    out = DEST / f"pipeline_retrieval_{units}.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
