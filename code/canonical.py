"""Canonical pipeline: one configuration, one set of derived numbers.

Freezes the choices that the second-check audit left open, and regenerates every
number the report quotes from that single configuration. Anything that is not
produced here should not be quoted as a result.

CANONICAL CONFIGURATION
-----------------------
hands    left and right tracked as separate channels. A derivative is taken only
         within one hand's own track, and is zeroed when the previous analysed
         frame for that hand was not the immediately preceding analysed frame
         ("gaps broken"). Matched sampling stride (every 5th frame) in both views;
         face-width normalisation in both views.
head     maincam head from `output/maincam`; solo head from `output/solo_head_std`
         (re-extracted on a common 1920 canvas). Both 18D, same builder.
domains  Head_SpeakerB, Head_SpeakerA, Hand_SpeakerB, Hand_SpeakerA.
consensus  mean over domains of 1 - (rank - 1)/14. Higher is better; 1.0 = first
           in every domain. Identical to `table_all_methods.rank_consensus`.

WHY THESE CHOICES
-----------------
Every one of them is forced by a defect found in the audit: the stride mismatch
(`cell_extractor` reads every 5th frame while the old maincam extraction wrote
every 3rd), the two different head-pose extractions, the two normalisation unit
systems, and the hand-identity switching caused by `_one_hand_per_frame`.

Usage:
    python canonical.py            # parts 1-3, writes output/paper_review/final/
"""
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from cell_extractor import run_dtw_domain, run_gmm_domain  # noqa: E402
from gmm_pure_fullbody import maincam_head  # noqa: E402
from hand_two_channel import build, build_single, cache_for  # noqa: E402
from handview_2x2 import SUBSAMPLE, hand_features  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402
from table_all_methods import solo_head_new  # noqa: E402

OUT = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output")
DEST = OUT / "paper_review" / "final"
vids = ["1", "2", "3", "4", "5", "6"]
PEOPLE = [("speakerB", "SpeakerB"), ("speakerA", "SpeakerA")]
PAIRS = [(a, b) for a, b in combinations(vids, 2)]
TARGET = "(4,5)"


# ===================== CONSENSUS (one definition, used everywhere) ==========
def consensus_scores(domain_ranks):
    """{'(i,j)': C} from {domain: {'(i,j)': rank}}."""
    return {
        f"({a},{b})": float(np.mean([1.0 - (domain_ranks[d][f"({a},{b})"] - 1) / 14.0
                                     for d in domain_ranks]))
        for a, b in PAIRS
    }


def rank_of(scores, pair=TARGET):
    """Average rank of `pair` among the 15 consensus scores (ties share a rank)."""
    target = scores[pair]
    above = sum(1 for v in scores.values() if v > target + 1e-12)
    tied = sum(1 for v in scores.values() if abs(v - target) <= 1e-12)
    return 1 + above + (tied - 1) / 2


def summarise(domain_ranks, pair=TARGET):
    C = consensus_scores(domain_ranks)
    return {
        "rank": rank_of(C, pair),
        "C": C[pair],
        "domains": {d: domain_ranks[d][pair] for d in domain_ranks},
        "scores": C,
    }


# ===================== CANONICAL FEATURES ==================================
def two_channel_split(rows, fr, mode="strict"):
    """{handedness: feature array} so each derivative stays inside one hand."""
    by_hand = {"left": [], "right": []}
    for r in rows:
        h = (r.get("handedness") or "").strip().lower()
        if h in by_hand:
            by_hand[h].append(r)

    out = {}
    for h in ["left", "right"]:
        side = sorted(by_hand[h], key=lambda r: int(r["frame"]))
        if len(side) < 3:
            continue
        f = hand_features(side, fr, min_speed=0)
        if f is None:
            continue
        if mode == "strict":
            keep = [int(r["frame"]) for r in side
                    if int(r["frame"]) % SUBSAMPLE == 0 and int(r["frame"]) in fr]
            if len(keep) == len(f):
                bad = np.array([False] + [keep[i] - keep[i - 1] != SUBSAMPLE
                                          for i in range(1, len(keep))])
                f = f.copy()
                f[bad, 2:6] = 0.0          # vx, vy, speed, accel
        if len(f) > 10:
            out[h] = f
    return out


def hand_channels(view, person, mode="strict"):
    """{handedness: {vid: array}} -- the two-channel representation, unconcatenated."""
    cache = cache_for(view, person)
    out = {"left": {}, "right": {}}
    for v in vids:
        if v not in cache:
            continue
        rows, fr = cache[v]
        for h, f in two_channel_split(rows, fr, mode=mode).items():
            out[h][v] = f
    return out


def canonical_heads(view):
    """18D head features for a view, keyed by person."""
    if view == "maincam":
        return {p: maincam_head(p) for p, _ in PEOPLE}
    return {p: solo_head_new(p) for p, _ in PEOPLE}


def canonical_hands(view, mode="strict"):
    """Two-channel hands concatenated across hands -- the GMM/aggregate input."""
    return {p: build(view, p, mode) for p, _ in PEOPLE}


def legacy_hands(view):
    """One hand per frame -- the pre-correction baseline, for audit only."""
    return {p: build_single(view, p) for p, _ in PEOPLE}


# ===================== DOMAIN RANKS ========================================
def gmm_domains(head, hand, quiet=True):
    dr = {}
    for person, name in PEOPLE:
        dr[f"Hand_{name}"] = _sk(run_gmm_domain(hand[person]))
        dr[f"Head_{name}"] = _sk(run_gmm_domain(head[person]))
    return dr


def dtw_domains(head, hand):
    dr = {}
    for person, name in PEOPLE:
        dr[f"Hand_{name}"] = _sk(run_dtw_domain(hand[person]))
        dr[f"Head_{name}"] = _sk(run_dtw_domain(head[person]))
    return dr


def dtw_domains_two_channel(view):
    """DTW under the canonical representation.

    Pooling two interleaved hands into one sequence would be wrong, so DTW is
    run per hand channel and the pair score is the mean over the channels
    present in both recordings. Handedness is treated as meaningful (left is
    compared with left); `assign` in dtw_two_hand.py relaxes that.
    """
    per_person = {}
    for person, name in PEOPLE:
        chans = hand_channels(view, person)
        scaled = {}
        for h in ["left", "right"]:
            d = chans[h]
            if len(d) >= 3:
                sc = RobustScaler().fit(np.vstack(list(d.values())))
                scaled[h] = {v: sc.transform(f) for v, f in d.items()}
        per_person[f"Hand_{name}"] = _dtw_from_scaled(scaled)
    return per_person


def _dtw_from_scaled(scaled):
    from fastdtw import fastdtw
    scores = {}
    for v1, v2 in PAIRS:
        ds = []
        for h in ["left", "right"]:
            if h in scaled and v1 in scaled[h] and v2 in scaled[h]:
                d, path = fastdtw(scaled[h][v1], scaled[h][v2], dist=2)
                ds.append(d / len(path))
        if ds:
            scores[(v1, v2)] = float(np.mean(ds))
    ranks = {}
    for i, (p, _) in enumerate(sorted(scores.items(), key=lambda x: x[1]), 1):
        ranks[p] = i
    return _sk(ranks)


def _sk(d):
    return {f"({k[0]},{k[1]})": v for k, v in d.items()}


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    result = {}

    print("Building canonical features ...")
    heads = {v: canonical_heads(v) for v in ["maincam", "fullbody"]}
    hands = {v: canonical_hands(v) for v in ["maincam", "fullbody"]}
    hands_legacy = {v: legacy_hands(v) for v in ["maincam", "fullbody"]}

    print("\nFrame counts, canonical two-channel hands")
    for view in ["maincam", "fullbody"]:
        for person, name in PEOPLE:
            c = [len(hands[view][person][v]) for v in vids if v in hands[view][person]]
            print(f"   {view:<9}{name:<9}" + " ".join(f"{x:>6}" for x in c)
                  + f"{sum(c):>8}")

    print("\nRunning canonical GMM-UBM domains ...")
    gmm = {v: gmm_domains(heads[v], hands[v]) for v in ["maincam", "fullbody"]}
    print("Running legacy one-hand GMM-UBM domains (audit only) ...")
    gmm_legacy = {v: gmm_domains(heads[v], hands_legacy[v])
                  for v in ["maincam", "fullbody"]}
    print("Running canonical two-channel DTW domains ...")
    dtw2 = {v: dtw_domains_two_channel(v) for v in ["maincam", "fullbody"]}
    print("Running canonical head DTW domains ...")
    for view in ["maincam", "fullbody"]:
        for p, n in PEOPLE:
            dtw2[view][f"Head_{n}"] = _sk(run_dtw_domain(heads[view][p]))
    print("Running legacy one-hand DTW domains (audit only) ...")
    dtw_legacy = {v: dtw_domains(heads[v], hands_legacy[v])
                  for v in ["maincam", "fullbody"]}

    # ---- part 1: canonical method table ----
    print()
    print("=" * 78)
    print("CANONICAL TABLE -- rank and consensus score of (4,5) out of 15")
    print("=" * 78)
    print(f"{'view':<10}{'method':<28}{'domains':<34}{'rank':>6}{'C':>8}")
    print("-" * 78)
    for view in ["maincam", "fullbody"]:
        for method, dr in [("GMM-UBM (canonical)", gmm[view]),
                           ("DTW (two-channel)", dtw2[view]),
                           ("GMM-UBM (legacy 1-hand)", gmm_legacy[view]),
                           ("DTW (legacy 1-hand)", dtw_legacy[view])]:
            s = summarise(dr)
            dom = " ".join(f"{d.split('_')[0][:1]}{d.split('_')[1][:2]}={r:>2}"
                           for d, r in s["domains"].items())
            print(f"{view:<10}{method:<28}{dom:<34}{s['rank']:>4}/15{s['C']:>8.3f}")
            result[f"{view}|{method}"] = s

    # ---- part 2: domain decomposition on the canonical representation ----
    print()
    print("=" * 78)
    print("DOMAIN DECOMPOSITION (canonical representation)")
    print("=" * 78)
    for view in ["maincam", "fullbody"]:
        dr = gmm[view]
        print(f"\nGMM-UBM, {view}")
        print(f"   {'subset':<24}{'rank(4,5)':>11}{'C':>9}")
        subsets = [("head only", ["Head_SpeakerB", "Head_SpeakerA"]),
                   ("hands only", ["Hand_SpeakerB", "Hand_SpeakerA"]),
                   ("full (4 domains)", list(dr))]
        for d in dr:
            subsets.append((f"drop {d}", [x for x in dr if x != d]))
        for name, keep in subsets:
            sub = {d: dr[d] for d in keep}
            s = summarise(sub)
            print(f"   {name:<24}{s['rank']:>8}/15{s['C']:>9.3f}")
            result[f"{view}|GMM|{name}"] = s["rank"]

    # ---- part 3: full 15-pair consensus for cross-view work ----
    result["_pair_scores"] = {
        "gmm_maincam": consensus_scores(gmm["maincam"]),
        "gmm_fullbody": consensus_scores(gmm["fullbody"]),
        "gmm_maincam_domains": gmm["maincam"],
        "gmm_fullbody_domains": gmm["fullbody"],
    }

    (DEST / "canonical.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {DEST / 'canonical.json'}")
