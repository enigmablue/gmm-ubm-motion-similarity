"""All methods, all configurations, in one table.

The paper's Evaluation 8 fullbody cell is not a fullbody measurement. From
`paper_review/cell_extractor.py`:

    "Head features are identical across camera conditions (both read the
     maincam video); only the hand landmarks differ (maincam vs fullbody)."

So the published fullbody result swaps only the HANDS and keeps the head from the
maincam. This script reports both conventions for every method:

    maincam              head maincam,  hands maincam,  mouth maincam
    fullbody unadjusted  head maincam,  hands fullbody, mouth fullbody  <- the paper
    fullbody adjusted    head solo,     hands fullbody, mouth solo

Aggregate uses six domains (Hand/Head/Mouth x Stella/Aru). GMM-UBM and DTW use
four (Hand/Head x Stella/Aru), since mouth is not one of their features. All
report the consensus rank of pair (4,5) out of 15.

Usage:
    python table_all_methods.py
"""
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from cell_extractor import (build_hand_maincam, run_gmm_domain,  # noqa: E402
                            run_dtw_domain)
from handview_2x2 import load_hand_raw, hand_features, _one_hand_per_frame  # noqa: E402
from aggregate_seven import (head_stats, hand_stats_maincam,  # noqa: E402
                             hand_stats_fullbody, mouth_stats, domain_rank)
from gmm_pure_fullbody import (maincam_head, solo_head, fullbody_hands,  # noqa: E402
                               _head_18d)

SOLO_STD = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output\solo_head_std")
MAINCAM_STD = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output\maincam_hands_std")


def load_maincam_hands_std(person):
    """Maincam hands re-extracted at every 5th frame (matching the fullbody)."""
    out = {}
    fp = "face1" if person == "stella" else "face2"
    for v in vids:
        lm = MAINCAM_STD / f"{v}_{person}_hand_landmarks.csv"
        ft = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output\maincam") / f"{v}_features.csv"
        if not lm.exists():
            continue
        fr = {}
        for r in csv.DictReader(open(ft)):
            if r.get(f"{fp}_detected") == "1":
                fr[int(r["frame"])] = (float(r[f"{fp}_x"]) + float(r[f"{fp}_w"]) / 2,
                                        float(r[f"{fp}_y"]) + float(r[f"{fp}_h"]) / 2,
                                        float(r[f"{fp}_w"]))
        rows = _one_hand_per_frame(list(csv.DictReader(open(lm))))
        out[v] = (sorted(rows, key=lambda r: int(r["frame"])), fr)
    return out

vids = ["1", "2", "3", "4", "5", "6"]
PEOPLE = [("stella", "Stella"), ("aru", "Aru")]
HAND_KEYS = ["CurlMean", "CurlStd", "CurlP25"]
HEAD_KEYS = ["HeadGest", "YawPitchRatio", "HeadDet"]
MOUTH_KEYS = ["talk_pct"]


def _strkeys(d):
    """The model pipelines key pairs as tuples; the aggregate uses '(1,2)'."""
    return {f"({k[0]},{k[1]})" if isinstance(k, tuple) else k: v for k, v in d.items()}


def rank_consensus(domain_ranks):
    """Consensus over any number of domains. Returns (rank, C, next pair, next C)."""
    C = {}
    for v1, v2 in combinations(vids, 2):
        n = f"({v1},{v2})"
        C[n] = float(np.mean([1.0 - (domain_ranks[d][n] - 1) / 14.0
                              for d in domain_ranks]))
    ordered = sorted(C.items(), key=lambda x: x[1], reverse=True)
    # Average ranks for ties, as stated in the Method: a pair tied for 3rd gets
    # 3.5 and the next pair gets 5. Ordinal positions would silently break ties.
    target = C["(4,5)"]
    above = sum(1 for _, c in ordered if c > target + 1e-12)
    tied = sum(1 for _, c in ordered if abs(c - target) <= 1e-12)
    pos = 1 + above + (tied - 1) / 2
    return pos, C["(4,5)"], ordered[1][0], ordered[1][1]


# ===================== MODEL CELLS =====================
def gmm_cell(head, hand):
    dr = {}
    for person, name in PEOPLE:
        dr[f"Hand_{name}"] = _strkeys(run_gmm_domain(hand[person]))
        dr[f"Head_{name}"] = _strkeys(run_gmm_domain(head[person]))
    return rank_consensus(dr)


def dtw_cell(head, hand):
    dr = {}
    for person, name in PEOPLE:
        dr[f"Hand_{name}"] = _strkeys(run_dtw_domain(hand[person]))
        dr[f"Head_{name}"] = _strkeys(run_dtw_domain(head[person]))
    return rank_consensus(dr)


def agg_cell(head, hand, mouth):
    dr = {}
    for person, name in PEOPLE:
        dr[f"Hand_{name}"] = domain_rank(hand[person], HAND_KEYS)[0]
        dr[f"Head_{name}"] = domain_rank(head[person], HEAD_KEYS)[0]
        dr[f"Mouth_{name}"] = domain_rank(mouth[person], MOUTH_KEYS)[0]
    return rank_consensus(dr)


# ===================== PIPELINE =====================
def solo_head_new(person):
    """Solo head pose re-extracted on a common canvas (output/solo_head_std)."""
    import csv as _csv
    raw = {}
    for v in vids:
        p = SOLO_STD / f"{v}_{person}_features.csv"
        if not p.exists():
            continue
        f = _head_18d(list(_csv.DictReader(open(p))), "face1")
        if f is not None and len(f) > 50:
            raw[v] = f
    if raw:
        gm = np.median(np.concatenate([f[:, 12] for f in raw.values()]))
        for v in raw:
            raw[v] = raw[v].copy()
            raw[v][:, 14] = (raw[v][:, 12] < gm).astype(float)
    return raw


def head_stats_solo_std(vid, person):
    """Aggregate head measures from the re-extracted solo head files."""
    p = SOLO_STD / f"{vid}_{person}_features.csv"
    hv, yv, pv, det = [], [], [], 0
    for r in csv.DictReader(open(p)):
        if r.get("face1_detected") != "1":
            continue
        det += 1
        for arr, key in [(hv, "head_vel"), (yv, "yaw_vel"), (pv, "pitch_vel")]:
            val = r.get(f"face1_{key}", "")
            if val not in ("", None):
                arr.append(float(val))
    gest = sum(1 for i in range(1, len(hv)) if hv[i] > 10.0 >= hv[i - 1])
    ratio = sum(abs(v) for v in yv) / (sum(abs(v) for v in pv) + 1e-6) if pv else 0.0
    return {"HeadGest": float(gest), "YawPitchRatio": float(ratio),
            "HeadDet": float(det)}


def maincam_hands():
    out = {}
    for person, _ in PEOPLE:
        cache = load_hand_raw("maincam", person)
        out[person] = {}
        for v in vids:
            if v in cache:
                f = hand_features(cache[v][0], cache[v][1], min_speed=0)
                if f is not None:
                    out[person][v] = f
    return out


def maincam_hands_std():
    """Face-width maincam hands, re-extracted at the analysis stride."""
    out = {}
    for person, _ in PEOPLE:
        cache = load_maincam_hands_std(person)
        out[person] = {}
        for v in vids:
            if v in cache:
                f = hand_features(cache[v][0], cache[v][1], min_speed=0)
                if f is not None:
                    out[person][v] = f
    return out


def maincam_hands_pixel():
    """The published maincam hand features: pixel-relative, not face-width."""
    out = {}
    for person, _ in PEOPLE:
        out[person] = {}
        for v in vids:
            f = build_hand_maincam(v, person)
            if f is not None and len(f) > 10:
                out[person][v] = f
    return out


def main():
    print("Building features...")
    hand_mc_pix = maincam_hands_pixel()          # published maincam hands
    hand_mc = maincam_hands()                     # unified face-width hands
    hand_mc_std = maincam_hands_std()             # face-width, matched stride
    hand_fb = {p: fullbody_hands(p) for p, _ in PEOPLE}
    head_mc = {p: maincam_head(p) for p, _ in PEOPLE}
    head_fb = {p: solo_head(p) for p, _ in PEOPLE}
    head_fb_new = {p: solo_head_new(p) for p, _ in PEOPLE}
    agg_head_fb_new = {p: {v: head_stats_solo_std(v, p) for v in vids}
                       for p, _ in PEOPLE}

    agg_head_mc = {p: {v: head_stats(v, p, "maincam") for v in vids} for p, _ in PEOPLE}
    agg_head_fb = {p: {v: head_stats(v, p, "fullbody") for v in vids} for p, _ in PEOPLE}
    agg_hand_mc = {p: {v: hand_stats_maincam(v, p) for v in vids} for p, _ in PEOPLE}
    agg_hand_fb = {p: {v: hand_stats_fullbody(v, p) for v in vids} for p, _ in PEOPLE}
    m_mc, m_fb = mouth_stats()
    agg_mouth_mc = {p: {v: {"talk_pct": m_mc[p][v]} for v in vids if v in m_mc[p]}
                    for p, _ in PEOPLE}
    agg_mouth_fb = {p: {v: {"talk_pct": m_fb[p][v]} for v in vids if v in m_fb[p]}
                    for p, _ in PEOPLE}

    res = {}
    print("\nAggregate")
    res["agg_mc"] = agg_cell(agg_head_mc, agg_hand_mc, agg_mouth_mc)
    res["agg_fb_adj"] = agg_cell(agg_head_mc, agg_hand_fb, agg_mouth_fb)
    res["agg_fb_unadj_old"] = agg_cell(agg_head_fb, agg_hand_fb, agg_mouth_fb)
    res["agg_fb_unadj_new"] = agg_cell(agg_head_fb_new, agg_hand_fb, agg_mouth_fb)

    print("GMM-UBM")
    res["gmm_mc_pix"] = gmm_cell(head_mc, hand_mc_pix)
    res["gmm_mc_fw"] = gmm_cell(head_mc, hand_mc)
    res["gmm_mc_std"] = gmm_cell(head_mc, hand_mc_std)
    res["gmm_fb_adj"] = gmm_cell(head_mc, hand_fb)
    res["gmm_fb_unadj_old"] = gmm_cell(head_fb, hand_fb)
    res["gmm_fb_unadj_new"] = gmm_cell(head_fb_new, hand_fb)

    print("DTW")
    res["dtw_mc_pix"] = dtw_cell(head_mc, hand_mc_pix)
    res["dtw_mc_fw"] = dtw_cell(head_mc, hand_mc)
    res["dtw_mc_std"] = dtw_cell(head_mc, hand_mc_std)
    res["dtw_fb_adj"] = dtw_cell(head_mc, hand_fb)
    res["dtw_fb_unadj_old"] = dtw_cell(head_fb, hand_fb)
    res["dtw_fb_unadj_new"] = dtw_cell(head_fb_new, hand_fb)

    print()
    print("=" * 88)
    print("CONSENSUS RANK OF PAIR (4,5) OUT OF 15")
    print("=" * 88)
    print(f"{'configuration':<42}{'Aggregate':>14}{'GMM-UBM':>14}{'DTW':>14}")
    print("-" * 88)
    lines = [
        ("maincam (published hands)", "agg_mc", "gmm_mc_pix", "dtw_mc_pix"),
        ("maincam (unified hands, old stride)", "agg_mc", "gmm_mc_fw", "dtw_mc_fw"),
        ("maincam (unified hands, MATCHED stride)", "agg_mc", "gmm_mc_std", "dtw_mc_std"),
        ("fullbody adjusted (maincam head)", "agg_fb_adj", "gmm_fb_adj", "dtw_fb_adj"),
        ("fullbody unadjusted (old solo head)", "agg_fb_unadj_old", "gmm_fb_unadj_old",
         "dtw_fb_unadj_old"),
        ("fullbody unadjusted (NEW solo head)", "agg_fb_unadj_new", "gmm_fb_unadj_new",
         "dtw_fb_unadj_new"),
    ]
    for label, a, b, c in lines:
        def cell(k):
            r, cc, _, _ = res[k]
            return f"{r:>3}/15 C={cc:.3f}"
        print(f"{label:<42}{cell(a):>14}{cell(b):>14}{cell(c):>14}")

    print()
    print("Row meanings")
    print("  adjusted    hands from the solo cameras, head taken from the maincam")
    print("              -- the paper's Evaluation 8 convention")
    print("  unadjusted  head AND hands from the solo cameras (the view's own measures)")
    print("  Aggregate C is over 6 domains (hand, head, mouth x 2 speakers);")
    print("  GMM-UBM and DTW C is over 4 domains (hand, head x 2 speakers).")


if __name__ == "__main__":
    main()
