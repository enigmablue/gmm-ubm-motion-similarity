"""Put all three methods through the two tests that have an external criterion.

WHY THIS EXISTS
---------------
The recommended RQ structure puts cross-view consistency and matched-session
retrieval at the centre, and leaves the post-hoc pair (4,5) as exploratory. Only
GMM-UBM has been through both tests. Comparing the methods on (4,5) is
selection-conditioned -- that pair was nominated post-hoc, possibly from a
distributional method's own output -- so it cannot carry a claim that one method
is better. These two tests can, because neither uses (4,5) at all:

  TEST 1  cross-view consistency. Correlate each method's 6x6 session-similarity
          matrix from the side view with the same method's matrix from the
          frontal view. Exact session-label permutation inference.
  TEST 2  matched-session retrieval. For each speaker and each session, score the
          side-view recording against all six frontal recordings of that speaker
          and ask whether the genuinely corresponding session comes first.
          The answer is fixed by recording time, not by any method.

Every method is scored on exactly the same trials, and the same session
permutation is applied to all of them, so the hit totals are directly comparable.

Usage:
    python review2_method_tests.py
"""
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
from fastdtw import fastdtw
from scipy import stats
from sklearn.preprocessing import RobustScaler

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from aggregate_seven import (hand_stats_fullbody, hand_stats_maincam,  # noqa: E402
                             head_stats, mouth_stats, pct_diff)
from canonical import canonical_hands, canonical_heads, hand_channels  # noqa: E402
from review2_checks import avg_ranks, consensus_from_scores, to_matrix  # noqa: E402
from session_retrieval import gmm_cell  # noqa: E402
from table_all_methods import head_stats_solo_std  # noqa: E402

OUT = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output")
DEST = OUT / "paper_review" / "final"
vids = ["1", "2", "3", "4", "5", "6"]
PEOPLE = [("speakerB", "SpeakerB"), ("speakerA", "SpeakerA")]
DOMAINS = ["Hand_SpeakerB", "Hand_SpeakerA", "Head_SpeakerB", "Head_SpeakerA"]
HEAD_KEYS = ["HeadGest", "YawPitchRatio", "HeadDet"]
HAND_KEYS = ["CurlMean", "CurlStd", "CurlP25"]


def _zscore(a):
    sd = a.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (a - a.mean(axis=0)) / sd


def _z1(d):
    """Standardise each scalar feature across the six recordings of one view.

    `d` is {recording: {feature: value}}. Centring one recording against itself is
    impossible when a domain is a single number per recording, so the camera offset
    is removed per feature instead: each view's six values are given zero mean and
    unit SD, and only their relative order survives.
    """
    keys = list(d[vids[0]])
    out = {x: {} for x in vids}
    for k in keys:
        v = np.array([d[x][k] for x in vids], float)
        sd = v.std() if v.std() > 1e-12 else 1.0
        for x in vids:
            out[x][k] = float((d[x][k] - v.mean()) / sd)
    return out


# ===================== DTW CROSS-VIEW CELL =================================
def dtw_cell(query, gallery, center=False):
    """Mean path cost for every (side-view query, frontal gallery) pair."""
    Q = {f"mc{v}": a for v, a in query.items()}
    G = {f"solo{v}": a for v, a in gallery.items()}
    if len(Q) < 2 or len(G) < 2:
        return None
    data = {**Q, **G}
    if center:
        data = {k: _zscore(a) for k, a in data.items()}
    sc = RobustScaler().fit(np.vstack(list(data.values())))
    scaled = {k: sc.transform(a) for k, a in data.items()}
    out = {}
    for q in Q:
        for g in G:
            d, path = fastdtw(scaled[q], scaled[g], dist=2)
            out[(q, g)] = float(d / len(path))
    return out


# ===================== AGGREGATE CROSS-VIEW CELL ===========================
def agg_cell(q, g, keys, center=False):
    """Mean percentage difference for every (query, gallery) pair. Lower = closer."""
    if center:
        q, g = _z1(q), _z1(g)
    out = {}
    for qi in vids:
        for gi in vids:
            out[(f"mc{qi}", f"solo{gi}")] = float(
                np.mean([pct_diff(q[qi][k], g[gi][k]) for k in keys]))
    return out


# ===================== RANKS AND HITS ======================================
def gallery_ranks(scores, lower_is_better=False):
    out = {}
    for q in vids:
        key = lambda g: scores.get((f"mc{q}", f"solo{g}"), 999.0)
        order = sorted(vids, key=key,
                       reverse=not lower_is_better)
        out[q] = {g: i + 1 for i, g in enumerate(order)}
    return out


def consensus_table(tabs):
    tab = {}
    for q in vids:
        mean = {g: float(np.mean([1.0 - (t[q][g] - 1) / 5.0 for t in tabs]))
                for g in vids}
        order = sorted(vids, key=lambda g: mean[g], reverse=True)
        tab[q] = {g: i + 1 for i, g in enumerate(order)}
    return tab


def mrr(tab):
    return float(np.mean([1.0 / tab[q][q] for q in vids]))


# ===================== METHOD RUNNERS ======================================
def run_gmm(center=False):
    tabs = {}
    for person, name in PEOPLE:
        q_ch, g_ch = hand_channels("maincam", person), hand_channels("fullbody", person)
        acc = {}
        for h in ["left", "right"]:
            if len(q_ch[h]) < 3 or len(g_ch[h]) < 3:
                continue
            s = gmm_cell(q_ch[h], g_ch[h], center=center)[0]
            if s is None:
                continue
            for k, v in s.items():
                acc.setdefault(k, []).append(v)
        tabs[f"Hand_{name}"] = gallery_ranks({k: float(np.mean(v))
                                              for k, v in acc.items()})
        hq, hg = canonical_heads("maincam")[person], canonical_heads("fullbody")[person]
        tabs[f"Head_{name}"] = gallery_ranks(gmm_cell(hq, hg, center=center)[0])
    return tabs


def run_dtw(center=False):
    tabs = {}
    for person, name in PEOPLE:
        q_ch, g_ch = hand_channels("maincam", person), hand_channels("fullbody", person)
        acc = {}
        for h in ["left", "right"]:
            if len(q_ch[h]) < 3 or len(g_ch[h]) < 3:
                continue
            s = dtw_cell(q_ch[h], g_ch[h], center=center)
            if s is None:
                continue
            for k, v in s.items():
                acc.setdefault(k, []).append(v)
        tabs[f"Hand_{name}"] = gallery_ranks({k: float(np.mean(v))
                                              for k, v in acc.items()},
                                             lower_is_better=True)
        hq, hg = canonical_heads("maincam")[person], canonical_heads("fullbody")[person]
        tabs[f"Head_{name}"] = gallery_ranks(dtw_cell(hq, hg, center=center),
                                             lower_is_better=True)
    return tabs


def run_aggregate(center=False):
    tabs = {}
    for person, name in PEOPLE:
        qh = {v: head_stats(v, person, "maincam") for v in vids}
        gh = {v: head_stats_solo_std(v, person) for v in vids}
        qg = {v: hand_stats_maincam(v, person) for v in vids}
        gg = {v: hand_stats_fullbody(v, person) for v in vids}
        tabs[f"Head_{name}"] = gallery_ranks(
            agg_cell(qh, gh, HEAD_KEYS, center), lower_is_better=True)
        tabs[f"Hand_{name}"] = gallery_ranks(
            agg_cell(qg, gg, HAND_KEYS, center), lower_is_better=True)
    return tabs


# ===================== COUNTS UNDER A SESSION PERMUTATION ==================
def counts(tabs, cons, perm):
    """Hits under a permutation of gallery session labels."""
    pd = {vids[i]: vids[perm[i]] for i in range(6)}
    dom = sum(1 for d in DOMAINS for q in vids if tabs[d][q][pd[q]] == 1)
    con = sum(1 for name in cons for q in vids if cons[name][q][pd[q]] == 1)
    return dom, con


def observed(tabs, cons):
    dom = sum(1 for d in DOMAINS for q in vids if tabs[d][q][q] == 1)
    con = sum(1 for name in cons for q in vids if cons[name][q][q] == 1)
    return dom, con


# ===================== CROSS-VIEW CONSISTENCY ==============================
def crossview(raw, higher_is_better):
    C = {}
    for view in ["maincam", "fullbody"]:
        C[view], _ = consensus_from_scores(raw[view], higher_is_better)
    return C


def exact_rho(M1, M2):
    iu = np.triu_indices(6, 1)
    obs = stats.spearmanr(M1[iu], M2[iu])[0]
    null = []
    for perm in permutations(range(6)):
        P = np.zeros((6, 6))
        for i, p in enumerate(perm):
            P[i, p] = 1.0
        r = stats.spearmanr(M1[iu], (P @ M2 @ P.T)[iu])[0]
        if np.isfinite(r):
            null.append(r)
    null = np.array(null)
    return float(obs), float(np.mean(null >= obs - 1e-12)), \
        float(np.mean(np.abs(null) >= abs(obs) - 1e-12)), null


def session_bootstrap(M1, M2, n=20000, seed=0):
    """Resample the six sessions, recompute both edge vectors, return rho pairs."""
    rng = np.random.default_rng(seed)
    a = np.array([int(v) - 1 for v in vids])
    out1, out2 = [], []
    for _ in range(n):
        idx = rng.choice(a, size=6, replace=True)
        e1, e2 = [], []
        for i in range(6):
            for j in range(i + 1, 6):
                e1.append(M1[idx[i], idx[j]])
                e2.append(M2[idx[i], idx[j]])
        if np.std(e1) < 1e-12 or np.std(e2) < 1e-12:
            continue
        out1.append(stats.spearmanr(e1, e2)[0])
    return np.array(out1)


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    out = {}
    perms = list(permutations(range(6)))

    # ---------------- TEST 2: retrieval, all three methods ----------------
    print("=" * 84)
    print("TEST 2  MATCHED-SESSION RETRIEVAL, ALL THREE METHODS")
    print("=" * 84)

    results = {}
    for label, fn, kw in [("GMM-UBM", run_gmm, {}),
                          ("DTW", run_dtw, {}),
                          ("Aggregate", run_aggregate, {}),
                          ("GMM-UBM, per-recording centred", run_gmm, {"center": True}),
                          ("DTW, per-recording centred", run_dtw, {"center": True}),
                          ("Aggregate, per-recording centred", run_aggregate,
                           {"center": True})]:
        tabs = fn(**kw)
        cons = {name: consensus_table([tabs[f"Hand_{name}"], tabs[f"Head_{name}"]])
                for _, name in PEOPLE}
        dom, con = observed(tabs, cons)
        dom_ranks = [tabs[d][q][q] for d in DOMAINS for q in vids]
        con_ranks = [cons[name][q][q] for name in cons for q in vids]
        results[label] = {"tabs": tabs, "cons": cons, "dom": dom, "con": con,
                          "domain_ranks": dom_ranks, "consensus_ranks": con_ranks,
                          "mrr_domain": float(np.mean([1.0 / r for r in dom_ranks])),
                          "mrr_consensus": float(np.mean([1.0 / r for r in con_ranks]))}
        print(f"\n   {label}")
        print(f"      domain self-ranks  : {dom_ranks}  -> {dom}/24 hits"
              f"   MRR {np.mean([1.0 / r for r in dom_ranks]):.3f}")
        print(f"      consensus ranks    : {con_ranks}  -> {con}/12 hits"
              f"   MRR {np.mean([1.0 / r for r in con_ranks]):.3f}")

    # exact permutation nulls, same 720 permutations for every method and variant
    print()
    print("   exact session-label permutation (same 720 permutations for all methods)")
    print(f"   {'variant':<34}{'domain':>9}{'p':>8}{'consensus':>11}{'p':>8}"
          f"{'MRR dom':>9}{'MRR con':>9}")
    nulls = {}
    out["retrieval"] = {}
    for label, r in results.items():
        nd, nc = [], []
        for perm in perms:
            a, b = counts(r["tabs"], r["cons"], perm)
            nd.append(a)
            nc.append(b)
        nd, nc = np.array(nd), np.array(nc)
        nulls[label] = (nd, nc)
        pd_ = float(np.mean(nd >= r["dom"]))
        pc_ = float(np.mean(nc >= r["con"]))
        print(f"   {label:<34}{r['dom']:>6}/24{pd_:>8.3f}{r['con']:>8}/12{pc_:>8.3f}"
              f"{r['mrr_domain']:>9.3f}{r['mrr_consensus']:>9.3f}")
        out["retrieval"][label] = {
            "domain_hits": r["dom"], "consensus_hits": r["con"],
            "domain_ranks": r["domain_ranks"],
            "consensus_ranks": r["consensus_ranks"],
            "mrr_domain": r["mrr_domain"], "mrr_consensus": r["mrr_consensus"],
            "p_domain": pd_, "p_consensus": pc_,
            "null_domain": [float(nd.min()), float(nd.max()), float(nd.mean())],
            "null_consensus": [float(nc.min()), float(nc.max()), float(nc.mean())]}

    # paired comparison against GMM-UBM
    print()
    print("   paired comparison — is the gap between methods larger than chance alone?")
    out["paired"] = {}
    pairs = [("GMM-UBM", "DTW"), ("GMM-UBM", "Aggregate"),
             ("GMM-UBM, per-recording centred", "DTW, per-recording centred"),
             ("GMM-UBM, per-recording centred", "Aggregate, per-recording centred")]
    for a_lab, b_lab in pairs:
        d_obs = results[a_lab]["dom"] - results[b_lab]["dom"]
        c_obs = results[a_lab]["con"] - results[b_lab]["con"]
        dd = nulls[a_lab][0] - nulls[b_lab][0]
        cc = nulls[a_lab][1] - nulls[b_lab][1]
        print(f"      {a_lab} minus {b_lab}")
        print(f"         domain {d_obs:+d} of 24 (exact p = {np.mean(dd >= d_obs):.3f}, "
              f"null {int(dd.min()):+d} to {int(dd.max()):+d})   "
              f"consensus {c_obs:+d} of 12 (exact p = {np.mean(cc >= c_obs):.3f}, "
              f"null {int(cc.min()):+d} to {int(cc.max()):+d})")
        out["paired"][f"{a_lab} minus {b_lab}"] = {
            "domain_diff": int(d_obs), "domain_p": float(np.mean(dd >= d_obs)),
            "consensus_diff": int(c_obs), "consensus_p": float(np.mean(cc >= c_obs)),
            "null_domain_range": [int(dd.min()), int(dd.max())],
            "null_consensus_range": [int(cc.min()), int(cc.max())]}

    # ---------------- TEST 1: cross-view consistency, all three methods ----
    print()
    print("=" * 84)
    print("TEST 1  CROSS-VIEW CONSISTENCY, ALL THREE METHODS")
    print("=" * 84)

    rawg = json.loads((DEST / "raw_gmm.json").read_text())
    rawd = json.loads((DEST / "raw_dtw.json").read_text())

    cv = {}
    G = crossview(rawg, True)
    cv["GMM-UBM"] = (to_matrix(G["maincam"]), to_matrix(G["fullbody"]))
    D = crossview(rawd, False)
    cv["DTW"] = (to_matrix(D["maincam"]), to_matrix(D["fullbody"]))

    m_mc, m_fb = mouth_stats()
    from crossview_exact import agg_domains  # noqa: E402
    Amc = {k: v for k, v in agg_domains("maincam", m_mc).items()
           if not k.startswith("Mouth")}
    Afb = {k: v for k, v in agg_domains("fullbody", m_fb).items()
           if not k.startswith("Mouth")}
    Cmc = consensus_from_scores(Amc, False)[0]
    Cfb = consensus_from_scores(Afb, False)[0]
    cv["Aggregate"] = (to_matrix(Cmc), to_matrix(Cfb))

    print(f"\n   {'method':<12}{'rho':>9}{'p one-sided':>14}{'p two-sided':>14}")
    for label, (M1, M2) in cv.items():
        rho, p1, p2, null = exact_rho(M1, M2)
        print(f"   {label:<12}{rho:>+9.3f}{p1:>14.3f}{p2:>14.3f}"
              f"   (null {null.min():+.2f} to {null.max():+.2f})")
        out[f"crossview {label}"] = {"rho": rho, "p_one": p1, "p_two": p2,
                                     "null_range": [float(null.min()), float(null.max())]}

    # ---------------- method comparison, cross-view rho --------------------
    print()
    print("   comparing the methods' cross-view correlations")
    boot = {}
    for label, (M1, M2) in cv.items():
        boot[label] = session_bootstrap(M1, M2)
        lo, hi = np.percentile(boot[label], [2.5, 97.5])
        print(f"      {label:<12} bootstrap 95% CI for its own rho: "
              f"[{lo:+.3f}, {hi:+.3f}]")

    for other in ["DTW", "Aggregate"]:
        diff = boot["GMM-UBM"] - boot[other]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        frac = float(np.mean(diff > 0))
        print(f"      GMM-UBM minus {other:<10} rho difference {diff.mean():+.3f}, "
              f"bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}], "
              f"P(difference > 0) = {frac:.3f}")
        out[f"rho difference GMM minus {other}"] = {
            "mean": float(diff.mean()), "ci": [float(lo), float(hi)],
            "p_gt_zero": frac}

    print("\n   NOTE. Permuting session labels sends every method's correlation to zero")
    print("   at once, so it cannot isolate a difference between two methods. The")
    print("   bootstrap above resamples the six sessions and is reported as approximate:")
    print("   with six units the intervals are wide by construction.")

    (DEST / "method_tests.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {DEST / 'method_tests.json'}")
