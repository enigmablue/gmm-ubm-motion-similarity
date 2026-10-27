"""Cross-view agreement, with inference that matches the sampling structure.

TWO PROBLEMS WITH THE OLD NUMBERS
---------------------------------
1. They were computed on the one-hand-per-frame fullbody features, i.e. before
   the hand-identity correction. They are not the canonical result.
2. The p-values treated the 15 recording pairs as 15 independent observations.
   They are not independent: each of the six sessions appears in five of the
   fifteen pairs. Any test that assumes independence is wrong here.

THE FIX
-------
Represent each view as a 6x6 session-similarity matrix, correlate the 15
upper-triangle entries, and get the p-value by permuting the six session labels
of one matrix over all 6! = 720 possible permutations. That is an exact test of
whether the two views order the sessions in a related way. The effect size is
still Spearman rho; only the inference changes.

Also reconciles the two aggregate consensus values quoted in the report
(C = .946 in section 3.2 and C = .893 in section 4.2). They differ because they
consensus over different domain counts, not because of numerical drift.

Usage:
    python crossview_exact.py
"""
import json
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from aggregate_seven import (domain_rank, hand_stats_fullbody,  # noqa: E402
                             hand_stats_maincam, head_stats, mouth_stats)
from canonical import consensus_scores  # noqa: E402
from table_all_methods import head_stats_solo_std  # noqa: E402

OUT = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output")
DEST = OUT / "paper_review" / "final"
vids = ["1", "2", "3", "4", "5", "6"]
PEOPLE = [("stella", "Stella"), ("aru", "Aru")]
PAIRS = [(a, b) for a, b in combinations(vids, 2)]
HAND_KEYS = ["CurlMean", "CurlStd", "CurlP25"]
HEAD_KEYS = ["HeadGest", "YawPitchRatio", "HeadDet"]


# ===================== EXACT MATRIX PERMUTATION TEST =======================
def to_matrix(scores):
    """{'(i,j)': value} -> 6x6 symmetric array indexed by session position."""
    idx = {v: i for i, v in enumerate(vids)}
    M = np.zeros((6, 6))
    for key, val in scores.items():
        a, b = key.strip("()").split(",")
        i, j = idx[a], idx[b]
        M[i, j] = M[j, i] = float(val)
    return M


def matrix_corr(M1, M2):
    iu = np.triu_indices(6, 1)
    return stats.spearmanr(M1[iu], M2[iu])[0]


def exact_matrix_test(s1, s2, label):
    """Exact session-label permutation p-value for a 6x6 matrix correlation."""
    M1, M2 = to_matrix(s1), to_matrix(s2)
    obs = matrix_corr(M1, M2)
    iu = np.triu_indices(6, 1)
    v1 = M1[iu]

    null = []
    for perm in permutations(range(6)):
        P = np.zeros((6, 6))
        for i, p in enumerate(perm):
            P[i, p] = 1.0
        Mp = P @ M2 @ P.T                      # relabel one matrix
        rho = stats.spearmanr(v1, Mp[iu])[0]
        if np.isfinite(rho):
            null.append(rho)
    null = np.array(null)

    tau, p_tau_naive = stats.kendalltau(v1, M2[iu])
    rho_naive, p_rho_naive = stats.spearmanr(v1, M2[iu])
    p_two = float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
    p_one = float(np.mean(null >= obs - 1e-12))

    # how far the observed value sits in the permutation distribution
    pct = float(np.mean(null < obs) * 100)

    print(f"\n{label}")
    print(f"   Spearman rho = {rho_naive:+.3f}   Kendall tau = {tau:+.3f}")
    print(f"   naive p (n=15 as if independent): rho {p_rho_naive:.3f}, "
          f"tau {p_tau_naive:.3f}   [NOT VALID]")
    print(f"   exact p over all {len(null)} session-label permutations: "
          f"two-sided {p_two:.3f}, one-sided {p_one:.3f}")
    print(f"   observed rho is at the {pct:.1f}th percentile of the null "
          f"(null range {null.min():+.2f} to {null.max():+.2f})")
    return {"rho": float(obs), "tau": float(tau), "p_exact_two": p_two,
            "p_exact_one": p_one, "p_naive": float(p_rho_naive),
            "percentile": pct}


def overlaps(s1, s2):
    o1 = sorted(s1, key=lambda k: s1[k], reverse=True)
    o2 = sorted(s2, key=lambda k: s2[k], reverse=True)
    return {k: len(set(o1[:k]) & set(o2[:k])) for k in (1, 3, 5, 10)}


# ===================== AGGREGATE DOMAINS ==================================
def agg_domains(view, mouths):
    out = {}
    hstats = ({p: {v: head_stats(v, p, "maincam") for v in vids} for p, _ in PEOPLE}
              if view == "maincam" else
              {p: {v: head_stats_solo_std(v, p) for v in vids} for p, _ in PEOPLE})
    gstats = ({p: {v: hand_stats_maincam(v, p) for v in vids} for p, _ in PEOPLE}
              if view == "maincam" else
              {p: {v: hand_stats_fullbody(v, p) for v in vids} for p, _ in PEOPLE})
    for person, name in PEOPLE:
        out[f"Head_{name}"] = domain_rank(hstats[person], HEAD_KEYS)[0]
        out[f"Hand_{name}"] = domain_rank(gstats[person], HAND_KEYS)[0]
        out[f"Mouth_{name}"] = domain_rank(
            {v: {"talk_pct": mouths[person][v]} for v in vids if v in mouths[person]},
            ["talk_pct"])[0]
    return out


if __name__ == "__main__":
    canon = json.loads((DEST / "canonical.json").read_text())
    dmc = canon["_pair_scores"]["gmm_maincam_domains"]
    dfb = canon["_pair_scores"]["gmm_fullbody_domains"]
    Cmc = consensus_scores(dmc)
    Cfb = consensus_scores(dfb)

    out = {}

    print("=" * 78)
    print("CROSS-VIEW AGREEMENT, CANONICAL REPRESENTATION (two-channel hands)")
    print("=" * 78)
    out["gmm"] = exact_matrix_test(Cmc, Cfb, "GMM-UBM, maincam vs fullbody (consensus)")
    ov = overlaps(Cmc, Cfb)
    print("   top-k overlap: " + "  ".join(f"k={k}: {v}/{k}" for k, v in ov.items()))
    out["gmm"]["overlap"] = ov

    print("\n   per-domain (same domain, both views):")
    out["per_domain"] = {}
    for d in ["Hand_Stella", "Hand_Aru", "Head_Stella", "Head_Aru"]:
        r = exact_matrix_test({f"({a},{b})": float(1.0 - (dmc[d][f"({a},{b})"] - 1) / 14.0)
                               for a, b in PAIRS},
                              {f"({a},{b})": float(1.0 - (dfb[d][f"({a},{b})"] - 1) / 14.0)
                               for a, b in PAIRS}, f"   {d}")
        out["per_domain"][d] = r

    print("\n   all 15 pairs, ordered by maincam consensus:")
    omc = sorted(Cmc, key=lambda k: Cmc[k], reverse=True)
    ofb = sorted(Cfb, key=lambda k: Cfb[k], reverse=True)
    diffs = []
    print(f"   {'pair':<9}{'maincam C':>11}{'rank':>6}{'fullbody C':>12}{'rank':>6}{'shift':>8}")
    for i, p in enumerate(omc, 1):
        j = ofb.index(p) + 1
        diffs.append(abs(i - j))
        print(f"   {p:<9}{Cmc[p]:>11.3f}{i:>6}{Cfb[p]:>12.3f}{j:>6}{abs(i - j):>8}")
    print(f"   mean absolute rank shift: {np.mean(diffs):.2f} (max {max(diffs)})")
    out["mean_rank_shift"] = float(np.mean(diffs))

    # ---- aggregate ----
    print()
    print("=" * 78)
    print("AGGREGATE BASELINE")
    print("=" * 78)
    m_mc, m_fb = mouth_stats()
    Amc_dom = agg_domains("maincam", m_mc)
    Afb_dom = agg_domains("fullbody", m_fb)

    for label, dom, keep6 in [("maincam", Amc_dom, True), ("fullbody", Afb_dom, True)]:
        four = {k: v for k, v in dom.items() if not k.startswith("Mouth")}
        C4 = consensus_scores(four)
        C6 = consensus_scores(dom)
        r4 = sorted(C4, key=lambda k: C4[k], reverse=True).index("(4,5)") + 1
        r6 = sorted(C6, key=lambda k: C6[k], reverse=True).index("(4,5)") + 1
        print(f"\n   {label}: 4-domain consensus (head+hands) rank {r4}/15, C = {C4['(4,5)']:.3f}")
        print(f"   {label}: 6-domain consensus (+mouth)      rank {r6}/15, C = {C6['(4,5)']:.3f}")
        out[f"agg_{label}_C4"] = {"rank": r4, "C": C4["(4,5)"]}
        out[f"agg_{label}_C6"] = {"rank": r6, "C": C6["(4,5)"]}

    Amc4 = consensus_scores({k: v for k, v in Amc_dom.items() if not k.startswith("Mouth")})
    Afb4 = consensus_scores({k: v for k, v in Afb_dom.items() if not k.startswith("Mouth")})
    out["agg"] = exact_matrix_test(Amc4, Afb4,
                                   "Aggregate (4 domains), maincam vs fullbody")

    (DEST / "crossview_exact.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {DEST / 'crossview_exact.json'}")
