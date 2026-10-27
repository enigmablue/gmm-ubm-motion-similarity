"""Statistics requested by the second review: tie handling, influence, multiplicity,
and a valid inference for the retrieval test.

WHAT THIS FIXES
---------------
1. TIES. Ranks were integers from a sort, so tied scores got different ranks by
   accident of pair order. Everything here uses AVERAGE RANKS instead: a pair's
   rank is 1 + (number strictly better) + 0.5 * (number equal). That is the rule
   Spearman assumes and the only one under which the chance reference (mean rank
   8.0 of 15) is exactly true.

2. INFLUENCE. rho = .567 on 15 pairwise edges, but the six sessions are the real
   units. Leave-one-session-out removes a session and its five edges and reports
   the correlation on the remaining ten. If one session carries the association,
   the six values will show it.

3. MULTIPLICITY. Five GMM cross-view tests were run (consensus + four domains).
   Benjamini-Hochberg across that family.

4. RETRIEVAL INFERENCE. 5/24 and 2/12 were tested with a binomial. The trials are
   not independent -- six queries share one gallery, four domains share the same
   recordings, both speakers come from the same conversations. Instead the six
   session labels are jointly permuted over all 6! = 720 arrangements and the
   total hit count is compared with the exact null distribution.

Usage:
    python review2_checks.py
"""
import json
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from aggregate_seven import domain_rank, hand_stats_fullbody, mouth_stats  # noqa: E402
from aggregate_seven import hand_stats_maincam, head_stats  # noqa: E402
from crossview_exact import agg_domains  # noqa: E402
from table_all_methods import head_stats_solo_std  # noqa: E402

OUT = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output")
DEST = OUT / "paper_review" / "final"
vids = ["1", "2", "3", "4", "5", "6"]
PEOPLE = [("stella", "Stella"), ("aru", "Aru")]
PAIRS = [f"({a},{b})" for a, b in combinations(vids, 2)]
TARGET = "(4,5)"
DOMAINS = ["Hand_Stella", "Head_Stella", "Hand_Aru", "Head_Aru"]


# ===================== AVERAGE RANKS ======================================
def avg_ranks(scores):
    """{key: score} -> {key: 1-based average rank}, 1 = best. Direction-free:
    pass scores already oriented so that LARGER is better, or use avg_ranks_rev."""
    keys = list(scores)
    vals = np.array([scores[k] for k in keys], dtype=float)
    out = {}
    for i, k in enumerate(keys):
        better = np.sum(vals > vals[i] + 1e-12)
        equal = np.sum(np.abs(vals - vals[i]) <= 1e-12) - 1
        out[k] = 1.0 + better + 0.5 * equal
    return out


def consensus_from_scores(domain_scores, higher_is_better=True):
    """Average-rank normalisation per domain, then mean over domains."""
    R = {}
    for d, s in domain_scores.items():
        r = avg_ranks(s if higher_is_better else {k: -v for k, v in s.items()})
        R[d] = r
    return {p: float(np.mean([1.0 - (R[d][p] - 1) / 14.0 for d in R])) for p in PAIRS}, R


def report_cell(label, C):
    r = avg_ranks(C)
    print(f"   {label:<34} rank {r[TARGET]:>5.2f}/15   C = {C[TARGET]:.3f}")
    return r[TARGET], C[TARGET], r


# ===================== EXACT SESSION-LABEL PERMUTATION =====================
def to_matrix(C):
    idx = {v: i for i, v in enumerate(vids)}
    M = np.zeros((6, 6))
    for key, val in C.items():
        a, b = key.strip("()").split(",")
        i, j = idx[a], idx[b]
        M[i, j] = M[j, i] = float(val)
    return M


def perm_null(M1, M2, keep=None):
    """Null distribution of the edge correlation, permuting session labels of M2.

    `keep` restricts to a subset of session indices (leave-one-session-out)."""
    iu = np.triu_indices(6, 1)
    if keep is not None:
        mask = np.array([(i in keep and j in keep) for i, j in zip(*iu)])
    else:
        mask = np.ones(len(iu[0]), dtype=bool)
    v1 = M1[iu][mask]
    out = []
    for perm in permutations(range(6)):
        P = np.zeros((6, 6))
        for i, p in enumerate(perm):
            P[i, p] = 1.0
        Mp = P @ M2 @ P.T
        r = stats.spearmanr(v1, Mp[iu][mask])[0]
        if np.isfinite(r):
            out.append(r)
    return np.array(out), v1, M2[iu][mask]


def exact_test(M1, M2, label, keep=None, verbose=True):
    null, v1, v2 = perm_null(M1, M2, keep)
    obs = stats.spearmanr(v1, v2)[0]
    p2 = float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
    p1 = float(np.mean(null >= obs - 1e-12))
    if verbose:
        n_edge = len(v1)
        print(f"   {label:<34} rho {obs:+.3f}  n_edges {n_edge:>3}  "
              f"p_exact(two) {p2:.3f}  p_one {p1:.3f}")
    return {"rho": float(obs), "p_two": p2, "p_one": p1}


def bh(pvals, family_name):
    """Benjamini-Hochberg. Returns adjusted p in the input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        val = min(prev, pvals[i] * m / rank)
        adj[i] = val
        prev = val
    print(f"\n   BH adjustment, family = {family_name} (m = {m})")
    for i, p in enumerate(pvals):
        print(f"      p = {p:.3f}  ->  p_BH = {adj[i]:.3f}")
    return adj


# ===================== MAIN ================================================
if __name__ == "__main__":
    out = {}
    rawg = json.loads((DEST / "raw_gmm.json").read_text())
    rawd = json.loads((DEST / "raw_dtw.json").read_text())

    print("=" * 78)
    print("1. RANKS OF (4,5) UNDER AVERAGE RANKS (ties shared)")
    print("=" * 78)

    cells = {}
    for view in ["maincam", "fullbody"]:
        C, _ = consensus_from_scores(rawg[view], higher_is_better=True)
        cells[f"GMM|{view}"] = C
        C, _ = consensus_from_scores(rawd[view], higher_is_better=False)
        cells[f"DTW|{view}"] = C

    m_mc, m_fb = mouth_stats()
    Amc = agg_domains("maincam", m_mc)
    Afb = agg_domains("fullbody", m_fb)

    def agg_rank_scores(dom, with_mouth):
        keep = dom if with_mouth else {k: v for k, v in dom.items()
                                       if not k.startswith("Mouth")}
        return consensus_from_scores(keep, higher_is_better=False)[0]

    cells["Aggregate6|maincam"] = agg_rank_scores(Amc, True)
    cells["Aggregate6|fullbody"] = agg_rank_scores(Afb, True)
    cells["Aggregate4|maincam"] = agg_rank_scores(Amc, False)
    cells["Aggregate4|fullbody"] = agg_rank_scores(Afb, False)

    out["avg_rank_45"] = {}
    for label, C in cells.items():
        r, c, _ = report_cell(label, C)
        out["avg_rank_45"][label] = {"rank": r, "C": c}

    # which cells contain a tie at or across (4,5)
    print("\n   ties involving (4,5):")
    for label, C in cells.items():
        tied = [p for p in PAIRS if p != TARGET
                and abs(C[p] - C[TARGET]) <= 1e-12]
        if tied:
            print(f"      {label:<34} tied with {', '.join(tied)} at C = {C[TARGET]:.3f}")

    # ===================== 2. INFLUENCE ==================================
    print()
    print("=" * 78)
    print("2. LEAVE-ONE-SESSION-OUT: is the cross-view association carried by one session?")
    print("=" * 78)
    M_mc, M_fb = to_matrix(cells["GMM|maincam"]), to_matrix(cells["GMM|fullbody"])
    full = exact_test(M_mc, M_fb, "all six sessions")
    out["crossview_full"] = full

    print("\n   leaving each session out in turn:")
    loo = {}
    for s in range(6):
        keep = [i for i in range(6) if i != s]
        r = exact_test(M_mc, M_fb, f"without session {vids[s]}", keep=keep)
        loo[vids[s]] = r
    out["crossview_loo"] = loo
    vals = [loo[v]["rho"] for v in vids]
    print(f"\n   six leave-one-out rho values: "
          + ", ".join(f"{v:+.3f}" for v in vals))
    print(f"   full rho {full['rho']:+.3f}; LOO range {min(vals):+.3f} to {max(vals):+.3f}")

    # per-domain, same treatment
    print("\n   per-domain cross-view tests, full and leave-one-out:")
    out["crossview_domains"] = {}
    p_vals = []
    for d in DOMAINS:
        s_mc = {p: rawg["maincam"][d][p] for p in PAIRS}
        s_fb = {p: rawg["fullbody"][d][p] for p in PAIRS}
        C_mc, _ = consensus_from_scores({d: s_mc})
        C_fb, _ = consensus_from_scores({d: s_fb})
        M1, M2 = to_matrix(C_mc), to_matrix(C_fb)
        r = exact_test(M1, M2, f"{d}")
        rl = {}
        for s in range(6):
            keep = [i for i in range(6) if i != s]
            rl[vids[s]] = exact_test(M1, M2, f"   {d} without {vids[s]}",
                                     keep=keep, verbose=False)["rho"]
        r["loo"] = rl
        out["crossview_domains"][d] = r
        p_vals.append(r["p_two"])

    # ===================== 3. MULTIPLICITY ===============================
    print()
    print("=" * 78)
    print("3. MULTIPLICITY ACROSS THE FIVE GMM CROSS-VIEW TESTS")
    print("=" * 78)
    labels = ["consensus"] + DOMAINS
    pvals = [full["p_two"]] + p_vals
    adj = bh(pvals, "five GMM cross-view tests")
    out["bh_five"] = {lab: {"p": float(p), "p_bh": float(a)}
                      for lab, p, a in zip(labels, pvals, adj)}

    # aggregate as a sixth, separate, exploratory comparison
    Amc4, _ = consensus_from_scores({k: v for k, v in Amc.items()
                                     if not k.startswith("Mouth")},
                                   higher_is_better=False)
    Afb4, _ = consensus_from_scores({k: v for k, v in Afb.items()
                                     if not k.startswith("Mouth")},
                                    higher_is_better=False)
    agg_test = exact_test(to_matrix(Amc4), to_matrix(Afb4),
                          "aggregate, 4 domains (separate)")
    out["crossview_aggregate"] = agg_test
    adj6 = bh(pvals + [agg_test["p_two"]], "five GMM tests + aggregate")
    out["bh_six"] = {"agg_p": agg_test["p_two"], "agg_p_bh": float(adj6[-1])}

    # ===================== 4. RETRIEVAL, EXACT ===========================
    print()
    print("=" * 78)
    print("4. RETRIEVAL TEST WITH AN EXACT SESSION-LABEL PERMUTATION")
    print("=" * 78)
    tabs = json.loads((DEST / "retrieval_tables.json").read_text())

    def cons_tables():
        """Per-speaker consensus rank table: mean rank-normalised Hand + Head."""
        ct = {}
        for _, name in PEOPLE:
            tab = {}
            for q in vids:
                mean = {g: float(np.mean([
                    1.0 - (tabs[f"{k}_{name}"][q][g] - 1) / 5.0
                    for k in ["Hand", "Head"]])) for g in vids}
                order = sorted(vids, key=lambda g: mean[g], reverse=True)
                tab[q] = {g: i + 1 for i, g in enumerate(order)}
            ct[name] = tab
        return ct

    ctab = cons_tables()

    obs_dom = sum(1 for d in DOMAINS for q in vids
                  if tabs[d][q][q] == 1)
    obs_con = sum(1 for name in ctab for q in vids if ctab[name][q][q] == 1)
    print(f"   observed: {obs_dom}/24 domain hits, {obs_con}/12 consensus hits")

    def hits(perm):
        """perm maps query session index -> gallery session index (a bijection)."""
        pd = {vids[i]: vids[perm[i]] for i in range(6)}
        dom = sum(1 for d in DOMAINS for q in vids if tabs[d][q][pd[q]] == 1)
        con = sum(1 for name in ctab for q in vids if ctab[name][q][pd[q]] == 1)
        return dom, con

    null_dom, null_con = [], []
    for perm in permutations(range(6)):
        a, b = hits(perm)
        null_dom.append(a)
        null_con.append(b)
    null_dom, null_con = np.array(null_dom), np.array(null_con)

    def tail(null, obs):
        return float(np.mean(null >= obs))

    p_dom = tail(null_dom, obs_dom)
    p_con = tail(null_con, obs_con)
    print(f"   exact permutation p, domain total:    {p_dom:.3f}"
          f"   (null mean {null_dom.mean():.2f}, range {null_dom.min()}-{null_dom.max()})")
    print(f"   exact permutation p, consensus total: {p_con:.3f}"
          f"   (null mean {null_con.mean():.2f}, range {null_con.min()}-{null_con.max()})")
    print(f"   the two statistics move together: corr = "
          f"{np.corrcoef(null_dom, null_con)[0, 1]:.2f}")

    from session_retrieval import binom_tail  # noqa: E402
    print(f"   for comparison, the invalid binomial values: "
          f"domain {binom_tail(obs_dom, 24, 1 / 6):.3f}, "
          f"consensus {binom_tail(obs_con, 12, 1 / 6):.3f}")

    out["retrieval_exact"] = {
        "domain_hits": obs_dom, "consensus_hits": obs_con,
        "p_domain": p_dom, "p_consensus": p_con,
        "null_domain_mean": float(null_dom.mean()),
        "null_consensus_mean": float(null_con.mean()),
        "p_domain_binomial_invalid": float(binom_tail(obs_dom, 24, 1 / 6)),
        "p_consensus_binomial_invalid": float(binom_tail(obs_con, 12, 1 / 6)),
    }
    out["consensus_tables"] = ctab

    (DEST / "review2_checks.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {DEST / 'review2_checks.json'}")
