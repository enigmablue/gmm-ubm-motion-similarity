"""Re-run the canonical scoring and keep the RAW scores, not just the ranks.

WHY
---
The canonical pipeline stores only 1-based integer ranks per domain. Ranks lose
the information needed to handle ties: two pairs with identical LLRs are given
different ranks by whatever order the sort happened to produce. Every "rank of
(4,5)" in the report therefore carries an unquantified tie ambiguity.

This script runs exactly the same fits as `canonical.py` -- same scaler, same
BIC K selection, same self-consistency r, same four-block cross-fitting -- but
returns the score dict. Downstream scripts can then use average ranks, which is
what the reported chance reference (mean rank 8.0 of 15) assumes.

Outputs (under output/paper_review/final/):
    raw_gmm.json         view -> domain -> pair -> mean LLR
    raw_dtw.json         view -> domain -> pair -> normalised path cost
    retrieval_tables.json   full gallery rank tables for the retrieval test

Nothing here changes a modelling choice. It only stops throwing away numbers.

Usage:
    python review2_raw_scores.py
"""
import copy
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from fastdtw import fastdtw
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from canonical import (PEOPLE, canonical_hands, canonical_heads,  # noqa: E402
                       hand_channels)
from session_retrieval import gmm_cell  # noqa: E402

OUT = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output")
DEST = OUT / "paper_review" / "final"
vids = ["1", "2", "3", "4", "5", "6"]
PAIRS = [(a, b) for a, b in combinations(vids, 2)]
R_CANDIDATES = [4, 8, 16, 32, 64, 128]


def _sab(X):
    """Alternating 50-frame A/B blocks, 50 on / 50 off."""
    n = len(X)
    ma = np.zeros(n, dtype=bool)
    for i in range(0, n, 100):
        ma[i:min(i + 50, n)] = True
    return X[ma], X[~ma]


# ===================== GMM-UBM, SCORES KEPT ================================
def gmm_scores(vid_to_frames):
    """Identical to cell_extractor.run_gmm_domain, returning scores."""
    if len(vid_to_frames) < 3:
        return None, None, None
    all_raw = np.vstack(list(vid_to_frames.values()))
    sc = RobustScaler().fit(all_raw)
    scaled = {v: sc.transform(f) for v, f in vid_to_frames.items()}
    all_X = np.vstack(list(scaled.values()))

    best_k, best_bic = 2, np.inf
    for k in range(2, 7):
        g = GaussianMixture(n_components=k, covariance_type="diag",
                            random_state=42, max_iter=200, reg_covar=1e-4, n_init=3)
        g.fit(all_X)
        if g.weights_.min() < 0.01:
            continue
        if g.bic(all_X) < best_bic:
            best_bic, best_k = g.bic(all_X), k

    ubm = GaussianMixture(n_components=best_k, covariance_type="diag",
                          random_state=42, max_iter=300, reg_covar=1e-4)
    ubm.fit(all_X)

    best_r, best_self = 16, -np.inf
    for r_test in R_CANDIDATES:
        ss = []
        for v in vids:
            if v not in scaled:
                continue
            Xa, Xb = _sab(scaled[v])
            if len(Xa) < 20 or len(Xb) < 20:
                continue
            resp = ubm.predict_proba(Xa)
            Nk = resp.sum(axis=0)
            xbar = (resp.T @ Xa) / (Nk[:, None] + 1e-12)
            ad = copy.deepcopy(ubm)
            a = Nk / (Nk + r_test)
            ad.means_ = a[:, None] * xbar + (1 - a[:, None]) * ubm.means_
            ss.append((ad.score_samples(Xb).sum()
                       - ubm.score_samples(Xb).sum()) / len(Xb))
        if ss and np.mean(ss) > best_self:
            best_self, best_r = float(np.mean(ss)), r_test

    scores = {}
    for v1, v2 in PAIRS:
        if v1 not in scaled or v2 not in scaled:
            continue
        X1a, X1b = _sab(scaled[v1])
        X2a, X2b = _sab(scaled[v2])
        llrs = []
        for af, so in [(X1a, X2b), (X2a, X1b), (X1b, X2a), (X2b, X1a)]:
            if len(af) < 10 or len(so) < 10:
                continue
            resp = ubm.predict_proba(af)
            Nk = resp.sum(axis=0)
            xbar = (resp.T @ af) / (Nk[:, None] + 1e-12)
            ad = copy.deepcopy(ubm)
            a = Nk / (Nk + best_r)
            ad.means_ = a[:, None] * xbar + (1 - a[:, None]) * ubm.means_
            llrs.append((ad.score_samples(so).sum()
                         - ubm.score_samples(so).sum()) / len(so))
        scores[f"({v1},{v2})"] = float(np.mean(llrs)) if llrs else float("-inf")
    return scores, int(best_k), int(best_r)


# ===================== DTW, SCORES KEPT ====================================
def dtw_scores_single(vid_to_seqs):
    """Single-sequence DTW (a head channel, or a pooled hand channel)."""
    if len(vid_to_seqs) < 3:
        return None
    sc = RobustScaler().fit(np.vstack(list(vid_to_seqs.values())))
    scaled = {v: sc.transform(s) for v, s in vid_to_seqs.items()}
    out = {}
    for v1, v2 in PAIRS:
        if v1 not in scaled or v2 not in scaled:
            continue
        d, path = fastdtw(scaled[v1], scaled[v2], dist=2)
        out[f"({v1},{v2})"] = float(d / len(path))
    return out


def dtw_scores_hands(person, view):
    """Per-hand-channel DTW, averaged over the channels present in both."""
    chans = hand_channels(view, person)
    scaled = {}
    for h in ["left", "right"]:
        d = chans[h]
        if len(d) >= 3:
            sc = RobustScaler().fit(np.vstack(list(d.values())))
            scaled[h] = {v: sc.transform(f) for v, f in d.items()}
    out = {}
    for v1, v2 in PAIRS:
        ds = []
        for h in ["left", "right"]:
            if h in scaled and v1 in scaled[h] and v2 in scaled[h]:
                d, path = fastdtw(scaled[h][v1], scaled[h][v2], dist=2)
                ds.append(d / len(path))
        if ds:
            out[f"({v1},{v2})"] = float(np.mean(ds))
    return out or None


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    gmm, dtw = {}, {}

    for view in ["maincam", "fullbody"]:
        heads = canonical_heads(view)
        hands = canonical_hands(view)
        gmm[view], dtw[view] = {}, {}
        for person, name in PEOPLE:
            for kind in ["Hand", "Head"]:
                label = f"{kind}_{name}"
                feats = hands[person] if kind == "Hand" else heads[person]
                if kind == "Hand":
                    sc, K, r = gmm_scores(feats)
                    ds = dtw_scores_hands(person, view)
                else:
                    sc, K, r = gmm_scores(feats)
                    ds = dtw_scores_single(feats)
                gmm[view][label] = sc
                dtw[view][label] = ds
                print(f"   {view:<9}{label:<14} GMM K={K} r={r}  "
                      f"DTW {'ok' if ds else 'n/a'}")

    (DEST / "raw_gmm.json").write_text(json.dumps(gmm, indent=2))
    (DEST / "raw_dtw.json").write_text(json.dumps(dtw, indent=2))
    print(f"\nwrote {DEST / 'raw_gmm.json'}")
    print(f"wrote {DEST / 'raw_dtw.json'}")

    # ---- retrieval gallery tables (full rank tables, not just the diagonal) ----
    tables = {}
    for person, name in PEOPLE:
        q_ch = hand_channels("maincam", person)
        g_ch = hand_channels("fullbody", person)
        acc = {}
        for h in ["left", "right"]:
            q, g = q_ch[h], g_ch[h]
            if len(q) < 3 or len(g) < 3:
                continue
            s, band, K, r = gmm_cell(q, g)
            if s is None:
                continue
            for k, v in s.items():
                acc.setdefault(k, []).append(v)
        hand_scores = {k: float(np.mean(v)) for k, v in acc.items()}
        hq = canonical_heads("maincam")[person]
        hg = canonical_heads("fullbody")[person]
        head_scores, band, K, r = gmm_cell(hq, hg)

        for kind, s in [("Hand", hand_scores), ("Head", head_scores)]:
            tab = {}
            for qi in vids:
                order = sorted(vids,
                               key=lambda g: s.get((f"mc{qi}", f"solo{g}"), -999.0),
                               reverse=True)
                tab[qi] = {g: i + 1 for i, g in enumerate(order)}
            tables[f"{kind}_{name}"] = tab
            print(f"   retrieval {kind}_{name}: self-ranks "
                  + ", ".join(f"{qi}:{tab[qi][qi]}" for qi in vids))

    (DEST / "retrieval_tables.json").write_text(json.dumps(tables, indent=2))
    print(f"wrote {DEST / 'retrieval_tables.json'}")


if __name__ == "__main__":
    main()
