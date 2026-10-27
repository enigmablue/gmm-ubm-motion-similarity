"""Session-matched retrieval test (12 known-answer trials).

GROUND TRUTH
------------
`maincam i`, `stella i` and `aru i` are the same person in the same recording
session. Verified two ways: embedded recording times agree to the minute, and
durations agree to within a few seconds (`probe_videos.py`). Six sessions were
recorded back-to-back on 31 July 2026.

TASK
----
For each speaker and each session i, score the maincam recording of that speaker
against all six solo recordings of the same speaker. The trial is a hit if the
solo recording from session i scores highest.

    6 sessions x 2 speakers = 12 trials.  Chance = 1/6 (expected 2/12).

This target does not depend on the aggregate reference standard or on any
measure designed for this paper: it is fixed by when the recordings were made.

Domains: Hand_Stella, Hand_Aru, Head_Stella, Head_Aru, plus a consensus over the
two domains belonging to each speaker.

Usage:
    python session_retrieval.py            # GMM-UBM only (fast)
    python session_retrieval.py --dtw      # also run DTW comparators
"""
import copy
import csv
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

import warnings
warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from handview_2x2 import load_hand_raw, hand_features, SUBSAMPLE  # noqa: E402

MC = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output")
vids = ["1", "2", "3", "4", "5", "6"]
EXCURSION_LAG = 3
R_CANDIDATES = [4, 8, 16, 32, 64, 128]


# ===================== HEAD 18D (maincam and solo alike) =====================
def _head_18d(rows, fp):
    """18D head features: angle(3) vel(3) |vel|(3) acc(3) speed accel still exc(3)."""
    records = []
    for r in rows:
        fi = int(r["frame"])
        if fi % SUBSAMPLE != 0:
            continue
        if r.get(f"{fp}_detected") != "1":
            continue
        ts = float(r.get("timestamp_sec") or fi / 30.0)
        angles = np.array([float(r.get(f"{fp}_yaw") or 0),
                           float(r.get(f"{fp}_pitch") or 0),
                           float(r.get(f"{fp}_roll") or 0)])
        records.append((fi, ts, angles))
    if len(records) < 30:
        return None

    frames, hist = [], []
    pa = pt = pvel = pvel_mag = None
    for fi, ts, angles in records:
        dt = ts - pt if pt else 0.033
        if pa is not None and dt > 1e-6:
            vel = ((angles - pa + 180.0) % 360.0 - 180.0) / dt
        else:
            vel = np.zeros(3)
        vm = float(np.linalg.norm(vel))
        if pvel is not None and dt > 1e-6:
            accel = (vel - pvel) / dt
            ang_accel = (vm - pvel_mag) / dt
        else:
            accel = np.zeros(3)
            ang_accel = 0.0
        hist.append(angles)
        if len(hist) > EXCURSION_LAG:
            exc = (angles - hist[-EXCURSION_LAG - 1] + 180.0) % 360.0 - 180.0
        else:
            exc = np.zeros(3)
        frames.append(np.concatenate([angles, vel, np.abs(vel), accel,
                                      [vm], [ang_accel], [0.0], exc]))
        pa, pt, pvel, pvel_mag = angles, ts, vel, vm
    return np.array(frames)


def head_source(vid, view, person):
    if view == "maincam":
        return MC / "maincam" / f"{vid}_features.csv", ("face1" if person == "stella" else "face2")
    if person == "stella":
        return MC / "stella" / f"{vid}_features.csv", "face1"
    return MC / "aru" / f"fb_aru_{int(vid):02d}_features.csv", "face1"


def build_head_set(view, person):
    out = {}
    for v in vids:
        path, fp = head_source(v, view, person)
        if not path.exists():
            continue
        f = _head_18d(list(csv.DictReader(open(path))), fp)
        if f is not None and len(f) > 50:
            out[v] = f
    if not out:
        return out
    gm = np.median(np.concatenate([f[:, 12] for f in out.values()]))
    for v in out:
        out[v] = out[v].copy()
        out[v][:, 14] = (out[v][:, 12] < gm).astype(float)
    return out


# ===================== SAME-VIEW PARITY CHECKS =====================
def build_hand_set(view, person):
    cache = load_hand_raw(view, person)
    out = {}
    for v in vids:
        if v not in cache:
            continue
        rows, fr = cache[v]
        f = hand_features(rows, fr, min_speed=0)
        if f is not None:
            out[v] = f
    return out


def effective_dims(query, gallery):
    """Dimensions that actually vary across the pooled query+gallery set.

    Head_pitch/roll from a side view and from a frontal view are not the same
    measurement. Any dimension that is constant within both sets carries no
    information and only adds noise to the distance, so report it.
    """
    kept, dropped = [], []
    for d in range(list(query.values())[0].shape[1]):
        vals = np.concatenate([a[:, d] for a in query.values()] +
                              [a[:, d] for a in gallery.values()])
        (kept if np.std(vals) > 1e-9 else dropped).append(d)
    return kept, dropped


# ===================== GMM-UBM RETRIEVAL =====================
def _zscore(a):
    sd = a.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (a - a.mean(axis=0)) / sd


def _ab_blocks(X):
    n = len(X)
    mask = np.zeros(n, dtype=bool)
    for i in range(0, n, 100):
        mask[i:min(i + 50, n)] = True
    return X[mask], X[~mask]


def gmm_cell(query, gallery, center=False):
    """Cross-fitted symmetric LLR for every (maincam query, solo gallery) cell.

    Returns (scores, band, K, r) with scores[(q, g)] in nats/frame.
    If center=True each recording is standardised on its own mean/SD first, so
    the comparison uses only within-recording shape and the systematic offset
    between the two camera views is removed.
    """
    Q = {f"mc{v}": a for v, a in query.items()}
    G = {f"solo{v}": a for v, a in gallery.items()}
    data = {**Q, **G}
    if center:
        data = {k: _zscore(a) for k, a in data.items()}
    if len(Q) < 2 or len(G) < 2:
        return None, None, None, None

    all_raw = np.vstack(list(data.values()))
    sc = RobustScaler().fit(all_raw)
    scaled = {k: sc.transform(a) for k, a in data.items()}
    all_X = np.vstack(list(scaled.values()))

    best_k, best_bic = 2, np.inf
    for k in range(2, 7):
        g = GaussianMixture(n_components=k, covariance_type="diag", random_state=42,
                            max_iter=200, reg_covar=1e-4, n_init=3)
        g.fit(all_X)
        if g.weights_.min() < 0.01:
            continue
        if g.bic(all_X) < best_bic:
            best_bic, best_k = g.bic(all_X), k

    ubm = GaussianMixture(n_components=best_k, covariance_type="diag", random_state=42,
                          max_iter=300, reg_covar=1e-4)
    ubm.fit(all_X)

    def adapt(Xa, r):
        resp = ubm.predict_proba(Xa)
        Nk = resp.sum(axis=0)
        xbar = (resp.T @ Xa) / (Nk[:, None] + 1e-12)
        alpha = Nk / (Nk + r)
        ad = copy.deepcopy(ubm)
        ad.means_ = alpha[:, None] * xbar + (1 - alpha[:, None]) * ubm.means_
        return ad

    best_r, best_self = 16, -np.inf
    for r_test in R_CANDIDATES:
        ss = []
        for Xv in scaled.values():
            Xa, Xb = _ab_blocks(Xv)
            if len(Xa) < 20 or len(Xb) < 20:
                continue
            ad = adapt(Xa, r_test)
            ss.append((ad.score_samples(Xb).sum() - ubm.score_samples(Xb).sum()) / len(Xb))
        if ss and np.mean(ss) > best_self:
            best_self, best_r = float(np.mean(ss)), r_test

    blocks = {k: _ab_blocks(X) for k, X in scaled.items()}

    scores = {}
    for qk, (Qa, Qb) in blocks.items():
        for gk, (Ga, Gb) in blocks.items():
            llrs = []
            for af, so in [(Qa, Gb), (Ga, Qb), (Qb, Ga), (Gb, Qa)]:
                if len(af) < 10 or len(so) < 10:
                    continue
                ad = adapt(af, best_r)
                llrs.append((ad.score_samples(so).sum() - ubm.score_samples(so).sum()) / len(so))
            scores[(qk, gk)] = float(np.mean(llrs)) if llrs else -999.0

    # between-query band: how separable are the six queries at all?
    qmeans = {}
    for q in Q:
        vals = [scores[(q, g)] for g in G]
        qmeans[q] = float(np.mean(vals))
    band = float(np.std(list(qmeans.values())))

    return scores, band, best_k, best_r


def dtw_cell(query, gallery):
    from fastdtw import fastdtw
    data = {**{f"mc{v}": a for v, a in query.items()},
            **{f"solo{v}": a for v, a in gallery.items()}}
    sc = RobustScaler().fit(np.vstack(list(data.values())))
    scaled = {k: sc.transform(a) for k, a in data.items()}
    scores = {}
    for qk, Q in scaled.items():
        for gk, G in scaled.items():
            d, path = fastdtw(Q, G, dist=2)
            scores[(qk, gk)] = -d / len(path)      # negative so higher = better
    return scores


# ===================== TRIALS =====================
def trials_from_scores(scores, n_gallery):
    """{session: rank of the correct gallery entry} (1 = correct)."""
    out = {}
    for q in vids:
        qk = f"mc{q}"
        order = sorted((f"solo{g}" for g in vids),
                       key=lambda gk: scores.get((qk, gk), -999.0), reverse=True)
        out[q] = order.index(f"solo{q}") + 1
    return out


def rank_norm(rank, n):
    return 1.0 - (rank - 1) / (n - 1.0)


def binom_tail(hits, n, p):
    return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(hits, n + 1))


def main():
    use_dtw = "--dtw" in sys.argv
    center = "--centered" in sys.argv
    if center:
        print("Mode: per-recording standardisation (removes the camera-view offset).")

    print("Building features (query = maincam, gallery = same speaker's solo)...")
    sets = {}
    for person in ["stella", "aru"]:
        sets[("Hand", person)] = (build_hand_set("maincam", person),
                                  build_hand_set("fullbody", person))
        sets[("Head", person)] = (build_head_set("maincam", person),
                                  build_head_set("solo", person))

    print(f"\n{'domain':<14} {'frames maincam (1-6)':<26} {'frames solo (1-6)':<26}")
    print("-" * 70)
    for (kind, person), (q, g) in sets.items():
        qs = " ".join(str(len(q[v])) if v in q else "-" for v in vids)
        gs = " ".join(str(len(g[v])) if v in g else "-" for v in vids)
        print(f"{kind + '_' + person:<14} {qs:<26} {gs:<26}")

    raw, meta = {}, {}
    for key, (q, g) in sets.items():
        label = f"{key[0]}_{key[1].capitalize()}"
        if use_dtw:
            raw[("DTW", label)] = dtw_cell(q, g)
        sc, band, K, r = gmm_cell(q, g, center=center)
        if sc is None:
            print(f"{label}: insufficient data")
            continue
        raw[("GMM", label)] = sc
        meta[label] = (K, r, band)

    def rank_table(scores):
        """{session: {gallery_vid: 1-based rank}}."""
        out = {}
        for q in vids:
            order = sorted(vids, key=lambda gv: scores.get((f"mc{q}", f"solo{gv}"), -999.0),
                           reverse=True)
            out[q] = {gv: i + 1 for i, gv in enumerate(order)}
        return out

    results = {(m, lab): rank_table(sc) for (m, lab), sc in raw.items()}

    # ---------------- report ----------------
    order = ["Hand_Stella", "Hand_Aru", "Head_Stella", "Head_Aru"]
    method = "DTW" if use_dtw else "GMM"

    print("\n" + "=" * 78)
    print(f"RETRIEVAL: rank of the correct session for each trial  (1 = correct)")
    print("=" * 78)
    hdr = f"{'domain':<14}" + "".join(f"{'s' + v:>5}" for v in vids) + f"{'hits/6':>9}"
    print(hdr)
    print("-" * len(hdr))
    for label in order:
        if (method, label) not in results:
            continue
        tb = results[(method, label)]
        rr = {v: tb[v][v] for v in vids}
        row = "".join(f"{rr[v]:>5}" for v in vids)
        hits = sum(1 for v in vids if rr[v] == 1)
        print(f"{label:<14}{row}{hits:>9}")

    print("=" * 78)
    print("CONSENSUS (rank aggregation across the two domains per speaker)")
    print("=" * 78)
    hdr2 = f"{'speaker':<10}" + "".join(f"{'s' + v:>5}" for v in vids) + f"{'hits/6':>9}"
    print(hdr2)
    print("-" * len(hdr2))
    cons_tot_hits = cons_tot = 0
    for person, pname in [("stella", "Stella"), ("aru", "Aru")]:
        doms = [d for d in [f"Hand_{pname}", f"Head_{pname}"] if (method, d) in results]
        if len(doms) < 2:
            continue
        rr = {}
        for q in vids:
            agg = {gv: float(np.mean([rank_norm(results[(method, d)][q][gv], 6)
                                      for d in doms])) for gv in vids}
            order_ = sorted(vids, key=lambda gv: agg[gv], reverse=True)
            rr[q] = order_.index(q) + 1
        row = "".join(f"{rr[v]:>5}" for v in vids)
        hits = sum(1 for v in vids if rr[v] == 1)
        cons_tot_hits += hits
        cons_tot += len(vids)
        print(f"{pname:<10}{row}{hits:>9}")

    # ---------------- summary ----------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    tot_hits = tot = 0
    for label in order:
        if (method, label) not in results:
            continue
        tb = results[(method, label)]
        hits = sum(1 for v in vids if tb[v][v] == 1)
        tot_hits += hits
        tot += len(vids)
        p_bin = binom_tail(hits, len(vids), 1 / 6)
        if label in meta:
            K, r, band = meta[label]
            print(f"{label:<14} {hits}/6 correct   (chance 1/6, binomial p = {p_bin:.4f})   "
                  f"K={K}, r={r}")
        else:
            print(f"{label:<14} {hits}/6 correct   (chance 1/6, binomial p = {p_bin:.4f})")
    if tot:
        print(f"\n{'4 domains':<14} {tot_hits}/{tot} correct   "
              f"(chance {tot / 6:.1f}/{tot}, binomial p = {binom_tail(tot_hits, tot, 1/6):.5f})")
    if cons_tot:
        print(f"{'consensus':<14} {cons_tot_hits}/{cons_tot} correct   "
              f"(chance {cons_tot / 6:.1f}/{cons_tot}, "
              f"binomial p = {binom_tail(cons_tot_hits, cons_tot, 1/6):.5f})")
    print("\nNote: trials are not fully independent (the six solo recordings all "
          "appear in\nevery trial's gallery), so the binomial p is a reference "
          "point, not a valid test.")


if __name__ == "__main__":
    main()
