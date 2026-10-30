"""Method (GMM-UBM vs DTW) x hand-view (maincam vs fullbody) analysis.

FIX A applied: BOTH views use an IDENTICAL 12D hand representation --
  rel_wrist(2) in face-widths, velocity(2) in face-widths/s, speed,
  accel, direction(2), finger ext mean/sd, aperture, spread (all in
  face-widths). Frames without a detected face are skipped in BOTH views,
  so the two builders are structurally identical.

This is a METHOD x HAND-VIEW comparison, NOT a camera-angle 2x2:
head features are maincam-only (fullbody has no head pose), so consensus
uses the two HAND domains only.

Also prints hand-detection rates per video/speaker/view (Fix B baseline).

min_speed: 0 = no threshold (paper baseline); 5 = the 5 px/s ablation (v6).
Both are runnable here without editing source.
"""
import csv, sys, copy
import numpy as np
from pathlib import Path
from itertools import combinations
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from fastdtw import fastdtw
import warnings
warnings.filterwarnings('ignore')

MC = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output")
FB = MC / "fullbody_std"          # FIX B: common-scale re-extraction
FB_OLD = MC / "fullbody"          # original (native-resolution) extraction
SUBSAMPLE = 5
FPS = 30
vids = ['1', '2', '3', '4', '5', '6']
pairs = list(combinations(vids, 2))


# ===================== RAW LOAD (face refs now include WIDTH) =====================
def _one_hand_per_frame(rows):
    """Keep exactly one row per frame, preferring the Right hand.

    Both views were extracted with up to two rows per frame (Left+Right).
    Consuming rows in order interleaves the two hands, so velocity and
    acceleration are computed between different hands. This selects one
    hand per frame with the same rule for both views.
    """
    by_frame = {}
    for r in rows:
        fi = int(r['frame'])
        h = (r.get('handedness') or '').strip().lower()
        cur = by_frame.get(fi)
        if cur is None or (cur[1] != 'right' and h == 'right'):
            by_frame[fi] = (r, h)
    return [by_frame[k][0] for k in sorted(by_frame)]


def load_hand_raw(view, person, one_hand=True):
    """{vid: (rows_sorted, face_refs)}; face_refs[frame] = (cx, cy, fw)."""
    out = {}
    for v in vids:
        if view == 'fullbody':
            lm = FB / "landmarks" / f"{v}_{person}_hand_landmarks.csv"
            ft = FB / "features" / f"{v}_{person}_features.csv"
            if not lm.exists() or not ft.exists():
                continue
            fr = {}
            for r in csv.DictReader(open(ft)):
                if r.get('face1_detected') == '1':
                    fr[int(r['frame'])] = (float(r['face1_x']) + float(r['face1_w']) / 2,
                                           float(r['face1_y']) + float(r['face1_h']) / 2,
                                           float(r['face1_w']))
        else:
            lm = MC / "landmarks" / f"{v}_{person}_hand_landmarks.csv"
            ft = MC / "maincam" / f"{v}_features.csv"
            if not lm.exists():
                continue
            fp = 'face1' if person == 'stella' else 'face2'
            fr = {}
            for r in csv.DictReader(open(ft)):
                if r.get(f'{fp}_detected') == '1':
                    fr[int(r['frame'])] = (float(r[f'{fp}_x']) + float(r[f'{fp}_w']) / 2,
                                           float(r[f'{fp}_y']) + float(r[f'{fp}_h']) / 2,
                                           float(r[f'{fp}_w']))
        rows = list(csv.DictReader(open(lm)))
        if one_hand:
            rows = _one_hand_per_frame(rows)
        out[v] = (sorted(rows, key=lambda r: int(r['frame'])), fr)
    return out


# ===================== UNIFIED 12D FACE-WIDTH HAND FEATURES =====================
def hand_features(rows, fr, f0=0, f1=10**9, min_speed=0, fallback_ref=None):
    frames = []
    pw = ps = pf = None
    for r in rows:
        fi = int(r['frame'])
        if fi % SUBSAMPLE != 0 or not (f0 <= fi < f1):
            continue
        if fi not in fr:                      # no face this frame
            if fallback_ref is None:          # 'skip' mode
                pw = np.array([float(r['lm0_x']), float(r['lm0_y'])])
                pf = fi
                continue
            cx, cy, fw = fallback_ref         # 'fallback' mode (per-video median face)
        else:
            cx, cy, fw = fr[fi]
        wrist = np.array([float(r['lm0_x']), float(r['lm0_y'])])
        rel = (wrist - np.array([cx, cy])) / fw
        dt = (fi - pf) / FPS if pf else 1 / FPS
        if pw is not None and dt > 0:
            v = (wrist - pw) / dt / fw         # face-widths / s
            sp = np.linalg.norm(v)
            ds, dc = (v[1] / sp, v[0] / sp) if sp > 1e-6 else (0.0, 0.0)
        else:
            v = np.zeros(2); sp = 0.0; ds = 0.0; dc = 0.0
        ac = (sp - ps) / dt if ps is not None and dt > 0 else 0.0
        wx, wy = wrist
        tips = np.array([np.linalg.norm([float(r[f'lm{ti}_x']) - wx,
                                         float(r[f'lm{ti}_y']) - wy])
                         for ti in [4, 8, 12, 16, 20]]) / fw
        ap = np.linalg.norm([float(r['lm4_x']) - float(r['lm8_x']),
                             float(r['lm4_y']) - float(r['lm8_y'])]) / fw
        tip_xy = np.array([[float(r[f'lm{ti}_x']), float(r[f'lm{ti}_y'])] for ti in [4, 8, 12, 16, 20]])
        spr = np.mean([np.linalg.norm(tip_xy[i] - tip_xy[j])
                       for i in range(5) for j in range(i + 1, 5)]) / fw
        frames.append(np.array([rel[0], rel[1], v[0], v[1], sp, ac, ds, dc,
                                np.mean(tips), np.std(tips), ap, spr]))
        pw, ps, pf = wrist, sp, fi
    frames = np.array(frames) if frames else np.empty((0, 12))
    if min_speed > 0 and len(frames) > 1:
        keep = np.ones(len(frames), dtype=bool)
        keep[1:] = frames[1:, 4] >= min_speed
        frames = frames[keep]
    return frames if len(frames) > 10 else None


# ===================== PIPELINE =====================
def gmm_ranks(vid_to_frames):
    if len(vid_to_frames) < 3:
        return None
    all_raw = np.vstack(list(vid_to_frames.values()))
    sc = RobustScaler().fit(all_raw)
    scaled = {v: sc.transform(f) for v, f in vid_to_frames.items()}
    all_X = np.vstack(list(scaled.values()))
    best_k, best_bic = 2, np.inf
    for k in range(2, 7):
        g = GaussianMixture(n_components=k, covariance_type='diag', random_state=42,
                            max_iter=200, reg_covar=1e-4, n_init=3)
        g.fit(all_X)
        if g.weights_.min() >= 0.01 and g.bic(all_X) < best_bic:
            best_bic, best_k = g.bic(all_X), k
    ubm = GaussianMixture(n_components=best_k, covariance_type='diag', random_state=42,
                          max_iter=300, reg_covar=1e-4)
    ubm.fit(all_X)
    best_r, best_self = 16, -np.inf
    for rt in [4, 8, 16, 32, 64, 128]:
        ss = []
        for v in scaled:
            X = scaled[v]; n = len(X)
            ma = np.zeros(n, dtype=bool)
            for i in range(0, n, 100):
                ma[i:min(i + 50, n)] = True
            mb = ~ma
            if ma.sum() < 20 or mb.sum() < 20:
                continue
            resp = ubm.predict_proba(X[ma]); Nk = resp.sum(axis=0); Fk = resp.T @ X[ma]
            xb = Fk / (Nk[:, None] + 1e-12); al = Nk / (Nk + rt)
            ad = copy.deepcopy(ubm); ad.means_ = al[:, None] * xb + (1 - al[:, None]) * ubm.means_
            ss.append((ad.score_samples(X[mb]).sum() - ubm.score_samples(X[mb]).sum()) / mb.sum())
        if ss and np.mean(ss) > best_self:
            best_self, best_r = np.mean(ss), rt
    adapted = {}
    for v in scaled:
        X = scaled[v]; resp = ubm.predict_proba(X); Nk = resp.sum(axis=0); Fk = resp.T @ X
        xb = Fk / (Nk[:, None] + 1e-12); al = Nk / (Nk + best_r)
        ad = copy.deepcopy(ubm); ad.means_ = al[:, None] * xb + (1 - al[:, None]) * ubm.means_
        adapted[v] = ad
    scores = {}
    for v1, v2 in pairs:
        if v1 not in adapted or v2 not in adapted:
            continue
        X1, X2 = scaled[v1], scaled[v2]

        def sab(X):
            n = len(X); ma = np.zeros(n, dtype=bool)
            for i in range(0, n, 100):
                ma[i:min(i + 50, n)] = True
            return X[ma], X[~ma]

        X1a, X1b = sab(X1); X2a, X2b = sab(X2)
        llrs = []
        for af, so in [(X1a, X2b), (X2a, X1b), (X1b, X2a), (X2b, X1a)]:
            if len(af) < 10 or len(so) < 10:
                continue
            resp = ubm.predict_proba(af); Nk = resp.sum(axis=0); Fk = resp.T @ af
            xb = Fk / (Nk[:, None] + 1e-12); al = Nk / (Nk + best_r)
            ad = copy.deepcopy(ubm); ad.means_ = al[:, None] * xb + (1 - al[:, None]) * ubm.means_
            llrs.append((ad.score_samples(so).sum() - ubm.score_samples(so).sum()) / len(so))
        scores[(v1, v2)] = np.mean(llrs) if llrs else -999
    ranks = {}
    for ri, (p, _) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True), 1):
        ranks[p] = ri
    return ranks


def dtw_ranks(vid_to_seqs):
    if len(vid_to_seqs) < 3:
        return None
    all_seq = np.vstack(list(vid_to_seqs.values()))
    sc = RobustScaler().fit(all_seq)
    scaled = {v: sc.transform(s) for v, s in vid_to_seqs.items()}
    scores = {}
    for v1, v2 in pairs:
        if v1 not in scaled or v2 not in scaled:
            continue
        dist, path = fastdtw(scaled[v1], scaled[v2], dist=2)
        scores[(v1, v2)] = dist / len(path)
    ranks = {}
    for ri, (p, _) in enumerate(sorted(scores.items(), key=lambda x: x[1]), 1):
        ranks[p] = ri
    return ranks


def hand_consensus(domain_ranks):
    domains = [d for d in domain_ranks if domain_ranks[d] is not None]
    if len(domains) < 2:
        return None
    C = {}
    for v1, v2 in pairs:
        C[(v1, v2)] = float(np.mean([1.0 - (domain_ranks[d].get((v1, v2), 15) - 1) / 14.0
                                     for d in domains]))
    ordered = sorted(C.items(), key=lambda x: x[1], reverse=True)
    rank45 = next(i + 1 for i, (p, _) in enumerate(ordered) if p == ('4', '5'))
    c45 = C[('4', '5')]
    others = [c for p, c in C.items() if p != ('4', '5')]
    return {'C45': c45, 'rank': rank45, 'gap': c45 - max(others),
            'hand_ranks': {d: domain_ranks[d].get(('4', '5'), 15) for d in domains}}


# ===================== MAIN =====================
if __name__ == "__main__":
    min_speed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    face_mode = sys.argv[2] if len(sys.argv) > 2 else 'skip'
    print(f"FIX A: identical 12D face-width hand features; min_speed={min_speed}; "
          f"no-face handling={face_mode}")
    print("=" * 72)

    cache = {}
    for view in ['maincam', 'fullbody']:
        for person in ['stella', 'aru']:
            cache[(view, person)] = load_hand_raw(view, person)

    # ---- hand-detection rates (Fix B baseline) ----
    print("Hand frames kept / subsampled frames with a detected face:")
    print(f"{'view':<9} {'speaker':<7} " + " ".join(f"{v:>13}" for v in vids))
    for view in ['maincam', 'fullbody']:
        for person in ['stella', 'aru']:
            cells = []
            for v in vids:
                if v not in cache[(view, person)]:
                    cells.append(f"{'-':>13}"); continue
                rows, fr = cache[(view, person)][v]
                sub_hand = [r for r in rows if int(r['frame']) % SUBSAMPLE == 0]
                sub_face = [f for f in fr if f % SUBSAMPLE == 0]
                cells.append(f"{len(sub_hand):>6}/{len(sub_face):<6}")
            print(f"{view:<9} {person:<7} " + " ".join(cells))

    # ---- face-width stability (is the normaliser noisy?) ----
    print("\nFace-width variability per video (sd/mean; lower = stabler normaliser):")
    print(f"{'view':<9} {'speaker':<7} " + " ".join(f"{v:>7}" for v in vids))
    for view in ['maincam', 'fullbody']:
        for person in ['stella', 'aru']:
            cvs = []
            for v in vids:
                if v not in cache[(view, person)]:
                    cvs.append(np.nan); continue
                rows, fr = cache[(view, person)][v]
                ws = [x[2] for x in fr.values()]
                cvs.append(np.std(ws) / np.mean(ws) if ws else np.nan)
            print(f"{view:<9} {person:<7} " + " ".join(f"{x:>7.3f}" for x in cvs))

    # ---- method x hand-view 2x2 (hand-only consensus) ----
    results = {}
    for view in ['maincam', 'fullbody']:
        for method in ['GMM', 'DTW']:
            dr = {}
            for person, label in [('stella', 'Stella'), ('aru', 'Aru')]:
                rows_by_vid = {}
                for v in vids:
                    if v in cache[(view, person)]:
                        rows, fr = cache[(view, person)][v]
                        fb = None
                        if face_mode == 'fallback' and fr:
                            vals = list(fr.values())
                            fb = (float(np.median([x[0] for x in vals])),
                                  float(np.median([x[1] for x in vals])),
                                  float(np.median([x[2] for x in vals])))
                        f = hand_features(rows, fr, min_speed=min_speed, fallback_ref=fb)
                        if f is not None:
                            rows_by_vid[v] = f
                dr[f"Hand_{label}"] = gmm_ranks(rows_by_vid) if method == 'GMM' else dtw_ranks(rows_by_vid)
            results[f"{method}_{view}"] = hand_consensus(dr)

    print("\n" + "=" * 72)
    print("METHOD x HAND-VIEW (hand-only consensus, identical 12D face-width features)")
    print("=" * 72)
    print(f"{'Cell':<16} {'C45':>6} {'Rank':>7} {'Gap':>7}  hand ranks")
    for k in ["GMM_maincam", "GMM_fullbody", "DTW_maincam", "DTW_fullbody"]:
        r = results[k]
        if r is None:
            print(f"{k:<16}  (insufficient data)"); continue
        print(f"{k:<16} {r['C45']:>6.3f} {r['rank']:>4}/15 {r['gap']:>+7.3f}  {r['hand_ranks']}")

    G2 = results["GMM_maincam"]['C45']; D2 = results["DTW_maincam"]['C45']
    GF = results["GMM_fullbody"]['C45']; DF = results["DTW_fullbody"]['C45']
    print("\nContrasts:")
    print(f"  method effect (maincam)  = {G2 - D2:+.3f}")
    print(f"  method effect (fullbody) = {GF - DF:+.3f}")
    print(f"  hand-view effect (GMM)   = {G2 - GF:+.3f}")
    print(f"  hand-view effect (DTW)   = {D2 - DF:+.3f}")
    print(f"  difference-in-differences= {(G2 - D2) - (GF - DF):+.3f}")
