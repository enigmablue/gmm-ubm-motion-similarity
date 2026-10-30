"""Four-cell extractor for the Evaluation-8 2x2.

Reproduces the four method x camera cells from a single canonical
configuration and reports per-cell statistics plus the four planned
descriptive contrasts. No resampling yet.

Cells:
  GMM-UBM x maincam
  GMM-UBM x fullbody
  DTW      x maincam
  DTW      x fullbody

Head features are identical across camera conditions (both read the
maincam video); only the hand landmarks differ (maincam vs fullbody).
"""
import csv, numpy as np, copy
from pathlib import Path
from itertools import combinations
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from fastdtw import fastdtw
import warnings
warnings.filterwarnings('ignore')

MC = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output")
FB = MC / "fullbody"
SUBSAMPLE = 5
EXCURSION_LAG = 3
vids = ['1', '2', '3', '4', '5', '6']
pairs = list(combinations(vids, 2))


# ===================== HEAD 18D (maincam, shared by both cameras) =====================
def build_head(vid, person):
    feat_path = MC / "maincam" / f"{vid}_features.csv"
    rows = list(csv.DictReader(open(feat_path)))
    fp = 'face1' if person == 'stella' else 'face2'
    records = []
    for r in rows:
        fi = int(r['frame'])
        if fi % SUBSAMPLE != 0:
            continue
        if r.get(f'{fp}_detected') != '1':
            continue
        ts = float(r.get('timestamp_sec', fi / 30))
        angles = np.array([
            float(r.get(f'{fp}_yaw', 0) or 0),
            float(r.get(f'{fp}_pitch', 0) or 0),
            float(r.get(f'{fp}_roll', 0) or 0)])
        records.append((fi, ts, angles))
    if len(records) < 30:
        return None

    frames, angles_history = [], []
    pa = pt = pvel = pvel_mag = None
    for fi, ts, angles in records:
        dt = ts - pt if pt else 0.033
        if pa is not None and dt > 1e-6:
            vel = ((angles - pa + 180.0) % 360.0 - 180.0) / dt
        else:
            vel = np.zeros(3)
        vel_mag = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
        if pvel is not None and dt > 1e-6:
            accel = (vel - pvel) / dt
            ang_accel = (vel_mag - pvel_mag) / dt
        else:
            accel = np.zeros(3)
            ang_accel = 0.0
        angles_history.append(angles)
        if len(angles_history) > EXCURSION_LAG:
            exc = (angles - angles_history[-EXCURSION_LAG - 1] + 180.0) % 360.0 - 180.0
        else:
            exc = np.zeros(3)
        feat = np.concatenate([angles, vel, np.abs(vel), accel,
                               [vel_mag], [ang_accel], [0.0], exc])
        frames.append(feat)
        pa, pt, pvel, pvel_mag = angles, ts, vel, vel_mag
    return np.array(frames)


def apply_global_stillness(vid_to_frames):
    all_speeds = []
    for f in vid_to_frames.values():
        if f is not None and len(f) > 0:
            all_speeds.append(f[:, 12])
    if not all_speeds:
        return vid_to_frames
    gm = np.median(np.concatenate(all_speeds))
    out = {}
    for v, f in vid_to_frames.items():
        f2 = f.copy()
        f2[:, 14] = (f2[:, 12] < gm).astype(float)
        out[v] = f2
    return out


# ===================== HAND 12D =====================
def build_hand_maincam(vid, person):
    lm_path = MC / "landmarks" / f"{vid}_{person}_hand_landmarks.csv"
    feat_path = MC / "maincam" / f"{vid}_features.csv"
    if not lm_path.exists():
        return None
    fp = 'face1' if person == 'stella' else 'face2'
    face_center = {}
    with open(feat_path) as f:
        for r in csv.DictReader(f):
            if r.get(f'{fp}_detected') == '1':
                face_center[int(r['frame'])] = np.array([
                    float(r[f'{fp}_x']) + float(r[f'{fp}_w']) / 2,
                    float(r[f'{fp}_y']) + float(r[f'{fp}_h']) / 2])
    rows = list(csv.DictReader(open(lm_path)))
    if len(rows) < 30:
        return None
    return _hand_loop(rows, face_center, normalize=False, median_fw=None)


def build_hand_fullbody(vid, person):
    lm_path = FB / "landmarks" / f"{vid}_{person}_hand_landmarks.csv"
    feat_path = FB / "features_m1" / f"{vid}_{person}_features.csv"
    if not lm_path.exists() or not feat_path.exists():
        return None
    fp = 'face1'
    face_info = {}
    with open(feat_path) as f:
        for r in csv.DictReader(f):
            if r.get(f'{fp}_detected') == '1':
                face_info[int(r['frame'])] = (
                    float(r[f'{fp}_x']) + float(r[f'{fp}_w']) / 2,
                    float(r[f'{fp}_y']) + float(r[f'{fp}_h']) / 2,
                    float(r[f'{fp}_w']))
    if not face_info:
        return None
    median_fw = np.median([v[2] for v in face_info.values()])
    rows = list(csv.DictReader(open(lm_path)))
    if len(rows) < 30:
        return None
    return _hand_loop(rows, face_info, normalize=True, median_fw=median_fw)


def _hand_loop(rows, face_ref, normalize, median_fw):
    frames = []
    pw = ps = pf = None
    for r in rows:
        fi = int(r['frame'])
        if fi % SUBSAMPLE != 0:
            continue
        wx = float(r["lm0_x"]); wy = float(r["lm0_y"])
        wrist = np.array([wx, wy])

        if normalize:
            if fi not in face_ref:
                # skip frames without face detection (matches fullbody scripts)
                pw = wrist; ps = ps; pf = fi
                continue
            cx, cy, fw = face_ref[fi]
            rel_wrist = np.array([wx - cx, wy - cy]) / fw
        else:
            rel_wrist = wrist - face_ref[fi] if fi in face_ref else np.array([0.0, 0.0])
            fw = 1.0

        dt = (fi - pf) / 30.0 if pf else 1 / 30
        if pw is not None and dt > 0:
            v = (wrist - pw) / dt / fw
            speed = np.linalg.norm(v)
            ds, dc = (v[1] / speed, v[0] / speed) if speed > 1e-6 else (0.0, 0.0)
        else:
            v = np.zeros(2); speed = 0.0; ds = 0.0; dc = 0.0
        accel = (speed - ps) / dt if ps is not None and dt > 0 else 0.0

        tips = np.array([np.linalg.norm([float(r[f"lm{ti}_x"]) - wx,
                                         float(r[f"lm{ti}_y"]) - wy])
                         for ti in [4, 8, 12, 16, 20]]) / fw
        fem, fesd = np.mean(tips), np.std(tips)
        aperture = np.linalg.norm([float(r["lm4_x"]) - float(r["lm8_x"]),
                                   float(r["lm4_y"]) - float(r["lm8_y"])]) / fw
        tip_xy = np.array([[float(r[f"lm{ti}_x"]), float(r[f"lm{ti}_y"])]
                           for ti in [4, 8, 12, 16, 20]])
        spread = np.mean([np.linalg.norm(tip_xy[i] - tip_xy[j])
                          for i in range(5) for j in range(i + 1, 5)]) / fw

        frames.append(np.array([rel_wrist[0], rel_wrist[1], v[0], v[1],
                                speed, accel, ds, dc, fem, fesd, aperture, spread]))
        pw, ps, pf = wrist, speed, fi
    return np.array(frames)


# ===================== GMM-UBM DOMAIN =====================
def run_gmm_domain(vid_to_frames, label=""):
    if len(vid_to_frames) < 3:
        return None
    all_raw = np.vstack(list(vid_to_frames.values()))
    sc = RobustScaler().fit(all_raw)
    scaled = {v: sc.transform(f) for v, f in vid_to_frames.items()}
    all_X = np.vstack([scaled[v] for v in scaled])

    # BIC K selection with 1% occupancy filter
    best_k, best_bic = 2, np.inf
    for k in range(2, 7):
        g = GaussianMixture(n_components=k, covariance_type='diag',
                            random_state=42, max_iter=200, reg_covar=1e-4, n_init=3)
        g.fit(all_X)
        if g.weights_.min() < 0.01:
            continue
        if g.bic(all_X) < best_bic:
            best_bic, best_k = g.bic(all_X), k

    ubm = GaussianMixture(n_components=best_k, covariance_type='diag',
                          random_state=42, max_iter=300, reg_covar=1e-4)
    ubm.fit(all_X)

    # self-consistency r (alternating 50-frame blocks)
    best_r, best_self = 16, -np.inf
    for r_test in [4, 8, 16, 32, 64, 128]:
        self_scores = []
        for v in vids:
            if v not in scaled:
                continue
            Xv = scaled[v]; n = len(Xv)
            ma = np.zeros(n, dtype=bool)
            for i in range(0, n, 100):
                ma[i:min(i + 50, n)] = True
            mb = ~ma
            if ma.sum() < 20 or mb.sum() < 20:
                continue
            resp = ubm.predict_proba(Xv[ma])
            Nk = resp.sum(axis=0); Fk = resp.T @ Xv[ma]
            xbar = Fk / (Nk[:, None] + 1e-12)
            alpha = Nk / (Nk + r_test)
            ad = copy.deepcopy(ubm)
            ad.means_ = alpha[:, None] * xbar + (1 - alpha[:, None]) * ubm.means_
            self_scores.append((ad.score_samples(Xv[mb]).sum()
                                - ubm.score_samples(Xv[mb]).sum()) / mb.sum())
        if self_scores and np.mean(self_scores) > best_self:
            best_self, best_r = np.mean(self_scores), r_test

    # MAP adaptation
    adapted = {}
    for v in vids:
        if v not in scaled:
            continue
        Xv = scaled[v]
        resp = ubm.predict_proba(Xv)
        Nk = resp.sum(axis=0); Fk = resp.T @ Xv
        xbar = Fk / (Nk[:, None] + 1e-12)
        alpha = Nk / (Nk + best_r)
        ad = copy.deepcopy(ubm)
        ad.means_ = alpha[:, None] * xbar + (1 - alpha[:, None]) * ubm.means_
        adapted[v] = ad

    # cross-fitted symmetric LLR
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
            resp = ubm.predict_proba(af)
            Nk = resp.sum(axis=0); Fk = resp.T @ af
            xbar = Fk / (Nk[:, None] + 1e-12)
            alpha = Nk / (Nk + best_r)
            ad = copy.deepcopy(ubm)
            ad.means_ = alpha[:, None] * xbar + (1 - alpha[:, None]) * ubm.means_
            llrs.append((ad.score_samples(so).sum() - ubm.score_samples(so).sum()) / len(so))
        scores[(v1, v2)] = np.mean(llrs) if llrs else -999

    ranks = {}
    for ri, (p, _) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True), 1):
        ranks[p] = ri
    if label:
        print(f"    {label}: (4,5) rank {ranks.get(('4','5'), '?')}/15, K={best_k}, r={best_r}")
    return ranks


# ===================== DTW DOMAIN =====================
def run_dtw_domain(vid_to_seqs, label=""):
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
        scores[(v1, v2)] = dist / len(path)  # path-length normalized
    ranks = {}
    for ri, (p, _) in enumerate(sorted(scores.items(), key=lambda x: x[1]), 1):
        ranks[p] = ri
    if label:
        print(f"    {label}: (4,5) rank {ranks.get(('4','5'), '?')}/15")
    return ranks


# ===================== CONSENSUS =====================
def consensus(domain_ranks):
    domains = list(domain_ranks.keys())
    C = {}
    for v1, v2 in pairs:
        rs = [1.0 - (domain_ranks[d].get((v1, v2), 15) - 1) / 14.0 for d in domains]
        C[(v1, v2)] = np.mean(rs)
    ordered = sorted(C.items(), key=lambda x: x[1], reverse=True)
    rank45 = next(i + 1 for i, (p, _) in enumerate(ordered) if p == ('4', '5'))
    c45 = C[('4', '5')]
    others = [c for p, c in C.items() if p != ('4', '5')]
    next_best = max(others)
    gap = c45 - next_best
    z = (c45 - np.mean(others)) / np.std(others) if np.std(others) > 0 else 0.0
    dom_ranks_45 = {d: domain_ranks[d].get(('4', '5'), 15) for d in domains}
    top3 = sum(1 for r in dom_ranks_45.values() if r <= 3)
    top4 = sum(1 for r in dom_ranks_45.values() if r <= 4)
    return {'C45': c45, 'rank': rank45, 'domain_ranks': dom_ranks_45,
            'mean_rank': np.mean(list(dom_ranks_45.values())),
            'top3': top3, 'top4': top4, 'next_best': next_best, 'gap': gap, 'z': z}


# ===================== FEATURE BUILDERS (module-level, importable) =====================
def build_all_features():
    """Return (head, hand_mc, hand_fb) feature dicts keyed by person -> {vid: array}."""
    head = {}
    for person in ['stella', 'aru']:
        raw = {}
        for v in vids:
            f = build_head(v, person)
            if f is not None and len(f) > 50:
                raw[v] = f
        head[person] = apply_global_stillness(raw)

    hand_mc = {}; hand_fb = {}
    for person in ['stella', 'aru']:
        hmc = {}; hfb = {}
        for v in vids:
            f = build_hand_maincam(v, person)
            if f is not None and len(f) > 10:
                hmc[v] = f
            g = build_hand_fullbody(v, person)
            if g is not None and len(g) > 10:
                hfb[v] = g
        hand_mc[person] = hmc
        hand_fb[person] = hfb
    return head, hand_mc, hand_fb


def run_all_cells(head, hand_mc, hand_fb, quiet=False):
    """Run the 4 method x camera cells and return the per-cell stats dict."""
    results = {}
    for camera, handdict, htag in [("maincam", hand_mc, "mc"), ("fullbody", hand_fb, "fb")]:
        dr_gmm = {}
        for person, label in [('stella', 'Stella'), ('aru', 'Aru')]:
            dr_gmm[f"Hand_{label}"] = run_gmm_domain(handdict[person], "" if quiet else f"Hand {label}")
            dr_gmm[f"Head_{label}"] = run_gmm_domain(head[person], "" if quiet else f"Head {label}")
        results[f"GMM_{htag}"] = consensus(dr_gmm)

        dr_dtw = {}
        for person, label in [('stella', 'Stella'), ('aru', 'Aru')]:
            dr_dtw[f"Hand_{label}"] = run_dtw_domain(handdict[person], "" if quiet else f"Hand {label}")
            dr_dtw[f"Head_{label}"] = run_dtw_domain(head[person], "" if quiet else f"Head {label}")
        results[f"DTW_{htag}"] = consensus(dr_dtw)
    return results


def contrasts(results):
    CG2 = results["GMM_mc"]['C45']; CF2 = results["DTW_mc"]['C45']
    CGF = results["GMM_fb"]['C45']; CDF = results["DTW_fb"]['C45']
    return {
        'method_2P': CG2 - CF2,
        'method_FB': CGF - CDF,
        'camera_GMM': CG2 - CGF,
        'camera_DTW': CF2 - CDF,
        'interaction': (CG2 - CF2) - (CGF - CDF),
    }


# ===================== MAIN =====================
def main():
    print("Building features...")
    head, hand_mc, hand_fb = build_all_features()
    results = run_all_cells(head, hand_mc, hand_fb)

    # ===================== REPORT =====================
    print("\n" + "=" * 74)
    print("FOUR-CELL SUMMARY")
    print("=" * 74)
    print(f"{'Cell':<12} {'C45':>6} {'Rank':>6} {'HandS':>6} {'HandA':>6} {'HeadS':>6} {'HeadA':>6} "
          f"{'Next':>6} {'Gap':>6} {'z':>6}")
    for key, label in [("GMM_mc", "GMM-maincam"), ("GMM_fb", "GMM-fullbody"),
                       ("DTW_mc", "DTW-maincam"), ("DTW_fb", "DTW-fullbody")]:
        r = results[key]
        dr = r['domain_ranks']
        print(f"{label:<12} {r['C45']:>6.3f} {r['rank']:>5}/15 "
              f"{dr.get('Hand_Stella',15):>6} {dr.get('Hand_Aru',15):>6} "
              f"{dr.get('Head_Stella',15):>6} {dr.get('Head_Aru',15):>6} "
              f"{r['next_best']:>6.3f} {r['gap']:>6.3f} {r['z']:>6.2f}")

    c = contrasts(results)
    print("\n" + "=" * 74)
    print("PLANNED DESCRIPTIVE CONTRASTS")
    print("=" * 74)
    print(f"  Method effect (2-person)  = C(GMM,2P) - C(DTW,2P) = {c['method_2P']:+.3f}")
    print(f"  Method effect (fullbody)  = C(GMM,FB) - C(DTW,FB) = {c['method_FB']:+.3f}")
    print(f"  Camera effect (GMM)       = C(GMM,2P) - C(GMM,FB) = {c['camera_GMM']:+.3f}")
    print(f"  Camera effect (DTW)       = C(DTW,2P) - C(DTW,FB) = {c['camera_DTW']:+.3f}")
    print(f"  delta_interaction         = Method(2P) - Method(FB) = {c['interaction']:+.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
