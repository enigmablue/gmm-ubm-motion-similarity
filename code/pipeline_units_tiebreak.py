"""Tie-break for the position-units choice, on a criterion the choice cannot game.

THE PROBLEM
-----------
Whether wrist position is expressed in pixels or in face widths changes two of
the paper's headline results:

    cross-view rho, BH-adjusted     fw .062      px .024
    speaker-level retrieval         fw 2/12      px 4/12 (p = .067)

Those are exactly the outcome-dependent comparisons that must not decide a
specification, so the choice needs a criterion measured somewhere the cross-view
question does not arise.

THE CRITERION, DECLARED BEFORE RUNNING
--------------------------------------
Within-recording separation: how far a recording's own held-out half stands
above the distribution of other recordings' halves, in standard deviations. It
is measured entirely inside one view, so it cannot be influenced by cross-view
agreement or by retrieval.

    all recordings -> alternating 50-frame A / B blocks
    UBM           -> fitted on pooled A blocks of that domain
    within(v)     -> adapt on A_v, score B_v
    between(v,w)  -> adapt on A_v, score B_w
    margin        -> mean over v of (within(v) - mean_w between(v,w)) / sd_w

Higher margin means the representation separates a recording from the others
more cleanly. The rule: whichever units give the larger mean margin across the
four side-view domains wins; the other is reported as sensitivity.

r is fixed at 64 rather than selected by self-consistency, because the
self-consistency rule is itself a within-recording criterion and would be
circular here.

READS   same feature sources as pipeline.py
WRITES  output/paper_review/final/units_tiebreak.json

Usage:  python pipeline_units_tiebreak.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from gmm_pure_fullbody import maincam_head  # noqa: E402
from pipeline import build_hand_frames, vids  # noqa: E402
from table_all_methods import solo_head_new  # noqa: E402

DEST = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output\paper_review\final")
PEOPLE = [("speakerB", "SpeakerB"), ("speakerA", "SpeakerA")]
R_FIXED = 64
UNITS = ["fw", "px"]


def ab_split(X):
    n = len(X)
    mask = np.zeros(n, dtype=bool)
    for i in range(0, n, 100):
        mask[i:min(i + 50, n)] = True
    return X[mask], X[~mask]


def margin(data):
    """Standardised within- vs between-recording separation for one domain."""
    if len(data) < 4:
        return None
    parts = {v: ab_split(X) for v, X in data.items()}
    parts = {v: p for v, p in parts.items() if len(p[0]) > 20 and len(p[1]) > 20}
    if len(parts) < 4:
        return None

    pooled = np.vstack([p[0] for p in parts.values()])
    sc = RobustScaler().fit(pooled)
    A = {v: sc.transform(p[0]) for v, p in parts.items()}
    B = {v: sc.transform(p[1]) for v, p in parts.items()}
    allA = np.vstack(list(A.values()))

    best_k, best_bic = 2, np.inf
    for k in range(2, 7):
        g = GaussianMixture(n_components=k, covariance_type="diag", random_state=42,
                            max_iter=200, reg_covar=1e-4, n_init=3).fit(allA)
        if g.weights_.min() < 0.01:
            continue
        if g.bic(allA) < best_bic:
            best_bic, best_k = g.bic(allA), k
    ubm = GaussianMixture(n_components=best_k, covariance_type="diag", random_state=42,
                          max_iter=300, reg_covar=1e-4).fit(allA)

    def adapt(X):
        import copy
        resp = ubm.predict_proba(X)
        Nk = resp.sum(axis=0)
        xbar = (resp.T @ X) / (Nk[:, None] + 1e-12)
        alpha = Nk / (Nk + R_FIXED)
        ad = copy.deepcopy(ubm)
        ad.means_ = alpha[:, None] * xbar + (1.0 - alpha[:, None]) * ubm.means_
        return ad

    models = {v: adapt(A[v]) for v in A}
    within, between = {}, {}
    for iv in A:
        for jv in B:
            ll = float(models[iv].score_samples(B[jv]).mean()
                       - ubm.score_samples(B[jv]).mean())
            if iv == jv:
                within[iv] = ll
            else:
                between.setdefault(iv, []).append(ll)

    margins = []
    for v in within:
        b = np.array(between[v])
        sd = np.std(b)
        if sd > 1e-12:
            margins.append((within[v] - np.mean(b)) / sd)
    return float(np.mean(margins)) if margins else None


def main():
    out = {}
    for view in ("maincam", "fullbody"):
        out[view] = {}
        for units in UNITS:
            out[view][units] = {}
            for person, name in PEOPLE:
                data = {}
                for v in vids:
                    f = build_hand_frames(view, v, person, units)
                    if f is not None and len(f) > 50:
                        data[v] = f
                out[view][units][f"Hand_{name}"] = margin(data)
            heads = maincam_head if view == "maincam" else solo_head_new
            for person, name in PEOPLE:
                out[view][units][f"Head_{name}"] = margin(heads(person))

    for view in ("maincam", "fullbody"):
        print(f"\n{view} view: standardised within-recording margin "
              f"(higher = cleaner separation)")
        print(f"  {'domain':<14}{'fw':>9}{'px':>9}")
        for d in sorted(out[view]["fw"]):
            a, b = out[view]["fw"][d], out[view]["px"][d]
            fa = "  n/a" if a is None else f"{a:>9.3f}"
            fb = "  n/a" if b is None else f"{b:>9.3f}"
            print(f"  {d:<14}{fa}{fb}")

    def mean_hand(view, units):
        vals = [out[view][units][f"Hand_{n}"] for _, n in PEOPLE]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    print()
    for view in ("maincam", "fullbody"):
        print(f"  {view:<9} mean hand margin   fw {mean_hand(view, 'fw'):+.3f}   "
              f"px {mean_hand(view, 'px'):+.3f}")

    side = {u: mean_hand("maincam", u) for u in UNITS}
    winner = max(side, key=lambda u: side[u])
    print(f"\n  DECLARED RULE: larger mean hand margin in the side view wins")
    print(f"  result: {winner}  "
          f"(fw {side['fw']:+.3f} vs px {side['px']:+.3f})")

    out["_side_mean_hand_margin"] = side
    out["_winner"] = winner
    (DEST / "units_tiebreak.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {DEST / 'units_tiebreak.json'}")


if __name__ == "__main__":
    main()
