"""The prose-level statistics that no table holds.

Table edits are mechanical. These are the sentences that quote derived numbers:

  H1a  the leave-one-session-out range and how many of the six stay significant
  H1b  top-five and top-ten overlap between views, and the mean rank shift
  H3a  the session-bootstrap interval for the difference between the framework
       and sequence alignment

All are computed from consensus scores over the same 15 pairs. Framework
consensus comes from the stored pipeline output; comparator consensus is
rebuilt here for both views so the bootstrap has a frontal ordering to resample.

WRITES  output/paper_review/final/pipeline_textstats_{units}.json
Usage:  python pipeline_textstats.py [--units fw|px]
"""
import argparse
import json
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from aggregate_seven import mouth_stats  # noqa: E402
from crossview_exact import agg_domains  # noqa: E402
from pipeline import DEST, PKEYS, consensus as pipe_consensus  # noqa: E402
from pipeline_comparators import consensus as dtw_consensus, domain_dtw  # noqa: E402
from review2_checks import consensus_from_scores  # noqa: E402

vids = ["1", "2", "3", "4", "5", "6"]
PAIRS = list(combinations(vids, 2))


def ordered(pairs, C):
    return sorted(pairs, key=lambda p: -C[f"({p[0]},{p[1]})"])


def top_overlap(side, front, k):
    a = {p for p in ordered(PAIRS, side)[:k]}
    b = {p for p in ordered(PAIRS, front)[:k]}
    return len(a & b)


def mean_shift(side, front):
    rs = {p: i + 1 for i, p in enumerate(ordered(PAIRS, side))}
    rf = {p: i + 1 for i, p in enumerate(ordered(PAIRS, front))}
    return float(np.mean([abs(rs[p] - rf[p]) for p in PAIRS]))


def rho(a, b):
    return float(stats.spearmanr(a, b)[0])


def perm_p(side, front, sessions, observed):
    """Exact two-sided p over relabellings of the sessions still present."""
    keys = [p for p in PAIRS
            if p[0] in sessions and p[1] in sessions]
    y = np.array([front[f"({a},{b})"] for a, b in keys])
    null = []
    for perm in permutations(sessions):
        m = dict(zip(sessions, perm))

        def pk(i, j):
            return f"({m[i]},{m[j]})" if m[i] < m[j] else f"({m[j]},{m[i]})"

        x = np.array([side[pk(a, b)] for a, b in keys])
        if np.std(x) > 1e-12:
            null.append(rho(x, y))
    null = np.array(null)
    return float(np.mean(np.abs(null) >= abs(observed) - 1e-12))


def loo(side, front):
    out = {}
    for drop in vids:
        keep = [v for v in vids if v != drop]
        keys = [p for p in PAIRS if p[0] in keep and p[1] in keep]
        a = np.array([side[f"({x},{y})"] for x, y in keys])
        b = np.array([front[f"({x},{y})"] for x, y in keys])
        r = rho(a, b)
        out[drop] = dict(rho=r, p=perm_p(side, front, keep, r))
    return out


def boot_diff(sideA, frontA, sideB, frontB, n=20000, seed=0):
    """Percentile interval for Spearman(A) - Spearman(B) over session resamples."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        pick = rng.choice(vids, size=len(vids), replace=True)
        keys = [(pick[i], pick[j])
                for i in range(len(pick)) for j in range(i + 1, len(pick))
                if pick[i] != pick[j]]
        if len(keys) < 3:
            continue
        ka = [f"({a},{b})" if a < b else f"({b},{a})" for a, b in keys]
        xa = np.array([sideA[k] for k in ka])
        ya = np.array([frontA[k] for k in ka])
        xb = np.array([sideB[k] for k in ka])
        yb = np.array([frontB[k] for k in ka])
        if min(np.std(xa), np.std(ya), np.std(xb), np.std(yb)) < 1e-12:
            continue
        diffs.append(rho(xa, ya) - rho(xb, yb))
    d = np.array(diffs)
    return dict(median=float(np.median(d)),
                lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)),
                n=int(len(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", choices=["fw", "px"], default="fw")
    args = ap.parse_args()
    units = args.units

    main_res = json.loads((DEST / f"pipeline_{units}.json").read_text())
    fw_side = main_res["consensus_side"]
    fw_front = main_res["consensus_frontal"]

    # comparator consensus, both views
    dtw = {}
    for view in ("maincam", "fullbody"):
        R, C = dtw_consensus(domain_dtw(view, units))
        dtw[view] = C
    m_mc, m_fb = mouth_stats()
    agg = {}
    for view, mouths in [("maincam", m_mc), ("fullbody", m_fb)]:
        raw = {k: v for k, v in agg_domains(view, mouths).items()
               if not k.startswith("Mouth")}
        agg[view], _ = consensus_from_scores(raw, False)

    out = {"units": units}

    out["overlap"] = dict(
        top5=top_overlap(fw_side, fw_front, 5),
        top10=top_overlap(fw_side, fw_front, 10),
        mean_shift=mean_shift(fw_side, fw_front))
    print(f"H1b  top5 {out['overlap']['top5']}/5  "
          f"top10 {out['overlap']['top10']}/10  "
          f"mean shift {out['overlap']['mean_shift']:.2f} places")

    out["loo"] = loo(fw_side, fw_front)
    rs = [v["rho"] for v in out["loo"].values()]
    below = sum(1 for v in out["loo"].values() if v["p"] >= .05)
    worst = min(out["loo"], key=lambda k: out["loo"][k]["rho"])
    best = max(out["loo"], key=lambda k: out["loo"][k]["rho"])
    out["loo_summary"] = dict(lo=min(rs), hi=max(rs), below_p05=below,
                              worst=worst, best=best)
    print(f"H1a  leave-one-out rho {min(rs):+.3f} (without {worst}) to "
          f"{max(rs):+.3f} (without {best}); {below}/6 not significant")

    out["boot_fw_minus_dtw"] = boot_diff(
        fw_side, fw_front, dtw["maincam"], dtw["fullbody"])
    b = out["boot_fw_minus_dtw"]
    print(f"H3a  framework minus sequence alignment: "
          f"95% [{b['lo']:+.3f}, {b['hi']:+.3f}]")

    out["boot_fw_minus_agg"] = boot_diff(
        fw_side, fw_front, agg["maincam"], agg["fullbody"])
    a = out["boot_fw_minus_agg"]
    print(f"     framework minus aggregate:             "
          f"95% [{a['lo']:+.3f}, {a['hi']:+.3f}]")

    p = DEST / f"pipeline_textstats_{units}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
