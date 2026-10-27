"""Does GMM-UBM's fullbody result survive when the head comes from the solo cameras?

The paper's Evaluation 8 fullbody cell is not a fullbody measurement. From
`paper_review/cell_extractor.py`:

    "Head features are identical across camera conditions (both read the
     maincam video); only the hand landmarks differ (maincam vs fullbody)."

So the published fullbody result (rank 1/15, C = .929) is maincam head + fullbody
hands. The aggregate measure, run on the same recordings with head taken from the
solo cameras, ranks (4,5) 10th of 15.

This script runs GMM-UBM both ways so the two can be compared like for like:

  A. shared head    head from maincam, hands from fullbody   (the paper's cell)
  B. pure fullbody  head from solo cameras, hands from fullbody

Hypothesis: if the head source is what carries the result, B should collapse
towards the aggregate's rank, and A should not.

Usage:
    python gmm_pure_fullbody.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from cell_extractor import run_gmm_domain, consensus  # noqa: E402
from handview_2x2 import load_hand_raw, hand_features  # noqa: E402

MC = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output")
vids = ["1", "2", "3", "4", "5", "6"]
SUBSAMPLE = 5
EXCURSION_LAG = 3


def _head_18d(rows, fp):
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


def solo_head(person):
    """18D head features for one speaker's six solo recordings."""
    raw = {}
    for v in vids:
        if person == "stella":
            p = MC / "stella" / f"{v}_features.csv"
        else:
            p = MC / "aru" / f"fb_aru_{int(v):02d}_features.csv"
        if not p.exists():
            continue
        f = _head_18d(list(csv.DictReader(open(p))), "face1")
        if f is not None and len(f) > 50:
            raw[v] = f
    if raw:
        gm = np.median(np.concatenate([f[:, 12] for f in raw.values()]))
        for v in raw:
            raw[v] = raw[v].copy()
            raw[v][:, 14] = (raw[v][:, 12] < gm).astype(float)
    return raw


def maincam_head(person):
    """18D head features for one speaker, as the paper computes them."""
    raw = {}
    fp = "face1" if person == "stella" else "face2"
    path = MC / "maincam"
    for v in vids:
        p = path / f"{v}_features.csv"
        if not p.exists():
            continue
        f = _head_18d(list(csv.DictReader(open(p))), fp)
        if f is not None and len(f) > 50:
            raw[v] = f
    if raw:
        gm = np.median(np.concatenate([f[:, 12] for f in raw.values()]))
        for v in raw:
            raw[v] = raw[v].copy()
            raw[v][:, 14] = (raw[v][:, 12] < gm).astype(float)
    return raw


def fullbody_hands(person):
    cache = load_hand_raw("fullbody", person)
    out = {}
    for v in vids:
        if v not in cache:
            continue
        rows, fr = cache[v]
        f = hand_features(rows, fr, min_speed=0)
        if f is not None:
            out[v] = f
    return out


def run(label, head_by_person, hand_by_person):
    domains = {}
    for person, name in [("stella", "Stella"), ("aru", "Aru")]:
        domains[f"Hand_{name}"] = run_gmm_domain(hand_by_person[person])
        domains[f"Head_{name}"] = run_gmm_domain(head_by_person[person])
    r = consensus(domains)
    dr = r["domain_ranks"]
    print(f"\n{label}")
    print(f"   domain ranks (4,5): Hand_St {dr['Hand_Stella']}/15, "
          f"Hand_Ar {dr['Hand_Aru']}/15, Head_St {dr['Head_Stella']}/15, "
          f"Head_Ar {dr['Head_Aru']}/15")
    print(f"   CONSENSUS rank {r['rank']}/15, C = {r['C45']:.3f}, "
          f"next best C = {r['next_best']:.3f}, gap = {r['gap']:+.3f}")
    return r


def main():
    hand_fb = {p: fullbody_hands(p) for p in ["stella", "aru"]}
    head_mc = {p: maincam_head(p) for p in ["stella", "aru"]}
    head_fb = {p: solo_head(p) for p in ["stella", "aru"]}

    print("=" * 74)
    print("GMM-UBM, FULLBODY HANDS, TWO HEAD SOURCES")
    print("=" * 74)

    a = run("A. Shared head  (head = maincam, hands = fullbody)  -- the paper's cell",
            head_mc, hand_fb)
    b = run("B. Pure fullbody (head = solo,    hands = fullbody)",
            head_fb, hand_fb)

    print()
    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"{'configuration':<46}{'rank':>8}{'C':>8}")
    print("-" * 74)
    print(f"{'GMM, maincam head + fullbody hands (paper)':<46}"
          f"{a['rank']:>5}/15{a['C45']:>8.3f}")
    print(f"{'GMM, solo head + fullbody hands (pure)':<46}"
          f"{b['rank']:>5}/15{b['C45']:>8.3f}")
    print()
    print("Aggregate, pure fullbody (7 measures incl. mouth) : 10/15, C = 0.440")
    print("Aggregate, maincam                                :  1/15, C = 0.893")


if __name__ == "__main__":
    main()
