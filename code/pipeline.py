"""Merged GMM-UBM pipeline: the extraction track joined to the method track.

WHY THIS FILE EXISTS
--------------------
Development ran along two tracks that each advanced correctly but were never
merged:

  extraction  extract_maincam_hands_std.py  ("re-extract maincam hand landmarks
              at the stride the analysis actually uses") and
              extract_solo_head_std.py
  method      gmm_ubm -> v2 -> v3 -> v4 -> v5 -> hand_two_channel -> canonical

`src/gmm/gmm_ubm_v5.py` reads `output/landmarks`, which is sampled at stride 3.
Its `SUBSAMPLE = 5` filter therefore keeps every 15th frame -- ~2 Hz, not the
6 Hz the manuscript states. The re-extraction that fixes this exists and is
dated a month later, but v5 never consumed it. Tables 2 and 3 of the manuscript
came from before that merge; Tables 5 and 6 came from after it.

This module is the merge point. It keeps:

  hands   within-hand derivatives, strict gap rule (method track)
          landmarks from maincam_hands_std / fullbody_std (extraction track)
  head    maincam_head / solo_head, unchanged, 18D
  model   BIC K selection with the 1% occupancy filter, self-consistency r,
          cross-fitted symmetric LLR, average-rank consensus

Only one thing is a free parameter: `UNITS`. Wrist position is either expressed
in pixels relative to the face centre (`px`) or in face widths (`fw`). The
manuscript normalises the frontal view by face width, so `fw` is the default;
`px` is retained because the choice moves the cross-view rho measurably and
that sensitivity is reported rather than hidden.

RECORDED, NOT DECIDED HERE
--------------------------
Retrieval across views and the DTW / aggregate comparators are NOT recomputed
here. Their manuscript values come from the same feature definitions and must
be re-derived before Table 5 is final.

Usage
    python pipeline.py              # face-width units
    python pipeline.py --units px

Writes output/paper_review/final/pipeline_{units}.json
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from gmm_pure_fullbody import maincam_head  # noqa: E402
from review2_raw_scores import gmm_scores  # noqa: E402
from table_all_methods import solo_head_new  # noqa: E402

OUT = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output")
DEST = OUT / "paper_review" / "final"
SUBSAMPLE = 5
FPS = 30.0
HANDS = ("left", "right")
vids = ["1", "2", "3", "4", "5", "6"]
PAIRS = [(a, b) for a, b in combinations(vids, 2)]
PKEYS = [f"({a},{b})" for a, b in PAIRS]
TARGET = "(4,5)"

VIEWS = {
    "maincam": dict(
        lm=OUT / "maincam_hands_std",
        feat=OUT / "maincam",
        feat_name="{v}_features.csv",
        per_person_face=True),
    "fullbody": dict(
        lm=OUT / "fullbody_std" / "landmarks",
        feat=OUT / "fullbody_std" / "features",
        feat_name="{v}_{person}_features.csv",
        per_person_face=False),
}


def key(p):
    return p if isinstance(p, str) else f"({p[0]},{p[1]})"


def face_ref(view, vid, person):
    """{frame: (cx, cy, face_width)}."""
    cfg = VIEWS[view]
    fp = ("face1" if person == "speakerB" else "face2") \
        if cfg["per_person_face"] else "face1"
    path = cfg["feat"] / cfg["feat_name"].format(v=vid, person=person)
    ref = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get(f"{fp}_detected") == "1":
                ref[int(r["frame"])] = (
                    float(r[f"{fp}_x"]) + float(r[f"{fp}_w"]) / 2,
                    float(r[f"{fp}_y"]) + float(r[f"{fp}_h"]) / 2,
                    float(r[f"{fp}_w"]))
    return ref


def hand_rows(rows, hand):
    keep = [r for r in rows
            if (r.get("handedness") or "").strip().lower() == hand]
    keep.sort(key=lambda r: int(r["frame"]))
    return [r for r in keep if int(r["frame"]) % SUBSAMPLE == 0]


def build_hand_frames(view, vid, person, units):
    """12D hand vector, differenced within each hand, both hands pooled."""
    lm_path = VIEWS[view]["lm"] / f"{vid}_{person}_hand_landmarks.csv"
    if not lm_path.exists():
        return None
    rows = list(csv.DictReader(open(lm_path)))
    if len(rows) < 30:
        return None
    ref = face_ref(view, vid, person)
    ws = [v[2] for v in ref.values() if v[2] > 0]
    fw_fallback = float(np.median(ws)) if ws else 1.0

    frames = []
    for hand in HANDS:
        side = hand_rows(rows, hand)
        if len(side) < 3:
            continue
        prev_wrist = prev_frame = prev_speed = None
        prev_ok = False
        for r in side:
            fi = int(r["frame"])
            if fi not in ref:
                # No face this frame, so the face-relative position is unknown.
                # Emitting [0, 0] would invent a wrist at the face centre, so the
                # row is dropped and the next row starts a fresh derivative.
                prev_frame = None
                prev_ok = False
                continue
            wx, wy = float(r["lm0_x"]), float(r["lm0_y"])
            wrist = np.array([wx, wy])
            cx, cy, fw = ref[fi]
            rel = wrist - np.array([cx, cy])
            scale = fw if (units == "fw" and fw > 0) else 1.0

            ok = prev_frame is not None and (fi - prev_frame) == SUBSAMPLE
            if ok:
                dt = SUBSAMPLE / FPS
                v = (wrist - prev_wrist) / dt / scale
                speed = float(np.linalg.norm(v))
                if speed > 1e-6:
                    dir_sin, dir_cos = v[1] / speed, v[0] / speed
                else:
                    dir_sin = dir_cos = 0.0
            else:
                v = np.zeros(2)
                speed = dir_sin = dir_cos = 0.0
            accel = ((speed - prev_speed) / (SUBSAMPLE / FPS)
                     if ok and prev_ok else 0.0)

            tips = np.array([np.linalg.norm([float(r[f"lm{ti}_x"]) - wx,
                                             float(r[f"lm{ti}_y"]) - wy])
                             for ti in [4, 8, 12, 16, 20]]) / scale
            tx, ty = float(r["lm4_x"]), float(r["lm4_y"])
            ix, iy = float(r["lm8_x"]), float(r["lm8_y"])
            aperture = float(np.linalg.norm([tx - ix, ty - iy])) / scale
            tip_xy = np.array([[float(r[f"lm{ti}_x"]), float(r[f"lm{ti}_y"])]
                               for ti in [4, 8, 12, 16, 20]])
            spread = float(np.mean([np.linalg.norm(tip_xy[i] - tip_xy[j])
                                    for i in range(5) for j in range(i + 1, 5)])) / scale

            frames.append(np.array([
                rel[0] / scale, rel[1] / scale, v[0], v[1], speed, accel,
                dir_sin, dir_cos, float(tips.mean()), float(tips.std()),
                aperture, spread,
            ]))
            prev_wrist, prev_speed, prev_frame, prev_ok = wrist, speed, fi, ok

    return np.array(frames) if frames else None


def domain_scores(view, units):
    """{(domain): (scores, K, r, n_rows)} for hands and heads."""
    out = {}
    for person, name in [("speakerB", "SpeakerB"), ("speakerA", "SpeakerA")]:
        raw = {}
        for v in vids:
            f = build_hand_frames(view, v, person, units)
            if f is not None and len(f) > 50:
                raw[v] = f
        s, k, r = gmm_scores(raw)
        out[f"Hand_{name}"] = ({key(p): x for p, x in s.items()}, k, r,
                               sum(len(x) for x in raw.values()))
    heads = maincam_head if view == "maincam" else solo_head_new
    for person, name in [("speakerB", "SpeakerB"), ("speakerA", "SpeakerA")]:
        s, k, r = gmm_scores(heads(person))
        out[f"Head_{name}"] = ({key(p): x for p, x in s.items()}, k, r,
                               None)
    return out


def average_rank(scores, pair):
    t = scores[pair]
    above = sum(1 for v in scores.values() if v > t + 1e-12)
    tied = sum(1 for v in scores.values() if abs(v - t) <= 1e-12)
    return 1 + above + (tied - 1) / 2


def consensus(dom):
    R = {d: {p: 1.0 - (average_rank(s, p) - 1) / 14.0 for p in PKEYS}
         for d, (s, *_rest) in dom.items()}
    return R, {p: float(np.mean([R[d][p] for d in R])) for p in PKEYS}


def exact_rho(side, front):
    a = np.array([side[p] for p in PKEYS])
    b = np.array([front[p] for p in PKEYS])
    obs = float(stats.spearmanr(a, b)[0])

    def pk(x, y):
        return f"({x},{y})" if x < y else f"({y},{x})"

    null = []
    for perm in permutations(vids):
        m = dict(zip(vids, perm))
        relabelled = np.array([side[pk(m[i], m[j])] for i, j in PAIRS])
        if np.std(relabelled) > 1e-12:
            null.append(float(stats.spearmanr(relabelled, b)[0]))
    null = np.array(null)
    return obs, float(np.mean(np.abs(null) >= abs(obs) - 1e-12))


def bh(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 1.0
    for rank, i in enumerate(order[::-1], 1):
        running = min(running, p[i] * m / (m - rank + 1))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", choices=["fw", "px"], default="fw")
    args = ap.parse_args()
    units = args.units

    print(f"merged GMM-UBM pipeline   units = {units}")
    dom = {}
    for view in ("maincam", "fullbody"):
        dom[view] = domain_scores(view, units)

    R = {}
    C = {}
    for view in ("maincam", "fullbody"):
        R[view], C[view] = consensus(dom[view])

    print("\nper-domain results, side view")
    print(f"  {'domain':<14}{'K':>3}{'r':>6}{'(4,5)':>11}{'rank':>7}{'rows':>8}")
    table2 = {}
    for d in sorted(dom["maincam"]):
        s, k, r, n = dom["maincam"][d]
        rk = average_rank(R["maincam"][d], TARGET)
        table2[d] = dict(dim=12 if d.startswith("Hand") else 18,
                         K=k, r=r, llr=s[TARGET], rank=rk, rows=n)
        print(f"  {d:<14}{k:>3}{r:>6}{s[TARGET]:>11.4f}{rk:>7.1f}"
              f"{'' if n is None else f'{n:>8}'}")

    print("\nper-domain results, frontal view")
    for d in sorted(dom["fullbody"]):
        s, k, r, n = dom["fullbody"][d]
        rk = average_rank(R["fullbody"][d], TARGET)
        print(f"  {d:<14}{k:>3}{r:>6}{s[TARGET]:>11.4f}{rk:>7.1f}"
              f"{'' if n is None else f'{n:>8}'}")

    order = sorted(C["maincam"].items(), key=lambda t: -t[1])
    pos = [i + 1 for i, (p, _) in enumerate(order) if p == TARGET][0]
    second = next((p, v) for p, v in order if p != TARGET)
    print(f"\n  consensus C(4,5) = {C['maincam'][TARGET]:.4f}, "
          f"rank {pos}/15")
    print(f"  next best {second[0]} = {second[1]:.4f}, "
          f"gap {C['maincam'][TARGET] - second[1]:.4f}")

    labels = ["consensus"] + sorted(dom["maincam"])
    pv, rows = [], []
    for lab in labels:
        side = C["maincam"] if lab == "consensus" else R["maincam"][lab]
        front = C["fullbody"] if lab == "consensus" else R["fullbody"][lab]
        rho, p = exact_rho(side, front)
        pv.append(p)
        rows.append((lab, rho, p))
    adj = bh(pv)

    print("\ncross-view consistency")
    for (lab, rho, p), a in zip(rows, adj):
        print(f"  {lab:<14} rho = {rho:+.3f}  p = {p:.3f}  p_BH = {a:.3f}")

    side_first = order[0][0]
    front_first = sorted(C["fullbody"].items(), key=lambda t: -t[1])[0][0]
    print(f"\n  side puts {side_first} first; frontal puts {front_first} first")

    script = Path(__file__).read_text(encoding="utf-8")
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(script.encode()).hexdigest()[:16],
        "units": units,
        "subsample": SUBSAMPLE, "fps": FPS,
        "hand_landmarks": {v: str(VIEWS[v]["lm"]) for v in VIEWS},
        "head_source": {"maincam": "output/maincam", "fullbody": "output/solo_head_std"},
        "table2": table2,
        "consensus_side": C["maincam"],
        "consensus_frontal": C["fullbody"],
        "consensus_rank_of_target": pos,
        "crossview": [{"test": l, "rho": r, "p": p, "p_bh": float(a)}
                      for (l, r, p), a in zip(rows, adj)],
        "side_first": side_first, "frontal_first": front_first,
        "not_computed": ["retrieval", "dtw", "aggregate"],
    }
    out = DEST / f"pipeline_{units}.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
