"""Table 7 rebuilt: hand-representation ablation on the corrected source.

Why this exists
---------------
`hand_ablations.py` reads `output/landmarks`, which is sampled at stride 3, and
filters it with SUBSAMPLE=5, so it actually runs at ~2 Hz. Its finger builder
also carries one `prev_wrist` across both hands, so 23% of its derivative rows
compare a left-hand landmark with a right-hand one. The "12D combined" row was
therefore scoring a different representation from the one the pipeline now
scores, and its 3/4/.857 no longer reproduces the corrected pipeline.

This script changes exactly one thing relative to the original: the data the
four hand builders read. Every builder now reads `output/maincam_hands_std`
(stride 5, ~6 Hz), differences within each hand, and refuses to form a
derivative across a gap or across a missing face row. The GMM machinery, the
head features, the consensus formula and the pair set are unchanged.

Four configurations, all normalised the same way:

  23D wrist-relative  five fingertips (xyz, wrist-relative) + five tip speeds
                      + three wrist terms
  8D  wrist only      columns 0-7 of the shared 12D base
  4D  finger only     columns 8-11 of the shared 12D base
  12D combined        the full base vector

Slicing the same base vector for the last three forces them to differ only in
which features they see, not in how the features were computed.

WRITES  output/paper_review/final/hand_ablation_{units}.json
Usage:  python pipeline_hand_ablation.py [--units fw|px]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from gmm_pure_fullbody import maincam_head  # noqa: E402
from pipeline import (DEST, FPS, SUBSAMPLE, VIEWS, average_rank,  # noqa: E402
                      build_hand_frames, consensus, face_ref, hand_rows, key,
                      vids)
from review2_raw_scores import gmm_scores  # noqa: E402

TIPS = [4, 8, 12, 16, 20]
PERSON = [("stella", "Stella"), ("aru", "Aru")]
MANUSCRIPT = {
    "23D wrist-relative": (23, 8, 8, 0.696, 2),
    "8D wrist only": (8, 4, 5, 0.821, 1),
    "4D finger only": (4, 3, 10, 0.750, 1),
    "12D combined": (12, 3, 4, 0.857, 1),
}


def build_23d(view, vid, person, units):
    """23D wrist-relative encoding, same rows and gap rule as the 12D builder."""
    lm_path = VIEWS[view]["lm"] / f"{vid}_{person}_hand_landmarks.csv"
    if not lm_path.exists():
        return None
    rows = list(__import__("csv").DictReader(open(lm_path)))
    if len(rows) < 30:
        return None
    ref = face_ref(view, vid, person)
    ws = [v[2] for v in ref.values() if v[2] > 0]
    fw_fallback = float(np.median(ws)) if ws else 1.0

    out = []
    for hand in ("left", "right"):
        side = hand_rows(rows, hand)
        if len(side) < 3:
            continue
        prev_tips = prev_wrist = None
        prev_frame = None
        prev_ok = False
        for r in side:
            fi = int(r["frame"])
            if fi not in ref:
                prev_frame = None
                prev_ok = False
                continue
            wrist = np.array([float(r["lm0_x"]), float(r["lm0_y"]),
                              float(r["lm0_z"])])
            cx, cy, fw = ref[fi]
            scale = fw if (units == "fw" and fw > 0) else 1.0

            tips = np.concatenate([
                np.array([float(r[f"lm{t}_x"]), float(r[f"lm{t}_y"]),
                          float(r[f"lm{t}_z"])]) - wrist
                for t in TIPS]) / scale

            ok = prev_frame is not None and (fi - prev_frame) == SUBSAMPLE
            if ok:
                dt = SUBSAMPLE / FPS
                speeds = np.linalg.norm((tips - prev_tips).reshape(5, 3),
                                        axis=1) / dt
                wrist_speed = (float(np.linalg.norm(wrist[:2] - prev_wrist[:2]))
                               / dt / scale)
            else:
                speeds = np.zeros(5)
                wrist_speed = 0.0

            out.append(np.concatenate([
                tips,
                speeds,
                [(float(r["lm0_x"]) - cx) / scale,
                 (float(r["lm0_y"]) - cy) / scale,
                 wrist_speed],
            ]))
            prev_tips, prev_wrist = tips, wrist
            prev_frame, prev_ok = fi, ok
    return np.array(out) if out else None


def config_frames(name, view, units):
    """{person: {vid: frames}} for one hand configuration."""
    got = {}
    for person, _ in PERSON:
        raw = {}
        for v in vids:
            if name == "12D combined":
                f = build_hand_frames(view, v, person, units)
            elif name == "23D wrist-relative":
                f = build_23d(view, v, person, units)
            else:
                base = build_hand_frames(view, v, person, units)
                f = None if base is None else (
                    base[:, 0:8] if name == "8D wrist only" else base[:, 8:12])
            if f is not None and len(f) > 50:
                raw[v] = f
        got[person] = raw
    return got


def score(raw):
    s, k, r = gmm_scores(raw)
    return None if s is None else ({key(p): x for p, x in s.items()}, k, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", choices=["fw", "px"], default="fw")
    ap.add_argument("--view", default="maincam")
    args = ap.parse_args()
    units, view = args.units, args.view

    heads = {}
    for person, name in PERSON:
        s, k, r = gmm_scores(maincam_head(person))
        heads[f"Head_{name}"] = ({key(p): x for p, x in s.items()}, k, r, None)
        print(f"head {name}: K={k}, r={r}  (held fixed across all rows)")

    print(f"\nhand representation ablation, {units}, {view}")
    print(f"{'configuration':<22}{'dims':>5}{'Hand-St':>9}{'Hand-Ar':>9}"
          f"{'C45':>8}{'cons':>8}   manuscript")
    payload = {}
    for name in MANUSCRIPT:
        got = config_frames(name, view, units)
        dom = dict(heads)
        dims = None
        for person, namep in PERSON:
            s, k, r = score(got[person])
            if s is None:
                print(f"  {name}: no data for {person}")
                return
            dims = got[person][vids[0]].shape[1]
            dom[f"Hand_{namep}"] = (s, k, r, None)
        _, cons = consensus(dom)
        c45 = cons["(4,5)"]
        order = sorted(cons.values(), reverse=True)
        cons_rank = 1 + sum(1 for x in order if x > c45 + 1e-12)
        r = {d: average_rank(dom[d][0], "(4,5)")
             for d in ("Hand_Stella", "Hand_Aru")}

        old = MANUSCRIPT[name]
        print(f"{name:<22}{dims:>5}{r['Hand_Stella']:>9.0f}"
              f"{r['Hand_Aru']:>9.0f}{c45:>8.3f}{cons_rank:>6}/15"
              f"   {old[1]}/{old[2]}/{old[3]:.3f}/{old[4]}")
        payload[name] = dict(dims=int(dims),
                             Hand_St=float(r["Hand_Stella"]),
                             Hand_Ar=float(r["Hand_Aru"]),
                             C45=float(c45), cons_rank=int(cons_rank),
                             K={p: dom[f"Hand_{n}"][1] for p, n in PERSON},
                             r={p: dom[f"Hand_{n}"][2] for p, n in PERSON})

    out = DEST / f"hand_ablation_{units}.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}")
    print("The 12D row must match the headline for these units; if it does not, "
          "the ablation is not on the same footing as the pipeline.")


if __name__ == "__main__":
    main()
