"""Seven-measure aggregate, both camera views, mouth included.

Extends the paper's Evaluation 5 aggregate (CurlMean, CurlStd, CurlP25 for hands;
HeadGest, YawPitchRatio, HeadDet for head) with mouth talking percentage, now
that mouth has been extracted for the solo recordings
(`paper_review/extract_fullbody_mouth.py` -> output/fullbody_mouth.csv).

Six domains per view, mirroring the four-domain structure already used:

    Hand Stella   CurlMean, CurlStd, CurlP25
    Hand Aru      CurlMean, CurlStd, CurlP25
    Head Stella   HeadGest, YawPitchRatio, HeadDet
    Head Aru      HeadGest, YawPitchRatio, HeadDet
    Mouth Stella  talk_pct
    Mouth Aru     talk_pct

Each domain ranks the 15 pairs by the mean percentage difference of its measures
(lower = more alike). The consensus is the rank-normalised mean across the six
domains, matching the consensus rule used by the GMM-UBM pipeline.

Hand sources: maincam view from output/{vid}_hands.csv; fullbody view from
output/fullbody_std/landmarks/, with curl computed exactly as
`src/utils/hand_all_videos.py` does for the maincam.
Head source: the features for that view (maincam/{vid}_features.csv, or
stella/{vid}_features.csv and aru/fb_aru_XX_features.csv).
Mouth source: talking_pct.json (maincam) and fullbody_mouth.csv (fullbody).

Usage:
    python aggregate_seven.py
"""
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

OUT = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output")
FB = OUT / "fullbody_std"
vids = ["1", "2", "3", "4", "5", "6"]
HEAD_THRESHOLD = 10.0
FINGERS = {"thumb": (4, 3, 2), "index": (8, 6, 5), "middle": (12, 10, 9),
           "ring": (16, 14, 13), "pinky": (20, 18, 17)}
SUBSAMPLE = 5


def pct_diff(a, b):
    if a == 0 and b == 0:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b)) * 100.0


# ===================== HEAD =====================
def head_stats(vid, person, view):
    if view == "maincam":
        path = OUT / "maincam" / f"{vid}_features.csv"
        fp = "face1" if person == "stella" else "face2"
    else:
        path = (OUT / "stella" / f"{vid}_features.csv" if person == "stella"
                else OUT / "aru" / f"fb_aru_{int(vid):02d}_features.csv")
        fp = "face1"

    hv, yv, pv, det = [], [], [], 0
    for r in csv.DictReader(open(path)):
        if r.get(f"{fp}_detected") != "1":
            continue
        det += 1
        for arr, key in [(hv, "head_vel"), (yv, "yaw_vel"), (pv, "pitch_vel")]:
            v = r.get(f"{fp}_{key}", "")
            if v not in ("", None):
                arr.append(float(v))
    gest = sum(1 for i in range(1, len(hv)) if hv[i] > HEAD_THRESHOLD >= hv[i - 1])
    ratio = sum(abs(v) for v in yv) / (sum(abs(v) for v in pv) + 1e-6) if pv else 0.0
    return {"HeadGest": float(gest), "YawPitchRatio": float(ratio),
            "HeadDet": float(det)}


# ===================== HANDS =====================
def hand_stats_maincam(vid, person):
    face_key = "f1_cx" if person == "stella" else "f2_cx"
    other_key = "f2_cx" if person == "stella" else "f1_cx"
    centers = {}
    for r in csv.DictReader(open(OUT / "maincam" / f"{vid}_features.csv")):
        c = {}
        for k, fp in [("f1_cx", "face1"), ("f2_cx", "face2")]:
            ok = r.get(f"{fp}_detected") == "1" and r.get(f"{fp}_x")
            c[k] = (float(r[f"{fp}_x"]) + float(r[f"{fp}_w"]) / 2) if ok else None
        centers[int(r["frame"])] = c

    curls = []
    for hr in csv.DictReader(open(OUT / f"{vid}_hands.csv")):
        fi = int(hr["frame"])
        if fi not in centers:
            continue
        cc = centers[fi]
        for h in ["Left", "Right"]:
            if hr.get(f"{h}_detected") != "1":
                continue
            wx = float(hr[f"{h}_wrist_x"])
            d_self = abs(wx - cc[face_key]) if cc[face_key] is not None else 99999
            d_other = abs(wx - cc[other_key]) if cc[other_key] is not None else 99999
            if d_self < d_other:
                curls.append(float(hr.get(f"{h}_avg_curl") or 0))
    return _curl_summary(curls)


def _one_hand_per_frame(rows):
    by_frame = {}
    for r in rows:
        fi = int(r["frame"])
        h = (r.get("handedness") or "").strip().lower()
        cur = by_frame.get(fi)
        if cur is None or (cur[1] != "right" and h == "right"):
            by_frame[fi] = (r, h)
    return [by_frame[k][0] for k in sorted(by_frame)]


def hand_stats_fullbody(vid, person):
    lm = FB / "landmarks" / f"{vid}_{person}_hand_landmarks.csv"
    ft = FB / "features" / f"{vid}_{person}_features.csv"
    if not lm.exists() or not ft.exists():
        return None
    face_ok = {int(r["frame"]) for r in csv.DictReader(open(ft))
               if r.get("face1_detected") == "1"}
    curls = []
    for r in _one_hand_per_frame(list(csv.DictReader(open(lm)))):
        fi = int(r["frame"])
        if fi % SUBSAMPLE != 0 or fi not in face_ok:
            continue
        pts = {i: np.array([float(r[f"lm{i}_x"]), float(r[f"lm{i}_y"])])
               for i in range(21)}
        scores = []
        for tip, pip, mcp in FINGERS.values():
            denom = np.linalg.norm(pts[pip] - pts[mcp]) + 1e-6
            scores.append(np.linalg.norm(pts[tip] - pts[mcp]) / denom)
        curls.append(float(np.mean(scores)))
    return _curl_summary(curls)


def _curl_summary(curls):
    if len(curls) <= 10:
        return {"CurlMean": 0.0, "CurlStd": 0.0, "CurlP25": 0.0}
    a = np.array(curls)
    return {"CurlMean": float(np.mean(a)), "CurlStd": float(np.std(a)),
            "CurlP25": float(np.percentile(a, 25))}


# ===================== MOUTH =====================
def mouth_stats():
    mc = json.load(open(OUT / "talking_pct.json"))
    maincam = {p: {v: mc[v][p]["talk_pct"] for v in vids} for p in ["stella", "aru"]}

    fullbody = {"stella": {}, "aru": {}}
    mp = OUT / "fullbody_mouth.csv"
    if mp.exists():
        for r in csv.DictReader(open(mp)):
            group = "aru" if r["group"] == "Aru" else "stella"
            name = r["video"]
            vid = (f"{int(name.split('_')[-1])}" if group == "aru" else name)
            if vid in vids:
                fullbody[group][vid] = float(r["talk_pct"])
    return maincam, fullbody


# ===================== BUILD + RANK =====================
def domain_rank(stats, keys):
    rows = []
    for v1, v2 in combinations(vids, 2):
        if v1 not in stats or v2 not in stats:
            continue
        d = [pct_diff(stats[v1][k], stats[v2][k]) for k in keys]
        rows.append((f"({v1},{v2})", float(np.mean(d))))
    rows.sort(key=lambda x: x[1])
    return {n: i + 1 for i, (n, _) in enumerate(rows)}, rows


def build(view, mouth_mc, mouth_fb):
    head, hand, mouth = {}, {}, (mouth_mc if view == "maincam" else mouth_fb)
    for person in ["stella", "aru"]:
        head[person] = {v: head_stats(v, person, view) for v in vids}
        if view == "maincam":
            hand[person] = {v: hand_stats_maincam(v, person) for v in vids}
        else:
            hand[person] = {}
            for v in vids:
                s = hand_stats_fullbody(v, person)
                if s:
                    hand[person][v] = s
        mouth[person] = {v: {"talk_pct": mouth[person][v]}
                         for v in vids if v in mouth[person]}
    return head, hand, mouth


def report(view, head, hand, mouth):
    domains = []
    for person in ["stella", "aru"]:
        domains.append((f"Hand {person.capitalize()}", hand[person],
                        ["CurlMean", "CurlStd", "CurlP25"]))
        domains.append((f"Head {person.capitalize()}", head[person],
                        ["HeadGest", "YawPitchRatio", "HeadDet"]))
        domains.append((f"Mouth {person.capitalize()}", mouth[person],
                        ["talk_pct"]))

    print(f"\n{view.upper()}")
    print("-" * 66)
    print(f"{'domain':<16}{'rank(4,5)':>11}{'(4,5)':>9}{'best pair':>12}")
    ranks = {}
    for name, stats, keys in domains:
        if not stats or len(stats.get("stella", stats.get("aru", {}))) < 6:
            pass
        r, rows = domain_rank(stats, keys)
        if not rows:
            continue
        ranks[name] = r
        pos = r["(4,5)"]
        val = next(x for x in rows if x[0] == "(4,5)")[1]
        print(f"{name:<16}{pos:>8}/15{val:>8.1f}%{rows[0][0]:>12}")

    C = {}
    for v1, v2 in combinations(vids, 2):
        n = f"({v1},{v2})"
        C[n] = float(np.mean([1.0 - (ranks[d][n] - 1) / 14.0 for d in ranks]))
    ordered = sorted(C.items(), key=lambda x: x[1], reverse=True)
    pos = [i for i, (n, _) in enumerate(ordered, 1) if n == "(4,5)"][0]
    others = [c for n, c in C.items() if n != "(4,5)"]
    print(f"{'CONSENSUS':<16}{pos:>8}/15{C['(4,5)']:>8.3f}{ordered[0][0]:>12}")
    print(f"   (consensus is over {len(ranks)} domains; "
          f"next best {ordered[1][0]} C={ordered[1][1]:.3f}, "
          f"gap {C['(4,5)'] - ordered[1][1]:+.3f})")
    return pos


def main():
    mouth_mc, mouth_fb = mouth_stats()
    missing = [f"{p}{v}" for p in ["stella", "aru"] for v in vids
               if v not in mouth_fb[p]]
    if missing:
        print("WARNING: no fullbody mouth for:", ", ".join(missing))

    print("=" * 66)
    print("SEVEN-MEASURE AGGREGATE (mouth included), both camera views")
    print("=" * 66)

    res = {}
    for view in ["maincam", "fullbody"]:
        head, hand, mouth = build(view, mouth_mc, mouth_fb)
        res[view] = report(view, head, hand, mouth)

    print()
    print("=" * 66)
    print("3 METHODS x 2 VIEWS  --  rank of (4,5) out of 15")
    print("=" * 66)
    print(f"{'method':<28}{'maincam':>14}{'fullbody':>14}")
    print("-" * 66)
    print(f"{'Aggregate (7 measures)':<28}"
          f"{str(res['maincam']) + '/15':>14}{str(res['fullbody']) + '/15':>14}")
    print(f"{'GMM-UBM':<28}{'1/15':>14}{'1/15':>14}")
    print(f"{'DTW':<28}{'15/15':>14}{'14/15':>14}")


if __name__ == "__main__":
    main()
