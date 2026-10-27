"""Mouth activity (MAR) for the 12 solo/fullbody recordings, saved to CSV.

Fills the gap that made mouth activity maincam-only. Uses the same measure and
threshold rule as the maincam pipeline (`src/audio/mar_all_videos.py`,
`src/audio/mouth_fullbody.py`):

    MAR = vertical lip distance / horizontal lip distance, outer lip landmarks
    resting   = 20th percentile of that recording's MAR values
    threshold = resting + 1.5 * SD
    talk_pct  = percentage of sampled frames above the threshold

Two speed-ups over `src/audio/mouth_fullbody.py`, neither of which changes the
measure (MAR is a ratio, so it is scale-invariant):

  1. face boxes are read from output/fullbody_std/features/, which Fix B already
     extracted with 100% face detection, instead of re-running YuNet;
  2. frames are downscaled to the same 1920-long-side canvas Fix B used before
     FaceMesh runs, so Stella's 4K videos cost the same as the 1080p ones.

Writes output/fullbody_mouth.csv
    group,video,mar_mean,mar_std,resting,threshold,talk_pct,samples

Usage:
    python extract_fullbody_mouth.py
"""
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BASE = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\v2")
OUT = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output")
FB = OUT / "fullbody_std"
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
              402, 317, 14, 87, 178, 88, 95]
SUBSAMPLE = 3
TARGET_LONG = 1920

mp_face_mesh = mp.solutions.face_mesh


def videos(group):
    folder = BASE / ("fullbody-Aru" if group == "Aru" else "fullbody-Stella")
    for p in sorted(folder.glob("*.MOV")):
        yield p


def solo_key(group, vpath):
    """Map a solo video file to its session number, 1-6."""
    stem = vpath.stem
    if group == "Aru":
        return f"fb_aru_{int(stem.split('_')[-1]):02d}"
    return stem


def features_path(group, vpath):
    person = "aru" if group == "Aru" else "stella"
    if group == "Aru":
        vid = f"{int(vpath.stem.split('_')[-1])}"
    else:
        vid = vpath.stem
    return FB / "features" / f"{vid}_{person}_features.csv"


def process(group, vpath):
    feat = features_path(group, vpath)
    if not feat.exists():
        print(f"  {group}/{vpath.name}: no face boxes at {feat}")
        return None

    boxes = {}
    for r in csv.DictReader(open(feat)):
        if r.get("face1_detected") == "1":
            boxes[int(r["frame"])] = (float(r["face1_x"]), float(r["face1_y"]),
                                      float(r["face1_w"]), float(r["face1_h"]))
    if not boxes:
        print(f"  {group}/{vpath.name}: no detected faces")
        return None

    cap = cv2.VideoCapture(str(vpath))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = TARGET_LONG / max(w, h)

    mar_vals = []
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.3,
                               min_tracking_confidence=0.3) as fm:
        for fi in range(0, total, SUBSAMPLE):
            if fi not in boxes:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                                   interpolation=cv2.INTER_AREA)
            x, y, bw, bh = boxes[fi]
            # Box coordinates are already in the 1920-long-side canvas that Fix B
            # wrote them in, and `frame` has just been resized to that same canvas,
            # so they must NOT be scaled again here.
            x1 = max(0, int(x) - 20)
            y1 = max(0, int(y) - 20)
            x2 = min(frame.shape[1], int(x + bw) + 20)
            y2 = min(frame.shape[0], int(y + bh) + 20)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            res = fm.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue
            hc, wc = crop.shape[:2]
            pts = np.array([[pt.x * wc, pt.y * hc]
                            for pt in res.multi_face_landmarks[0].landmark])
            lip = pts[LIPS_OUTER]
            v = float(np.linalg.norm(lip[2] - lip[6]))
            hd = float(np.linalg.norm(lip[0] - lip[4]))
            if hd > 1e-6:
                mar_vals.append(v / hd)
    cap.release()

    if not mar_vals:
        print(f"  {group}/{vpath.name}: NO MAR DATA")
        return None

    arr = np.array(mar_vals)
    resting = float(np.percentile(arr, 20))
    sd = float(np.std(arr))
    threshold = resting + 1.5 * sd
    talk = float(np.sum(arr > threshold) / len(arr) * 100)
    print(f"  {group}/{vpath.name}: MAR={arr.mean():.3f} sd={sd:.3f} "
          f"talking={talk:.1f}% ({len(arr)} samples)")
    return {"mar_mean": float(arr.mean()), "mar_std": sd, "resting": resting,
            "threshold": threshold, "talk_pct": talk, "samples": len(arr)}


def main():
    groups = sys.argv[1:] if len(sys.argv) > 1 else ["Aru", "Stella"]
    rows = []
    for group in groups:
        for vpath in videos(group):
            r = process(group, vpath)
            if r:
                rows.append((group, solo_key(group, vpath), r))

    out_path = OUT / "fullbody_mouth.csv"
    if out_path.exists() and len(groups) < 2:
        print(f"\nNot overwriting {out_path} (partial run: {groups})")
        print("Rows computed:")
        for group, name, r in rows:
            print(f"  {group},{name},{r['talk_pct']:.1f}%")
        return

    with open(out_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["group", "video", "mar_mean", "mar_std", "resting",
                     "threshold", "talk_pct", "samples"])
        for group, name, r in rows:
            wr.writerow([group, name, r["mar_mean"], r["mar_std"], r["resting"],
                         r["threshold"], r["talk_pct"], r["samples"]])
    print(f"\nSaved {len(rows)} rows to {out_path}")

    print("\nTalking percentage, solo recordings:")
    for group in ["Aru", "Stella"]:
        vals = [(n, r["talk_pct"]) for g, n, r in rows if g == group]
        print(f"  {group}: " + "  ".join(f"{n}={v:.1f}%" for n, v in vals))


if __name__ == "__main__":
    main()
