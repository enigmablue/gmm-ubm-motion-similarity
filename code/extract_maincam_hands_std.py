"""Re-extract maincam hand landmarks at the stride the analysis actually uses.

THE BUG
-------
`src/features/extract_landmarks.py` extracts maincam hands every 3rd frame. The
analysis reads every 5th frame. Only frames that are multiples of both (every
15th) survive, so two thirds of the recorded hand data is discarded, and what
remains yields velocity over ~0.5 s instead of ~0.17 s.

The fullbody hand files were extracted every 5th frame, so they are unaffected.
Consequence: "hand speed" and "hand acceleration" meant different things in the
two camera views, on top of the different normalisation units.

WHAT THIS DOES
--------------
Re-runs MediaPipe Hands on the six maincam videos at every 5th frame, at native
resolution (1920x1080, which already equals the fullbody canvas convention of
"longer side 1920", so no resize is needed). Hands are assigned to SpeakerB or SpeakerA
by wrist x-proximity to the face box centre, as the original script did.

Writes output/maincam_hands_std/{vid}_{person}_hand_landmarks.csv
    frame,timestamp,handedness,lm0_x..lm20_z     same columns as the originals

Originals in output/landmarks/ are left untouched.

Usage:
    python extract_maincam_hands_std.py          # all six videos
    python extract_maincam_hands_std.py 4        # one video
"""
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp

BASE = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\v2\maincam")
OUT = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output")
DEST = OUT / "maincam_hands_std"
SUBSAMPLE = 5


def face_centres(vid):
    """frame -> (speakerB_cx, speakerA_cx), None where not detected."""
    out = {}
    for r in csv.DictReader(open(OUT / "maincam" / f"{vid}_features.csv")):
        c = {}
        for name, fp in [("speakerB", "face1"), ("speakerA", "face2")]:
            ok = r.get(f"{fp}_detected") == "1" and r.get(f"{fp}_x")
            c[name] = ((float(r[f"{fp}_x"]) + float(r[f"{fp}_w"]) / 2)
                       if ok else None)
        out[int(r["frame"])] = (c["speakerB"], c["speakerA"])
    return out


def extract(vid):
    video = BASE / f"{vid}.MOV"
    if not video.exists():
        print(f"  missing {video}")
        return None, None
    centres = face_centres(vid)

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = {"speakerB": [], "speakerA": []}
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as hands:
        for fi in range(0, total, SUBSAMPLE):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.multi_hand_landmarks:
                continue
            sc, ac = centres.get(fi, (None, None))
            for idx, lm in enumerate(res.multi_hand_landmarks):
                label = res.multi_handedness[idx].classification[0].label
                pts = [(p.x * w, p.y * h, p.z) for p in lm.landmark]
                wx, wy = pts[0][0], pts[0][1]
                d_s = abs(wx - sc) if sc is not None else 99999
                d_a = abs(wx - ac) if ac is not None else 99999
                if sc is None and ac is None:
                    continue
                person = "speakerB" if d_s <= d_a else "speakerA"
                rows[person].append((fi, f"{fi / fps:.3f}", label, pts))
    cap.release()

    DEST.mkdir(parents=True, exist_ok=True)
    counts = {}
    for person in ["speakerB", "speakerA"]:
        path = DEST / f"{vid}_{person}_hand_landmarks.csv"
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            hdr = ["frame", "timestamp", "handedness"]
            for i in range(21):
                hdr += [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"]
            wr.writerow(hdr)
            for fi, ts, label, pts in rows[person]:
                row = [fi, ts, label]
                for x, y, z in pts:
                    row += [f"{x:.4f}", f"{y:.4f}", f"{z:.6f}"]
                wr.writerow(row)
        frames = len({r[0] for r in rows[person]})
        counts[person] = frames
    return counts, (total // SUBSAMPLE + 1)


def main():
    vids = sys.argv[1:] if len(sys.argv) > 1 else ["1", "2", "3", "4", "5", "6"]
    print(f"{'video':<7}{'speakerB frames':>15}{'speakerA frames':>13}{'extracted':>11}")
    print("-" * 48)
    for v in vids:
        counts, expected = extract(v)
        if counts is None:
            continue
        print(f"{v:<7}{counts['speakerB']:>15}{counts['speakerA']:>13}{expected:>11}")
    print(f"\nWrote to {DEST}")
    print("Compare with output/landmarks/: those files hold every 3rd frame, so the")
    print("analysis could only read about a third of them.")


if __name__ == "__main__":
    main()
