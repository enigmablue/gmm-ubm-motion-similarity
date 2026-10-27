"""Re-extract head pose from the 12 solo recordings under controlled conditions.

WHY
---
`paper_review/cell_extractor.py` shares the head features between camera
conditions, taking them from the maincam. When the head is instead taken from the
solo recordings, the SpeakerB head domain collapses (12/15) in both GMM-UBM and the
aggregate. But the solo head features (`output/speakerB/`, `output/speakerA/`) were
produced by an ad-hoc command-line run at each video's native resolution:
SpeakerB's videos 1-4 are 4K (2160x3840) and 5-6 are 1080p. Fix B rebuilt hands and
face boxes on a common canvas but never touched head pose.

This script re-extracts solo head pose the same way, but on a common 1920-long-side
canvas so all twelve recordings are processed identically. Then the "own head"
comparison can be made fairly.

WHAT IT REUSES
--------------
The same detector, landmark-to-pose solver, and CSV format as
`src/features/yunet_head_movement.py` (YuNet 2023mar + solvePnP with the generic
5-point reference model). Only two things change: the common canvas, and
processing every 5th frame, which is the rate the analysis pipeline consumes.

Output: output/solo_head_std/{vid}_{person}_features.csv
    same columns as output/maincam/{vid}_features.csv, so it is a drop-in
    replacement for the head builder.

Usage:
    python extract_solo_head_std.py            # all 12
    python extract_solo_head_std.py 4 speakerB   # one recording
"""
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\v2")
OUT = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output")
MODEL = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\face_detection_yunet_2023mar.onnx")
DEST = OUT / "solo_head_std"

TARGET_LONG = 1920
SUBSAMPLE = 5

REFERENCE_5 = np.array([
    [-30.0, 30.0, 30.0],
    [30.0, 30.0, 30.0],
    [0.0, 0.0, 50.0],
    [-25.0, -25.0, 35.0],
    [25.0, -25.0, 35.0],
], dtype=np.float64)


def head_pose_5pt(landmarks_5, image_size):
    h, w = image_size
    focal = w
    camera_matrix = np.array([[focal, 0, w / 2.0],
                              [0, focal, h / 2.0],
                              [0, 0, 1]], dtype=np.float64)
    ok, rot_vec, _ = cv2.solvePnP(REFERENCE_5, landmarks_5.astype(np.float64),
                                  camera_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return 0.0, 0.0, 0.0
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    sy = math.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(-rot_mat[2, 0], sy)
        yaw = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
        roll = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
    else:
        pitch = math.atan2(-rot_mat[2, 0], sy)
        yaw = math.atan2(-rot_mat[1, 1], rot_mat[1, 1])
        roll = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def header():
    cols = ["frame", "timestamp_sec", "face_count"]
    for f in [1, 2]:
        cols += [f"face{f}_detected", f"face{f}_x", f"face{f}_y",
                 f"face{f}_w", f"face{f}_h",
                 f"face{f}_yaw", f"face{f}_pitch", f"face{f}_roll",
                 f"face{f}_yaw_vel", f"face{f}_pitch_vel", f"face{f}_roll_vel",
                 f"face{f}_head_vel"]
    return cols


def velocity(series, times):
    """Central-difference angular velocity in deg/s over the sampled series."""
    n = len(series)
    vel = np.zeros(n)
    if n < 2:
        return vel
    for i in range(n):
        if i == 0:
            dt = times[1] - times[0]
            vel[i] = (series[1] - series[0]) / dt if dt > 1e-9 else 0.0
        elif i == n - 1:
            dt = times[-1] - times[-2]
            vel[i] = (series[-1] - series[-2]) / dt if dt > 1e-9 else 0.0
        else:
            dt = times[i + 1] - times[i - 1]
            vel[i] = (series[i + 1] - series[i - 1]) / dt if dt > 1e-9 else 0.0
    return vel


def extract(video_path, out_path):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = TARGET_LONG / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    detector = cv2.FaceDetectorYN.create(str(MODEL), "", (nw, nh), 0.5, 0.3, 5000)

    rows = []          # (frame, timestamp, detected, x, y, bw, bh, yaw, pitch, roll)
    for fi in range(0, total, SUBSAMPLE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        _, faces = detector.detect(frame)
        ts = fi / fps
        if faces is None or len(faces) == 0:
            rows.append((fi, ts, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0))
            continue
        face = max(faces, key=lambda f: f[2] * f[3])   # solo: keep the largest
        x, y, bw, bh = face[0:4]
        pts = np.array([[face[4], face[5]], [face[6], face[7]], [face[8], face[9]],
                        [face[10], face[11]], [face[12], face[13]]])
        yaw, pitch, roll = head_pose_5pt(pts, (nh, nw))
        rows.append((fi, ts, 1, float(x), float(y), float(bw), float(bh),
                     yaw, pitch, roll))
    cap.release()

    times = np.array([r[1] for r in rows])
    det = np.array([r[2] for r in rows])
    yaws = np.array([r[7] for r in rows])
    pitches = np.array([r[8] for r in rows])
    rolls = np.array([r[9] for r in rows])
    # unwrap so velocity is not corrupted by the +-180 jump
    yaws_u = np.degrees(np.unwrap(np.radians(yaws)))
    pitches_u = np.degrees(np.unwrap(np.radians(pitches)))
    rolls_u = np.degrees(np.unwrap(np.radians(rolls)))
    yv, pv, rv = velocity(yaws_u, times), velocity(pitches_u, times), velocity(rolls_u, times)
    hv = np.sqrt(yv ** 2 + pv ** 2 + rv ** 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header())
        for i, r in enumerate(rows):
            fi, ts, d, x, y, bw, bh, yaw, pitch, roll = r
            row = [fi, f"{ts:.3f}", 1 if d else 0]
            if d:
                row += [1, f"{x:.1f}", f"{y:.1f}", f"{bw:.1f}", f"{bh:.1f}",
                        f"{yaw:.4f}", f"{pitch:.4f}", f"{roll:.4f}",
                        f"{yv[i]:.3f}", f"{pv[i]:.3f}", f"{rv[i]:.3f}", f"{hv[i]:.3f}"]
            else:
                row += [0, "", "", "", "", "", "", "", "", "", "", ""]
            row += [0] + [""] * 12
            wr.writerow(row)

    rate = 100.0 * det.sum() / len(rows) if rows else 0.0
    return len(rows), rate, float(np.std(yaws_u)), float(np.std(pitches_u)), float(np.std(rolls_u))


def main():
    if len(sys.argv) >= 3:
        jobs = [(sys.argv[1], sys.argv[2])]
    else:
        jobs = [(str(v), s) for v in range(1, 7) for s in ["speakerB", "speakerA"]]

    print(f"{'recording':<24}{'rows':>7}{'face %':>9}{'sd yaw':>9}{'sd pitch':>10}{'sd roll':>9}")
    print("-" * 68)
    for vid, person in jobs:
        base = BASE / ("fullbody-SpeakerA" if person == "speakerA" else "fullbody-SpeakerB")
        pattern = f"fb_speakerA_{int(vid):02d}.MOV" if person == "speakerA" else f"{int(vid)}.MOV"
        src = base / pattern
        if not src.exists():
            print(f"{person}/{vid}: MISSING {src}")
            continue
        dst = DEST / f"{vid}_{person}_features.csv"
        n, rate, sy, sp, sr = extract(src, dst)
        print(f"{person}/{vid:<20}{n:>7}{rate:>8.1f}%{sy:>9.2f}{sp:>10.2f}{sr:>9.2f}")


if __name__ == "__main__":
    main()
