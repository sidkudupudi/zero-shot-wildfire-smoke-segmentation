"""
Tracks the centroid of the largest smoke mask across frames, draws a
growing trajectory line, overlays an FPS counter, and stitches everything
into a final mp4.

Usage:
    python track_and_render.py --annotated outputs/annotated --out outputs/final_demo.mp4 --fps-label 45.2
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def largest_smoke_centroid(detections_for_frame, out_dir: Path):
    smoke_dets = [d for d in detections_for_frame if "smoke" in d["phrase"].lower()]
    if not smoke_dets:
        return None

    best = max(smoke_dets, key=lambda d: d["confidence"])
    mask = np.load(best["mask_path"])
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps-label", type=float, default=None,
                         help="TensorRT FPS number from benchmark.py to display on-screen")
    args = parser.parse_args()

    with open(args.annotated / "detections.json") as f:
        detections = json.load(f)

    frame_names = sorted(detections.keys())
    trajectory = []

    first_frame = cv2.imread(str(args.annotated / frame_names[0]))
    h, w = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, 5, (w, h))

    for name in frame_names:
        frame = cv2.imread(str(args.annotated / name))
        centroid = largest_smoke_centroid(detections[name], args.annotated)
        if centroid:
            trajectory.append(centroid)

        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)
        if trajectory:
            cv2.circle(frame, trajectory[-1], 6, (0, 0, 255), -1)

        if args.fps_label:
            cv2.putText(frame, f"TensorRT: {args.fps_label:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        writer.write(frame)

    writer.release()
    print(f"Wrote {len(frame_names)} frames to {args.out}")
    print(f"Smoke centroid tracked across {len(trajectory)} frames")


if __name__ == "__main__":
    main()
