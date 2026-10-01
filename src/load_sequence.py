"""
Loads a folder of sequential wildfire lookout camera images and exposes
them as a simulated live feed (sorted by filename, which for HPWREN/AI for
Mankind style datasets typically encodes timestamp order).

Usage:
    python load_sequence.py --dir data/raw --preview
    python load_sequence.py --dir data/raw   # just validates, no display
"""
import argparse
from pathlib import Path

import cv2


class FrameSequence:
    def __init__(self, frames_dir: Path):
        self.frames = sorted(
            list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.jpeg")) + list(frames_dir.glob("*.png"))
        )
        if not self.frames:
            raise FileNotFoundError(f"No .jpg/.png frames found in {frames_dir}")

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        for path in self.frames:
            frame = cv2.imread(str(path))
            if frame is not None:
                yield path, frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    seq = FrameSequence(args.dir)
    print(f"Loaded {len(seq)} frames from {args.dir}")

    if args.preview:
        for path, frame in seq:
            cv2.imshow("Simulated feed (press q to quit)", frame)
            if cv2.waitKey(150) == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
