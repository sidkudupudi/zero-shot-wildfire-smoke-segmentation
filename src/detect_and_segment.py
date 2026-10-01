"""
Zero-shot scene understanding pipeline: Grounding DINO detects objects from
a text prompt (no training), boxes are fed into SAM for pixel-accurate
masks. Saves annotated frames and a per-frame detections JSON.

Usage:
    python detect_and_segment.py --frames data/raw --out outputs/annotated \
        --checkpoints checkpoints/
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import groundingdino
from groundingdino.util.inference import load_model, load_image, predict, annotate
from segment_anything import sam_model_registry, SamPredictor

GROUNDING_DINO_CONFIG = os.path.join(
    os.path.dirname(groundingdino.__file__), "config", "GroundingDINO_SwinT_OGC.py"
)

TEXT_PROMPT = "smoke plume . wildfire smoke . haze . smoke . dry vegetation . power line . building"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20


def run_pipeline(frames_dir: Path, out_dir: Path, checkpoints_dir: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dino_model = load_model(
        GROUNDING_DINO_CONFIG,
        str(checkpoints_dir / "groundingdino_swint_ogc.pth"),
    )

    sam = sam_model_registry["vit_h"](checkpoint=str(checkpoints_dir / "sam_vit_h_4b8939.pth"))
    sam.to(device)
    predictor = SamPredictor(sam)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_detections = {}

    frame_paths = sorted(
        list(frames_dir.glob("*.jpg"))
        + list(frames_dir.glob("*.jpeg"))
        + list(frames_dir.glob("*.png"))
    )
    print(f"Processing {len(frame_paths)} frames...")

    for i, frame_path in enumerate(frame_paths):
        image_source, image = load_image(str(frame_path))

        boxes, logits, phrases = predict(
            model=dino_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        h, w, _ = image_source.shape
        annotated_frame = image_source.copy()
        frame_detections = []

        if len(boxes) > 0:
            predictor.set_image(image_source)

            # Grounding DINO boxes are normalized cxcywh -> convert to xyxy pixels
            boxes_xyxy = boxes.clone()
            boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
            boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
            boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
            boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h

            for box, phrase, conf in zip(boxes_xyxy, phrases, logits):
                box_np = box.cpu().numpy()
                mask, score, _ = predictor.predict(
                    box=box_np, multimask_output=False
                )
                mask = mask[0]

                # overlay mask
                color = np.random.randint(0, 255, 3)
                overlay = np.zeros_like(annotated_frame)
                overlay[mask] = color
                annotated_frame = cv2.addWeighted(annotated_frame, 1.0, overlay, 0.4, 0)

                cv2.rectangle(
                    annotated_frame,
                    (int(box_np[0]), int(box_np[1])),
                    (int(box_np[2]), int(box_np[3])),
                    color.tolist(), 2,
                )
                cv2.putText(
                    annotated_frame, f"{phrase} {float(conf):.2f}",
                    (int(box_np[0]), int(box_np[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 1,
                )

                mask_path = out_dir / f"{frame_path.stem}_{phrase.replace(' ', '_')}_mask.npy"
                np.save(mask_path, mask)

                frame_detections.append({
                    "phrase": phrase,
                    "confidence": float(conf),
                    "box_xyxy": box_np.tolist(),
                    "mask_path": str(mask_path),
                })

        out_path = out_dir / frame_path.name
        cv2.imwrite(str(out_path), cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        all_detections[frame_path.name] = frame_detections

        print(f"[{i+1}/{len(frame_paths)}] {frame_path.name}: {len(frame_detections)} detections")

    with open(out_dir / "detections.json", "w") as f:
        json.dump(all_detections, f, indent=2)

    print(f"\nDone. Annotated frames + detections.json written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    args = parser.parse_args()
    run_pipeline(args.frames, args.out, args.checkpoints)
