"""
Score the zero-shot Grounding DINO detections against the dataset's ground-truth smoke boxes.

Every image in the AI For Mankind / HPWREN set has exactly one annotated smoke box (Pascal VOC XML).
A detection counts as "smoke" when its phrase mentions smoke or plume (Grounding DINO sometimes merges phrases,
e.g. "smoke plume wildfire smoke"). Reported per prompt version:
  * image recall      - images with at least one smoke detection anywhere
  * hit@IoU           - images where some smoke box overlaps the GT box with IoU >= 0.3 / 0.5
  * precision@0.5     - share of smoke boxes that match the GT box at IoU >= 0.5 (one GT per image)
  * AP50              - VOC-style all-point average precision, ranked by Grounding DINO confidence

Usage:
    python src/eval_against_gt.py --xmls data/annotated_bounding_box_hpwren/xmls \
        --detections v1=results/metrics/detections_v1.json v2=results/metrics/detections_v2.json --out results/metrics
"""
import argparse, json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


def load_gt(xml_dir: Path):
    gt = {}
    for x in sorted(p for p in xml_dir.glob("*.xml") if not p.name.startswith("._")):   # skip macOS resource forks
        root = ET.parse(x).getroot()
        b = root.find("object/bndbox")
        gt[root.find("filename").text] = [float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
    return gt


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def is_smoke(phrase):
    p = phrase.lower()
    return "smoke" in p or "plume" in p


def evaluate(gt, dets):
    rows, scored = [], []
    for name, g in gt.items():
        smoke = [d for d in dets.get(name, []) if is_smoke(d["phrase"])]
        ious = [iou(d["box_xyxy"], g) for d in smoke]
        rows.append(dict(image=name, smoke_boxes=len(smoke), best_iou=max(ious, default=0.0),
                         all_boxes=len(dets.get(name, []))))
        # AP bookkeeping: each GT can be matched once, highest-confidence box first
        matched = False
        for d, v in sorted(zip(smoke, ious), key=lambda t: -t[0]["confidence"]):
            tp = (v >= 0.5) and not matched
            matched = matched or tp
            scored.append((d["confidence"], tp))
    per_image = pd.DataFrame(rows)
    scored.sort(key=lambda t: -t[0])
    tp = np.cumsum([s[1] for s in scored]); fp = np.cumsum([not s[1] for s in scored])
    rec = tp / len(gt); prec = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap50 = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    n_smoke = per_image.smoke_boxes.sum()
    summary = dict(images=len(gt), smoke_boxes=int(n_smoke),
                   image_recall=float((per_image.smoke_boxes > 0).mean()),
                   hit_iou03=float((per_image.best_iou >= 0.3).mean()), hit_iou05=float((per_image.best_iou >= 0.5).mean()),
                   precision_iou05=float(tp[-1] / n_smoke) if n_smoke else 0.0, AP50=ap50,
                   mean_boxes_per_image=float(per_image.all_boxes.mean()))
    return per_image, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmls", type=Path, required=True)
    ap.add_argument("--detections", nargs="+", required=True, help="name=path/to/detections.json")
    ap.add_argument("--out", type=Path, default=Path("results/metrics"))
    args = ap.parse_args()
    gt = load_gt(args.xmls)
    summaries = {}
    for item in args.detections:
        name, path = item.split("=", 1)
        per_image, summaries[name] = evaluate(gt, json.load(open(path)))
        per_image.to_csv(args.out / f"gt_eval_per_image_{name}.csv", index=False)
    table = pd.DataFrame(summaries).T
    table.to_csv(args.out / "gt_eval_summary.csv")
    print(table.round(3).to_string())
