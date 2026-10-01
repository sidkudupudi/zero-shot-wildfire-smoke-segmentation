# Limitations & Scoping Notes

- **Zero-shot, not trained.** Grounding DINO and SAM are used off the shelf with text and box prompts; nothing is fine-tuned. This measures what composed foundation models can do, not what a trained smoke detector can do.
- **Prompt sensitivity (measured).** Adding the synonyms `wildfire smoke` / `smoke` and the distractor `haze` raised plume localisation (IoU ≥ 0.5) from 35.5% to 50.8% of images, but cut box precision from 57% to 24% because each synonym boxes the same plume again. Cross-phrase NMS recovers precision to 47% (AP50 0.304). Box threshold 0.25 / text threshold 0.20 for every phrase.
- **What still fails.** Small, distant and faint plumes are missed or swallowed by a whole-sky `haze` box. No smoke plume is mislabelled as another phrase (IoU ≥ 0.3 check).
- **Not measured.** False alarms on smoke-free scenes (cloud, fog, dust): all 744 evaluated images contain smoke.
- **What was optimised.** Only the segmentation encoder. MobileSAM's TinyViT encoder was exported to ONNX and built with TensorRT (FP32 graph, 4.31 ms per 1024×1024 frame). Grounding DINO stays in PyTorch because its text branch is dynamic. The detection pass itself used SAM ViT-H in PyTorch; MobileSAM is not yet wired into the pipeline, and the PyTorch-vs-TensorRT comparison (`src/benchmark.py`) was not run.
- **Dataset scope.** 744 images from many HPWREN cameras and days. It is not a time-lapse, so the centroid "tracking" video built from it is not meaningful. Tracking needs per-camera sequences.
- **Not a production system.** No per-phrase threshold calibration, no multi-camera fusion, no deployment on edge hardware (Jetson). TensorRT timings are from a desktop RTX 5080.
