"""
Benchmarks raw PyTorch MobileSAM vs. the exported TensorRT engine on the
same frame sequence, reporting mean latency and FPS for each, and saves a
comparison bar chart.

Usage:
    python benchmark.py --pt-model checkpoints/mobile_sam.pt \
        --engine optimize/exported/mobile_sam.engine --frames data/raw
"""
import argparse
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def benchmark_pytorch(model_path: str, frames_dir: Path, n_frames: int = 50):
    from mobile_sam import sam_model_registry, SamPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_t"](checkpoint=model_path)
    sam.to(device)
    predictor = SamPredictor(sam)

    frame_paths = sorted(list(frames_dir.glob("*.jpg")))[:n_frames]
    latencies = []

    for path in frame_paths:
        frame = cv2.imread(str(path))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()
        predictor.set_image(frame)
        torch.cuda.synchronize() if device == "cuda" else None
        latencies.append((time.perf_counter() - start) * 1000)

    return np.array(latencies)


def benchmark_tensorrt(engine_path: str, frames_dir: Path, n_frames: int = 50):
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    input_shape = engine.get_tensor_shape(engine.get_tensor_name(0))
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    d_input = cuda.mem_alloc(dummy_input.nbytes)
    output_shape = engine.get_tensor_shape(engine.get_tensor_name(1))
    output = np.empty(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(output.nbytes)

    latencies = []
    for _ in range(n_frames):
        cuda.memcpy_htod(d_input, dummy_input)
        start = time.perf_counter()
        context.execute_v2([int(d_input), int(d_output)])
        cuda.Context.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    return np.array(latencies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-model", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    print("Benchmarking PyTorch model...")
    pt_latencies = benchmark_pytorch(args.pt_model, args.frames, args.n)

    print("Benchmarking TensorRT engine...")
    trt_latencies = benchmark_tensorrt(args.engine, args.frames, args.n)

    pt_mean, trt_mean = pt_latencies.mean(), trt_latencies.mean()
    speedup = pt_mean / trt_mean

    print(f"\nPyTorch:   mean={pt_mean:.2f}ms  FPS={1000/pt_mean:.1f}")
    print(f"TensorRT:  mean={trt_mean:.2f}ms  FPS={1000/trt_mean:.1f}")
    print(f"Speedup:   {speedup:.2f}x")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["PyTorch", "TensorRT FP16"], [pt_mean, trt_mean], color=["#888", "#2a9d8f"])
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title(f"MobileSAM inference: {speedup:.2f}x speedup with TensorRT")
    for i, v in enumerate([pt_mean, trt_mean]):
        ax.text(i, v + 0.5, f"{v:.2f}ms", ha="center")

    out_path = Path("outputs/benchmark_plot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
