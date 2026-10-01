"""
Exports MobileSAM's image encoder (the heavy, per-frame part of the
model) to ONNX. The pip-installed mobile_sam package doesn't ship an
export script (that lives only in the GitHub repo's scripts/ folder,
which isn't packaged into the wheel), so this does it directly: load
the checkpoint, pull out the image_encoder submodule, export with a
dummy input of the shape the encoder expects.

Usage:
    python export_onnx.py --checkpoint checkpoints/mobile_sam.pt \
        --output optimize/exported/mobile_sam_encoder.onnx
"""
import argparse
from pathlib import Path

import torch
from mobile_sam import sam_model_registry


def export(checkpoint: str, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)

    sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
    sam.eval()

    image_encoder = sam.image_encoder
    dummy_input = torch.randn(1, 3, 1024, 1024)

    torch.onnx.export(
        image_encoder,
        dummy_input,
        str(output),
        input_names=["image"],
        output_names=["image_embeddings"],
        opset_version=17,
        dynamic_axes=None,
        dynamo=False,
    )

    print(f"Exported image encoder to {output}")
    print(f"Input: image [1, 3, 1024, 1024]")
    print("Next: trtexec --onnx=<this file> --saveEngine=mobile_sam.engine")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output)
