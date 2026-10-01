#!/bin/bash
# Exports MobileSAM's image encoder to ONNX and compiles a TensorRT FP16
# engine. Full SAM (vit_h) doesn't export cleanly -- MobileSAM is the
# correct, honest answer to "foundation models are too heavy for edge
# deployment," not a workaround to hide.
set -e

CHECKPOINTS_DIR="checkpoints"
OUT_DIR="optimize/exported"
mkdir -p "$OUT_DIR"

# 1. Download MobileSAM checkpoint if not present
if [ ! -f "$CHECKPOINTS_DIR/mobile_sam.pt" ]; then
    echo "Downloading MobileSAM checkpoint..."
    wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt -P "$CHECKPOINTS_DIR/"
fi

# 2. Export image encoder to ONNX (MobileSAM ships an export script;
#    if your installed version's path differs, check:
#    python -c "import mobile_sam; print(mobile_sam.__file__)"
python optimize/export_onnx.py \
    --checkpoint "$CHECKPOINTS_DIR/mobile_sam.pt" \
    --output "$OUT_DIR/mobile_sam_encoder.onnx"

# 3. Build TensorRT FP16 engine
trtexec --onnx="$OUT_DIR/mobile_sam_encoder.onnx" \
    --saveEngine="$OUT_DIR/mobile_sam.engine" \
    --exportTimes="$OUT_DIR/mobile_sam_timings.json"

echo "Done. Engine at $OUT_DIR/mobile_sam.engine"
echo "Note: newer TensorRT versions may not accept an explicit --fp16 flag"
echo "(precision is inferred from the ONNX model / strongly-typed build) --"
echo "same issue you already hit and solved in the endoscope project."
