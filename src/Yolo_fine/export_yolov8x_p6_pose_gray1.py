"""Export a trained gray1 YOLOv8x-Pose-P6 checkpoint to a fixed-shape ONNX model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_ULTRALYTICS_ROOT = PROJECT_ROOT / "ultralytics"

if str(LOCAL_ULTRALYTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRALYTICS_ROOT))

python_path = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = os.pathsep.join(
    [item for item in (str(LOCAL_ULTRALYTICS_ROOT), python_path) if item]
)

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export gray1 YOLOv8x-Pose-P6 to ONNX")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def first_conv_in_channels(model: YOLO) -> int:
    for module in model.model.modules():
        if hasattr(module, "in_channels") and hasattr(module, "weight"):
            return int(module.in_channels)
    raise RuntimeError("Unable to locate the first convolution")


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if args.imgsz != 1280:
        raise ValueError("Gray1 FPGA ABI requires --imgsz 1280")

    model = YOLO(str(args.weights.resolve()))
    if first_conv_in_channels(model) != 1:
        raise ValueError("Checkpoint first convolution is not single-channel")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            batch=1,
            dynamic=False,
            simplify=False,
            opset=args.opset,
            half=args.half,
            device=args.device,
        )
    )
    destination = args.output_dir.resolve() / exported.name
    if exported.resolve() != destination:
        exported.replace(destination)
    print("input_semantics=gray1")
    print("expected_input_shape=[1,1,1280,1280]")
    print("expected_output_shape=[1,31,34000]")
    print(f"onnx={destination}")


if __name__ == "__main__":
    main()
