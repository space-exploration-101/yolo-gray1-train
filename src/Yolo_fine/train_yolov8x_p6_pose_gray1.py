"""Train a grayscale-native YOLOv8x-pose-p6 model from scratch.

The input ABI is one grayscale channel at 1280x1280. This model family does not
use pretrained three-channel weights or the legacy RChannelOnly transform.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_ULTRALYTICS_ROOT = PROJECT_ROOT / "ultralytics"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "yolov8x-pose-p6-gray1.yaml"
DEFAULT_PROJECT = Path("/experiments")

if str(LOCAL_ULTRALYTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRALYTICS_ROOT))

# DDP child processes import Ultralytics independently, so propagate the local
# checkout explicitly instead of allowing a globally installed version.
python_path = os.environ.get("PYTHONPATH", "")
python_paths = [str(LOCAL_ULTRALYTICS_ROOT)]
if python_path:
    python_paths.append(python_path)
os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)

from ultralytics import YOLO
from ultralytics.utils import yaml_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train grayscale-native YOLOv8x-pose-p6 from scratch."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Fixed FPGA-compatible training image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Images per batch; use -1 for Ultralytics AutoBatch.",
    )
    parser.add_argument("--device", default="cpu", help="Explicit device; GPU use requires a separate readiness gate.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="landmarks_yolov8x_p6_pose_gray1_from_scratch")
    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Save a periodic checkpoint every N epochs (default: 10).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume only from a compatible gray1 checkpoint.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {args.data}\n"
            "Run prepare_r_channel_yolo_dataset.py first."
        )
    if not args.model_config.is_file():
        raise FileNotFoundError(f"Model config not found: {args.model_config}")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.imgsz < 32:
        raise ValueError("--imgsz must be >= 32.")
    if args.batch == 0 or args.batch < -1:
        raise ValueError("--batch must be -1 or a positive integer.")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0.")
    if args.save_period < 1:
        raise ValueError("--save-period must be >= 1.")
    if args.imgsz != 1280:
        raise ValueError("Gray1 FPGA ABI requires --imgsz 1280.")

    model_cfg = yaml_load(args.model_config)
    if model_cfg.get("ch") != 1 or model_cfg.get("nc") != 21 or model_cfg.get("kpt_shape") != [2, 3]:
        raise ValueError("Model config must declare ch: 1, nc: 21, and kpt_shape: [2, 3].")

    data_cfg = yaml_load(args.data)
    if data_cfg.get("channels") != 1 or data_cfg.get("input_semantics") != "gray1":
        raise ValueError("Dataset YAML must declare channels: 1 and input_semantics: gray1.")
    if data_cfg.get("r_channel_only", False):
        raise ValueError("Gray1 datasets must not enable r_channel_only.")
    if data_cfg.get("kpt_shape") != [2, 3] or len(data_cfg.get("names", [])) != 21:
        raise ValueError("Dataset must declare 21 classes and kpt_shape: [2, 3].")


def first_conv_in_channels(model: YOLO) -> int:
    for module in model.model.modules():
        if hasattr(module, "in_channels") and hasattr(module, "weight"):
            return int(module.in_channels)
    raise RuntimeError("Unable to locate the first convolution.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    model = YOLO(str(args.resume.resolve())) if args.resume is not None else YOLO(str(args.model_config.resolve()), task="pose")
    if first_conv_in_channels(model) != 1:
        raise ValueError("The model first convolution is not single-channel.")
    train_kwargs = {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(args.project.resolve()),
        "name": args.name,
        "save": True,
        "save_period": args.save_period,
        "amp": True,
        "patience": 100,
        "cache": False,
        "multi_scale": False,
        "plots": True,
        # yolo_full is already augmented offline by generate_landmark_aug.py.
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "bgr": 0.0,
        "erasing": 0.0,
    }
    if args.resume is not None:
        train_kwargs["resume"] = True

    print(f"Ultralytics root: {LOCAL_ULTRALYTICS_ROOT}")
    print(f"Model config: {args.model_config.resolve()}")
    print(f"Initialization: {'gray1 resume checkpoint' if args.resume else 'from scratch'}")
    print("Input contract: gray1 [B,1,1280,1280]")
    print(f"Dataset: {args.data.resolve()}")
    print(f"Output: {(args.project / args.name).resolve()}")
    print(f"Periodic checkpoint: every {args.save_period} epochs")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
