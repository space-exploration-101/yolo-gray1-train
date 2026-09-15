"""Train YOLOv8x-pose-p6 on the generated landmark dataset.

The script intentionally imports the local Ultralytics 8.3.98 checkout under
H:/auto_landmark/Yolo_fine/ultralytics instead of a globally installed package.
Periodic checkpoints are written with Ultralytics' official save_period option.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_ULTRALYTICS_ROOT = PROJECT_ROOT / "ultralytics"
DEFAULT_WEIGHTS = LOCAL_ULTRALYTICS_ROOT / "yolov8x-pose-p6.pt"
DEFAULT_DATA = PROJECT_ROOT.parent / "Dataset" / "yolo_full" / "dataset.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8x-pose-p6 for the 20-landmark pose dataset."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Training image size. Increase only when GPU memory allows.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Images per batch; use -1 for Ultralytics AutoBatch.",
    )
    parser.add_argument("--device", default="0", help="CUDA device, for example 0 or 0,1.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="landmarks_yolov8x_p6_pose")
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
        help="Resume from a previous checkpoint instead of the pretrained weights.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {args.data}\n"
            "Run prepare_r_channel_yolo_dataset.py first."
        )
    if not args.weights.is_file() and args.resume is None:
        raise FileNotFoundError(f"Pretrained weights not found: {args.weights}")
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


def main() -> None:
    args = parse_args()
    validate_args(args)

    checkpoint = args.resume if args.resume is not None else args.weights
    model = YOLO(str(checkpoint))
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
    print(f"Model: {checkpoint.resolve()}")
    print(f"Dataset: {args.data.resolve()}")
    print(f"Output: {(args.project / args.name).resolve()}")
    print(f"Periodic checkpoint: every {args.save_period} epochs")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
