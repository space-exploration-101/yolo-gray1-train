"""Train a grayscale-native YOLOv8x-pose-p6 model from scratch at 640 or 1280.

This model family uses one grayscale channel and does not use pretrained
three-channel weights or the legacy RChannelOnly transform.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import tempfile
from pathlib import Path

import cv2
import yaml


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
from ultralytics.utils import LOCAL_RANK, yaml_load


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
        default=640,
        help="Square training image size; supported values are 640 and 1280.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Images per batch; use -1 for Ultralytics AutoBatch.",
    )
    parser.add_argument("--device", default="cpu", help="Explicit device; GPU use requires a separate readiness gate.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early-stopping patience; 0 disables early stopping (default: 0).",
    )
    parser.add_argument("--box-gain", type=float, default=7.5)
    parser.add_argument("--pose-gain", type=float, default=12.0)
    parser.add_argument("--kobj-gain", type=float, default=1.0)
    parser.add_argument("--coord-loss-gain", type=float, default=0.02)
    parser.add_argument("--coord-loss-beta-px", type=float, default=2.0)
    parser.add_argument(
        "--coord-loss-start-epoch",
        type=int,
        default=5,
        help="Keep coordinate-loss gain at zero for this many initial epochs.",
    )
    parser.add_argument(
        "--coord-loss-ramp-epochs",
        type=int,
        default=15,
        help="Half-cosine ramp length after --coord-loss-start-epoch.",
    )
    parser.add_argument("--pixel-metric-conf", type=float, default=0.25)
    parser.add_argument("--pixel-metric-match-iou", type=float, default=0.5)
    parser.add_argument("--pixel-metric-kpt-conf", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.1)
    parser.add_argument("--copy-paste", type=float, default=0.1)
    parser.add_argument("--flipud", type=float, default=0.5)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--degrees", type=float, default=10.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=2.0)
    parser.add_argument("--perspective", type=float, default=0.0005)
    parser.add_argument("--erasing", type=float, default=0.2)
    parser.add_argument(
        "--multi-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable dynamic 0.5x-1.5x batch resizing around --imgsz (default: disabled for fixed-size training).",
    )
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
    parser.add_argument("--val-plot-period", type=int, default=10)
    parser.add_argument("--test-plot-period", type=int, default=20)
    parser.add_argument("--plot-sample-stride", type=int, default=5)
    parser.add_argument("--plot-min-images", type=int, default=10)
    parser.add_argument(
        "--difficulty-curriculum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train cumulatively on easy (epochs 1-10), then medium (11), hard (21), and extreme (51) views.",
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
    if args.patience < 0:
        raise ValueError("--patience must be >= 0.")
    if args.box_gain <= 0 or args.pose_gain <= 0 or args.kobj_gain <= 0:
        raise ValueError("Loss gains must be positive.")
    if args.coord_loss_gain < 0:
        raise ValueError("--coord-loss-gain must be >= 0.")
    if args.coord_loss_beta_px <= 0:
        raise ValueError("--coord-loss-beta-px must be > 0.")
    if args.coord_loss_start_epoch < 0 or args.coord_loss_ramp_epochs < 1:
        raise ValueError("Coordinate-loss start epoch must be >= 0 and ramp epochs must be >= 1.")
    for name in ("pixel_metric_conf", "pixel_metric_match_iou", "pixel_metric_kpt_conf"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    for name in ("mosaic", "mixup", "copy_paste", "flipud", "fliplr", "erasing"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    if args.degrees < 0 or args.translate < 0 or args.scale < 0 or args.shear < 0 or args.perspective < 0:
        raise ValueError("Geometric augmentation magnitudes must be >= 0.")
    if args.translate > 1 or args.scale > 1 or args.perspective > 0.001:
        raise ValueError("Require translate/scale <= 1 and perspective <= 0.001.")
    if args.save_period < 1:
        raise ValueError("--save-period must be >= 1.")
    if args.val_plot_period < 1 or args.test_plot_period < 1:
        raise ValueError("Periodic plot intervals must be >= 1.")
    if args.plot_sample_stride < 1 or args.plot_min_images < 1:
        raise ValueError("Plot sample stride and minimum image count must be >= 1.")
    if args.imgsz not in (640, 1280):
        raise ValueError("This training recipe requires --imgsz 640 or 1280.")

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


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
BRIGHT_BLUE_BGR = (255, 191, 0)  # RGB #00BFFF (DeepSkyBlue)
EPOCH_CHECKPOINT_PATTERN = re.compile(r"epoch(\d+)\.pt")
EPOCH_CHECKPOINTS_TO_KEEP = 5


def split_image_paths(data_file: Path, split: str) -> list[Path]:
    """Resolve one image-directory split from an Ultralytics dataset YAML."""
    config = yaml_load(data_file)
    root = Path(config.get("path", data_file.parent))
    if not root.is_absolute():
        root = (data_file.parent / root).resolve()
    entries = config.get(split)
    if entries is None:
        raise ValueError(f"Dataset has no {split!r} split: {data_file}")
    entries = entries if isinstance(entries, list) else [entries]
    paths: list[Path] = []
    for entry in entries:
        location = Path(entry)
        if not location.is_absolute():
            location = root / location
        if location.is_dir():
            paths.extend(
                path for path in location.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        elif location.is_file():
            paths.extend(
                Path(line.strip()) for line in location.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        else:
            raise FileNotFoundError(f"Dataset {split} path does not exist: {location}")
    result = sorted({path.resolve() for path in paths})
    if not result:
        raise FileNotFoundError(f"Dataset {split} split contains no images")
    return result


def periodic_plot_sample(
    paths: list[Path], stride: int, minimum: int, seed: int
) -> list[Path]:
    """Randomly sample about one image in every N, reproducibly per epoch/split."""
    shuffled = paths.copy()
    random.Random(seed).shuffle(shuffled)
    sampled = shuffled[::stride]
    if len(sampled) < min(minimum, len(paths)):
        sampled = shuffled[:minimum]
    return sorted(sampled)


def save_bright_blue_prediction(result, output_path: Path) -> None:
    """Save an overlay with bright-blue detection boxes, labels, and keypoints."""
    annotated = result.orig_img.copy()
    # Predictions originate from Gray1 arrays. OpenCV ignores all but the first
    # color component when drawing onto a one-channel canvas, which made the
    # previous boxes and label backgrounds appear white.
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    elif annotated.shape[2] == 1:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    height, width = annotated.shape[:2]
    boxes = result.boxes
    if boxes is not None:
        line_width = max(round(sum(annotated.shape[:2]) * 0.003 / 2), 2)
        font_scale = line_width / 3
        text_thickness = max(line_width - 1, 1)
        for detection in boxes:
            x1, y1, x2, y2 = (int(round(value)) for value in detection.xyxy.squeeze().tolist())
            x1, x2 = sorted((max(0, min(x1, width - 1)), max(0, min(x2, width - 1))))
            y1, y2 = sorted((max(0, min(y1, height - 1)), max(0, min(y2, height - 1))))
            class_id = int(detection.cls.item())
            class_name = result.names[class_id]
            label = f"{class_name} {float(detection.conf.item()):.2f}"
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                BRIGHT_BLUE_BGR,
                thickness=line_width,
                lineType=cv2.LINE_AA,
            )
            (_, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_thickness,
            )
            text_y = y1 - baseline - 3 if y1 >= text_height + baseline + 6 else y1 + text_height + baseline + 3
            text_origin = (x1, min(text_y, height - baseline - 1))
            # A thin dark outline keeps the blue text readable over bright image regions.
            cv2.putText(
                annotated,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness=text_thickness + 2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                BRIGHT_BLUE_BGR,
                thickness=text_thickness,
                lineType=cv2.LINE_AA,
            )
    keypoints = result.keypoints
    if keypoints is not None:
        keypoint_radius = 5
        for detection_keypoints in keypoints.data.detach().cpu().numpy():
            for keypoint in detection_keypoints:
                x_coord, y_coord = float(keypoint[0]), float(keypoint[1])
                if len(keypoint) == 3 and float(keypoint[2]) < 0.5:
                    continue
                if not (0 < x_coord < width and 0 < y_coord < height):
                    continue
                cv2.circle(
                    annotated,
                    (int(round(x_coord)), int(round(y_coord))),
                    keypoint_radius,
                    BRIGHT_BLUE_BGR,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
    if not cv2.imwrite(str(output_path), annotated):
        raise OSError(f"Unable to save periodic visualization: {output_path}")


def prune_epoch_checkpoints(weights_dir: Path, keep: int = EPOCH_CHECKPOINTS_TO_KEEP) -> list[Path]:
    """Delete old epochN.pt files while leaving best.pt and last.pt untouched."""
    checkpoints: list[tuple[int, Path]] = []
    for checkpoint in weights_dir.glob("epoch*.pt"):
        match = EPOCH_CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
        if match is not None and checkpoint.is_file():
            checkpoints.append((int(match.group(1)), checkpoint))
    checkpoints.sort(key=lambda item: (item[0], item[1].name))
    removed: list[Path] = []
    for _, checkpoint in checkpoints[:-keep]:
        try:
            checkpoint.unlink()
            removed.append(checkpoint)
        except FileNotFoundError:
            continue
        except OSError as error:
            print(f"WARNING: unable to remove old checkpoint {checkpoint}: {error}", flush=True)
    return removed


def install_periodic_visualizations(model: YOLO, args: argparse.Namespace) -> None:
    """Save periodic Gray1 prediction overlays without changing training metrics."""

    def render_split(trainer, epoch: int, split: str) -> None:
        all_paths = split_image_paths(args.data.resolve(), split)
        paths = periodic_plot_sample(
            all_paths,
            args.plot_sample_stride,
            args.plot_min_images,
            seed=epoch * 2 + (0 if split == "val" else 1),
        )
        output_dir = trainer.save_dir / "periodic_visualizations" / f"epoch_{epoch:03d}" / split
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "epoch": epoch,
            "split": split,
            "source_images": len(all_paths),
            "selection": "deterministic_random",
            "sample_stride": args.plot_sample_stride,
            "selected_images": len(paths),
            "files": [str(path) for path in paths],
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        training_batch_size = args.batch if args.batch > 0 else max(1, int(trainer.batch_size))
        # Never hand the live training module to the predictor: AutoBackend fuses it
        # and sets every parameter's requires_grad=False. A fresh checkpoint-backed
        # model keeps the next training epoch's autograd graph intact.
        visualizer = YOLO(str(trainer.last))
        batch_size = min(training_batch_size, 4)
        predict_kwargs = {
            "imgsz": args.imgsz,
            "batch": batch_size,
            "device": str(trainer.device),
            "stream": False,
            "verbose": False,
            "save": False,
        }
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise FileNotFoundError(f"Unable to read Gray1 image: {path}")
                images.append(gray[..., None])
            results = visualizer.predict(source=images, **predict_kwargs)
            if len(results) != len(batch_paths):
                raise RuntimeError(
                    f"Periodic {split} prediction count mismatch: "
                    f"{len(results)} != {len(batch_paths)}"
                )
            for source_path, result in zip(batch_paths, results, strict=True):
                save_bright_blue_prediction(
                    result,
                    output_dir / f"{source_path.stem}_pred.jpg",
                )
        del visualizer
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"PERIODIC_VISUALIZATION split={split} epoch={epoch} "
            f"selected={len(paths)} output={output_dir}",
            flush=True,
        )

    def on_fit_epoch_end(trainer) -> None:
        if getattr(trainer, "rank", -1) not in (-1, 0):
            return
        epoch = int(trainer.epoch) + 1
        if epoch % args.val_plot_period == 0:
            render_split(trainer, epoch, "val")
        if epoch % args.test_plot_period == 0:
            render_split(trainer, epoch, "test")
        removed = prune_epoch_checkpoints(Path(trainer.wdir))
        if removed:
            print(
                "EPOCH_CHECKPOINT_RETENTION "
                f"keep={EPOCH_CHECKPOINTS_TO_KEEP} removed={','.join(path.name for path in removed) or 'none'}",
                flush=True,
            )

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def coordinate_loss_gain(epoch: int, final_gain: float, start_epoch: int, ramp_epochs: int) -> float:
    """Return a zero-then-half-cosine coordinate-loss gain for a one-based epoch."""
    if epoch <= start_epoch:
        return 0.0
    progress = min(max((epoch - start_epoch) / ramp_epochs, 0.0), 1.0)
    return final_gain * 0.5 * (1.0 - math.cos(math.pi * progress))


def install_coordinate_loss_schedule(model: YOLO, args: argparse.Namespace) -> None:
    """Apply the coordinate-loss schedule; the logged coord_loss is its weighted contribution."""

    def on_train_epoch_start(trainer) -> None:
        epoch = int(trainer.epoch) + 1
        gain = coordinate_loss_gain(
            epoch,
            args.coord_loss_gain,
            args.coord_loss_start_epoch,
            args.coord_loss_ramp_epochs,
        )
        train_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        if getattr(train_model, "criterion", None) is None:
            train_model.criterion = train_model.init_criterion()
        if not hasattr(train_model.criterion, "coord_gain"):
            raise TypeError("Active pose criterion does not support coordinate loss.")
        train_model.criterion.coord_gain = gain
        trainer.coord_loss_gain_current = gain
        print(
            f"COORD_LOSS_SCHEDULE epoch={epoch} gain={gain:.8f} "
            f"beta_px={args.coord_loss_beta_px:.3f}",
            flush=True,
        )

    def on_train_epoch_end(trainer) -> None:
        trainer.lr["coord/gain"] = float(getattr(trainer, "coord_loss_gain_current", 0.0))

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)


def prepare_difficulty_curriculum(data_file: Path) -> tuple[Path, dict[int, tuple[tuple[str, ...], Path, int]]]:
    """Write stage image lists/YAMLs from metadata and return the epoch schedule."""
    config = yaml_load(data_file)
    root = Path(config.get("path", data_file.parent))
    if not root.is_absolute():
        root = (data_file.parent / root).resolve()
    metadata_path = root / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Difficulty curriculum requires {metadata_path}")
    with metadata_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    difficulty_by_id = {
        row["sample_id"]: row["difficulty"]
        for row in rows
        if row.get("split") == "train"
    }
    allowed = ("easy", "medium", "hard", "extreme")
    if not difficulty_by_id or set(difficulty_by_id.values()) - set(allowed):
        raise ValueError("Training metadata must declare easy/medium/hard/extreme difficulty for every train sample")
    train_paths = split_image_paths(data_file, "train")
    missing = [path.stem for path in train_paths if path.stem not in difficulty_by_id]
    if missing:
        raise ValueError(f"Metadata lacks difficulty for train samples, e.g. {missing[:5]}")
    # Ultralytics exposes epochs one-based to callbacks.  Keep the requested
    # inclusive stages: 1-10, 11-20, 21-50, and 51 onward.
    stages = ((1, ("easy",)), (11, ("easy", "medium")), (21, ("easy", "medium", "hard")), (51, allowed))
    # Training datasets are mounted read-only in the release container, so
    # stage lists must live in a writable runtime directory rather than beside
    # dataset.yaml.
    directory = Path(tempfile.gettempdir()) / f"gray1-difficulty-curriculum-{data_file.stem}"
    directory.mkdir(exist_ok=True)
    schedule: dict[int, tuple[tuple[str, ...], Path, int]] = {}
    for start_epoch, difficulties in stages:
        images = [path for path in train_paths if difficulty_by_id[path.stem] in difficulties]
        if not images:
            raise ValueError(f"Curriculum stage {difficulties} has no training images")
        tag = "_".join(difficulties)
        image_list = directory / f"train_{tag}.txt"
        image_list.write_text("".join(f"{path}\n" for path in images), encoding="utf-8")
        stage_config = dict(config)
        stage_config["train"] = str(image_list)
        stage_yaml = directory / f"dataset_{tag}.yaml"
        stage_yaml.write_text(yaml.safe_dump(stage_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        schedule[start_epoch] = (difficulties, stage_yaml, len(images))
    return schedule[1][1], schedule


def install_difficulty_curriculum(model: YOLO, schedule: dict[int, tuple[tuple[str, ...], Path, int]]) -> None:
    """Replace the train loader at curriculum boundaries while retaining optimizer state."""
    def on_train_epoch_start(trainer) -> None:
        epoch = int(trainer.epoch) + 1
        stage = schedule.get(epoch)
        if stage is None:
            return
        difficulties, stage_yaml, count = stage
        stage_config = yaml_load(stage_yaml)
        trainer.data["train"] = stage_config["train"]
        trainer.trainset = stage_config["train"]
        world_size = max(int(getattr(trainer, "world_size", 1)), 1)
        trainer.train_loader = trainer.get_dataloader(
            trainer.trainset, batch_size=trainer.batch_size // world_size, rank=LOCAL_RANK, mode="train"
        )
        print(
            f"DIFFICULTY_CURRICULUM epoch={epoch} difficulties={','.join(difficulties)} images={count}",
            flush=True,
        )
    model.add_callback("on_train_epoch_start", on_train_epoch_start)


def main() -> None:
    args = parse_args()
    validate_args(args)

    model = YOLO(str(args.resume.resolve())) if args.resume is not None else YOLO(str(args.model_config.resolve()), task="pose")
    if first_conv_in_channels(model) != 1:
        raise ValueError("The model first convolution is not single-channel.")
    install_periodic_visualizations(model, args)
    install_coordinate_loss_schedule(model, args)
    curriculum_data = args.data.resolve()
    if args.difficulty_curriculum:
        curriculum_data, curriculum_schedule = prepare_difficulty_curriculum(args.data.resolve())
        install_difficulty_curriculum(model, curriculum_schedule)
    train_kwargs = {
        "data": str(curriculum_data),
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
        "patience": args.patience,
        "box": args.box_gain,
        "pose": args.pose_gain,
        "kobj": args.kobj_gain,
        "coord_beta_px": args.coord_loss_beta_px,
        "pixel_metric_conf": args.pixel_metric_conf,
        "pixel_metric_match_iou": args.pixel_metric_match_iou,
        "pixel_metric_kpt_conf": args.pixel_metric_kpt_conf,
        "cache": False,
        "multi_scale": args.multi_scale,
        "plots": True,
        # Keep color augmentation disabled for native grayscale input, while
        # enabling spatial/mixing augmentations requested for pose training.
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "bgr": 0.0,
        "erasing": args.erasing,
    }
    if args.resume is not None:
        train_kwargs["resume"] = True

    print(f"Ultralytics root: {LOCAL_ULTRALYTICS_ROOT}")
    print(f"Model config: {args.model_config.resolve()}")
    print(f"Initialization: {'gray1 resume checkpoint' if args.resume else 'from scratch'}")
    print(f"Input contract: gray1 [B,1,{args.imgsz},{args.imgsz}]")
    print(f"Dataset: {args.data.resolve()}")
    if args.difficulty_curriculum:
        print("Difficulty curriculum: 1-10 easy; 11-20 +medium; 21-50 +hard; 51+ +extreme")
    print(f"Output: {(args.project / args.name).resolve()}")
    print(f"Periodic checkpoint: every {args.save_period} epochs")
    print(
        "Coordinate loss: "
        f"Smooth L1 beta={args.coord_loss_beta_px:g}px, "
        f"gain=0 for epochs 1-{args.coord_loss_start_epoch}, "
        f"half-cosine ramp over {args.coord_loss_ramp_epochs} epochs to {args.coord_loss_gain:g}"
    )
    print(
        "Pixel eval: "
        "L1 |dx|+|dy| per keypoint, "
        f"conf>={args.pixel_metric_conf:g}, box IoU>={args.pixel_metric_match_iou:g}, "
        f"keypoint conf>={args.pixel_metric_kpt_conf:g}"
    )
    print(
        "Augmentations: "
        f"mosaic={args.mosaic:g}, mixup={args.mixup:g}, copy_paste={args.copy_paste:g}, "
        f"flipud={args.flipud:g}, fliplr={args.fliplr:g}, degrees={args.degrees:g}, "
        f"translate={args.translate:g}, scale={args.scale:g}, shear={args.shear:g}, "
        f"perspective={args.perspective:g}, erasing={args.erasing:g}, multi_scale={args.multi_scale}"
    )
    print(
        "Periodic visualizations: "
        f"val every {args.val_plot_period} epochs, "
        f"test every {args.test_plot_period} epochs, "
        f"sample stride {args.plot_sample_stride}, minimum {args.plot_min_images} images"
    )
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
