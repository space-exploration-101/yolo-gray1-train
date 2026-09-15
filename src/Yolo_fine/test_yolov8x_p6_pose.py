"""Evaluate the trained YOLOv8 pose model on the held-out test split.

The script reports Ultralytics' standard box/pose metrics and additional
per-keypoint coordinate statistics.  The latter are computed after matching
predictions to the YOLO ground-truth boxes, so background images and missed
objects are included in the detection summary.

Examples:
    python Yolo_fine/test_yolov8x_p6_pose.py --device 0 --batch 16
    python Yolo_fine/test_yolov8x_p6_pose.py --device 0 --save-visualizations
    python Yolo_fine/test_yolov8x_p6_pose.py --skip-val --max-images 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_ULTRALYTICS_ROOT = PROJECT_ROOT / "ultralytics"
DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "runs"
    / "landmarks_yolov8x_p6_pose_pretrained_1024_no_online_aug_b64_run2"
    / "weights"
    / "best.pt"
)
DEFAULT_DATA = PROJECT_ROOT.parent / "Dataset" / "yolo_full" / "dataset.yaml"
DEFAULT_IMAGES = PROJECT_ROOT.parent / "Dataset" / "yolo_full" / "images" / "test"
DEFAULT_LABELS = PROJECT_ROOT.parent / "Dataset" / "yolo_full" / "labels" / "test"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "test_landmarks_yolov8x_p6_pose_best"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

if str(LOCAL_ULTRALYTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRALYTICS_ROOT))

# Keep prediction and validation on the checked-out Ultralytics version.
python_path = os.environ.get("PYTHONPATH", "")
python_paths = [str(LOCAL_ULTRALYTICS_ROOT)]
if python_path:
    python_paths.append(python_path)
os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)

from ultralytics import YOLO  # noqa: E402


@dataclass
class GroundTruth:
    """One YOLO pose annotation in normalized image coordinates."""

    class_id: int
    bbox: np.ndarray  # xyxy, normalized to [0, 1]
    keypoints: np.ndarray  # (K, 3), normalized x/y and visibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLOv8 pose checkpoint on Dataset/yolo_full test images."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for validation and prediction. Reduce it if GPU memory is insufficient.",
    )
    parser.add_argument("--device", default="0", help="CUDA device, for example 0 or 0,1; use cpu for CPU.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for the detailed per-image prediction pass.",
    )
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold for prediction.")
    parser.add_argument("--match-iou", type=float, default=0.5, help="Box IoU threshold for custom matching.")
    parser.add_argument(
        "--keypoint-conf",
        type=float,
        default=0.5,
        help="Minimum per-keypoint confidence used by the custom PCK/error statistics.",
    )
    parser.add_argument(
        "--pck-thresholds",
        default="0.05,0.10,0.20",
        help="Comma-separated thresholds normalized by the ground-truth box diagonal.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Evaluate only the first N images; 0 means all.")
    parser.add_argument(
        "--samples-per-landmark",
        type=int,
        default=0,
        help="Randomly sample this many positive test images per landmark; 0 means use all images.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=20260824,
        help="Random seed used by --samples-per-landmark.",
    )
    parser.add_argument("--skip-val", action="store_true", help="Skip the official Ultralytics test validation pass.")
    parser.add_argument(
        "--skip-custom",
        action="store_true",
        help="Skip the detailed prediction pass and custom keypoint statistics.",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save annotated prediction images during the detailed prediction pass.",
    )
    parser.add_argument(
        "--save-pred-txt",
        action="store_true",
        help="Save normalized prediction TXT files during the detailed prediction pass.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable plots generated by official validation.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if not args.images.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {args.images}")
    if not args.labels.is_dir():
        raise FileNotFoundError(f"Test label directory not found: {args.labels}")
    if args.imgsz < 32:
        raise ValueError("--imgsz must be >= 32")
    if args.batch < 1:
        raise ValueError("--batch must be >= 1")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be in [0, 1]")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be in [0, 1]")
    if not 0.0 <= args.match_iou <= 1.0:
        raise ValueError("--match-iou must be in [0, 1]")
    if not 0.0 <= args.keypoint_conf <= 1.0:
        raise ValueError("--keypoint-conf must be in [0, 1]")
    if args.max_images < 0:
        raise ValueError("--max-images must be >= 0")
    if args.samples_per_landmark < 0:
        raise ValueError("--samples-per-landmark must be >= 0")


def parse_pck_thresholds(value: str) -> list[float]:
    try:
        thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --pck-thresholds value: {value!r}") from exc
    if not thresholds or any(item <= 0.0 for item in thresholds):
        raise ValueError("--pck-thresholds must contain positive numbers")
    return thresholds


def image_files(images_dir: Path, max_images: int) -> list[Path]:
    files = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if max_images:
        files = files[:max_images]
    if not files:
        raise FileNotFoundError(f"No image files found in {images_dir}")
    return files


def sample_per_landmark(
    image_paths: list[Path], labels_dir: Path, samples_per_landmark: int, seed: int
) -> tuple[list[Path], dict[int, int]]:
    """Select a deterministic random sample for each landmark class."""

    if not samples_per_landmark:
        return image_paths, {}

    by_class: dict[int, list[Path]] = {}
    for image_path in image_paths:
        annotations = read_label_file(labels_dir / f"{image_path.stem}.txt")
        for class_id in {item.class_id for item in annotations}:
            by_class.setdefault(class_id, []).append(image_path)

    rng = random.Random(seed)
    selected: set[Path] = set()
    sampled_counts: dict[int, int] = {}
    for class_id in sorted(by_class):
        candidates = sorted(by_class[class_id])
        chosen = rng.sample(candidates, min(samples_per_landmark, len(candidates)))
        selected.update(chosen)
        sampled_counts[class_id] = len(chosen)

    if not selected:
        raise ValueError("No labeled images available for --samples-per-landmark")
    return sorted(selected), sampled_counts


def read_label_file(label_file: Path) -> list[GroundTruth]:
    """Read one YOLO pose label file, including empty background files."""

    annotations: list[GroundTruth] = []
    if not label_file.is_file():
        raise FileNotFoundError(f"Label file not found: {label_file}")
    for line_number, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        if len(values) < 7:
            raise ValueError(f"Invalid label at {label_file}:{line_number}: expected bbox and keypoints")
        numeric = np.asarray([float(value) for value in values], dtype=np.float32)
        keypoint_values = numeric[5:]
        if keypoint_values.size % 3 == 0:
            keypoints = keypoint_values.reshape(-1, 3)
        elif keypoint_values.size % 2 == 0:
            keypoints = np.concatenate(
                [keypoint_values.reshape(-1, 2), np.ones((keypoint_values.size // 2, 1), dtype=np.float32)],
                axis=1,
            )
        else:
            raise ValueError(f"Invalid keypoint fields at {label_file}:{line_number}")
        x_center, y_center, width, height = numeric[1:5]
        bbox = np.asarray(
            [x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2],
            dtype=np.float32,
        )
        annotations.append(GroundTruth(int(numeric[0]), bbox, keypoints))
    return annotations


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU for arrays of xyxy boxes."""

    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection_wh = np.maximum(bottom_right - top_left, 0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    area_a = np.maximum(boxes_a[:, 2] - boxes_a[:, 0], 0.0) * np.maximum(boxes_a[:, 3] - boxes_a[:, 1], 0.0)
    area_b = np.maximum(boxes_b[:, 2] - boxes_b[:, 0], 0.0) * np.maximum(boxes_b[:, 3] - boxes_b[:, 1], 0.0)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.maximum(union, np.finfo(np.float32).eps)


def as_numpy(value: object, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Convert a torch/numpy result attribute to a CPU numpy array."""

    if value is None:
        result = np.empty(shape or (0,), dtype=np.float32)
    elif hasattr(value, "detach"):
        result = value.detach().cpu().numpy()
    else:
        result = np.asarray(value)
    return result


def result_arrays(result: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract boxes, classes, confidences, and keypoint data from a Results object."""

    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None:
        boxes = np.empty((0, 4), dtype=np.float32)
        classes = np.empty((0,), dtype=np.int64)
        confidences = np.empty((0,), dtype=np.float32)
    else:
        boxes = as_numpy(boxes_obj.xyxy).reshape(-1, 4).astype(np.float32, copy=False)
        classes = as_numpy(boxes_obj.cls).reshape(-1).astype(np.int64, copy=False)
        confidences = as_numpy(boxes_obj.conf).reshape(-1).astype(np.float32, copy=False)

    keypoints_obj = getattr(result, "keypoints", None)
    if keypoints_obj is None:
        keypoints = np.empty((0, 0, 3), dtype=np.float32)
    else:
        keypoints = as_numpy(keypoints_obj.data).astype(np.float32, copy=False)
        if keypoints.ndim == 2:
            keypoints = keypoints[None, ...]
    return boxes, classes, confidences, keypoints


def match_predictions(
    gt: list[GroundTruth],
    pred_boxes_normalized: np.ndarray,
    pred_classes: np.ndarray,
    pred_confidences: np.ndarray,
    match_iou: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedily match high-confidence predictions to same-class GT boxes."""

    if not gt:
        return [], [], list(range(len(pred_boxes_normalized)))
    ious = box_iou(pred_boxes_normalized, np.stack([item.bbox for item in gt]))
    matched: list[tuple[int, int, float]] = []
    used_gt: set[int] = set()
    false_positive: list[int] = []
    for pred_index in np.argsort(-pred_confidences):
        candidates = [
            gt_index
            for gt_index, item in enumerate(gt)
            if gt_index not in used_gt and int(pred_classes[pred_index]) == item.class_id
        ]
        if not candidates:
            false_positive.append(int(pred_index))
            continue
        best_gt = max(candidates, key=lambda item: float(ious[pred_index, item]))
        best_iou = float(ious[pred_index, best_gt])
        if best_iou >= match_iou:
            used_gt.add(best_gt)
            matched.append((int(pred_index), best_gt, best_iou))
        else:
            false_positive.append(int(pred_index))
    missed = [index for index in range(len(gt)) if index not in used_gt]
    return matched, missed, false_positive


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def landmark_from_image(path: str | Path) -> str:
    """Use the leading filename token as the landmark name (e.g. agaier_123...)."""
    return Path(path).stem.split("_")[0]


def aggregate_per_landmark(per_image: list[dict[str, object]], pck_thresholds: list[float]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in per_image:
        groups.setdefault(landmark_from_image(str(row["image"])), []).append(row)
    output: list[dict[str, object]] = []
    for landmark in sorted(groups):
        rows = groups[landmark]
        def total(field: str) -> int:
            return sum(int(row.get(field) or 0) for row in rows)
        matched = total("matched_instances")
        false_positive = total("false_positive")
        gt = total("gt_instances")
        visible = total("visible_keypoints")
        predicted = total("predicted_keypoints")
        mae_values = [float(row["keypoint_mae_px_on_predicted"]) * int(row["predicted_keypoints"])
                      for row in rows if row.get("keypoint_mae_px_on_predicted") is not None]
        result: dict[str, object] = {
            "landmark": landmark, "images": len(rows), "gt_instances": gt,
            "predicted_instances": total("predictions"), "matched_instances": matched,
            "missed_instances": total("missed_instances"), "false_positive_instances": false_positive,
            "detection_precision": safe_ratio(matched, matched + false_positive),
            "detection_recall": safe_ratio(matched, gt),
            "visible_keypoints": visible, "predicted_keypoints": predicted,
            "keypoint_recall": safe_ratio(predicted, visible),
            "keypoint_mae_px": safe_ratio(sum(mae_values), predicted),
        }
        for threshold in pck_thresholds:
            hits = sum(float(row.get(f"pck@{threshold:g}") or 0) * int(row.get("visible_keypoints") or 0) for row in rows)
            result[f"pck@{threshold:g}"] = safe_ratio(hits, visible)
        output.append(result)
    return output


def run_custom_metrics(
    model: YOLO,
    image_paths: list[Path],
    labels_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    pck_thresholds: list[float],
) -> dict[str, object]:
    """Run prediction and calculate detection and normalized keypoint metrics."""

    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    # Passing a Python list to Ultralytics 8.3.98 triggers ``autocast_list`` and
    # loads every image into memory before inference, so ``batch`` is ignored.
    # A text manifest keeps the loader on its streaming ``LoadImagesAndVideos``
    # path and preserves the requested batch size (including ``--max-images``).
    source_manifest = output_dir / "prediction_sources.txt"
    source_manifest.write_text(
        "".join(f"{path}\n" for path in image_paths),
        encoding="utf-8",
    )
    predict_kwargs = {
        "source": str(source_manifest),
        "stream": True,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": 300,
        "save": args.save_visualizations,
        "save_txt": args.save_pred_txt,
        "save_conf": args.save_pred_txt,
        "project": str(prediction_dir),
        "name": "images",
        "exist_ok": True,
        "verbose": False,
    }

    per_image: list[dict[str, object]] = []
    error_by_keypoint: list[list[float]] = []
    pck_hits_by_keypoint: list[dict[float, int]] = []
    visible_by_keypoint: list[int] = []
    predicted_by_keypoint: list[int] = []
    all_errors: list[float] = []
    all_pck_hits = {threshold: 0 for threshold in pck_thresholds}
    total_visible = 0
    total_predicted_keypoints = 0
    total_gt_instances = 0
    total_matched_instances = 0
    total_missed_instances = 0
    total_false_positive = 0
    total_predictions = 0
    total_gt_images = 0
    total_background_images = 0

    results = model.predict(**predict_kwargs)
    seen_paths: set[Path] = set()
    for result in results:
        image_path = Path(str(getattr(result, "path"))).resolve()
        if not image_path.is_file():
            # This fallback also handles predictors that return a relative path.
            image_path = next(
                (item.resolve() for item in image_paths if item.name == Path(image_path).name), image_path
            )
        seen_paths.add(image_path)
        label_file = labels_dir / f"{image_path.stem}.txt"
        gt = read_label_file(label_file)
        total_gt_instances += len(gt)
        total_gt_images += bool(gt)
        total_background_images += not bool(gt)

        image_height, image_width = getattr(result, "orig_shape", (0, 0))
        boxes, classes, confidences, keypoints = result_arrays(result)
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Invalid original image shape for {image_path}: {(image_height, image_width)}")
        scale = np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)
        pred_boxes_normalized = boxes / scale if len(boxes) else boxes.reshape(0, 4)
        total_predictions += len(boxes)
        matched, missed, false_positive = match_predictions(
            gt, pred_boxes_normalized, classes, confidences, args.match_iou
        )
        total_matched_instances += len(matched)
        total_missed_instances += len(missed)
        total_false_positive += len(false_positive)

        image_errors: list[float] = []
        image_pck_hits = {threshold: 0 for threshold in pck_thresholds}
        image_visible = 0
        image_predicted = 0
        matched_by_gt = {gt_index: pred_index for pred_index, gt_index, _matched_iou in matched}
        for gt_index, target in enumerate(gt):
            target_width = max(float(target.bbox[2] - target.bbox[0]) * image_width, 1.0)
            target_height = max(float(target.bbox[3] - target.bbox[1]) * image_height, 1.0)
            target_diagonal = math.hypot(target_width, target_height)
            pred_index = matched_by_gt.get(gt_index)
            pred_keypoints = (
                keypoints[pred_index] if pred_index is not None and pred_index < len(keypoints) else np.empty((0, 3))
            )
            for keypoint_index, target_keypoint in enumerate(target.keypoints):
                if target_keypoint[2] <= 0:
                    continue
                while len(error_by_keypoint) <= keypoint_index:
                    error_by_keypoint.append([])
                    pck_hits_by_keypoint.append({threshold: 0 for threshold in pck_thresholds})
                    visible_by_keypoint.append(0)
                    predicted_by_keypoint.append(0)
                visible_by_keypoint[keypoint_index] += 1
                total_visible += 1
                image_visible += 1
                target_xy_px = target_keypoint[:2] * np.asarray([image_width, image_height], dtype=np.float32)
                has_prediction = (
                    keypoint_index < len(pred_keypoints)
                    and pred_keypoints.shape[1] >= 3
                    and float(pred_keypoints[keypoint_index, 2]) >= args.keypoint_conf
                )
                if not has_prediction:
                    continue
                predicted_by_keypoint[keypoint_index] += 1
                total_predicted_keypoints += 1
                image_predicted += 1
                pred_xy_px = pred_keypoints[keypoint_index, :2]
                error_px = float(np.linalg.norm(pred_xy_px - target_xy_px))
                normalized_error = error_px / target_diagonal
                error_by_keypoint[keypoint_index].append(error_px)
                all_errors.append(error_px)
                image_errors.append(error_px)
                for threshold in pck_thresholds:
                    if normalized_error <= threshold:
                        all_pck_hits[threshold] += 1
                        image_pck_hits[threshold] += 1
                        pck_hits_by_keypoint[keypoint_index][threshold] += 1

        image_row: dict[str, object] = {
            "image": str(image_path),
            "gt_instances": len(gt),
            "predictions": len(boxes),
            "matched_instances": len(matched),
            "missed_instances": len(missed),
            "false_positive": len(false_positive),
            "best_match_iou": max((item[2] for item in matched), default=None),
            "visible_keypoints": image_visible,
            "predicted_keypoints": image_predicted,
            "keypoint_mae_px_on_predicted": mean_or_none(image_errors),
        }
        for threshold in pck_thresholds:
            image_row[f"pck@{threshold:g}"] = safe_ratio(image_pck_hits[threshold], image_visible)
        per_image.append(image_row)

    missing_results = [path for path in image_paths if path.resolve() not in seen_paths]
    if missing_results:
        raise RuntimeError(
            f"Predictor returned {len(missing_results)} fewer results than expected; "
            f"first missing: {missing_results[0]}"
        )

    per_keypoint: list[dict[str, object]] = []
    for keypoint_index, errors in enumerate(error_by_keypoint):
        row: dict[str, object] = {
            "keypoint": keypoint_index,
            "visible_gt": visible_by_keypoint[keypoint_index],
            "predicted": predicted_by_keypoint[keypoint_index],
            "prediction_recall": safe_ratio(predicted_by_keypoint[keypoint_index], visible_by_keypoint[keypoint_index]),
            "mae_px": mean_or_none(errors),
        }
        for threshold in pck_thresholds:
            row[f"pck@{threshold:g}"] = safe_ratio(
                pck_hits_by_keypoint[keypoint_index][threshold], visible_by_keypoint[keypoint_index]
            )
        per_keypoint.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_image.csv", per_image)
    write_csv(output_dir / "per_keypoint.csv", per_keypoint)
    per_landmark = aggregate_per_landmark(per_image, pck_thresholds)
    write_csv(output_dir / "per_landmark.csv", per_landmark)
    summary: dict[str, object] = {
        "images": len(image_paths),
        "images_with_ground_truth": total_gt_images,
        "background_images": total_background_images,
        "ground_truth_instances": total_gt_instances,
        "predicted_instances": total_predictions,
        "matched_instances": total_matched_instances,
        "missed_instances": total_missed_instances,
        "false_positive_instances": total_false_positive,
        "detection_precision_at_match_iou": safe_ratio(
            total_matched_instances, total_matched_instances + total_false_positive
        ),
        "detection_recall_at_match_iou": safe_ratio(total_matched_instances, total_gt_instances),
        "visible_keypoints": total_visible,
        "predicted_keypoints": total_predicted_keypoints,
        "keypoint_prediction_recall": safe_ratio(total_predicted_keypoints, total_visible),
        "keypoint_mae_px_on_predicted": mean_or_none(all_errors),
        "pck_normalization": "ground_truth_box_diagonal",
        "keypoint_conf_threshold": args.keypoint_conf,
        "match_iou_threshold": args.match_iou,
        "per_keypoint": per_keypoint,
        "per_landmark": per_landmark,
    }
    for threshold in pck_thresholds:
        summary[f"pck@{threshold:g}"] = safe_ratio(all_pck_hits[threshold], total_visible)
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scalar_metrics(metrics: object) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in getattr(metrics, "results_dict", {}).items():
        try:
            values[key] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    for key in (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(P)",
        "metrics/recall(P)",
        "metrics/mAP50(P)",
        "metrics/mAP50-95(P)",
    ):
        if key in metrics:
            print(f"  {key}: {metrics[key]:.6f}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    pck_thresholds = parse_pck_thresholds(args.pck_thresholds)
    image_paths = image_files(args.images, args.max_images)
    image_paths, sampled_counts = sample_per_landmark(
        image_paths, args.labels, args.samples_per_landmark, args.sample_seed
    )
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Ultralytics root: {LOCAL_ULTRALYTICS_ROOT}")
    print(f"Weights: {args.weights.resolve()}")
    sample_description = (
        f" ({len(image_paths)} selected; {args.samples_per_landmark} per landmark, seed={args.sample_seed})"
        if args.samples_per_landmark
        else f" ({len(image_paths)} selected)"
    )
    print(f"Test images: {args.images.resolve()}{sample_description}")
    print(f"Test labels: {args.labels.resolve()}")
    print(f"Output: {args.output.resolve()}")

    model = YOLO(str(args.weights.resolve()))
    report: dict[str, object] = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "images": str(args.images.resolve()),
        "labels": str(args.labels.resolve()),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "selected_images": len(image_paths),
        "samples_per_landmark": args.samples_per_landmark,
        "sample_seed": args.sample_seed,
        "sampled_counts_by_class_id": {str(key): value for key, value in sampled_counts.items()},
    }

    if not args.skip_val:
        if args.max_images:
            print(
                "Note: official validation uses the complete test split from the dataset YAML; "
                "--max-images only limits the custom pass."
            )
        print("\nRunning official Ultralytics validation on split=test ...")
        validation = model.val(
            data=str(args.data.resolve()),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            conf=0.001,
            plots=not args.no_plots,
            project=str(args.output.parent.resolve()),
            name=f"{args.output.name}_val",
            exist_ok=True,
        )
        official = scalar_metrics(validation)
        report["official_metrics"] = official
        report["official_validation_images"] = "all test images from dataset YAML"
        print_metrics("Official validation metrics", official)

    if not args.skip_custom:
        print("\nRunning detailed prediction pass and keypoint statistics ...")
        custom = run_custom_metrics(model, image_paths, args.labels, args.output, args, pck_thresholds)
        report["custom_metrics"] = custom
        print("\nCustom detection/keypoint metrics")
        for key in (
            "detection_precision_at_match_iou",
            "detection_recall_at_match_iou",
            "keypoint_prediction_recall",
            "keypoint_mae_px_on_predicted",
        ):
            print(f"  {key}: {custom[key]}")
        for threshold in pck_thresholds:
            print(f"  pck@{threshold:g}: {custom[f'pck@{threshold:g}']}")
        print(f"  Per-image CSV: {args.output / 'per_image.csv'}")
        print(f"  Per-keypoint CSV: {args.output / 'per_keypoint.csv'}")

    report_path = args.output / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nMetrics JSON: {report_path}")


if __name__ == "__main__":
    main()
