#!/usr/bin/env python3
"""Build a self-contained Gray1 training smoke dataset from a YOLO pose pilot.

The source pilot is test-only, RGB, and has 20 populated classes.  This tool
creates an explicitly non-production smoke dataset with a deterministic
per-primary-landmark 80/20 train/validation split.  Images are converted to
true one-channel PNGs and resized from 1024x1024 to 1280x1280.  Normalized
YOLO boxes and keypoints therefore remain unchanged.

The Gray1 model contract requires 21 class names.  Index 20 is deliberately
named ``unrepresented_smoke_class_20``; it has no samples and must not be
interpreted as a real class during model-quality evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TARGET_SIZE = 1280
EXPECTED_SOURCE_CLASSES = 20
EXPECTED_MODEL_CLASSES = 21
LABEL_FIELD_COUNT = 11  # class, xywh, 2 * (x, y, visibility)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Source test-only YOLO pose dataset.")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; it must not exist.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path recorded in output dataset.yaml (the container-visible output path).",
    )
    parser.add_argument("--seed", type=int, default=20261021, help="Deterministic split seed.")
    parser.add_argument(
        "--val-per-primary",
        type=int,
        default=4,
        help="Validation images selected from each primary-landmark group.",
    )
    return parser.parse_args()


def load_source_names(source: Path) -> list[str]:
    data_file = source / "data.yaml"
    if not data_file.is_file():
        raise FileNotFoundError(f"Source data.yaml is missing: {data_file}")
    with data_file.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, dict):
        names = [names[index] for index in range(len(names))]
    if not isinstance(names, list) or len(names) != EXPECTED_SOURCE_CLASSES:
        raise ValueError("Source dataset must provide exactly 20 ordered class names")
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("Source dataset contains an invalid class name")
    return names


def source_images(source: Path) -> list[Path]:
    images_dir = source / "images" / "test"
    labels_dir = source / "labels" / "test"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError("Source must contain images/test and labels/test")
    images = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError("Source images/test is empty")
    missing_labels = [path.name for path in images if not (labels_dir / f"{path.stem}.txt").is_file()]
    if missing_labels:
        raise ValueError(f"Source images without labels: {missing_labels[:5]}")
    image_stems = {path.stem for path in images}
    orphan_labels = sorted(path.name for path in labels_dir.glob("*.txt") if path.stem not in image_stems)
    if orphan_labels:
        raise ValueError(f"Source labels without images: {orphan_labels[:5]}")
    return images


def primary_group(path: Path) -> str:
    match = re.match(r"^(.*)_xyz_opk_\d+$", path.stem)
    if not match:
        raise ValueError(f"Cannot determine primary-landmark group from {path.name}")
    return match.group(1)


def split_images(images: list[Path], seed: int, val_per_primary: int) -> dict[str, list[Path]]:
    if val_per_primary < 1:
        raise ValueError("--val-per-primary must be at least 1")
    groups: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        groups[primary_group(image)].append(image)
    split = {"train": [], "val": []}
    for group_name, members in sorted(groups.items()):
        members = sorted(members)
        if len(members) <= val_per_primary:
            raise ValueError(f"Group {group_name!r} has insufficient images for train/val split")
        group_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{group_name}".encode("utf-8")).digest()[:8], "big"
        )
        val_names = {path.name for path in random.Random(group_seed).sample(members, val_per_primary)}
        split["val"].extend(path for path in members if path.name in val_names)
        split["train"].extend(path for path in members if path.name not in val_names)
    return {name: sorted(paths) for name, paths in split.items()}


def validate_label(label_path: Path) -> int:
    count = 0
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != LABEL_FIELD_COUNT:
            raise ValueError(f"{label_path}:{line_number}: expected {LABEL_FIELD_COUNT} fields, got {len(fields)}")
        try:
            values = [float(field) for field in fields]
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: non-numeric field") from exc
        class_id = int(values[0])
        if class_id != values[0] or not 0 <= class_id < EXPECTED_SOURCE_CLASSES:
            raise ValueError(f"{label_path}:{line_number}: class must be an integer in [0, 19]")
        x_center, y_center, width, height = values[1:5]
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{label_path}:{line_number}: invalid normalized xywh")
        for keypoint_start in (5, 8):
            x, y, visibility = values[keypoint_start : keypoint_start + 3]
            if visibility not in (0, 1, 2):
                raise ValueError(f"{label_path}:{line_number}: keypoint visibility must be 0, 1, or 2")
            if visibility and not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError(f"{label_path}:{line_number}: visible keypoint is outside normalized range")
        count += 1
    return count


def write_dataset_yaml(path: Path, dataset_path: Path, names: list[str]) -> None:
    config = {
        "path": str(dataset_path),
        "train": "images/train",
        "val": "images/val",
        "channels": 1,
        "input_semantics": "gray1",
        "kpt_shape": [2, 3],
        "flip_idx": [0, 1],
        "names": {index: name for index, name in enumerate(names)},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def validate_output(output: Path) -> dict[str, dict[str, int]]:
    with (output / "dataset.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("channels") != 1 or config.get("input_semantics") != "gray1":
        raise ValueError("Output dataset.yaml does not declare Gray1 semantics")
    if config.get("kpt_shape") != [2, 3] or len(config.get("names", [])) != EXPECTED_MODEL_CLASSES:
        raise ValueError("Output dataset.yaml does not declare the required 21-class Gray1 model contract")

    summary: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        images = sorted(path for path in image_dir.glob("*.png") if path.is_file())
        labels = sorted(path for path in label_dir.glob("*.txt") if path.is_file())
        if not images or len(images) != len(labels):
            raise ValueError(f"Output {split} images and labels are missing or unmatched")
        if {path.stem for path in images} != {path.stem for path in labels}:
            raise ValueError(f"Output {split} image/label filenames do not match")
        objects = 0
        for image_path in images:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None or image.ndim != 2 or image.shape != (TARGET_SIZE, TARGET_SIZE):
                raise ValueError(f"Output image is not {TARGET_SIZE}x{TARGET_SIZE} Gray1: {image_path}")
            objects += validate_label(label_dir / f"{image_path.stem}.txt")
        summary[split] = {"images": len(images), "labels": len(labels), "objects": objects}
    return summary


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    partial = output.with_name(f"{output.name}.partial")
    if partial.exists():
        raise FileExistsError(f"Refusing to reuse partial output: {partial}")

    names = load_source_names(source)
    names.append("unrepresented_smoke_class_20")
    images = source_images(source)
    split = split_images(images, args.seed, args.val_per_primary)

    for split_name in ("train", "val"):
        (partial / "images" / split_name).mkdir(parents=True)
        (partial / "labels" / split_name).mkdir(parents=True)

    labels_dir = source / "labels" / "test"
    for split_name, paths in split.items():
        for source_image in paths:
            source_label = labels_dir / f"{source_image.stem}.txt"
            validate_label(source_label)
            bgr = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Unable to read source image: {source_image}")
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
            target_image = partial / "images" / split_name / f"{source_image.stem}.png"
            if not cv2.imwrite(str(target_image), resized):
                raise OSError(f"Unable to write target image: {target_image}")
            shutil.copy2(source_label, partial / "labels" / split_name / source_label.name)

    write_dataset_yaml(partial / "dataset.yaml", args.dataset_path, names)
    output_summary = validate_output(partial)
    manifest = {
        "purpose": "functional smoke test only; not a production-quality dataset",
        "source": str(source),
        "conversion": "RGB source converted with OpenCV BGR-to-gray; resized 1024x1024 to 1280x1280",
        "seed": args.seed,
        "val_per_primary": args.val_per_primary,
        "class_names": names,
        "unrepresented_class": 20,
        "splits": output_summary,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (partial / "manifest.json").write_bytes(manifest_bytes)
    (partial / "VERIFIED").write_text(
        "prepared_by=prepare_gray1_smoke_dataset.py\n"
        f"manifest_sha256={hashlib.sha256(manifest_bytes).hexdigest()}\n"
        f"train_images={output_summary['train']['images']}\n"
        f"val_images={output_summary['val']['images']}\n",
        encoding="utf-8",
    )
    partial.rename(output)
    print(f"GRAY1_SMOKE_DATASET_PASS output={output}")
    print(json.dumps(output_summary, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GRAY1_SMOKE_DATASET_FAIL: {exc}", file=sys.stderr)
        raise
