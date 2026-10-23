#!/usr/bin/env python3
"""Prepare a Gray1 pose dataset while preserving upstream train/test splits.

The upstream generator owns the split: source ``train`` becomes output ``train``
and source ``test`` becomes output ``test``.  There is no independent validation
set upstream, so this script deliberately maps YAML ``val`` to ``images/test``.
It never moves a source-training example to validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


TARGET_SIZE = 1280
EXPECTED_SOURCE_CLASSES = 20
EXPECTED_MODEL_CLASSES = 21
LABEL_FIELD_COUNT = 11
SPLITS = ("train", "test")
# Pillow < 9 exposes resampling filters directly on Image.
RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Completed upstream dataset containing data.yaml and train/test.")
    parser.add_argument(
        "--names-source", type=Path,
        help="Optional data.yaml used only when --source/data.yaml is not available (useful for a smoke run while generation is active).",
    )
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; it must not already exist.")
    parser.add_argument("--dataset-path", type=Path, required=True,
                        help="Path written to output dataset.yaml (usually the container path).")
    parser.add_argument("--size", type=int, default=TARGET_SIZE,
                        help=f"Square Gray1 image size (default: {TARGET_SIZE}).")
    parser.add_argument(
        "--max-per-split", type=int,
        help="Process at most this many sorted samples from each split (smoke testing only).",
    )
    return parser.parse_args()


def load_names(source: Path, names_source: Path | None) -> list[str]:
    data_file = source / "data.yaml"
    if not data_file.is_file() and names_source is not None:
        data_file = names_source.resolve()
    if not data_file.is_file():
        raise FileNotFoundError(f"Missing completed upstream data.yaml: {data_file}")
    config = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    names = config.get("names")
    if isinstance(names, dict):
        try:
            names = [names[index] for index in range(len(names))]
        except KeyError as exc:
            raise ValueError("Source names must have contiguous class IDs starting at 0") from exc
    # Accept a previously prepared Gray1 dataset as a source as well.  Such a
    # YAML carries the reserved, intentionally unrepresented model class 20.
    if isinstance(names, list) and len(names) == EXPECTED_MODEL_CLASSES and names[-1] == "unrepresented_class_20":
        names = names[:-1]
    if not isinstance(names, list) or len(names) != EXPECTED_SOURCE_CLASSES:
        raise ValueError("Source data.yaml must define exactly 20 ordered classes")
    return [str(name) for name in names]


def split_files(source: Path, split: str) -> list[Path]:
    image_dir = source / "images" / split
    label_dir = source / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing source image/label directories for split {split!r}")
    images = sorted(image_dir.glob("*.png"))
    if not images:
        raise ValueError(f"Source split {split!r} contains no PNG images; is generation complete?")
    image_names = {path.stem for path in images}
    label_names = {path.stem for path in label_dir.glob("*.txt")}
    if image_names != label_names:
        missing_labels = sorted(image_names - label_names)[:5]
        missing_images = sorted(label_names - image_names)[:5]
        raise ValueError(
            f"Source {split!r} image/label mismatch; "
            f"missing labels={missing_labels}, missing images={missing_images}"
        )
    return images


def validate_label(path: Path) -> int:
    objects = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != LABEL_FIELD_COUNT:
            raise ValueError(f"{path}:{line_number}: expected {LABEL_FIELD_COUNT} fields")
        values = [float(field) for field in fields]
        class_id = int(values[0])
        if values[0] != class_id or not 0 <= class_id < EXPECTED_SOURCE_CLASSES:
            raise ValueError(f"{path}:{line_number}: invalid class ID")
        if not all(0.0 <= value <= 1.0 for value in values[1:5]) or min(values[3:5]) <= 0:
            raise ValueError(f"{path}:{line_number}: invalid normalized box")
        for start in (5, 8):
            x, y, visibility = values[start : start + 3]
            if visibility not in (0, 1, 2):
                raise ValueError(f"{path}:{line_number}: invalid keypoint visibility")
            if visibility and not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"{path}:{line_number}: invalid keypoint")
        objects += 1
    return objects


def convert_image(source: Path, target: Path, size: int) -> None:
    with Image.open(source) as opened:
        image = opened.convert("L").resize((size, size), RESAMPLE_BILINEAR)
        image.save(target, format="PNG", optimize=False)


def write_dataset_yaml(path: Path, dataset_path: Path, names: list[str]) -> None:
    # ``val`` intentionally aliases the physical test directory: the generator
    # supplies only train/test, and preserving train is a hard requirement.
    config = {
        "path": str(dataset_path),
        "train": "images/train",
        "val": "images/test",
        "test": "images/test",
        "channels": 1,
        "input_semantics": "gray1",
        "kpt_shape": [2, 3],
        "flip_idx": [0, 1],
        "names": {index: name for index, name in enumerate(names)},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def copy_metadata(source: Path, output: Path, expected_ids: set[str]) -> bool:
    metadata = source / "metadata.csv"
    if not metadata.is_file():
        return False
    with metadata.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = {row.get("sample_id", "") for row in rows}
    if ids != expected_ids:
        raise ValueError("Source metadata sample IDs do not exactly match source images")
    shutil.copy2(metadata, output / "metadata.csv")
    (output / "metadata.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return True


def main() -> None:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be positive")
    if args.max_per_split is not None and args.max_per_split <= 0:
        raise ValueError("--max-per-split must be positive")
    source = args.source.resolve()
    output = args.output.resolve()
    partial = output.with_name(f"{output.name}.partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {partial}")

    names = load_names(source, args.names_source)
    model_names = names + ["unrepresented_class_20"]
    source_images = {split: split_files(source, split) for split in SPLITS}
    if args.max_per_split is not None:
        source_images = {
            split: images[:args.max_per_split] for split, images in source_images.items()
        }
    summary: dict[str, dict[str, int]] = {}
    expected_ids: set[str] = set()
    try:
        for split, images in source_images.items():
            image_dir = partial / "images" / split
            label_dir = partial / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            objects = 0
            for index, image in enumerate(images, 1):
                label = source / "labels" / split / f"{image.stem}.txt"
                objects += validate_label(label)
                convert_image(image, image_dir / image.name, args.size)
                shutil.copy2(label, label_dir / label.name)
                expected_ids.add(image.stem)
                if index % 500 == 0 or index == len(images):
                    print(f"{split}: {index}/{len(images)}", flush=True)
            summary[split] = {"images": len(images), "labels": len(images), "objects": objects}

        metadata_copied = False
        # A partial smoke run intentionally has fewer image IDs than source metadata.
        if args.max_per_split is None:
            metadata_copied = copy_metadata(source, partial, expected_ids)
        # The validation and test entries deliberately refer to exactly one set
        # of files; do not create a duplicate val directory.
        summary["val"] = {**summary["test"], "source": "test"}
        write_dataset_yaml(partial / "dataset.yaml", args.dataset_path, model_names)
        manifest = {
            "purpose": "Gray1 pose dataset preserving upstream train/test splits",
            "source": str(source),
            "conversion": f"RGB converted to one-channel BT.601 luminance and bilinear-resized to {args.size}x{args.size}",
            "split_policy": {
                "train": "upstream train copied without resplitting",
                "val": "aliases upstream test via dataset.yaml val: images/test",
                "test": "upstream test copied without resplitting",
            },
            "metadata_copied": metadata_copied,
            "max_per_split": args.max_per_split,
            "class_names": model_names,
            "unrepresented_class": 20,
            "splits": summary,
        }
        manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        (partial / "manifest.json").write_bytes(manifest_bytes)
        (partial / "VERIFIED").write_text(
            "prepared_by=prepare_gray1_preserve_upstream_splits.py\n"
            f"manifest_sha256={hashlib.sha256(manifest_bytes).hexdigest()}\n"
            f"train_images={summary['train']['images']}\n"
            f"val_images={summary['val']['images']} (aliases test)\n"
            f"test_images={summary['test']['images']}\n",
            encoding="utf-8",
        )
        partial.rename(output)
    except Exception:
        # Keep the partial directory for inspection/recovery; never publish it.
        raise
    print(f"GRAY1_PRESERVE_SPLITS_PASS output={output}")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GRAY1_PRESERVE_SPLITS_FAIL: {exc}", file=sys.stderr)
        raise
