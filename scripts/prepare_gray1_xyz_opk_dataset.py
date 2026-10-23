#!/usr/bin/env python3
"""Prepare a stratified Gray1 train/val/test dataset from XYZ/OPK output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


TARGET_SIZE = 1280
EXPECTED_SOURCE_CLASSES = 20
EXPECTED_MODEL_CLASSES = 21
LABEL_FIELD_COUNT = 11
IMAGE_PATTERN = re.compile(r"^(?P<group>landmark_\d{2}_.+)_xyz_opk_(?P<number>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--train-per-primary", type=int, default=480)
    parser.add_argument("--val-per-primary", type=int, default=60)
    parser.add_argument("--test-per-primary", type=int, default=60)
    return parser.parse_args()


def load_names(source: Path) -> list[str]:
    with (source / "data.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    names = config.get("names")
    if isinstance(names, dict):
        names = [names[index] for index in range(len(names))]
    if not isinstance(names, list) or len(names) != EXPECTED_SOURCE_CLASSES:
        raise ValueError("Source data.yaml must define exactly 20 ordered classes")
    return [str(name) for name in names]


def grouped_images(source: Path) -> dict[str, list[Path]]:
    image_dir = source / "images/test"
    label_dir = source / "labels/test"
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(image_dir.glob("*.png")):
        match = IMAGE_PATTERN.match(path.stem)
        if match is None:
            raise ValueError(f"Unexpected source image name: {path.name}")
        if not (label_dir / f"{path.stem}.txt").is_file():
            raise FileNotFoundError(f"Missing label for {path.name}")
        groups[match.group("group")].append(path)
    if len(groups) != EXPECTED_SOURCE_CLASSES:
        raise ValueError(f"Expected 20 primary-landmark groups, found {len(groups)}")
    return dict(groups)


def split_groups(
    groups: dict[str, list[Path]], seed: int, counts: dict[str, int]
) -> dict[str, list[Path]]:
    expected = sum(counts.values())
    result = {name: [] for name in counts}
    for group, members in sorted(groups.items()):
        if len(members) != expected:
            raise ValueError(f"{group} has {len(members)} images; expected {expected}")
        group_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big"
        )
        shuffled = sorted(members)
        random.Random(group_seed).shuffle(shuffled)
        offset = 0
        for split, count in counts.items():
            result[split].extend(shuffled[offset : offset + count])
            offset += count
    return {split: sorted(paths) for split, paths in result.items()}


def validate_label(path: Path) -> int:
    objects = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != LABEL_FIELD_COUNT:
            raise ValueError(f"{path}:{line_number}: expected 11 fields")
        values = [float(field) for field in fields]
        class_id = int(values[0])
        if values[0] != class_id or not 0 <= class_id < EXPECTED_SOURCE_CLASSES:
            raise ValueError(f"{path}:{line_number}: invalid class")
        if not all(0.0 <= value <= 1.0 for value in values[1:5]) or min(values[3:5]) <= 0:
            raise ValueError(f"{path}:{line_number}: invalid box")
        for start in (5, 8):
            x, y, visibility = values[start : start + 3]
            if visibility not in (0, 1, 2):
                raise ValueError(f"{path}:{line_number}: invalid keypoint visibility")
            if visibility and not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"{path}:{line_number}: invalid keypoint")
        objects += 1
    return objects


def write_yaml(path: Path, dataset_path: Path, names: list[str]) -> None:
    config = {
        "path": str(dataset_path),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "channels": 1,
        "input_semantics": "gray1",
        "kpt_shape": [2, 3],
        "flip_idx": [0, 1],
        "names": {index: name for index, name in enumerate(names)},
    }
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def convert_image(source: Path, target: Path) -> None:
    with Image.open(source) as opened:
        gray = opened.convert("L")
        resized = gray.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.BILINEAR)
        resized.save(target, format="PNG", optimize=False)


def copy_split_metadata(source: Path, output: Path, split_by_id: dict[str, str]) -> None:
    metadata_path = source / "metadata.csv"
    if not metadata_path.is_file():
        return
    with metadata_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(split_by_id):
        raise ValueError("Source metadata count does not match image count")
    for row in rows:
        row["split"] = split_by_id[row["sample_id"]]
    fieldnames = list(rows[0])
    with (output / "metadata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output / "metadata.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    partial = output.with_name(f"{output.name}.partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {partial}")
    counts = {
        "train": args.train_per_primary,
        "val": args.val_per_primary,
        "test": args.test_per_primary,
    }
    if any(count <= 0 for count in counts.values()):
        raise ValueError("All per-primary split counts must be positive")

    names = load_names(source)
    names.append("unrepresented_class_20")
    split = split_groups(grouped_images(source), args.seed, counts)
    split_by_id = {path.stem: name for name, paths in split.items() for path in paths}
    source_labels = source / "labels/test"
    summary: dict[str, dict[str, int]] = {}
    for split_name, paths in split.items():
        image_dir = partial / "images" / split_name
        label_dir = partial / "labels" / split_name
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        objects = 0
        primary_counts: Counter[str] = Counter()
        for index, source_image in enumerate(paths, 1):
            label = source_labels / f"{source_image.stem}.txt"
            objects += validate_label(label)
            convert_image(source_image, image_dir / source_image.name)
            shutil.copy2(label, label_dir / label.name)
            match = IMAGE_PATTERN.match(source_image.stem)
            assert match is not None
            primary_counts[match.group("group")] += 1
            if index % 500 == 0 or index == len(paths):
                print(f"{split_name}: {index}/{len(paths)}", flush=True)
        if set(primary_counts.values()) != {counts[split_name]}:
            raise RuntimeError(f"Unbalanced {split_name} primary counts: {primary_counts}")
        summary[split_name] = {
            "images": len(paths), "labels": len(paths), "objects": objects
        }

    write_yaml(partial / "dataset.yaml", args.dataset_path, names)
    copy_split_metadata(source, partial, split_by_id)
    manifest = {
        "purpose": "production-sized synthetic XYZ/OPK Gray1 pose dataset",
        "source": str(source),
        "pose_source": "synthetic_xyz_opk_no_real_trajectory_or_iers",
        "conversion": "RGB converted to true one-channel BT.601 luminance and bilinear-resized from 1024x1024 to 1280x1280",
        "seed": args.seed,
        "per_primary": counts,
        "class_names": names,
        "unrepresented_class": 20,
        "splits": summary,
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    (partial / "manifest.json").write_bytes(manifest_bytes)
    (partial / "VERIFIED").write_text(
        "prepared_by=prepare_gray1_xyz_opk_dataset.py\n"
        f"manifest_sha256={hashlib.sha256(manifest_bytes).hexdigest()}\n"
        + "".join(f"{name}_images={summary[name]['images']}\n" for name in counts),
        encoding="utf-8",
    )
    partial.rename(output)
    print(f"GRAY1_XYZ_OPK_DATASET_PASS output={output}")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GRAY1_XYZ_OPK_DATASET_FAIL: {exc}", file=sys.stderr)
        raise
