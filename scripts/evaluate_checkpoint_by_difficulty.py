#!/usr/bin/env python3
"""Evaluate one Gray1 pose checkpoint on four held-out difficulty subsets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = REPO_ROOT / "src" / "Yolo_fine"
LOCAL_ULTRALYTICS_ROOT = YOLO_ROOT / "ultralytics"
sys.path.insert(0, str(LOCAL_ULTRALYTICS_ROOT))
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(LOCAL_ULTRALYTICS_ROOT), os.environ.get("PYTHONPATH", "")]
).rstrip(os.pathsep)

from ultralytics import YOLO  # noqa: E402


DIFFICULTIES = ("easy", "medium", "hard", "extreme")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def scalar(value: object) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def write_report(output: Path, checkpoint: Path, counts: dict[str, int], rows: list[dict[str, object]]) -> None:
    json_path = output / "difficulty_metrics.json"
    json_path.write_text(
        json.dumps({"checkpoint": str(checkpoint), "counts": counts, "results": rows}, indent=2),
        encoding="utf-8",
    )
    keys = [
        "difficulty",
        "images",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/mAP50(P)",
        "metrics/mAP50-95(P)",
        "metrics/kpt_l1_mean_px",
        "metrics/kpt_l1_median_px",
        "metrics/kpt_l1_p95_px",
        "metrics/kpt_pixel_coverage",
    ]
    with (output / "difficulty_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in rows)

    lines = [
        "# Epoch 120 evaluation by difficulty",
        "",
        f"Checkpoint: `{checkpoint}`",
        "",
        "| Difficulty | Images | Box mAP50 | Box mAP50-95 | Pose mAP50 | Pose mAP50-95 | Mean L1 px | Median L1 px | P95 L1 px | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {difficulty} | {images} | {box50:.4f} | {box5095:.4f} | {pose50:.4f} | "
            "{pose5095:.4f} | {mean:.3f} | {median:.3f} | {p95:.3f} | {coverage:.4f} |".format(
                difficulty=row["difficulty"],
                images=row["images"],
                box50=row["metrics/mAP50(B)"],
                box5095=row["metrics/mAP50-95(B)"],
                pose50=row["metrics/mAP50(P)"],
                pose5095=row["metrics/mAP50-95(P)"],
                mean=row["metrics/kpt_l1_mean_px"],
                median=row["metrics/kpt_l1_median_px"],
                p95=row["metrics/kpt_l1_p95_px"],
                coverage=row["metrics/kpt_pixel_coverage"],
            )
        )
    (output / "difficulty_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    metadata = dataset / "metadata.csv"
    data_yaml = dataset / "dataset.yaml"
    if not args.weights.is_file() or not metadata.is_file() or not data_yaml.is_file():
        raise FileNotFoundError("weights, metadata.csv, and dataset.yaml must exist")
    args.output.mkdir(parents=True, exist_ok=True)

    with metadata.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    ids = {
        difficulty: [
            row["sample_id"]
            for row in records
            if row.get("split") == "test" and row.get("difficulty") == difficulty
        ]
        for difficulty in DIFFICULTIES
    }
    expected = {"easy": 1600, "medium": 800, "hard": 800, "extreme": 800}
    counts = {difficulty: len(sample_ids) for difficulty, sample_ids in ids.items()}
    if counts != expected:
        raise ValueError(f"Unexpected test difficulty counts: {counts}, expected {expected}")

    base_config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    model = YOLO(str(args.weights.resolve()))
    rows: list[dict[str, object]] = []
    for difficulty in DIFFICULTIES:
        image_paths = [dataset / "images" / "test" / f"{sample_id}.png" for sample_id in ids[difficulty]]
        label_paths = [dataset / "labels" / "test" / f"{sample_id}.txt" for sample_id in ids[difficulty]]
        missing = [str(path) for path in image_paths + label_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {difficulty} files, e.g. {missing[:3]}")

        image_list = args.output / f"test_{difficulty}.txt"
        image_list.write_text("".join(f"{path}\n" for path in image_paths), encoding="utf-8")
        subset_config = dict(base_config)
        subset_config["path"] = str(dataset)
        subset_config["val"] = str(image_list.resolve())
        subset_config["test"] = str(image_list.resolve())
        subset_yaml = args.output / f"dataset_{difficulty}.yaml"
        subset_yaml.write_text(yaml.safe_dump(subset_config, sort_keys=False), encoding="utf-8")

        metrics = model.val(
            data=str(subset_yaml),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            plots=False,
            project=str(args.output),
            name=f"val_{difficulty}",
            exist_ok=True,
        )
        row: dict[str, object] = {"difficulty": difficulty, "images": counts[difficulty]}
        row.update({key: scalar(value) for key, value in metrics.results_dict.items()})
        rows.append(row)
        write_report(args.output, args.weights.resolve(), counts, rows)
        print(
            f"DIFFICULTY_RESULT difficulty={difficulty} images={counts[difficulty]} "
            f"box_mAP50_95={row['metrics/mAP50-95(B)']:.6f} "
            f"pose_mAP50_95={row['metrics/mAP50-95(P)']:.6f} "
            f"mean_l1_px={row['metrics/kpt_l1_mean_px']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
