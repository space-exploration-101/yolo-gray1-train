#!/usr/bin/env python3
"""Wait for a validated upstream train/test dataset, then prepare Gray1 data.

The upstream parallel generator publishes ``summary.json`` only after its own
validation has completed.  This program additionally verifies file counts,
image/label pairing, and metadata IDs before it invokes the Gray1 preparer.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


REQUIRED_FILES = ("data.yaml", "metadata.csv", "metadata.json", "summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, default=36_000)
    parser.add_argument("--expected-test", type=int, default=4_000)
    parser.add_argument("--size", type=int, default=1280)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--once", action="store_true", help="Check once and exit if upstream is not ready.")
    return parser.parse_args()


def split_ids(root: Path, kind: str, split: str, suffix: str) -> set[str]:
    directory = root / kind / split
    if not directory.is_dir():
        raise ValueError(f"missing directory: {directory}")
    return {path.stem for path in directory.glob(f"*{suffix}")}


def readiness_error(source: Path, expected: dict[str, int]) -> str | None:
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        return f"waiting for final upstream artifacts: {', '.join(missing)}"

    all_ids: set[str] = set()
    for split, count in expected.items():
        image_ids = split_ids(source, "images", split, ".png")
        label_ids = split_ids(source, "labels", split, ".txt")
        if len(image_ids) != count or len(label_ids) != count:
            return f"{split}: images={len(image_ids)}, labels={len(label_ids)}, expected={count}"
        if image_ids != label_ids:
            return f"{split}: image/label filenames do not match"
        all_ids.update(image_ids)

    with (source / "metadata.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    split_counts = Counter(row.get("split") for row in rows)
    metadata_ids = {row.get("sample_id", "") for row in rows}
    if len(rows) != sum(expected.values()) or dict(split_counts) != expected:
        return f"metadata split counts={dict(split_counts)}, expected={expected}"
    if len(metadata_ids) != len(rows) or metadata_ids != all_ids:
        return "metadata sample IDs do not exactly match image sample IDs"
    return None


def main() -> None:
    args = parse_args()
    if args.expected_train <= 0 or args.expected_test <= 0 or args.size <= 0:
        raise ValueError("expected split counts and size must be positive")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    source = args.source.resolve()
    output = args.output.resolve()
    expected = {"train": args.expected_train, "test": args.expected_test}

    while True:
        error = readiness_error(source, expected)
        if error is None:
            print(f"UPSTREAM_READY source={source}", flush=True)
            command = [
                sys.executable,
                str(Path(__file__).with_name("prepare_gray1_preserve_upstream_splits.py")),
                "--source", str(source),
                "--output", str(output),
                "--dataset-path", str(args.dataset_path),
                "--size", str(args.size),
            ]
            subprocess.run(command, check=True)
            return
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        print(f"UPSTREAM_NOT_READY {stamp}: {error}", flush=True)
        if args.once:
            raise SystemExit(2)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
