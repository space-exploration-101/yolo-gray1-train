#!/usr/bin/env python3
"""Render Etosha difficulty samples with their YOLO ground-truth annotations."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "data" / "earth_gray1_00to19_xyz_opk_40000_view_difficulty"
OUTPUT = Path(__file__).resolve().parent / "images"
DIFFICULTIES = ("easy", "medium", "hard", "extreme")
DISPLAY_SIZE = 640
ANNOTATION_COLOR = "#00A8FF"


def load_annotation(label_path: Path) -> tuple[float, float, float, float, list[tuple[float, float]]]:
    fields = label_path.read_text(encoding="utf-8").splitlines()[0].split()
    values = [float(value) for value in fields[1:]]
    x_center, y_center, width, height = values[:4]
    keypoints = []
    for index in range(4, len(values), 3):
        x, y, visibility = values[index : index + 3]
        if visibility > 0:
            keypoints.append((x, y))
    return x_center, y_center, width, height, keypoints


def render(difficulty: str) -> None:
    sample = f"landmark_00_etosha_{difficulty}_test_0001"
    image_path = DATASET / "images" / "test" / f"{sample}.png"
    label_path = DATASET / "labels" / "test" / f"{sample}.txt"
    image = Image.open(image_path).convert("RGB")
    x_center, y_center, width, height, keypoints = load_annotation(label_path)
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    left = (x_center - width / 2) * image_width
    top = (y_center - height / 2) * image_height
    right = (x_center + width / 2) * image_width
    bottom = (y_center + height / 2) * image_height
    draw.rectangle((left, top, right, bottom), outline=ANNOTATION_COLOR, width=6)
    for x, y in keypoints:
        px, py = x * image_width, y * image_height
        radius = 10
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=ANNOTATION_COLOR, outline="white", width=2)
    font = ImageFont.load_default()
    title = f"Etosha | {difficulty} | bright blue: GT box and visible GT keypoint"
    draw.rectangle((0, 0, image_width, 36), fill="black")
    draw.text((10, 10), title, fill="white", font=font)
    image.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS).save(OUTPUT / f"etosha_{difficulty}_gt.png")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for difficulty in DIFFICULTIES:
        render(difficulty)


if __name__ == "__main__":
    main()
