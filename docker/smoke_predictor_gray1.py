"""CPU smoke: Gray1 predictor warmup and single-channel preprocess must succeed."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import cv2

from ultralytics import YOLO
from ultralytics.engine.predictor import BasePredictor
from ultralytics.nn.autobackend import AutoBackend


def first_conv_in_channels(model: YOLO) -> int:
    for module in model.model.modules():
        if hasattr(module, "in_channels") and hasattr(module, "weight"):
            return int(module.in_channels)
    raise RuntimeError("Unable to locate the first convolution")


def main() -> None:
    weights = Path(sys.argv[1])
    image_path = Path(sys.argv[2])

    warmup_src = inspect.getsource(BasePredictor.stream_inference)
    if "model_yaml.get(\"ch\", 3)" not in warmup_src and "model_yaml.get('ch', 3)" not in warmup_src:
        raise SystemExit("FAIL: predictor warmup still does not read model yaml channels")
    if ", 3, *self.imgsz)" in warmup_src:
        raise SystemExit("FAIL: predictor warmup still hard-codes 3 channels")

    preprocess_src = inspect.getsource(BasePredictor.preprocess)
    if "im[..., ::-1].transpose((0, 3, 1, 2))" in preprocess_src:
        raise SystemExit("FAIL: predictor preprocess still always reverses channels")
    if "im.shape[-1] == 3" not in preprocess_src:
        raise SystemExit("FAIL: predictor preprocess does not gate BGR reversal on 3-channel inputs")

    pre_transform_src = inspect.getsource(BasePredictor.pre_transform)
    if "output[..., None]" not in pre_transform_src:
        raise SystemExit("FAIL: predictor pre_transform does not restore squeezed gray1 channel")

    recorded_warmup: list[tuple[int, ...]] = []
    recorded_pre: list[tuple[int, ...]] = []
    original_warmup = AutoBackend.warmup
    original_preprocess = BasePredictor.preprocess

    def wrapped_warmup(self, imgsz=(1, 3, 640, 640)):  # type: ignore[no-untyped-def]
        recorded_warmup.append(tuple(imgsz) if isinstance(imgsz, (tuple, list)) else (imgsz,))
        print(f"WARMUP_IMGSZ={imgsz}", flush=True)
        return original_warmup(self, imgsz)

    def wrapped_preprocess(self, im):  # type: ignore[no-untyped-def]
        out = original_preprocess(self, im)
        recorded_pre.append(tuple(int(v) for v in out.shape))
        print(f"PREPROCESS_SHAPE={tuple(int(v) for v in out.shape)}", flush=True)
        return out

    AutoBackend.warmup = wrapped_warmup
    BasePredictor.preprocess = wrapped_preprocess

    model = YOLO(str(weights))
    in_ch = first_conv_in_channels(model)
    print(f"FIRST_CONV_IN_CHANNELS={in_ch}", flush=True)
    if in_ch != 1:
        raise SystemExit(f"FAIL: expected 1-channel first conv, got {in_ch}")

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"FAIL: cannot read {image_path}")
    if gray.shape != (1280, 1280):
        raise SystemExit(f"FAIL: expected 1280x1280, got {gray.shape}")

    results = model.predict(
        source=[gray[..., None]],
        imgsz=1280,
        device="cpu",
        verbose=False,
        save=False,
        save_txt=False,
    )
    if not results:
        raise SystemExit("FAIL: predictor returned no results")
    if not recorded_warmup:
        raise SystemExit("FAIL: warmup was not called")
    warmup_shape = recorded_warmup[-1]
    print(f"WARMUP_SHAPE={warmup_shape}", flush=True)
    if warmup_shape != (1, 1, 1280, 1280):
        raise SystemExit(f"FAIL: warmup shape {warmup_shape} != (1, 1, 1280, 1280)")
    if not recorded_pre:
        raise SystemExit("FAIL: preprocess was not called")
    pre_shape = recorded_pre[-1]
    if len(pre_shape) != 4 or pre_shape[1] != 1:
        raise SystemExit(f"FAIL: preprocess tensor is not 1-channel: {pre_shape}")
    if results[0].orig_shape != (1280, 1280):
        raise SystemExit(f"FAIL: orig_shape {results[0].orig_shape} != (1280, 1280)")
    print(f"RESULT_ORIG_SHAPE={results[0].orig_shape}", flush=True)
    print(f"RESULT_PRED_COUNT={0 if results[0].boxes is None else len(results[0].boxes)}", flush=True)
    print("SMOKE_GRAY1_PREDICT_PASS", flush=True)


if __name__ == "__main__":
    main()
