# syntax=docker/dockerfile:1

FROM ubuntu:22.04 AS env-unpack

COPY docker/yolo-runtime.tar.gz /tmp/yolo-runtime.tar.gz
RUN mkdir -p /opt/runtime \
    && tar -xzf /tmp/yolo-runtime.tar.gz -C /opt/runtime \
    && rm -f /tmp/yolo-runtime.tar.gz

FROM ubuntu:22.04

COPY --from=env-unpack /opt/runtime /opt/ywang/yolo-runtime
COPY src/Yolo_fine /workspace/src/Yolo_fine

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    CONDA_PREFIX=/opt/ywang/yolo-runtime \
    PATH=/opt/ywang/yolo-runtime/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/workspace/src/Yolo_fine/ultralytics \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_INPUT_CHANNELS=1 \
    YOLO_INPUT_SEMANTICS=gray1 \
    YOLO_IMGSZ=1280

WORKDIR /workspace/src/Yolo_fine

RUN python -c "import sys, torch, torchvision, cv2, numpy, albumentations, PIL, yaml, tqdm; assert sys.version.split()[0] == '3.12.12'; assert torch.__version__ == '2.8.0+cu128'; assert torchvision.__version__ == '0.23.0+cu128'; assert cv2.__version__ == '4.11.0'; assert numpy.__version__ == '1.26.4'; assert albumentations.__version__ == '2.0.8'; assert PIL.__version__ == '12.2.0'; assert yaml.__version__ == '6.0.3'; assert tqdm.__version__ == '4.67.3'"

CMD ["/bin/bash"]
