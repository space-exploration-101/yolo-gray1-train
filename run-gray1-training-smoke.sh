#!/usr/bin/env bash
set -euo pipefail

TRAIN_IMAGE="ywang/yolo-gray1-train:ultralytics-8.3.98-v1"
GPU_UUID="${1:-}"
TRAIN_DATASET="${2:-}"
TRAIN_RUN_ID="${3:-e2e_smoke_$(date +%Y%m%d_%H%M%S)}"
TRAIN_RUNS_ROOT="${TRAIN_RUNS_ROOT:-${PWD}/runs}"
TRAIN_RUN_ROOT="${TRAIN_RUNS_ROOT%/}/${TRAIN_RUN_ID}"
CONTAINER_NAME="gray1-train-${TRAIN_RUN_ID}"

if [[ -z "$GPU_UUID" || -z "$TRAIN_DATASET" ]]; then
  echo "Usage: $0 <gpu-uuid> <prepared-dataset-path> [run-id]" >&2
  echo "Optional environment: TRAIN_RUNS_ROOT=<absolute-writable-runs-directory>" >&2
  exit 2
fi

if [[ "$TRAIN_RUNS_ROOT" != /* ]]; then
  echo "ERROR: TRAIN_RUNS_ROOT must be an absolute path: $TRAIN_RUNS_ROOT" >&2
  exit 2
fi

if [[ "$GPU_UUID" != GPU-* ]]; then
  echo "ERROR: expected a full GPU UUID beginning with GPU-" >&2
  exit 2
fi

if [[ ! -f "$TRAIN_DATASET/VERIFIED" ]]; then
  echo "ERROR: dataset is not verified: $TRAIN_DATASET" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_DATASET/dataset.yaml" ]]; then
  echo "ERROR: dataset.yaml is missing: $TRAIN_DATASET" >&2
  exit 1
fi

if [[ -e "$TRAIN_RUN_ROOT" ]]; then
  echo "ERROR: refusing to reuse existing run directory: $TRAIN_RUN_ROOT" >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "ERROR: container name already exists: $CONTAINER_NAME" >&2
  exit 1
fi

docker image inspect "$TRAIN_IMAGE" >/dev/null
nvidia-smi --id="$GPU_UUID" \
  --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu \
  --format=csv
nvidia-smi dmon -i "$GPU_UUID" -s pucm -c 5

if nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader | grep -Fq "$GPU_UUID"; then
  echo "ERROR: the selected GPU has an active compute process; training was not started" >&2
  nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv
  exit 1
fi

read -r GPU_UTIL GPU_FREE < <(
  nvidia-smi --id="$GPU_UUID" \
    --query-gpu=utilization.gpu,memory.free \
    --format=csv,noheader,nounits |
    awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1, $2}'
)

if (( GPU_UTIL > 20 || GPU_FREE < 32768 )); then
  echo "ERROR: GPU gate failed: utilization=${GPU_UTIL}%, free=${GPU_FREE} MiB" >&2
  exit 1
fi

mkdir -p "$TRAIN_RUN_ROOT/experiments"
cp "$TRAIN_DATASET/dataset.yaml" "$TRAIN_RUN_ROOT/dataset.container.yaml"
sed -i 's|^path:.*|path: /dataset|' "$TRAIN_RUN_ROOT/dataset.container.yaml"

cleanup_container() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup_container EXIT INT TERM

echo "TRAIN_IMAGE=$TRAIN_IMAGE"
echo "TRAIN_DATASET=$TRAIN_DATASET"
echo "TRAIN_RUN_ID=$TRAIN_RUN_ID"
echo "TRAIN_RUNS_ROOT=$TRAIN_RUNS_ROOT"
echo "TRAIN_RUN_ROOT=$TRAIN_RUN_ROOT"
echo "GPU_UUID=$GPU_UUID"

set +e
timeout --signal=TERM --kill-after=30s 55m \
docker run --rm \
  --name "$CONTAINER_NAME" \
  --stop-timeout 30 \
  --gpus "device=$GPU_UUID" \
  --user "$(id -u):$(id -g)" \
  --network none \
  --read-only \
  --shm-size 64g \
  --tmpfs /tmp:rw,nosuid,size=2g \
  -e HOME=/tmp \
  -e YOLO_CONFIG_DIR=/tmp/ultralytics \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  -e PYTHONPATH=/workspace/src/Yolo_fine/ultralytics \
  -v "$TRAIN_DATASET:/dataset:ro" \
  -v "$TRAIN_RUN_ROOT/dataset.container.yaml:/config/dataset.yaml:ro" \
  -v "$TRAIN_RUN_ROOT/experiments:/experiments:rw" \
  --entrypoint python \
  "$TRAIN_IMAGE" \
  /workspace/src/Yolo_fine/train_yolov8x_p6_pose_gray1.py \
  --data /config/dataset.yaml \
  --epochs 1 \
  --imgsz 1280 \
  --batch 4 \
  --device 0 \
  --workers 4 \
  --project /experiments \
  --name "$TRAIN_RUN_ID" \
  --save-period 1
TRAIN_STATUS=$?
set -e

if (( TRAIN_STATUS != 0 )); then
  echo "ERROR: training exited with status $TRAIN_STATUS" >&2
  echo "Partial outputs, if any: $TRAIN_RUN_ROOT" >&2
  exit "$TRAIN_STATUS"
fi

MODEL="$TRAIN_RUN_ROOT/experiments/$TRAIN_RUN_ID/weights/best.pt"
if [[ ! -s "$MODEL" ]]; then
  echo "ERROR: training exited successfully but best.pt is missing: $MODEL" >&2
  exit 1
fi

sha256sum "$MODEL"
echo "TRAINING_SMOKE_PASS model=$MODEL"
