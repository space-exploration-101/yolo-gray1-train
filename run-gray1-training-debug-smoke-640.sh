#!/usr/bin/env bash
# RTX 5090 functional smoke test using the standard fixed 640x640 recipe.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
TRAIN_IMAGE="${TRAIN_IMAGE:-yolo-gray1:1280-train-predict-20261016}"
GPU_UUID="${1:-}"
TRAIN_DATASET="${2:-}"
TRAIN_RUN_ID="${3:-debug640_smoke_$(date +%Y%m%d_%H%M%S)}"
TRAIN_RUNS_ROOT="${TRAIN_RUNS_ROOT:-${PWD}/runs}"
TRAIN_RUN_ROOT="${TRAIN_RUNS_ROOT%/}/${TRAIN_RUN_ID}"
CONTAINER_NAME="gray1-debug640-${TRAIN_RUN_ID}"
MIN_FREE_MIB="${MIN_FREE_MIB:-24000}"
DEBUG_BATCH="${DEBUG_BATCH:-2}"
DEBUG_EPOCHS="${DEBUG_EPOCHS:-1}"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-55m}"
BOX_GAIN="${BOX_GAIN:-7.5}"
POSE_GAIN="${POSE_GAIN:-12.0}"
KOBJ_GAIN="${KOBJ_GAIN:-1.0}"

if [[ -z "$GPU_UUID" || -z "$TRAIN_DATASET" ]]; then
  echo "Usage: $0 <gpu-uuid> <prepared-dataset-path> [run-id]" >&2
  exit 2
fi
if [[ "$TRAIN_RUNS_ROOT" != /* || "$GPU_UUID" != GPU-* ]]; then
  echo "ERROR: TRAIN_RUNS_ROOT must be absolute and GPU_UUID must begin with GPU-" >&2
  exit 2
fi
if [[ ! "$DEBUG_BATCH" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: DEBUG_BATCH must be a positive integer" >&2
  exit 2
fi
if [[ ! "$DEBUG_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: DEBUG_EPOCHS must be a positive integer" >&2
  exit 2
fi
for gain_name in BOX_GAIN POSE_GAIN KOBJ_GAIN; do
  gain_value="${!gain_name}"
  if [[ ! "$gain_value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || [[ "$gain_value" == "0" ]] || [[ "$gain_value" == "0.0" ]]; then
    echo "ERROR: ${gain_name} must be a positive number" >&2
    exit 2
  fi
done
if [[ ! -f "$TRAIN_DATASET/VERIFIED" || ! -f "$TRAIN_DATASET/dataset.yaml" ]]; then
  echo "ERROR: prepared dataset must contain VERIFIED and dataset.yaml" >&2
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

docker image inspect "$TRAIN_IMAGE" --format 'TRAIN_IMAGE_ID={{.Id}}'
nvidia-smi --id="$GPU_UUID" \
  --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu \
  --format=csv
nvidia-smi dmon -i "$GPU_UUID" -s pucm -c 5
if nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader | grep -Fq "$GPU_UUID"; then
  echo "ERROR: selected GPU has an active compute process" >&2
  exit 1
fi
read -r GPU_UTIL GPU_FREE < <(
  nvidia-smi --id="$GPU_UUID" --query-gpu=utilization.gpu,memory.free \
    --format=csv,noheader,nounits | awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1, $2}'
)
if (( GPU_UTIL > 20 || GPU_FREE < MIN_FREE_MIB )); then
  echo "ERROR: debug GPU gate failed: utilization=${GPU_UTIL}%, free=${GPU_FREE} MiB, required=${MIN_FREE_MIB} MiB" >&2
  exit 1
fi

mkdir -p "$TRAIN_RUN_ROOT/experiments"
cp "$TRAIN_DATASET/dataset.yaml" "$TRAIN_RUN_ROOT/dataset.container.yaml"
sed -i 's|^path:.*|path: /dataset|' "$TRAIN_RUN_ROOT/dataset.container.yaml"
cleanup_container() { docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup_container EXIT INT TERM

echo "DEBUG_TRAINING=imgsz=640,batch=${DEBUG_BATCH},epochs=${DEBUG_EPOCHS},workers=2,min_free_mib=${MIN_FREE_MIB},timeout=${TRAIN_TIMEOUT},box=${BOX_GAIN},pose=${POSE_GAIN},kobj=${KOBJ_GAIN}"
set +e
timeout --signal=TERM --kill-after=30s "$TRAIN_TIMEOUT" \
docker run --rm \
  --name "$CONTAINER_NAME" \
  --stop-timeout 30 \
  --gpus "device=$GPU_UUID" \
  --user "$(id -u):$(id -g)" \
  --network none \
  --read-only \
  --shm-size 16g \
  --tmpfs /tmp:rw,nosuid,size=2g \
  -e HOME=/tmp \
  -e YOLO_CONFIG_DIR=/tmp/ultralytics \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  -e PYTHONPATH=/workspace/src/Yolo_fine/ultralytics \
  -v "$REPO_ROOT/src/Yolo_fine:/workspace/src/Yolo_fine:ro" \
  -v "$TRAIN_DATASET:/dataset:ro" \
  -v "$TRAIN_RUN_ROOT/dataset.container.yaml:/config/dataset.yaml:ro" \
  -v "$TRAIN_RUN_ROOT/experiments:/experiments:rw" \
  --entrypoint python \
  "$TRAIN_IMAGE" \
  /workspace/src/Yolo_fine/train_yolov8x_p6_pose_gray1.py \
  --data /config/dataset.yaml \
  --epochs "$DEBUG_EPOCHS" \
  --imgsz 640 \
  --batch "$DEBUG_BATCH" \
  --box-gain "$BOX_GAIN" \
  --pose-gain "$POSE_GAIN" \
  --kobj-gain "$KOBJ_GAIN" \
  --mosaic 1.0 \
  --mixup 0.1 \
  --copy-paste 0.1 \
  --flipud 0.5 \
  --fliplr 0.5 \
  --degrees 10 \
  --translate 0.1 \
  --scale 0.5 \
  --shear 2 \
  --perspective 0.0005 \
  --erasing 0.2 \
  --no-multi-scale \
  --device 0 \
  --workers 2 \
  --project /experiments \
  --name "$TRAIN_RUN_ID" \
  --save-period 1
TRAIN_STATUS=$?
set -e
if (( TRAIN_STATUS != 0 )); then
  echo "ERROR: debug training exited with status $TRAIN_STATUS; outputs retained at $TRAIN_RUN_ROOT" >&2
  exit "$TRAIN_STATUS"
fi

MODEL="$TRAIN_RUN_ROOT/experiments/$TRAIN_RUN_ID/weights/best.pt"
if [[ ! -s "$MODEL" ]]; then
  echo "ERROR: training exited successfully but best.pt is missing: $MODEL" >&2
  exit 1
fi
sha256sum "$MODEL"
echo "DEBUG_TRAINING_SMOKE_PASS model=$MODEL"
