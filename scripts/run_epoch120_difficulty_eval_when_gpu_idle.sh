#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/ai-i-lizhaojin/Code/yolo-gray1-train
RUN_ROOT="$REPO_ROOT/runs/earth_gray1_40000_640_b24_e150_pose30_curriculum_1to10_11to20_21to50"
DATASET_ROOT="$REPO_ROOT/data/earth_gray1_00to19_xyz_opk_40000_view_difficulty"
WEIGHTS="$RUN_ROOT/weights/epoch120.pt"
OUTPUT="$RUN_ROOT/eval_epoch120_by_difficulty"
IMAGE=yolo-gray1:1280-train-predict-20261016

while true; do
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF {found=1} END {exit !found}'; then
    echo "WAITING_FOR_GPU $(date --iso-8601=seconds)"
    sleep 60
    continue
  fi
  sleep 15
  if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF {found=1} END {exit !found}'; then
    break
  fi
done

echo "START_DIFFICULTY_EVAL $(date --iso-8601=seconds)"
docker run --rm \
  --name gray1-epoch120-difficulty-eval \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  --network none \
  --read-only \
  --shm-size 64g \
  --tmpfs /tmp:rw,nosuid,size=8g \
  -e YOLO_CONFIG_DIR=/tmp/ultralytics \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  -e PYTHONPATH=/workspace/src/Yolo_fine/ultralytics \
  -v "$REPO_ROOT/src/Yolo_fine:/workspace/src/Yolo_fine:ro" \
  -v "$REPO_ROOT/scripts:/workspace/scripts:ro" \
  -v "$DATASET_ROOT:/dataset:ro" \
  -v "$RUN_ROOT:/run:rw" \
  --entrypoint python \
  "$IMAGE" \
  /workspace/scripts/evaluate_checkpoint_by_difficulty.py \
  --weights /run/weights/epoch120.pt \
  --dataset /dataset \
  --output /run/eval_epoch120_by_difficulty \
  --imgsz 640 \
  --batch 24 \
  --device 0 \
  --workers 4
echo "DIFFICULTY_EVAL_COMPLETE $(date --iso-8601=seconds) output=$OUTPUT"
