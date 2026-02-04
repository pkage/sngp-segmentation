#!/usr/bin/env bash
set -euo pipefail

export WANDB_PROJECT=sngp-seg
export LSCRATCH=/media/hdd02/phd/sngp-segmentation/lscratch
# export LSCRATCH=/Volumes/Dock/phd/sngp-segmentation/lscratch/
export WANDB_API_KEY=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
export WANDB_ENTITY=pkage

mkdir -p "$LSCRATCH"

DEVICE_TYPE=${SNGP_DEVICE_TYPE:-cuda}
BACKEND=${TORCH_DISTRIBUTED_BACKEND:-nccl}
NPROC_PER_NODE=${SNGP_NPROC_PER_NODE:-1}

PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mps)
      DEVICE_TYPE=mps
      BACKEND=gloo
      NPROC_PER_NODE=1
      export PYTORCH_ENABLE_MPS_FALLBACK=1
      shift
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done

export SNGP_DEVICE_TYPE="$DEVICE_TYPE"
export TORCH_DISTRIBUTED_BACKEND="$BACKEND"

TORCHRUN_ARGS=(
  "--nproc_per_node=${NPROC_PER_NODE}"
  "--nnodes=1"
  "--node_rank=0"
  "--master_addr=localhost"
  "--master_port=12355"
)

echo "Launching torchrun with backend=${BACKEND}, device=${DEVICE_TYPE}, nproc_per_node=${NPROC_PER_NODE}"
echo "Pass --no-dahps along with training hyperparameters to disable DAHPS."

uv run torchrun "${TORCHRUN_ARGS[@]}" main_ddp.py "${PASSTHROUGH[@]}" --scratch_path $LSCRATCH --voc_path $LSCRATCH/datasets/voc --coco_path $LSCRATCH/datasets/coco --cityscapes_path $LSCRATCH/datasets --checkpoint_path $LSCRATCH/checkpoints

