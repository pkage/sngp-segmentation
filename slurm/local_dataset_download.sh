#! /bin/bash

export WANDB_PROJECT=sngp-seg
export LSCRATCH=/Volumes/Dock/phd/sngp-segmentation/lscratch/
export WANDB_API_KEY=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
export WANDB_ENTITY=pkage

mkdir -p "$LSCRATCH"

bash dataset_download.sh
