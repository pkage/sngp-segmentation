export WANDB_ENTITY=ai2es
export WANDB_PROJECT=unc-pl-seg
export LSCRATCH=./lscratch
export WANDB_API_KEY=6ac799cb76304b17ce74f5161bc27f7a80b6ecee

torchrun --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:0 \
        --standalone \
        --nnodes=1 \
        --nproc-per-node=1 \
        main_ddp.py \
        --batch_size 4 \
        --test_batch_size 4 \
        --epochs 32 \
        --model deep_ensemble \
        --ul_fraction 0.4 \
        --strategy self \
        --accumulate 4 \