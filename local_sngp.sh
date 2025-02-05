export WANDB_PROJECT=ai2es
export LSCRATCH=./lscratch
export WANDB_API_KEY=6ac799cb76304b17ce74f5161bc27f7a80b6ecee

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --standalone --nnodes=1 --nproc-per-node=1 main_ddp.py --batch_size 16 --test_batch_size 8 --epochs 64 --model sngp --ul_fraction 0.5 --strategy self --accumulate 1