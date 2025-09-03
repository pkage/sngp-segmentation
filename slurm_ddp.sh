#!/bin/bash
#SBATCH --partition=ai2es_h100
#SBATCH --exclude=c856
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Thread count:
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
# memory in MB
#SBATCH --mem=64G
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/jroth/sngp_seg/slurm/sngp_out_%a.txt
#SBATCH --error=/ourdisk/hpc/ai2es/jroth/sngp_seg/slurm/sngp_err_%a.txt
#SBATCH --time=48:00:00
#SBATCH --job-name=sngp
#SBATCH --mail-user=jay.c.rothenberger@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/ourdisk/hpc/ai2es/jroth/sngp_seg
#SBATCH --array=[0-0]
#################################################

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b")

echo Node IP: $head_node_ip


. /home/fagg/tf_setup.sh
conda activate /home/jroth/.conda/envs/mct

export WANDB_ENTITY=ai2es
export WANDB_PROJECT=unc-pl-seg
export WANDB_API_KEY=<your-key>

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint "$head_node_ip:64425" \
 main_ddp.py --batch_size 16 --test_batch_size 8 --epochs 25 --model deeplab --ul_fraction 0.0 --strategy baseline --accumulate 1
