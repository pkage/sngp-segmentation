#!/bin/bash
# Jay Rothenberger
#
#
#SBATCH --partition=disc
#SBATCH --cpus-per-task=64
# The number of cores you get
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/jroth/sngp_seg/out.txt
#SBATCH --error=/ourdisk/hpc/ai2es/jroth/sngp_seg/err.txt
#SBATCH --time=09:30:00
#SBATCH --job-name=sngp_seg
#SBATCH --mail-user=jay.c.rothenberger@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/ourdisk/hpc/ai2es/jroth/sngp_seg/
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b")

echo Node IP: $head_node_ip

. /home/fagg/tf_setup.sh
conda activate /home/jroth/.conda/envs/torch

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint "$head_node_ip:64425" \
main_ddp.py