#!/bin/bash
# aiai valluvar setup (segmentation)

#SBATCH --partition=AIAI_GPU
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:A40:4
#SBATCH --output=/home/s1734411/sngp-segmentation/stdout.txt
#SBATCH --error=/home/s1734411/sngp-segmentation/stderr.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=sngp_seg
#SBATCH --mail-user=p.kage@ed.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/s1734411/sngp-segmentation/

export PROJ_DIR=/home/s1734411/sngp-segmentation/
export LSCRATCH=/disk/scratch/s1734411

cd $PROJ_DIR

# load python, ensure the env is installed
export PATH=/home/s1734411/py3.11.4/bin:$PATH
cd $PROJ_DIR

# ... and the voc
if ! [ -f ./VOCtrainval_11-May-2012.tar ]; then
    echo "voc not found, downloading..."
    curl -L -O -J http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
else
    echo "voc has been downloaded already"
fi

# kickoff
echo "beginning train..."
poetry run python main_ddp.py

# srun torchrun \
#     --nnodes $SLURM_JOB_NUM_NODES \
#     --nproc_per_node 2 \
#     --rdzv_id $RANDOM \
#     --rdzv_backend c10d \
#     --rdzv_endpoint "$head_node_ip:64425" \
#     main_ddp.py
