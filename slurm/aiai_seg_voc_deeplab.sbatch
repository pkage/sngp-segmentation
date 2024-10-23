#!/bin/bash
# aiai valluvar setup (segmentation)

#SBATCH --partition=AIAI_GPU
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:A40:1
#SBATCH --output=/home/s1734411/sngp-segmentation/slurm/stdout_pvc_dpl.txt
#SBATCH --error=/home/s1734411/sngp-segmentation/slurm/stderr_pvc_dpl.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=sngp_pvc_dpl
#SBATCH --mail-user=p.kage@ed.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/s1734411/sngp-segmentation/

export PROJ_DIR=/home/s1734411/sngp-segmentation/
export LSCRATCH=/disk/scratch/s1734411/sngp-seg

# set both
export TORCH_HUB=$LSCRATCH
export TORCH_HOME=$LSCRATCH

cd $PROJ_DIR

# load python, ensure the env is installed
export PATH=/home/s1734411/py3.11.4/bin:$PATH
cd $PROJ_DIR

#
mkdir -p $LSCRATCH/datasets


... and the voc
if ! [ -f ./VOCtrainval_11-May-2012.tar ]; then
    echo "voc not found, downloading..."
    curl -L -O -J http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
else
    echo "voc has been downloaded already"
fi

# perf tuning
export OMP_NUM_THREADS=16

# kickoff
echo "beginning train..."
# poetry run python main_ddp.py

cd $PROJ_DIR && poetry run torchrun \
    --nnodes 1 \
    --nproc_per_node 1 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint "localhost:64425" \
    main_ddp.py \
    --batch_size 24 \
    --test_batch_size 24
    --model deeplab
