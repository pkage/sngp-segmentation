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
poetry install

# kickoff
echo "beginning projection..."
poetry run python main_seg.py

