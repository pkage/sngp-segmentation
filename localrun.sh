#! /bin/sh
# local run setup

export PROJ_DIR=$(dirname $0)
export LSCRATCH=/media/hdd01/scratch/

poetry install
poetry run python main_project.py
