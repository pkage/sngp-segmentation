#! /bin/sh
# local run setup

export PROJ_DIR=$(dirname $0)
export LSCRATCH=/media/hdd01/scratch/

poetry install

# check that we've got our vit model checkpoint....
if ! [ -f ./IN1K-vit.h.14-300e.pth.tar ]; then
    echo "vit not found, downloading..."
    curl -L -O -J https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar
else
    echo "vit has been downloaded already"
fi

# ... and the voc ...
if ! [ -f ./VOCtrainval_11-May-2012.tar ]; then
    echo "voc not found, downloading..."
    curl -L -O -J http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
else
    echo "voc has been downloaded already"
fi

# ... and the sngp-vit checkpoint
if ! [ -f ./ijepa_sngp_epoch190.pth ]; then
    echo "sngp-vit checkpoint not found, downloading..."
    curl -L -O -J http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
else
    echo "sngp-vit checkpoint has been downloaded already"
fi

poetry run python main_project.py
