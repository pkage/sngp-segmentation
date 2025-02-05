export DOWNLOAD_FUNCTION_URL=https://ledoejtumvyg2uiumzgibwc2ze0zzjgx.lambda-url.us-east-2.on.aws

mkdir -p $LSCRATCH/datasets
mkdir -p $LSCRATCH/datasets/coco

echo "downloading cityscapes"

if ! [ -f $LSCRATCH/datasets/gtFine_trainvaltest.zip ]; then
    echo "gtFine_trainvaltest.zip not found, downloading..."
    curl -L $DOWNLOAD_FUNCTION_URL/gtFine_trainvaltest.zip -o $LSCRATCH/datasets/gtFine_trainvaltest.zip
else
    echo "gtFine_trainvaltest.zip has already been downloaded."
fi

if ! [ -f $LSCRATCH/datasets/leftImg8bit_trainvaltest.zip ]; then
    echo "leftImg8bit_trainvaltest.zip not found, downloading..."
    curl -L $DOWNLOAD_FUNCTION_URL/leftImg8bit_trainvaltest.zip -o $LSCRATCH/datasets/leftImg8bit_trainvaltest.zip
else
    echo "leftImg8bit_trainvaltest.zip has already been downloaded."
fi

if ! [ -d $LSCRATCH/datasets/leftImg8bit ]; then
    echo "leftImg8bit_trainvaltest.zip has not been unzipped, expanding now..."
    cd $LSCRATCH/datasets/ && unzip leftImg8bit_trainvaltest.zip
else
    echo "leftImg8bit_trainvaltest.zip has already been unzipped."
fi
if ! [ -d $LSCRATCH/datasets/gtFine ]; then
    echo "gtFine_trainvaltest.zip has not been unzipped, expanding now..."
    cd $LSCRATCH/datasets/ && unzip gtFine_trainvaltest.zip
else
    echo "gtFine_trainvaltest.zip has already been unzipped."
fi


echo "downloading coco"
if ! [ -f $LSCRATCH/datasets/coco/train2014.zip ]; then
    echo "COCO train2014.zip not found, downloading..."
    curl -LJ http://images.cocodataset.org/zips/train2014.zip -o $LSCRATCH/datasets/coco/train2014.zip
    curl -L $DOWNLOAD_FUNCTION_URL/coco_train2014.zip -o $LSCRATCH/datasets/coco/train2014.zip
else
    echo "COCO train2014.zip has already been downloaded."
fi
if ! [ -d $LSCRATCH/datasets/coco/train2014 ]; then
    echo "COCO train2014.zip has not been unzipped, expanding now..."
    cd $LSCRATCH/datasets/coco && unzip train2014.zip
else
    echo "train2014.zip has already been unzipped."
fi



# ... and the voc
# if ! [ -f ./VOCtrainval_11-May-2012.tar ]; then
#     echo "voc not found, downloading..."
#     curl -L -O -J http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# else
#     echo "voc has been downloaded already"
# fi
