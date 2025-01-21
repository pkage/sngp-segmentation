export DOWNLOAD_FUNCTION_URL=https://ledoejtumvyg2uiumzgibwc2ze0zzjgx.lambda-url.us-east-2.on.aws
export DOWNLOAD_FUNCTION_URL=https://ledoejtumvyg2uiumzgibwc2ze0zzjgx.lambda-url.us-east-2.on.aws
mkdir -p $LSCRATCH/datasets

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


# ... and the voc
# if ! [ -f ./VOCtrainval_11-May-2012.tar ]; then
#     echo "voc not found, downloading..."
#     curl -L -O -J http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# else
#     echo "voc has been downloaded already"
# fi
