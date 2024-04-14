##
# project voc samples into ijepa space, and save them back out
# this way, we don't have to produce all the representations on-the-fly,
# and save our VRAM for other models

import torch
import os
import copy
from pathlib import Path
from .ijepa import init_model, load_checkpoint
import yaml
import torchvision.datasets
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from .utils import LabelToTensor, get_rank
import shutil


def load_ijepa_without_probe(checkpoint_path: Path, yaml_path: Path, device: str):
    with open(yaml_path, 'rb') as fp:
        yam = yaml.load(fp, yaml.Loader)

    encoder, predictor = init_model(
            device=device,
            patch_size=yam['mask']['patch_size'],
            crop_size=yam['data']['crop_size'],
            pred_depth=yam['meta']['pred_depth'],
            pred_emb_dim=yam['meta']['pred_emb_dim'],
            model_name=yam['meta']['model_name'])

    target_encoder = copy.deepcopy(encoder)

    # i'm betting that this just ignores the SNGP probe trained onto the model
    encoder, predictor, target_encoder, epoch = load_checkpoint(
        device,
        checkpoint_path,
        encoder,
        predictor,
        target_encoder,
    )
    
    return target_encoder


def load_and_project(checkpoint_path: Path, yaml_path: Path, voc_path: Path):
    device = get_rank() % torch.cuda.device_count()

    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    target_trans = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(224),
        LabelToTensor(255)
    ])

    # load the model itself
    target_encoder = load_pretrained_model(
        checkpoint_path,
        yaml_path,
        0
    )


    # initialize the dataset
    shutil.copy(voc_path, os.environ['LSCRATCH'])
    voc_seg_trn = torchvision.datasets.VOCSegmentation(
        os.environ['LSCRATCH'], 
        image_set='train',
        transform=trans,
        target_transform=target_trans,
        download=True
    )
    voc_seg_tst = torchvision.datasets.VOCSegmentation(
        os.environ['LSCRATCH'], 
        image_set='test',
        transform=trans,
        target_transform=target_trans,
        download=True
    )

    


