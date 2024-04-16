##
# project voc samples into ijepa space, and save them back out
# this way, we don't have to produce all the representations on-the-fly,
# and save our VRAM for other models

import copy
import os
from pathlib import Path
import shutil

import h5py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import yaml

from .ijepa import init_model
from .utils import LabelToTensor, get_rank

def load_checkpoint_for_projection(
    device,
    r_path,
    encoder
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        
        # if 'epoch' in checkpoint:
        #     epoch = checkpoint['epoch']
        # else:
        #     epoch = 0
        #     print('no epoch info, assuming epoch zero')
        # -- loading encoder

        
        # manually rebuild the checkpoint dictionary
        pretrained_dict = {}
        for key, val in checkpoint.items():
            if not key.startswith('module.backbone'):
                continue

            key = key.replace('module.backbone.', '')
            pretrained_dict[key] = val


        msg = encoder.load_state_dict(pretrained_dict)

        print(f'loaded pretrained encoder with msg: {msg}')
        del checkpoint # do we need this?
        return encoder

    except Exception as e:
        raise

    # return encoder, predictor, target_encoder, epoch


def load_ijepa_without_probe(checkpoint_path: Path, yaml_path: Path, device: str):
    with open(yaml_path, 'rb') as fp:
        yam = yaml.load(fp, yaml.Loader)

    encoder, _ = init_model(
            device=device,
            patch_size=yam['mask']['patch_size'],
            crop_size=yam['data']['crop_size'],
            pred_depth=yam['meta']['pred_depth'],
            pred_emb_dim=yam['meta']['pred_emb_dim'],
            model_name=yam['meta']['model_name'])

    encoder = load_checkpoint_for_projection(
        device,
        checkpoint_path,
        encoder
    )
    
    return encoder


def project_loader(model, loader, device):
    features  = []
    labels    = []
    encodings = []

    for X, y in tqdm(loader):
        with torch.inference_mode():
            X = X.to(device)
            encoded = model(X)

        features.append(X.to('cpu'))
        labels.append(y.to('cpu'))
        encodings.append(encoded.to('cpu'))

    print('concatenating tensors...')
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    encodings = torch.cat(encodings, dim=0)

    return features, labels, encodings


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
    target_encoder = load_ijepa_without_probe(
        checkpoint_path,
        yaml_path,
        0
    )


    # initialize the dataset
    shutil.copy(voc_path, os.environ['LSCRATCH'])

    output_filename = 'voc_projection.h5'
    print(f'creating h5 dataset at {output_filename}...')
    with h5py.File(output_filename, 'w') as f:
        for ds_key in ['train', 'trainval', 'val']:
            print(f'projecting VOC split {ds_key}')

            voc_seg = torchvision.datasets.VOCSegmentation(
                os.environ['LSCRATCH'], 
                image_set=ds_key,
                transform=trans,
                target_transform=target_trans,
                download=True
            )
            loader = DataLoader(voc_seg, batch_size=4, pin_memory=True, shuffle=False, num_workers=12)

            features, labels, encodings = project_loader(
                target_encoder,
                loader,
                device
            )

            f.create_dataset(f'{ds_key}.features', data=features.numpy())
            f.create_dataset(f'{ds_key}.labels', data=labels.numpy())
            f.create_dataset(f'{ds_key}.encodings', data=encodings.numpy())

            print(f'completed projection of {ds_key}')

    print(f'dataset projection complete.')


