import copy
import os
import shutil

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import wandb
import yaml
from yaml import Loader

from .utils import LabelToTensor, test_ddp, train_ddp, get_rank
from .ijepa import init_model, load_checkpoint
from .sngp import SNGP_probe

def load_pretrained_model(yaml_path, checkpoint_path, device):
    with open(yaml_path, 'rb') as fp:
        yam = yaml.load(fp, Loader)

    encoder, predictor = init_model(
            device=device,
            patch_size=yam['mask']['patch_size'],
            crop_size=yam['data']['crop_size'],
            pred_depth=yam['meta']['pred_depth'],
            pred_emb_dim=yam['meta']['pred_emb_dim'],
            model_name=yam['meta']['model_name'])

    target_encoder = copy.deepcopy(encoder)

    encoder, predictor, target_encoder, epoch = load_checkpoint(
        device,
        checkpoint_path,
        encoder,
        predictor,
        target_encoder,
    )
    
    return target_encoder



def training_process(args):
    num_classes = 20 + 1
    device = int(os.environ['RANK']) % torch.cuda.device_count()

    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])

    target_trans = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(224),
        LabelToTensor(255)
    ])

    # load the voc file into our scratch space
    shutil.copy(args.voc, os.environ['LSCRATCH'])

    ds_train = torchvision.datasets.VOCSegmentation(os.environ['LSCRATCH'], image_set='train', transform=trans, target_transform=target_trans, download=True)
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=12)

    ds_val = torchvision.datasets.VOCSegmentation(os.environ['LSCRATCH'], image_set='val', transform=trans, target_transform=target_trans, download=True)
    loader_val = DataLoader(ds_val, batch_size=args.test_batch_size, pin_memory=True, shuffle=False, num_workers=12)

    target_encoder = load_pretrained_model(
        args.vit_cfg,
        args.vit_ckpt,
        0
    )
    target_encoder.requires_grad = False
    # target_encoder = load_pretrained_model('./in1k_vith14_ep300.yaml', '../models/IN1K-vit.h.14-300e.pth.tar', 0)


    model = SNGP_probe(target_encoder, 1280, num_classes, 14).to(device)

    # model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=num_classes, init_features=32, pretrained=False).to(device)

    model = DDP(model, device_ids=[device], find_unused_parameters=True)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)



    for epoch in range(args.epochs):
        train_ddp(
            get_rank(),
            device,
            epoch,
            model,
            loader_train,
            loss_fn,
            optimizer
        )

        test_ddp(
            get_rank(),
            device,
            model,
            loader_val,
            loss_fn
        )

