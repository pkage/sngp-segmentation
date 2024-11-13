import copy
import os
import shutil
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import Cityscapes
from torchvision.transforms import Compose, PILToTensor
import wandb
import yaml
from yaml import Loader
import functools

from .utils import LabelToTensor, test_ddp, train_ddp, mpl_ddp, get_rank
from .data import SplitVOCDataset
from .unet import SNGPUnet
from .sngp import SNGP_probe, SNGP_FPFT
from .deeplab import SNGPDeepLabV3_Resnet50, Stacked_DeepLabV3_Resnet50_Ensemble, DeepLabV3_Resnet50

# def load_pretrained_model(yaml_path, checkpoint_path, device):
#     from .ijepa import init_model, load_checkpoint
#     with open(yaml_path, 'rb') as fp:
#         yam = yaml.load(fp, Loader)

#     encoder, predictor = init_model(
#             device=device,
#             patch_size=yam['mask']['patch_size'],
#             crop_size=yam['data']['crop_size'],
#             pred_depth=yam['meta']['pred_depth'],
#             pred_emb_dim=yam['meta']['pred_emb_dim'],
#             model_name=yam['meta']['model_name'])

#     target_encoder = copy.deepcopy(encoder)

#     encoder, predictor, target_encoder, epoch = load_checkpoint(
#         device,
#         checkpoint_path,
#         encoder,
#         predictor,
#         target_encoder,
#     )
    
#     return target_encoder


def get_datasets(args):
    if args.dataset == 'cityscapes':
        ds_train = Cityscapes(
            args.cityscapes,
            split='train',
            mode='fine',
            target_type='semantic',
            transform=Compose([
                transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
                PILToTensor()
            ]),
            target_transform=Compose([
                transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
                PILToTensor()
            ])
        )
        ds_val  = Cityscapes(
            args.cityscapes,
            split='test',
            mode='fine',
            target_type='semantic',
            transform=Compose([
                transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
                PILToTensor()
            ]),
            target_transform=Compose([
                transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
                PILToTensor()
            ])
        )
        return ds_train, ds_val
    elif args.dataset == 'pascal-voc':
        shutil.copy(args.voc, os.environ['LSCRATCH'])
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

        ds_train = torchvision.datasets.VOCSegmentation(os.environ['LSCRATCH'], image_set='train', transform=trans, target_transform=target_trans, download=True)
        ds_val = torchvision.datasets.VOCSegmentation(os.environ['LSCRATCH'], image_set='val', transform=trans, target_transform=target_trans, download=True)
        return ds_train, ds_val
    else:
        raise ValueError(f'unknown dataset: {args.dataset}')


def training_process(args):
    num_classes = 20 + 1
    device = int(os.environ['RANK']) % torch.cuda.device_count()
    print(f'rank {os.environ["RANK"]} running on device {device} (of {torch.cuda.device_count()})')
    torch.cuda.set_device(device)

    checkpoint_path = os.path.join(os.environ['LSCRATCH'], 'checkpoints')

    if device == 0:
        # load the voc file into our scratch space
        shutil.copy(args.voc, os.environ['LSCRATCH'])

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
    else:
        while not os.path.exists(checkpoint_path):
            time.sleep(10)

    ds_train, ds_val = get_datasets(args)

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=12)
    loader_val = DataLoader(ds_val, batch_size=args.test_batch_size, pin_memory=True, shuffle=False, num_workers=12)

    if args.model == 'unet':
        model = SNGPUnet(
            3,
            num_classes,
        ).to(device)
    elif args.model == 'deeplab':
        model = SNGPDeepLabV3_Resnet50(
            3,
            num_classes,
            weights=args.model_weights
        ).to(device)
    else:
        raise ValueError(f'no such model {args.model}')

    model = DDP(
        model,
        device_ids=[device],
        find_unused_parameters=True
    )

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    best_jacc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        train_ddp(
            get_rank(),
            device,
            epoch,
            model,
            loader_train,
            loss_fn,
            optimizer,
            accumulate=2
        )

        acc, jacc, loss = test_ddp(
            get_rank(),
            device,
            model,
            loader_val,
            loss_fn
        )

        if jacc > best_jacc:
            best_state = copy.deepcopy(model.state_dict())
            jacc = best_jacc
    
    return best_state


def fpft_training_process(args, state_dict=None):
    num_classes = 20 + 1
    device = int(os.environ['RANK']) % torch.cuda.device_count()

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

    # load the voc file into our scratch space

    ds_train, ds_val = get_datasets(args)

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=12)
    loader_val = DataLoader(ds_val, batch_size=args.test_batch_size, pin_memory=True, shuffle=False, num_workers=12)

    if args.model == 'unet':
        model = SNGPUnet(
            3,
            num_classes,
        ).to(device)
    elif args.model == 'deeplab':
        model = SNGPDeepLabV3_Resnet50(
            3,
            num_classes,
            weights=args.model_weights
        ).to(device)
    else:
        raise ValueError(f'no such model {args.model}')

    # model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=num_classes, init_features=32, pretrained=False).to(device)
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=10_000_000
    )
    model = FSDP(model, 
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=torch.distributed.fsdp.MixedPrecision(
                    param_dtype=torch.bfloat16, 
                   reduce_dtype=torch.float32, 
                    buffer_dtype=torch.float32, 
                    cast_forward_inputs=True)
                )

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

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

    return model.state_dict()
"""
        if get_rank() == 0 and epoch % 10 == 0:
            torch.save(
                model.state_dict(), # technically this should be "model.module.state_dict", however the 
                                    # load_checkpoint code expects to have to strip the .module DDP cruft
                                    # so we're gonna leave it in
                os.path.join(checkpoint_path, f'ijepa_sngp_epoch{epoch}.pth')
            )
"""

def self_training_process(args):
    num_classes = 20 + 1
    device = int(os.environ['RANK']) % torch.cuda.device_count()
    print(f'rank {os.environ["RANK"]} running on device {device} (of {torch.cuda.device_count()})')
    torch.cuda.set_device(device)

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

    checkpoint_path = os.path.join(os.environ['LSCRATCH'], 'checkpoints')

    if device == 0:
        # load the voc file into our scratch space
        shutil.copy(args.voc, os.environ['LSCRATCH'])

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
    else:
        while not os.path.exists(checkpoint_path):
            time.sleep(10)

    ds = SplitVOCDataset(torchvision.datasets.VOCSegmentation(os.environ['LSCRATCH'], image_set='train', transform=trans, target_transform=target_trans, download=True), fraction_labeled=1 - args.ul_fraction)
    ds_train = ds.get_labeled()
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=12, drop_last=True)

    ds_val = torchvision.datasets.VOCSegmentation(os.environ['LSCRATCH'], image_set='val', transform=trans, target_transform=target_trans, download=True)
    loader_val = DataLoader(ds_val, batch_size=args.test_batch_size, pin_memory=True, shuffle=False, num_workers=12, drop_last=True)

    if args.model == 'unet':
        model = SNGPUnet(
            3,
            num_classes,
        ).to(device)
    elif args.model == 'deeplab':
        model = DeepLabV3_Resnet50(
            3,
            num_classes,
            weights=args.model_weights
        ).to(device)
    elif args.model == 'deep_ensemble':
        model = Stacked_DeepLabV3_Resnet50_Ensemble(
            3,
            num_classes,
            weights=args.model_weights
        ).to(device)
    elif args.model == 'sngp':
        model = SNGPDeepLabV3_Resnet50(
            3,
            num_classes,
            weights=args.model_weights
        ).to(device)
    else:
        raise ValueError(f'no such model {args.model}')

        """
        model = DDP(
            model,
            device_ids=[device],
            find_unused_parameters=True
        )
        """

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    best_jacc = 0.0
    best_state = None

    for iteration in range(args.iterations):
        for epoch in range(args.epochs):
            train_ddp(
                get_rank(),
                device,
                epoch,
                model,
                loader_train,
                loss_fn,
                optimizer,
                accumulate=2,
                warmup=args.warmup
            )

            acc, jacc, loss = test_ddp(
                get_rank(),
                device,
                model,
                loader_val,
                loss_fn
            )

            if jacc > best_jacc:
                best_state = copy.deepcopy(model.state_dict())
                jacc = best_jacc

        model.load_state_dict(best_state)
        ds.reset()
        ds.pseudo_label(model, args.pl_fraction, args.with_replacement)
        ds_train = ds.get_labeled()
        loader_train = DataLoader(ds_train, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=12, drop_last=True)

        print(len(ds_train))

        if iteration < args.iterations - 1:
            if args.model == 'unet':
                model = SNGPUnet(
                    3,
                    num_classes,
                ).to(device)
            elif args.model == 'deeplab':
                model = SNGPDeepLabV3_Resnet50(
                    3,
                    num_classes,
                    weights=args.model_weights
                ).to(device)
            else:
                raise ValueError(f'no such model {args.model}')
            """
            model = DDP(
                model,
                device_ids=[device],
                find_unused_parameters=True
            )
            """
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.learning_rate
            )
            
    return best_state


def mpl_training_process(args):
    num_classes = 20 + 1
    device = int(os.environ['RANK']) % torch.cuda.device_count()
    print(f'rank {os.environ["RANK"]} running on device {device} (of {torch.cuda.device_count()})')
    torch.cuda.set_device(device)

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

    checkpoint_path = os.path.join(os.environ['LSCRATCH'], 'checkpoints')

    if device == 0:
        # load the voc file into our scratch space
        shutil.copy(args.voc, os.environ['LSCRATCH'])

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
    else:
        while not os.path.exists(checkpoint_path):
            time.sleep(10)

    ds = SplitVOCDataset(torchvision.datasets.VOCSegmentation(os.environ['LSCRATCH'], image_set='train', transform=trans, target_transform=target_trans, download=True))
    ds_train = ds.get_labeled()
    ds_unlabeled = ds.get_unlabeled()
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=12)
    loader_unlabeled = DataLoader(ds_train, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=12)

    ds_val = torchvision.datasets.VOCSegmentation(os.environ['LSCRATCH'], image_set='val', transform=trans, target_transform=target_trans, download=True)
    loader_val = DataLoader(ds_val, batch_size=args.test_batch_size, pin_memory=True, shuffle=False, num_workers=12)

    if args.model == 'unet':
        student = SNGPUnet(
            3,
            num_classes,
        ).to(device)

        teacher = SNGPUnet(
            3,
            num_classes,
        ).to(device)
    elif args.model == 'deeplab':
        student = SNGPDeepLabV3_Resnet50(
            3,
            num_classes,
            weights=args.model_weights
        ).to(device)
        teacher = SNGPDeepLabV3_Resnet50(
            3,
            num_classes,
            weights=args.model_weights
        ).to(device)
    else:
        raise ValueError(f'no such model {args.model}')

    student = DDP(
        student,
        device_ids=[device],
        find_unused_parameters=True
    )

    teacher = DDP(
        teacher,
        device_ids=[device],
        find_unused_parameters=True
    )

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    student_optimizer = optim.Adam(
        student.parameters(),
        lr=args.learning_rate
    )

    teacher_optimizer = optim.Adam(
        teacher.parameters(),
        lr=args.learning_rate
    )

    for epoch in range(args.warmup):
        train_ddp(
            get_rank(),
            device,
            epoch,
            teacher,
            loader_train,
            loss_fn,
            teacher_optimizer,
            accumulate=1
        )

        test_ddp(
            get_rank(),
            device,
            teacher,
            loader_val,
            loss_fn
        )

    for epoch in range(args.epochs):
        mpl_ddp(
            get_rank(),
            device,
            epoch,
            teacher,
            student,
            loader_train,
            loader_unlabeled,
            loss_fn,
            teacher_optimizer,
            student_optimizer,
            accumulate=1
        )

        test_ddp(
            get_rank(),
            device,
            teacher,
            loader_val,
            loss_fn
        )
    
    return student.state_dict()
