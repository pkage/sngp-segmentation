import copy
import os
from pathlib import Path
import shutil
import time
from dataclasses import dataclass, asdict
from typing import Literal, Set
from pprint import pprint

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
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
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
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
from .data import OneHotLabelEncode, SplitVOCDataset, CityscapesCategoryTransform, VOCLabelTransform
from .unet import SNGPUnet
from .sngp import SNGP_probe, SNGP_FPFT
from .deeplab import SNGPDeepLabV3_Resnet50, Stacked_DeepLabV3_Resnet50_Ensemble, DeepLabV3_Resnet50

# --- CONFIGURATION ---

@dataclass
class TrainingArgs:
    '''formalize the contents of `args` so we can be consistent about how we're invoking the runs'''
    # common
    epochs: int
    warmup: int
    batch_size: int
    test_batch_size: int

    learning_rate: float
    patience: float

    strategy: Set[Literal['mpl', 'self', 'baseline']]
    model: Literal['deeplab', 'unet', 'deep_ensemble', 'sngp']
    dataset: Literal['cityscapes', 'pascal-voc', 'coco']

    train_iterations: int
    pl_fraction: float

    scratch_path: Path
    checkpoint_path: Path | None # default to scratch_path / 'checkpoints'

    cityscapes_path: Path | None
    voc_path: Path | None
    coco_path: Path | None
    deeplab_weights_path: Path | None
    
    # optional params
    fsdp: bool = False
    with_replacement: bool = True


def validate_args(args: TrainingArgs):
    # basics
    assert args.epochs > 0
    assert args.batch_size > 0
    assert args.test_batch_size > 0

    assert args.learning_rate > 0
    assert args.patience > 0

    assert args.dataset in {'cityscapes', 'coco', 'pascal-voc'}

    assert args.scratch_path.exists()

    if 'baseline' in args.strategy:
        assert len(args.strategy) == 1, 'no combination of strategies allowed with baseline'
    if 'mpl' in args.strategy:
        assert 'self' in args.strategy, 'mpl requires self labeling'

    if 'self' in args.strategy:
        assert args.train_iterations > 0
        assert args.pl_fraction >= 0
# --- DATASET LOADING ---

    
def copy_datasets(args: TrainingArgs):
    if args.dataset == 'cityscapes':
        assert args.cityscapes_path is not None

        # only copy if we have to
        if not (args.scratch_path / args.cityscapes_path.name).exists():
            shutil.copy(args.cityscapes_path, os.environ['LSCRATCH'])
    
    elif args.dataset == 'pascal-voc':
        assert args.voc_path is not None

        # only copy if we have to
        if not (args.scratch_path / args.voc_path.name).exists():
            shutil.copy(args.voc_path, os.environ['LSCRATCH'])



def get_datasets(args: TrainingArgs):
    if args.dataset == 'cityscapes':
        assert args.cityscapes_path is not None
        n_classes = 8

        ds_train = Cityscapes(
            str(args.cityscapes_path), # shouldn't need string cast but it helps
            split='train',
            mode='fine',
            target_type='semantic',
            transform=Compose([
                transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
                PILToTensor()
            ]),
            target_transform=Compose([
                CityscapesCategoryTransform(),
                transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
                PILToTensor(),
                OneHotLabelEncode(n_classes)
            ])
        )
        ds_val  = Cityscapes(
            str(args.cityscapes_path),
            split='test',
            mode='fine',
            target_type='semantic',
            transform=Compose([
                transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
                PILToTensor()
            ]),
            target_transform=Compose([
                CityscapesCategoryTransform(),
                transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
                PILToTensor(),
                VOCLabelTransform(),
                OneHotLabelEncode(n_classes)
            ])
        )
        return ds_train, ds_val, n_classes

    elif args.dataset == 'pascal-voc':
        assert args.voc_path is not None
        n_classes = 20 + 1 + 1

        # imagenet transforms
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
            LabelToTensor(255),
            VOCLabelTransform(),
            OneHotLabelEncode(n_classes)
        ])

        ds_train = torchvision.datasets.VOCSegmentation(
            os.environ['LSCRATCH'],
            image_set='train',
            transform=trans,
            target_transform=target_trans,
            download=True
        )
        ds_val = torchvision.datasets.VOCSegmentation(
            os.environ['LSCRATCH'],
            image_set='val',
            transform=trans,
            target_transform=target_trans,
            download=True
        )
        return ds_train, ds_val, n_classes

    else:
        raise ValueError(f'unknown dataset: {args.dataset}')


# -- training loops

def training_process(args: TrainingArgs):
    pprint(args)
    # convenience
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # device setup (todo: support MPS for testing)
    device = rank % torch.cuda.device_count()
    print(f'rank {rank} running on device {device} (of {torch.cuda.device_count()})')
    torch.cuda.set_device(device)


    # copy the datasets into our scratch space if we're the lead worker
    if device == 0:
        copy_datasets(args)

        # cram in a lead worker creation of the checkpoint path
        if not args.checkpoint_path.exists():
            args.checkpoint_path.mkdir()

    # sync all workers
    dist.barrier()

    # load the datsets
    ds_train, ds_val, num_classes = get_datasets(args)
    print('num_classes', num_classes)

    # make a function for creating loaders
    def create_loader(dataset, val_mode=False):
        if val_mode:
            # don't need a sampler for val
            return DataLoader(
                dataset,
                batch_size=args.test_batch_size,
                pin_memory=True,
                shuffle=False, # does not need a sampler either
                num_workers=12,
                drop_last=True
            )


        sampler_train = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            sampler=sampler_train,
            num_workers=12,
            drop_last=True
        )


    

    if 'self' in args.strategy:
        ds_splitter = SplitVOCDataset(ds_train, fraction_labeled=args.pl_fraction)
    else:
        ds_splitter = None

    # this is where the real branching happens
    def create_model():
        if args.model == 'unet':
            print(f'creaing unet model with 3 input channels and {num_classes} outputs')
            model = SNGPUnet(
                3,
                num_classes,
            ).to(device)
        elif args.model == 'deeplab':
            print(f'creaing deeplab model with 3 input channels and {num_classes} outputs')
            model = DeepLabV3_Resnet50(
                3,
                num_classes,
                weights=args.deeplab_weights_path
            ).to(device)
        elif args.model == 'deep_ensemble':
            print(f'creaing deep_ensemble model with 3 input channels and {num_classes} outputs')
            model = Stacked_DeepLabV3_Resnet50_Ensemble(
                3,
                num_classes,
                weights=args.deeplab_weights_path
            ).to(device)
        elif args.model == 'sngp':
            print(f'creaing sngb_deeplab model with 3 input channels and {num_classes} outputs')
            model = SNGPDeepLabV3_Resnet50(
                3,
                num_classes,
                weights=args.deeplab_weights_path
            ).to(device)
        else:
            raise ValueError(f'no such model {args.model}')

        model = DDP(
            model,
            device_ids=[device],
            find_unused_parameters=True
        )

        if args.fsdp:
            auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy, min_num_params=10_000_000
            )
            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16, 
                    reduce_dtype=torch.float32, 
                    buffer_dtype=torch.float32, 
                    cast_forward_inputs=True
                )
            )

        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate
        )

        return model, optimizer

    # shared between all processes
    # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model, optimizer = create_model()

    if 'mpl' in args.strategy:
        teacher_model, teacher_optimizer = create_model()
    else:
        teacher_model, teacher_optimizer = None, None

    dist.barrier()

    for train_iteration in range(args.train_iterations):
        print(f'Starting iteration {train_iteration+1} of {args.train_iterations}...')
        for epoch in range(args.epochs):
            print(f'Epoch {epoch}/{args.epochs}')

            # creating these on the fly
            if 'self' in args.strategy:
                assert ds_splitter is not None
                loader_train = create_loader(ds_splitter.get_labeled(), val_mode=False)
            else:
                loader_train = create_loader(ds_train, val_mode=False)

            loader_train.sampler.set_epoch(epoch)

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

            loader_val = create_loader(ds_val, val_mode=True)

            test_ddp(
                get_rank(),
                device,
                model,
                loader_val,
                loss_fn
            )

        if 'mpl' in args.strategy:
            dist.barrier()

            loader_train = create_loader(ds_train, val_mode=False)
            loader_val = create_loader(ds_val, val_mode=True)
            assert ds_splitter is not None

            for epoch in range(args.epochs):
                mpl_ddp(
                    get_rank(),
                    device,
                    epoch,
                    teacher_model,
                    model,
                    loader_train,
                    ds_splitter.get_labeled(),
                    loss_fn,
                    teacher_optimizer,
                    optimizer,
                    accumulate=1
                )

                test_ddp(
                    get_rank(),
                    device,
                    teacher_model,
                    loader_val,
                    loss_fn
                )

        if 'self' in args.strategy:
            assert ds_splitter is not None
            ds_splitter.pseudo_label(model, args.pl_fraction, args.with_replacement)
        
        dist.barrier()
        
    
    return {
        'teacher': teacher_model, # probably None
        'model': model.state_dict(),
        'config': asdict(args)
    }


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
                mixed_precision=MixedPrecision(
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
