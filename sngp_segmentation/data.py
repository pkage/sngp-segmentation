from abc import ABC, abstractmethod
from copy import deepcopy as copy
import glob
import os
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
import einops
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Cityscapes
from torchvision.datasets.folder import default_loader


class SplitVOCDataset:
    def __init__(self, dataset, fraction_labeled=0.1, pl_directory='./pl'):
        self.dataset = dataset
        self.fraction_labeled = fraction_labeled
        self.pseudo_label_directory = pl_directory

        if not os.path.isdir(pl_directory):
            os.mkdir(pl_directory)

        # this is what we need to split to 
        imgs = self.dataset.images
        targets = self.dataset.masks

        generator = torch.Generator()
        generator.manual_seed(13)

        self.randperm = torch.randperm(len(self.dataset), generator = generator)
        unlabeled_inds = list(self.randperm[:int(len(self.dataset)*(1 - self.fraction_labeled))])
        labeled_inds = list(self.randperm[int(len(self.dataset)*(1 - self.fraction_labeled)):])

        self.unlabeled = copy(self.dataset)

        self.unlabeled.images.clear()
        self.unlabeled.masks.clear()

        for i in unlabeled_inds:
            self.unlabeled.images.append(self.dataset.images[i])
            self.unlabeled.masks.append(self.dataset.masks[i])

        self.labeled = copy(self.dataset)

        self.labeled.images.clear()
        self.labeled.masks.clear()

        for i in labeled_inds:
            self.labeled.images.append(self.dataset.images[i])
            self.labeled.masks.append(self.dataset.masks[i])

        self.num_labeled_examples = len(self.labeled)
        self.num_unlabeled_examples = len(self.unlabeled)

    def reset(self):
        generator = torch.Generator()
        generator.manual_seed(13)

        self.randperm = torch.randperm(len(self.dataset), generator = generator)

        unlabeled_inds = list(self.randperm[:int(len(self.dataset)*(1 - self.fraction_labeled))])
        labeled_inds = list(self.randperm[int(len(self.dataset)*(1 - self.fraction_labeled)):])

        self.unlabeled = copy(self.dataset)

        self.unlabeled.images.clear()
        self.unlabeled.masks.clear()

        for i in unlabeled_inds:
            self.unlabeled.images.append(self.dataset.images[i])
            self.unlabeled.masks.append(self.dataset.masks[i])

        self.labeled = copy(self.dataset)

        self.labeled.images.clear()
        self.labeled.masks.clear()

        for i in labeled_inds:
            self.labeled.images.append(self.dataset.images[i])
            self.labeled.masks.append(self.dataset.masks[i])


        self.num_labeled_examples = len(self.labeled)
        self.num_unlabeled_examples = len(self.unlabeled)

    def get_unlabeled(self):
        return copy(self.unlabeled)
    
    def get_labeled(self):
        return copy(self.labeled)
    
    @torch.no_grad()
    def rank_unlabeled_by_confidence(self, model, num_examples, batch_size=32, label_type='hard', sngp=False, use_amp=False):

        assert label_type in ['hard', 'soft'], 'label type must be one of ["hard", "soft"]'

        device = next(model.parameters()).device
        current_batch_size = batch_size

        def build_loader(bs):
            return DataLoader(
                self.unlabeled,
                batch_size=bs,
                pin_memory=True,
                shuffle=False,
                num_workers=12,
                drop_last=False
            )

        while True:
            try:
                if sngp:
                    loader_unlabeled = build_loader(current_batch_size)
                    unc_min = None
                    unc_max = None
                    for x, _ in loader_unlabeled:
                        x = x.to(device, non_blocking=True)
                        with torch.no_grad():
                            with torch.cuda.amp.autocast(enabled=use_amp):
                                _, unc = model(x, with_variance=True)
                        batch_min = unc.min().item()
                        batch_max = unc.max().item()
                        unc_min = batch_min if unc_min is None else min(unc_min, batch_min)
                        unc_max = batch_max if unc_max is None else max(unc_max, batch_max)

                    loader_unlabeled = build_loader(current_batch_size)
                    scores = []
                    denom = max(unc_max - unc_min, 1e-8)
                    for x, _ in loader_unlabeled:
                        x = x.to(device, non_blocking=True)
                        with torch.no_grad():
                            with torch.cuda.amp.autocast(enabled=use_amp):
                                soft_pl, unc = model(x, with_variance=True)
                                soft_pl = torch.nn.functional.softmax(soft_pl, 1)
                        unc = (unc - unc_min) / denom
                        unc = unc.clamp(min=0.0, max=1.0) ** 0.5
                        probs = torch.log((1 + 1e-4) - unc).mean(-1).mean(-1)
                        scores.append(probs.cpu())
                else:
                    loader_unlabeled = build_loader(current_batch_size)
                    scores = []
                    for x, _ in loader_unlabeled:
                        x = x.to(device, non_blocking=True)
                        with torch.no_grad():
                            with torch.cuda.amp.autocast(enabled=use_amp):
                                soft_pl = model(x)  # b, c, h, w
                                soft_pl = torch.nn.functional.softmax(soft_pl, 1)
                        unc = 1 - soft_pl.max(1)[0]
                        probs = torch.log((1 + 1e-4) - unc).mean(-1).mean(-1)
                        scores.append(probs.cpu())

                scores = torch.cat(scores, 0)
                order = torch.argsort(scores, descending=True)
                topk = order[:num_examples]

                topk_list = topk.tolist()
                if len(topk_list) == 0:
                    empty = torch.empty((0,), dtype=torch.uint8 if label_type == 'hard' else torch.float32)
                    return topk, empty

                topk_set = set(topk_list)
                rank_map = {idx: rank for rank, idx in enumerate(topk_list)}
                labels = [None] * len(topk_list)

                loader_unlabeled = build_loader(current_batch_size)
                offset = 0
                for x, _ in loader_unlabeled:
                    x = x.to(device, non_blocking=True)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            if sngp:
                                soft_pl, _ = model(x, with_variance=True)
                            else:
                                soft_pl = model(x)
                            soft_pl = torch.nn.functional.softmax(soft_pl, 1)

                    if label_type == 'hard':
                        hard_pl = torch.argmax(soft_pl, 1)
                        hard_pl = torch.where(soft_pl.max(1)[0] < 1e-3, 255, hard_pl)
                        hard_pl = hard_pl.cpu().type(torch.uint8)
                        for i in range(hard_pl.shape[0]):
                            global_idx = offset + i
                            if global_idx in topk_set:
                                labels[rank_map[global_idx]] = hard_pl[i]
                    else:
                        soft_pl = soft_pl.cpu()
                        for i in range(soft_pl.shape[0]):
                            global_idx = offset + i
                            if global_idx in topk_set:
                                labels[rank_map[global_idx]] = soft_pl[i]

                    offset += x.shape[0]

                if label_type == 'hard':
                    labels = torch.stack(labels, 0)
                    assert (labels <= 255).all(), 'class out of range'
                    assert (0 <= labels).all(), 'class out of range'
                    assert not (labels == 255).all(), f'only none class found {torch.unique(labels)}'
                else:
                    labels = torch.stack(labels, 0)
                break
            except torch.OutOfMemoryError:
                if current_batch_size == 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                torch.cuda.empty_cache()

        if label_type == 'hard':
            print((labels == 255).type(torch.float32).mean() * 100, '% masked')
            print(labels.shape)

        return topk, labels
        

    def save_pseudo_labels(self, labels, inds):

        label_paths = []

        for label, ind in zip(labels, inds):
            path = os.path.join(self.pseudo_label_directory, 'pl_' + str(ind) + '.png')
            Image.fromarray(np.array(label)).save(path)
            label_paths.append(path)

        return label_paths


    def pseudo_label(self, model, num_examples=0.05, with_replacement=True, batch_size=8, sngp=False, use_amp=False):
        if num_examples <= 1:
            num_examples = int(num_examples*len(self.unlabeled))
        else:
            num_examples = int(num_examples)

        if with_replacement:
            self.reset()

        # current number of pseudo labels is the amount of original unlabeled data minus the remaining amount
        num_pl = self.num_unlabeled_examples - len(self.unlabeled)

        if with_replacement:
            assert num_pl == 0

        # generate the predictions
        pl_inds_by_confidence, labels = self.rank_unlabeled_by_confidence(
            model,
            num_examples,
            batch_size=batch_size,
            sngp=sngp,
            use_amp=use_amp
        )

        if labels.numel() == 0:
            return

        label_inds = list(range(num_pl, num_pl + num_examples))
        # save the prediction images to a unique path
        label_paths = self.save_pseudo_labels(labels[:num_examples], label_inds)

        assert len(label_paths) == len(label_inds), f'expected number of pseudo label saved images to match the number of pseudo labels, found {len(label_paths)} and {len(label_inds)}'

        # add the paths to the labeled data array
        for _, (path, u_ind) in enumerate(zip(label_paths[:num_examples], pl_inds_by_confidence[:num_examples])):
            self.labeled.images.append(self.unlabeled.images[u_ind])
            self.labeled.masks.append(path)

        # remove the added paths from the unlabeled data array
        for i in sorted(pl_inds_by_confidence[:num_examples], reverse=True):
            self.unlabeled.images.pop(i)
            self.unlabeled.masks.pop(i)

    def __len__(self):
        return len(self.dataset)
    

class LabelTransform(ABC):
    mapping: Dict
    
    def __init__(self):
        self.mapping = self.build_mapping()
        
    @abstractmethod
    def build_mapping(self) -> Dict:
        pass

    def apply_mapping(self, target):
        arr = np.array(target)
        
        out_arr = arr.copy()
        for old_val, new_val in self.mapping.items():
            # create list of indices we care about for this rule
            idxs = arr == old_val
            out_arr[idxs] = new_val
        
        return Image.fromarray(out_arr)
    
    def __call__(self, target):
        return self.apply_mapping(target)

    
class CityscapesCategoryTransform(LabelTransform):
    def build_mapping(self):
        mapping = {}
        for ctycls in Cityscapes.classes:
            mapping[ctycls.id] = ctycls.category_id
        
        return mapping

class CityscapesTrainIDTransform(LabelTransform):
    def build_mapping(self):
        mapping = {}
        for ctycls in Cityscapes.classes:
            mapping[ctycls.id] = ctycls.train_id
        
        return mapping

class VOCLabelTransform():
    mapping: Dict
    
    def __init__(self):
        self.mapping = self.build_mapping()
        
    def build_mapping(self):
        return {
            255: 255
        }

    def apply_mapping(self, target):
        arr = np.array(target)
        
        out_arr = arr.copy()
        for old_val, new_val in self.mapping.items():
            # create list of indices we care about for this rule
            idxs = arr == old_val
            out_arr[idxs] = new_val
        
        return torch.tensor(out_arr)
    
    def __call__(self, target):
        return torchvision.transforms.Resize(
            (520, 520), interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )(torch.tensor(np.array(target)).to(torch.int64).unsqueeze(0)).squeeze(0)


def slice_off_last_channel(img):
    return img[:-1]


class OneHotLabelEncode:
    n_classes: int

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, labels):
        labels = labels.to(torch.int64)

        one_hot = F.one_hot(
            labels,
            num_classes=self.n_classes
        )

        # bad hack
        if len(one_hot.shape) == 5:
            one_hot = one_hot.squeeze(0)

        return einops.rearrange(one_hot, 'b h w c -> b c h w').squeeze(0)


class UnlabeledImageDataset(Dataset):
    def __init__(self, root_path: Path | str, file_types: List[str] = ['png', 'jpg'], transform=None):
        """
        Load a folder of images without labels
        """
        
        # build a glob list
        glob_targets = [f'{root_path}/**/*.{ft}' for ft in file_types]
        image_paths = []
        for target in glob_targets:
            image_paths += glob.glob(target)
        
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = default_loader(img_path)  # Loads image as PIL.Image probably
        
        if self.transform:
            image = self.transform(image)

        return image

