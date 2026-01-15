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
    def rank_unlabeled_by_confidence(self, model, batch_size=32, label_type='hard', sngp=False):

        assert label_type in ['hard', 'soft'], 'label type must be one of ["hard", "soft"]'

        loader_unlabeled = DataLoader(self.unlabeled, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=12, drop_last=False)

        hard_pls = []
        soft_pls = []
        uncs = []

        if sngp:
            for x, _ in loader_unlabeled:
                with torch.no_grad():
                    soft_pl, unc = model(x, with_variance=True)
                    soft_pl = torch.nn.functional.softmax(soft_pl, 1)
                uncs.append(unc.cpu())
                # none_class = soft_pl.shape[1] - 1
                soft_pls.append(soft_pl.cpu())

                # argmax the prediction
                hard_pl = torch.argmax(soft_pl, 1) # b, h, w

                # replace the last index with 255
                hard_pl = torch.where(soft_pl.max(1)[0] < 1e-3, 255, hard_pl)

                # append the batch to the 
                hard_pls.append(hard_pl.cpu())

            soft_pls = torch.cat(soft_pls, 0)
            hard_pls = torch.cat(hard_pls, 0).type(torch.uint8)
            uncs = torch.cat(uncs)

            uncs = (uncs - uncs.min())
            uncs = (uncs / uncs.max()) ** 0.5
        else:
            for x, _ in loader_unlabeled:
                with torch.no_grad():
                    # generate the prediction
                    soft_pl = model(x) # b, c, h, w
                    soft_pl = torch.nn.functional.softmax(soft_pl, 1)
                # none_class = soft_pl.shape[1] - 1
                soft_pls.append(soft_pl.cpu())

                # argmax the prediction
                hard_pl = torch.argmax(soft_pl, 1) # b, h, w

                # replace the last index with 255
                hard_pl = torch.where(soft_pl.max(1)[0] < 1e-3, 255, hard_pl)

                # append the batch to the 
                hard_pls.append(hard_pl.cpu())

            soft_pls = torch.cat(soft_pls, 0)
            hard_pls = torch.cat(hard_pls, 0).type(torch.uint8)
            uncs = 1 - soft_pls.max(1)[0]
        
        assert (hard_pls <= 255).all(), 'class out of range'
        assert (0 <= hard_pls).all(), 'class out of range'
        # assert (hard_pls == 255).any(), f'no none class found {torch.unique(hard_pls)}, {soft_pl.shape}'
        assert not (hard_pls == 255).all(), f'only none class found {torch.unique(hard_pls)}, {soft_pl.shape}'

        print((hard_pls == 255).type(torch.float32).mean() * 100, '% masked')
        print(uncs.shape, hard_pls.shape)

        probs = torch.log((1 + 1e-4) - uncs).mean(-1).mean(-1)
        order = torch.argsort(probs, descending=True)

        print(torch.unique(hard_pls))

        if label_type == 'hard':
            return order, hard_pls[order]
        elif label_type == 'soft':
            return order, soft_pls[order]
        

    def save_pseudo_labels(self, labels, inds):

        label_paths = []

        for label, ind in zip(labels, inds):
            path = os.path.join(self.pseudo_label_directory, 'pl_' + str(ind) + '.png')
            Image.fromarray(np.array(label)).save(path)
            label_paths.append(path)

        return label_paths


    def pseudo_label(self, model, num_examples=0.05, with_replacement=True, batch_size=8, sngp=False):
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
        pl_inds_by_confidence, labels = self.rank_unlabeled_by_confidence(model, batch_size=batch_size, sngp=sngp)

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

