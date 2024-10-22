from copy import deepcopy as copy
from typing import Tuple
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image


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
        labeled_inds = list(self.randperm[:int(len(self.dataset)*self.fraction_labeled)])

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
        labeled_inds = list(self.randperm[:int(len(self.dataset)*self.fraction_labeled)])

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
    def rank_unlabeled_by_confidence(self, model, batch_size=32, label_type='hard'):

        assert label_type in ['hard', 'soft'], 'label type must be one of ["hard", "soft"]'

        loader_unlabeled = DataLoader(self.unlabeled, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=12)

        hard_pls = []
        soft_pls = []

        for x, _ in loader_unlabeled:
            # generate the prediction
            soft_pl = model(x) # b, c, h, w
            none_class = soft_pl.shape[1]
            soft_pls.append(soft_pl.cpu())

            # argmax the prediction
            hard_pl = torch.argmax(soft_pl, 1) # b, h, w

            # replace the last index with 255
            hard_pl = torch.where(hard_pl == none_class, 255, hard_pl)

            # append the batch to the 
            hard_pls.append(hard_pl.cpu())

        soft_pls = torch.cat(soft_pls, 0)
        hard_pls = torch.cat(hard_pls, 0).type(torch.uint8)

        probs = torch.nn.functional.softmax(soft_pls, 1).mean(-1).mean(-1).max(-1)[0]
        order = torch.argsort(probs)

        if label_type == 'hard':
            return order, hard_pls
        elif label_type == 'soft':
            return order, soft_pls
        

    def save_pseudo_labels(self, labels, inds):

        label_paths = []

        for label, ind in zip(labels, inds):
            path = os.path.join(self.pseudo_label_directory, 'pl_' + str(ind) + '.png')
            Image.fromarray(np.array(label)).save(path)
            label_paths.append(path)

        return label_paths


    def pseudo_label(self, model, num_examples=0.05, with_replacement=True, batch_size=32):
        if isinstance(num_examples, float):
            num_examples = int(num_examples*len(self.unlabeled))

        if with_replacement:
            self.reset()

        # current number of pseudo labels is the amount of original unlabeled data minus the remaining amount
        num_pl = self.num_unlabeled_examples - len(self.unlabeled)

        # generate the predictions
        pl_inds_by_confidence, labels = self.rank_unlabeled_by_confidence(model, batch_size=batch_size)

        label_inds = list(range(num_pl, num_pl + num_examples))
        # save the prediction images to a unique path
        label_paths = self.save_pseudo_labels(labels[:num_examples], label_inds)

        assert len(label_paths) == len(label_inds), f'expected number of pseudo label saved images to match the number of pseudo labels, found {len(label_paths)} and {len(label_inds)}'

        # add the paths to the labeled data array
        for i, (path, u_ind) in enumerate(zip(label_paths, pl_inds_by_confidence)):
            self.labeled.images.append(self.unlabeled.images[u_ind])
            self.labeled.masks.append(path)

        # remove the added paths from the unlabeled data array
        for i in sorted(pl_inds_by_confidence[:num_examples])[::-1]:
            self.unlabeled.images.pop(i)
            self.unlabeled.masks.pop(i)

    def __len__(self):
        return len(self.dataset)
    

