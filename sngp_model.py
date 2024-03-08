# adapted from https://github.com/alartum/sngp-pytorch with minor fixes by Jay Rothenbeger
import math
from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.nn.utils import spectral_norm

class Cos(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor):
        return torch.cos(X)


class RandomFeatureGaussianProcess(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        backbone: nn.Module = nn.Identity(),
        n_inducing: int = 1024,
        momentum: float = 0.9,
        ridge_penalty: float = 1e-6,
        activation: nn.Module = Cos(),
        verbose: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_inducing = n_inducing
        self.momentum = momentum
        self.ridge_penalty = ridge_penalty
        self.verbose = verbose
        self.backbone = backbone

        # Random Fourier features (RFF) layer
        projection = nn.Linear(in_features, n_inducing)
        projection.weight.requires_grad_(False)
        projection.bias.requires_grad_(False)

        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L96
        nn.init.kaiming_normal_(projection.weight, a=math.sqrt(5))
        nn.init.uniform_(projection.bias, 0, 2 * math.pi)

        self.rff = nn.Sequential(
            OrderedDict(
                [
                    ("backbone", backbone),
                    ("projection", projection),
                    ("activation", activation),
                ]
            )
        )

        # Weights for RFF
        self.weight = nn.Linear(n_inducing, out_features, bias=False)
        # Should be normally distributed a priori
        nn.init.kaiming_normal_(self.weight.weight, a=math.sqrt(5))

        self.pipeline = nn.Sequential(self.rff, self.weight)

        # RFF precision and covariance matrices
        self.is_fitted = False
        self.covariance = Parameter(
            1 / self.ridge_penalty * torch.eye(self.n_inducing),
            requires_grad=False,
        )
        # Ridge penalty is used to stabilize the inverse computation
        self.precision_initial = self.ridge_penalty * torch.eye(
            self.n_inducing, requires_grad=False
        )
        self.precision = Parameter(
            self.precision_initial,
            requires_grad=False,
        )

    def forward(
        self,
        X: torch.Tensor,
        with_variance: bool = False,
        update_precision: bool = False,
    ):
    
        features = self.rff(X).detach()

        if update_precision:
            self.update_precision_(features)

        logits = self.weight(features)
        if not with_variance:
            return logits
        else:
            if not self.is_fitted:
                raise ValueError(
                    "`compute_covariance` should be called before setting "
                    "`with_variance` to True"
                )
            with torch.no_grad():
                variances = torch.bmm(
                    features[:, None, :],
                    (features @ self.covariance)[:, :, None],
                ).reshape(-1)

            return logits, variances

    def reset_precision(self):
        self.precision[...] = self.precision_initial.detach()

    def update_precision_(self, features: torch.Tensor):
        with torch.no_grad():
            if self.momentum < 0:
                # Use this to compute the precision matrix for the whole
                # dataset at once
                self.precision[...] = self.precision + features.T @ features
            else:
                self.precision[...] = (
                    self.momentum * self.precision
                    + (1 - self.momentum) * features.T @ features
                )

    def update_precision(self, X: torch.Tensor):
        with torch.no_grad():
            features = self.rff(X)
            
            #features_list = [torch.zeros_like(features) for i in range(int(os.environ['WORLD_SIZE']))]
            #torch.distributed.all_gather(features_list, features)
            #features = torch.cat(features_list)

            self.update_precision_(features)

    def update_covariance(self):
        if not self.is_fitted:
            self.covariance[...] = (
                self.ridge_penalty * self.precision.cholesky_inverse()
            )
            self.is_fitted = True

    def reset_covariance(self):
        self.is_fitted = False
        self.covariance.zero_()


def get_uncertainty(ds, model, batch_size=1024, verbose=True):
    dataloader = DataLoader(ds, batch_size=batch_size)
    uncs = []

    for (X, y) in tqdm(dataloader, total=len(dataloader), disable=not verbose):
        with torch.no_grad():
            _, unc = model(X, with_variance=True)
        uncs.append(unc.cpu())

    uncs = torch.concat(uncs)
    uncs = (uncs - uncs.min())
    uncs = (uncs / uncs.max()) ** 0.5
    return uncs.detach().cpu()


def patches_to_image(x, patch_size=14):
    # x shape is batch_size x num_patches x c
    batch_size, num_patches, c = x.size()
    grid_size = int(num_patches ** 0.5)

    out_channels = c // (patch_size ** 2)

    x_image = x.view(batch_size, grid_size, grid_size, c)

    output_h = grid_size * patch_size
    output_w = grid_size * patch_size

    x_image = x_image.permute(0, 3, 1, 2).contiguous()

    x_image = x_image.view(batch_size, out_channels, output_h, output_w)
    return x_image


def SN_wrapper(layer, use_sn):
    if use_sn:
        return spectral_norm(layer)
    else:
        return layer

class Identity_module(nn.Module):
    def __init__(self):
        super(Identity_module, self).__init__()

    def forward(self, x):
        return x

class SNGP_probe(nn.Module):
    def __init__(self, backbone, backbone_features, num_classes=3, patch_size=14, embed_features=16):
        super(SNGP_probe, self).__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.linear = SN_wrapper(torch.nn.Linear(backbone_features, patch_size * patch_size * embed_features), True)
        self.logits = RandomFeatureGaussianProcess(
                                                    in_features=embed_features,
                                                    out_features=num_classes,
                                                    backbone=Identity_module(),
                                                    n_inducing=1024,
                                                    momentum = 0.9,
                                                    ridge_penalty = 1e-6,
                                                    activation = Cos(),
                                                    verbose = False,
                                                )

    def forward(self, x, update_precision=False, with_variance=False):
        with torch.no_grad():
            x = self.backbone(x).detach()
        x = self.linear(x)
        x = patches_to_image(x, self.patch_size) # b, c, w, h
        
        x = x.permute(0, 2, 3, 1) # b, w, h, c
        x = self.logits(x, with_variance, update_precision)
        x = x.permute(0, 3, 1, 2) # b, c, w, h
        return x
            