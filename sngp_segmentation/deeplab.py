from pathlib import Path
from copy import deepcopy as copy

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50
import torchvision

from .utils import convleaves, getattrrecur, setattrrecur
from .unet import RandomFeatureGaussianProcess, Cos


def construct_deeplabv3_resnet50(
        weights: Path | None,
        num_classes
    ):

    # if weights:
    #     assert weights.exists()

    model = deeplabv3_resnet50(
        weights=weights,
        num_classes=num_classes
        # i think everything else should be okay as defaults, we're directly
        # comparing against a pascal-VOC baseline.
    )

    return model


def construct_deeplabv3_resnet101(
        weights: Path | None,
        num_classes
    ):
    if weights is not None:
        assert weights.exists()

    model = deeplabv3_resnet101(
        weights=weights,
        num_classes=num_classes
        # i think everything else should be okay as defaults, we're directly
        # comparing against a pascal-VOC baseline.
    )

    return model


class SNGPDeepLabV3_Resnet50(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, weights: Path | None = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1):
        super().__init__()
        module = construct_deeplabv3_resnet50(weights, n_classes)
        
        for name, _ in convleaves(module):
            setattrrecur(module, name, spectral_norm(getattrrecur(module, name)))

        self.deeplab = module

        self.deeplab.classifier._modules['4'] = torch.nn.Identity() # remove the classification head

        self.rfgp = RandomFeatureGaussianProcess(
            in_features=256,
            out_features=n_classes,
            backbone=nn.Identity(),
            n_inducing=512,
            momentum = 0.99,
            ridge_penalty = 1e-8,
            activation = torch.nn.ReLU(), #Cos(), # torch.nn.Sigmoid(),
            verbose = False,
        )
        
    def forward(self, x, with_variance: bool = False, update_precision: bool = True, freeze_backbone: bool = True):

        if freeze_backbone:
            with torch.no_grad():
                x = self.deeplab(x.float())['out']
        else:
            x = self.deeplab(x.float())['out']


        return self.rfgp(x, with_variance, update_precision)
    

class DeepLabV3_Resnet50(nn.Module):
    def __init__(self, n_channels, n_classes, spectral_norm=False, weights: Path | None = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, **kwargs):
        super().__init__()
        module = construct_deeplabv3_resnet50(weights, n_classes)
        
        if spectral_norm:
            for name, _ in convleaves(module):
                setattrrecur(module, name, spectral_norm(getattrrecur(module, name)))

        self.deeplab = module
        
    def forward(self, x, **kwargs):

        return self.deeplab(x)['out']


class DeepLabV3_Resnet101(nn.Module):
    def __init__(self, n_channels, n_classes, spectral_norm=False, weights: Path | None = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1, **kwargs):
        super().__init__()
        module = construct_deeplabv3_resnet101(weights, n_classes)
        
        if spectral_norm:
            for name, _ in convleaves(module):
                setattrrecur(module, name, spectral_norm(getattrrecur(module, name)))

        self.deeplab = module
        
    def forward(self, x, **kwargs):

        return self.deeplab(x.float())['out']


class Stacked_DeepLabV3_Resnet50_Ensemble(torch.nn.Module):
    def __init__(self, n_channels, n_classes, members=1, bilinear=False, weights: Path | None = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1):
        super().__init__()

        self.ensemble = [construct_deeplabv3_resnet50(weights, n_classes) for _ in range(members)]

        self.members = members
        """

        stacked module state parameters and buffers are tensors not nn modules
        when sending this module to a device, these will not be sent

        """
        self.params, self.buffers = torch.func.stack_module_state(self.ensemble)

        for param in self.params:
            # put some noise on the weights to create diversity in initialization
            self.params[param] = torch.nn.Parameter(self.params[param]) # torch.nn.Parameter(torch.normal(self.params[param], self.params[param].abs() / 100))
            self.register_parameter(param.replace('.', '#'), self.params[param])

        """

        meta tensor models are nn modules, so they will be sent to device with
        this one.  As such we must wrap them with a python class to avoid them
        being sent to a non-meta device, which will cause an error.

        """
        self.base_model = [copy(self.ensemble[0]).to('meta')]

    def ensemble_wrapper(self, params, buffers, data):
        return torch.func.functional_call(self.base_model[0], (params, buffers), (data,))

    def ensemble_fwd(self, data):
        return torch.vmap(self.ensemble_wrapper, randomness='same')(self.params, self.buffers, data.unsqueeze(0).expand(self.members, -1, -1, -1, -1))

    def forward(self, x, with_variance=False, **kwargs):
        predictions = self.ensemble_fwd(x.float())['out']

        # based on: https://arxiv.org/abs/2006.10108

        if not self.train:
          if with_variance:
              # return the prediction with the uncertainty value
              pred = torch.mean(predictions, 0) / (1 + (0.3*torch.var(predictions, 0)))**(0.5)
              pred_idx = torch.argmax(pred, -1)
              smax = torch.nn.functional.softmax(predictions, -1)
              unc = 1.0 - torch.gather(smax, -1, pred_idx.unsqueeze(-1)).squeeze()
              return pred, unc
          else:
              # return the calibrated prediction
              return torch.mean(predictions, 0) / (1 + (0.3*torch.var(predictions, 0)))**(0.5)
        else:
          return torch.mean(predictions, 0)

    def to(self, device):
      super().to(device)

      # we overrode this method to send our parameters and buffers
      # it must be done with no grad to avoid non-leaf tensor errors
      with torch.no_grad():
        for param in self.params:
            # must set the requires grad or it will be false
            param_requires_grad = self.params[param].requires_grad
            self.params[param] = self.params[param].to(device)
            self.params[param].requires_grad = param_requires_grad

        for buffer in self.buffers:
            if self.buffers[buffer] is not None:
              buffer_requires_grad = self.buffers[buffer].requires_grad
              self.buffers[buffer] = self.buffers[buffer].to(device)
              self.buffers[buffer].requires_grad = buffer_requires_grad

      return self
    # overrode this because the other parameters are not updated by optimizer step
    def parameters(self, recurse: bool = True):
      return [self.params[param] for param in self.params]

    def buffers(self, recurse: bool = True):
      return [self.buffers[buffer] for buffer in self.buffers]

    def state_dict(self, *args, **kwargs):
      state_dict1 = super().state_dict(*args, **kwargs)
      state_dict1.update({'params': copy(self.params), 'buffers': copy(self.buffers)})
      return state_dict1

    def load_state_dict(self, state_dict, *args, **kwargs):
      with torch.no_grad():
        for param in self.params:
            self.params[param].data = state_dict['params'][param].data

        self.buffers = state_dict['buffers']
        del state_dict['params']
        del state_dict['buffers']
      super().load_state_dict(state_dict, *args, **kwargs)
