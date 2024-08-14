# code adapted from https://github.com/milesial/Pytorch-UNet

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy as copy


class DoubleConv(nn.Module):
    '''(convolution => [BN] => ReLU) * 2'''

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    '''Downscaling with maxpool then double conv'''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    '''Upscaling then double conv'''

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class Stacked_UNet_Ensemble(torch.nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, members=10):
        super().__init__()
        self.members = members

        self.ensemble = [UNet(n_channels, n_classes, bilinear) for member in range(members)]
        """

        stacked module state parameters and buffers are tensors not nn modules
        when sending this module to a device, these will not be sent

        """
        self.params, self.buffers = torch.func.stack_module_state(self.ensemble)

        for param in self.params:
            self.params[param] = torch.nn.Parameter(self.params[param])
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
        return torch.vmap(self.ensemble_wrapper, randomness='same')(self.params, self.buffers, data.unsqueeze(0).expand(self.members, -1, -1))

    def forward(self, x, with_variance=False):
        predictions = self.ensemble_fwd(x)

        # based on: https://arxiv.org/abs/2006.10108

        if not self.train:
          if with_variance:
              # return the prediction with the uncertainty value
              pred = (torch.sum(predictions, 0) / predictions.shape[0]) / (1 + (0.3*torch.var(predictions, 0)))**(0.5)
              pred_idx = torch.argmax(pred, -1)
              smax = torch.nn.functional.softmax(predictions, -1)
              unc = 1.0 - torch.gather(smax, -1, pred_idx.unsqueeze(-1)).squeeze()
              return pred, unc
          else:
              # return the calibrated prediction
              return (torch.sum(predictions, 0) / self.members) / (1 + (0.3*torch.var(predictions, 0)))**(0.5)
        else:
          return torch.sum(predictions, 0) / self.members

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
