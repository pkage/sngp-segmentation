from pretrained_ijepa import load_checkpoint, init_model
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import yaml
from yaml import Loader
import copy
from sngp_model import SNGP_probe
import argparse
import torch.optim as optim
from utils import train, test, LabelToTensor
from torchvision.transforms.functional import InterpolationMode
import numpy as np

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

    encoder, predictor, target_encoder, epoch = load_checkpoint(device,
                                                                checkpoint_path,
                                                                encoder,
                                                                predictor,
                                                                target_encoder,
                                                                )
    
    return target_encoder



def training_process(args):
    num_classes = 20 + 1

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


    ds_train = torchvision.datasets.VOCSegmentation('.', image_set='train', transform=trans, target_transform=target_trans, download=True)
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=12)

    ds_val = torchvision.datasets.VOCSegmentation('.', image_set='val', transform=trans, target_transform=target_trans, download=True)
    loader_val = DataLoader(ds_val, batch_size=args.test_batch_size, pin_memory=True, shuffle=False, num_workers=12)

    target_encoder = load_pretrained_model('./in1k_vith14_ep300.yaml', '../models/IN1K-vit.h.14-300e.pth.tar', 0)


    model = SNGP_probe(target_encoder, 1280, num_classes, 14).to(0)

    # model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=num_classes, init_features=32, pretrained=False).to(0)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        train(0, epoch, model, loader_train, loss_fn, optimizer)
        test(0, model, loader_val, loss_fn)


def parse_args():
    parser = argparse.ArgumentParser(description='co-training')

    parser.add_argument('-e', '--epochs', type=int, default=10, 
                        help='training epochs (default: %(default)s)')
    parser.add_argument('-b', '--batch_size', type=int, default=32, 
                        help='batch size for training (default: %(default)s)')
    parser.add_argument('-tb', '--test_batch_size', type=int, default=32, 
                        help=' batch size for testing (default: %(default)s)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='learning rate for Adam (default: %(default)s)')
    parser.add_argument('-p', '--patience', type=float, default=64,
                        help='number of epochs to train for without improvement (default: %(default)s)')
    return parser.parse_args()

def main():
    args = parse_args()
    training_process(args)


if __name__ == '__main__':
    main()