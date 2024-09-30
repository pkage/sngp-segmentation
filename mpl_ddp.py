import argparse

from dotenv import load_dotenv

from sngp_segmentation.train import mpl_training_process
from sngp_segmentation.utils import cleanup, get_rank, setup, wandb_setup

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description='co-training')

    parser.add_argument('-e', '--epochs', type=int, default=10, 
                        help='training epochs (default: %(default)s)')
    parser.add_argument('-w', '--warmup', type=int, default=10, 
                        help='training epochs (default: %(default)s)')
    parser.add_argument('-b', '--batch_size', type=int, default=64, 
                        help='batch size for training (default: %(default)s)')
    parser.add_argument('-tb', '--test_batch_size', type=int, default=64, 
                        help=' batch size for testing (default: %(default)s)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='learning rate for Adam (default: %(default)s)')
    parser.add_argument('-p', '--patience', type=float, default=64,
                        help='number of epochs to train for without improvement (default: %(default)s)')
    parser.add_argument('-i', '--iterations', type=float, default=10,
                        help='number of iteratons of self-training (default: %(default)s)')
    parser.add_argument('-pf', '--pl_fraction', type=float, default=0.05,
                        help='fraction of unlabeled examples to label in each iteration (default: %(default)s)')
    parser.add_argument('-rep', '--with_replacement', type=bool, default=True,
                        help='whether or not to reset the labeled and unlabeled sets after each iteration (default: %(default)s)')
    parser.add_argument('--voc', help='VOC file', default='./VOCtrainval_11-May-2012.tar')

    # parser.add_argument('--vit-ckpt', help='ViT checkpoint', default='../models/IN1K-vit.h.14-300e.pth.tar') # defaults to not breaking jay's code
    # parser.add_argument('--vit-cfg', help='ViT configuration', default='./in1k_vith14_ep300.yaml')

    return parser.parse_args()

def main():
    import os
    import torch
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()

    if get_rank() == 0:
        wandb_setup(args)

    setup()
    mpl_training_process(args)
    cleanup()


if __name__ == '__main__':
    main()