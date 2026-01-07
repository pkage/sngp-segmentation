import argparse
from pathlib import Path
import os
import torch

from dotenv import load_dotenv

from sngp_segmentation.train import training_process
from sngp_segmentation.utils import cleanup, get_rank, setup, wandb_setup
from dahps import DistributedAsynchronousRandomSearch as DARS
from dahps.torch_utils import sync_parameters
from config import experiment_config as config

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description='semi-supervised with calibrated uncertainty.')

    parser.add_argument('--fsdp', type=bool, default=False,
                        help='whether or not to reset the labeled and unlabeled sets after each iteration (default: %(default)s)')
    parser.add_argument('-warm', '--warmup', type=float, default=20,
                        help='number of epochs to freeze the backbone for (default: %(default)s)')
    parser.add_argument('-strat', '--strategy', type=str, default='self', help='Training strategy (default: %(default)s)')
    parser.add_argument('--voc_path', help='VOC file', type=Path, default='./VOCtrainval_11-May-2012.tar')
    parser.add_argument('--cityscapes_path', help='cityscapes directory', type=Path, default='./cityscapes')
    parser.add_argument('--coco_path', help='coco directory', type=Path, default='./coco_21')
    parser.add_argument('--deeplab_weights_path', help='deeplabv3 weights directory', type=Path, default='./deeplab_weights')
    parser.add_argument('--scratch_path', help='local scratch partition path', type=Path, default=os.environ['LSCRATCH'])
    parser.add_argument('--checkpoint_path', help='model checkpoint location', type=Path, default='./checkpoints')
    parser.add_argument('--path', help='hparam search directory path', default='./sngp_hparam', type=str)

    # parser.add_argument('--vit-ckpt', help='ViT checkpoint', default='../models/IN1K-vit.h.14-300e.pth.tar') # defaults to not breaking jay's code
    # parser.add_argument('--vit-cfg', help='ViT configuration', default='./in1k_vith14_ep300.yaml')

    return parser.parse_args()

def main():
    args = parse_args()

    setup()

    rank = int(os.environ["RANK"])

    device = rank % torch.cuda.device_count()
    print(f"rank {rank} running on device {device} (of {torch.cuda.device_count()})")
    torch.cuda.set_device(device)

    agent = DARS.from_config(args.path, config)

    agent = sync_parameters(rank, agent)

    args = agent.update_namespace(args)

    if get_rank() == 0:
        wandb_setup(args)

    states, metric = training_process(args)

    if rank == 0:
        print("saving checkpoint")
        agent.save_checkpoint(states)
        agent.finish_combination(metric)

    print("cleanup")
    cleanup()



if __name__ == '__main__':
    main()
