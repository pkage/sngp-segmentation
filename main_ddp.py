import argparse
from pathlib import Path
import os

from dotenv import load_dotenv

from sngp_segmentation.train import training_process
from sngp_segmentation.utils import cleanup, get_rank, setup, wandb_setup
from dahps import DistributedAsynchronousRandomSearch as DARS
from dahps.torch_utils import sync_parameters
from config import experiment_config as config

load_dotenv()

DEFAULT_LSCRATCH = Path(os.environ.get('LSCRATCH', './lscratch'))


def str2bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {'true', 't', '1', 'yes', 'y', 'on'}:
            return True
        if value in {'false', 'f', '0', 'no', 'n', 'off'}:
            return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def normalize_args(args):
    path_fields = (
        'scratch_path',
        'checkpoint_path',
        'voc_path',
        'cityscapes_path',
        'coco_path',
        'deeplab_weights_path',
    )
    for field in path_fields:
        value = getattr(args, field, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, field, Path(value))

    strategy_value = getattr(args, 'strategy', 'self')
    if isinstance(strategy_value, str):
        strategy_set = {item.strip() for item in strategy_value.split(',') if item.strip()}
    elif isinstance(strategy_value, (list, tuple, set)):
        strategy_set = {str(item).strip() for item in strategy_value if str(item).strip()}
    elif strategy_value:
        strategy_set = {str(strategy_value).strip()}
    else:
        strategy_set = set()
    if not strategy_set:
        strategy_set = {'self'}
    setattr(args, 'strategy', strategy_set)

    for field in ('dataset', 'model'):
        value = getattr(args, field, None)
        if isinstance(value, str):
            setattr(args, field, value.strip().lower())

    bool_fields = ('fsdp', 'with_replacement')
    for field in bool_fields:
        setattr(args, field, bool(getattr(args, field, False)))

    if not hasattr(args, 'train_iterations') and hasattr(args, 'iterations'):
        setattr(args, 'train_iterations', getattr(args, 'iterations'))

    int_fields = (
        ('epochs', 3),
        ('accumulate', 2),
        ('batch_size', 16),
        ('test_batch_size', 16),
        ('train_iterations', 10),
    )
    for field, fallback in int_fields:
        value = getattr(args, field, None)
        if value is None:
            value = fallback
        setattr(args, field, int(value))

    float_fields = ('learning_rate', 'patience', 'pl_fraction', 'ul_fraction', 'warmup')
    for field in float_fields:
        value = getattr(args, field, None)
        if value is not None:
            setattr(args, field, float(value))

    return args


def parse_args():
    parser = argparse.ArgumentParser(description='semi-supervised with calibrated uncertainty.')

    parser.add_argument('--epochs', type=int, default=3,
                        help='number of epochs to train for (default: %(default)s)')
    parser.add_argument('--accumulate', type=int, default=2,
                        help='number of steps to accumulate gradients before an optimizer step (default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='training batch size per device (default: %(default)s)')
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help='validation batch size per device (default: %(default)s)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='base learning rate (default: %(default)s)')
    parser.add_argument('--patience', type=float, default=10,
                        help='early stopping patience (default: %(default)s)')
    parser.add_argument('--dataset', type=str, choices=['pascal-voc', 'coco', 'cityscapes'], default='pascal-voc',
                        help='dataset to train on (default: %(default)s)')
    parser.add_argument('--model', type=str, choices=['deeplab', 'unet', 'deep_ensemble', 'sngp'], default='deeplab',
                        help='model architecture to use (default: %(default)s)')
    parser.add_argument('--ul_fraction', type=float, default=0.875,
                        help='fraction of data treated as unlabeled (default: %(default)s)')
    parser.add_argument('--train_iterations', '--iterations', dest='train_iterations', type=int, default=10,
                        help='number of self-training iterations to run (default: %(default)s)')
    parser.add_argument('--pl_fraction', type=float, default=1.0,
                        help='fraction of unlabeled data to pseudo-label per iteration (default: %(default)s)')
    parser.add_argument('--with_replacement', type=str2bool, default=True,
                        help='whether to sample unlabeled data with replacement when pseudo-labeling (default: %(default)s)')

    parser.add_argument('--fsdp', type=str2bool, default=False,
                        help='enable Fully Sharded Data Parallel (default: %(default)s)')
    parser.add_argument('-warm', '--warmup', type=float, default=0,
                        help='epochs to freeze the backbone (default: %(default)s)')
    parser.add_argument('-strat', '--strategy', type=str, default='self',
                        help='comma separated training strategies from {baseline,self,mpl} (default: %(default)s)')
    parser.add_argument('--voc_path', help='VOC file', type=Path, default=Path(f'{DEFAULT_LSCRATCH}/VOCtrainval_11-May-2012.tar'))
    parser.add_argument('--cityscapes_path', help='cityscapes directory', type=Path, default=Path(f'{DEFAULT_LSCRATCH}/cityscapes'))
    parser.add_argument('--coco_path', help='coco directory', type=Path, default=Path('./coco_21'))
    parser.add_argument('--deeplab_weights_path', help='deeplabv3 weights directory', type=Path, default=Path('./deeplab_weights'))
    parser.add_argument('--scratch_path', help='local scratch partition path', type=Path, default=DEFAULT_LSCRATCH)
    parser.add_argument('--checkpoint_path', help='model checkpoint location', type=Path, default=Path('./checkpoints'))
    parser.add_argument('--path', help='hparam search directory path', default='./sngp_hparam', type=str)
    parser.add_argument('--no-dahps', action='store_true',
                        help='disable DAHPS and run with the explicitly provided hyperparameters')

    return parser.parse_args()


def main():
    args = parse_args()

    setup()

    rank = int(os.environ["RANK"])

    agent = None
    if not args.no_dahps:
        agent = DARS.from_config(args.path, config)
        agent = sync_parameters(rank, agent)
        args = agent.update_namespace(args)

    args = normalize_args(args)

    if get_rank() == 0:
        wandb_setup(args)

    states, metric = training_process(args)

    if agent is not None and rank == 0:
        print("saving checkpoint")
        agent.save_checkpoint(states)
        agent.finish_combination(metric)

    print("cleanup")
    cleanup()


if __name__ == '__main__':
    main()
