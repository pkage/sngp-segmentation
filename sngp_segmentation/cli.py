import click
from pathlib import Path
from dotenv import load_dotenv

from .train import training_process, TrainingArgs, validate_args
from .utils import cleanup, get_rank, setup, wandb_setup

@click.group(help='entrypoint for all sngpseg commands')
def cli():
    # entrypoint
    load_dotenv()

@cli.command('train', help='train a model')
@click.option('-e', '--epochs', type=int, default=10, help='number of training epochs (default 10)')
@click.option('-b', '--batch_size', type=int, required=True, help='number of instances per batch (shared betwen training)')
@click.option('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate for Adam (default 1e-4)')
@click.option('-p', '--patience', type=float, default=64, help='number of epochs to train for without improvement (default: 64)')
@click.option('-i', '--iterations', type=float, default=10, help='number of iteratons of self-training (default: 10)')
@click.option('-pf', '--pl_fraction', type=float, default=0.05,
                    help='fraction of unlabeled examples to label in each iteration (default: 0.05)')
@click.option('-rep', '--with_replacement', type=bool, is_flag=True,
                    help='whether or not to reset the labeled and unlabeled sets after each iteration')
@click.option('--strategy', type=str, default='baseline', help='learning strategies to use, unpacked into a set. Avaiable: mpl,self,baseline')
@click.option('--dataset', help='dataset to use', default='pascal-voc', type=click.Choice(['pascal-voc', 'coco', 'cityscapes']))
@click.option('--model', help='model to use (one of deeplab, unet), default %(default)', default='unet', type=click.Choice(['deeplab', 'unet']))
@click.option('--model_weights', type=click.Path(readable=True), help='if using a deeplab-based model, the pretrained weights to load')
@click.option('--voc_path', help='VOC source file', default='./VOCtrainval_11-May-2012.tar', type=click.Path(readable=True))
@click.option('--cityscapes_path', help='cityscapes directory', default='./cityscapes', type=click.Path(readable=True))
@click.option('--coco_path', help='VOC source file', default='./VOCtrainval_11-May-2012.tar', type=click.Path(readable=True))
@click.option('--fsdp', type=bool, is_flag=True, help='whether to ues mixed precision + fullyshardeddataparallel')
@click.option('--scratch_path', type=click.Path(file_okay=False, dir_okay=True, writable=True), required=True, help='scratch path to use')
@click.option('--checkpoint_path', type=click.Path(file_okay=False, dir_okay=True, writable=True), help='checkpoint path path to use')
def train(
        epochs,
        batch_size,
        learning_rate,
        patience,
        iterations,
        pl_fraction,
        with_replacement,
        strategy,
        dataset,
        model,
        model_weights,
        voc_path,
        cityscapes_path,
        coco_path,
        fsdp,
        scratch_path,
        checkpoint_path
    ):
    strategy = set(strategy.split(','))

    args = TrainingArgs(
        epochs=epochs,
        warmup=0, # currently unused
        batch_size=batch_size,
        test_batch_size=batch_size, # also unused but lets set it

        learning_rate=learning_rate,
        patience=patience,

        strategy=strategy,
        model=model,
        dataset=dataset,

        train_iterations=iterations,
        pl_fraction=pl_fraction,
        
        scratch_path=scratch_path,
        checkpoint_path=checkpoint_path,

        cityscapes_path=Path(cityscapes_path),
        voc_path=voc_path,
        coco_path=coco_path,
        deeplab_weights_path=model_weights,

        fsdp=fsdp,
        with_replacement=with_replacement
    )

    validate_args(args)

    if get_rank() == 0:
        wandb_setup(args)

    setup()
    training_process(args)
    cleanup()
