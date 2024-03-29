import torch
import torch.distributed as dist
import numpy as np
from torchmetrics import JaccardIndex
import wandb
import os

def setup():
    dist.init_process_group("nccl", rank=int(os.environ['RANK']), world_size=int(os.environ['WORLD_SIZE']))

def cleanup():
    dist.destroy_process_group()

def get_rank():
    return int(os.environ['RANK'])

def wandb_setup(args):
    if get_rank() != 0:
        print('initing wandb from a non-zero-rank process, weird stuff might happen')

    assert os.environ['WANDB_PROJECT'] is not None
    assert os.environ['WANDB_API_KEY'] is not None

    wandb.init(
        project=os.environ['WANDB_PROJECT'],
        config=vars(args)
    )


def train_ddp(rank, device, epoch, model, loader, loss_fn, optimizer, accumulate=2):
    jaccard = None
    step = 0
    ddp_loss = torch.zeros(5).to(device)
    model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        output = model(X) # ), update_precision=False)

        if jaccard is None:
            jaccard = JaccardIndex(
                task="multiclass",
                num_classes=output.shape[1],
                ignore_index=255
            ).to(device)

        loss = loss_fn(output, y.squeeze().type(torch.int64))
        loss.backward()
        if step % accumulate == (accumulate - 1):
            optimizer.step()
            optimizer.zero_grad()

        ddp_loss[0] += loss.item()
        ddp_loss[1] += torch.where(y != loss_fn.ignore_index, (output.argmax(1) == y), 0.0).sum().item()
        ddp_loss[2] += torch.where(y != loss_fn.ignore_index, 1.0, 0.0).sum().item()
        ddp_loss[3] += jaccard(output, y)
        ddp_loss[4] += 1

        step += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_acc = ddp_loss[1] / ddp_loss[2] 
    train_loss = ddp_loss[0] / ddp_loss[2]
    test_jaccard = ddp_loss[3] / ddp_loss[4]

    if rank == 0:

        accuracy = 100 * (ddp_loss[1] / ddp_loss[2])
        jaccard  = ddp_loss[3] / ddp_loss[4]
        avg_loss = ddp_loss[0] / ddp_loss[2]

        print(
            'Train Epoch: {} \tAccuracy: {:.2f}% \tJaccard: {:.2f} \tAverage Loss: {:.6f}'
            .format(
                epoch, 
                accuracy,
                jaccard,
                avg_loss
                # 100*(ddp_loss[1] / ddp_loss[2]), 
                # ddp_loss[3] / ddp_loss[4],
                # ddp_loss[0] / ddp_loss[2]
            )
        )

        wandb.log({
            'trn_epoch': epoch,
            'trn_accuracy': accuracy,
            'trn_jaccard': jaccard,
            'trn_avg_loss': avg_loss
        })


    return train_acc, test_jaccard, train_loss 


def test_ddp(rank, device, model, loader, loss_fn):
    ddp_loss = torch.zeros(5).to(device)
    model.eval()
    jaccard = None
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            if jaccard is None:
                jaccard = JaccardIndex(task="multiclass", num_classes=output.shape[1], ignore_index=255).to(device)
            loss = loss_fn(output, y.squeeze().type(torch.int64))
            ddp_loss[0] += loss.item()
            ddp_loss[1] += torch.where(y != loss_fn.ignore_index, (output.argmax(1) == y), 0).type(torch.float).sum().item()
            ddp_loss[2] += torch.where(y != loss_fn.ignore_index, 1.0, 0.0).sum().item()
            ddp_loss[3] += jaccard(output.argmax(1), y)
            ddp_loss[4] += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    test_acc = ddp_loss[1] / ddp_loss[2] 
    test_loss = ddp_loss[0] / ddp_loss[2]
    test_jaccard = ddp_loss[3] / ddp_loss[4]

    if rank == 0:

        accuracy = 100 * (ddp_loss[1] / ddp_loss[2])
        jaccard  = ddp_loss[3] / ddp_loss[4]
        avg_loss = ddp_loss[0] / ddp_loss[2]

        print('\tAccuracy: {:.2f}% \tJaccard: {:.2f} \tAverage Loss: {:.6f}'
            .format( 
                accuracy,
                jaccard,
                avg_loss
                # 100*(ddp_loss[1] / ddp_loss[2]), 
                # ddp_loss[3] / ddp_loss[4],
                # ddp_loss[0] / ddp_loss[2])
            )
        )

        wandb.log({
            'tst_accuracy': accuracy,
            'tst_jaccard': jaccard,
            'tst_avg_loss': avg_loss
        })

    return test_acc, test_jaccard, test_loss



def train(device, epoch, model, loader, loss_fn, optimizer):
    jaccard = None
    ddp_loss = torch.zeros(5).to(device)
    model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X, update_precision=False)
        if jaccard is None:
            jaccard = JaccardIndex(task="multiclass", num_classes=output.shape[1], ignore_index=255).to(device)
        loss = loss_fn(output, y.squeeze().type(torch.int64))
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += torch.where(y == loss_fn.ignore_index, 0.0, (output.argmax(1) == y)).sum().item()
        ddp_loss[2] += torch.where(y == loss_fn.ignore_index, 0.0, 1.0).sum().item()
        ddp_loss[3] += jaccard(output, y)
        ddp_loss[4] += 1

    train_acc = ddp_loss[1] / ddp_loss[2] 
    train_loss = ddp_loss[0] / ddp_loss[2]


    print('Train Epoch: {} \tAccuracy: {:.2f}% \tJaccard: {:.2f} \tAverage Loss: {:.6f}'
            .format(epoch, 
                    100*(ddp_loss[1] / ddp_loss[2]), 
                    ddp_loss[3] / ddp_loss[4],
                    ddp_loss[0] / ddp_loss[2])
                    )

    return train_acc, train_loss 


def test(device, model, loader, loss_fn):
    ddp_loss = torch.zeros(5).to(device)
    model.eval()
    jaccard = None
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            if jaccard is None:
                jaccard = JaccardIndex(task="multiclass", num_classes=output.shape[1], ignore_index=255).to(device)
            loss = loss_fn(output, y.squeeze().type(torch.int64))
            ddp_loss[0] += loss.item()
            ddp_loss[1] += torch.where(y == loss_fn.ignore_index, 0.0, (output.argmax(1) == y)).type(torch.float).sum().item()
            ddp_loss[2] += torch.where(y == loss_fn.ignore_index, 0.0, 1.0).sum().item()
            ddp_loss[3] += jaccard(output, y)
            ddp_loss[4] += 1


    test_acc = ddp_loss[1] / ddp_loss[2] 
    test_loss = ddp_loss[0] / ddp_loss[2]

    print('\tAccuracy: {:.2f}% \tJaccard: {:.2f} \tAverage Loss: {:.6f}'
            .format( 
                    100 * (ddp_loss[1] / ddp_loss[2]), 
                    ddp_loss[3] / ddp_loss[4],
                    ddp_loss[0] / ddp_loss[2])
                    )

    return test_acc, test_loss


class LabelToTensor:
    def __init__(self, nothing_class=255):
        self.nothing_class = nothing_class

    def __call__(self, y):
        y = torch.tensor(np.array(y)).type(torch.int64)
        y = torch.where(y == self.nothing_class, 255, y)
        return y
