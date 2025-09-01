import torch
import torch.distributed as dist
import numpy as np
from torchmetrics.classification import MulticlassJaccardIndex
import wandb
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

import torchvision
import torchvision.transforms.functional as TF
import random
from PIL import Image


def setup():
    dist.init_process_group(
        "nccl", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"])
    )


def cleanup():
    dist.destroy_process_group()


def get_rank():
    if not "RANK" in os.environ:
        print("warning: no rank information in environment")
        return 0
    return int(os.environ["RANK"])

def linearleaves(module):
    # returns a list of pairs of (parent, submodule_name) pairs for all submodule leaves of the current module
    if isinstance(module, torch.nn.Linear):
        return [(module, None)]

    linear_children = []
    for name, mod in module.named_modules():
        if isinstance(mod, torch.nn.Linear) or isinstance(mod, torch.nn.Conv2d):
            linear_children.append((name, module))
    return linear_children
        

def getattrrecur(mod, s):
    s = s.split('.')
    for substr in s:
        mod = getattr(mod, substr)
    return mod


def setattrrecur(mod, s, value):
    s = s.split('.')
    for substr in s[:-1]:
        mod = getattr(mod, substr)
    setattr(mod, s[-1], value)


def spectral_normalize(model):
    for name, mod in linearleaves(model):
        setattrrecur(model, name, torch.nn.utils.parametrizations.spectral_norm(getattrrecur(mod, name)))
    
    return model


def wandb_setup(args):
    if get_rank() != 0:
        print("initing wandb from a non-zero-rank process, weird stuff might happen")

    assert os.environ.get("WANDB_PROJECT") is not None
    assert os.environ.get("WANDB_API_KEY") is not None
    assert os.environ.get("WANDB_ENTITY") is not None

    wandb.init(project=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"], config=vars(args))


def train_ddp(rank, device, epoch, model, loader, loss_fn, optimizer, accumulate=2, warmup=5):
    jaccard = None
    step = 0
    ddp_loss = torch.zeros(5).to(device)
    model.train()
    for X, y in tqdm(loader):
        X, y = X.to(device), y.to(device)
        output = model(X, freeze_backbone=epoch < warmup)  # ), update_precision=False)

        if jaccard is None:
            jaccard = MulticlassJaccardIndex(
                num_classes=output.shape[1], ignore_index=255, average='macro', zero_division=1,
            ).to(device)

        # output_lng = output.type(torch.int64)
        # y_lng      = y.squeeze().type(torch.int64)

        loss = loss_fn(output, y.squeeze().type(torch.int64)) / accumulate
        # loss = loss_fn(
        #     output.type(torch.float32),
        #     y.squeeze()
        # ) / accumulate
        loss.backward()
        if step % accumulate == (accumulate - 1):
            optimizer.step()
            optimizer.zero_grad()


        ddp_loss[0] += loss.item()
        # we should be smarter about this when we have an ignore index:
        ddp_loss[1] += (
            torch.where(y != loss_fn.ignore_index, (output.argmax(1) == y), 0.0)
            .sum()
            .item()
        )
        ddp_loss[2] += torch.where(y != loss_fn.ignore_index, 1.0, 0.0).sum().item()
        # ddp_loss[1] += (output.argmax(1) == y.argmax(1)).sum().item()
        # ddp_loss[2] += y.argmax(1).numel()
        ddp_loss[3] += jaccard(output.argmax(1), y)
        ddp_loss[4] += 1

        step += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_acc = ddp_loss[1] / ddp_loss[2]
    train_loss = ddp_loss[0] / ddp_loss[2]
    test_jaccard = ddp_loss[3] / ddp_loss[4]

    if rank == 0:
        accuracy = 100 * (ddp_loss[1] / ddp_loss[2])
        jaccard = ddp_loss[3] / ddp_loss[4]
        avg_loss = ddp_loss[0] / ddp_loss[2]

        print(
            "Train Epoch: {} \tAccuracy: {:.2f}% \tJaccard: {:.2f} \tAverage Loss: {:.6f}".format(
                epoch,
                accuracy,
                jaccard,
                avg_loss,
                # 100*(ddp_loss[1] / ddp_loss[2]),
                # ddp_loss[3] / ddp_loss[4],
                # ddp_loss[0] / ddp_loss[2]
            )
        )

        wandb.log(
            {
                "trn_epoch": epoch,
                "trn_accuracy": accuracy,
                "trn_jaccard": jaccard,
                "trn_avg_loss": avg_loss,
            }
        )

        gc.collect()

    return train_acc, test_jaccard, train_loss


def mpl_ddp(
    rank,
    device,
    epoch,
    teacher,
    student,
    train_loader,
    unlabeled_loader,
    loss_fn,
    teacher_optimizer,
    student_optimizer,
    accumulate=2,
):
    jaccard = None
    step = 0
    ddp_loss = torch.zeros(5).to(device)
    teacher.train()
    student.train()
    for (X, y), (U, _) in tqdm(zip(train_loader, unlabeled_loader)):
        output = MPL_Seg(
            U.to(device),
            X.to(device),
            y.to(device),
            student,
            teacher,
            student_optimizer,
            teacher_optimizer,
            loss=loss_fn,
            sup_teacher=False,
            approx=False,
        )

        if jaccard is None:
            jaccard = MulticlassJaccardIndex(
                num_classes=output.shape[1], ignore_index=255
            ).to(device)

        ddp_loss[0] += 0.0
        # ddp_loss[1] += (
        #     torch.where(y.to(device) != loss_fn.ignore_index, (output.argmax(1) == y.to(device)), 0.0)
        #     .sum()
        #     .item()
        # )
        # ddp_loss[2] += torch.where(y != loss_fn.ignore_index, 1.0, 0.0).sum().item()
        ddp_loss[1] += (output.argmax(1) == y).sum().item()
        ddp_loss[2] += y.numel()
        ddp_loss[3] += jaccard(
            output.argmax(1),
            y.to(device).argmax(1)
        )
        ddp_loss[4] += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_acc = ddp_loss[1] / ddp_loss[2]
    train_loss = ddp_loss[0] / ddp_loss[2]
    test_jaccard = ddp_loss[3] / ddp_loss[4]

    if rank == 0:
        accuracy = 100 * (ddp_loss[1] / ddp_loss[2])
        jaccard = ddp_loss[3] / ddp_loss[4]
        avg_loss = ddp_loss[0] / ddp_loss[2]

        print(
            "Train Epoch: {} \tAccuracy: {:.2f}% \tJaccard: {:.2f} \tAverage Loss: {:.6f}".format(
                epoch,
                accuracy,
                jaccard,
                avg_loss,
                # 100*(ddp_loss[1] / ddp_loss[2]),
                # ddp_loss[3] / ddp_loss[4],
                # ddp_loss[0] / ddp_loss[2]
            )
        )

        wandb.log(
            {
                "trn_epoch": epoch,
                "trn_accuracy": accuracy,
                "trn_jaccard": jaccard,
                "trn_avg_loss": avg_loss,
            }
        )

        gc.collect()

    return train_acc, test_jaccard, train_loss


def test_ddp(rank, device, model, loader, loss_fn):
    ddp_loss = torch.zeros(5).to(device)
    model.eval()
    jaccard = None
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X, with_variance=False, update_precision=False)
            if jaccard is None:
                jaccard = MulticlassJaccardIndex(
                    num_classes=output.shape[1], ignore_index=255
                ).to(device)
            loss = loss_fn(output.type(torch.float32), y)
            ddp_loss[0] += loss.item()
            ddp_loss[1] += (
                torch.where(y != loss_fn.ignore_index, (output.argmax(1) == y), 0)
                .type(torch.float)
                .sum()
                .item()
            )
            # ddp_loss[2] += torch.where(y != loss_fn.ignore_index, 1.0, 0.0).sum().item()
            # ddp_loss[3] += jaccard(output.argmax(1), y)
            # ddp_loss[4] += 1
            # ddp_loss[1] += (output.argmax(1) == y.argmax(1)).sum().item()
            ddp_loss[2] += y.numel()
            ddp_loss[3] += jaccard(
                output.argmax(1),
                y.to(device)
            )
            ddp_loss[4] += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    test_acc = ddp_loss[1] / ddp_loss[2]
    test_loss = ddp_loss[0] / ddp_loss[2]
    test_jaccard = ddp_loss[3] / ddp_loss[4]

    if rank == 0:
        accuracy = 100 * (ddp_loss[1] / ddp_loss[2])
        jaccard = ddp_loss[3] / ddp_loss[4]
        avg_loss = ddp_loss[0] / ddp_loss[2]

        print(
            "\tAccuracy: {:.2f}% \tJaccard: {:.2f} \tAverage Loss: {:.6f}".format(
                accuracy,
                jaccard,
                avg_loss,
                # 100*(ddp_loss[1] / ddp_loss[2]),
                # ddp_loss[3] / ddp_loss[4],
                # ddp_loss[0] / ddp_loss[2])
            )
        )

        wandb.log(
            {"tst_accuracy": accuracy, "tst_jaccard": jaccard, "tst_avg_loss": avg_loss}
        )

        gc.collect()

    return test_acc, test_jaccard, test_loss


def train(device, epoch, model, loader, loss_fn, optimizer, accumulate=1):
    jaccard = None
    ddp_loss = torch.zeros(5).to(device)
    model.train()
    for (X, y), i in enumerate(loader):
        X, y = X.to(device), y.to(device)
        output = model(X, update_precision=False)
        if jaccard is None:
            jaccard = MulticlassJaccardIndex(
                num_classes=output.shape[1], ignore_index=255
            ).to(device)
        loss = loss_fn(output, y.type(torch.int64))
        loss.backward()
        if i % accumulate == accumulate - 1:
            optimizer.step()
            optimizer.zero_grad()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += (
            torch.where(y == loss_fn.ignore_index, 0.0, (output.argmax(1) == y))
            .sum()
            .item()
        )
        ddp_loss[2] += torch.where(y == loss_fn.ignore_index, 0.0, 1.0).sum().item()
        ddp_loss[3] += jaccard(output, y)
        ddp_loss[4] += 1

    train_acc = ddp_loss[1] / ddp_loss[2]
    train_loss = ddp_loss[0] / ddp_loss[2]

    print(
        "Train Epoch: {} \tAccuracy: {:.2f}% \tJaccard: {:.2f} \tAverage Loss: {:.6f}".format(
            epoch,
            100 * (ddp_loss[1] / ddp_loss[2]),
            ddp_loss[3] / ddp_loss[4],
            ddp_loss[0] / ddp_loss[2],
        )
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
                jaccard = MulticlassJaccardIndex(
                    num_classes=output.shape[1], ignore_index=255
                ).to(device)
            loss = loss_fn(output, y.squeeze().type(torch.int64))
            ddp_loss[0] += loss.item()
            ddp_loss[1] += (
                torch.where(y == loss_fn.ignore_index, 0.0, (output.argmax(1) == y))
                .type(torch.float)
                .sum()
                .item()
            )
            ddp_loss[2] += torch.where(y == loss_fn.ignore_index, 0.0, 1.0).sum().item()
            ddp_loss[3] += jaccard(output, y)
            ddp_loss[4] += 1

    test_acc = ddp_loss[1] / ddp_loss[2]
    test_loss = ddp_loss[0] / ddp_loss[2]

    print(
        "\tAccuracy: {:.2f}% \tJaccard: {:.2f} \tAverage Loss: {:.6f}".format(
            100 * (ddp_loss[1] / ddp_loss[2]),
            ddp_loss[3] / ddp_loss[4],
            ddp_loss[0] / ddp_loss[2],
        )
    )

    return test_acc, test_loss


class LabelToTensor:
    def __init__(self, nothing_class=255):
        self.nothing_class = nothing_class

    def __call__(self, y):
        y = torch.tensor(np.array(y)).type(torch.int64)
        y = torch.where(y == self.nothing_class, 255, y)
        return y


def convleaves(module):
    # returns a list of pairs of (parent, submodule_name) pairs for all submodule leaves of the current module
    if isinstance(module, torch.nn.Conv2d):
        return [(module, None)]

    conv_children = []
    for name, mod in module.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            conv_children.append((name, module))
    return conv_children


def getattrrecur(mod, s):
    s = s.split(".")
    for substr in s:
        mod = getattr(mod, substr)
    return mod


def setattrrecur(mod, s, value):
    s = s.split(".")
    for substr in s[:-1]:
        mod = getattr(mod, substr)
    setattr(mod, s[-1], value)


def get_uncertainty(ds, model, batch_size=1024, verbose=True):
    rank = int(os.environ["RANK"])
    device = rank % torch.cuda.device_count()
    dataloader = DataLoader(ds, batch_size=batch_size)
    uncs = []
    preds = []

    for X, y in tqdm(dataloader, total=len(dataloader), disable=not verbose):
        with torch.no_grad():
            pred, unc = model(X.to(device), with_variance=True)
        uncs.append(unc.cpu())
        preds.append(pred.cpu())

    uncs = torch.cat(uncs)
    preds = torch.cat(preds)

    uncs = uncs - uncs.min()
    uncs = (uncs / uncs.max()) ** 0.5

    return preds.detach().cpu(), uncs.detach().cpu()


def MPL_Seg(
    U,
    L,
    Y,
    student,
    teacher,
    student_optimizer,
    teacher_optimizer,
    loss=torch.nn.CrossEntropyLoss(),
    sup_teacher=False,
    approx=False,
):
    SPL = teacher(U)  # compute the soft pseudo labels
    probs = torch.nn.Softmax(1)(SPL).detach().cpu().numpy()
    PL = torch.tensor(
        [[[
            np.random.choice(
                np.arange(SPL.shape[1]),
                None,
                False,
                probs[xi, :, xj, xk],
            )
            for xk in range(SPL.shape[-1])
        ]
        for xj in range(SPL.shape[-2])
        ]
        for xi in range(SPL.shape[0])
        ]
    ).to(L.device)  # sample from the SPL distribution

    # compute the gradient of the student parameters with respect to the pseudo labels
    if approx:
        student_initial_output = student(L)
        student_loss_initial_l = loss(student_initial_output, Y).detach().clone()
    student_optimizer.zero_grad()

    student_initial_output = student(U)
    student_loss_initial = loss(student_initial_output, PL)

    student_optimizer.zero_grad()
    student_loss_initial.backward()
    grads1 = [dist.all_reduce(param.grad.data.detach().clone(), torch.distributed.ReduceOp.AVG) if param.grad is not None else None for param in student.parameters()]
    # update the student parameters based on pseudo labels from teacher
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    student_optimizer.step()

    student_optimizer.zero_grad()
    # compute the gradient of the student parameters with respect to the real labels
    student_final_output = student(L)
    student_loss_final = loss(student_final_output, Y)
    student_loss_final_l = student_loss_final.detach().clone()
    student_loss_final.backward()
    grads2 = [dist.all_reduce(param.grad.data.detach().clone(), torch.distributed.ReduceOp.AVG) if param.grad is not None else None for param in student.parameters()]

    student_optimizer.zero_grad()

    # compute the teacher MPL loss
    if approx:
        # https://github.com/google-research/google-research/issues/536
        # h is approximable by: student_loss_final - loss(student_initial(L), Y) where student_initial is before the gradient update for U
        h_approx = student_loss_initial_l - student_loss_final_l
        # this is the first order taylor approximation of the above loss, and apparently has finite deviation from the true quantity.
        # for correctness, I include instead the theoretically correct computation of h
        teacher_loss_mpl = h_approx.detach() * loss(SPL, PL)
    else:
        # compute h
        h = sum([(grad1 * grad2).sum() if (grad1 is not None and grad2 is not None) else 0.0 for grad1, grad2 in zip(grads1, grads2)])
        teacher_loss_mpl = h * loss(SPL, PL)

    if sup_teacher:  # optionally compute the teacher's supervised loss
        teacher_out = teacher(L)
        teacher_loss_sup = loss(teacher_out, Y)
        teacher_loss = teacher_loss_mpl + teacher_loss_sup
    else:
        teacher_loss = teacher_loss_mpl
    # update teacher based on student performance
    teacher_optimizer.zero_grad()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
    teacher_loss.backward()
    teacher_optimizer.step()
    # return the student output for the batch
    return student_final_output
    

class LikeTransformDataset:
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform

    @property
    def images(self):
        return self.ds.images
    
    @property
    def masks(self):
        return self.ds.masks

    def __iter__(self):
        for i in range(len(self)):
            assert self[i].dtype == torch.int64
            yield self[i]

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
