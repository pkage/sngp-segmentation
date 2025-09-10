import random
import time
import os
import pickle
import arrow
import pathlib
import torch

# remove all the files older than a day
def clean_dir(path):
    for f in os.listdir(path):
        if os.stat(os.path.join(path,f)).st_mtime < (time.time() - 86400):
            os.remove(os.path.join(path,f))


def sync_parameters(rank, agent):
    """
    Synchronizes the parameters between the ranks of a process that is performing
    a single model training run.

    Arguments

    args : argparse.Namespace
        Arguments parsed by an argparse.ArgumentParser.  The arguments parsed which are to
        be used in the search_space should be lists of values which the arguments can take.
        The other arguments should be singletons which will be used by the script.
    rank : int
        Rank of this process in the process world
    search_space : list[str, ...]
        List of keys from the args which the hyperparameter search agent searches over.
        These are the parameters which need to be synchronized between process ranks.
    agent_class : object
        One of the classes in this file (or a custom class that you define) whose parameters
        will be synchronized across the ranks.

    Returns

    hyperparameter search agent class instance
    """
    # the path to the hyperparameter search directory
    path = agent.root
    # generate and broadcast a unique integer - this will specify a path
    # broadcast the integer
    agree = random.randrange(0, 2**32)
    agree = torch.Tensor([agree]).to(torch.cuda.current_device())

    torch.distributed.broadcast(agree, 0)
    torch.distributed.all_reduce(agree)
    agree = int(agree.cpu()[0])

    if rank == 0:
        try:
            os.mkdir(os.path.join(path, "tmp"))
        except Exception as e:
            print(e)
        print(path)
        with open(os.path.join(path, f"tmp/{agree}.pkl"), "wb") as fp:
            pickle.dump(agent, fp)
    else:
        # wait for the root process to make the pickle file
        while not os.path.exists(os.path.join(path, f"tmp/{agree}.pkl")):
            time.sleep(1)

    # remove any tmp files older than a day
    clean_dir(os.path.join(path, "tmp"))

    # load the mutual file which holds the hyperparameters
    with open(os.path.join(path, f"tmp/{agree}.pkl"), "rb") as fp:
        agent = pickle.load(fp)
    # return the synchronized agent object
    return agent
