### Usage Example:

```python

# handles the low resolution datasets
low_res_config = {
    'key': 'image_size',
    'values': [64, 128],
    'default': {
        'key': 'crop_size',
        'values': [16, 32, 64],
        'default': None
    }
}
# handles the high resolution datasets
high_res_config = {
    'key': 'image_size',
    'values': [128, 256],
    'default': {
        'key': 'crop_size',
        'values': [16, 32, 64, 128],
        'default': None
    }
}

dataset_config = {
    'key': 'dataset',
    'values': ['caltech101', 'caltech256', 'food101', 'inat2021', 'cifar10', 'cifar100'],
    'default': high_res_config,
    'cifar10': low_res_config,
    'cifar100': low_res_config
},

optimization_config = {
    'key': 'learning_rate',
    'values': [1e-3, 1e-4],
    'default': {
        'key': 'batch_size',
        'values': [64, 128, 256, 512],
        'default': dataset_config
    }
}

deit_config = {
    'key': 'activation',
    'values': [],
    'default': {
        'key': 'patch_size',
        'values': [1, 2, 4, 16],
        'default': {
            'key': 'num_heads',
            'values': [3, 6, 12],
            'default': {
                'key': 'embed_dim',
                'values': [192, 384, 768],
                'default': {
                    'key': 'conv_first',
                    'values': [True, False],
                    'default': optimization_config
                }
            }
        }
    }
}

resnet_config = {
    'key': 'model',
    'values': ['complex_resnet18', 'complex_resnet34', 'complex_resnet50', 'complex_resnet101'],
    'default': optimization_config
}

model_config = {
    'key': 'model_type',
    'values': ['deit', 'resnet'],
    'default': resnet_config,
    'deit': deit_config
}

preprocessing_config = {
    'key': 'normalize',
    'values': [True, False],
    'default': {
        'key': 'magphase',
        'values': [True, False],
        'default': {
            'key': 'symlog',
            'values': [True, False],
            'default': model_config
        }
    }
}

experiment_config = {
    'root': {
        'key': 'experiment_type',
        'values': ['shearlet', 'fourier', 'baseline'],
        'default': model_config,
        'shearlet': {
            'key': 'n_shearlets',
            'value': [1, 3, 10],
            'default': preprocessing_config
        },
        'fourier': preprocessing_config
    },
    'check_unique': True,
    'repetitions': 1
}

```

```python
from config import experiment_config as config

def create_parser():
    parser = argparse.ArgumentParser(description="Shearlet NN")

    parser.add_argument(
        "--epochs", type=int, default=20, help="training epochs (default: 10)"
    )
    parser.add_argument(
        "--path", type=str, default='./hp_test', help="path for the hyperparameter search data"
    )


    return parser


def main(args, rank, world_size):
    setup(rank, world_size)

    device = rank % torch.cuda.device_count()
    print(f'rank {rank} running on device {device} (of {torch.cuda.device_count()})')
    torch.cuda.set_device(device)

    agent = DARS.from_config(args.path, config)

    agent = sync_parameters(rank, agent)

    args = agent.to_namespace(agent.combination)

    states = training_process(args, rank, world_size)

    if rank == 0:
        print('saving checkpoint')
        agent.save_checkpoint(states)

    print('cleanup')
    cleanup()

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.multiprocessing.set_start_method("spawn")

    main(args, rank, world_size)
```