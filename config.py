training_config = {
    'key': 'epochs',
    'values': [25],
    'default': {
        'key': 'accumulate',
        'values': [4],
        'default': {
            'key': 'batch_size',
            'values': [32],
            'default': {
                'key': 'test_batch_size',
                'values': [32],
                'default': {
                    'key': 'learning_rate',
                    'values': [1e-3],
                    'default': {
                        'key': 'patience',
                        'values': [10],
                        'default': {
                            'key': 'dataset',
                            'values': ['pascal-voc'],
                            'default': None
                        },
                    },
                },
            },
        },
    },
}

pl_config = {
    'key': 'strategy',
    'values': ['mpl', 'self', 'baseline'],
    'default': training_config,
    'self': {
        'key': 'pl_fraction',
        'values': [0.1, 0.5, 1.0],
        'default': {
            'key': 'train_iterations',
            'values': [10],
            'default': {
                'key': 'with_replacement',
                'values': [True, False],
                'default': None
            },
        },
    },
}

experiment_config = {
    'root': {
        'key': 'model',
        'values': ['deeplab', 'unet', 'deep_ensemble', 'sngp'],
        'default': {
            'key': 'ul_fraction',
            'values': [i / 20 for i in range(20)],
            'default': pl_config
        } 
    },
    'check_unique': True,
    'repetitions': 1
}