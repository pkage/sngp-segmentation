training_config = {
    'key': 'epochs',
    'values': [3],
    'default': {
        'key': 'accumulate',
        'values': [2],
        'default': {
            'key': 'batch_size',
            'values': [16],
            'default': {
                'key': 'test_batch_size',
                'values': [16],
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
    'values': ['self'],
    'default': training_config,
    'self': {
        'key': 'pl_fraction',
        'values': [1],
        'default': {
            'key': 'train_iterations',
            'values': [10],
            'default': {
                'key': 'with_replacement',
                'values': [True],
                'default': training_config
            },
        },
    },
}

experiment_config = {
    'root': {
        'key': 'model',
        'values': ['deeplab'],
        'default': {
            'key': 'ul_fraction',
            'values': [1 - 0.125],
            'default': pl_config
        } 
    },
    'check_unique': True,
    'repetitions': 1
}