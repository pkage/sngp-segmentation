# Pseudo-Labeling Algorithms for Improving Segementation with Learned Uncertainty Quantification



## Parameters relevant for pseudo-labeling methods

+ `strategy`

    The strategy that will be used for pseudo-labeling, one of `['mpl', 'self', 'baseline']`.  The `baseline` strategy uses no pseudo-labeling and only performs supervised training.  The other two options are [Meta Pseudo-Labels](https://arxiv.org/abs/2003.10580) and Self-Training.

### Self-Training:
+ `pl_fraction`

    The fraction of examples to bring in at each iteration of self-training.  This is the fraction of examples from the unlabeled set to which will actually be used in the training set for the next iteration.
+ `ul_fraction`

    The fraction of examples to hold out from the labeled set.  This is the complement of fraction of labels we use to train the model.  `ul_fraction` of `0.1` corresponds to using 90% of the labels available for training in this experiment.
+ `train_iterations`

    Iterations of self-training to perform.  This is the number of times that `pl_fraction` examples will be assigned a pseudo-label and introduced into the training set.
+ `with_replacement`

    Whether or not to return the pseudo-labeled examples to the unlabeled set after each iteration of self-training.  If this is set to true then the training set will never be larger than it is during the second iteration of self-training.

### Meta Pseudo-Labels

There are no parameter which are uniquely relevant to meta pseudo-labels experiments.

## Parameters relevant for UQ methods

### SNGP

+ `warmup`
    The number of epochs to train the SNGP layer with a frozen deeplab backbone.  This might be more effective than just training the whole model with a freshly initialized SNGP layer.

### Deep Ensembles

There are no parameters which are uniquely relevasnt to deep ensembles as a model choice, though it may make sense in the future to add the number of ensemble members as a parameter as theoretically a larger number of ensemble members should correspond to a monotonic increase in the performance of the uncertainty estimate.

## Parameters relevant for supervised training

+ `epochs`

    The number of epochs per iteration self-training or for the entire training run otherwise.
+ `accumulate`

    The number of gradient accumulation steps per gradient update.  The effective batch size is `batch_size * accumulate`.  On some hardware configurations using gradient accumulation can lead to higher throughput than not using it and simply leveraging a higher batch size (I/O latency, augmentation transform latency, layer latency etc. may not scale linearly or sub-linearly with batch size).

    This parameter is also commonly used to achieve a larger batch size than will fit in the VRAM of a GPU.
+ `batch_size`

    The batch size used during training.
+ `test_batch_size`

    The batch size used for evaluating the models - this can typically be 2x or more the batch size and may increase the throughput of the evaluation process.
+ `learning_rate`

    The initial learning rate used for Adam during the training runs.  The learning rate is used to start a [Cosine Annealing schedule](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
+ `patience`

    The number of epochs to wait for a jaccard index improvement before stopping the training run early.
+ `model`


    One of `['deeplab', 'unet', 'deep_ensemble', 'sngp']` which choose the model to train with the algorithm selected.  This determines the type of uncertainty quantification used and the starting weights.  

+ `dataset`


    One of `['cityscapes', 'pascal-voc', 'coco']`, the dataset on which train the model with the algorithm.

## wandb key setup

on the target machine:

```sh
$ cp template.env .env
$ vi .env
```

and replace the placeholder wandb api key.
