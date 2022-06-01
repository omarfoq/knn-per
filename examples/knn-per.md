# kNN-Per example

We provide an example to evaluate kNN-Per. We suppose that we
have access to `train_loaders`
and `test_loaders` each given as a list of `torch.utils.data.DataLoader`
objects. We can use `utils.py/get_loaders`, for example, to generate
the data loaders:

```
from utils.utils import get_loaders

_, train_iterators, test_iterators = \
        get_loaders(
            type_=LOADER_TYPE[args_.experiment],
            data_dir=data_dir,
            batch_size=args_.bz,
            is_validation=args_.validation
        )
      
```

In addition to the data loaders, we need to initialize a `Learner`
object. A `Learner` object takes care of training and evaluating a
machine learning model. A `Learner` object can be initialized using
pretrained weights, available as a `.pt` (`.pth`) file stored in
`args.chkpts_path`. One can initialize a `Learner` object from
pretrained weights as follows:

```
from utils.utils import get_learner

learner = \
    get_learner(
        name=args_.experiment,
        model_name=args_.model_name,
        device=args_.device,
        optimizer_name="sgd",
        scheduler_name='constant',
        initial_lr=0.,
        mu=0.,
        n_rounds=0,
        seed=rng_seed,
        chkpts_path=args_.chkpts_path
    )
```

*When evaluating kNN-Per, the model is not updated,
that's why we set the learning rate, and the number of rounds to 0.*

For every `train loader` and `test loader`, we initialize a `KNNPerClient`
object. The `KNNPerClient` object models a client and its associated
datastore. It can be initialized as follows:

```
client = \
    KNNPerClient(
        learner=learner,
        train_iterator=train_loader,
        val_iterator=None,
        test_iterator=test_loader,
        logger=None,
        local_steps=None,
        k=args_.n_neighbors,
        interpolate_logits=args_.interpolate_logits,
        features_dimension=EMBEDDING_DIM[args_.experiment],
        num_classes=N_CLASSES[args_.experiment],
        capacity=-1,
        strategy=args_.strategy,
        rng=rng,
        device=args_.device,
    )
```

We then compute the features and model outputs using 
`client.compute_features_and_model_outputs()`, we populate the
clients' datastore with the computed features using
`client.build_datastore()`, and we gather the kNN outputs both on
the train samples and the test samples using `client.gather_knn_outputs()`.
The final prediction can be obtained using `client_.evaluate(weight)`,
where `weight` is the interpolation parameter between the outputs of the
global model, and the local memorization mechanism.  




