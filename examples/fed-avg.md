# Simulating FedAvg

We provide an example for simulating a federated training using
FedAvg. We suppose that we have access to `train_loaders`, `val_loaders`
and `test_loaders` each given as a list of `torch.utils.data.DataLoader`
objects. We can use `utils.py/get_loaders`, for example, to generate
the data loaders:

```
from utils.utils import get_loaders

train_iterators, val_iterators, test_iterators = \
        get_loaders(
            type_=LOADER_TYPE[args_.experiment],
            data_dir=data_dir,
            batch_size=args_.bz,
            is_validation=args_.validation
        )
      
```

In addition to the data loaders, a client needs a `Learner` object
to be initialized. The `Learner` object takes care of training and
evaluating a  machine learning model. One can use
`utils.py/get_learner`, for example, to initialize a `learner`:

```
from client import Client
from utils.utils import get_learner

all_clients = []
for train_iterator, val_iterator, test_iterator in \
    zip(train_iterators, val_iterators, test_iterators):
    learner = get_learner(
        name=args_.experiment,
        model_name=args_.model_name,
        device=args_.device,
        optimizer_name=args_.optimizer,
        scheduler_name=args_.lr_scheduler,
        initial_lr=args_.lr,
        n_rounds=args_.n_rounds,
        seed=args_.seed,
        input_dimension=args_.input_dimension,
        hidden_dimension=args_.hidden_dimension,
        mu=args_.mu
    )
        
    client = get_client(
        client_type=args_.client_type,
        learner=learner,
        train_iterator=train_iterator,
        val_iterator=val_iterator,
        test_iterator=test_iterator,
        logger=logger,
        local_steps=args_.local_steps,
        client_id=task_id
    )
    
    all_client.append(client)
    
```

Once the list `all_clients` is generated, the aggregator can be
initialized as follows:

```
from aggregator import CentralizedAggregator


global_learner = get_learner(
    name=args_.experiment,
    model_name=args_.model_name,
    device=args_.device,
    optimizer_name=args_.optimizer,
    scheduler_name=args_.lr_scheduler,
    initial_lr=args_.lr,
    n_rounds=args_.n_rounds,
    seed=args_.seed,
    mu=args_.mu,
    input_dimension=args_.input_dimension,
    hidden_dimension=args_.hidden_dimension
)

aggregator = CentralizedAggregator(
    aggregator_type=args_.aggregator_type,
    clients=clients,
    global_learner=global_learner,
    sampling_rate=args_.sampling_rate,
    log_freq=args_.log_freq,
    global_train_logger=global_train_logger,
    global_test_logger=global_test_logger,
    test_clients=test_clients,
    verbose=args_.verbose,
    seed=args_.seed
)

```

The main training loop is as follows:

```
for ii in range(args_.n_rounds):
    aggregator.mix()

    if (ii % args_.log_freq) == (args_.log_freq - 1):
        aggregator.save_state(chkpts_dir)
        aggregator.write_logs()
        
```
