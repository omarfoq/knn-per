from aggregator import *
from client import *
from learners.learner import *
from models import *
from datasets import *

from .constants import *
from .metrics import *
from .optim import *

from torch.utils.data import DataLoader

from tqdm import tqdm


def get_data_dir(experiment_name):
    """
    returns a string representing the path where to find the datafile corresponding to the experiment
    :param experiment_name: name of the experiment
    :return: str
    """
    data_dir = os.path.join("data", experiment_name, "all_data")

    return data_dir


def get_loader(type_, path, batch_size, train, inputs=None, targets=None):
    """
    constructs a torch.utils.DataLoader object from the given path
    :param type_: type of the dataset; possible are `tabular`, `cifar10` and `cifar100`, `emnist`,
     `femnist` and `shakespeare`
    :param path: path to the data file
    :param batch_size:
    :param train: flag indicating if train loader or test loader
    :param inputs: tensor storing the input data; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :param targets: tensor storing the labels; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :return: torch.utils.DataLoader
    """
    if type_ == "tabular":
        dataset = TabularDataset(path)
    elif type_ == "cifar10":
        dataset = SubCIFAR10(path, cifar10_data=inputs, cifar10_targets=targets)
    elif type_ == "cifar100":
        dataset = SubCIFAR100(path, cifar100_data=inputs, cifar100_targets=targets)
    elif type_ == "femnist":
        dataset = SubFEMNIST(path)
    elif type_ == "shakespeare":
        dataset = CharacterDataset(path, chunk_len=SHAKESPEARE_CONFIG["chunk_len"])
    else:
        raise NotImplementedError(f"{type_} not recognized type; possible are {list(LOADER_TYPE.keys())}")

    if len(dataset) == 0:
        return

    # drop last batch, because of BatchNorm layer used in mobilenet_v2
    drop_last = (len(dataset) > batch_size) and train

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last, num_workers=NUM_WORKERS)


def get_loaders(type_, data_dir, batch_size, is_validation):
    """
    constructs lists of `torch.utils.DataLoader` object from the given files in `root_path`;
     corresponding to `train_iterator`, `val_iterator` and `test_iterator`;
     `val_iterator` iterates on the same dataset as `train_iterator`, the difference is only in drop_last
    :param type_: type of the dataset;
    :param data_dir: directory of the data folder
    :param batch_size:
    :param is_validation: (bool) if `True` validation part is used as test
    :return:
        train_iterator, val_iterator, test_iterator
        (List[torch.utils.DataLoader], List[torch.utils.DataLoader], List[torch.utils.DataLoader])
    """
    if type_ == "cifar10":
        inputs, targets = get_cifar10()
    elif type_ == "cifar100":
        inputs, targets = get_cifar100()
    else:
        inputs, targets = None, None

    train_iterators, val_iterators, test_iterators = [], [], []

    for task_id, task_dir in enumerate(tqdm(os.listdir(data_dir))):
        task_data_path = os.path.join(data_dir, task_dir)

        train_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=True
            )

        val_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            )

        if is_validation:
            test_set = "val"
        else:
            test_set = "test"

        test_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"{test_set}{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            )

        train_iterators.append(train_iterator)
        val_iterators.append(val_iterator)
        test_iterators.append(test_iterator)

    return train_iterators, val_iterators, test_iterators


def get_model(name, model_name, device, input_dimension=None, hidden_dimension=None, chkpts_path=None):
    """
    create model and initialize it from checkpoints

    :param name: experiment's name

    :param model_name: the name of the model to be used, only used when experiment is CIFAR-10, CIFAR-100 or FEMNIST
            possible are mobilenet and resnet

    :param device: either cpu or cuda

    :param input_dimension:

    :param hidden_dimension:

    :param chkpts_path: path to chkpts; if specified the weights of the model are initialized from chkpts,
                        otherwise the weights are initialized randomly; default is None.
    """
    if name == "synthetic":
        model = TwoLinearLayers(
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
            output_dimension=1
        )
    elif name == "cifar10":
        if model_name == "mobilenet":
            model = get_mobilenet(n_classes=10, pretrained=True)
        else:
            error_message = f"{model_name } is not a possible arrival process, available are:"
            for model_name_ in ALL_MODELS:
                error_message += f" `{model_name_};`"

            raise NotImplementedError(error_message)

    elif name == "cifar100":
        if model_name == "mobilenet":
            model = get_mobilenet(n_classes=100, pretrained=True)
        else:
            error_message = f"{model_name} is not a possible model, available are:"
            for model_name_ in ALL_MODELS:
                error_message += f" `{model_name_};`"

            raise NotImplementedError(error_message)
    elif name == "femnist":
        if model_name == "mobilenet":
            model = get_mobilenet(n_classes=62, pretrained=True)
        else:
            error_message = f"{model_name} is not a possible arrival process, available are:"
            for model_name_ in ALL_MODELS:
                error_message += f" `{model_name_};`"

            raise NotImplementedError(error_message)
    elif name == "shakespeare":
        model = NextCharacterLSTM(
                input_size=SHAKESPEARE_CONFIG["input_size"],
                embed_size=SHAKESPEARE_CONFIG["embed_size"],
                hidden_size=SHAKESPEARE_CONFIG["hidden_size"],
                output_size=SHAKESPEARE_CONFIG["output_size"],
                n_layers=SHAKESPEARE_CONFIG["n_layers"]
            )

    else:
        raise NotImplementedError(
            f"{name} is not available!"
            f" Possible are: `cifar10`, `cifar100`, `emnist`, `femnist` and `shakespeare`."
        )

    if chkpts_path is not None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model.load_state_dict(torch.load(chkpts_path, map_location=map_location)['model_state_dict'])
        except KeyError:
            try:
                model.load_state_dict(torch.load(chkpts_path, map_location=map_location)['net'])
            except KeyError:
                model.load_state_dict(torch.load(chkpts_path, map_location=map_location))

    model = model.to(device)

    return model


def get_client(
        client_type,
        learner,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps,
        client_id=None,
        save_path=None,
        k=None,
        interpolate_logits=None,
        features_dimension=None,
        num_classes=None,
        capacity=None,
        strategy=None,
        rng=None,
):
    """

    :param client_type:
    :param learner:
    :param train_iterator:
    :param val_iterator:
    :param test_iterator:
    :param logger:
    :param local_steps:
    :param client_id:
    :param save_path:
    :param k: number of neighbours used in KNNPerClient; default is None
    :param interpolate_logits: if selected logits are interpolated instead of probabilities
    :param features_dimension: feature space dimension of the embedding space;
                                only used with KNNPerClient; default is None
    :param num_classes: number of classes; only used with KNNPerClient; default is None
    :param capacity: datastore capacity; only used with KNNPerClient; default is None
    :param strategy: strategy to build the datastore; only used with KNNPerClient; default is None
    :param rng: random number generator; only used with KNNPerClient; default is None

    :return:
        Client

    """
    if client_type == "fedknn":
        return KNNPerClient(
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            interpolate_logits=interpolate_logits,
            k=k,
            features_dimension=features_dimension,
            num_classes=num_classes,
            capacity=capacity,
            strategy=strategy,
            rng=rng
        )
    else:
        return Client(
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            id_=client_id,
            save_path=save_path
        )


def get_learner(
        name,
        model_name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        n_rounds,
        seed,
        input_dimension=None,
        hidden_dimension=None,
        chkpts_path=None,
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param name: name of the experiment to be used; possible are
            {`synthetic`, `cifar10`, `emnist`, `shakespeare`}

    :param model_name: the name of the model to be used, only used when experiment is CIFAR-10, CIFAR-100 or FEMNIST
            possible are mobilenet and resnet

    :param device: used device; possible `cpu` and `cuda`

    :param optimizer_name: passed as argument to utils.optim.get_optimizer

    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler

    :param initial_lr: initial value of the learning rate

    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`

    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;

    :param seed:

    :param input_dimension:

    :param hidden_dimension:

    :param chkpts_path: path to chkpts; if specified the weights of the model are initialized from chkpts,
            otherwise the weights are initialized randomly; default is None.

    :return: Learner

    """
    torch.manual_seed(seed)

    if name == "synthetic":
        criterion = nn.MSELoss(reduction="none").to(device)
        metric = mse
        is_binary_classification = True
    elif name == "cifar10":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        is_binary_classification = False
    elif name == "cifar100":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        is_binary_classification = False
    elif name == "femnist":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        is_binary_classification = False
    elif name == "shakespeare":
        all_characters = string.printable
        labels_weight = torch.ones(len(all_characters), device=device)
        for character in CHARACTERS_WEIGHTS:
            labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
        labels_weight = labels_weight * 8
        criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)
        metric = accuracy
        is_binary_classification = False
    else:
        raise NotImplementedError

    model = \
        get_model(
            name=name,
            model_name=model_name,
            device=device,
            chkpts_path=chkpts_path,
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension
        )

    optimizer =\
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr_initial=initial_lr,
            mu=mu
        )

    lr_scheduler =\
        get_lr_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            n_rounds=n_rounds
        )

    if name == "shakespeare":
        return LanguageModelingLearner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )
    else:
        return Learner(
            model=model,
            model_name=model_name,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )


def get_aggregator(
        aggregator_type,
        clients,
        global_learner,
        sampling_rate,
        log_freq,
        global_train_logger,
        global_test_logger,
        test_clients,
        verbose,
        seed=None
):
    """

    :param aggregator_type:
    :param clients:
    :param global_learner:
    :param sampling_rate:
    :param log_freq:
    :param global_train_logger:
    :param global_test_logger:
    :param test_clients
    :param verbose: level of verbosity
    :param seed: default is None
    :return:

    """
    seed = (seed if (seed is not None and seed >= 0) else int(time.time()))

    if aggregator_type == "local":
        return NoCommunicationAggregator(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "centralized":
        return CentralizedAggregator(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )

    else:
        raise NotImplementedError(
            f"{aggregator_type} is not available!"
            f" Possible are: `local` and `centralized`."
        )
