"""Simulate Federated Learning Training

This script allows to simulate federated learning; the experiment name, the method and  be precised along side with the
 hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * train - simulate federated learning training

"""

from utils.utils import *
from utils.constants import *
from utils.args import TrainArgumentsManager

from torch.utils.tensorboard import SummaryWriter


def init_clients(args_, data_dir, logs_dir, chkpts_dir):
    """
    initialize clients from data folders

    :param args_:
    :param data_dir: path to directory containing data folders
    :param logs_dir: directory to save the logs
    :param chkpts_dir: directory to save chkpts
    :return: List[Client]

    """
    os.makedirs(chkpts_dir, exist_ok=True)

    print("===> Building data iterators..")
    train_iterators, val_iterators, test_iterators = \
        get_loaders(
            type_=LOADER_TYPE[args_.experiment],
            data_dir=data_dir,
            batch_size=args_.bz,
            is_validation=args_.validation
        )

    print("===> Initializing clients..")
    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue

        learner =\
            get_learner(
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

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        client = get_client(
            client_type=args_.client_type,
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=args_.local_steps,
            client_id=task_id,
            save_path=os.path.join(chkpts_dir, "task_{}.pt".format(task_id))
        )

        clients_.append(client)

    return clients_


def run_experiment(arguments_manager_):
    """

    :param arguments_manager_:
    :type arguments_manager_: ArgumentsManager

    """

    if not arguments_manager_.initialized:
        arguments_manager_.parse_arguments()

    args_ = arguments_manager_.args

    torch.manual_seed(args_.seed)

    data_dir = get_data_dir(args_.experiment)

    if "logs_dir" in args_:
        logs_dir = args_.logs_dir
    else:
        logs_dir = os.path.join("logs", arguments_manager_.args_to_string())

    if "chkpts_dir" in args_:
        chkpts_dir = args_.chkpts_dir
    else:
        chkpts_dir = os.path.join("chkpts", arguments_manager_.args_to_string())

    print("==> Clients initialization..")
    clients = \
        init_clients(
            args_,
            data_dir=os.path.join(data_dir, "train"),
            logs_dir=os.path.join(logs_dir, "train"),
            chkpts_dir=os.path.join(chkpts_dir, "train")
        )

    print("==> Test Clients initialization..")
    test_clients = \
        init_clients(
            args_,
            data_dir=os.path.join(data_dir, "test"),
            logs_dir=os.path.join(logs_dir, "test"),
            chkpts_dir=os.path.join(chkpts_dir, "test")
        )

    logs_path = os.path.join(logs_dir, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_dir, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    global_learner = \
        get_learner(
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

    aggregator = \
        get_aggregator(
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

    aggregator.write_logs()

    print("Training..")
    for ii in tqdm(range(args_.n_rounds)):
        aggregator.mix()

        if (ii % args_.log_freq) == (args_.log_freq - 1):
            aggregator.save_state(chkpts_dir)
            aggregator.write_logs()

    aggregator.save_state(chkpts_dir)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    arguments_manager = TrainArgumentsManager()
    arguments_manager.parse_arguments()

    run_experiment(arguments_manager)
