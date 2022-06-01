from utils.utils import *
from utils.args import TestArgumentsManager


def eval_knnper_grid(client_, weights_grid_, capacities_grid_):
    client_results = np.zeros((len(weights_grid_), len(capacities_grid_)))

    for ii, capacity in enumerate(capacities_grid_):
        client_.capacity = capacity
        client_.clear_datastore()
        client_.build_datastore()
        client_.gather_knn_outputs()

        for jj, weight in enumerate(weights_grid_):
            client_results[jj, ii] = client_.evaluate(weight) * client_.n_test_samples

    return client_results


def run(arguments_manager_):

    if not arguments_manager_.initialized:
        arguments_manager_.parse_arguments()

    args_ = arguments_manager_.args

    rng_seed = (args_.seed if (("seed" in args_) and (args_.seed >= 0)) else int(time.time()))
    rng = np.random.default_rng(seed=rng_seed)

    learner =\
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

    data_dir = get_data_dir(args_.experiment)

    weights_grid_ = np.arange(0, 1. + 1e-6, args_.weights_grid_resolution)
    capacities_grid_ = np.arange(0., 1. + 1e-6, args_.capacities_grid_resolution)

    all_scores_ = []
    n_test_samples_ = []

    _, train_loaders, test_loaders = \
        get_loaders(
            type_=LOADER_TYPE[args_.experiment],
            data_dir=os.path.join(data_dir, "train"),
            batch_size=args_.bz,
            is_validation=False
        )

    for train_loader, test_loader in tqdm(
            zip(train_loaders, test_loaders),
            total=len(train_loaders)
    ):
        if args_.verbose > 0:
            print(f"N_Train: {len(train_loader.dataset)} | N_Test: {len(test_loader.dataset)}")

        # TODO: need to adapt features_dimension in the case of ResNet
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

        if client.n_train_samples == 0 or client.n_test_samples == 0:
            continue

        client.compute_features_and_model_outputs()
        client.clear_datastore()

        client_scores = eval_knnper_grid(client, weights_grid_, capacities_grid_)

        n_test_samples_.append(client.n_test_samples)

        all_scores_.append(client_scores)

    all_scores_ = np.array(all_scores_)
    n_test_samples_ = np.array(n_test_samples_)

    return all_scores_, n_test_samples_, weights_grid_, capacities_grid_


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    arguments_manager = TestArgumentsManager()
    arguments_manager.parse_arguments()

    all_scores, n_test_samples, weights_grid, capacities_grid = run(arguments_manager)

    if "results_dir" in arguments_manager.args:
        results_dir = arguments_manager.args.results_dir
    else:
        results_dir = os.path.join("results", arguments_manager.args_to_string())

    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "all_scores.npy"), all_scores)
    np.save(os.path.join(results_dir, "n_test_samples.npy"), n_test_samples)
    np.save(os.path.join(results_dir, "weights_grid.npy"), weights_grid)
    np.save(os.path.join(results_dir, "capacities_grid.npy"), capacities_grid)
