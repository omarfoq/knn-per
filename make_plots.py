from utils.plots import *
from utils.args import PlotsArgumentsManager


if __name__ == "__main__":
    arguments_manager = PlotsArgumentsManager()
    arguments_manager.parse_arguments()

    results_dir = arguments_manager.args.results_dir
    if "save_path" in arguments_manager.args:
        save_path = arguments_manager.args.save_path
    else:
        save_path = None

    if arguments_manager.args.plot_name == "capacity_effect":
        fig, ax = plt.subplots(figsize=(12, 10))
        plot_capacity_effect(ax, results_dir=results_dir, save_path=save_path)

    elif arguments_manager.args.plot_name == "weight_effect":
        plot_weight_effect(results_dir=results_dir, save_path=save_path)

    elif arguments_manager.args.plot_name == "hetero_effect":
        plot_hetero_effect(results_dir=results_dir, save_path=save_path)

    else:
        raise NotImplementedError(
            f"{arguments_manager.args.plot_name} is not a valid plot name, possible are:"
            "{'capacity_effect', 'weight_effect', 'hetero_effect', 'n_neighbors_effect'} "
        )
