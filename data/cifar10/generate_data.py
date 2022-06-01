"""
Download CIFAR-10 dataset, and splits it among clients.
"""
import os
import argparse
import pickle

import numpy as np

from torchvision.datasets import CIFAR10
from torch.utils.data import ConcatDataset

from sklearn.model_selection import train_test_split

from utils import by_labels_non_iid_split, pathological_non_iid_split, iid_split


N_CLASSES = 10
RAW_DATA_PATH = "raw_data/"
PATH = "all_data/"


def save_data(l, path_):
    with open(path_, 'wb') as f:
        pickle.dump(l, f)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Splits dataset among n_tasks. Default usage splits dataset in an IID fashion. Can be used with '
                    '`--pathological_split` or `--by_labels_split` for different methods of non-IID splits.'
                    'In case `--pathological_split` and `--by_labels_split` are both selected,'
                    ' `--by_ labels_split` will be used.'
    )
    parser.add_argument(
        '--n_tasks',
        help='number of tasks/clients;',
        type=int,
        required=True
    )
    parser.add_argument(
        '--pathological_split',
        help='if selected, the dataset will be split as follow:'
             '  1) sort the data by label'
             '  2) divide it into `n_clients * n_classes_per_client` shards, of equal size.'
             '  3) assign each of the `n_clients` with `n_classes_per_client` shards'
             'Similar to "Communication-Efficient Learning of Deep Networks from Decentralized Data"'
             '__(https://arxiv.org/abs/1602.05629);',
        action='store_true'
    )
    parser.add_argument(
        '--by_labels_split',
        help='if selected, the dataset will be split as follow:'
             '  1) classes are grouped into `n_clusters`'
             '  2) for each cluster `c`, samples are partitioned across clients using dirichlet distribution'
             'Inspired by "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440);',
        action='store_true'
    )
    parser.add_argument(
        '--n_shards',
        help='number of shards given to each clients/task; ignored if `--pathological_split` is not used;'
             'default is 2',
        type=int,
        default=2
    )
    parser.add_argument(
        '--n_components',
        help='number of components/clusters; ignored if `--by_labels_split` is not used; default is -1',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--alpha',
        help='parameter controlling tasks dissimilarity, the smaller alpha is the more tasks are dissimilar;'
             'ignored if `--by_labels_split` is not used; default is 0.5',
        type=float,
        default=0.5
    )
    parser.add_argument(
        '--s_frac',
        help='fraction of the dataset to be used; default: 1.0;',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--test_tasks_frac',
        help='fraction of tasks / clients not participating to the training; default is 0.0',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes; default is 12345',
        type=int,
        default=12345
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset =\
        ConcatDataset([
            CIFAR10(root=RAW_DATA_PATH, download=True, train=True),
            CIFAR10(root=RAW_DATA_PATH, download=False, train=False)
        ])

    if args.pathological_split:
        clients_indices =\
            pathological_non_iid_split(
                dataset=dataset,
                n_classes=N_CLASSES,
                n_clients=args.n_tasks,
                n_classes_per_client=args.n_shards,
                frac=args.s_frac,
                seed=args.seed
            )

    elif args.by_labels_split:
        clients_indices = \
            by_labels_non_iid_split(
                dataset=dataset,
                n_classes=N_CLASSES,
                n_clients=args.n_tasks,
                n_clusters=args.n_components,
                alpha=args.alpha,
                frac=args.s_frac,
                seed=args.seed
            )
    else:
        clients_indices = \
            iid_split(
                dataset=dataset,
                n_clients=args.n_tasks,
                frac=args.s_frac,
                seed=args.seed
            )

    if args.test_tasks_frac > 0:
        train_clients_indices, test_clients_indices = \
            train_test_split(clients_indices, test_size=args.test_tasks_frac, random_state=args.seed)
    else:
        train_clients_indices, test_clients_indices = clients_indices, []

    os.makedirs(os.path.join(PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(PATH, "test"), exist_ok=True)

    for mode, clients_indices in [('train', train_clients_indices), ('test', test_clients_indices)]:
        for client_id, indices in enumerate(clients_indices):

            indices = np.array(indices)
            train_indices = indices[indices < 50_000]
            test_indices = indices[indices >= 50_000]

            if (len(train_indices) == 0) or (len(test_indices) == 0):
                continue

            client_path = os.path.join(PATH, mode, "task_{}".format(client_id))
            os.makedirs(client_path, exist_ok=True)

            save_data(train_indices, os.path.join(client_path, "train.pkl"))
            save_data(test_indices, os.path.join(client_path, "test.pkl"))


if __name__ == "__main__":
    main()
