"""
Download CIFAR-100 dataset, and splits it among clients.
"""
import os
import argparse
import pickle

from torchvision.datasets import CIFAR100
from torch.utils.data import ConcatDataset

from sklearn.model_selection import train_test_split

from utils import *


RAW_DATA_PATH = "raw_data/"
PATH = "all_data/"

N_FINE_LABELS = 100
N_COARSE_LABELS = 20

COARSE_LABELS =\
    np.array([
        4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
        3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
        6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
        0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
        5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
        16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
        10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
        2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
        16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
        18, 1, 2, 15, 6, 0, 17, 8, 14, 13
    ])


def save_data(l, path_):
    with open(path_, 'wb') as f:
        pickle.dump(l, f)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Splits dataset among n_tasks. Default usage splits dataset in an IID fashion. Can be used with '
                    '`--pathological_split`, `--by_labels_split` or `pachinko_allocation` for different methods '
                    'of non-IID splits.'
                    'In case if usage `--pathological_split`, `--by_labels_split` and `--pachinko_allocation`'
                    'simultaneously, the order of preference is:'
                    ' `--pachinko_allocation`, `--by_labels_split` then `--pathological_split`.'
    )
    parser.add_argument(
        '--n_tasks',
        help='number of tasks/clients;',
        type=int,
        required=True)
    parser.add_argument(
        '--pachinko_allocation_split',
        help='split classification dataset among `n_clients` using pachinko allocation.'
             'This method works for datasets with coarse (super) labels, e.g. cifar-100.'
             'The dataset is split as follow:'
             '  1) Each client  has a symmetric Dirichlet distribution with parameter `alpha` over the coarse labels.'
             '  2) Each coarse label has a symmetric Dirichlet distribution with parameter `beta` over its fine labels.'
             '  3) To generate a sample for the client, we first select  a coarse label by drawing from the coarse'
             '      label multinomial distribution, and then draw a fine label using the coarse-to-fine multinomial'
             '      distribution. We then randomly draw a sample from CIFAR-100 with that label (without replacement).'
             '  4) If this exhausts the set of samples with this label, we remove the label from the coarse-to-fine'
             '      multinomial and re-normalize the multinomial distribution.'
             'Implementation follows the description in'
             ' "Adaptive Federated Optimization"__(https://arxiv.org/abs/2003.00295)',
        action='store_true'
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
        default=-1)
    parser.add_argument(
        '--alpha',
        help='parameter controlling tasks dissimilarity, the smaller alpha is the more tasks are dissimilar;'
             'ignored if `--by_labels_split` is not used; default is 0.5',
        type=float,
        default=0.5)
    parser.add_argument(
        '--beta',
        help='parameter controlling tasks dissimilarity, the smaller beta is the more tasks are dissimilar;'
             'used with `pachinko_allocation_split`; default is 10.0'
             'default is 10.0',
        type=float,
        default=10.
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
        default=12345)

    return parser.parse_args()


def main():
    args = parse_args()

    dataset =\
        ConcatDataset([
            CIFAR100(root=RAW_DATA_PATH, download=True, train=True),
            CIFAR100(root=RAW_DATA_PATH, download=False, train=False)
        ])

    if args.pathological_split:
        clients_indices =\
            pathological_non_iid_split(
                dataset=dataset,
                n_classes=N_FINE_LABELS,
                n_clients=args.n_tasks,
                n_classes_per_client=args.n_shards,
                frac=args.s_frac,
                seed=args.seed)

    elif args.by_labels_split:
        clients_indices = \
            by_labels_non_iid_split(
                dataset=dataset,
                n_classes=N_FINE_LABELS,
                n_clients=args.n_tasks,
                n_clusters=args.n_components,
                alpha=args.alpha,
                frac=args.s_frac,
                seed=args.seed
            )
    elif args.pachinko_allocation_split:
        clients_indices = \
            pachinko_allocation_split(
                dataset=dataset,
                n_clients=args.n_tasks,
                coarse_labels=COARSE_LABELS,
                n_fine_labels=N_FINE_LABELS,
                n_coarse_labels=N_COARSE_LABELS,
                alpha=args.alpha,
                beta=args.beta,
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
            train_test_split(
                clients_indices,
                test_size=args.test_tasks_frac,
                random_state=args.seed
            )
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
