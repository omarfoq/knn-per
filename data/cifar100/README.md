# CIFAR10 Dataset

## Introduction
Splits `CIFAR100` dataset among `n_clients`, four methods are available:

### IID split (Default)
The dataset is shuffled and partitioned among `n_clients`

### By Labels Non-IID split
The dataset is split among `n_clients` as follow:
1. classes are grouped into `n_clusters`.
2. for each cluster `c` in `n_clusters`, samples are partitioned across clients using dirichlet distribution with parameter `alpha`.

Inspired by the split in [Federated Learning with Matched Averaging](https://arxiv.org/abs/2002.06440).

In order to use this mode, you should use argument `--by_labels_split`.

### Pathological Non-IID split
The dataset is split as follow:
1) sort the data by label
2) divide it into `n_clients * n_classes_per_client` shards, of equal size.
3) assign each of the `n_clients` with `n_classes_per_client` shards

Similar to [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)

In order to use this mode, you should use argument `--pathological_split`.

### Pachinko Allocation Split

Split CIFAR100 dataset among `n_clients` using a hierarchical Latent Dirichlet
Allocation (LDA) process, referred to as the
[Pachinko Allocation Method](https://people.cs.umass.edu/~mccallum/papers/pam-icml06.pdf) (PAM).
This method uses a two-stage LDA process, where each client has an associated
multinomial distribution over the coarse labels of CIFAR-100, and a
coarse-to-fine label multinomial distribution for that coarse label over the
labels under that coarse label. The coarse label multinomial is drawn from a
symmetric Dirichlet with the parameter $\alpha$, and each coarse-to-fine multinomial
distribution is drawn from a symmetric Dirichlet with the parameter $\beta$.
To generate a sample for the client, we first select
a coarse label by drawing from the coarse label multinomial distribution, and
then draw a fine label using the coarse-to-fine multinomial distribution. We
then randomly draw a sample from CIFAR-100 with that label (without
replacement). If this exhausts the set of samples with this label, we
remove the label from the coarse-to-fine multinomial and re-normalize the
multinomial distribution.

Inspired by the split in
[Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295)

## Instructions
Run generate_data.py with a choice of the following arguments:

- `--n_tasks`: number of tasks/clients, written as integer;
- `--pathological_split`: (`bool`) if selected; "pathological non-iid split" is used
- `--by_labels_split`: (`bool`) if selected; "by labels non-iid split" is used;
- `--n_shards`: number of shards given to each clients/task; ignored if `--pathological_split` is not used; default=``2``;
- `--n_components`: number of mixture components, written as integer, ignored if `--by_labels_split`  is not used; default=``-1``;
- `--alpha`: parameter controlling tasks dissimilarity, the smaller alpha is the more tasks are dissimilar; default=``0.5``;
- `--s_frac`: fraction of the dataset to be used; default=``1.0``;
- `--seed` := seed to be used before random sampling of data, default is `12345`;

### Remarks
- In case `--pathological_split` and `--by_labels_split` are both selected, `--by_ labels_split` will be used.
- If `n_components=-1`, then `n_components` will be set to be equal to `n_classes(=10)`.

## Paper Experiments

In order to generate the data split used in the paper, run

```
python generate_data.py \
    --n_tasks 200 \
    --by_labels_split \
    --n_components -1 \
    --alpha 0.3 \
    --s_frac 1.0 \
    --test_tasks_frac 0.0 \
    --seed 12345
```