# CIFAR-10 Example

We provide an example to run an experiment with CIFAR-10 dataset

## Generate Dataset

In order to download the dataset and generate the federated split

```
cd data/cifar10

python generate_data.py \
    --n_tasks 200 \
    --by_labels_split \
    --n_components -1 \
    --alpha 0.1 \
    --s_frac 1.0 \
    --test_tasks_frac 0.0 \
    --seed 12345 

cd ../../
```

## Download Pretrained Models

To download CIFAR-10 pretrained models, run

```
mkdir chkpts
cd chkpts

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1g1qQsGWFPBb5yDWOXro9i8Gwd_XRohzj' -O 'cifar10-fedavg-alpha-1.0.pth'
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HAaVBzolYPGmmvSJr5jgkugR5pjgC3L4' -O 'cifar10-fedavg-alpha-0.7.pth'
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1SPhDq9-SVmAHS_XQQIa4QFk2fnyjYA7M' -O 'cifar10-fedavg-alpha-0.5.pth'
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=14SsQ5uXNa7kvjR01g1eufOAR9t5sMfMy' -O 'cifar10-fedavg-alpha-0.3.pth'
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yxSMq7m6e2-Pm8bKRJWW1AaF96LkDq_9' -O 'cifar10-fedavg-alpha-0.1.pth'

cd ../
```

## Evaluate kNN-Per

To evaluate kNN-Per, run

```eval
python eval_knnper.py \
    cifar10 \
    random \
    chkpts/cifar10-fedavg-alpha-0.3.pth \
    7 \
    --capacities_grid_resolution 0.01 \
    --weights_grid_resolution 0.01 \
    --bz 256 \
    --device cpu \
    --verbose 1 \
    --results_dir results/cifar10/alpha-0.3 \
    --seed 12345
```

## Plots

To plot the capacity effect, run

```plot
python make_plots.py  capacity_effect results/cifar10/alpha-0.3 --save_path plots/cifar10/alpha-0.3/capacity-effect.png
```

To plot the weights effect, run

```plot
python make_plots.py  weight_effect results/cifar10/alpha-0.3 --save_path plots/cifar10/alpha-0.3/weight-effect.png
```

