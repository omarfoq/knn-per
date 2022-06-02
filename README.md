# Personalized Federated Learning through Local Memorization

This repository is the official implementation of
[Personalized Federated Learning through Local Memorization](https://arxiv.org/abs/2111.09360)

Federated learning allows clients to collaboratively learn statistical
models while keeping their data local. Federated learning was originally
used to train a unique global model to be served to all clients,
but this approach might be sub-optimal when clients' local data
distributions are heterogeneous. In order to tackle this limitation,
recent personalized federated learning methods train a separate model
for each client while still leveraging the knowledge available at other
clients. In this work, we exploit the ability of deep neural networks
to extract high quality vectorial representations (embeddings) from
non-tabular data, e.g., images and text, to propose a personalization
mechanism based on local memorization. Personalization is obtained
interpolating a pre-trained global model with a k-nearest neighbors
(kNN) model based on the shared representation provided by the global
model. We provide generalization bounds for the proposed approach,
and we show on a suite of federated datasets that this approach
achieves significantly higher accuracy and fairness than
state-of-the-art methods.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Additionally, FAISS should be installed. Instructions for the
installation of FAISS can be found
[here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)

## Usage

### Federated Training

We provide code to simulate federated training of machine learning. 
The core objects are `Aggregator` and `Client`; different federated learning
algorithms can be implemented by implementing the local update method
`Client.step()` and/or the aggregation protocol defined in
`Aggregator.mix()` and `Aggregator.update_client()`.

In addition to the trivial baseline consisting of training models locally without
any collaboration, this repository supports the following federated learning
algorithms:

* FedAvg ([McMahan et al. 2017](http://proceedings.mlr.press/v54/mcmahan17a.html))
* FdProx ([Li et al. 2018](https://arxiv.org/abs/1812.06127))
* Clustered FL ([Sattler et al. 2019](https://ieeexplore.ieee.org/abstract/document/9174890))
* pFedMe ([Dinh et al. 2020](https://proceedings.neurips.cc/paper/2020/file/f4f1f13c8289ac1b1ee0ff176b56fc60-Paper.pdf))
* L2SGD ([Hanzely et al. 2020](https://proceedings.neurips.cc/paper/2020/file/187acf7982f3c169b3075132380986e4-Paper.pdf))
* APFL ([Deng et al. 2020](https://arxiv.org/abs/2003.13461))
* q-FFL ([Li et al. 2020](https://openreview.net/forum?id=ByexElSYDr))
* AFL ([Mohri et al. 2019](http://proceedings.mlr.press/v97/mohri19a.html))
* Ditto ([Li et al. 2021](https://proceedings.mlr.press/v139/li21h.html))
* FedRep ([Collins et al. 2021](https://arxiv.org/abs/2102.07078))

An example for simulating a federated training using
FedAvg is provided in [examples/fed-avg.md](examples/fed-avg.md)


### kNN-Per

This repository implements kNN-Per described in
[Personalized Federated Learning through Local Memorization](https://arxiv.org/abs/2111.09360).
The object `KNNPerClient` represents a client with a local memory,
represented as a `Datastore` object.

An example of kNN-Per with CIFAR-10 dataset is provided in `examples/cifar-10.md`.

## Datasets

We provide four federated benchmark datasets spanning a wide range
of machine learning tasks: image classification (CIFAR10 and CIFAR100),
handwritten character recognition (FEMNIST), and language
modelling (Shakespeare), in addition to a synthetic dataset

Shakespeare dataset (resp. FEMNIST) was naturally partitioned by assigning
all lines from the same characters (resp. all images from the same writer)
to the same client.  We created federated versions of CIFAR10  by
distributing samples with the same label across the clients according to a 
symmetric Dirichlet distribution with parameter 0.3. For CIFAR100,
we exploited the availability of "coarse" and "fine" labels, using a
two-stage Pachinko allocation method  to assign 600 sample to each of
the 100 clients.

The following table summarizes the datasets and models

|Dataset         | Task |  Model |
| ------------------  |  ------|------- |
| FEMNIST   |     Handwritten character recognition       |     MobileNet-v2  |
| CIFAR10   |     Image classification        |      MobileNet-v2 |
| CIFAR100    |     Image classification         |      MobileNet-v2  |
| Shakespeare |     Next character prediction        |      Stacked LSTM    |

See the `README.md` files of respective dataset, i.e., `data/$DATASET`,
for instructions on generating data.


## Training

To train the base models used for Fed-kNN, run this command 

```train
python train.py 
    <dataset_name> \
    --aggregator_type centralized \
    --n_rounds 200 \
    --bz 128 \
    --lr 0.03 \
    --lr_scheduler multi_step \
    --log_freq 10 \
    --device cuda \
    --optimizer sgd \
    --seed 1234 \
    --logs_dir ./logs \
    --chkpts_dir ./chkpts/cifar10_fedavg
    --verbose 1
```


## Evaluation

To evaluate the score (accuracy) of kNN-Per, run this command

```eval
python eval_knnper.py \
    <dataset_name> \
    random \
    <chkpts_path> \
    <n_neighbors>\
    --capacities_grid_resolution 0.01 \
    --weights_grid_resolution 0.01 \
    --bz 256 \
    --device cuda \
    --verbose 1 \
    --results_dir <results_dir> \
    --seed 12345
```

This scripts will create an array (saved as an `.npy` file) of shape
(101, 101), each entry corresponds to the score (accuracy) of kNN-Per
for a value of $\lambda$ (weight) and datastore capacity.

## Pre-trained Models
You can download pretrained models here:
* Models trained using FedAvg on CIFAR-10 for different levels
of heterogeneity can be found
[here](https://drive.google.com/drive/folders/1lu0XdKO7GRtQdU99p6matevddU9TOr__?usp=sharing).
* Models trained using FedAvg on CIFAR-100 for different levels
of heterogeneity can be found
[here](https://drive.google.com/drive/folders/14VCYKYX9fHX9NXFXGvKboUaZ7-nCEUDh?usp=sharing).
* Model trained using FedAvg on Shakespeare can be found
[here](https://drive.google.com/drive/folders/1jlUFpXAcLyZ5EXmt7Qjz_yI4gdCMxCnm?usp=sharing)

To download all the pretrained models, run

```
mkdir chkpts
cd chkpts

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1g1qQsGWFPBb5yDWOXro9i8Gwd_XRohzj' -O 'cifar10-fedavg-alpha-1.0.pth'
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HAaVBzolYPGmmvSJr5jgkugR5pjgC3L4' -O 'cifar10-fedavg-alpha-0.7.pth'
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1SPhDq9-SVmAHS_XQQIa4QFk2fnyjYA7M' -O 'cifar10-fedavg-alpha-0.5.pth'
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=14SsQ5uXNa7kvjR01g1eufOAR9t5sMfMy' -O 'cifar10-fedavg-alpha-0.3.pth'
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yxSMq7m6e2-Pm8bKRJWW1AaF96LkDq_9' -O 'cifar10-fedavg-alpha-0.1.pth'

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lo3t4suq7gF2jFQWq8hCbC5lHk2zODyY' -O 'cifar100-fedavg-alpha-1.0.pth'
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ms4mvRPuKVFIfl3D7Ka0apnekaW8wxxa' -O 'cifar100-fedavg-alpha-0.7.pth'
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1mzIfbYElguv0hIbYPP-0-LzEudsp66bI' -O 'cifar100-fedavg-alpha-0.5.pth'
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1disK-4yUPNpVVul8YR4bhpSRxwy9GClf' -O 'cifar100-fedavg-alpha-0.3.pth'
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1JVyhhnBacFvhJ6q93lBn3P12A9IQT85m' -O 'cifar100-fedavg-alpha-0.1.pth'

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1oKBBHmMKT7aHKDXAhYynjOXDlNtKYjUu' -O 'shakespeare-fedavg.pth'

cd ../
```

## Results

### Average performance of personalized models

The performance  of  each  personalized  model
(which coincides with the global one in the case of FedAvg) is  evaluated
on  the  local  test  dataset  (unseen  at training).
The table below shows the average weighted accuracy with weights proportional 
to local dataset sizes. kNN-Per consistently achieves the highest 
accuracy across all datasets.

|Dataset         | Local | FedAvg| FedAvg+ | Clustered FL| Ditto | FedRep| APFL| kNN-Per (Ours)|
| ------|------|-------|-------------|-----------|----------|----------|-----------|------------|
| FEMNIST   |  71.0  |  83.4  | 84.3|83.7| 84.3| 85.3| 84.1 | **88.2** |
| CIFAR10   |  57.6  |  72.8    |75.2 |73.3|  80.0 | 77.7 | 78.9 | **83.0**
| CIFAR100    | 31.5  |  47.4 |51.4|47.2| 52.0| 53.2| 51.7 | **55.0**|
| Shakespeare |  32.0  |   48.1 |  47.0 |46.7|47.9 | 47.2 | 45.9 | **51.4**|


### Effect of local dataset's size

To plot the effect of the datastore capacity on the accuracy obtained
by kNN-Per, run

```plot
python make_plots.py  capacity_effect <results_dir> --save_path <save_path>
```


### Effect of the mixing weight

To plot the effect of the mixing weight  on the accuracy obtained by Fed-kNN, run

```plot
python make_plots.py weight_effect <results_dir> --save_path <save_path>
```

![weight_effect](https://user-images.githubusercontent.com/42912620/171421358-54217c16-d634-4d0c-9de8-4307d85f632a.png)

### Effect of data heterogeneity (only for CIFAR-10 and CIFAR-100)

To plot the effect of the data heterogeneity on the score obtained
by Fed-kNN, run this command

```
cd scripts/<dataset_name>
chmod +x heterogeneity_effect.sh
./heterogeneity_effect.sh
```

![hetero_effect](https://user-images.githubusercontent.com/42912620/171421936-73ff5429-d266-4e28-84bf-57e20d9e8a1d.png)


## Citation

If you use our code or wish to refer to our results,
please use the following BibTex entry:

```
@article{marfoq2021personalized,
  title={Personalized Federated Learning through Local Memorization},
  author={Marfoq, Othmane and Neglia, Giovanni and Kameni, Laetitia and Vidal, Richard},
  journal={arXiv preprint arXiv:2111.09360},
  year={2021}
}
```
