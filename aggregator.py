import os
import time
import random

from copy import deepcopy

from abc import ABC, abstractmethod

from utils.torch_utils import *

from learners.learners_ensemble import *

from tqdm import tqdm

import numpy as np
import numpy.linalg as LA

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering


class Aggregator(ABC):
    r""" Base class for Aggregator. `Aggregator` dictates communications between clients

    Attributes
    ----------
    clients

    test_clients

    n_clients:

    n_test_clients

    clients_weights:

    global_learner: List[Learner]

    model_dim: dimension if the used model

    device:

    sampling_rate: proportion of clients used at each round; default is `1.`

    sample_with_replacement: is True, client are sampled with replacement; default is False

    n_clients_per_round:

    sampled_clients:

    c_round: index of the current communication round

    global_train_logger:

    global_test_logger:

    log_freq:

    verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

    rng: random number generator

    Methods
    ----------
    __init__

    mix

    update_clients

    update_test_clients

    write_logs

    save_state

    load_state

    """
    def __init__(
            self,
            clients,
            global_learner,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None
    ):

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        self.global_learner = global_learner
        self.model_dim = self.global_learner.model_dim
        self.device = self.global_learner.device

        if test_clients is None:
            test_clients = []

        self.clients = clients
        self.test_clients = test_clients

        self.n_clients = len(clients)
        self.n_test_clients = len(test_clients)

        self.clients_weights =\
            torch.tensor(
                [client.n_train_samples for client in self.clients],
                dtype=torch.float32,
                device=self.device
            )

        self.clients_weights = self.clients_weights / self.clients_weights.sum()

        self.sampling_rate = sampling_rate
        self.sample_with_replacement = sample_with_replacement
        self.n_clients_per_round = max(1, int(self.sampling_rate * self.n_clients))
        self.sampled_clients_ids = list()
        self.sampled_clients = list()

        self.global_train_logger = global_train_logger
        self.global_test_logger = global_test_logger
        self.log_freq = log_freq
        self.verbose = verbose

        self.c_round = 0

    @abstractmethod
    def mix(self):
        pass

    @abstractmethod
    def toggle_client(self, client_id, mode):
        """
        toggle client at index `client_id`, if `mode=="train"`, `client_id` is selected in `self.clients`,
        otherwise it is selected in `self.test_clients`.

        :param client_id: (int)
        :param mode: possible are "train" and "test"

        """
        pass

    def toggle_clients(self):
        for client_id in range(self.n_clients):
            self.toggle_client(client_id, mode="train")

    def toggle_sampled_clients(self):
        for client_id in self.sampled_clients_ids:
            self.toggle_client(client_id, mode="train")

    def toggle_test_clients(self):
        for client_id in range(self.n_test_clients):
            self.toggle_client(client_id, mode="test")

    def write_logs(self):
        self.toggle_test_clients()

        for global_logger, clients, mode in [
            (self.global_train_logger, self.clients, "train"),
            (self.global_test_logger, self.test_clients, "test")
        ]:
            if len(clients) == 0:
                continue

            global_train_loss = 0.
            global_train_acc = 0.
            global_test_loss = 0.
            global_test_acc = 0.

            total_n_samples = 0
            total_n_test_samples = 0

            for client_id, client in enumerate(clients):

                train_loss, train_acc, test_loss, test_acc = client.write_logs()

                if self.verbose > 1:
                    print("*" * 30)
                    print(f"Client {client_id}..")
                    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.3f}%|", end="")
                    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.3f}% |")

                global_train_loss += train_loss * client.n_train_samples
                global_train_acc += train_acc * client.n_train_samples
                global_test_loss += test_loss * client.n_test_samples
                global_test_acc += test_acc * client.n_test_samples

                total_n_samples += client.n_train_samples
                total_n_test_samples += client.n_test_samples

            global_train_loss /= total_n_samples
            global_test_loss /= total_n_test_samples
            global_train_acc /= total_n_samples
            global_test_acc /= total_n_test_samples

            if self.verbose > 0:
                print("+" * 30)
                print("Global..")
                print(f"Train Loss: {global_train_loss:.3f} | Train Acc: {global_train_acc * 100:.3f}% |", end="")
                print(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_acc * 100:.3f}% |")
                print("+" * 50)

            global_logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
            global_logger.add_scalar("Train/Metric", global_train_acc, self.c_round)
            global_logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
            global_logger.add_scalar("Test/Metric", global_test_acc, self.c_round)

            # Additional logging
            global_logger.add_scalar("ClientsSampleSize", len(self.sampled_clients), self.c_round)

        if self.verbose > 0:
            print("#" * 80)

    def evaluate(self):
        """
        evaluate the aggregator, returns the performance of every client in the aggregator

        :return
            clients_results: (np.array of size (self.n_clients, 2, 2))
                number of correct predictions and total number of samples per client both for train part and test part
            test_client_results: (np.array of size (self.n_test_clients))
                number of correct predictions and total number of samples per client both for train part and test part

        """

        clients_results = []
        test_client_results = []

        for results, clients, mode in [
            (clients_results, self.clients, "train"),
            (test_client_results, self.test_clients, "test")
        ]:
            if len(clients) == 0:
                continue

            print(f"evaluate {mode} clients..")
            for client_id, client in enumerate(tqdm(clients)):
                if not client.is_ready():
                    self.toggle_client(client_id, mode=mode)

                _, train_acc, _, test_acc = client.write_logs()

                results.append([
                    [train_acc * client.n_train_samples, client.n_train_samples],
                    [test_acc * client.n_test_samples, client.n_test_samples]
                ])

                client.free_memory()

        return np.array(clients_results, dtype=np.uint16), np.array(test_client_results, dtype=np.uint16)

    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of  `global_learner` as `.pt` file,
         and the state of each client in `self.clients`.

        :param dir_path:

        """
        save_path = os.path.join(dir_path, "global.pt")
        torch.save(self.global_learner.model.state_dict(), save_path)

        for client_id, client in enumerate(self.clients):
            self.toggle_client(client_id, mode="train")
            client.save_state()
            client.free_memory()

    def load_state(self, dir_path):
        """
        load the state of the aggregator

        :param dir_path:

        """
        chkpts_path = os.path.join(dir_path, f"global.pt")
        self.global_learner.model.load_state_dict(torch.load(chkpts_path))
        for client_id, client in self.clients:
            self.toggle_client(client_id, mode="train")
            client.load_state()
            client.free_memory()

    def sample_clients(self):
        """
        sample a list of clients without repetition

        """
        if self.sample_with_replacement:
            self.sampled_clients_ids = \
                self.rng.choices(
                    population=range(self.n_clients),
                    weights=self.clients_weights,
                    k=self.n_clients_per_round,
                )
        else:
            self.sampled_clients_ids = self.rng.sample(range(self.n_clients), k=self.n_clients_per_round)

        self.sampled_clients = [self.clients[id_] for id_ in self.sampled_clients_ids]


class CentralizedAggregator(Aggregator):
    r""" Standard Centralized Aggregator.
     All clients get fully synchronized with the average client.

    """
    def mix(self):
        self.sample_clients()
        self.toggle_sampled_clients()

        for client in self.sampled_clients:
            client.step()

        learners = [client.learner for client in self.sampled_clients]

        average_learners(
            learners=learners,
            target_learner=self.global_learner,
            weights=self.clients_weights[self.sampled_clients_ids] / self.sampling_rate,
            average_params=True,
            average_gradients=False
        )

        for client in self.clients:
            copy_model(client.learner.model, self.global_learner.model)

        self.c_round += 1

    def toggle_client(self, client_id, mode):
        if mode == "train":
            client = self.clients[client_id]
        else:
            client = self.test_clients[client_id]

        if client.is_ready():
            copy_model(client.learner.model, self.global_learner.model)
        else:
            client.learner = deepcopy(self.global_learner)

        if callable(getattr(client.learner.optimizer, "set_initial_params", None)):
            client.learner.optimizer.set_initial_params(
                self.global_learner.model.parameters()
            )

    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of  `global_learner` as `.pt` file,
         and the state of each client in `self.clients`.

        :param dir_path:

        """
        save_path = os.path.join(dir_path, f"global_{self.c_round}.pt")
        torch.save(self.global_learner.model.state_dict(), save_path)

    def load_state(self, dir_path):
        """
        load the state of the aggregator

        :param dir_path:

        """
        chkpts_path = os.path.join(dir_path, f"global_{self.c_round}.pt")
        self.global_learner.model.load_state_dict(torch.load(chkpts_path))


class NoCommunicationAggregator(Aggregator):
    r"""Clients do not communicate. Each client work locally

    """
    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step()

        self.c_round += 1

    def toggle_client(self, client_id, mode):
        pass


class PersonalizedAggregator(CentralizedAggregator):
    r"""Implements Personalized Aggregator. Consider the same objective function as in
    "Federated Learning of a Mixture of Global and Local Models"__(https://arxiv.org/pdf/2002.05516.pdf), i.e.,

    .. math::
        min_x \sum_{i=1}^{n}F_{i}(x_i) + \lambda \sum_{i=1}^{n}|| x_{i} - \bar{x}||^ {2}

    In our implementation, it should be combined with `prox_sgd` optimizer.

    """
    def toggle_client(self, client_id, mode):
        if mode == "train":
            client = self.clients[client_id]
        else:
            client = self.test_clients[client_id]

        if callable(getattr(client.learner.optimizer, "set_initial_params", None)):
            client.learner.optimizer.set_initial_params(self.global_learner.model.parameters())


class ClusteredAggregator(Aggregator):
    """
    Implements
     `Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints`.

     Follows implementation from https://github.com/felisat/clustered-federated-learning

    """
    def __init__(
            self,
            clients,
            global_learner,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            test_clients=None,
            verbose=0,
            tol_1=0.4,
            tol_2=1.6,
            seed=None
    ):

        super(ClusteredAggregator, self).__init__(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )

        assert self.sampling_rate == 1.0, f"`sampling_rate` is {sampling_rate}, should be {1.0}," \
                                          f" ClusteredAggregator only supports full clients participation."

        self.tol_1 = tol_1
        self.tol_2 = tol_2

        self.global_learners = [self.global_learner]
        self.clusters_indices = [np.arange(len(clients)).astype("int")]
        self.n_clusters = 1

    def mix(self):
        clients_updates = np.zeros((self.n_clients, self.model_dim))

        for client_id, client in enumerate(self.clients):
            clients_updates[client_id] = client.step()

        similarities = pairwise_distances(clients_updates, metric="cosine")
        similarities = similarities.mean(axis=0)

        new_cluster_indices = []
        for indices in self.clusters_indices:
            max_update_norm = LA.norm(clients_updates[indices], axis=1).max()
            mean_update_norm = LA.norm(np.mean(clients_updates[indices], axis=0))

            if mean_update_norm < self.tol_1 and max_update_norm > self.tol_2 and len(indices) > 2:
                clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete")
                clustering.fit(similarities[indices][:, indices])
                cluster_1 = np.argwhere(clustering.labels_ == 0).flatten()
                cluster_2 = np.argwhere(clustering.labels_ == 1).flatten()
                new_cluster_indices += [cluster_1, cluster_2]
            else:
                new_cluster_indices += [indices]

        self.clusters_indices = new_cluster_indices

        self.n_clusters = len(self.clusters_indices)

        self.global_learners = [deepcopy(self.clients[0].learner) for _ in range(self.n_clusters)]

        for cluster_id, indices in enumerate(self.clusters_indices):
            cluster_clients = [self.clients[i] for i in indices]

            average_learners(
                learners=[client.learner for client in cluster_clients],
                target_learner=self.global_learners[cluster_id],
                weights=self.clients_weights[indices] / self.clients_weights[indices].sum()
            )

        self.toggle_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def toggle_client(self, client_id, mode):
        pass


class LoopLessLocalSGDAggregator(PersonalizedAggregator):
    """
    Implements L2SGD introduced in
    'Federated Learning of a Mixture of Global and Local Models'__. (https://arxiv.org/pdf/2002.05516.pdf)


    """

    def __init__(
            self,
            clients,
            global_learner,
            log_freq,
            global_train_logger,
            global_test_logger,
            communication_probability,
            penalty_parameter,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None
    ):
        super(LoopLessLocalSGDAggregator, self).__init__(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )

        self.communication_probability = communication_probability
        self.penalty_parameter = penalty_parameter

    @property
    def communication_probability(self):
        return self.__communication_probability

    @communication_probability.setter
    def communication_probability(self, communication_probability):
        self.__communication_probability = communication_probability

    def mix(self):
        communication_flag = self.np_rng.binomial(1, self.communication_probability, 1)

        if communication_flag:
            for learner_id, learner in enumerate(self.global_learner):
                learners = [client.learners_ensemble[learner_id] for client in self.clients]
                average_learners(learners, learner, weights=self.clients_weights)

                partial_average(
                    learners,
                    average_learner=learner,
                    alpha=self.penalty_parameter/self.communication_probability
                )

                self.toggle_clients()

                self.c_round += 1

                if self.c_round % self.log_freq == 0:
                    self.write_logs()

        else:
            self.sample_clients()
            for client in self.sampled_clients:
                client.step(single_batch_flag=True)


class AgnosticAggregator(CentralizedAggregator):
    """
    Implements
     `Agnostic Federated Learning`__(https://arxiv.org/pdf/1902.00146.pdf).

    """
    def __init__(
            self,
            clients,
            global_learner,
            log_freq,
            global_train_logger,
            global_test_logger,
            lr_lambda,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None
    ):
        super(AgnosticAggregator, self).__init__(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )

        self.lr_lambda = lr_lambda

    def mix(self):
        self.sample_clients()

        clients_losses = []
        for client in self.sampled_clients:
            client_losses = client.step()
            clients_losses.append(client_losses)

        clients_losses = torch.tensor(clients_losses)

        learners = [client.learner for client in self.clients]

        average_learners(
            learners=learners,
            target_learner=self.global_learner,
            weights=self.clients_weights,
            average_gradients=True
        )

        # update parameters
        self.global_learner.optimizer_step()

        # update clients weights
        self.clients_weights += self.lr_lambda * clients_losses.mean(dim=1)
        self.clients_weights = simplex_projection(self.clients_weights)

        # assign the updated model to all clients
        self.toggle_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()


class FFLAggregator(CentralizedAggregator):
    """
    Implements q-FedAvg from
     `FAIR RESOURCE ALLOCATION IN FEDERATED LEARNING`__(https://arxiv.org/pdf/1905.10497.pdf)

    """
    def __init__(
            self,
            clients,
            global_learner,
            log_freq,
            global_train_logger,
            global_test_logger,
            lr,
            q=1,
            sampling_rate=1.,
            sample_with_replacement=True,
            test_clients=None,
            verbose=0,
            seed=None
    ):
        super(FFLAggregator, self).__init__(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed)

        self.q = q
        self.lr = lr
        assert self.sample_with_replacement, 'FFLAggregator only support sample with replacement'

    def mix(self):
        self.sample_clients()

        hs = 0
        for client in self.sampled_clients:
            hs += client.step(lr=self.lr)

        hs /= (self.lr * len(self.sampled_clients))  # take account for the lr used inside optimizer

        learners = [client.learner for client in self.sampled_clients]
        average_learners(
            learners=learners,
            target_learner=self.global_learner,
            weights=hs*torch.ones(len(learners)),
            average_params=False,
            average_gradients=True
        )

        # update parameters
        self.global_learner.optimizer_step()

        # assign the updated model to all clients
        self.toggle_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()


class APFLAggregator(Aggregator):
    """

    """
    def __init__(
            self,
            clients,
            global_learner,
            log_freq,
            global_train_logger,
            global_test_logger,
            alpha,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None
    ):
        super(APFLAggregator, self).__init__(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )
        self.global_learners_ensemble = LearnersEnsemble(
            learners=[global_learner, deepcopy(global_learner)],
            learners_weights=torch.tensor([alpha, 1-alpha], device=self.device)
        )

        for client in self.clients:
            client.learners_ensemble = deepcopy(self.global_learners_ensemble)

        self.alpha = alpha

    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            for _ in range(client.local_steps):
                client.step(single_batch_flag=True)

                partial_average(
                    learners=[client.learners_ensemble[1]],
                    average_learner=client.learners_ensemble[0],
                    alpha=self.alpha
                )

        average_learners(
            learners=[client.learners_ensemble[0] for client in self.clients],
            target_learner=self.global_learners_ensemble[0],
            weights=self.clients_weights
        )

        # assign the updated model to all clients
        self.toggle_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def toggle_client(self, client_id, mode):
        if mode == "train":
            client = self.clients[client_id]
        else:
            client = self.test_clients[client_id]

        copy_model(client.learners_ensemble[0].model, self.global_learners_ensemble[0].model)

        if callable(getattr(client.learners_ensemble[0].optimizer, "set_initial_params", None)):
            client.learners_ensemble[0].optimizer.set_initial_params(
                self.global_learners_ensemble[0].model.parameters()
            )
