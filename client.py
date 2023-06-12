import numpy as np

from datastore import *
from utils.torch_utils import *
from utils.constants import *

from copy import deepcopy


class Client(object):
    r"""
    Implements one clients

    Attributes
    ----------
    learner
    train_iterator
    val_iterator
    test_iterator
    n_train_samples
    n_test_samples
    local_steps
    logger
    counter
    __save_path
    __id

    Methods
    ----------
    __init__
    step
    write_logs
    update_tuned_learners

    """
    def __init__(
            self,
            learner,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            save_path=None,
            id_=None,
            *args,
            **kwargs
    ):
        """

        :param learner:
        :param train_iterator:
        :param val_iterator:
        :param test_iterator:
        :param logger:
        :param local_steps:
        :param save_path:

        """
        self.learner = learner

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator

        self.n_train_samples = len(self.train_iterator.dataset)
        self.n_test_samples = len(self.test_iterator.dataset)

        self.local_steps = local_steps

        self.save_path = save_path

        self.id = -1
        if id_ is not None:
            self.id = id_

        self.counter = 0
        self.logger = logger

    def is_ready(self):
        return self.learner.is_ready

    def step(self, *args, **kwargs):
        self.counter += 1

        self.learner.fit_epochs(
            iterator=self.train_iterator,
            n_epochs=self.local_steps,
        )

    def write_logs(self):

        train_loss, train_acc = self.learner.evaluate_iterator(self.val_iterator)
        test_loss, test_acc = self.learner.evaluate_iterator(self.test_iterator)

        self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        self.logger.add_scalar("Train/Metric", train_acc, self.counter)
        self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        self.logger.add_scalar("Test/Metric", test_acc, self.counter)

        return train_loss, train_acc, test_loss, test_acc

    def save_state(self, path=None):
        """

        :param path: expected to be a `.pt` file

        """
        if path is None:
            if self.save_path is None:
                warnings.warn("client state was not saved", RuntimeWarning)
                return
            else:
                self.learner.save_checkpoint(self.save_path)
                return

        self.learner.save_checkpoint(path)

    def load_state(self, path=None):
        if path is None:
            if self.save_path is None:
                warnings.warn("client state was not loaded", RuntimeWarning)
                return
            else:
                self.learner.load_checkpoint(self.save_path)
                return

        self.learner.load_checkpoint(path)

    def free_memory(self):
        self.learner.free_memory()


class FedRepClient(Client):
    """
    Client used to implement
        "Exploiting Shared Representations for Personalized FederatedLearning"__(https://arxiv.org/pdf/2102.07078.pdf)

    """
    def __init__(
            self,
            learner,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            save_path=None,
            id_=None,
            *args,
            **kwargs
    ):
        super(FedRepClient, self).__init__(
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            save_path=save_path,
            id_=id_,
            *args,
            **kwargs
        )
        self.head = deepcopy(self.learner.get_head())

    def step(self):
        head = self.learner.get_head()
        copy_model(source=self.head, target=head)

        self.learner.freeze_body()

        # train the head
        self.learner.fit_epochs(
            iterator=self.train_iterator,
            n_epochs=LOCAL_HEAD_UPDATES,
        )
        self.head = deepcopy(self.learner.get_head())

        # train the body with fixed head
        self.learner.unfreeze_body()

        head = self.learner.get_head()
        client_updates = \
            self.learner.fit_epochs(
                iterator=self.train_iterator,
                n_epochs=self.local_steps,
                frozen_modules=[head]
            )

        return client_updates


class PerFedAvgClient(Client):
    """
    Client used to implement
        "Personalized Federated Learning: A Meta-Learning Approach"__(https://arxiv.org/pdf/2002.07948.pdf)

    """

    def step(self, *args, **kwargs):
        self.counter += 1

        self.learner.fit_maml_epochs(
            iterator=self.train_iterator,
            n_epochs=self.local_steps,
        )


class AgnosticFLClient(Client):
    def __init__(
            self,
            learner,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally,
            *args,
            **kwargs
    ):
        super(AgnosticFLClient, self).__init__(
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            *args,
            **kwargs
        )
        self.train_loader = iter(self.train_iterator)

    def step(self, *args, **kwargs):
        self.counter += 1

        batch = self.__get_next_batch()
        losses = self.learner.compute_gradients_and_loss(batch)

        return losses

    def __get_next_batch(self):
        try:
            batch = next(self.train_loader)
        except StopIteration:
            self.train_loader = iter(self.train_iterator)
            batch = next(self.train_loader)

        return batch


class FFLClient(Client):
    r"""
    Implements client for q-FedAvg from
     `FAIR RESOURCE ALLOCATION IN FEDERATED LEARNING`__(https://arxiv.org/pdf/1905.10497.pdf)

    """

    def __init__(
            self,
            learner,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally,
            q,
            *args,
            **kwargs
    ):
        """

        :param learner:
        :param train_iterator:
        :param val_iterator:
        :param test_iterator:
        :param logger:
        :param local_steps:
        :param tune_locally:
        :param q: fairness parameter used in q-FFL

        """
        super(FFLClient, self).__init__(
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            *args,
            **kwargs
        )
        self.q = q

    def step(self, lr, *args, **kwargs):

        hs = 0
        for learner in self.learner:
            initial_state_dict = self.learner[0].model.state_dict()
            learner.fit_epochs(iterator=self.train_iterator, n_epochs=self.local_steps)

            client_loss, _ = learner.evaluate_iterator(self.train_iterator)
            client_loss = torch.tensor(client_loss)
            client_loss += 1e-10

            # assign the difference to param.grad for each param in learner.parameters()
            differentiate_learner(
                target=learner,
                reference_state_dict=initial_state_dict,
                coeff=torch.pow(client_loss, self.q) / lr
            )

            hs = self.q * torch.pow(client_loss, self.q-1) * torch.pow(torch.linalg.norm(learner.get_grad_tensor()), 2)
            hs /= torch.pow(torch.pow(client_loss, self.q), 2)
            hs += torch.pow(client_loss, self.q) / lr

        return hs / len(self.learner)


class KNNPerClient(Client):
    """

    Attributes
    ----------
    model:
    features_dimension:
    num_classes:
    train_loader:
    test_loader:
    n_train_samples:
    n_test_samples:
    local_steps:
    logger:
    binary_classification_flag:
    counter:
    capacity: datastore capacity of the client
    strategy: strategy to select samples to keep on the datastore
    rng (numpy.random._generator.Generator):
    datastore (datastore.DataStore):
    datastore_flag (bool):
    features_dimension (int):
    num_classes (int):
    train_features: (n_train_samples x features_dimension)
    test_features: (n_train_samples x features_dimension)
    features_flag (bool):
    model_outputs: (n_test_samples x num_classes)
    model_outputs_flag (bool):
    knn_outputs:
    knn_outputs_flag (bool)
    interpolate_logits (bool): if selected logits are interpolated instead of probabilities

    Methods
    -------
    __init__

    build

    compute_features_and_model_outputs

    build_datastore

    gather_knn_outputs

    evaluate

    clear_datastore

    """

    def __init__(
            self,
            learner,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            k,
            interpolate_logits,
            features_dimension,
            num_classes,
            capacity,
            strategy,
            rng,
            *args,
            **kwargs
    ):
        """

        :param learner:
        :param train_iterator:
        :param val_iterator:
        :param test_iterator:
        :param logger:
        :param local_steps:
        :param k:
        :param features_dimension:
        :param num_classes:
        :param capacity:
        :param strategy:
        :param rng:

        """
        super(KNNPerClient, self).__init__(
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            *args,
            **kwargs
        )

        self.k = k
        self.interpolate_logits = interpolate_logits

        self.model = self.learner.model
        self.features_dimension = features_dimension
        self.num_classes = num_classes

        self.train_iterator = train_iterator
        self.test_iterator = test_iterator

        self.n_train_samples = len(train_iterator.dataset)
        self.n_test_samples = len(test_iterator.dataset)

        self.capacity = capacity
        self.strategy = strategy
        self.rng = rng
        self.device = self.learner.device

        self.model = self.model.to(self.device)
        self.model.eval()

        self.datastore = DataStore(self.capacity, self.strategy, self.features_dimension, self.rng)
        self.datastore_flag = False

        self.train_features = np.zeros(shape=(self.n_train_samples, self.features_dimension), dtype=np.float32)
        self.train_labels = np.zeros(shape=self.n_train_samples, dtype=np.float32)
        self.test_features = np.zeros(shape=(self.n_test_samples, self.features_dimension), dtype=np.float32)
        self.test_labels = np.zeros(shape=self.n_test_samples, dtype=np.float32)
        self.features_flag = False

        self.train_model_outputs = np.zeros(shape=(self.n_train_samples, self.num_classes), dtype=np.float32)
        self.train_model_outputs_flag = False

        self.test_model_outputs = np.zeros(shape=(self.n_test_samples, self.num_classes), dtype=np.float32)
        self.test_model_outputs_flag = False

        self.train_knn_outputs = np.zeros(shape=(self.n_train_samples, self.num_classes), dtype=np.float32)
        self.train_knn_outputs_flag = False

        self.test_knn_outputs = np.zeros(shape=(self.n_test_samples, self.num_classes), dtype=np.float32)
        self.test_knn_outputs_flag = False

    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, k):
        self.__k = int(k)

    @property
    def capacity(self):
        return self.__capacity

    @capacity.setter
    def capacity(self, capacity):
        if 0 <= capacity <= 1 and isinstance(capacity, float):
            capacity = int(capacity * self.n_train_samples)
        else:
            capacity = int(capacity)

        if capacity < 0:
            capacity = self.n_train_samples

        self.__capacity = capacity

    def step(self):
        pass

    def compute_features_and_model_outputs(self):
        """
        extract features from `train_iterator` and `test_iterator` .
        and computes the predictions of the base model (i.e., `self.model`) over `test_iterator`.

        """
        self.features_flag = True
        self.train_model_outputs_flag = True
        self.test_model_outputs_flag = True

        self.train_features, self.train_model_outputs, self.train_labels = \
            self.learner.compute_embeddings_and_outputs(
                iterator=self.train_iterator,
                embedding_dim=self.features_dimension,
                n_classes=self.num_classes,
                apply_softmax=(not self.interpolate_logits)
            )

        self.test_features, self.test_model_outputs, self.test_labels = \
            self.learner.compute_embeddings_and_outputs(
                iterator=self.test_iterator,
                embedding_dim=self.features_dimension,
                n_classes=self.num_classes,
                apply_softmax=(not self.interpolate_logits)
            )

    def build_datastore(self):
        assert self.features_flag, "Features should be computed before building datastore!"
        self.datastore_flag = True

        self.datastore.build(self.train_features, self.train_labels)

    def gather_knn_outputs(self, mode="test", scale=1.):
        """
        computes the k-NN predictions

        :param mode: possible are "train" and "test", default is "test"
        :param scale: scale of the gaussian kernel, default is 1.0
        """
        if self.capacity <= 0:
            warnings.warn("trying to gather knn outputs with empty datastore", RuntimeWarning)
            return

        assert self.features_flag, "Features should be computed before building datastore!"
        assert self.datastore_flag, "Should build datastore before computing knn outputs!"

        if mode == "train":
            features = self.train_features
            self.train_knn_outputs_flag = True
        else:
            features = self.test_features
            self.test_knn_outputs_flag = True

        distances, indices = self.datastore.index.search(features, self.k)
        similarities = np.exp(-distances / (self.features_dimension * scale))
        neighbors_labels = self.datastore.labels[indices]

        masks = np.zeros(((self.num_classes,) + similarities.shape))
        for class_id in range(self.num_classes):
            masks[class_id] = neighbors_labels == class_id

        outputs = (similarities * masks).sum(axis=2) / similarities.sum(axis=1)

        if mode == "train":
            self.train_knn_outputs = outputs.T
        else:
            self.test_knn_outputs = outputs.T

    def evaluate(self, weight, mode="test"):
        """
        evaluates the client for a given weight parameter

        :param weight: float in [0, 1]
        :param mode: possible are "train" and "test", default is "test"

        :return:
            accuracy score

        """
        if mode == "train":
            flag = self.train_knn_outputs_flag
            knn_outputs = self.train_knn_outputs
            model_outputs = self.train_model_outputs
            labels = self.train_labels

        else:
            flag = self.test_knn_outputs_flag
            knn_outputs = self.test_knn_outputs
            model_outputs = self.test_model_outputs
            labels = self.test_labels

        if flag:
            outputs = weight * knn_outputs + (1 - weight) * model_outputs
        else:
            warnings.warn("evaluation is done only with model outputs, datastore is empty", RuntimeWarning)
            outputs = model_outputs

        predictions = np.argmax(outputs, axis=1)

        correct = (labels == predictions).sum()
        total = len(labels)

        if total == 0:
            acc = 1
        else:
            acc = correct / total

        return acc

    def write_logs(self):
        train_loss, train_acc, test_loss, test_acc = super().write_logs()

        if self.counter % 10 == 0:
            self.capacity = capacity
            self.clear_datastore()
            self.build_datastore()
            self.gather_knn_outputs()

            # todo: weight and capacity, take average?
            weights = np.arange(0.1, 1. + 1e-6, 0.1)
            for w in weights:
                personal_acc = self.evaluate(w) * self.n_test_samples
                self.logger.add_scalar("Personal/Metric{}".format(w), personal_acc, self.counter)

        return train_loss, train_acc, test_loss, test_acc

    def clear_datastore(self):
        """
        clears `datastore`

        """
        self.datastore.clear()
        self.datastore.capacity = self.capacity

        self.datastore_flag = False
        self.train_knn_outputs_flag = False
        self.test_knn_outputs_flag = False
