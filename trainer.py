import os.path
import numpy as np
from optimizer import *

train_params = "params.pkl"

class trainer:
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, min_batch_size=100,
                 optimizer='sgd', optimizer_param={'lr': 0.001},
                 evaluate_sample_num_per_epoch=100,
                 verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = min_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        optimizer_class_dict = {'sgd': sgd, 'adam': adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        self.train_size = self.x_train.shape[0]
        self.iter_per_epoch = max(self.train_size//self.batch_size, 1)
        self.max_iter = int(self.epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose:
            print("train loss: " + str(loss))
        if (self.current_iter % self.iter_per_epoch == 0) or (self.current_iter + 1 == self.max_iter):
            self.current_epoch += 1
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)
            if self.verbose:
                print("epoch: " + str(self.current_epoch) + ", train acc: " + str(train_acc) + ", test acc: " + str(test_acc))
        self.current_iter += 1

    def train(self):
        if os.path.exists(train_params):
            self.network.load_params()
            return
        for i in range(self.max_iter):
            self.train_step()
        test_acc = self.network.accuracy(self.x_test, self.t_test)
        self.network.save_params()
        if self.verbose:
            print("final test accuracy: " + str(test_acc))