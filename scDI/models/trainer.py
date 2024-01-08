from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset

# from .utils import *

from scDI.models.utils import CreateDataset, train_test_split, HSIC, mean_nll, eval_train, eval_test, trades_loss, \
    adjust_bias, IRM_CustomDatasetFromAdata, penalty
from scDI.models.utils import EarlyStopping


class Trainer:
    def __init__(self,
                 model,
                 adata,
                 label_key=None,
                 condition_key="condition",
                 seed=2,
                 print_every=10,
                 learning_rate=0.001,
                 validation_itr=5,
                 train_frac=0.85,
                 s_x=1,
                 s_y=2
                 ):

        self.model = model
        self.adata = adata
        self.label_key = label_key
        self.condition_key = condition_key
        self.seed = seed
        self.print_loss = print_every
        self.lr = learning_rate
        self.val_check = validation_itr
        self.train_frac = train_frac
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.device = self.device
        self.logs = defaultdict(list)
        self.model.to(self.device)
        self.s_x = s_x
        self.s_y = s_y

    def make_dataset(self):
        """
        generating dataset for each of the NLL, HCIC and TRADES
        :return:
        """
        self.train_adata, self.validation_adata = train_test_split(self.adata, self.train_frac)
        data_set_train = CreateDataset(self.train_adata, self.label_key)
        self.model.label_encoder = data_set_train.get_label_encoder()
        data_set_valid = CreateDataset(self.validation_adata, self.label_key, le=self.model.label_encoder)

        return data_set_train, data_set_valid

    def irm_make_dataset(self):
        """
        generating dataset for the IRM training
        :return:
        """
        self.train_data, self.validation_data = train_test_split(self.adata, self.train_frac)
        data_set_train = IRM_CustomDatasetFromAdata(self.train_data,
                                                    label_key=self.label_key,
                                                    condition_key=self.condition_key)

        self.model.condition_encoder = data_set_train.get_condition_ecnoder()
        self.model.label_encoder = data_set_train.get_label_encoder()

        data_set_valid = IRM_CustomDatasetFromAdata(self.validation_data,
                                                    label_key=self.label_key,
                                                    condition_key=self.condition_key,
                                                    le_ct=self.model.label_encoder,
                                                    le_cnd=self.model.condition_encoder)
        return data_set_train, data_set_valid

    def on_epoch_begin(self):
        if self.n_epochs_kl_warmup is not None:
            self.kl_weight = min(1, self.epoch / self.n_epochs_kl_warmup)
        else:
            self.kl_weight = 1.0

    def train_HSIC(self, n_epochs=200, batch_size=32, early_patience=20, weight_decay=0.01):
        """
                    Trains a MLP model `n_epochs` times with given `batch_size` with HSIC loss function. This function is using `early stopping`
                    technique to prevent overfitting.
                    # Parameters
                        n_epochs: int
                            number of epochs to iterate and optimize network weights
                        batch_size: int
                            number of samples to be used in each batch for network weights optimization
                        early_patience: int
                            number of consecutive epochs in which network loss is not going lower.
                            After this limit, the network will stop training.
                        weight_decay: float
                            l2 norm for reqularization
                    # Returns
                        Nothing will be returned

        """
        es = EarlyStopping(patience=early_patience)
        dataset_train, dataset_valid = self.make_dataset()
        data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                        shuffle=True,
                                                        batch_size=batch_size,
                                                        drop_last=True)
        data_loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                        shuffle=True,
                                                        batch_size=batch_size,
                                                        drop_last=True)
        num_classes = self.model.label_encoder.classes_.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.model.train()
        self.logs = defaultdict(list)
        for epoch in range(n_epochs):
            train_loss = 0
            for iteration, (x, y) in enumerate(data_loader_train):
                y = (y == torch.arange(num_classes).reshape(1, num_classes)).float()
                # c = euclidean_distances(x, squared=True)
                # m = x.shape[0]
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                yhat = self.model(x)

                loss = HSIC(x, yhat - y, self.s_x, self.s_y)

                loss.backward()
                optimizer.step()
                train_loss += loss

            if iteration % self.print_loss == 0 or iteration == len(data_loader_train) - 1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss: {:9.4f}".format(
                    epoch, n_epochs, iteration, len(data_loader_train) - 1, loss.item()))

            self.logs['loss_train'].append(train_loss / iteration)
            valid_acc, valid_loss = self.validate_hsic(data_loader_valid)
            self.logs['acc_loss__valid'].append(valid_acc)
            if es.step(valid_loss.cpu().numpy()):
                print("Training stoped with early stopping")
                break

            if epoch % self.val_check == 0 and epoch != 0:
                print("Epoch {:02d}, Loss_valid: {:9.4f}, acc_valid: {:9.4f}".format(
                    epoch, valid_loss, valid_acc))
        self.accuracy = valid_acc
        adjust_bias(data_loader_train)

    def train_nll(self, n_epochs=200, batch_size=512, early_patience=20, weight_decay=0.01):
        """
                    Trains a MLP model `n_epochs` times with given `batch_size` with NLL loss function. This function is using `early stopping`
                    technique to prevent overfitting.
                    # Parameters
                        n_epochs: int
                            number of epochs to iterate and optimize network weights
                        batch_size: int
                            number of samples to be used in each batch for network weights optimization
                        early_patience: int
                            number of consecutive epochs in which network loss is not going lower.
                            After this limit, the network will stop training.
                        weight_decay: float
                            l2 norm for reqularization
                    # Returns
                        Nothing will be returned

        """
        es = EarlyStopping(patience=early_patience)
        dataset_train, dataset_valid = self.make_dataset()
        data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        drop_last=True)
        data_loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        drop_last=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.model.train()
        self.logs = defaultdict(list)
        for epoch in range(n_epochs):
            train_loss = 0
            for iteration, (x, y) in enumerate(data_loader_train):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                yhat = self.model(x)
                loss = mean_nll(yhat, y.reshape(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss

            if iteration % self.print_loss == 0 or iteration == len(data_loader_train) - 1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss: {:9.4f}".format(
                    epoch, n_epochs, iteration, len(data_loader_train) - 1, loss.item()))

            self.logs['loss_train'].append(train_loss / iteration)
            valid_acc, valid_loss = self.validate_nll(data_loader_valid)
            self.logs['acc_loss__valid'].append(valid_acc)
            if es.step(valid_loss.cpu().numpy()):
                print("Training stoped with early stopping")
                break

            if epoch % self.val_check == 0 and epoch != 0:
                print("Epoch {:02d}, Loss_valid: {:9.4f}, acc_valid: {:9.4f}".format(
                    epoch, valid_loss, valid_acc))
        self.accuracy = valid_acc

    def train_trades(self,
                     n_epochs=200,
                     batch_size=512,
                     weight_decay=0.01,
                     step_size=0.01,
                     epsilon=1.5,
                     num_steps=40,
                     beta=3,
                     distance='l_2',
                     print_eval=False):

        """
        Trains a MLP model `n_epochs` times with given `batch_size` with TRADES loss function. This function is using
            `early stopping` technique to prevent overfitting.
        :param n_epochs: number of epochs
        :param batch_size: batch size
        :param weight_decay: l2 normalizer
        :param step_size: step size for the PGD attack
        :param epsilon: epsilon baoundary for the PGD attack
        :param num_steps: number of steps for attack on a sample
        :param beta: a number between 1 to 10. if beta is near 1 we have the higher accuracy and lower robustness
            and vice versa.
        :param distance: type of distance measure `l_inf` or `l_2`
        :param print_eval: period of printing accuracy and loss
        :return: 
        """

        dataset_train, dataset_valid = self.make_dataset()
        data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        drop_last=True)
        data_loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        drop_last=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.model.train()
        self.logs = defaultdict(list)
        for epoch in range(n_epochs):
            train_loss = 0
            for iteration, (x, y) in enumerate(data_loader_train):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = trades_loss(model=self.model,
                                   x_natural=x,
                                   y=y,
                                   optimizer=optimizer,
                                   step_size=step_size,
                                   epsilon=epsilon,
                                   perturb_steps=num_steps,
                                   beta=beta,
                                   distance=distance)
                loss.backward()
                optimizer.step()
                train_loss += loss

            if print_eval:
                eval_train(self.model, self.device, data_loader_train)
                eval_test(self.model, self.device, data_loader_valid)


    def train_IRM(self,
                  n_epochs=200,
                  batch_size=32,
                  early_patience=30,
                  weight_decay=0):
        """
                    Trains a MLP model `n_epochs` times with given `batch_size` with HSIC loss function. This function
                        is using `early stopping` technique to prevent overfitting.
                    # Parameters
                        n_epochs: int
                            number of epochs to iterate and optimize network weights
                        batch_size: int
                            number of samples to be used in each batch for network weights optimization
                        early_patience: int
                            number of consecutive epochs in which network loss is not going lower.
                            After this limit, the network will stop training.
                        weight_decay: float
                            l2 norm for reqularization
                    # Returns
                        Nothing will be returned

        """


        es = EarlyStopping(patience=early_patience)
        dataset_train, dataset_valid = self.irm_make_dataset()
        data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        drop_last=True)
        data_loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        drop_last=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.logs = defaultdict(list)
        self.model.train()
        self.dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(self.device)
        for epoch in range(n_epochs):
            train_loss = 0
            nll_loss_train = 0
            penalty_train = 0
            for iteration, (x, y, y_ct) in enumerate(data_loader_train):
                x, y, y_ct = x.to(self.device), y.to(self.device), y_ct.to(self.device)
                nll_loss = 0
                penalty_itr = 0
                acc = 0
                for y_env in torch.unique(y):
                    x_env = x[(y == y_env).nonzero()[:, 0]]
                    y_ct_env = y_ct[(y == y_env).nonzero()[:, 0]]
                    pred_env = self.model(x_env)

                    nll_loss_env = mean_nll(pred_env, y_ct_env.reshape(-1))
                    #                     acc_env = mean_accuracy(pred_env, y_ct_env)
                    penalty_env = penalty(pred_env, y_ct_env)

                    nll_loss += nll_loss_env
                    penalty_itr += penalty_env
                #                     acc += acc_env

                if self.model.linear0_anneal1 == 0:
                    penalty_weight = self.model.reg_weight * (epoch / n_epochs)
                elif self.model.linear0_anneal1 == 1:
                    if epoch >= self.model.penalty_anneal_iters:
                        penalty_weight = self.model.reg_weight
                    else:
                        penalty_weight = 0
                else:
                    penalty_weight = self.model.reg_weight

                loss = nll_loss + penalty_weight * penalty_itr
                # loss /= penalty_weight
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss
                nll_loss_train += nll_loss
                penalty_train += penalty_weight * penalty_itr

            if iteration % self.print_loss == 0 or iteration == len(data_loader_train) - 1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss: {:9.4f},"
                      " nll_loss: {:9.4f}, IRM_loss:  {:9.4f}, acc: {:9.4f}, penalty_weight: {:.4f}".format(
                    epoch, n_epochs, iteration, len(data_loader_train) - 1,
                    loss.item(), nll_loss, penalty_weight * penalty_itr.item(), acc / len(torch.unique(y)),
                    penalty_weight))
            self.logs['loss_train'].append(train_loss / iteration)
            self.logs["nll_loss_train"].append(nll_loss / iteration)
            self.logs["IRM_loss"].append(penalty_itr / iteration)
            valid_acc, valid_loss = self.validate_irm(data_loader_valid)
            self.logs['nll_loss__valid'].append(valid_loss)
            self.logs['acc_loss__valid'].append(valid_acc)
            if es.step(valid_loss.cpu().numpy()):
                print("Training stoped with early stopping")
                break
            if epoch % self.val_check == 0 and epoch != 0:
                print("Epoch {:02d}, Loss_valid: {:9.4f}, acc_valid: {:9.4f}".format(
                    epoch, valid_loss, valid_acc))

        self.accuracy = valid_acc

    def optuna_HSIC(self, trial, n_epochs=200, batch_size=512, early_patience=20, weight_decay=0.01):

        """
        finding best model with HSIC loss using optuna
        :param trial:
        :param n_epochs: number of epochs
        :param batch_size: batch_size
        :param early_patience:  early patience
        :param weight_decay: l2 reqularizer
        :return:
        """
        es = EarlyStopping(patience=early_patience)
        dataset_train, dataset_valid = self.make_dataset()
        data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        drop_last=True)
        data_loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        drop_last=True)
        num_classes = self.model.label_encoder.classes_.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

        self.model.train()
        self.logs = defaultdict(list)
        for epoch in range(n_epochs):
            train_loss = 0
            for iteration, (x, y) in enumerate(data_loader_train):
                y = (y == torch.arange(num_classes).reshape(1, num_classes)).float()
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                yhat = self.model(x)
                loss = HSIC(x, yhat - y, self.s_x, self.s_y)
                loss.backward()
                optimizer.step()
                train_loss += loss

            if iteration % self.print_loss == 0 or iteration == len(data_loader_train) - 1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss: {:9.4f}".format(
                    epoch, n_epochs, iteration, len(data_loader_train) - 1, loss.item()))

            self.logs['loss_train'].append(train_loss / iteration)
            valid_acc, valid_loss = self.validate_hsic(data_loader_valid)
            self.logs['acc_loss__valid'].append(valid_acc)
            if es.step(valid_loss.cpu().numpy()):
                print("Training stoped with early stopping")
                break

            if epoch % self.val_check == 0 and epoch != 0:
                print("Epoch {:02d}, Loss_valid: {:9.4f}, acc_valid: {:9.4f}".format(
                    epoch, valid_loss, valid_acc))
            trial.report(valid_acc, epoch)

        self.accuracy = valid_acc

    def optuna_nll(self, trial, n_epochs=200, batch_size=512, early_patience=20, weight_decay=0.01):
        """
        finding best model with nll loss using optuna
        :param trial:
        :param n_epochs: number of epochs
        :param batch_size: batch_size
        :param early_patience:  early patience
        :param weight_decay: l2 reqularizer
        :return:
        """

        es = EarlyStopping(patience=early_patience)
        dataset_train, dataset_valid = self.make_dataset()
        data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        drop_last=True)
        data_loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        drop_last=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

        self.model.train()
        self.logs = defaultdict(list)
        for epoch in range(n_epochs):
            train_loss = 0
            for iteration, (x, y) in enumerate(data_loader_train):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                yhat = self.model(x)
                loss = mean_nll(yhat, y.reshape(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss

            if iteration % self.print_loss == 0 or iteration == len(data_loader_train) - 1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss: {:9.4f}".format(
                    epoch, n_epochs, iteration, len(data_loader_train) - 1, loss.item()))

            self.logs['loss_train'].append(train_loss / iteration)
            valid_acc, valid_loss = self.validate_nll(data_loader_valid)
            self.logs['acc_loss__valid'].append(valid_acc)
            if es.step(valid_loss.cpu().numpy()):
                print("Training stoped with early stopping")
                break

            if epoch % self.val_check == 0 and epoch != 0:
                print("Epoch {:02d}, Loss_valid: {:9.4f}, acc_valid: {:9.4f}".format(
                    epoch, valid_loss, valid_acc))
            trial.report(valid_acc, epoch)

        self.accuracy = valid_acc

    def validate_nll(self, validation_data):
        """

        :param validation_data: validation data
        :return:
        """
        self.model.eval()
        with torch.no_grad():
            correct = 0
            nll_loss = 0
            for iteration, (x, y) in enumerate(validation_data):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
                nll_loss += mean_nll(output, y.reshape(-1))

        accuracy = correct / len(validation_data.dataset)
        return accuracy, nll_loss

    def validate_hsic(self, validation_data):
        """

        :param validation_data: validation data
        :return:
        """
        self.model.eval()
        num_classes = self.model.label_encoder.classes_.shape[0]
        with torch.no_grad():
            correct = 0
            hsic_loss = 0
            for iteration, (x, y) in enumerate(validation_data):
                y_h = (y == torch.arange(num_classes).reshape(1, num_classes)).float()
                x, y, y_h = x.to(self.device), y.to(self.device), y_h.to(self.device)
                output = self.model(x)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
                hsic_loss += HSIC(x, output - y_h, self.s_x, self.s_y)

        accuracy = correct / len(validation_data.dataset)
        return accuracy, hsic_loss

    def validate_irm(self, validation_data):
        """

        :param validation_data: validation data
        :return:
        """
        self.model.eval()
        with torch.no_grad():
            correct = 0
            nll_loss = 0
            for iteration, (x, y, y_ct) in enumerate(validation_data):
                x, y_ct = x.to(self.device), y_ct.to(self.device)
                pred_env = self.model(x)
                pred = pred_env.argmax(dim=1, keepdim=True)
                correct += pred.eq(y_ct.view_as(pred)).sum().item()
                nll_loss += mean_nll(pred_env, y_ct.reshape(-1))

        accuracy = correct / len(validation_data.dataset)
        return accuracy, nll_loss
