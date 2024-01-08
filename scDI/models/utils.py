import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import random_split


def mean_nll(logits, y):
    return nn.functional.cross_entropy(logits, y)


def mean_accuracy(logits, y):
    pred = torch.argmax(logits, 1)
    return ((pred - y).abs() < 1e-2).float().mean()


def label_encoder(adata, label_encoder=None, label_key='label'):
    if label_encoder is None:
        le = LabelEncoder()
        labels = le.fit_transform(adata.obs[label_key]).reshape(-1, 1)
    else:
        le = label_encoder
        labels = le.transform(adata.obs[label_key]).reshape(-1, 1)
    return labels, le


# dataset definition
class CreateDataset(Dataset):
    # load the dataset
    def __init__(self, adata, label_key=None, le=None):
        self.adata = adata
        self.X = np.array(adata.X)
        self.X = self.X.astype('float32')
        self.y, self.le = label_encoder(self.adata, le, label_key=label_key)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.2):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

    def get_label_encoder(self):
        return self.le


class IRM_CustomDatasetFromAdata(Dataset):
    def __init__(self, adata, label_key=None, condition_key=None, le_ct=None, le_cnd=None):
        self.condition_key = condition_key
        self.label_key = label_key
        self.X = np.array(adata.X)
        self.X = self.X.astype('float32')
        self.cond_labels, self.le_cnd = label_encoder(adata, label_encoder=le_cnd, label_key=condition_key)
        self.ct_labels, self.le_ct = label_encoder(adata, label_encoder=le_ct, label_key=label_key)

    def __getitem__(self, index):
        return [self.X[index], self.cond_labels[index], self.ct_labels[index]]

    def __len__(self):
        return len(self.X)

    def get_condition_ecnoder(self):
        return self.le_cnd

    def get_label_encoder(self):
        return self.le_ct

def train_test_split(adata, train_frac=0.80, seed=2):
    np.random.seed(seed)
    train_size = int(adata.shape[0] * train_frac)
    indices = np.arange(adata.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    train_data = adata[train_idx, :]
    valid_data = adata[test_idx, :]
    return train_data, valid_data


def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / (sigma ** 2))


def HSIC(x, y, s_x=10, s_y=0.5):
    m = x.size(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        K = GaussianKernelMatrix(x, s_x).cuda()
        L = GaussianKernelMatrix(y, s_y).cuda()
        H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
        H = H.float().cuda()
    else:
        K = GaussianKernelMatrix(x, s_x)
        L = GaussianKernelMatrix(y, s_y)
        H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
        H = H.float()

    HSIC = torch.trace(torch.mm(K, torch.mm(H, torch.mm(L, H)))) / ((m - 1) ** 2)
    return HSIC

def compute_penalty(loss, dummy_w):
    g1 = grad(loss, dummy_w, create_graph=True)[0]
    return torch.sum(g1 ** 2)


def penalty(logits, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        scale = torch.tensor(1.).cuda().requires_grad_()
    else:
        scale = torch.tensor(1.).requires_grad_()
    loss = mean_nll(logits * scale, y.reshape(-1))
    grad_ = grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad_ ** 2)

# adopted from https://github.com/yaodongyu/TRADES/blob/master/trades.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
import torch.optim as optim


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y.reshape(-1))
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target.reshape(-1), size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target.reshape(-1), size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_bias(model, training_data, device):
    model.eval()
    num_classes = model.label_encoder.classes_.shape[0]
    with torch.no_grad():
        correct = 0
        b = torch.zeros((1, num_classes))
        for iteration, (x, y) in enumerate(training_data):
            y_h = (y == torch.arange(num_classes).reshape(1, num_classes)).float()
            x, y, y_h = x.to(device), y.to(device), y_h.to(device)
            output = model(x)
            b += (y_h - output).sum(axis=0).cpu()

    bias = b / len(training_data.dataset)
    model.b = bias.to(device)


# taken from https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a <= best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a >= best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                        best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                        best * min_delta / 100)
