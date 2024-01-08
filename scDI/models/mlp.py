import torch.nn as nn
import torch


class MLP(nn.Module):
    """
            # Parameters
                input_dim: integer
                    Number of input features (i.e. gene in case of scRNA-seq).
                num_classes: integer
                    Number of classes (conditions) the data contain. if `None` the model
                    will be a normal VAE instead of conditional VAE.
                use_batch_norm: boolean
                    if `True` batch normalization will applied to hidden layers
                dr_rate: float
                    Dropput rate applied to hidden layer, if `dr_rate`==0 no dropput will be applied.
                out_activation: str
                    activation type
        """

    def __init__(self, input_dim, num_classes=None, layer_sizes=[128, 64],
                 reg_weight=1, penalty_anneal_iters=10, dr_rate=None, linear0_anneal1=2, use_bn=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_cls = num_classes
        self.reg_weight = reg_weight
        self.penalty_anneal_iters = penalty_anneal_iters
        self.linear0_anneal1 = linear0_anneal1
        layer_sizes.append(num_classes)
        self.FC = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip([input_dim] + layer_sizes[:-1], layer_sizes)):
            if i + 1 < len(layer_sizes):
                self.FC.add_module(
                    name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=False))
                if use_bn:
                    self.FC.add_module("B{:d}".format(i), module=nn.BatchNorm1d(out_size, affine=True))
                self.FC.add_module(name="A{:d}".format(i), module=nn.ReLU())
                if dr_rate is not None:
                    self.FC.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dr_rate))

            else:

                self.FC.add_module(name="output".format(i), module=nn.Linear(in_size, out_size))

    def forward(self, x):
        logits = self.FC(x)
        return logits

    def predict(self, x):
        x = torch.tensor(x).to(self.device)
        x = self.forward(x)
        return torch.argmax(x, 1).cpu().data.numpy()

    def predict_hsic(self, x):
        x = torch.tensor(x).to(self.device)
        x = self.forward(x)
        x += self.b
        return torch.argmax(x, 1).cpu().data.numpy()


