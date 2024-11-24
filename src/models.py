import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

from .plottings import plot_line
from .topology import topological_complexity


class split_tanh(nn.Module):
    """
    An activation function that smoothes the initial activation function and is of form
    x \mapsto x + c * tanh(\lamb * (x - a))
    """

    def __init__(self, lamb=10.0):
        super(split_tanh, self).__init__()
        self.c = torch.nn.parameter.Parameter(torch.rand(1))
        self.a = torch.nn.parameter.Parameter(torch.randn(1))
        self.lamb = torch.nn.parameter.Parameter(torch.Tensor([lamb]))

    def forward(self, x):
        return x + self.c * torch.tanh(self.lamb * (x - self.a))


class split_sign(nn.Module):
    """
    An activation function that splits the data in the naive way
    x \mapsto x + c * sign(x)
    """

    def __init__(self):
        super(split_sign, self).__init__()
        self.c = torch.nn.parameter.Parameter(torch.rand(1))
        self.c.requiresGrad = True

    def forward(self, x):
        return x + self.c * torch.sign(x)


class split_sincos(nn.Module):
    """
    An activation function that splits the data in the sin-cos way
    """

    def __init__(self):
        super(split_sincos, self).__init__()
        self.a = torch.nn.parameter.Parameter(torch.pi * torch.rand(1) / 2)
        self.b = torch.nn.parameter.Parameter(torch.rand(1))

    def forward(self, x):
        cosa = torch.cos(self.a)
        sina = torch.sin(self.a)
        tga = torch.tan(self.a)

        res = torch.zeros_like(x)

        idx1 = x <= -cosa
        res[idx1] = self.b * x[idx1] + self.b * cosa - sina

        idx2 = (-cosa < x) & (x < cosa)
        res[idx2] = x[idx2] * tga

        idx3 = x >= cosa
        res[idx3] = x[idx3] + sina - cosa

        return res


ACTIVATIONS = {
    "split_tanh": torch.jit.script(split_tanh()),
    "split_sign": torch.jit.script(split_sign()),
    "split_sincos": torch.jit.script(split_sincos()),
    "relu": F.relu,
}


class Classifier1L(nn.Module):
    def __init__(
        self,
        dim_of_in=2,
        num_of_hidden=2,
        dim_of_hidden=3,
        activation=F.relu,
        use_batch_norm=False,
        initialize_weights=True,
        save_snapshots=True,
    ):
        """
        layers_signature = (dim_of_in, num_of_hidden, dim_of_hidden, is_with_split, split_constant)
        """
        super().__init__()

        self.fc_in = nn.Linear(dim_of_in, dim_of_hidden)
        self.norm = nn.BatchNorm1d(dim_of_hidden) if use_batch_norm else None
        self.hiddens = nn.ModuleList([nn.Linear(dim_of_hidden, dim_of_hidden) for _ in range(num_of_hidden)])
        self.fc_out = nn.Linear(dim_of_hidden, 2)

        self.activation = torch.jit.script(activation) if isinstance(activation, torch.nn.Module) else activation

        if initialize_weights:
            self.apply(self.__initialize_weights)

        if save_snapshots:
            self.snapshots = []

    def forward(self, x, save=False):
        x = self.fc_in(x)
        if self.norm:
            x = self.norm(x)
        x = self.activation(x)

        for l in self.hiddens[:-1]:
            x = F.relu(l(x))

        x = F.relu(self.hiddens[-1](x))
        if save:
            self.snapshots.append(x.cpu().detach().numpy())

        x = self.fc_out(x)

        return x

    def __initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            if isinstance(self.activation, torch.nn.modules.activation.ReLU):
                torch.nn.init.xavier_uniform_(module.weight)  #! check the paper of ReLU initialization and loss surface study
            else:
                torch.nn.init.xavier_normal_(module.weight)

            if module.bias is not None:
                module.bias.data.zero_()


class ClassifierAL(nn.Module):
    def __init__(
        self,
        dim_of_in=2,
        num_of_hidden=2,
        dim_of_hidden=3,
        activation=F.relu,
        use_batch_norm=False,
        initialize_weights=True,
    ):
        """
        layers_signature = (dim_of_in, num_of_hidden, dim_of_hidden, is_with_split, split_constant)
        """
        super().__init__()

        self.fc_in = nn.Linear(dim_of_in, dim_of_hidden)
        self.norm = nn.BatchNorm1d(dim_of_hidden) if use_batch_norm else None
        self.hiddens = nn.ModuleList([nn.Linear(dim_of_hidden, dim_of_hidden) for _ in range(num_of_hidden)])
        self.fc_out = nn.Linear(dim_of_hidden, 2)

        self.activation = torch.jit.script(activation) if isinstance(activation, torch.nn.Module) else activation

        self.topo_info = [None] * (num_of_hidden + 2)

        if initialize_weights:
            self.apply(self.__initialize_weights)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.activation(x)
        for _, l in enumerate(self.hiddens[:-1]):
            x = self.activation(l(x))
        x = F.relu(self.hiddens[-1](x))
        x = self.fc_out(x)

        return x

    def forward_with_save(self, x, **kwargs):
        self.topo_info[0] = topological_complexity(x.cpu().detach().numpy(), **kwargs)
        x = self.fc_in(x)
        if self.norm:
            x = self.norm(x)
        x = self.activation(x)
        self.topo_info[1] = topological_complexity(x.cpu().detach().numpy(), **kwargs)

        for i, l in enumerate(self.hiddens[:-1]):
            x = self.activation(l(x))
            self.topo_info[i + 2] = topological_complexity(x.cpu().detach().numpy(), **kwargs)

        x = F.relu(self.hiddens[-1](x))
        self.topo_info[-1] = topological_complexity(x.cpu().detach().numpy(), **kwargs)

        x = self.fc_out(x)

        return x

    def __initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            if isinstance(self.activation, torch.nn.modules.activation.ReLU):
                torch.nn.init.xavier_uniform_(module.weight)  #! check the paper of ReLU initialization and loss surface study
            else:
                torch.nn.init.xavier_normal_(module.weight)

            if module.bias is not None:
                module.bias.data.zero_()


class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, true_labels = batch
        logits = self.forward(features)
        loss = F.cross_entropy(logits, true_labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        features, true_labels = batch
        logits = self.forward(features)
        loss = F.cross_entropy(logits, true_labels)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def main():
    x_range = torch.linspace(-3, 3, 50)
    func = split_sincos()
    y_range = func(x_range).detach().numpy()
    plot_line(x_range, y_range, save=True, filename="sincos.png")


if __name__ == "__main__":
    main()
