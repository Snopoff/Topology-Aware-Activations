import torch
from torch import nn
from torch import optim
from .datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train(model, X, y, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, X, y):
    model.eval()
    logits = model(X)
    _, top_class = logits.topk(1, dim=1)
    equals = top_class == y.view(*top_class.shape)
    acc = torch.mean(equals.type(torch.FloatTensor))
    return logits, acc


def train_eval_loop(
    model: nn.Module,
    data: Dataset,
    epochs=500,
    optimizer_lr=0.003,
    test_ratio=0.2,
    verbose=False,
    save_max=10,
    return_losses=False,
    return_loss_plot=False,
    return_topo_changes=False,
):
    model.to(DEVICE)
    train_x, train_y, test_x, test_y = data.train_test_split(
        test_ratio=test_ratio, datatype="torch", device=DEVICE
    )
    train_y, test_y = train_y.type(torch.LongTensor).to(DEVICE), test_y.type(
        torch.LongTensor
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=optimizer_lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, test_losses = [], []

    save_each = epochs // save_max

    for e in range(epochs):
        running_loss = 0
        running_loss += train(model, train_x, train_y, optimizer, criterion)

        test_loss = 0
        accuracy = 0

        with torch.no_grad():
            logits, acc = evaluate(model, test_x, test_y)
            test_loss += criterion(logits, test_y).cpu()
            accuracy += acc

            train_losses.append(running_loss)
            test_losses.append(test_loss)

            if e % 50 == 0 and verbose:
                print(
                    "Epoch: {}/{}.. ".format(e, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss),
                    "Test Loss: {:.3f}.. ".format(test_loss),
                    "Test Accuracy: {:.3f}".format(accuracy),
                )

    if return_losses:
        return (np.array(train_losses), np.array(test_losses))

    if return_loss_plot:
        fig = plt.Figure()
        plt.plot(range(epochs), train_losses, label="train loss")
        plt.plot(range(epochs), test_losses, label="test loss")
        plt.legend(loc="best")
        plt.xlabel("epochs")
        return fig

    if return_topo_changes:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        test_x = test_x.to(device)
        with torch.no_grad():
            model.eval()
            _ = model.forward_with_save(test_x)
        print(model.topo_info)
        return model.topo_info
