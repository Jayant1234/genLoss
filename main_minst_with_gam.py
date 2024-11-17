import random
import time
import math
from argparse import ArgumentParser
from collections import defaultdict
from itertools import islice
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision

from grokfast import *


# Dictionary for activation functions
activation_dict = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'GELU': nn.GELU
}

# Optimizer dictionary
optimizer_dict = {
    'AdamW': torch.optim.AdamW,
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD
}

# Loss function dictionary
loss_function_dict = {
    'MSE': nn.MSELoss,
    'CrossEntropy': nn.CrossEntropyLoss
}


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def compute_accuracy(network, dataset, device, N=2000, batch_size=50):
    """Computes accuracy of `network` on `dataset`."""
    with torch.no_grad():
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        correct = 0
        total = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            logits = network(x.to(device))
            predicted_labels = torch.argmax(logits, dim=1)
            correct += torch.sum(predicted_labels == labels.to(device))
            total += x.size(0)
        return (correct / total).item()


def compute_loss(network, dataset, loss_function, device, N=2000, batch_size=50):
    """Computes mean loss of `network` on `dataset`."""
    with torch.no_grad():
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss_fn = loss_function_dict[loss_function](reduction='sum')
        one_hots = torch.eye(10, 10).to(device)
        total = 0
        points = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            y = network(x.to(device))
            if loss_function == 'CrossEntropy':
                total += loss_fn(y, labels.to(device)).item()
            elif loss_function == 'MSE':
                total += loss_fn(y, one_hots[labels]).item()
            points += len(labels)
        return total / points


def train_baseline(model, train_data, test_data, optimizer, loss_fn, device, args):
    model.train()
    steps = 0
    one_hots = torch.eye(10, 10).to(device)

    # Loading datasets for training
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
    log_steps = []

    with tqdm(total=args.optimization_steps, dynamic_ncols=True) as pbar:
        for x, labels in islice(cycle(train_loader), args.optimization_steps):
            model.zero_grad()
            y = model(x.to(device))

            if args.loss_function == 'CrossEntropy':
                loss = loss_fn(y, labels.to(device))
            elif args.loss_function == 'MSE':
                loss = loss_fn(y, one_hots[labels])

            loss.backward()
            optimizer.step()

            if steps % 100 == 0:
                # Log metrics every 100 steps
                train_losses.append(compute_loss(model, train_data, args.loss_function, device, N=len(train_data)))
                train_accuracies.append(compute_accuracy(model, train_data, device, N=len(train_data)))
                test_losses.append(compute_loss(model, test_data, args.loss_function, device, N=len(test_data)))
                test_accuracies.append(compute_accuracy(model, test_data, device, N=len(test_data)))
                log_steps.append(steps)

                pbar.set_description(
                    "L: {0:1.1e}|{1:1.1e}. A: {2:2.1f}%|{3:2.1f}%".format(
                        train_losses[-1],
                        test_losses[-1],
                        train_accuracies[-1] * 100,
                        test_accuracies[-1] * 100,
                    )
                )

            steps += 1
            pbar.update(1)

            # Saving results and plotting
            if steps % 1000 == 0:
                plt.plot(log_steps, train_accuracies, label="train")
                plt.plot(log_steps, test_accuracies, label="val")
                plt.legend()
                plt.title(f"MNIST Image Classification")
                plt.xlabel("Optimization Steps")
                plt.ylabel("Accuracy")
                plt.xscale("log", base=10)
                plt.grid()
                plt.savefig(f"results/mnist_acc_{args.label}.png", dpi=150)
                plt.close()

                plt.plot(log_steps, train_losses, label="train")
                plt.plot(log_steps, test_losses, label="val")
                plt.legend()
                plt.title(f"MNIST Image Classification")
                plt.xlabel("Optimization Steps")
                plt.ylabel(f"{args.loss_function} Loss")
                plt.xscale("log", base=10)
                plt.yscale("log", base=10)
                plt.grid()
                plt.savefig(f"results/mnist_loss_{args.label}.png", dpi=150)
                plt.close()

                torch.save({
                    'its': log_steps,
                    'train_acc': train_accuracies,
                    'train_loss': train_losses,
                    'val_acc': test_accuracies,
                    'val_loss': test_losses,
                }, f"results/mnist_{args.label}.pt")


def main(args):
    print("MAIN method called")
    log_freq = math.ceil(args.optimization_steps / 150)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    torch.set_default_dtype(dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    train = torchvision.datasets.MNIST(root=args.download_directory, train=True, 
        transform=torchvision.transforms.ToTensor(), download=True)
    test = torchvision.datasets.MNIST(root=args.download_directory, train=False, 
        transform=torchvision.transforms.ToTensor(), download=True)
    train = torch.utils.data.Subset(train, range(args.train_points))
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)

    # Ensure activation function is valid
    assert args.activation in activation_dict, f"Unsupported activation function: {args.activation}"
    activation_fn = activation_dict[args.activation]

    # Define model
    layers = [nn.Flatten()]
    for i in range(args.depth):
        if i == 0:
            layers.append(nn.Linear(784, args.width))
            layers.append(activation_fn())
        elif i == args.depth - 1:
            layers.append(nn.Linear(args.width, 10))
        else:
            layers.append(nn.Linear(args.width, args.width))
            layers.append(activation_fn())
    model = nn.Sequential(*layers).to(device)

    # Initialize model weights
    with torch.no_grad():
        for p in model.parameters():
            p.data = args.initialization_scale * p.data
    nparams = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f'Number of parameters: {nparams}')

    # Define optimizer
    assert args.optimizer in optimizer_dict, f"Unsupported optimizer choice: {args.optimizer}"
    optimizer = optimizer_dict[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Define loss function
    assert args.loss_function in loss_function_dict
    loss_fn = loss_function_dict[args.loss_function]()

    # Training using train_baseline
    train_baseline(model, train, test, optimizer, loss_fn, device, args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_points", type=int, default=1000)
    parser.add_argument("--optimization_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--loss_function", type=str, default="MSE")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--initialization_scale", type=float, default=8.0)
    parser.add_argument("--download_directory", type=str, default=".")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--activation", type=str, default="ReLU")

    # Grokfast
    parser.add_argument("--filter", type=str, choices=["none", "ma", "ema", "fir"], default="none")
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--lamb", type=float, default=5.0)
    args = parser.parse_args()

    # Generate experiment label based on arguments
    filter_str = ('_' if args.label != '' else '') + args.filter
    window_size_str = f'_w{args.window_size}'
    alpha_str = f'_a{args.alpha:.3f}'.replace('.', '')
    lamb_str = f'_l{args.lamb:.2f}'.replace('.', '')

    if args.filter == 'none':
        filter_suffix = ''
    elif args.filter == 'ma':
        filter_suffix = window_size_str + lamb_str
    elif args.filter == 'ema':
        filter_suffix = alpha_str + lamb_str
    else:
        raise ValueError(f"Unrecognized filter type {args.filter}")

    optim_suffix = ''
    if args.weight_decay != 0:
        optim_suffix = optim_suffix + f'_wd{args.weight_decay:.1e}'.replace('.', '')
    if args.lr != 1e-3:
        optim_suffix = optim_suffix + f'_lrx{int(args.lr / 1e-3)}'

    args.label = args.label + filter_str + filter_suffix + optim_suffix
    print(f'Experiment results saved under name: {args.label}')
    print(args)
    main(args)
