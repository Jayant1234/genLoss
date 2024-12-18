import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm
from itertools import islice
import matplotlib.pyplot as plt
import random
import time
import math
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np

from grokfast import *


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def compute_accuracy(network, dataset, device, N=2000, batch_size=50):
    """Computes accuracy of `network` on `dataset`.
    """
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
    """Computes mean loss of `network` on `dataset`.
    """
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


optimizer_dict = {
    'AdamW': torch.optim.AdamW,
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD
}

activation_dict = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'GELU': nn.GELU
}

loss_function_dict = {
    'MSE': nn.MSELoss,
    'CrossEntropy': nn.CrossEntropyLoss
}


def main(args):
    print("MAIN method with baseline GLAM implementation called")
    log_freq = math.ceil(args.optimization_steps / 150)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    torch.set_default_dtype(dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load dataset
    train = torchvision.datasets.MNIST(root=args.download_directory, train=True, 
        transform=torchvision.transforms.ToTensor(), download=True)
    test = torchvision.datasets.MNIST(root=args.download_directory, train=False, 
        transform=torchvision.transforms.ToTensor(), download=True)
    train = torch.utils.data.Subset(train, range(args.train_points))
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)

    # create model
    layers = [nn.Flatten()]
    for i in range(args.depth):
        if i == 0:
            layers.append(nn.Linear(784, args.width))
            layers.append(activation_dict[args.activation]())
        elif i == args.depth - 1:
            layers.append(nn.Linear(args.width, 10))
        else:
            layers.append(nn.Linear(args.width, args.width))
            layers.append(activation_dict[args.activation]())
    mlp = nn.Sequential(*layers).to(device)
    with torch.no_grad():
        for p in mlp.parameters():
            p.data = args.initialization_scale * p.data
    nparams = sum([p.numel() for p in mlp.parameters() if p.requires_grad])
    print(f'Number of parameters: {nparams}')

    # create optimizer
    optimizer = optimizer_dict[args.optimizer](mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # define loss function
    loss_fn = nn.CrossEntropyLoss()

    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
    log_steps = []
    grads = None
    steps = 0
    cosine_similarities = []

    with tqdm(total=args.optimization_steps, dynamic_ncols=True) as pbar:
        for x, labels in islice(cycle(train_loader), args.optimization_steps):
            do_log = (steps < 30) or (steps < 150 and steps % 10 == 0) or steps % log_freq == 0
            if do_log:
                train_losses.append(compute_loss(mlp, train, 'CrossEntropy', device, N=len(train)))
                train_accuracies.append(compute_accuracy(mlp, train, device, N=len(train)))
                test_losses.append(compute_loss(mlp, test, 'CrossEntropy', device, N=len(test)))
                test_accuracies.append(compute_accuracy(mlp, test, device, N=len(test)))
                log_steps.append(steps)
                cosine_similarities.append(cosine_sim.item())

                pbar.set_description(
                    "L: {0:1.1e}|{1:1.1e}. A: {2:2.1f}%|{3:2.1f}%".format(
                        train_losses[-1],
                        test_losses[-1],
                        train_accuracies[-1] * 100, 
                        test_accuracies[-1] * 100,
                    )
                )

            # Prepare data
            x, labels = x.to(device), labels.to(device)

            # Get B1 loss and gradients
            logits_B1 = mlp(x)
            loss_B1 = loss_fn(logits_B1, labels)

            # Compute B1 gradients for potential similarity computation
            mlp.zero_grad()
            loss_B1.backward(create_graph=True)
            g_B1 = [p.grad.detach().clone() for p in mlp.parameters()]

            # Prepare second batch by cycling through dataset
            try:
                x2, labels2 = next(iter(cycle(train_loader)))
                x2, labels2 = x2.to(device), labels2.to(device)
            except StopIteration:
                # Reset iterator if needed
                train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
                x2, labels2 = next(iter(train_loader))
                x2, labels2 = x2.to(device), labels2.to(device)

            # Compute B2 loss and gradients
            logits_B2 = mlp(x2)
            loss_B2 = loss_fn(logits_B2, labels2)

            # Recompute first batch gradients to ensure proper graph
            mlp.zero_grad()
            loss_B1.backward(create_graph=True)
            g_B1 = [p.grad.detach().clone() for p in mlp.parameters()]
            
            # Compute B2 gradients
            mlp.zero_grad()
            loss_B2.backward(create_graph=True)
            g_B2 = [p.grad.detach().clone() for p in mlp.parameters()]

            # Compute gradient similarity
            s = sum((g1 * g2).sum() for g1, g2 in zip(g_B1, g_B2))
            norm_g_B1 = torch.sqrt(sum((g1 ** 2).sum() for g1 in g_B1))
            norm_g_B2 = torch.sqrt(sum((g2 ** 2).sum() for g2 in g_B2))
            cosine_sim = s / (norm_g_B1 * norm_g_B2 + 1e-8)
            

            # Determine gradient composition based on early stopping steps
            mlp.zero_grad()
            if args.early_stopping_steps <= 0:
                # Standard SGD: just sum the gradients
                total_grad = [g1 + g2 for g1, g2 in zip(g_B1, g_B2)]
            else:
                # GLAM method: use cosine similarity gradient for the first n steps
                if steps < args.early_stopping_steps:
                    # Compute gradient of cosine similarity
                    grad_s = torch.autograd.grad((1-cosine_sim), mlp.parameters())
                    total_grad = [g1 + g2 + gs for g1, g2, gs in zip(g_B1, g_B2, grad_s)]
                else:
                    # Fallback to standard gradient summation
                    total_grad = [g1 + g2 for g1, g2 in zip(g_B1, g_B2)]

            # Assign total gradient
            for p, g in zip(mlp.parameters(), total_grad):
                p.grad = g

            # Optional gradient filtering
            trigger = False
            if args.filter == "none":
                pass
            elif args.filter == "ma":
                grads = gradfilter_ma(mlp, grads=grads, window_size=args.window_size, lamb=args.lamb, trigger=trigger)
            elif args.filter == "ema":
                grads = gradfilter_ema(mlp, grads=grads, alpha=args.alpha, lamb=args.lamb)
            else:
                raise ValueError(f"Invalid gradient filter type `{args.filter}`")

            # Optimization step
            optimizer.step()
            steps += 1
            pbar.update(1)

            # Periodic logging and plotting
            if do_log:
                title = "MNIST Image Classification"

                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.plot(log_steps, train_accuracies, label="train")
                plt.plot(log_steps, test_accuracies, label="val")
                plt.legend()
                plt.title(title + " - Accuracy")
                plt.xlabel("Optimization Steps")
                plt.ylabel("Accuracy")
                plt.xscale("log", base=10)
                plt.grid()

                plt.subplot(122)
                plt.plot(log_steps, train_losses, label="train")
                plt.plot(log_steps, test_losses, label="val")
                plt.legend()
                plt.title(title + " - Loss")
                plt.xlabel("Optimization Steps")
                plt.ylabel("Loss")
                plt.xscale("log", base=10)
                plt.yscale("log", base=10)
                plt.grid()

                plt.tight_layout()
                plt.savefig(f"results/mnist_acc_{args.label}.png", dpi=150)
                plt.close()
                # Cosine Similarity plot
                plt.figure(figsize=(10, 5))
                plt.plot(log_steps, cosine_similarities, label="Cosine Similarity", color="purple")
                plt.legend()
                plt.title(title + " - Cosine Similarity")
                plt.xlabel("Optimization Steps")
                plt.ylabel("Cosine Similarity")
                plt.xscale("log", base=10)
                plt.grid()

                plt.tight_layout()
                plt.savefig(f"results/mnist_acc_{args.label}.png", dpi=150)
                plt.close()


                torch.save({
                    'its': log_steps,
                    'train_acc': train_accuracies,
                    'train_loss': train_losses,
                    'val_acc': test_accuracies,
                    'val_loss': test_losses,
                }, f"results/mnist_{args.label}.pt")


if __name__ == '__main__':
    # [Rest of the argument parsing code remains the same as in the original script]
    parser = ArgumentParser()
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--train_points", type=int, default=1000)
    parser.add_argument("--optimization_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--loss_function", type=str, default="CrossEntropy")
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
    parser.add_argument("--early_stopping_steps", type=int, default=-1)
    
    args = parser.parse_args()

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