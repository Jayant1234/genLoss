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

    assert args.activation in activation_dict, f"Unsupported activation function: {args.activation}"
    activation_fn = activation_dict[args.activation]

    # create model
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
    mlp = nn.Sequential(*layers).to(device)
    with torch.no_grad():
        for p in mlp.parameters():
            p.data = args.initialization_scale * p.data
    nparams = sum([p.numel() for p in mlp.parameters() if p.requires_grad])
    print(f'Number of parameters: {nparams}')

    # create optimizer
    assert args.optimizer in optimizer_dict, f"Unsupported optimizer choice: {args.optimizer}"
    optimizer = optimizer_dict[args.optimizer](mlp.parameters(), lr=args.lr) #weight_decay=args.weight_decay)

    # define loss function
    assert args.loss_function in loss_function_dict
    loss_fn = loss_function_dict[args.loss_function]()


    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
    norms, last_layer_norms, log_steps, sim = [], [], [], []
    grads = None

    steps = 0
    one_hots = torch.eye(10, 10).to(device)
    with tqdm(total=args.optimization_steps, dynamic_ncols=True) as pbar:
        for x, labels in islice(cycle(train_loader), args.optimization_steps):
            do_log = (steps < 30) or (steps < 150 and steps % 10 == 0) or steps % log_freq == 0
            if do_log:
                train_losses.append(compute_loss(mlp, train, args.loss_function, device, N=len(train)))
                train_accuracies.append(compute_accuracy(mlp, train, device, N=len(train)))
                test_losses.append(compute_loss(mlp, test, args.loss_function, device, N=len(test)))
                test_accuracies.append(compute_accuracy(mlp, test, device, N=len(test)))
                log_steps.append(steps)

                pbar.set_description(
                    "L: {0:1.1e}|{1:1.1e}. A: {2:2.1f}%|{3:2.1f}%".format(
                        train_losses[-1],
                        test_losses[-1],
                        train_accuracies[-1] * 100, 
                        test_accuracies[-1] * 100,
                    )
                )
           # Split the batch into two halves
            mid_point = x.size(0) // 2
            x1, x2 = x[:mid_point], x[mid_point:]
            labels1, labels2 = labels[:mid_point], labels[mid_point:]

            # Compute predictions for both halves
            y1 = mlp(x1.to(device))
            y2 = mlp(x2.to(device))

            # Compute losses for both halves
            if args.loss_function == 'CrossEntropy':
                L_B1 = loss_fn(y1, labels1.to(device))
                L_B2 = loss_fn(y2, labels2.to(device))
            elif args.loss_function == 'MSE':
                L_B1 = loss_fn(y1, one_hots[labels1])
                L_B2 = loss_fn(y2, one_hots[labels2])

            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                model.zero_grad()
                if(args.similarity_type =="cosine"): 
                    g_B1 = torch.autograd.grad(L_B1, model.parameters(), create_graph=True)
                    g_B2 = torch.autograd.grad(L_B2, model.parameters(), create_graph=True)
                    
                    s = sum((g1 * g2).sum() for g1, g2 in zip(g_B1, g_B2))

                    #cosine_sim = sum((g1 * g2).sum()/(torch.sqrt(sum((g1 ** 2)))*torch.sqrt(sum((g2 ** 2)))+ 1e-8) for g1, g2 in zip(g_B1, g_B2)) #trying pair-wise cosine similarity

                    # Compute the norms of g_B1 and g_B2
                    norm_g_B1 = torch.sqrt(sum((g1 ** 2).sum() for g1 in g_B1))
                    norm_g_B2 = torch.sqrt(sum((g2 ** 2).sum() for g2 in g_B2))

                    #Normalize the dot product
                    similarity = s / (norm_g_B1 * norm_g_B2 + 1e-8)  # Add epsilon for numerical stability

                    # Compute gradient of s with respect to model parameters
                    grad_s = torch.autograd.grad((1-similarity), model.parameters())
                    if steps % 1000 == 0 or num_batch == 0: 
                        #print("gradient for coherence is:", grad_s)
                        #print("gradient for baseline is:", g_B1)
                        print("similarity of both gradients is::::",similarity)
                        print(num_batch)
                    
                    #curious case of barely doing it and still getting same results. 
                    if steps >args.cosine_steps: 
                        total_grad = [g1+g2 for g1,g2 in zip(g_B1, g_B2)]
                    #Compute total gradient
                    else: 
                        total_grad = [g1+g2 + gs for g1,g2, gs in zip(g_B1, g_B2, grad_s)]
                        
                elif (args.similarity_type =="euc"): 
                    # Compute gradient lists
                    g_B1 = torch.autograd.grad(L_B1, model.parameters(), create_graph=True)
                    g_B2 = torch.autograd.grad(L_B2, model.parameters(), create_graph=True)

                    # Flatten and concatenate all gradients into a single 1D vector
                    flat_g_B1 = torch.cat([g.view(-1) for g in g_B1], dim=0)
                    flat_g_B2 = torch.cat([g.view(-1) for g in g_B2], dim=0)

                    # Compute the Euclidean distance between the two gradient vectors
                    diff = flat_g_B1 - flat_g_B2
                    # Add a small epsilon for numerical stability
                    euclidean_dist = torch.sqrt(torch.sum(diff ** 2) + 1e-8)

                    # Normalize the Euclidean distance
                    # One option: divide by sum of their norms to get a scale-invariant measure
                    norm_g_B1 = torch.sqrt(torch.sum(flat_g_B1 ** 2) + 1e-8)
                    norm_g_B2 = torch.sqrt(torch.sum(flat_g_B2 ** 2) + 1e-8)
                    similarity = euclidean_dist / (norm_g_B1 + norm_g_B2)

                    # Compute the gradient of the loss (1 - normalized_euclidean_dist) to encourage minimization of distance
                    grad_s = torch.autograd.grad((1 - similarity), model.parameters(), create_graph=True)

                    if steps % 1000 == 0 or num_batch == 0: 
                        print("Normalized Euclidean distance of both gradients:", similarity)
                        print(num_batch)

                    # Combine gradients as before
                    if steps > args.cosine_steps:
                        total_grad = [g1 + g2 for g1, g2 in zip(g_B1, g_B2)]
                    else:
                        total_grad = [g1 + g2 + gs for g1, g2, gs in zip(g_B1, g_B2, grad_s)]

                
                #total_grad = [g1+g2 + gs for g1,g2, gs in zip(g_B1, g_B2, grad_s)]
                
                #Assign gradients to parameters
                for p, g in zip(model.parameters(), total_grad):
                    p.grad = g
                #######

            #######

            trigger = False

            if args.filter == "none":
                pass
            elif args.filter == "ma":
                grads = gradfilter_ma(mlp, grads=grads, window_size=args.window_size, lamb=args.lamb, trigger=trigger)
            elif args.filter == "ema":
                grads = gradfilter_ema(mlp, grads=grads, alpha=args.alpha, lamb=args.lamb)
            else:
                raise ValueError(f"Invalid gradient filter type `{args.filter}`")

            #######

            optimizer.step()

            steps += 1
            pbar.update(1)

            if do_log:
                title = (f"MNIST Image Classification")

                plt.plot(log_steps, train_accuracies, label="train")
                plt.plot(log_steps, test_accuracies, label="val")
                plt.legend()
                plt.title(title)
                plt.xlabel("Optimization Steps")
                plt.ylabel("Accuracy")
                plt.xscale("log", base=10)
                plt.grid()
                plt.savefig(f"results/mnist_acc_{args.label}.png", dpi=150)
                plt.close()

                plt.plot(log_steps, train_losses, label="train")
                plt.plot(log_steps, test_losses, label="val")
                plt.legend()
                plt.title(title)
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--train_points", type=int, default=1000)
    parser.add_argument("--optimization_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--loss_function", type=str, default="MSE")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--initialization_scale", type=float, default=8.0)
    parser.add_argument("--download_directory", type=str, default=".")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--activation", type=str, default="ReLU")
    parser.add_argument("--similarity_type", type=str, choices=["euc", "cosine"], default="cosine")
    parser.add_argument("--cosine_steps",type=int, default=100000)

    # Grokfast
    parser.add_argument("--filter", type=str, choices=["none", "ma", "ema", "fir"], default="none")
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--lamb", type=float, default=5.0)
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

    main(args)
