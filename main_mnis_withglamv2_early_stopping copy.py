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
from itertools import islice
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision

from grokfast import *
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

def train_mnist_baseline(model, train_data, valid_data, optimizer, scheduler, device, args):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    
    log_freq = math.ceil(args.optimization_steps / 150)

    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
    log_steps, sim = [], []
    grads = None
    steps = 0

    one_hots = torch.eye(10, 10).to(device)
    loss_fn = loss_function_dict[args.loss_function]()

    with tqdm(total=args.optimization_steps, dynamic_ncols=True) as pbar:
        for (x1, labels1), (x2, labels2) in zip(cycle(train_loader), cycle(train_loader)):
            do_log = (steps < 30) or (steps < 150 and steps % 10 == 0) or steps % log_freq == 0
            
            if do_log:
                # Compute losses and accuracies
                train_losses.append(compute_loss(model, train_data, args.loss_function, device, N=len(train_data)))
                train_accuracies.append(compute_accuracy(model, train_data, device, N=len(train_data)))
                test_losses.append(compute_loss(model, valid_data, args.loss_function, device, N=len(valid_data)))
                test_accuracies.append(compute_accuracy(model, valid_data, device, N=len(valid_data)))
                log_steps.append(steps)

                pbar.set_description(
                    "L: {0:1.1e}|{1:1.1e}. A: {2:2.1f}%|{3:2.1f}%".format(
                        train_losses[-1],
                        test_losses[-1],
                        train_accuracies[-1] * 100, 
                        test_accuracies[-1] * 100,
                    )
                )

            # Process first batch
            y1 = model(x1.to(device))
            if args.loss_function == 'CrossEntropy':
                L_B1 = loss_fn(y1, labels1.to(device))
            elif args.loss_function == 'MSE':
                L_B1 = loss_fn(y1, one_hots[labels1].to(device))

            # Process second batch
            y2 = model(x2.to(device))
            if args.loss_function == 'CrossEntropy':
                L_B2 = loss_fn(y2, labels2.to(device))
            elif args.loss_function == 'MSE':
                L_B2 = loss_fn(y2, one_hots[labels2].to(device))

            optimizer.zero_grad()

            # Compute gradient for first loss
            g_B1 = torch.autograd.grad(L_B1, model.parameters(), create_graph=True)
            # Compute gradient for second loss
            g_B2 = torch.autograd.grad(L_B2, model.parameters(), create_graph=True)
            
            # Compute dot product s = g_B2^T g_B1
            s = sum((g1 * g2).sum() for g1, g2 in zip(g_B1, g_B2))

            # Compute the norms of g_B1 and g_B2
            norm_g_B1 = torch.sqrt(sum((g1 ** 2).sum() for g1 in g_B1))
            norm_g_B2 = torch.sqrt(sum((g2 ** 2).sum() for g2 in g_B2))

            # Normalize the dot product
            cosine_sim = s / (norm_g_B1 * norm_g_B2 + 1e-8)
            
            # Only use gradient of cosine similarity in first 100 steps
            if steps < args.early_stopping_steps or args.early_stopping_steps == -1:
                # Here we are using cosine similarity for the first early stopping steps
                grad_s = torch.autograd.grad((1-cosine_sim), model.parameters())
                total_grad = [g1+g2 + gs for g1, g2, gs in zip(g_B1, g_B2, grad_s)]
                sim.append(cosine_sim.item())
            else:
                # here we are using the sum of gradients of both batches that is g_B1 and g_B2 which means its a simple SGD
                total_grad = [g1+g2 for g1, g2 in zip(g_B1, g_B2)]
                sim.append(0)  # Append 0 to maintain list length
            
            # Set model parameters' gradients
            for p, g in zip(model.parameters(), total_grad):
                p.grad = g

            trigger = False
            if args.filter == "none":
                pass
            elif args.filter == "ma":
                grads = gradfilter_ma(model, grads=grads, window_size=args.window_size, lamb=args.lamb, trigger=trigger)
            elif args.filter == "ema":
                grads = gradfilter_ema(model, grads=grads, alpha=args.alpha, lamb=args.lamb)
            else:
                raise ValueError(f"Invalid gradient filter type `{args.filter}`")

            optimizer.step()
            scheduler.step()

            steps += 1
            pbar.update(1)

            if steps >= args.optimization_steps:
                break

            # Logging and plotting logic
            if do_log:
                title = "MNIST Image Classification"

                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.plot(log_steps, train_accuracies, label="train")
                plt.plot(log_steps, test_accuracies, label="val")
                plt.legend()
                plt.title("MNIST Classification Accuracy")
                plt.xlabel("Optimization Steps")
                plt.ylabel("Accuracy")
                plt.xscale("log", base=10)
                plt.grid()

                plt.subplot(1, 3, 2)
                plt.plot(log_steps, train_losses, label="train")
                plt.plot(log_steps, test_losses, label="val")
                plt.legend()
                plt.title("MNIST Classification Loss")
                plt.xlabel("Optimization Steps")
                plt.ylabel(f"{args.loss_function} Loss")
                plt.xscale("log", base=10)
                plt.yscale("log", base=10)
                plt.grid()

                plt.subplot(1, 3, 3)
                plt.plot(log_steps, sim, label="cosine")
                plt.legend()
                plt.title("Gradient Similarity")
                plt.xlabel("Optimization Steps")
                plt.ylabel("Similarity")
                plt.xscale("log", base=10)
                plt.grid()

                plt.tight_layout()
                plt.savefig(f"results/mnist_acc_{args.label}.png", dpi=150)
                plt.close()

                # Save results
                torch.save({
                    'its': log_steps,
                    'train_acc': train_accuracies,
                    'train_loss': train_losses,
                    'val_acc': test_accuracies,
                    'val_loss': test_losses,
                    'sim': sim
                }, f"results/mnist_res_{args.label}.pt")
# Update the main function to use this implementation
def main(args):
    print("MAIN method with baseline GLAM implementation called")
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
    
    # Create model
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
    
    # Initialize weights
    with torch.no_grad():
        for p in mlp.parameters():
            p.data = args.initialization_scale * p.data
            
    optimizer = optimizer_dict[args.optimizer](mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)  # Constant LR


    # Train using baseline implementation
    train_mnist_baseline(mlp, train, test, optimizer, scheduler, device, args)



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

   # Ablation studies
    parser.add_argument("--two_stage", action='store_true')
    parser.add_argument("--save_weights", action='store_true')

    # Batch size and budget for the   GLAM method
    parser.add_argument("--budget", type=int, default=3e5)
    # parser.add_argument("--batch_size", type=int, default=512)

    # Grokfast
    parser.add_argument("--filter", type=str, choices=["none", "ma", "ema", "fir"], default="none")
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--lamb", type=float, default=5.0)
   
    # Early stopping steps
    # Here, by default the early stopping steps are set to -1, which means that the cosine similarity is not used
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

