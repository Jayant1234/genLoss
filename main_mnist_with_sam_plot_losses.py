import random
import time
import math
from argparse import ArgumentParser
from collections import defaultdict
from itertools import islice
from pathlib import Path
from sam import SAM  # Using original SAM import

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision

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
    print("MAIN method called")
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
    base_optimizer = optimizer_dict[args.optimizer]
    optimizer = SAM(mlp.parameters(), base_optimizer, rho=0.05, lr=args.lr, weight_decay=args.weight_decay)
    
    # define loss function
    assert args.loss_function in loss_function_dict
    loss_fn = loss_function_dict[args.loss_function]()

    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
    original_losses, perturbed_losses = [], []  # Additional tracking for SAM investigation
    log_steps = []

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

            # Original forward pass and loss calculation
            y = mlp(x.to(device))
            if args.loss_function == 'CrossEntropy':
                initial_loss = loss_fn(y, labels.to(device))
            elif args.loss_function == 'MSE':
                initial_loss = loss_fn(y, one_hots[labels])
            
            original_loss_value = initial_loss.item()

            optimizer.zero_grad()
            initial_loss.backward()
            optimizer.first_step(zero_grad=True)

            # Calculate loss after perturbation
            y = mlp(x.to(device))
            if args.loss_function == 'CrossEntropy':
                perturbed_loss = loss_fn(y, labels.to(device))
            elif args.loss_function == 'MSE':
                perturbed_loss = loss_fn(y, one_hots[labels])
            
            perturbed_loss_value = perturbed_loss.item()

            if do_log:
                print(f"\nStep {steps}:")
                print(f"Original Loss: {original_loss_value:.6f}")
                print(f"Perturbed Loss: {perturbed_loss_value:.6f}")
                print(f"Loss Difference: {perturbed_loss_value - original_loss_value:.6f}")

                original_losses.append(original_loss_value)
                perturbed_losses.append(perturbed_loss_value)

                # Original plotting code
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

                # Additional plot for SAM loss investigation
                plt.figure(figsize=(10, 6))
                plt.plot(log_steps, original_losses, label="Original Loss")
                plt.plot(log_steps, perturbed_losses, label="Perturbed Loss")
                plt.legend()
                plt.title("SAM Loss Comparison")
                plt.xlabel("Optimization Steps")
                plt.ylabel(f"{args.loss_function} Loss")
                plt.xscale("log", base=10)
                plt.yscale("log", base=10)
                plt.grid(True)
                plt.savefig(f"results/mnist_sam_loss_comparison_{args.label}.png", dpi=150)
                plt.close()

                torch.save({
                    'its': log_steps,
                    'train_acc': train_accuracies,
                    'train_loss': train_losses,
                    'val_acc': test_accuracies,
                    'val_loss': test_losses,
                }, f"results/mnist_{args.label}.pt")

            perturbed_loss.backward()
            optimizer.second_step(zero_grad=True)

            steps += 1
            pbar.update(1)

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

    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    main(args)