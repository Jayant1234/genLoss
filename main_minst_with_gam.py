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

# Define the train_baseline function
def train_baseline(model, train_data, valid_data, optimizer, scheduler, device, args):
    steps_per_epoch = math.ceil(len(train_data) / args.batch_size)
    
    its, train_acc, val_acc, train_loss, val_loss, sim = [], [], [], [], [], []
    grads = None
    i = 0

    for e in tqdm(range(int(args.optimization_steps) // steps_per_epoch)):

        # Randomly shuffle train data
        train_data = train_data[torch.randperm(train_data.size(0))]

        for data, is_train in [(train_data, True), (valid_data, False)]:

            model.train(is_train)
            total_loss = 0
            total_acc = 0
            avg_sim = 0

            # Split data into batches
            dl = torch.split(data, args.batch_size)
            num_batches = len(dl)

            for num_batch in range(num_batches):
                input_data = dl[num_batch].to(device)

                with torch.set_grad_enabled(is_train):
                    logits = model(input_data)
                    loss = nn.CrossEntropyLoss()(logits, input_data)  # Change this according to the dataset structure
                    total_loss += loss.item() * input_data.size(0)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                acc = (logits.argmax(dim=1) == input_data).float().mean()  # Modify this part for actual labels
                total_acc += acc.item() * input_data.size(0)

            if is_train:
                train_acc.append(total_acc / len(train_data))
                train_loss.append(total_loss / len(train_data))
                its.append(i)
            else:
                val_acc.append(total_acc / len(valid_data))
                val_loss.append(total_loss / len(valid_data))

        # Save progress at specific intervals
        if (e + 1) % 100 == 0 or e == (int(args.optimization_steps) // steps_per_epoch - 1):
            plt.plot(its, train_acc, label="Train")
            plt.plot(its, val_acc, label="Validation")
            plt.legend()
            plt.title("MNIST Training Progress")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Accuracy")
            plt.xscale("log", base=10)
            plt.grid()
            plt.savefig(f"results/mnist_acc_{args.label}.png", dpi=150)
            plt.close()

            plt.plot(its, train_loss, label="Train")
            plt.plot(its, val_loss, label="Validation")
            plt.legend()
            plt.title("MNIST Training Progress")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Loss")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            plt.grid()
            plt.savefig(f"results/mnist_loss_{args.label}.png", dpi=150)
            plt.close()

            torch.save({
                'its': its,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, f"results/mnist_{args.label}.pt")

# Modified main function to use train_baseline
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

    # Define model
    assert args.activation in activation_dict, f"Unsupported activation function: {args.activation}"
    activation_fn = activation_dict[args.activation]

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
    train_baseline(model, train_loader.dataset.data, test.data, optimizer, None, device, args)

if __name__ == '__main__':
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
