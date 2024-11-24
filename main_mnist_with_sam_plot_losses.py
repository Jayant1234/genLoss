import random
import time
import math
from argparse import ArgumentParser
from collections import defaultdict
from itertools import islice
from pathlib import Path
from sam import SAM

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision

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
    
    # create optimizer
    assert args.optimizer in optimizer_dict, f"Unsupported optimizer choice: {args.optimizer}"
    base_optimizer = optimizer_dict[args.optimizer]
    optimizer = SAM(mlp.parameters(), base_optimizer, rho=0.05, lr=args.lr, weight_decay=args.weight_decay)
    
    # define loss function
    assert args.loss_function in loss_function_dict
    loss_fn = loss_function_dict[args.loss_function]()

    train_losses, test_losses = [], []
    original_losses, perturbed_losses = [], []  # Track losses before and after perturbation
    log_steps = []

    steps = 0
    one_hots = torch.eye(10, 10).to(device)
    
    with tqdm(total=args.optimization_steps, dynamic_ncols=True) as pbar:
        for x, labels in islice(cycle(train_loader), args.optimization_steps):
            do_log = (steps < 30) or (steps < 150 and steps % 10 == 0) or steps % log_freq == 0
            
            # Calculate initial loss
            y = mlp(x.to(device))
            if args.loss_function == 'CrossEntropy':
                initial_loss = loss_fn(y, labels.to(device))
            elif args.loss_function == 'MSE':
                initial_loss = loss_fn(y, one_hots[labels])
            
            original_loss_value = initial_loss.item()
            
            # First step of SAM (includes weight perturbation)
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
            
            # Log the losses
            if do_log:
                original_losses.append(original_loss_value)
                perturbed_losses.append(perturbed_loss_value)
                log_steps.append(steps)
                
                print(f"\nStep {steps}:")
                print(f"Original Loss: {original_loss_value:.6f}")
                print(f"Perturbed Loss: {perturbed_loss_value:.6f}")
                print(f"Loss Difference: {perturbed_loss_value - original_loss_value:.6f}")
                
                # Plot the losses
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
                plt.savefig(f"results/sam_loss_comparison_{args.label}.png", dpi=150)
                plt.close()
            
            # Complete the SAM update
            perturbed_loss.backward()
            optimizer.second_step(zero_grad=True)
            
            steps += 1
            pbar.update(1)

if __name__ == '__main__':
    # [Previous argument parsing code remains the same]
    parser = ArgumentParser()
    # ... [rest of the argument parsing code]
    args = parser.parse_args()
    main(args)