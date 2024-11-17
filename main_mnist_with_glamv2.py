def train_mnist_baseline(model, train_data, valid_data, optimizer, scheduler, device, args):
    """
    Adapted from train_baseline to work with MNIST while keeping the same GLAM implementation
    """
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    
    steps_per_epoch = math.ceil(len(train_data) / args.batch_size)
    
    its, train_acc, val_acc, train_loss, val_loss, sim = [], [], [], [], [], []
    grads = None
    i = 0

    for e in tqdm(range(int(args.optimization_steps) // steps_per_epoch)):
        # Convert loaders to tensors for similar processing as original
        train_batches = []
        for images, labels in train_loader:
            # Combine images and labels into single tensor
            combined = torch.cat((images.view(images.size(0), -1), 
                                labels.unsqueeze(1).float()), dim=1)
            train_batches.append(combined)
        train_data_tensor = torch.cat(train_batches, dim=0)
        
        valid_batches = []
        for images, labels in valid_loader:
            combined = torch.cat((images.view(images.size(0), -1), 
                                labels.unsqueeze(1).float()), dim=1)
            valid_batches.append(combined)
        valid_data_tensor = torch.cat(valid_batches, dim=0)

        # Randomly shuffle train data
        train_data_tensor = train_data_tensor[torch.randperm(train_data_tensor.shape[0])]

        for data, is_train in [(train_data_tensor, True), (valid_data_tensor, False)]:
            model.train(is_train)
            total_loss = 0
            total_acc = 0
            avg_sim = 0
            
            # Split into batches like original implementation
            dl = torch.split(data, args.batch_size, dim=0)
            num_batches = len(dl)

            for num_batch in range(num_batches):
                input = dl[num_batch].to(device)
                b2_input = dl[(num_batch + 1) % num_batches].to(device)
                
                # Split combined tensor back into images and labels
                images = input[:, :-1].view(input.size(0), 1, 28, 28)
                labels = input[:, -1].long()
                b2_images = b2_input[:, :-1].view(b2_input.size(0), 1, 28, 28)
                b2_labels = b2_input[:, -1].long()

                with torch.set_grad_enabled(is_train):
                    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                        logits = model(images)
                        L_B1 = F.cross_entropy(logits, labels)
                        total_loss += L_B1.item() * input.shape[0]
                        
                        logits_b2 = model(b2_images)
                        L_B2 = F.cross_entropy(logits_b2, b2_labels)

                if is_train:
                    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                        model.zero_grad()

                        g_B1 = torch.autograd.grad(L_B1, model.parameters(), create_graph=True)
                        g_B2 = torch.autograd.grad(L_B2, model.parameters(), create_graph=True)
                        
                        # Compute dot product s = g_B2^T g_B1
                        s = sum((g1 * g2).sum() for g1, g2 in zip(g_B1, g_B2))

                        # Compute the norms of g_B1 and g_B2
                        norm_g_B1 = torch.sqrt(sum((g1 ** 2).sum() for g1 in g_B1))
                        norm_g_B2 = torch.sqrt(sum((g2 ** 2).sum() for g2 in g_B2))

                        # Normalize the dot product
                        cosine_sim = s / (norm_g_B1 * norm_g_B2 + 1e-8)
                        
                        # Compute gradient of s with respect to model parameters
                        grad_s = torch.autograd.grad((1-cosine_sim), model.parameters())
                        
                        if i % 1000 == 0 or num_batch == 0:
                            print("similarity of both gradients is::::", cosine_sim)
                            print(num_batch)
                        
                        total_grad = [g1+g2 + gs for g1, g2, gs in zip(g_B1, g_B2, grad_s)]
                        
                        for p, g in zip(model.parameters(), total_grad):
                            p.grad = g

                        trigger = i < 500 if args.two_stage else False

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
                        i += 1

                avg_sim += cosine_sim.item()
                acc = (logits.argmax(-1) == labels).float().mean()
                total_acc += acc.item() * input.shape[0]

            if is_train:
                train_acc.append(total_acc / len(train_data))
                train_loss.append(total_loss / len(train_data))
                sim.append(100 * avg_sim / num_batches)
                its.append(i)
            else:
                val_acc.append(total_acc / len(valid_data))
                val_loss.append(total_loss / len(valid_data))

        # Plotting and saving logic
        do_save = (e + 1) % 100 == 0
        if do_save:
            steps = torch.arange(len(train_acc)).numpy() * steps_per_epoch
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.plot(steps, train_acc, label="train")
            plt.plot(steps, val_acc, label="val")
            plt.legend()
            plt.title("MNIST Classification Accuracy")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Accuracy")
            plt.xscale("log", base=10)
            plt.grid()

            plt.subplot(1, 3, 2)
            plt.plot(steps, train_loss, label="train")
            plt.plot(steps, val_loss, label="val")
            plt.legend()
            plt.title("MNIST Classification Loss")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Loss")
            plt.xscale("log", base=10)
            plt.grid()

            plt.subplot(1, 3, 3)
            plt.plot(steps, sim, label="cosine")
            plt.legend()
            plt.title("Gradient Similarity")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Similarity")
            plt.xscale("log", base=10)
            plt.grid()

            plt.tight_layout()
            plt.savefig(f"results/mnist_acc_{args.label}.png", dpi=150)
            plt.close()

            results = {
                'its': its,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'sim': sim
            }

            torch.save(results, f"results/mnist_res_{args.label}.pt")

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

def main_with_glam(args):
    print("MAIN method with GLAM called")
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
    
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    similarities = []
    log_steps = []
    grads = None
    steps = 0
    
    # Get paired batches for GLAM
    train_iter = iter(train_loader)
    
    with tqdm(total=args.optimization_steps, dynamic_ncols=True) as pbar:
        while steps < args.optimization_steps:
            try:
                batch1 = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch1 = next(train_iter)
                
            try:
                batch2 = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch2 = next(train_iter)
                
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
            
            # Process first batch
            x1, labels1 = batch1
            x1, labels1 = x1.to(device), labels1.to(device)
            logits1 = mlp(x1)
            loss1 = F.cross_entropy(logits1, labels1)
            
            # Process second batch
            x2, labels2 = batch2
            x2, labels2 = x2.to(device), labels2.to(device)
            logits2 = mlp(x2)
            loss2 = F.cross_entropy(logits2, labels2)
            
            # Compute gradients with graph retention
            mlp.zero_grad()
            g1 = torch.autograd.grad(loss1, mlp.parameters(), create_graph=True)
            g2 = torch.autograd.grad(loss2, mlp.parameters(), create_graph=True)
            
            # Compute gradient similarity
            s = sum((g1_i * g2_i).sum() for g1_i, g2_i in zip(g1, g2))
            norm_g1 = torch.sqrt(sum((g1_i ** 2).sum() for g1_i in g1))
            norm_g2 = torch.sqrt(sum((g2_i ** 2).sum() for g2_i in g2))
            cosine_sim = s / (norm_g1 * norm_g2 + 1e-8)
            
            # Compute gradient of similarity
            grad_s = torch.autograd.grad((1-cosine_sim), mlp.parameters())
            
            # Combine gradients
            total_grad = [g1_i + g2_i + gs_i for g1_i, g2_i, gs_i in zip(g1, g2, grad_s)]
            
            # Apply gradients
            for p, g in zip(mlp.parameters(), total_grad):
                p.grad = g
                
            if args.filter == "ma":
                grads = gradfilter_ma(mlp, grads=grads, window_size=args.window_size, lamb=args.lamb, trigger=False)
            elif args.filter == "ema":
                grads = gradfilter_ema(mlp, grads=grads, alpha=args.alpha, lamb=args.lamb)
                
            optimizer.step()
            
            if do_log:
                similarities.append(cosine_sim.item())
                
                # Plot training curves
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.plot(log_steps, train_accuracies, label="train")
                plt.plot(log_steps, test_accuracies, label="test")
                plt.legend()
                plt.title("Accuracy")
                plt.xlabel("Steps")
                plt.xscale("log")
                
                plt.subplot(1, 3, 2)
                plt.plot(log_steps, train_losses, label="train")
                plt.plot(log_steps, test_losses, label="test")
                plt.legend()
                plt.title("Loss")
                plt.xlabel("Steps")
                plt.xscale("log")
                plt.yscale("log")
                
                plt.subplot(1, 3, 3)
                plt.plot(log_steps, similarities)
                plt.title("Gradient Similarity")
                plt.xlabel("Steps")
                plt.xscale("log")
                
                plt.tight_layout()
                plt.savefig(f"results/mnist_glam_{args.label}.png", dpi=150)
                plt.close()
                
                # Save results
                torch.save({
                    'steps': log_steps,
                    'train_acc': train_accuracies,
                    'test_acc': test_accuracies,
                    'train_loss': train_losses,
                    'test_loss': test_losses,
                    'similarities': similarities,
                }, f"results/mnist_glam_{args.label}.pt")
            
            steps += 1
            pbar.update(1)

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
    print(args.activation)
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
    optimizer = optimizer_dict[args.optimizer](mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # define loss function
    assert args.loss_function in loss_function_dict
    loss_fn = loss_function_dict[args.loss_function]()


    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
    norms, last_layer_norms, log_steps = [], [], []
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

            y = mlp(x.to(device))
            if args.loss_function == 'CrossEntropy':
                loss = loss_fn(y, labels.to(device))
            elif args.loss_function == 'MSE':
                loss = loss_fn(y, one_hots[labels])

            optimizer.zero_grad()
            loss.backward()

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

