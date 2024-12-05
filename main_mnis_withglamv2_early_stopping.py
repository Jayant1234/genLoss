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


def train_mnist_baseline(model, train_data, valid_data, optimizer, scheduler, device, args,early_stopping_steps=-1):
    """
    Modified GLAM implementation where cosine similarity is only used for first 100 steps
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
                        
                        # Only use gradient of cosine similarity in first 100 steps
			
                        if i < early_stopping_steps:

                            grad_s = torch.autograd.grad((1-cosine_sim), model.parameters())
                            total_grad = [g1+g2 + gs for g1, g2, gs in zip(g_B1, g_B2, grad_s)]
                        else:
                            total_grad = [g1+g2 for g1, g2 in zip(g_B1, g_B2)]
                        
                        if i % 1000 == 0 or num_batch == 0:
                            print("similarity of both gradients is::::", cosine_sim)
                            print(num_batch)
                        
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

