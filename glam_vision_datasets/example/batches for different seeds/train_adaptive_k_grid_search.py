import argparse
import torch
import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

class AdaptiveLookahead(torch.optim.Optimizer):
    def __init__(self, base_optimizer, method, alpha=0.5, initial_k=5, k_multiplier=5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not initial_k >= 1:
            raise ValueError(f"Invalid initial_k: {initial_k}")

        self.base_optimizer = base_optimizer
        self.alpha = alpha
        self.k = initial_k
        self.k_multiplier = k_multiplier
        self._step_count = 0
        self.method = method
        
        self.last_lr = self.base_optimizer.param_groups[0]['lr']
        self.slow_params = []
        for group in self.base_optimizer.param_groups:
            sp = [p.clone().detach() for p in group['params']]
            self.slow_params.append(sp)

    def check_lr_change(self):
        current_lr = self.base_optimizer.param_groups[0]['lr']
        if self.method == 'adaptive_decrease' and current_lr < self.last_lr:
            self.k = max(1, self.k - self.k_multiplier)
        elif self.method == 'adaptive_increase' and current_lr < self.last_lr:
            self.k += self.k_multiplier
            print(f"\nLearning rate decreased from {self.last_lr:.6f} to {current_lr:.6f}. New k: {self.k}")
        self.last_lr = current_lr

    def step(self, closure=None):
        self.check_lr_change()
        loss = self.base_optimizer.step(closure)
        self._step_count += 1
        
        if self._step_count % self.k == 0:
            for group_idx, group in enumerate(self.base_optimizer.param_groups):
                for p_idx, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    slow = self.slow_params[group_idx][p_idx]
                    slow += self.alpha * (p.data - slow)
                    p.data.copy_(slow)
        return loss

def train_model(args, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    initialize(args, seed=seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = Cifar(args.batch_size, args.threads)
    log = Log(filename=args.label + f"lr{args.learning_rate}_k{args.initial_k}_seed{seed}", log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
    
    initial_weights = [p.clone().detach() for p in model.parameters()]
    epoch_weight_changes = []
    base_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = AdaptiveLookahead(base_optimizer, alpha=args.alpha, method=args.method, initial_k=args.initial_k, k_multiplier=args.k_multiplier)
    scheduler = StepLR(base_optimizer, args.learning_rate, args.epochs)
    
    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))
        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            
            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
            scheduler(epoch)
        
        weight_changes = [(p.data - p_init).norm(2).item() for p, p_init in zip(model.parameters(), initial_weights)]
        epoch_weight_changes.append(sum(weight_changes) / len(weight_changes))
        
        model.eval()
        log.eval(len_dataset=len(dataset.test))
        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
    
    final_accuracy = log.get_final_accuracy()
    return final_accuracy, log.train_losses, log.val_losses, log.train_accuracies, log.val_accuracies, epoch_weight_changes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--depth", default=16, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--learning_rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument("--weight_decay", default=0.0005, type=float)
    parser.add_argument("--width_factor", default=8, type=int)
    parser.add_argument("--label", default="Adaptive_K_Lookahead", type=str)
    parser.add_argument("--initial_k", default=5, type=int)
    parser.add_argument("--k_multiplier", default=5, type=int)
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--method", default="adaptive_decrease", type=str)
    args = parser.parse_args()
    
    k_values = [10, 20]
    lr_values = [0.1, 0.2,0.3,0.4]
    results = {}
    
    for k in k_values:
        for lr in lr_values:
            args.initial_k = k
            args.learning_rate = lr
            #generate a random number without using the random package by leveraging operating system randomness
            seed = int.from_bytes(os.urandom(4), 'big') % 100
            # seed = random.randint(0, 100)
            final_acc, train_losses, val_losses, train_accs, val_accs, weight_changes = train_model(args, seed)
            results[(k, lr, seed)] = (train_losses, val_losses, train_accs, val_accs, weight_changes)
    
    for (k, lr, seed), (train_losses, val_losses, train_accs, val_accs, weight_changes) in results.items():
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.legend()
        plt.title(f"Loss Curves (k={k}, lr={lr})")
        plt.savefig(f"loss_k{k}_lr{lr}_seed{seed}.png")
        plt.close()
    
        plt.figure()
        plt.plot(train_accs, label="Train Accuracy")
        plt.plot(val_accs, label="Validation Accuracy")
        plt.legend()
        plt.title(f"Accuracy Curves (k={k}, lr={lr})")
        plt.savefig(f"accuracy_k{k}_lr{lr}_seed{seed}.png")
        plt.close()
    
        plt.figure()
        plt.plot(weight_changes, marker='o')
        plt.title(f"Weight Change (k={k}, lr={lr})")
        plt.savefig(f"weight_change_k{k}_lr{lr}_seed{seed}.png")
        plt.close()
