import argparse
import torch
import sys
from torch.utils.data import DataLoader, random_split

sys.path.append("..")
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

import numpy as np
import matplotlib.pyplot as plt

# ---------------- AdaptiveLookahead Optimizer ----------------

class AdaptiveLookahead(torch.optim.Optimizer):
    def __init__(self, base_optimizer, alpha_min=0.1, alpha_max=0.8, k=5, smoothing_factor=0.9, patience=3):
        """
        Implements an Adaptive Lookahead Optimizer that adjusts alpha based on validation performance.

        :param base_optimizer: The inner optimizer (e.g., SGD with momentum)
        :param alpha_min: Minimum value of alpha
        :param alpha_max: Maximum value of alpha
        :param k: Number of inner updates before slow update
        :param smoothing_factor: EMA smoothing factor for validation loss and accuracy
        :param patience: Number of epochs to wait before reducing alpha if validation worsens
        """
        if not 0.0 <= alpha_min <= alpha_max <= 1.0:
            raise ValueError(f"Invalid alpha range: {alpha_min} to {alpha_max}")
        if not k >= 1:
            raise ValueError(f"Invalid k: {k}")

        self.base_optimizer = base_optimizer
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.k = k
        self.smoothing_factor = smoothing_factor  # EMA smoothing for validation metrics
        self.patience = patience  # Number of stagnant epochs before decreasing alpha

        self._step_count = 0
        self.val_loss_history = []
        self.val_acc_history = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.ema_val_loss = None  # Exponential moving average of validation loss
        self.ema_val_acc = None  # Exponential moving average of validation accuracy
        self.no_improvement_epochs = 0  # Counter for overfitting detection

        # Adaptive α settings
        self.current_alpha = (alpha_min + alpha_max) / 2  
        self.ema_alpha = self.current_alpha  # EMA for α updates

        # Slow weight storage
        self.slow_params = []
        for group in self.base_optimizer.param_groups:
            sp = [p.clone().detach() for p in group['params']]
            self.slow_params.append(sp)

    def sigmoid(self, x):
        """Scaled sigmoid function for stable α updates"""
        return 1 / (1 + np.exp(-x))

    def update_alpha(self, val_loss, val_acc):
        """Dynamically adjust α based on validation performance"""
        self.val_loss_history.append(val_loss)
        self.val_acc_history.append(val_acc)

        if self.ema_val_loss is None:
            self.ema_val_loss = val_loss
            self.ema_val_acc = val_acc
        else:
            # Smooth validation loss & accuracy using EMA
            self.ema_val_loss = (
                self.smoothing_factor * self.ema_val_loss + (1 - self.smoothing_factor) * val_loss
            )
            self.ema_val_acc = (
                self.smoothing_factor * self.ema_val_acc + (1 - self.smoothing_factor) * val_acc
            )

        # Compute improvement signals
        loss_improvement = (self.best_val_loss - self.ema_val_loss) / max(self.best_val_loss, 1e-8)
        acc_improvement = (self.ema_val_acc - self.best_val_acc) / max(1 - self.best_val_acc, 1e-8)

        # Clipping extreme values to stabilize α updates
        loss_improvement = np.clip(loss_improvement, -0.5, 0.5)
        acc_improvement = np.clip(acc_improvement, -0.5, 0.5)

        # Track best validation loss and accuracy
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.no_improvement_epochs = 0  # Reset counter if improvement detected
        else:
            self.no_improvement_epochs += 1  # Count stagnant/worsening epochs

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

        # Adaptive α strategy:
        if loss_improvement > 0 and acc_improvement > 0:
            # Both metrics improving → increase α slightly
            change_signal = 0.6 * loss_improvement + 0.4 * acc_improvement
        elif self.no_improvement_epochs >= self.patience:
            # Overfitting detected → decrease α
            change_signal = -0.7
            self.no_improvement_epochs = 0  # Reset counter after decreasing α
        else:
            # Neutral case → small adjustment
            change_signal = 0.2 * loss_improvement + 0.2 * acc_improvement

        # Compute new α using sigmoid scaling
        sigmoid_value = self.sigmoid(5 * change_signal)  # Scale input for smooth updates
        new_alpha = self.alpha_min + sigmoid_value * (self.alpha_max - self.alpha_min)

        # Apply EMA momentum to α updates
        self.ema_alpha = 0.8 * self.ema_alpha + 0.2 * new_alpha

        print(f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}, "
              f"Smoothed Loss: {self.ema_val_loss:.6f}, Smoothed Acc: {self.ema_val_acc:.4f}, "
              f"Alpha: {self.ema_alpha:.4f}")
        return self.ema_alpha

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self, closure=None, val_loss=None, val_acc=None):
        """Perform fast updates and periodic slow weight updates"""
        loss = self.base_optimizer.step(closure)
        self._step_count += 1

        # If per-batch update is desired (here we update alpha at epoch level only)
        if val_loss is not None and val_acc is not None:
            self.current_alpha = self.update_alpha(val_loss, val_acc)

        if self._step_count % self.k == 0:
            for group_idx, group in enumerate(self.base_optimizer.param_groups):
                for p_idx, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    slow = self.slow_params[group_idx][p_idx]
                    slow += self.ema_alpha * (p.data - slow)
                    p.data.copy_(slow)

        return loss

# ---------------- Alpha History Plot Function ----------------

alpha_history = []
def plot_alpha_history(alpha_history, filename="alpha_history.png"):
    """
    Plot the history of alpha values during training
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(alpha_history)), alpha_history, 'b-', linewidth=2)
    plt.title('Adaptive Alpha Value Over Training', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Alpha Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1)  # Alpha is between 0 and 1
    
    # Add horizontal lines for min and max alpha values
    if alpha_history:
        plt.axhline(y=min(alpha_history), color='r', linestyle='--', alpha=0.5, 
                   label=f'Min: {min(alpha_history):.4f}')
        plt.axhline(y=max(alpha_history), color='g', linestyle='--', alpha=0.5, 
                   label=f'Max: {max(alpha_history):.4f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ---------------- Main Training Script ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--label", default="AdaptiveLookaheadv7", type=str, help="Label for the experiment.")
    parser.add_argument("--method_type", default="adaptive_lookahead", type=str, help="Method type for training.")
    parser.add_argument("--alpha_min", default=0.2, type=float, help="Minimum alpha value for adaptive lookahead.")
    parser.add_argument("--alpha_max", default=0.9, type=float, help="Maximum alpha value for adaptive lookahead.")
    parser.add_argument("--lookahead_k", default=10, type=int, help="Number of fast updates before slow update in lookahead.")
    parser.add_argument("--sigmoid_scale", default=5.0, type=float, help="Controls the steepness of the sigmoid function.")
    parser.add_argument("--sigmoid_shift", default=0.0, type=float, help="Controls the center point of the sigmoid function.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    parser.add_argument("--alpha", default=0.5, type=float, help="Alpha value for adaptive lookahead.")
    
    args = parser.parse_args()
    
    lr = [0.1]
    alpha_values = [0.5]
    for i in range(len(lr)):
        for j in range(len(alpha_values)):
            args.learning_rate = lr[i]
            args.alpha_max = alpha_values[j]
            seed = args.seed
            if args.alpha_max - 0.2 > 0.1:
                args.alpha_min = args.alpha_max - 0.2
        
            args.label = f"AdaptiveLookaheadv7_lr{args.learning_rate}_alpha{args.alpha_max}_seed{seed}"
            print(args.label)
            initialize(args, seed=seed)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Initialize the CIFAR dataset
            cifar = Cifar(args.batch_size, args.threads)
            
            # Split training (95%) and validation (5%)
            full_train_dataset = cifar.train.dataset
            train_size = int(0.95 * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=args.threads
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=args.threads
            )
            
            # Test set remains unchanged
            test_loader = cifar.test
            
            log = Log(filename=args.label, log_each=10)
            model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

            base_optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
            
            if args.method_type in ['lookahead', 'adaptive_lookahead']:
                optimizer = AdaptiveLookahead(base_optimizer, 
                                              alpha_min=args.alpha_min, 
                                              alpha_max=args.alpha_max, 
                                              k=args.lookahead_k)
                optimizer.current_alpha = args.alpha_max
                
                # (Optional) set sigmoid parameters if needed
                if hasattr(optimizer, 'sigmoid_scale'):
                    optimizer.sigmoid_scale = args.sigmoid_scale
                if hasattr(optimizer, 'sigmoid_shift'):
                    optimizer.sigmoid_shift = args.sigmoid_shift
            else:
                optimizer = base_optimizer
                
            scheduler = StepLR(base_optimizer, args.learning_rate, args.epochs)
            
            current_val_loss = None
            current_val_acc = None

            for epoch in range(args.epochs):
                # Training Phase
                model.train()
                log.train(len_dataset=len(train_loader.dataset))
                for batch in train_loader:
                    inputs, targets = (b.to(device) for b in batch)
                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                    
                    optimizer.zero_grad()
                    loss.mean().backward()
                    # In training loop, update optimizer without validation parameters.
                    optimizer.step()
                    
                    with torch.no_grad():
                        correct = torch.argmax(predictions, 1) == targets
                        log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                        scheduler(epoch)
                
                # Validation Phase (compute loss and accuracy)
                model.eval()
                val_losses = []
                val_correct = 0
                total_val = 0
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, targets = (b.to(device) for b in batch)
                        predictions = model(inputs)
                        loss = smooth_crossentropy(predictions, targets)
                        val_losses.append(loss.mean().item())
                        correct = (torch.argmax(predictions, 1) == targets).sum().item()
                        val_correct += correct
                        total_val += targets.size(0)
                current_val_loss = sum(val_losses) / len(val_losses) if val_losses else None
                current_val_acc = val_correct / total_val if total_val > 0 else None
                print(f"Epoch {epoch+1}/{args.epochs} - Validation Loss: {current_val_loss:.6f}, Val Acc: {current_val_acc:.4f}")
                
                # Update adaptive alpha based on epoch-level validation metrics
                if args.method_type in ['lookahead', 'adaptive_lookahead'] and current_val_loss is not None and current_val_acc is not None:
                    new_alpha = optimizer.update_alpha(current_val_loss, current_val_acc)
                    alpha_history.append(new_alpha)
                
                # Test Phase (optional)
                model.eval()
                test_losses = []
                log.eval(len_dataset=len(test_loader.dataset))
                test_correct = 0
                test_total = 0
                with torch.no_grad():
                    for batch in test_loader:
                        inputs, targets = (b.to(device) for b in batch)
                        predictions = model(inputs)
                        loss = smooth_crossentropy(predictions, targets)
                        test_losses.append(loss.mean().item())
                        pred = torch.argmax(predictions, 1)
                        test_correct += (pred == targets).sum().item()
                        log(model, loss.cpu(), (pred == targets).cpu())
                        test_total += targets.size(0)
                test_loss = sum(test_losses) / len(test_losses) if test_losses else 0
                test_accuracy = 100.0 * test_correct / test_total if test_total > 0 else 0
                print(f"Epoch {epoch+1}/{args.epochs} - Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.2f}%")
            
            log.flush()
            # Save training/validation plots
            log.save_loss_plot(log.train_losses, log.val_losses, filename=f'{args.label}_training_validation_loss.png')
            log.save_accuracy_plot(log.train_accuracies, log.val_accuracies, filename=f'{args.label}_training_validation_accuracy.png')
            plot_alpha_history(alpha_history, filename=f'{args.label}_alpha_history.png')
            # Print alpha statistics
            if alpha_history:
                print("Alpha Statistics:")
                print(f"  Initial: {alpha_history[0]:.4f}")
                print(f"  Final: {alpha_history[-1]:.4f}")
                print(f"  Min: {min(alpha_history):.4f}")
                print(f"  Max: {max(alpha_history):.4f}")
                print(f"  Mean: {sum(alpha_history)/len(alpha_history):.4f}")
