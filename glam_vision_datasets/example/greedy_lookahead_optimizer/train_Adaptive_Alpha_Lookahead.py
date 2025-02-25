import argparse
import torch
import sys
import math
from torch.utils.data import DataLoader, random_split

# Add parent directory to path so that our modules can be imported.
sys.path.append("..")
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

class AdaptiveAlphaLookahead(torch.optim.Optimizer):
    """
    This optimizer wraps a base optimizer (e.g. SGD) and every k steps:
      - Evaluates the current model on the validation set (via eval_func) to obtain
        a validation loss.
      - Updates a running average of the validation loss and the best validation loss so far.
      - Computes an adaptive interpolation parameter α using a sigmoid:
            α = 1 / (1 + exp(-γ (L_best - L_run)))
        where γ = 2.944 / Δ* (with Δ* a characteristic difference, e.g. 0.1).
      - Uses α to interpolate the fast (current) weights toward the slow (stable) weights.
    """
    def __init__(self, base_optimizer, k=5, eval_func=None, beta=0.9, delta_star=0.1):
        # base_optimizer: underlying optimizer (e.g. SGD)
        # k: number of steps between adaptive lookahead updates
        # eval_func: function that returns current validation loss (float)
        # beta: smoothing factor for the running average (EMA) of validation loss
        # delta_star: characteristic loss difference for normalization
        self.base_optimizer = base_optimizer
        self.k = k
        self.eval_func = eval_func  # should return the validation loss (float)
        self._step_count = 0
        
        # Adaptive α parameters:
        self.beta = beta
        self.delta_star = delta_star
        self.gamma = 2.944 / self.delta_star  # so that at Δ = delta_star, α ≈ 0.95
        self.running_val_loss = float('inf')  # initialize running average
        self.best_val_loss = float('inf')     # initialize best loss
        
        # Initialize slow weights (deep copies of the fast weights)
        self.slow_params = []
        for group in self.base_optimizer.param_groups:
            sp = []
            for p in group['params']:
                sp.append(p.clone().detach())
            self.slow_params.append(sp)
    
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _update_adaptive_alpha(self):
        # Evaluate current validation loss (using the provided eval_func)
        current_val_loss = self.eval_func()
        
        # If running average is not yet set, initialize it.
        if self.running_val_loss == float('inf'):
            self.running_val_loss = current_val_loss
        else:
            self.running_val_loss = self.beta * self.running_val_loss + (1 - self.beta) * current_val_loss
        
        # Update the best validation loss seen so far.
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
        
        # Compute difference Δ between best and running average.
        delta = self.best_val_loss - self.running_val_loss
        # Compute adaptive α via the sigmoid:
        adaptive_alpha = 1.0 / (1.0 + math.exp(-self.gamma * delta))
        
        # Print the adaptive α and related values for debugging/monitoring.
        print(f"Adaptive alpha: {adaptive_alpha:.4f} | Δ: {delta:.4f} | current_val_loss: {current_val_loss:.4f} | running_val_loss: {self.running_val_loss:.4f} | best_val_loss: {self.best_val_loss:.4f}")
        return adaptive_alpha

    def step(self, closure=None):
        # Perform one fast update using the base optimizer.
        loss = self.base_optimizer.step(closure)
        self._step_count += 1
        
        # Every k steps, update the slow weights using the adaptive α.
        if self._step_count % self.k == 0:
            adaptive_alpha = self._update_adaptive_alpha()
            for group_idx, group in enumerate(self.base_optimizer.param_groups):
                for p_idx, p in enumerate(group['params']):
                    best_slow = self.slow_params[group_idx][p_idx]
                    # Interpolate: new_fast = best_slow + α * (fast - best_slow)
                    p.data.copy_(best_slow + adaptive_alpha * (p.data - best_slow))
                    # Update the stored slow weights to reflect the new fast weights.
                    self.slow_params[group_idx][p_idx].copy_(p.data)
        return loss

def evaluate_model_loss(model, loader, device):
    """
    Evaluate the model on the given loader (validation set) and return the average loss.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = smooth_crossentropy(outputs, targets)
            loss_val = loss.mean().item()  # ensure a scalar
            total_loss += loss_val * targets.size(0)
            total_samples += targets.size(0)
    model.train()
    return total_loss / total_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="Use the adaptive Lookahead variant.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size for training/validation.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers in the network.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of training epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing value (0 for none).")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Initial learning rate.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for data loading.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter (if using SAM variants).")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="Network width factor compared to normal ResNet.")
    parser.add_argument("--label", default="AdaptiveAlphaLookahead", type=str, help="Label for the experiment/log files.")
    parser.add_argument("--method_type", default="lookahead", type=str, help="Either 'lookahead' or 'lookdeep'.")
    parser.add_argument("--k_steps", default=5, type=int, help="k: number of fast steps before lookahead update.")
    parser.add_argument("--beta", default=0.9, type=float, help="EMA beta for running validation loss.")
    parser.add_argument("--delta_star", default=0.1, type=float, help="Characteristic Δ for adaptive alpha normalization.")
    args = parser.parse_args()

    # Initialize experiment (sets seeds, logging, etc.) and device.
    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load CIFAR dataset.
    dataset = Cifar(args.batch_size, args.threads)
    full_train_dataset = dataset.train.dataset  # Extract underlying dataset.

    # Split into training (95%) and validation (5%) subsets.
    train_size = int(0.95 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    
    # Set up logging and the model.
    log = Log(filename=args.label, log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    # Define an evaluation function that returns the current validation loss.
    def eval_func():
        return evaluate_model_loss(model, val_loader, device)

    # Create a base optimizer (SGD) and wrap it with the adaptive lookahead optimizer.
    base_optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    k = args.k_steps  # Number of fast steps before updating slow weights.
    optimizer = AdaptiveAlphaLookahead(
        base_optimizer,
        k=k,
        eval_func=eval_func,
        beta=args.beta,
        delta_star=args.delta_star,
    )
    scheduler = StepLR(base_optimizer, args.learning_rate, args.epochs)

    # Training loop.
    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(train_loader))
        
        if args.method_type == 'lookahead':
            for batch in train_loader:
                inputs, targets = (b.to(device) for b in batch)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()

                with torch.no_grad():
                    correct = (torch.argmax(predictions, 1) == targets)
                    log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                    scheduler(epoch)

        elif args.method_type == 'lookdeep':
            for batch in train_loader:
                inner_steps = 5  # Number of inner lookdeep updates.
                inputs, targets = (b.to(device) for b in batch)
                for i in range(inner_steps):
                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()
                    
                    with torch.no_grad():
                        correct = (torch.argmax(predictions, 1) == targets)
                        log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                        scheduler(epoch)

        # Evaluate on the test set at the end of each epoch.
        model.eval()
        log.eval(len_dataset=len(dataset.test))
        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = (torch.argmax(predictions, 1) == targets)
                log(model, loss.cpu(), correct.cpu())
    
    log.flush()
    # Save loss and accuracy plots.
    log.save_loss_plot(log.train_losses, log.val_losses, filename='training_validation_loss.png')
    log.save_accuracy_plot(log.train_accuracies, log.val_accuracies, filename='training_validation_accuracy.png')
