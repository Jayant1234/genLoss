import argparse
import torch
import sys

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

sys.path.append("..")
from sam import SAM  # Optional: if you want to use SAM

class GreedySoftWeightLookahead(torch.optim.Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=5, eval_func=None):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if k < 1:
            raise ValueError(f"Invalid k: {k}")

        self.base_optimizer = base_optimizer
        self.alpha = alpha
        self.k = k
        self.eval_func = eval_func  # Callback to evaluate current fast weights on validation set
        self._step_count = 0

        # Initialize slow weights and best slow weights (deep copies)
        self.slow_params = []
        self.best_slow_params = []
        for group in self.base_optimizer.param_groups:
            sp = []
            bsp = []
            for p in group['params']:
                sp.append(p.clone().detach())
                bsp.append(p.clone().detach())
            self.slow_params.append(sp)
            self.best_slow_params.append(bsp)
        
        # Buffer for best accuracy (based on slow weights)
        self.best_acc = 0.0

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self, closure=None):
        """
        1. Perform one fast update via the base optimizer.
        2. Every k steps, evaluate the fast weights:
           - If they have better accuracy than the best recorded,
             update the best slow weights buffer.
           - Otherwise, interpolate the fast weights with the best slow weights.
        """
        loss = self.base_optimizer.step(closure)
        self._step_count += 1

        if self._step_count % self.k == 0:
            if self.eval_func is not None:
                current_acc = self.eval_func()
                if current_acc > self.best_acc:
                    # Fast weights are better: update best accuracy and best slow weights buffer
                    self.best_acc = current_acc
                    for group_idx, group in enumerate(self.base_optimizer.param_groups):
                        for p_idx, p in enumerate(group['params']):
                            self.best_slow_params[group_idx][p_idx] = self.slow_params[group_idx][p_idx].clone().detach()
                else:
                    # Fast weights did not improve: interpolate with best slow weights
                    for group_idx, group in enumerate(self.base_optimizer.param_groups):
                        for p_idx, p in enumerate(group['params']):
                            best_slow = self.best_slow_params[group_idx][p_idx]
                            # Interpolate: new_fast = best_slow + alpha * (fast - best_slow)
                            p.data.copy_(best_slow + self.alpha * (p.data - best_slow))
                            # Update the slow weights to reflect the new fast weights
                            self.slow_params[group_idx][p_idx].copy_(p.data)
            else:
                # If no evaluation function is provided, perform a standard Lookahead update
                for group_idx, group in enumerate(self.base_optimizer.param_groups):
                    for p_idx, p in enumerate(group['params']):
                        if p.grad is None:
                            continue
                        slow = self.slow_params[group_idx][p_idx]
                        slow += self.alpha * (p.data - slow)
                        p.data.copy_(slow)
        return loss

def evaluate_model(model, val_loader, device):
    """
    Evaluate the model on the validation set.
    Returns the accuracy (as a float between 0 and 1).
    """
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
    model.train()
    return total_correct / total

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
    parser.add_argument("--label", default="Lookahead", type=str, help="Label for the experiment.")
    parser.add_argument("--method_type", default="lookahead", type=str, help="Method type: 'lookahead' or 'lookdeep'.")
    args = parser.parse_args()

    # Initialize experiment
    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Prepare dataset, log, and model
    dataset = Cifar(args.batch_size, args.threads)
    log = Log(filename=args.label, log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    # Define an evaluation function for the Lookahead optimizer
    def eval_func():
        return evaluate_model(model, dataset.test, device)

    # Create the base optimizer and then wrap it with Lookahead
    base_optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    optimizer = GreedySoftWeightLookahead(base_optimizer, alpha=0.5, k=10, eval_func=eval_func)
    scheduler = StepLR(base_optimizer, args.learning_rate, args.epochs)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))
        
        if args.method_type == 'lookahead':
            for batch in dataset.train:
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
            for batch in dataset.train:
                k = 5  # Number of inner lookdeep updates
                inputs, targets = (b.to(device) for b in batch)
                for i in range(k):
                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()
                    
                    with torch.no_grad():
                        correct = (torch.argmax(predictions, 1) == targets)
                        log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                        scheduler(epoch)

        # Evaluation on the test set
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
    # Save plots after training
    log.save_loss_plot(log.train_losses, log.val_losses, filename='training_validation_loss.png')
    log.save_accuracy_plot(log.train_accuracies, log.val_accuracies, filename='training_validation_accuracy.png')
