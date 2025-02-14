import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
import argparse
import sys
import os
# Add the parent folder to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

# Define OuterLoopLookahead Optimizer
class OuterLoopLookahead(Optimizer):
    def __init__(self, outer_optimizer, alpha=0.5, k=5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not k >= 1:
            raise ValueError(f"Invalid k: {k}")
        
        self.outer_optimizer = outer_optimizer
        self.alpha = alpha
        self.k = k
        self._step_count = 0
        
        # Copy of the slow parameters
        self.slow_params = []
        self.momentum_buffer = []
        
        for group in self.outer_optimizer.param_groups:
            sp, mb = [], []
            for p in group['params']:
                sp.append(p.clone().detach())
                mb.append(torch.zeros_like(p))
            self.slow_params.append(sp)
            self.momentum_buffer.append(mb)
    
    @property
    def param_groups(self):
        return self.outer_optimizer.param_groups
    
    def zero_grad(self):
        self.outer_optimizer.zero_grad()
    
    def step(self, closure=None):
        self._step_count += 1
        self.outer_optimizer.step(closure)
        
        if self._step_count % self.k == 0:
            for group_idx, group in enumerate(self.outer_optimizer.param_groups):
                for p_idx, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    
                    slow = self.slow_params[group_idx][p_idx]
                    momentum = self.momentum_buffer[group_idx][p_idx]
                    
                    # Compute the difference between fast and slow weights
                    delta = p.data - slow
                    
                    # Apply Lookahead Update
                    slow += self.alpha * delta
                    p.data.copy_(slow)
                    
                    # Store in momentum buffer for outer optimizer updates
                    momentum.mul_(0.9).add_(delta)
                    p.grad.add_(momentum)
                    
            # Ensure gradients are updated correctly before calling the outer optimizer step
            for group in self.outer_optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
                    
            self.outer_optimizer.step()
        
        return


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
    parser.add_argument("--label", default="Adaptive_K_Lookahead", type=str, help="Label for the experiment.")
    parser.add_argument("--method_type", default="lookahead", type=str, help="Label for the experiment.")
    parser.add_argument("--initial_k", default=5, type=int, help="Initial k value for AdaptiveLookahead.")
    parser.add_argument("--k_multiplier", default=5, type=int, help="Factor to multiply k when learning rate changes.")
    parser.add_argument("--alpha", default=0.5, type=float, help="Alpha parameter for AdaptiveLookahead.")
    parser.add_argument("--method", default="adaptive_increase", type=str, help="Method for adjusting k.")
    parser.add_argument("--seed", default=42, type=int, help="Seed for random number generators.")
    args = parser.parse_args()

    initialize(args, seed=45)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(args.batch_size, args.threads)
    print(dataset.train.shape)
    print(dataset)
    log = Log(filename=args.label, log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    # Initialize base_optimizer as SGD
    base_optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Use OuterLoopLookahead for Lookahead functionality
    optimizer = OuterLoopLookahead(
        base_optimizer,
        alpha=args.alpha,
        k=args.initial_k  # You can adjust this as needed
    )

    scheduler = StepLR(base_optimizer, args.learning_rate, args.epochs)

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
                    correct = torch.argmax(predictions.data, 1) == targets
                    log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                    scheduler(epoch)

        elif args.method_type == 'lookdeep':
            for batch in dataset.train:
                k = 5
                inputs, targets = (b.to(device) for b in batch)
                for i in range(k):
                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)

                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()

                    with torch.no_grad():
                        correct = torch.argmax(predictions.data, 1) == targets
                        log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                        scheduler(epoch)

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
    
    log.flush()
    log.save_loss_plot(log.train_losses, log.val_losses, filename='training_validation_loss.png')
    log.save_accuracy_plot(log.train_accuracies, log.val_accuracies, filename='training_validation_accuracy.png')
