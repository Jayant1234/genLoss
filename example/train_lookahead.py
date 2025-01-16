
import argparse
import torch

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
import sys; sys.path.append("..")
from sam import SAM

import torch

class Lookahead(torch.optim.Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=5, lk_momentum=0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not k >= 1:
            raise ValueError(f"Invalid k: {k}")

        if not 0.0 <= lk_momentum < 1.0:
            raise ValueError(f"Invalid momentum: {lk_momentum}")

        self.base_optimizer = base_optimizer
        self.alpha = alpha
        self.k = k
        self.lk_momentum = lk_momentum
        # Track number of "fast" updates so we know when to do the slow update
        self._step_count = 0

        # Copy of the fast params to “slow” buffer
        self.slow_params = []
        self.momentum_buffer = []

        for group in self.base_optimizer.param_groups:
            sp = []
            mb = []
            for p in group['params']:
                sp.append(p.clone().detach())
                mb.append(torch.zeros_like(p))
            self.slow_params.append(sp)
            self.momentum_buffer.append(mb)

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self, closure=None):
        """
        1. Perform one 'fast' step with the base optimizer
        2. Every k steps, update slow weights
        """
        loss = self.base_optimizer.step(closure)
        self._step_count += 1
        
        if self._step_count % self.k == 0:
            
            # Slow update
            with torch.no_grad():
                for group_idx, group in enumerate(self.base_optimizer.param_groups):
                    for p_idx, p in enumerate(group['params']):
                        #if p.grad is None:
                            #continue # this is a bug since it causes model to not update some param that may have values changed from previous gradients.
                        
                        slow = self.slow_params[group_idx][p_idx]
                        momentum = self.momentum_buffer[group_idx][p_idx]
                        
                        # Compute the difference between fast and slow weights
                        delta = p.data - slow

                        # Update the momentum buffer in place
                        momentum.mul_(self.lk_momentum).add_(delta)

                        # Update the slow weights with the momentum buffer
                        slow.add_(self.alpha * momentum)
                        # # slow <- slow + alpha * (fast - slow)
                        # slow += self.alpha * (p.data - slow)
                        # Then copy back to fast parameters
                        p.data.copy_(slow)

        return loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--lk_momentum", default=0.0, type=float, help="Momentum for lookahead")

    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--label", default="Baseline SGD", type=str, help="Label for the experiment.")
    parser.add_argument("--k", default=5, type=int, help="k for the lookahead")
    parser.add_argument("--alpha", default=0.5, type=float, help="k for the lookahead")

    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(args.batch_size, args.threads)
    log = Log(filename=args.label,log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    # base_optimizer = torch.optim.SGD
    # optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    base_optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    optimizer = Lookahead(base_optimizer, alpha=args.alpha, k=args.k, lk_momentum=args.lk_momentum)
    scheduler = StepLR(base_optimizer, args.learning_rate, args.epochs)
    
    epoch=0
    while (epoch<=args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            # # first forward-backward step
            # enable_running_stats(model)

            # Here i Will use the simple sgd optimizer
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            # print("Predictions Shape: ", predictions.shape)
            # print("Targets Shape: ", targets.shape)
            # print("Loss Shape Before: ", loss.shape)
            
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                # print("Loss Shape After: ", loss.shape)
                # print("Correct Shape: ", correct.shape)
                # print("Targets Shape: ", targets.shape)
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)
        epoch+=1

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
    # Save the plots after all epochs
    log.save_loss_plot(log.train_losses, log.val_losses, filename='training_validation_loss.png')
    log.save_accuracy_plot(log.train_accuracies, log.val_accuracies, filename='training_validation_accuracy.png')
    