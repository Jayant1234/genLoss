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

class AdaptiveLookahead(torch.optim.Optimizer):
    def __init__(self, base_optimizer, alpha_min=0.1, alpha_max=0.9, k=5):
        if not 0.0 <= alpha_min <= alpha_max <= 1.0:
            raise ValueError(f"Invalid alpha range: {alpha_min} to {alpha_max}")
        if not k >= 1:
            raise ValueError(f"Invalid k: {k}")

        self.base_optimizer = base_optimizer
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.k = k

        # Track number of "fast" updates so we know when to do the slow update
        self._step_count = 0
        
        # For adaptive alpha calculation
        self.val_loss_history = []
        self.best_val_loss = float('inf')
        self.running_val_loss = None
        self.window_size = 5  # For running average calculation
        self.current_alpha = (alpha_min + alpha_max) / 2  # Start with middle value
        
        # For loss normalization
        self.loss_min = float('inf')
        self.loss_max = float('-inf')
        self.loss_mean = 0
        self.loss_std = 1
        self.num_samples = 0
        
        # Sigmoid parameters for better control
        self.sigmoid_scale = 5.0  # Controls the steepness of the sigmoid
        self.sigmoid_shift = 0.0  # Controls the center point of the sigmoid

        # Copy of the fast params to "slow" buffer
        self.slow_params = []
        for group in self.base_optimizer.param_groups:
            sp = []
            for p in group['params']:
                sp.append(p.clone().detach())
            self.slow_params.append(sp)

    def sigmoid(self, x):
        """Sigmoid function with adjustable scale and shift parameters"""
        return 1 / (1 + np.exp(-self.sigmoid_scale * (x - self.sigmoid_shift)))

    def normalize_loss(self, loss):
        """
        Normalize loss values using running statistics to center the data
        so positive and negative fluctuations can be captured accurately
        """
        # Update running statistics
        self.num_samples += 1
        delta = loss - self.loss_mean
        self.loss_mean += delta / self.num_samples
        
        # Use Welford's algorithm for stable variance calculation
        if self.num_samples > 1:
            delta2 = loss - self.loss_mean
            self.loss_std = np.sqrt(((self.num_samples - 1) * (self.loss_std ** 2) + delta * delta2) / self.num_samples)
        
        # Update min/max values
        self.loss_min = min(self.loss_min, loss)
        self.loss_max = max(self.loss_max, loss)
        
        # Apply z-score normalization if we have enough samples
        if self.num_samples > 1 and self.loss_std > 0:
            return (loss - self.loss_mean) / self.loss_std
        else:
            return 0.0  # Default to neutral value if not enough data

    def update_alpha(self, val_loss):
        """Update alpha based on validation loss metrics with improved normalization"""
        # Record the new validation loss
        self.val_loss_history.append(val_loss)
        
        # Update best validation loss if needed
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        
        # Calculate running average of validation loss
        if len(self.val_loss_history) >= self.window_size:
            self.running_val_loss = sum(self.val_loss_history[-self.window_size:]) / self.window_size
        else:
            self.running_val_loss = sum(self.val_loss_history) / len(self.val_loss_history)
        
        # No update if we don't have enough history
        if len(self.val_loss_history) < 2 or self.running_val_loss <= 0 or self.best_val_loss <= 0:
            return self.current_alpha
        
        # Calculate improvement ratio
        improvement_ratio = self.best_val_loss / self.running_val_loss
        
        # Normalize the improvement ratio to handle fluctuations better
        normalized_improvement = self.normalize_loss(improvement_ratio)
        
        # Calculate a "trend" factor based on recent loss changes
        if len(self.val_loss_history) >= 3:
            # Positive trend means loss is decreasing (good)
            recent_trend = (self.val_loss_history[-3] - self.val_loss_history[-1]) / max(self.val_loss_history[-3], 1e-8)
            # Normalize the trend
            normalized_trend = self.normalize_loss(recent_trend)
            
            # Combine improvement ratio with trend
            combined_signal = 0.7 * normalized_improvement + 0.3 * normalized_trend
        else:
            combined_signal = normalized_improvement
        
        # Apply sigmoid to map to (0,1) range with controlled scaling
        sigmoid_value = self.sigmoid(combined_signal)
        
        # Calculate new alpha
        new_alpha = self.alpha_min + sigmoid_value * (self.alpha_max - self.alpha_min)
        
        # Add damping to prevent large oscillations in alpha
        damping_factor = 0.8
        self.current_alpha = damping_factor * self.current_alpha + (1 - damping_factor) * new_alpha
        
        # Print diagnostics with more information
        print("\n")            
        print(f"Val Loss: {val_loss:.6f}, Running Avg: {self.running_val_loss:.6f}, Best: {self.best_val_loss:.6f}")
        print(f"Signal: {combined_signal:.4f}, Sigmoid: {sigmoid_value:.4f}, Alpha: {self.current_alpha:.4f}")
        
        return self.current_alpha

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self, closure=None, val_loss=None):
        """
        1. Perform one 'fast' step with the base optimizer
        2. Every k steps, update slow weights with adaptive alpha
        """
        loss = self.base_optimizer.step(closure)
        self._step_count += 1

        # If validation loss is provided, update alpha
        if val_loss is not None:
            self.update_alpha(val_loss)

        if self._step_count % self.k == 0:
            # Slow update with adaptive alpha
            for group_idx, group in enumerate(self.base_optimizer.param_groups):
                for p_idx, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    slow = self.slow_params[group_idx][p_idx]
                    # slow <- slow + adaptive_alpha * (fast - slow)
                    slow += self.current_alpha * (p.data - slow)
                    # Then copy back to fast parameters
                    p.data.copy_(slow)

        return loss

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
    parser.add_argument("--label", default="AdaptiveLookahead", type=str, help="Label for the experiment.")
    parser.add_argument("--method_type", default="adaptive_lookahead", type=str, help="Method type for training.")
    parser.add_argument("--alpha_min", default=0.1, type=float, help="Minimum alpha value for adaptive lookahead.")
    parser.add_argument("--alpha_max", default=0.9, type=float, help="Maximum alpha value for adaptive lookahead.")
    parser.add_argument("--lookahead_k", default=10, type=int, help="Number of fast updates before slow update in lookahead.")
    parser.add_argument("--sigmoid_scale", default=5.0, type=float, help="Controls the steepness of the sigmoid function.")
    parser.add_argument("--sigmoid_shift", default=0.0, type=float, help="Controls the center point of the sigmoid function.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    parser.add_argument("--alpha", default=0.5, type=float, help="Alpha value for adaptive lookahead.")
    


    args = parser.parse_args()
    
    lr = [0.1]
    alpha = [0.5]
    for i in range(len(lr)):
        for j in range(len(alpha)):
            args.learning_rate = lr[i]
            args.alpha_max = alpha[j]
            seed = args.seed
            args.label = f"{args.label}_{args.learning_rate}_alpha{args.alpha_max}_seed{seed}"
            print(args.label)
            initialize(args, seed=seed)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Initialize the Cifar dataset
            cifar = Cifar(args.batch_size, args.threads)
            
            # Extract the full training dataset from the DataLoader
            full_train_dataset = cifar.train.dataset
            
            # Split into training (95%) and validation (5%) sets
            train_size = int(0.95 * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
            
            # Create new DataLoaders for the split datasets
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
            
            # Keep the original test set
            test_loader = cifar.test
            
            log = Log(filename=args.label, log_each=10)
            model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

            base_optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
            
            if args.method_type == 'lookahead' or args.method_type == 'adaptive_lookahead':
                optimizer = AdaptiveLookahead(base_optimizer, 
                                        alpha_min=args.alpha_min, 
                                        alpha_max=args.alpha_max, 
                                        k=args.lookahead_k)
                
                # Set sigmoid parameters if provided
                if hasattr(optimizer, 'sigmoid_scale'):
                    optimizer.sigmoid_scale = args.sigmoid_scale
                if hasattr(optimizer, 'sigmoid_shift'):
                    optimizer.sigmoid_shift = args.sigmoid_shift
            else:
                optimizer = base_optimizer
                
            scheduler = StepLR(base_optimizer, args.learning_rate, args.epochs)
            
            # Track validation loss for adaptive alpha
            current_val_loss = None

            for epoch in range(args.epochs):
                # Training phase
                model.train()
                log.train(len_dataset=len(train_loader.dataset))

                if args.method_type in ['lookahead', 'adaptive_lookahead']: 
                    for batch in train_loader:
                        inputs, targets = (b.to(device) for b in batch)

                        predictions = model(inputs)
                        loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                        
                        optimizer.zero_grad()
                        loss.mean().backward()
                        
                        # Pass current validation loss to optimizer if available
                        if current_val_loss is not None and hasattr(optimizer, 'update_alpha'):
                            optimizer.step(val_loss=current_val_loss)
                        else:
                            optimizer.step()

                        with torch.no_grad():
                            correct = torch.argmax(predictions.data, 1) == targets
                            log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                            scheduler(epoch)

                elif args.method_type == 'lookdeep': 
                    pass
                    # for batch in train_loader:
                    #     k = 5
                    #     inputs, targets = (b.to(device) for b in batch)
                    #     for i in range(k): 
                    #         predictions = model(inputs)
                    #         loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                            
                    #         optimizer.zero_grad()
                    #         loss.mean().backward()
                    #         optimizer.step()

                    #         with torch.no_grad():
                    #             correct = torch.argmax(predictions.data, 1) == targets
                    #             log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                    #             scheduler(epoch)

                # Validation phase
                model.eval()
                # log.eval(len_dataset=len(val_loader.dataset))
                val_losses = []

                with torch.no_grad():
                    for batch in val_loader:
                        inputs, targets = (b.to(device) for b in batch)

                        predictions = model(inputs)
                        loss = smooth_crossentropy(predictions, targets)
                        val_losses.append(loss.mean().item())
                        correct = torch.argmax(predictions, 1) == targets
                        # log(model, loss.cpu(), correct.cpu())
                
                # Update current validation loss for next epoch
                current_val_loss = sum(val_losses) / len(val_losses) if val_losses else None
                print(f"Epoch {epoch+1}/{args.epochs} - Validation Loss: {current_val_loss:.6f}")
                
                
                # Test phase (optional)
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
                        correct = torch.argmax(predictions, 1) == targets
                        log(model, loss.cpu(), correct.cpu())
                        test_total += targets.size(0)
                
                test_loss = sum(test_losses) / len(test_losses) if test_losses else 0
                test_accuracy = 100.0 * test_correct / test_total if test_total > 0 else 0
                print(f"Epoch {epoch+1}/{args.epochs} - Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.2f}%")
            
            log.flush()
            # Save the plots after all epochs
            log.save_loss_plot(log.train_losses, log.val_losses, filename=f'{args.label}_training_validation_loss.png')
            log.save_accuracy_plot(log.train_accuracies, log.val_accuracies, filename=f'{args.label}_training_validation_accuracy.png')
