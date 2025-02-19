import argparse
import sys
import os
# Add the parent folder to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import argparse
import torch
import os
import sys
import copy
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.optim import SGD
from utility.log import Log
from utility.step_lr import StepLR
from model.wide_res_net import WideResNet
from data.cifar import Cifar
from utility.initialize import initialize
from model.smooth_cross_entropy import smooth_crossentropy


def train():
    # Parse arguments
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

    # Initialize the experiment
    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = Cifar(args.batch_size, args.threads)
    full_train_dataset = dataset.train.dataset  # Extract the dataset from DataLoader

    # Split dataset into training (90%) and validation (10%)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    
    # Logging and model setup
    log = Log(filename=args.label, log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    # Initialize base_optimizer as SGD
    base_optimizer = SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Loss function (criterion) and scheduler
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(base_optimizer, args.learning_rate, args.epochs)

    # Greedy lookahead parameters
    k = 10  # Number of batches before performing validation
    current_weights_fraction = 0.5  # Fraction of the current model weights
    best_model_weights_fraction = 0.5  # Fraction of the best model weights
    best_val_accuracy = 0.0  # Best validation accuracy so far
    best_model_weights = None  # Best model weights buffer
    batch_count = 0  # Batch count for periodic validation check
    epoch_step = 0
    # Training loop
    for epoch in range(args.epochs):
        epoch_step += 1
        model.train()
        log.train(len_dataset=len(train_loader))

        for batch in train_loader:
            inputs, targets = (b.to(device) for b in batch)

            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)

            base_optimizer.zero_grad()
            loss.mean().backward()

            # Regular SGD optimization step
            base_optimizer.step()

            # Increment batch count
            batch_count += 1

            # Periodic validation check
            if batch_count % k == 0 and epoch_step > 130:
                # Evaluate the model on the validation set
                current_val_accuracy = evaluate(model, val_loader, criterion, device)
                
                # If the current model is better than the best model, update the best model weights and accuracy
                if current_val_accuracy > best_val_accuracy:
                    best_val_accuracy = current_val_accuracy
                    best_model_weights = copy.deepcopy(model.state_dict())
                else:
                    # If the current model is not better, average the current weights with the best weights
                    average_weights(model, best_model_weights,current_weight_fraction=current_weights_fraction,best_weight_fraction=best_model_weights_fraction)

            # Log the loss and accuracy for the current batch
            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        # Evaluation on validation set at the end of each epoch
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


def evaluate(model, val_loader, criterion, device):
    """Evaluates the model on the validation set and returns the accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            # Move inputs and targets to the same device as the model
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    model.train()
    return accuracy


def average_weights(model, best_model_weights, current_weight_fraction=0.5, best_weight_fraction=0.5):
    """
    Averages the current model weights with the best model weights, and loads the result back into the model.
    
    Parameters:
    - model: The current model.
    - best_model_weights: The best model weights.
    - current_weight_fraction: The fraction of the current model weights (in [0, 1]).
    - best_weight_fraction: The fraction of the best model weights (in [0, 1]).
    """
    assert current_weight_fraction + best_weight_fraction == 1.0, "The fractions must sum to 1."
    
    current_weights = model.state_dict()
    for key in current_weights.keys():
        current_weights[key] = current_weight_fraction * current_weights[key] + best_weight_fraction * best_model_weights[key]
    model.load_state_dict(current_weights)


if __name__ == "__main__":
    train()

