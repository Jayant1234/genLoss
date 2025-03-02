import argparse
import torch
import sys
import os
from torch.utils.data import random_split, DataLoader

# Add the parent folder to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="Use Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total training epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing factor.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Initial learning rate.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum.")
    parser.add_argument("--threads", default=2, type=int, help="CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="Width factor for WideResNet.")
    parser.add_argument("--label", default="Baseline SGD", type=str, help="Experiment label.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    args = parser.parse_args()

    initialize(args, seed=args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = Cifar(args.batch_size, args.threads)
    full_train_dataset = dataset.train.dataset  # Extract dataset from DataLoader

    # Split into training (90%) and validation (10%) sets
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    test_loader = dataset.test  # Use original test set

    log = Log(filename=args.label, log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        # Training loop
        model.train()
        log.train(len_dataset=len(train_loader.dataset))

        for batch in train_loader:
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

        # Validation loop
        model.eval()
        log.eval(len_dataset=len(test_loader.dataset))

        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

    """ # Final testing loop
    model.eval()
    log.eval(len_dataset=len(test_loader.dataset))

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = (b.to(device) for b in batch)

            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            correct = torch.argmax(predictions, 1) == targets
            log(model, loss.cpu(), correct.cpu())"""

    log.flush()
    # Save loss and accuracy plots
    log.save_loss_plot(log.train_losses, log.val_losses, filename="training_validation_loss.png")
    log.save_accuracy_plot(log.train_accuracies, log.val_accuracies, filename="training_validation_accuracy.png")

