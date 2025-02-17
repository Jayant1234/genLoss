# Run SGDM(SGD with momentum) baseline on training set 90%, validation set 10% and testing set.
import argparse
import torch
import sys
import os
from torch.utils.data import DataLoader, random_split

# Add the parent folder to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR

def evaluate(model, data_loader, device):
    """Evaluates the model on the given dataset and returns accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size for training.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Initial learning rate.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="Width factor for WideResNet.")
    parser.add_argument("--label", default="Baseline_SGD_Momentum", type=str, help="Experiment label.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    args = parser.parse_args()

    initialize(args, seed=args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = Cifar(args.batch_size, args.threads)
    full_train_dataset = dataset.train.dataset  # Extract dataset from DataLoader

    # Split dataset: 90% training, 10% validation
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    test_loader = DataLoader(dataset.test.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)

    # Logging and model setup
    log = Log(filename=args.label, log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(train_loader))

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss.mean().backward()
            optimizer.step()

            with torch.no_grad():
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                
        # Scheduler step
        scheduler(epoch)

        # Validation accuracy
        val_accuracy = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} - Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Final evaluation on test set
    test_accuracy = evaluate(model, test_loader, device)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

    log.flush()
    log.save_loss_plot(log.train_losses, log.val_losses, filename='training_validation_loss.png')
    log.save_accuracy_plot(log.train_accuracies, log.val_accuracies, filename='training_validation_accuracy.png')

