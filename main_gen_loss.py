import math
from argparse import ArgumentParser
from itertools import permutations
import copy

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from grokfast import *


class Block(nn.Module):
    """Causal transformer block
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
        attn_mask[torch.isnan(attn_mask)] = 0.0 # fixes all 'nan' on 'mps' device

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class Decoder(nn.Module):
    """Causal Transformer decoder
    """

    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=97, seq_len=5):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(dim, num_heads))

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)

    def forward(self, x):
        h = self.token_embeddings(x)
        positions = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)
        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits


def multiplication_mod_p_data(p, eq_token, op_token):
    """x◦y = x/y (mod p) for 0 ≤ x < p, 0 < y < p
    """
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    result = x * y % p

    # "All of our experiments used a small transformer trained on datasets of
    # equations of the form a◦b = c, where each of “a”, “◦”, “b”, “=”, and “c”
    # is a seperate token"
    return torch.stack([x, op, y, eq, result])

def split_data_into_10_parts(data):
    # Shuffle the data indices
    indices = torch.randperm(data.shape[1])
    
    # Split the indices into 10 roughly equal parts
    parts = torch.split(indices, data.shape[1] // 10)
    parts = list(parts)

    # If there are any remaining indices (because data.shape[1] may not be perfectly divisible by 10),
    # add them to the last part
    if len(parts) > 10:
        parts[-2] = torch.cat((parts[-2], parts[-1]))
        parts = parts[:-1]
    
    # Create a dictionary to label each part
    labeled_parts = {f"part_{i+1}": parts[i] for i in range(len(parts))}
    
    return labeled_parts

def train_progressive(model, parts, data, optimizer, scheduler, device, lambda_weight):
    criterion = nn.CrossEntropyLoss()
    
    cumulative_indices = torch.tensor([], dtype=torch.long)
    total_steps = 0  # Track the total number of steps taken
    
    # Use the last part as the validation set
    validation_indices = parts[f"part_{10}"]
    validation_data = data[:, validation_indices]
    
    # Separate features and labels for validation data
    features_validation = validation_data[:4].T.to(device)
    labels_validation = validation_data[4].to(device)
    
    # Remove the last part from the training parts
    training_parts = {k: parts[k] for k in list(parts.keys())[:-1]}
    
    # Containers to save training and validation metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    its = []
    max_epochs=500
    for i in range(1, len(training_parts) + 1):
        # Accumulate parts
        print(f"Accumulating data for Part {i}")
        cumulative_indices = torch.cat((cumulative_indices, training_parts[f"part_{i}"]))
        cumulative_data = data[:, cumulative_indices]
        new_data_indices = training_parts[f"part_{i}"]
        new_data = data[:, new_data_indices]
        print(f"Cumulative data shape after adding Part {i}: {cumulative_data.shape}")
        
        # Separate features and labels (assuming x, y -> result as features -> label)
        features_cumulative = cumulative_data[:4].T.to(device)  # Using x, op, y as features
        labels_cumulative = cumulative_data[4].to(device)       # Using result as label
        
        features_new = new_data[:4].T.to(device)  # Features from newly added part
        labels_new = new_data[4].to(device)       # Labels from newly added part

        # Create matching sizes for old data and new data by repeating the smaller dataset
        if features_cumulative.shape[0] > features_new.shape[0] and features_new.shape[0] > 0:
            repeats = features_cumulative.shape[0] // features_new.shape[0] + 1
            features_new_repeated = features_new.repeat(repeats, 1)[:features_cumulative.shape[0]]
            labels_new_repeated = labels_new.repeat(repeats)[:features_cumulative.shape[0]]
        else:
            features_new_repeated = features_new
            labels_new_repeated = labels_new
                
        print(f"Features shape: {features_cumulative.shape}, Labels shape: {labels_cumulative.shape}")
        
        # Training loop for the current cumulative dataset
        for epoch in range(max_epochs):  # Set max_epochs or use a stopping condition like a small training loss
            model.train()

            # Shuffle indices for each epoch to get new batches
            shuffled_indices = torch.randperm(features_cumulative.shape[0])
            features_cumulative = features_cumulative[shuffled_indices]
            labels_cumulative = labels_cumulative[shuffled_indices]

            batch_size = args.batch_size
            num_batches = math.ceil(features_cumulative.shape[0] / batch_size)
            print("num_batch",num_batches)
            print("batch_size",batch_size)
            for batch_idx in range(num_batches):
                # Calculate the start and end indices for the current batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, features_cumulative.shape[0])  # Prevent out-of-bounds indexing

                # Extract the features and labels for the current batch
                batch_features_cumulative = features_cumulative[start_idx:end_idx]
                batch_labels_cumulative = labels_cumulative[start_idx:end_idx]
                print("Shape of batch_labels_cumulative, batch_features_cumulative:",batch_labels_cumulative.shape, batch_features_cumulative.shape)

                optimizer.zero_grad()

                # Calculate losses
                outputs_cumulative = model(batch_features_cumulative.T)
                print("Shape of outputs cumulative, batch_features_cumulative, batch cumulative:",outputs_cumulative.shape, batch_features_cumulative.shape, batch_labels_cumulative.shape)
                # Reshape the output and labels for cross-entropy
                outputs_cumulative = outputs_cumulative.permute(1, 0, 2).reshape(-1, outputs_cumulative.shape[-1])  # Shape: (batch_size * seq_len, num_tokens)
                batch_labels_cumulative = batch_labels_cumulative.repeat(4)  # Shape: (batch_size * seq_len)

                loss_cumulative = criterion(outputs_cumulative, batch_labels_cumulative)

                outputs_new = model(features_new_repeated.T)
                # Reshape the output and labels for cross-entropy
                outputs_new = outputs_new.permute(1, 0, 2).reshape(-1, outputs_new.shape[-1])  # Shape: (batch_size * seq_len, num_tokens)
                labels_new_repeated = labels_new_repeated.repeat(4)

                generalization_loss = criterion(outputs_new, labels_new_repeated)

                # Combined loss
                combined_loss = loss_cumulative + lambda_weight * generalization_loss
                combined_loss.backward()
                optimizer.step()
                scheduler.step()
                total_steps += 1

                print(f"Part {i}, Epoch {epoch+1}, Step {total_steps}, Batch {batch_idx+1}/{num_batches}, Loss: {combined_loss.item():.4f}, Loss Cumulative: {loss_cumulative.item():.4f}, Generalization Loss: {generalization_loss.item():.4f}")

                # Save training metrics
                train_losses.append(loss_cumulative.item())
                train_accuracies.append((outputs_cumulative.argmax(dim=-1) == batch_labels_cumulative).float().mean().item())
                its.append(total_steps)

            # Stop training for this part when the cumulative loss is zero (or very close)
            if loss_cumulative.item() < 1e-6:
                break

        
            # Validation
            model.eval()
            val_loss = 0
            val_accuracy = 0
            val_num_batches = math.ceil(features_validation.shape[0] / batch_size)

            with torch.no_grad():
                for val_batch_idx in range(val_num_batches):
                    val_batch_features = features_validation[val_batch_idx * batch_size : (val_batch_idx + 1) * batch_size]
                    val_batch_labels = labels_validation[val_batch_idx * batch_size : (val_batch_idx + 1) * batch_size]

                    val_outputs = model(val_batch_features)
                    val_batch_loss = criterion(val_outputs, val_batch_labels.long())
                    val_loss += val_batch_loss.item()
                    val_accuracy += (val_outputs.argmax(dim=-1) == val_batch_labels).float().mean().item()

            avg_val_loss = val_loss / val_num_batches
            avg_val_accuracy = val_accuracy / val_num_batches
            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_accuracy)

            print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")

            # Stop training for this part when the cumulative loss is very close to zero
            if avg_loss < 1e-3:
                break

    return its, train_losses, train_accuracies, val_losses, val_accuracies

def main(args):
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokens for <op> and <=>.
    eq_token = args.p
    op_token = args.p + 1

    model = Decoder(
        dim=128, num_layers=2, num_heads=4, num_tokens=args.p + 2, seq_len=5
    ).to(device)
    nparams = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(model)
    print(f'Total number of parameters: {nparams}')

    data = multiplication_mod_p_data(args.p, eq_token, op_token)
    parts = split_data_into_10_parts(data)

    train_idx, valid_idx = torch.randperm(data.shape[1]).split(data.shape[1] // 2)
    train_data, valid_data = data[:, train_idx], data[:, valid_idx]

    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )

    steps_per_epoch = math.ceil(train_data.shape[1] / args.batch_size)

    its, train_acc, val_acc, train_loss, val_loss = [], [], [], [], []
    grads = None
    i = 0

    if args.method_type == "progressive":
        its, train_loss, train_acc, val_loss, val_acc = train_progressive(model, parts, data, optimizer, scheduler, device, lambda_weight=args.lambda_weight)
    else:
        for e in tqdm(range(int(args.budget) // steps_per_epoch)):
            train_data = train_data[:, torch.randperm(train_data.shape[1])]

            for data, is_train in [(train_data, True), (valid_data, False)]:
                model.train(is_train)
                total_loss = 0
                total_acc = 0

                dl = torch.split(data, args.batch_size, dim=1)
                for input in dl:
                    input = input.to(device)

                    with torch.set_grad_enabled(is_train):
                        logits = model(input[:-1])
                        loss = F.cross_entropy(logits[-1], input[-1])
                        total_loss += loss.item() * input.shape[-1]

                    if is_train:
                        model.zero_grad()
                        loss.backward()

                        trigger = i < 500 if args.two_stage else False

                        if args.filter == "none":
                            pass
                        elif args.filter == "ma":
                            grads = gradfilter_ma(model, grads=grads, window_size=args.window_size, lamb=args.lamb, trigger=trigger)
                        elif args.filter == "ema":
                            grads = gradfilter_ema(model, grads=grads, alpha=args.alpha, lamb=args.lamb)
                        else:
                            raise ValueError(f"Invalid gradient filter type `{args.filter}`")

                        optimizer.step()
                        scheduler.step()
                        i += 1

                    acc = (logits[-1].argmax(-1) == input[-1]).float().mean()
                    total_acc += acc.item() * input.shape[-1]

                if is_train:
                    train_acc.append(total_acc / train_data.shape[-1])
                    train_loss.append(total_loss / train_data.shape[-1])
                    its.append(i)
                else:
                    val_acc.append(total_acc / valid_data.shape[-1])
                    val_loss.append(total_loss / valid_data.shape[-1])

    if args.save_weights:
        steps = np.array(its) if args.method_type == "progressive" else torch.arange(len(train_acc)).numpy() * steps_per_epoch
        plt.plot(steps, train_acc, label="train")
        plt.plot(steps, val_acc, label="val")
        plt.legend()
        plt.title("Modular Multiplication (training on 50% of data)")
        plt.xlabel("Optimization Steps")
        plt.ylabel("Accuracy")
        plt.xscale("log", base=10)
        plt.grid()
        plt.savefig(f"results/acc_{args.label}.png", dpi=150)
        plt.close()

        plt.plot(steps, train_loss, label="train")
        plt.plot(steps, val_loss, label="val")
        plt.legend()
        plt.title("Modular Multiplication (training on 50% of data)")
        plt.xlabel("Optimization Steps")
        plt.ylabel("Loss")
        plt.xscale("log", base=10)
        plt.grid()
        plt.savefig(f"results/loss_{args.label}.png", dpi=150)
        plt.close()

        results = {
                'its': its,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_loss': val_loss,
            }

        if args.save_weights:
                net_its.append(e)
                nets.append(copy.deepcopy(model.state_dict()))
                results['net_its'] = net_its
                results['net'] = nets

        torch.save(results, f"results/res_{args.label}.pt")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=3e5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--method_type", default="progressive")
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lambda_weight", default=1.5)

    # Grokfast
    parser.add_argument("--filter", type=str, choices=["none", "ma", "ema", "fir"], default="none")
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--lamb", type=float, default=5.0)

    # Ablation studies
    parser.add_argument("--two_stage", action='store_true')
    parser.add_argument("--save_weights", action='store_true')
    args = parser.parse_args()

    filter_str = ('_' if args.label != '' else '') + args.filter
    window_size_str = f'_w{args.window_size}'
    alpha_str = f'_a{args.alpha:.3f}'.replace('.', '')
    lamb_str = f'_l{int(args.lamb)}'

    if args.filter == 'none':
        filter_suffix = ''
    elif args.filter == 'ma':
        filter_suffix = window_size_str + lamb_str
    elif args.filter == 'ema':
        filter_suffix = alpha_str + lamb_str
    else:
        raise ValueError(f"Unrecognized filter type {args.filter}")

    optim_suffix = ''
    if args.weight_decay != 0:
        optim_suffix = optim_suffix + f'_wd{args.weight_decay:.1e}'.replace('.', '')
    if args.lr != 1e-3:
        optim_suffix = optim_suffix + f'_lrx{int(args.lr / 1e-3)}'

    args.label = args.label + filter_str + filter_suffix + optim_suffix
    print(f'Experiment results saved under name: {args.label}')

    main(args)