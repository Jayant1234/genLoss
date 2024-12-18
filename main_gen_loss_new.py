import math
from argparse import ArgumentParser
from itertools import permutations
import copy

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

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

def split_data_into_n_parts(data,n_parts):
    # Shuffle the data indices
    indices = torch.randperm(data.shape[1])
    
    # Split the indices into 10 roughly equal parts
    parts = torch.split(indices, data.shape[1] // n_parts)
    parts = list(parts)

    # If there are any remaining indices (because data.shape[1] may not be perfectly divisible by 10),
    # add them to the last part
    if len(parts) > n_parts:
        parts[-2] = torch.cat((parts[-2], parts[-1]))
        parts = parts[:-1]
    
    # Create a dictionary to label each part
    labeled_parts = {f"part_{i+1}": parts[i] for i in range(len(parts))}
    
    return labeled_parts
    
def train_progressive(model, data, valid_data, optimizer, scheduler, device, args):

    cumulative_indices = torch.tensor([], dtype=torch.long)
    total_steps = 0  # Track the total number of steps taken
    
    parts=split_data_into_n_parts(data,args.parts)
    # Use the last part as the validation set
    internal_val_indices = parts[f"part_{args.parts}"]
    internal_val_data = data[:, internal_val_indices]

    
    # Remove the last part from the training parts
    training_parts = {k: parts[k] for k in list(parts.keys())[:-1]}
    #print("Training parts are:",training_parts)
    # Containers to save training and validation metrics
    its, train_acc, gen_acc, val_acc, in_val_loss, in_val_acc, gen_loss, train_loss, val_loss = [], [], [], [], [], [], [], [], []

    e=0 # total epoch counter
    i=0 # iteration counter
    cutoff=1e-6
    gen_loss_type= 'standard' #MSE, KLdivergence are other options
    pbar = tqdm()
    lambda_weight = args.lambda_weight
    
    part=1
    repetition =0
    while(part < len(training_parts)+1):
        # Accumulate parts
        print(f"Accumulating data for Part {part}")
        
        if part<len(training_parts)+1:
            cumulative_indices = torch.cat((cumulative_indices, training_parts[f"part_{part}"]))
            
        train_data = data[:, cumulative_indices]
        print(f"Cumulative training data shape before adding Part {part}: {train_data.shape}")
        gen_data=None
        if part<len(training_parts) or part==1:  #part==1 is for baseline case
            gen_data_indices = training_parts[f"part_{part+1}"]
            gen_data = data[:, gen_data_indices]
            print(f"gen data shape with Part {part}: {gen_data.shape}")

        if cumulative_indices.numel() == 0: # first case
            print("First case switch of train and gen data happens")
            print("Is gen_data None?:", bool(gen_data is None))
            train_data=gen_data
            gen_data =None
        if gen_data is not None and cumulative_indices.numel() != 0:
            # Create matching sizes for gen data and new data by repeating the gen dataset
            if train_data.shape[1] > gen_data.shape[1] and gen_data.shape[1] > 0:
                repeats = train_data.shape[1] // gen_data.shape[1] + 1
                gen_data = gen_data.repeat(1,repeats)[:, :train_data.shape[1]]

        epochs=0 #epoch counter for concentrated training
        
        max_epochs_counter = args.last_max_epochs if part == len(training_parts)+1 else args.max_epochs
        
        internal_val_counter=[]# counter to stop the training when gen loss is sufficiently minimized

        while epochs <= max_epochs_counter:
            
            train_data = train_data[:, torch.randperm(train_data.shape[1])]
            
            if gen_data is not None: 
                gen_data =   gen_data[:, torch.randperm(gen_data.shape[1])]
            
            for the_data, g_data, is_train, is_in in [(train_data, gen_data, True, False), (valid_data, None, False, False), (internal_val_data, None, False, True)]:
                
                model.train(is_train)

                if g_data is not None and the_data is not None and is_train:
                    total_train_loss = 0
                    total_train_acc = 0
                    total_gen_loss = 0
                    total_gen_acc = 0
                    # torch.split faster than dataloader with tensor
                    train_batches = torch.split(the_data, args.batch_size, dim=1)
                    gen_batches = torch.split(g_data, args.batch_size, dim=1)
                    # Print the count of batches
                    #print(f"Number of train batches: {len(train_batches)}")
                    #print(f"Number of gen batches: {len(gen_batches)}")

                    for train_input,gen_input in zip(train_batches,gen_batches):
                        train_input = train_input.to(device)
                        gen_input = gen_input.to(device)
                        
                        with torch.set_grad_enabled(is_train):
                            train_logits = model(train_input[:-1])
                            # calculate loss only on the answer part of the equation (last element
                            t_loss_per_sample = F.cross_entropy(train_logits[-1], train_input[-1], reduction='none')
                            # Compute average loss
                            t_loss = t_loss_per_sample.mean()

                            total_train_loss += t_loss.item() * train_input.shape[-1]
                            
                            gen_logits = model(gen_input[:-1])
                            # calculate loss only on the answer part of the equation (last element
                            g_loss_per_sample = F.cross_entropy(gen_logits[-1], gen_input[-1],reduction='none')
                            g_loss = g_loss_per_sample.mean()
                            total_gen_loss += g_loss.item() * gen_input.shape[-1]
                            if args.loss_type =="cross_entropy": 
                                loss= (1-lambda_weight)*t_loss + lambda_weight*g_loss
                            elif args.loss_type =="mse": 
                                loss= (1-lambda_weight)*t_loss + 0.5*lambda_weight*(g_loss-t_loss).pow(2)
                            elif args.loss_type =="mae":  
                                loss= (1-lambda_weight)*t_loss + lambda_weight*torch.abs(g_loss-t_loss)
                            elif args.loss_type =='huber':
                                huber_loss_fn = SmoothL1Loss(beta=delta)  # beta parameter is the delta in PyTorch 1.6+
                                loss= (1-lambda_weight)*t_loss + lambda_weight*huber_loss_fn(t_loss, g_loss)
                            elif args.loss_type =='relative_difference': 
                                epsilon = 1e-8  # Small constant to prevent division by zero
                                denominator = t_loss + g_loss + epsilon
                                gap_loss = torch.abs(t_loss - g_loss) / denominator
                                loss= (1-lambda_weight)*t_loss + lambda_weight* gap_loss
                            elif args.loss_type == 'earth_mover': 
                                # Sort the losses
                                losses_seen_sorted, _ = torch.sort(t_loss_per_sample)
                                losses_unseen_sorted, _ = torch.sort(g_loss_per_sample)

                                # Compute Wasserstein distance
                                gap_loss = torch.mean(torch.abs(losses_seen_sorted - losses_unseen_sorted))
                                loss= (1-lambda_weight)*t_loss + lambda_weight* gap_loss
                            elif args.loss_type == 'kl': 
                                # Compute KL divergence from seen to unseen
                                kl_divergence = F.kl_div(t_loss_per_sample, g_loss_per_sample, reduction='batchmean')

                                # Alternatively, compute symmetric KL divergence
                                kl_divergence_symmetric = 0.5 * (
                                    F.kl_div(g_loss_per_sample, t_loss_per_sample, reduction='batchmean') +
                                    F.kl_div(t_loss_per_sample, g_loss_per_sample, reduction='batchmean')
                                )

                                # Choose one of the KL divergence measures
                                loss= (1-lambda_weight)*t_loss + lambda_weight* kl_divergence_symmetric
                                
                        model.zero_grad()
                        loss.backward()

                        optimizer.step()
                        scheduler.step()
                        i += 1
                        total_steps+=1

                        acc = (train_logits[-1].argmax(-1) == train_input[-1]).float().mean()
                        total_train_acc += acc.item() * train_input.shape[-1]

                        acc = (gen_logits[-1].argmax(-1) == gen_input[-1]).float().mean()
                        total_gen_acc += acc.item() * gen_input.shape[-1]

                    train_acc.append(total_train_acc / train_data.shape[-1])
                    train_loss.append(total_train_loss / train_data.shape[-1])
                    its.append(i)
                    gen_acc.append(total_gen_acc / gen_data.shape[-1])
                    gen_loss.append(total_gen_loss / gen_data.shape[-1])
                    gen_loss_counter=total_gen_loss / gen_data.shape[-1]

                else:
                    total_loss = 0
                    total_acc = 0
                    
                    # torch.split faster than dataloader with tensor
                    dl = torch.split(the_data, args.batch_size, dim=1)
                    for input in dl:
                        input = input.to(device)

                        with torch.set_grad_enabled(is_train):
                            logits = model(input[:-1])
                            # calculate loss only on the answer part of the equation (last element
                            loss = F.cross_entropy(logits[-1], input[-1])
                            total_loss += loss.item() * input.shape[-1]

                        if is_train:
                            model.zero_grad()
                            loss.backward()

                            optimizer.step()
                            scheduler.step()
                            i += 1
                            total_steps+=1

                        acc = (logits[-1].argmax(-1) == input[-1]).float().mean()
                        total_acc += acc.item() * input.shape[-1]

                    if is_train:
                        train_acc.append(total_acc / train_data.shape[-1])
                        train_loss.append(total_loss / train_data.shape[-1])
                        gen_acc.append(0)
                        gen_loss.append(0)
                        its.append(i)
                            
                    else:
                        if is_in:
                            in_val_acc.append(total_acc / internal_val_data.shape[-1])
                            in_val_loss.append(total_loss / internal_val_data.shape[-1])

                        else:
                            val_acc.append(total_acc / valid_data.shape[-1])
                            val_loss.append(total_loss / valid_data.shape[-1])

            do_save = e <= 500 or (e > 500 and (e + 1) % 10 == 0)
            if do_save:
                steps = torch.arange(len(train_acc)).numpy() # steps calculation is tricky so will leave it for future. 
                plt.plot(steps, train_acc, label="train")
                plt.plot(steps, val_acc, label="val")
                plt.plot(steps, gen_acc, label="gen")
                plt.plot(steps, in_val_acc, label="in_val")
                plt.legend()
                plt.title("Modular Multiplication")
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.xscale("log", base=10)
                plt.grid()
                file_name = f"results/acc_method_{args.method_type}_loss_type_{args.loss_type}_lambda_{args.lambda_weight}_maxepochs_{args.max_epochs}_lastmaxepochs_{args.last_max_epochs}_minerror_{args.min_error}_parts_{args.parts}.png"
                plt.savefig(file_name, dpi=150)
                plt.close()

                plt.plot(steps, train_loss, label="train")
                plt.plot(steps, val_loss, label="val")
                plt.plot(steps, gen_loss, label="gen")
                plt.plot(steps, in_val_loss, label="in_val")

                plt.legend()
                plt.title("Modular Multiplication (training on 50% of data)")
                plt.xlabel("Optimization Steps")
                plt.ylabel("Loss")
                #plt.xscale("log", base=10)
                plt.grid()
                file_name = f"results/loss_method_{args.method_type}_loss_type_{args.loss_type}_lambda_{args.lambda_weight}_maxepochs_{args.max_epochs}_lastmaxepochs_{args.last_max_epochs}_minerror_{args.min_error}_parts_{args.parts}.png"
                plt.savefig(file_name, dpi=150)
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
            pbar.update(1)
            e+=1
            epochs+=1
            
            if len(in_val_loss) >2 and args.early_stopping:
                if in_val_loss[-1] > in_val_loss[-2] and part>1:
                    break

    
        if part<len(training_parts)+1:
            cumulative_indices = torch.cat((cumulative_indices, training_parts[f"part_{part}"]))

        if part==len(training_parts) and args.early_stopping and repetition <25:# is_repeat is used for early stopping
            part=1
            repetition+=1
            cumulative_indices = torch.tensor([], dtype=torch.long)
        part+=1
        
    print("Total number of optimizer steps:", total_steps)
    pbar.close()
    
    return its, train_acc, gen_acc, val_acc, gen_loss, train_loss, val_loss

def train_baseline(model, train_data, valid_data, optimizer, scheduler, device, args):

    steps_per_epoch = math.ceil(train_data.shape[1] / args.batch_size)
    
    its, train_acc, val_acc, train_loss, val_loss = [], [], [], [], []
    grads = None
    i = 0

    # For logging network weights.
    net_its, nets = [], []

    for e in tqdm(range(int(args.budget) // steps_per_epoch)):

        # randomly shuffle train data
        train_data = train_data[:, torch.randperm(train_data.shape[1])]

        for data, is_train in [(train_data, True), (valid_data, False)]:

            model.train(is_train)
            total_loss = 0
            total_acc = 0
            
            # torch.split faster than dataloader with tensor
            dl = torch.split(data, args.batch_size, dim=1)
            for input in dl:
                input = input.to(device)

                with torch.set_grad_enabled(is_train):
                    logits = model(input[:-1])
                    # calculate loss only on the answer part of the equation (last element
                    loss = F.cross_entropy(logits[-1], input[-1])
                    total_loss += loss.item() * input.shape[-1]

                if is_train:
                    model.zero_grad()
                    loss.backward()

                    #######

                    trigger = i < 500 if args.two_stage else False

                    if args.filter == "none":
                        pass
                    elif args.filter == "ma":
                        grads = gradfilter_ma(model, grads=grads, window_size=args.window_size, lamb=args.lamb, trigger=trigger)
                    elif args.filter == "ema":
                        grads = gradfilter_ema(model, grads=grads, alpha=args.alpha, lamb=args.lamb)
                    else:
                        raise ValueError(f"Invalid gradient filter type `{args.filter}`")

                    #######

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
            do_save = e <= 500 or (e > 500 and (e + 1) % 100 == 0) or e == int(args.budget) // steps_per_epoch - 1
        else:
            do_save = (e + 1) % 100 == 0
        if do_save:
            steps = torch.arange(len(train_acc)).numpy() * steps_per_epoch
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

def main(args):
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # tokens for <op> and <=>. It's not clear why <=> is needed at all since it
    # has no effect on the output, but we'll leave it in to best follow the
    # paper.
    eq_token = args.p
    op_token = args.p + 1

    # "We trained a standard decoder-only transformer (Vaswani et al., 2017)
    # with causal attention masking, and calculated loss and accuracy only on
    # the answer part of the equation. For all experiments we used a
    # transformer with 2 layers, width 128, and 4 attention heads"
    model = Decoder(
        dim=128, num_layers=2, num_heads=4, num_tokens=args.p + 2, seq_len=5
    ).to(device)
    nparams = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(model)
    print(f'Total number of parameters: {nparams}')

    data = multiplication_mod_p_data(args.p, eq_token, op_token)

    # Get the number of columns (data points)
    num_data_points = data.shape[1]

    # Compute the 90-10 split point
    split_point = int(0.5 * num_data_points)

    # Shuffle the data indices
    train_idx, valid_idx = torch.randperm(num_data_points).split([split_point, num_data_points - split_point])

    # Split the data
    train_data, valid_data = data[:, train_idx], data[:, valid_idx]


    # For most experiments we used AdamW optimizer with learning rate 10−3,
    # weight decay 1, β1 = 0.9, β2 = 0.98
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    #  linear learning rate warmup over the first 10 updates
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )
    
    if args.method_type =="progressive":
        train_progressive(model, train_data, valid_data, optimizer, scheduler, device, args)
    else: 
        train_baseline(model, train_data, valid_data, optimizer, scheduler, device, args)

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
    parser.add_argument("--optimizer", default="Adam")

    #Generalization Loss
    parser.add_argument("--method_type", default="progressive")
    parser.add_argument("--lambda_weight", type=float, default=0.9)
    parser.add_argument("--max_epochs",type=int, default=20)
    parser.add_argument("--last_max_epochs",type=int, default=200)
    parser.add_argument("--min_error",type=float, default=1e-3)
    parser.add_argument("--parts",type=int, default=10)
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--part_wise", action="store_true", help="Enable early stopping")
    parser.add_argument("--loss_type", default="cross_entropy",  choices=[
        "cross_entropy", 
        "mse", 
        "mae", 
        "huber", 
        "relative_difference", 
        "earth_mover", 
        "kl"
        ])

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
