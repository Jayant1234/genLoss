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


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

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

def calculate_weight_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:  # Only consider parameters with gradients
            param_norm = param.data.norm(2)  # L2 norm of the parameter
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5  # Take the square root to get the L2 norm
    return total_norm

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


def train_progressive(model, train_data, valid_data, optimizer, scheduler, device, args):
    total_steps = 0  # Track the total number of steps taken
    flip_all = False
    total_ex=train_data.shape[1]
    # Split 10% of train_data as gen_set
    gen_size = int(0.1 * total_ex)
    gen_idx, train_idx = torch.randperm(total_ex).split([gen_size, total_ex - gen_size])

    train_data, gen_set = train_data[:, train_idx], train_data[:, gen_idx]

    steps_per_epoch = math.ceil(train_data.shape[1] / args.batch_size)

    # Containers for tracking training, validation, and generalization metrics
    its, train_acc, val_acc, gen_acc, train_loss, val_loss, gen_loss = [], [], [], [], [], [], []
    
    epochs = 0  # Epoch counter
    i = 0  # Step counter
    pbar = tqdm()
    counter=0
    while total_steps <= args.budget:
        # Shuffle training data for each epoch
        train_data = train_data[:, torch.randperm(train_data.shape[1])]
        gen_set = gen_set[:, torch.randperm(gen_set.shape[1])]
        
        for data, is_train, dataset_name in [(train_data, True, "train"), (gen_set, False, "gen"), (valid_data, False, "val")]:
            model.train(is_train)
            total_loss = 0
            total_acc = 0

            # Split data into batches
            data_batches = torch.split(data, args.batch_size, dim=1)

            for input_batch in data_batches:
                input_batch = input_batch.to(device)
                
                with torch.set_grad_enabled(is_train):
                    logits = model(input_batch[:-1])
                    loss = F.cross_entropy(logits[-1], input_batch[-1])
                    total_loss += loss.item() * input_batch.shape[-1]

                if is_train and dataset_name == "train":
                    # Regular training step
                    model.zero_grad()
                    loss.backward()

                    # Gradient manipulation for "progressive_signed" method
                    if args.method_type == "anti-learning":
                        
                        #average_grad_norm = total_grad_norm / param_count if param_count > 0 else 0
                        #print(f"Average Gradient Norm: {average_grad_norm}")
                        generalization_gap = [val - train for val, train in zip(gen_loss, train_loss)]
                        if len(generalization_gap)>6: 
                            if generalization_gap[-1]>0 and generalization_gap[-1]> sum(generalization_gap[-7:-2])/5:
                                if flip_all: 
                                    flip_all=False
                                else: 
                                    flip_all=True
                        if flip_all:
                            counter+=1
                            if counter>20: 
                                flip_all=False
                                counter=0
                        #p = 40  # Set the probability of flipping all gradients
                        with torch.no_grad():  # Use no_grad to prevent tracking in autograd
                            # Generate a single random probability to decide if we flip all gradients
                            for name, param in model.named_parameters():
                                if param.grad is not None:
                                    g_B = param.grad
                                    
                                    # If flip_all is True, flip the sign of the gradient for all parameters
                                    if flip_all:
                                        param.grad = -g_B  # Flip all signs

                    optimizer.step()
                    scheduler.step()
                    i += 1
                    total_steps += 1

                elif is_train and dataset_name == "gen":
                    
                    # Calculate averaged gradients and update each parameter
                    model.zero_grad()
                    loss.backward()  # Backprop for `gen_set`
                    
                    # # Generalization set logic for weight norm learning
                    # if len(gen_loss) >= 6:
                        # last_loss = gen_loss[-1]
                        # avg_last_five = sum(gen_loss[-6:-1]) / 5
                        # last_to_last=gen_loss[-2]
                        # if avg_last_five> 0.1: 
                            # with torch.no_grad():
                                # global_grad_norm = torch.sqrt(sum(p.grad.norm(2)**2 for p in model.parameters() if p.grad is not None))
                                # for param in model.parameters():
                                    # if param.grad is not None:
                                        # # Find the average of the gradients for non-individual weights
                                        # if param.grad.dim() > 1:  # Ensure it's not a single weight
                                            # #grad_norm = param.grad.norm()
                                            # # Fill the gradient with this signed norm value
                                            # update = torch.full_like(param.grad, global_grad_norm)
                                            # update = update*torch.sign(param.data)

                                            # if not (last_loss < avg_last_five) and last_loss>last_to_last:
                                                # param.data-=args.lr*update #anti-learning norm grad step
                                            # #else: 
                                                # #param.data+=args.lr*update #learning norm grad step

                # Compute accuracy
                acc = (logits[-1].argmax(-1) == input_batch[-1]).float().mean()
                total_acc += acc.item() * input_batch.shape[-1]

            # Track metrics
            avg_loss = total_loss / data.shape[-1]
            avg_acc = total_acc / data.shape[-1]
            if dataset_name == "train":
                train_acc.append(avg_acc)
                train_loss.append(avg_loss)
                its.append(i)
            elif dataset_name == "gen":
                gen_acc.append(avg_acc)
                gen_loss.append(avg_loss)
            else:
                val_acc.append(avg_acc)
                val_loss.append(avg_loss)

        # Plot and save results periodically
        do_save = (epochs <= 500 and (epochs + 1) % 10 == 0) or (epochs > 500 and (epochs + 1) % 100 == 0)
        if do_save:
            steps = torch.arange(len(train_acc)).numpy()*steps_per_epoch
            
            # Plot accuracy
            plt.plot(steps, train_acc, label="train")
            plt.plot(steps, val_acc, label="val")
            plt.plot(steps, gen_acc, label="gen")
            plt.legend()
            plt.title("Modular Multiplication Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.xscale("log")
            plt.grid()
            plt.savefig(f"results/acc_{args.label}.png", dpi=150)
            plt.close()
            
            # Plot loss
            plt.plot(steps, train_loss, label="train")
            plt.plot(steps, val_loss, label="val")
            plt.plot(steps, gen_loss, label="gen")
            plt.legend()
            plt.title("Modular Multiplication Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.xscale("log")
            plt.grid()
            plt.savefig(f"results/loss_{args.label}.png", dpi=150)
            plt.close()

            # Save model and results
            results = {
                'its': its,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'gen_acc': gen_acc,
                'gen_loss': gen_loss,
            }

            if args.save_weights:
                results['net'] = [copy.deepcopy(model.state_dict())]

            torch.save(results, f"results/res_{args.label}.pt")
        
        pbar.update(1)
        epochs += 1

    print("Total number of optimizer steps:", total_steps)
    pbar.close()

    return its, train_acc, gen_acc, val_acc, gen_loss, train_loss, val_loss

def train_baseline(model, train_data, valid_data, optimizer, scheduler, device, args):

    steps_per_epoch = math.ceil(train_data.shape[1] / args.batch_size)
    
    its, train_acc, val_acc, train_loss, val_loss, sim = [], [], [], [], [], []
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
            avg_sim=0
            
            # torch.split faster than dataloader with tensor
            dl = torch.split(data, args.batch_size, dim=1)
            num_batches = len(dl)

            # Loop through each batch with the next one as reference
            for num_batch in range(num_batches):
                
                input = dl[num_batch].to(device)
                b2_input = dl[(num_batch + 1) % num_batches].to(device)

                with torch.set_grad_enabled(is_train):
                    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                        logits = model(input[:-1])
                        # calculate loss only on the answer part of the equation (last element
                        L_B1 = F.cross_entropy(logits[-1], input[-1])
                        total_loss += L_B1.item() * input.shape[-1]
                        
                        
                        logits_b2 = model(b2_input[:-1])
                        # calculate loss only on the answer part of the equation (last element
                        L_B2 = F.cross_entropy(logits_b2[-1], b2_input[-1])

                if is_train:
                    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                        model.zero_grad()

                        g_B1 = torch.autograd.grad(L_B1, model.parameters(), create_graph=True)
                        g_B2 = torch.autograd.grad(L_B2, model.parameters(), create_graph=True)
                        
                        #g_B1 = torch.autograd.grad(L_B1, model.parameters(), retain_graph=True)
                        #g_B2 = torch.autograd.grad(L_B2, model.parameters(), retain_graph=True)

                        # Convert gradients to vectors
                        #g_B1_vector = torch.cat([g.view(-1) for g in g_B1])
                        #g_B2_vector = torch.cat([g.view(-1) for g in g_B2])

                        # Compute cosine similarity
                        # cos_sim = torch.nn.functional.cosine_similarity(g_B1_vector, g_B2_vector, dim=0)

                        # # Define total loss
                        # L_total = L_B1 - cos_sim
                        # L_total.backward()
                        # Compute dot product s = g_B2^T g_B1
                        
                        s = sum((g1 * g2).sum() for g1, g2 in zip(g_B1, g_B2))

                        #cosine_sim = sum((g1 * g2).sum()/(torch.sqrt(sum((g1 ** 2)))*torch.sqrt(sum((g2 ** 2)))+ 1e-8) for g1, g2 in zip(g_B1, g_B2)) #trying pair-wise cosine similarity

                        # Compute the norms of g_B1 and g_B2
                        norm_g_B1 = torch.sqrt(sum((g1 ** 2).sum() for g1 in g_B1))
                        norm_g_B2 = torch.sqrt(sum((g2 ** 2).sum() for g2 in g_B2))

                        #Normalize the dot product
                        cosine_sim = s / (norm_g_B1 * norm_g_B2 + 1e-8)  # Add epsilon for numerical stability

                        
                        # Compute gradient of s with respect to model parameters
                        grad_s = torch.autograd.grad((1-cosine_sim), model.parameters())
                        if i % 1000 == 0 or num_batch == 0: 
                            #print("gradient for coherence is:", grad_s)
                            #print("gradient for baseline is:", g_B1)
                            print("similarity of both gradients is::::",cosine_sim)
                            print(num_batch)
                        
                        #curious case of barely doing it and still getting same results. 
                        # if i >10: 
                            # total_grad = [g1+g2 + 0*gs for g1,g2, gs in zip(g_B1, g_B2, grad_s)]
                        # #Compute total gradient
                        # else: 
                            # total_grad = [g1+g2 + gs for g1,g2, gs in zip(g_B1, g_B2, grad_s)]

                        
                        total_grad = [g1+g2 + gs for g1,g2, gs in zip(g_B1, g_B2, grad_s)]
                        
                        #Assign gradients to parameters
                        for p, g in zip(model.parameters(), total_grad):
                            p.grad = g
                        #######

                        trigger = i < 500 if args.two_stage else False

                        if args.filter == "none":
                            pass
                        elif args.filter == "anti":
                            total_grad_norm = 0.0
                            param_count = 0
                            with torch.no_grad():
                                for name, param in model.named_parameters():
                                    if param.grad is not None:
                                    # Exclude bias or single-element parameters
                                        if len(param.shape) > 1:
                                            grad_norm = param.grad.norm(2).item()
                                            
                                            # Accumulate grad norm and count for averaging
                                            total_grad_norm += grad_norm
                                            param_count += 1
                                            # Create a tensor that scales the gradient's direction by the gradient norm's opposite sign
                                            # We normalize the gradient to keep the direction and scale by grad_norm
                                            if grad_norm != 0:
                                                # Normalize the gradient and reverse its direction
                                                direction = -param.grad / grad_norm  # Normalized direction of gradient, flipped
                                                # Calculate the adjustment to param.data
                                                adjustment = direction * grad_norm
                                                # Update param.data directly
                                                param.data += adjustment  # Move param.data in the opposite direction by grad_norm

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
                avg_sim+=cosine_sim.item()
                acc = (logits[-1].argmax(-1) == input[-1]).float().mean()
                total_acc += acc.item() * input.shape[-1]

            if is_train:
                train_acc.append(total_acc / train_data.shape[-1])
                train_loss.append(total_loss / train_data.shape[-1])
                sim.append(100*avg_sim/train_data.shape[-1])
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
            plt.plot(steps, sim, label="cosine")
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
    for name, param in model.named_parameters():
        print(name)
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
    #optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)

    #  linear learning rate warmup over the first 10 updates
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )
    
    #train_baseline(model, train_data, valid_data, optimizer, scheduler, device, args)
    train_progressive(model, train_data, valid_data, optimizer, scheduler, device, args)


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
    parser.add_argument("--method_type", default="progressive_grad")
    parser.add_argument("--lambda_weight", type=float, default=0.9)
    parser.add_argument("--max_epochs",type=int, default=500)
    parser.add_argument("--last_max_epochs",type=int, default=10000)
    parser.add_argument("--min_error",type=float, default=1e-3)
    parser.add_argument("--parts",type=int, default=8)
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
    parser.add_argument("--filter", type=str, choices=["none", "anti", "ma", "ema", "fir"], default="none")
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
    elif args.filter == 'anti':
        filter_suffix = "anti"
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
