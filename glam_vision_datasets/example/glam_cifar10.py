
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
    parser.add_argument("--disable_glam", default=False, type=bool, help="Disable the gradient local alignment mechanism.")
    
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(args.batch_size, args.threads)
    log = Log(filename="Glam",log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    # base_optimizer = torch.optim.SGD
    # optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    steps = 0
    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            # # first forward-backward step
            # enable_running_stats(model)

            half_size = inputs.size(0) // 2
            inputs_half1, inputs_half2 = inputs[:half_size], inputs[half_size:]
            targets_half1,targets_half2 = targets[:half_size], targets[half_size:]


            predictions_half1 = model(inputs_half1)
            predictions_half2 = model(inputs_half2)
            
            L_B1 = smooth_crossentropy(predictions_half1, targets_half1, smoothing=args.label_smoothing)
            L_B2 = smooth_crossentropy(predictions_half2, targets_half2, smoothing=args.label_smoothing)



            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                model.zero_grad()
                g_B1 = torch.autograd.grad(L_B1.mean(), model.parameters(), create_graph=True)
                g_B2 = torch.autograd.grad(L_B2.mean(), model.parameters(), create_graph=True)
                    
                s = sum((g1 * g2).sum() for g1, g2 in zip(g_B1, g_B2))

                
                # Compute the norms of g_B1 and g_B2
                norm_g_B1 = torch.sqrt(sum((g1 ** 2).sum() for g1 in g_B1))
                norm_g_B2 = torch.sqrt(sum((g2 ** 2).sum() for g2 in g_B2))

                #Normalize the dot product
                similarity = s / (norm_g_B1 * norm_g_B2 + 1e-8)  # Add epsilon for numerical stability

                # Compute gradient of s with respect to model parameters
                grad_s = torch.autograd.grad((1-similarity), model.parameters())
                if steps % 1000 == 0: 
                        #print("gradient for coherence is:", grad_s)
                        #print("gradient for baseline is:", g_B1)
                        print("similarity of both gradients is::::",similarity)
                    
            if args.disable_glam:
                total_grad = [g1+g2 for g1,g2 in zip(g_B1, g_B2)]
            else:      
                total_grad = [g1+g2 + gs for g1,g2, gs in zip(g_B1, g_B2, grad_s)]
                        
            for p, g in zip(model.parameters(), total_grad):
                    p.grad = g
            
            
            optimizer.step()
            steps += 1


            # # optimizer.first_step(zero_grad=True)

            # # second forward-backward step
            # disable_running_stats(model)
            # smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            # optimizer.second_step(zero_grad=True)
            # Accumulate results
            predictions = torch.cat([predictions_half1, predictions_half2], dim=0)
            targets_combined = torch.cat([targets_half1, targets_half2], dim=0)
            # Calculate loss and accuracy for both halves combined
            loss_combined = torch.cat([L_B1, L_B2], dim=0)
            # print("predictions_half1 Shape:", predictions_half1.shape)
            # print("L1 Shape:", L_B1.shape)
            # print("predictions_half2 Shape:", predictions_half2.shape)
            # print("L2 Shape:", L_B2.shape)
     
            
            # print("Predictions Shape:", predictions.shape)
            # print("Targets Shape:", targets_combined.shape)
            # print("Loss Combined Shape:", loss_combined.shape)
            correct = torch.argmax(predictions.data, 1) == targets_combined
            
            with torch.no_grad():
                log(model, loss_combined.cpu(), correct.cpu(), scheduler.lr())
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
    # Save the plots after all epochs
    log.save_loss_plot(log.train_losses, log.val_losses, filename='training_validation_loss.png')
    log.save_accuracy_plot(log.train_accuracies, log.val_accuracies, filename='training_validation_accuracy.png')

    log.flush()
