#!/bin/bash 
#SBATCH --job-name=greedy_lookahead_k100_lr_0.1
#SBATCH --output=greedy_lookahead_k100_lr0.1_output.log
#SBATCH --error=greedy_lookahead_k100_lr0.1_error.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:L4:1# Request 1 L4 per node
#SBATCH --mem=78G # Request 64GB memory
#SBATCH --time=24:00:00 # Request 12 hours runtime

module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

#activate your environment
source /home/bg91882/environments/gen/bin/activate

python train_greedy_lookahead_optimizer.py --label greedy_lookahead_k_100_lr0.1_cw_0.5_bw_0.5 --learning_rate 0.1 


