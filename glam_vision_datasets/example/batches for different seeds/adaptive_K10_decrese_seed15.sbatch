#!/bin/bash 
#SBATCH --job-name=adaptive_K10_dec_seed_15
#SBATCH --output=adaptive_K10_dec_seed_15_output.log
#SBATCH --error=adaptive_K10_dec_seed_15_error.log
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

python train_adaptive_K_lookahead.py --label adaptive_K10_dec_seed_15 --method adaptive_decrease --inital_K 10 --k_multiplier 5 --seed 15

