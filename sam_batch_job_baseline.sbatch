#!/bin/bash  
#SBATCH --job-name=mnist_sam_baseline
#SBATCH --output=sam_baselinem_o.log
#SBATCH --error=sam_baseline_o.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:L4:1# Request 1 L4 per node
#SBATCH --mem=64 # Request 64GB memory
#SBATCH --time=12:00:00 # Request 12 hours runtime

module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

#activate your environment
source /home/jp98084/environments/gen/bin/activate

#here i am running SAM with the same parameter as the baseline for the 
#main mnist default parameters
python main_mnist_with_sam.py --label baseline_sam_with_main_mnist 
