#!/bin/bash
#SBATCH --job-name=main_mnis_withglamv2_early_stopping.py           # Job name
#SBATCH --output=main_mnis_with_glamv2_early_stopping.py_o.log            # Standard output log $
#SBATCH --error=main_mnis_with_glamv2_early_stopping_o.log             # Standard error log f$
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks to run
#SBATCH --ntasks-per-node=1            # Tasks per node
#SBATCH --partition=gpu_p              # Partition (GPU) to run on
#SBATCH --gres=gpu:L4:1                # Request 1 L4 GPU
#SBATCH --mem=78G                      # Request 64GB of memory
#SBATCH --time=07:00:00                # Job time limit (12 hours)

# Load required modules
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate Python virtual environment
source /home/bg91882/environments/gen/bin/activate

# Run the Python script with specified parameters


python main_mnist_with_glamv2.py --weight_decay 0 --label glam_v2_weight_decay_zero
