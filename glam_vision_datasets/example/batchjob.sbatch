#!/bin/bash 
#SBATCH --job-name=sam_example
#SBATCH --output=sam_example_ouput_o.log
#SBATCH --error=sam_example_error_o.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:L4:1# Request 1 L4 per node
#SBATCH --mem=78 # Request 64GB memory
#SBATCH --time=08:00:00 # Request 12 hours runtime

module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

#activate your environment
source /home/bg91882/environments/gen/bin/activate

python sam/example/train.py
