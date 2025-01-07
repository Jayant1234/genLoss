#!/bin/bash 
#SBATCH --job-name=sam_example
#SBATCH --output=glam_0.15_similarity_cifar_ouput_o.log
#SBATCH --error=glam_0.15_similairty_cifar_error_o.log
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

python glam_cifar10.py --label glam_0.15_similarity_batch_256_200_epochs --batch_size 256
