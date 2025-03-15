#!/bin/bash 
<<<<<<< Updated upstream
#SBATCH --job-name=adaptive_alpha_v4_finetuning
#SBATCH --error=adaptive_alpha_v4__finetuning__lr0.1error.log
#SBATCH --output=adaptive_alpha_v4__finetuning__lr0.1output.log
=======
#SBATCH --job-name=adaptive_alpha_v6_2
#SBATCH --error=adaptive_alpha_v6_2error.log
#SBATCH --output=adaptive_alpha_v6_@output.log
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
python adaptive_alpha_v4_finetuning.py 
=======
python adaptive_alpha_v4.py --label adaptive_alpha_v6_2 --seed 42
>>>>>>> Stashed changes

