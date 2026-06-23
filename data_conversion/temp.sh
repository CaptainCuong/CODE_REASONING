#!/bin/bash


#-- SLURM Job Directives --#

#SBATCH --nodes=1                   # Request a single node

#SBATCH --ntasks-per-node=2         # Request 2 CPU cores

#SBATCH --time=00:00:30              # Set a 12-hour time limit

#SBATCH --partition=h200_normal_q   # Specify the GPU partition: h200_normal_q, a100_normal_q on Tinkercliffs | a30_normal_q o>

#SBATCH --account=cuong            # Your class-specific account: cuong, ece6514

#SBATCH --gres=gpu:1                # Request 2 GPUs

# Instructions for running ARC:
# https://docs.arc.vt.edu/usage/vscode_remote_ssh.html

module load CUDA/12.6.0

source ~/.bashrc
conda activate llama310
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"