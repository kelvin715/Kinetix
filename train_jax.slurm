#!/bin/bash
#SBATCH --job-name=kinetix_ppo
#SBATCH --partition=gpu              # or whatever your GPU partition is
#SBATCH --gres=gpu:h100-96:1                 # number of GPUs
#SBATCH --cpus-per-task=32            # adjust as needed
#SBATCH --mem=128G                    # adjust as needed
#SBATCH --time=08:00:00              # walltime

#SBATCH --output=kinetix_plr_%j.out   # <--- stdout file

# 1. Go to your project
cd /home/y/youngjin/Kinetix

# 2. Load modules (if your cluster uses them)
module load cuda/12.9                # or the exact CUDA your JAX wheel expects

# 3. Activate your virtualenv
source .venv/bin/activate

# Optional: JAX/XLA memory flags (example)
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# or something like:
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# 4. Run your JAX script
python3 experiments/plr.py train_levels=green_pos_100 env_size=l eval=eval_auto
