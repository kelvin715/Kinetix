#!/bin/bash
#SBATCH --job-name=kinetix_sfl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=sfl_hard_pos_100_y_indix_%j.out

cd /home/y/youngjin/Kinetix
module load cuda/12.9
source .venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python3 experiments/sfl.py train_levels=hard_pos_100_y_indix env_size=l eval=eval_auto
