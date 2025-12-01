#!/bin/bash
#SBATCH --job-name=kinetix_ppo
#SBATCH --partition=gpu              # or whatever your GPU partition is
#SBATCH --gres=gpu:h100-47:1                 # number of GPUs
#SBATCH --cpus-per-task=32            # adjust as needed
#SBATCH --mem=128G                    # adjust as needed
#SBATCH --time=08:00:00              # walltime

#SBATCH --output=plr_hard_pos_100_y_indix_%j.out   # <--- stdout file

# 1. Go to your project
cd /home/y/youngjin/Kinetix

# 2. Load modules (if your cluster uses them)
module load cuda/12.9                # or the exact CUDA your JAX wheel expects

# 3. Activate your virtualenv
source .venv/bin/activate

# Optional: JAX/XLA memory flags (example)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# or something like:
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# 4. Run your JAX script
# NOTE: num_steps × outer_rollout_steps MUST equal env_params.max_timesteps (256)
python3 experiments/plr.py train_levels=hard_pos_100_y_indix env_size=l eval=eval_auto \
  learning.num_train_envs=512 \
  learning.outer_rollout_steps=4 \
  learning.num_steps=64 \
  ued.level_buffer_capacity=1000 \
  eval.eval_num_attempts=5 \
  misc.video_frequency=50