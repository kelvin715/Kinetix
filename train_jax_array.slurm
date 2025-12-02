#!/bin/bash
#SBATCH --job-name=kinetix_ppo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-3                       # <-- Number of jobs (0 to N-1)
#SBATCH --output=ppo_%A_%a.out            # %A = array job ID, %a = task index

cd /home/y/youngjin/Kinetix
module load cuda/12.9
source .venv/bin/activate

# export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Define your train_levels here (one per array index)
TRAIN_LEVELS=(
    "vary_joints_100"
    "hard_pos_100"
    "hard_pos_100_y"
    "hard_pos_100_y_indix"
)

# Get the train_level for this array task
LEVEL=${TRAIN_LEVELS[$SLURM_ARRAY_TASK_ID]}

echo "Running job $SLURM_ARRAY_TASK_ID with train_levels=$LEVEL"

python3 experiments/ppo.py train_levels=$LEVEL env_size=l eval=eval_grasper misc.group_auto_prefix="${LEVEL}-"

