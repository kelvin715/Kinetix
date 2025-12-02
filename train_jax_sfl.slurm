#!/bin/bash
#SBATCH --job-name=kinetix_sfl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:2              # Request 2 GPUs for pmap
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00

#SBATCH --output=sfl_hard_pos_100_y_indix_%j.out

# 1. Go to your project
cd /home/y/youngjin/Kinetix

# 2. Load modules
module load cuda/12.9

# 3. Activate your virtualenv
source .venv/bin/activate

# 4. JAX multi-GPU settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Ensure JAX sees all GPUs (usually automatic, but explicit is safer)
# export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=false"

# 5. Verify GPU detection (optional debug)
python3 -c "import jax; print(f'JAX devices: {jax.devices()}')"

# 6. Run SFL
# Key: num_train_envs, num_batches, num_to_save must be divisible by num_gpus (2)
python3 experiments/sfl.py train_levels=hard_pos_100_y_indix env_size=l eval=eval_auto