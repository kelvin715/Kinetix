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

# 4. JAX/CUDA multi-GPU settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Fix NCCL duplicate GPU issue
export NCCL_DEBUG=WARN
# Use shared memory for intra-node communication (more stable on some clusters)
export NCCL_SHM_DISABLE=0
# Disable P2P if GPUs are on different NUMA nodes or have connectivity issues
export NCCL_P2P_DISABLE=0
# Set unique NCCL ID per job
export NCCL_UNIQUE_ID=$SLURM_JOB_ID

# 5. Verify GPU detection
echo "=== GPU Debug Info ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L
python3 -c "
import jax
print(f'JAX devices: {jax.devices()}')
print(f'Device count: {jax.device_count()}')
print(f'Local device count: {jax.local_device_count()}')
# Check if devices have unique IDs
for i, d in enumerate(jax.devices()):
    print(f'  Device {i}: {d.id} - {d.device_kind}')
"
echo "======================"

# 6. Run SFL
# Key: num_train_envs, num_batches, num_to_save must be divisible by num_gpus (2)
python3 experiments/sfl.py train_levels=hard_pos_100_y_indix env_size=l eval=eval_auto