#!/bin/bash
#SBATCH --job-name=cuda_app
#SBATCH --output=cuda.out.%j
#SBATCH --error=cuda.err.%j
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
#SBATCH --mem=32G

nvidia-smi
module load cuda

make clean
make

# Run
./driver_accuracy
./driver_profile
./driver_perf
