#!/bin/bash
#SBATCH --job-name="overfit - no-sparsity"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=8GB
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out 
srun ./train-remote.sh
