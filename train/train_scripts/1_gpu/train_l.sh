#!/bin/bash

#SBATCH -o logs/deployment/l_1_gpu.txt
#SBATCH -e logs/deployment_errors/l_1_gpu.txt
#SBATCH -J dino_1g
#SBATCH -p gpu_p

#SBATCH --mem=80G
#SBATCH -q gpu_normal
#SBATCH --time=48:00:00
#SBATCH --nice=10000
#SBATCH --ntasks=5
#SBATCH --gres=gpu:1
#SBATCH -C a100_80gb


# Environment setup
source $HOME/.bashrc
conda activate dinov2

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH



NAME="l_1_gpu"

python dinov2/train/train.py --name $NAME --no-resume --config-file dinov2/configs/train/deployment/1_gpu/l.yaml
