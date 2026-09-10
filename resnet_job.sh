#!/bin/bash
#SBATCH --job-name=resnet18_dawnbench
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=dawnbench_resnet_%j.out
#SBATCH --error=dawnbench_resnet_%j.err

# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate torch

# full = train then test; use --mode inference to re-test saved weights, --mode train to skip testing
python dawnbench_resnet.py --mode full --weights resnet18_DAWNBench.pth

# Re-test an existing checkpoint:
# python -u dawnbench_resnet.py --mode inference --weights resnet18_cifar10.pth
