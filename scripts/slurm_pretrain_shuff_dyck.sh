#!/bin/bash
#SBATCH --job-name=pretrain_shuff_dyck
#SBATCH --output=logs/pretrain_shuff_dyck_%j.out
#SBATCH --error=logs/pretrain_shuff_dyck_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1 -C "a100|h100"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load any necessary modules (adjust based on your cluster)
# module load python/3.9
# module load cuda/11.8

# Activate your virtual environment if needed
# source /path/to/venv/bin/activate

# Run the pre-pretraining on shuff_dyck
source .venv/bin/activate
python train.py \
    --model_name EleutherAI/pythia-160m \
    --data_dir ./data/tokenized/shuff_dyck \
    --output_dir ./output/shuff_dyck/pythia-160m \
    --save_steps 500 \
    --max_steps 500 \
    --reinit True
