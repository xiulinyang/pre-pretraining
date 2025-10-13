#!/bin/bash
#SBATCH --job-name=pretrain_c4
#SBATCH --output=logs/pretrain_c4_%j.out
#SBATCH --error=logs/pretrain_c4_%j.err
#SBATCH --time=48:00:00
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

# Run the pretraining on c4 (vanilla pretraining)
source .venv/bin/activate
python train.py \
    --model_name EleutherAI/pythia-160m \
    --data_dir ./data/tokenized/c4 \
    --output_dir ./output/c4/pythia-160m \
    --save_steps 2000 \
    --max_steps 10000 \
    --reinit True
