#!/bin/bash
#SBATCH --job-name=gen_predictions
#SBATCH --output=logs/generate_predictions_%j.out
#SBATCH --error=logs/generate_predictions_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00

echo "=========================================="
echo "SLURM Job: Generate Predictions Cache"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

cd ~/nsga2_medical_ensemble
source ~/nsga2_medical_ensemble/venv/bin/activate
mkdir -p logs

python models/generate_predictions.py \
    --batch_size 64 \
    --models_dir models/backbones \
    --output_dir data/cache

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
