#!/bin/bash
#SBATCH --job-name=ft_densenet121
#SBATCH --output=logs/finetune_densenet121_%j.out
#SBATCH --error=logs/finetune_densenet121_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=23:00:00

echo "=========================================="
echo "SLURM Job: Fine-tune DenseNet121"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Navigate to project directory
cd ~/nsga2_medical_ensemble

# Activate virtual environment
source ~/nsga2_medical_ensemble/venv/bin/activate

# Create logs directory
mkdir -p logs

# Run training
python models/train_backbone.py \
    --model_name densenet121 \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --checkpoint_dir models/checkpoints \
    --resume

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
