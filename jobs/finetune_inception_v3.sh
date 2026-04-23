#!/bin/bash
#SBATCH --job-name=ft_inception_v3
#SBATCH --output=logs/finetune_inception_v3_%j.out
#SBATCH --error=logs/finetune_inception_v3_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=23:00:00

echo "=========================================="
echo "SLURM Job: Fine-tune Inception-v3"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

cd ~/nsga2_medical_ensemble
source ~/nsga2_medical_ensemble/venv/bin/activate
mkdir -p logs

python models/train_backbone.py \
    --model_name inception_v3 \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --checkpoint_dir models/checkpoints \
    --resume

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
