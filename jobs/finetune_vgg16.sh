#!/bin/bash
#SBATCH --job-name=ft_vgg16
#SBATCH --output=logs/finetune_vgg16_%j.out
#SBATCH --error=logs/finetune_vgg16_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=23:00:00

echo "=========================================="
echo "SLURM Job: Fine-tune VGG16"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

cd ~/nsga2_medical_ensemble
source ~/nsga2_medical_ensemble/venv/bin/activate
mkdir -p logs

python models/train_backbone.py \
    --model_name vgg16 \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --checkpoint_dir models/checkpoints \
    --resume

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
