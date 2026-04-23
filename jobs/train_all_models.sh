#!/bin/bash
#SBATCH --job-name=train_all_backbones
#SBATCH --output=logs/train_all_%j.out
#SBATCH --error=logs/train_all_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=23:00:00

echo "=========================================="
echo "PHASE 1: TRAIN ALL 7 BACKBONES"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

cd ~/nsga2_medical_ensemble
source ~/nsga2_medical_ensemble/venv/bin/activate
mkdir -p logs models/checkpoints models/backbones

MODELS=(densenet121 resnet50 resnet101 efficientnet_b4 vgg16 inception_v3 mobilenet_v2)
TOTAL=${#MODELS[@]}
FAILED=()

for i in "${!MODELS[@]}"; do
    MODEL=${MODELS[$i]}
    NUM=$((i+1))
    echo ""
    echo "=========================================="
    echo "[$NUM/$TOTAL] Training: $MODEL"
    echo "Started: $(date)"
    echo "=========================================="

    python models/train_backbone.py \
        --model_name $MODEL \
        --epochs 20 \
        --batch_size 32 \
        --lr 1e-4 \
        --checkpoint_dir models/checkpoints \
        --resume

    if [ $? -eq 0 ]; then
        echo "✅ [$NUM/$TOTAL] $MODEL DONE at $(date)"
    else
        echo "❌ [$NUM/$TOTAL] $MODEL FAILED"
        FAILED+=($MODEL)
    fi
done

echo ""
echo "=========================================="
echo "PHASE 1 COMPLETE at $(date)"
echo "Successful: $((TOTAL - ${#FAILED[@]}))/$TOTAL"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed: ${FAILED[*]}"
fi
echo "=========================================="
