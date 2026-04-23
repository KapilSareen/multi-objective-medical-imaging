#!/bin/bash
# Submit all 7 model training jobs to Slurm
# They will queue sequentially due to 1 GPU/user limit

echo "=========================================="
echo "Submitting all 7 model training jobs"
echo "=========================================="

cd "$(dirname "$0")/.."

# Submit all jobs
for model in densenet121 resnet50 resnet101 efficientnet_b4 vgg16 inception_v3 mobilenet_v2; do
    job_id=$(sbatch jobs/finetune_${model}.sh | awk '{print $4}')
    echo "✅ Submitted $model (Job ID: $job_id)"
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor with: watch -n 10 'squeue -u \$USER'"
echo "Check logs:   tail -f logs/finetune_*_<job_id>.out"
echo ""
