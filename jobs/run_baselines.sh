#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --cpus-per-task=40
#SBATCH --mem=40G
#SBATCH --time=08:00:00

echo "=========================================="
echo "SLURM Job: Baseline Comparisons"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="

cd ~/nsga2_medical_ensemble
source ~/nsga2_medical_ensemble/venv/bin/activate
mkdir -p logs results/analysis

python analysis/compute_baselines.py \
    --cache_dir   data/cache \
    --results_dir results/nsga2 \
    --output_dir  results/analysis \
    --n_bootstrap 1000 \
    --pop_size    100 \
    --n_gen       100 \
    --n_workers   40

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
