#!/bin/bash
#SBATCH --job-name=nsga2
#SBATCH --output=logs/nsga2_%j.out
#SBATCH --error=logs/nsga2_%j.err
#SBATCH --cpus-per-task=40
#SBATCH --mem=80G
#SBATCH --time=23:00:00

echo "=========================================="
echo "SLURM Job: NSGA-II Evolution"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="

cd ~/nsga2_medical_ensemble
source ~/nsga2_medical_ensemble/venv/bin/activate
mkdir -p logs results/nsga2

python nsga2/run_nsga2.py \
    --pop_size 100 \
    --n_gen 100 \
    --n_workers 40 \
    --cache_dir data/cache \
    --output_dir results/nsga2 \
    --checkpoint_interval 10 \
    --resume

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
