#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

echo "=========================================="
echo "SLURM Job: Algorithm Comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

cd ~/nsga2_medical_ensemble
source ~/nsga2_medical_ensemble/venv/bin/activate
mkdir -p results/analysis

python analysis/compare_algorithms.py

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
