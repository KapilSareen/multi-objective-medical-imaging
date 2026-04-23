#!/bin/bash
#SBATCH --job-name=pareto
#SBATCH --output=logs/pareto_%j.out
#SBATCH --error=logs/pareto_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

echo "=========================================="
echo "SLURM Job: Visualize Pareto"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

cd ~/nsga2_medical_ensemble
. ~/nsga2_medical_ensemble/venv/bin/activate
mkdir -p logs results/analysis

python analysis/visualize_pareto.py

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
