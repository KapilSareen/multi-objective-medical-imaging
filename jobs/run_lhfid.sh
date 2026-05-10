#!/bin/bash
#SBATCH --job-name=lhfid
#SBATCH --output=logs/lhfid_%j.out
#SBATCH --error=logs/lhfid_%j.err
#SBATCH --cpus-per-task=40
#SBATCH --mem=80G
#SBATCH --time=23:00:00

echo "=========================================="
echo "SLURM Job: LHFiD Evolution"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="

cd ~/nsga2_medical_ensemble
source ~/nsga2_medical_ensemble/venv/bin/activate
mkdir -p logs results/lhfid

python lhfid/run_lhfid.py \
    --n_gen 100 \
    --n_partitions 13 \
    --cache_dir data/cache \
    --output_dir results/lhfid \
    --seed 42

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
