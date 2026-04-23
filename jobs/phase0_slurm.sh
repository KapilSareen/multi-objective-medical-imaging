#!/bin/bash
#SBATCH --job-name=phase0_coreset
#SBATCH --output=logs/phase0_%j.out
#SBATCH --error=logs/phase0_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

source ~/nsga2_medical_ensemble/venv/bin/activate
cd ~/nsga2_medical_ensemble

bash scripts/run_phase0.sh
