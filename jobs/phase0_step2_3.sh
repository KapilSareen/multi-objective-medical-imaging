#!/bin/bash
#SBATCH --job-name=phase0_step2_3
#SBATCH --output=logs/phase0_step2_3_%j.out
#SBATCH --error=logs/phase0_step2_3_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

source ~/nsga2_medical_ensemble/venv/bin/activate
cd ~/nsga2_medical_ensemble

echo "=================================="
echo "PHASE 0 - STEPS 2-3"
echo "=================================="
echo ""

# Step 2: Generate CLIP embeddings (uses GPU)
python scripts/02_generate_embeddings.py --non-interactive

# Step 3: Run ZCore coreset selection (CPU only)
python scripts/03_run_zcore.py

echo ""
echo "✅ Phase 0 complete!"
