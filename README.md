# NSGA-II Multi-Objective Medical Ensemble

Multi-objective optimization of chest X-ray diagnostic ensembles using evolutionary algorithms to balance utility, calibration, and fairness.

## Overview

This repository demonstrates how to use **NSGA-II** to find optimal ensemble weights for medical imaging, optimizing three objectives simultaneously:

1. **Utility (AUC)** — Diagnostic accuracy
2. **Calibration (ACE)** — Prediction confidence matches true probability
3. **Equity (Fairness)** — Demographic parity across sex groups

**Key Result:** The optimized 3-objective ensemble achieves **AUC=0.9866, ACE=0.0053, equity gap=0.0003**, significantly improving calibration and fairness vs. equal-weight baselines with minimal AUC trade-off.

## Quick Start

### View Results

All analysis and visualizations are in the `results/` directory:

**Interactive Plots** — Open any `.html` file in a web browser:
- `results/analysis/pareto_3d.html` — 3D Pareto front (trade-off between AUC, ACE, equity gap)
- `results/analysis/pareto_2d.html` — 2D trade-off projections
- `results/analysis/knee_weights.html` — Optimal ensemble model weights (knee point)

**Data Files:**
- `results/analysis/pareto_solutions.csv` — All 100 Pareto solutions (weights + fitness values)
- `results/analysis/knee_point.json` — Optimal solution weights and metrics
- `results/nsga2/pareto_weights.npy` — Pareto front weights (100 × 7)
- `results/nsga2/pareto_fitness.npy` — Fitness values (100 × 3: AUC, ACE, equity gap)
- `results/nsga2/summary.json` — NSGA-II metadata (generations, population size, runtime)

### Baseline Comparison

| Method | AUC | ACE | Equity Gap |
|--------|-----|-----|------------|
| Single: ResNet101 (best) | 0.9837 | 0.0156 | 0.0012 |
| Ensemble: Equal weights | 0.9877 | 0.0211 | 0.0010 |
| **Ensemble: NSGA-II 3-obj (ours)** | **0.9866** | **0.0053** | **0.0003** |
| Ensemble: NSGA-II 2-obj (AUC+equity) | 0.9872 | 0.0097 | 0.0004 |

**Interpretation:** Your method sacrifices 0.0011 AUC vs. equal weights to gain **0.0158 on calibration** and **0.0007 on equity**—a favorable trade-off for trustworthy, fair medical AI.

## Key Configuration

Runtime and algorithmic parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Population Size** | 100 | NSGA-II population |
| **Generations** | 100 | Evolution iterations |
| **Workers** | 40 | Parallel CPU cores |
| **Crossover (SBX η)** | 20 | Crossover distribution |
| **Mutation (η)** | 20 | Mutation distribution |
| **Crossover Prob** | 0.9 | Probability per pair |
| **Mutation Prob** | 0.2 | Individual gene probability |
| **Ensemble Size** | 7 | Backbone models |
| **Batch Size** | 64 | Inference batch size |

### Objectives

- **f1 (Utility):** Negative AUC (minimize = maximize accuracy)
- **f2 (Calibration):** Adaptive Calibration Error (ACE) with 10 equal-mass bins
- **f3 (Fairness):** Absolute AUC difference between sex groups (M vs F)

### Models

7 diverse CNN backbones pre-trained on medical imaging:
- DenseNet121, ResNet50, ResNet101, EfficientNet-B4, VGG16, Inception-v3, MobileNetV2

## How It Works

```
Phase 0: Data Preparation (Optional)
├─ Download NIH ChestX-ray14 dataset
├─ Select coreset using ZCore (representative subset)
└─ Output: ~74K labeled chest X-ray images

Phase 1: Model Training
├─ Fine-tune 7 diverse CNN backbones on Pleural Effusion task
│  └─ Models: DenseNet121, ResNet50/101, EfficientNet-B4, VGG16, Inception-v3, MobileNetV2
├─ Train on full dataset (73,719 samples)
└─ Output: 7 trained model checkpoints

Phase 2: Generate Predictions Cache
├─ Run all 7 models on 73,719 samples (inference)
├─ Store: P_cache (73,719 × 7 predictions), labels, demographics
└─ Output: Cached predictions for efficient multi-objective optimization

Phase 3: NSGA-II Evolution
├─ Initialize: 100 random ensemble weight vectors
├─ For 100 generations:
│  ├─ Selection, crossover, mutation
│  ├─ Evaluate: f1 (AUC), f2 (ACE), f3 (equity gap)
│  └─ NSGA-II selection (100 survivors)
└─ Output: Pareto front (~100 non-dominated solutions)

Phase 4: Pareto Analysis
├─ Visualize Pareto (3D + 2D trade-off plots)
├─ Find knee point (best compromise solution)
└─ Output: Interactive HTML plots, optimal weights

Phase 5: Baseline Comparisons & Statistics
├─ Evaluate 11 methods (7 single models, 4 ensemble variants)
├─ 1000 bootstrap samples per method (confidence intervals)
├─ Permutation tests (statistical significance)
└─ Output: Comparison table + p-values
```

## Repository Structure

```
nsga2_medical_ensemble/
├── nsga2/
│   ├── objectives.py           # Fitness functions (f1, f2, f3)
│   └── run_nsga2.py            # NSGA-II evolution loop
├── models/
│   ├── generate_predictions.py # Run inference, save cache
│   ├── train_backbone.py       # Model training logic
│   └── backbones/              # Model weights
├── analysis/
│   ├── visualize_pareto.py     # Pareto plots + knee selection
│   ├── compute_baselines.py    # Baseline comparisons
├── results/
│   ├── nsga2/
│   │   ├── pareto_weights.npy  # Pareto solutions (100×7)
│   │   ├── pareto_fitness.npy  # Fitness values (100×3)
│   │   └── summary.json        # Metadata
│   └── analysis/
│       ├── pareto_3d.html      # 3D visualization
│       ├── pareto_2d.html      # 2D projections
│       ├── knee_weights.html   # Model importance
│       ├── baseline_comparison.csv
│       └── baseline_summary.json
├── logs/
│   ├── generate_predictions_*.out/err
│   ├── nsga2_*.out/err
│   ├── pareto_*.out/err
│   └── baselines_*.out/err
├── requirements.txt
└── README.md
```

## Data

- **Dataset:** NIH ChestX-ray14
- **Task:** Pleural Effusion detection (binary)
- **Samples:** 73,719 (Effusion + NoFinding)
- **Split:** All samples used for NSGA-II (no held-out test in current version)

## Technical Details

### NSGA-II Parameters
- **Crossover:** Simulated Binary Crossover (SBX) with η=20
- **Mutation:** Polynomial mutation with η=20, indpb=0.2
- **Selection:** NSGA-II elitism
- **Parallel Eval:** Multiprocessing (40 workers)

### Metrics
- **AUC:** Area under ROC curve
- **ACE:** Adaptive Calibration Error (equal-mass binning)
- **Equity Gap:** max(AUC_M, AUC_F) - min(AUC_M, AUC_F)

## Results

### Pareto Front
- **Size:** ~100 non-dominated solutions
- **Knee Point:** Solution 24 (best compromise)
- **Model Weights:** ResNet50 (31%), VGG16 (56%), others minor

### Baseline Performance
The 3-obj method outperforms all single models and equals 2-obj on AUC, while dramatically improving calibration and fairness.

### Interpretation
The Pareto front visualizes the fundamental trade-off: improving one objective typically requires sacrificing others. The knee point balances all three.

## Running the Pipeline

This is a **results repository**—the code and outputs are included. To reproduce:

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the NSGA-II optimization on your data (see scripts in `nsga2/` and `analysis/`)
4. View results in `results/analysis/`

## References

- **NSGA-II:** Deb et al. (2002) - "A fast and elitist multiobjective genetic algorithm: NSGA-II"
- **ACE:** Nixon et al. (2019) - "Measuring Calibration in Deep Learning"
- **NIH ChestX-ray14:** Wang et al. (2017)
- **TorchXRayVision:** Cohen et al. (2020)