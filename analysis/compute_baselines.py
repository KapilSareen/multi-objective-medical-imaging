"""
Phase 4: Baseline Comparisons + Statistical Analysis
Compares NSGA-II knee-point ensemble against all baselines using bootstrap CIs.

Baselines:
  1. Single models (7x)          - best individual model
  2. Equal-weight ensemble       - P_cache average
  3. Max-AUC greedy ensemble     - full weight on best single model
  4. 2-objective NSGA-II         - re-run without ACE (f2 disabled)
  5. NSGA-II knee point          - our method (from Phase 2)

Metrics per baseline: AUC, ACE, Equity Gap (M vs F AUC gap)
Bootstrap: 1000 resamples, 95% CI on each metric
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "nsga2"))
from objectives import compute_ace, compute_demographic_auc_gap, evaluate_ensemble

MODEL_NAMES = ['DenseNet121', 'ResNet50', 'ResNet101',
               'EfficientNet-B4', 'VGG16', 'Inception-v3', 'MobileNetV2']

N_BOOTSTRAP = 1000
CI_LEVEL     = 0.95
RNG          = np.random.default_rng(42)


# ─── Metric helpers ──────────────────────────────────────────────────────────

def compute_all_metrics(y_pred, y_true, demographics):
    """Return (AUC, ACE, equity_gap) for a prediction vector."""
    demographics = np.array(demographics).astype(str)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float('nan')
    ace = compute_ace(y_true, y_pred, n_bins=10)
    gap = compute_demographic_auc_gap(y_true, y_pred, demographics)
    return auc, ace, gap


def bootstrap_ci(y_pred, y_true, demographics, n=N_BOOTSTRAP):
    """
    Bootstrap 95% CIs for AUC, ACE, equity_gap.
    Returns dict with keys: auc, ace, gap — each (mean, lower, upper).
    """
    n_samples = len(y_true)
    aucs, aces, gaps = [], [], []

    for _ in range(n):
        idx = RNG.integers(0, n_samples, size=n_samples)
        yp = y_pred[idx]
        yt = y_true[idx]
        dm = demographics[idx]
        if len(np.unique(yt)) < 2:
            continue
        a, c, g = compute_all_metrics(yp, yt, dm)
        if not np.isnan(a):
            aucs.append(a)
            aces.append(c)
            gaps.append(g)

    alpha = (1 - CI_LEVEL) / 2

    def ci(vals):
        arr = np.array(vals)
        return float(arr.mean()), float(np.percentile(arr, alpha*100)), float(np.percentile(arr, (1-alpha)*100))

    return {
        'auc': ci(aucs),
        'ace': ci(aces),
        'gap': ci(gaps),
    }


def fmt(mean, lo, hi, higher_better=True):
    """Format metric as 'mean (lo–hi)' with arrow indicating direction."""
    arrow = '↑' if higher_better else '↓'
    return f"{mean:.4f} [{lo:.4f}–{hi:.4f}] {arrow}"


# ─── 2-objective NSGA-II baseline ────────────────────────────────────────────

def run_2obj_nsga2(P_cache, y_true, demographics,
                   pop_size=100, n_gen=100, n_workers=40):
    """
    Re-run NSGA-II with only 2 objectives: AUC + equity gap (no ACE).
    Returns the knee-point weight vector.
    """
    from deap import base, creator, tools
    import multiprocessing as mp

    # selTournamentDCD requires pop_size divisible by 4
    pop_size = pop_size + (4 - pop_size % 4) % 4
    print("\n   Running 2-objective NSGA-II (AUC + equity, no calibration)...")

    # Fresh creator names to avoid collision with 3-obj run
    if not hasattr(creator, "Fitness2Obj"):
        creator.create("Fitness2Obj", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "Ind2Obj"):
        creator.create("Ind2Obj", list, fitness=creator.Fitness2Obj)

    n_models = P_cache.shape[1]

    def eval_2obj(ind):
        w = np.array(ind)
        w = np.abs(w) / (np.abs(w).sum() + 1e-10)
        y_pred = np.clip(P_cache @ w, 1e-7, 1 - 1e-7)
        if len(np.unique(y_true)) < 2:
            return 1.0, 1.0
        try:
            f1 = -roc_auc_score(y_true, y_pred)
        except Exception:
            f1 = 1.0
        try:
            f3 = compute_demographic_auc_gap(y_true, y_pred, demographics)
        except Exception:
            f3 = 1.0
        return f1, f3

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Ind2Obj,
                     lambda: np.random.dirichlet(np.ones(n_models)).tolist())
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_2obj)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    pop[:] = toolbox.select(pop, pop_size)

    t0 = time.time()
    for gen in range(n_gen):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < 0.9:
                a, b = np.array(c1[:]), np.array(c2[:])
                alpha = np.random.rand(n_models)
                new1 = np.abs(alpha * a + (1 - alpha) * b)
                new2 = np.abs((1 - alpha) * a + alpha * b)
                c1[:] = (new1 / new1.sum()).tolist()
                c2[:] = (new2 / new2.sum()).tolist()
                del c1.fitness.values, c2.fitness.values

        for mut in offspring:
            if np.random.rand() < 0.2:
                w = np.abs(np.array(mut[:]) + np.random.randn(n_models) * 0.1)
                mut[:] = (w / w.sum()).tolist()
                del mut.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = toolbox.select(pop + offspring, pop_size)

        if (gen + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (gen + 1) * (n_gen - gen - 1)
            print(f"   Gen {gen+1}/{n_gen}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

    pareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

    # Knee point: min distance to ideal in normalized space
    fits = np.array([ind.fitness.values for ind in pareto])
    norm = (fits - fits.min(0)) / (fits.max(0) - fits.min(0) + 1e-10)
    knee_idx = int(np.argmin(np.sqrt(np.sum(norm**2, axis=1))))
    knee_weights = np.array(pareto[knee_idx][:])
    knee_weights = np.abs(knee_weights) / knee_weights.sum()

    print(f"   2-obj Pareto size: {len(pareto)}, knee AUC={-fits[knee_idx,0]:.4f}, gap={fits[knee_idx,1]:.4f}")
    return knee_weights


def permutation_test(y_a, y_b, y_true, demographics, metric='ace', n_perm=1000):
    """
    One-sided permutation test: is metric(A) significantly lower than metric(B)?
    obs_diff = metric(A) - metric(B)  — negative means A is better.
    p-value = fraction of permutations where perm_diff <= obs_diff.
    Low p-value means the observed improvement is unlikely by chance.
    """
    def get_metric(yp):
        _, ace, gap = compute_all_metrics(yp, y_true, demographics)
        return ace if metric == 'ace' else gap

    obs_diff = get_metric(y_a) - get_metric(y_b)   # negative = A is better
    count = 0
    for _ in range(n_perm):
        # Under null hypothesis, randomly reassign which predictor is "A"
        mask     = RNG.random(len(y_true)) > 0.5
        y_perm_a = np.where(mask, y_a, y_b)
        y_perm_b = np.where(mask, y_b, y_a)
        perm_diff = get_metric(y_perm_a) - get_metric(y_perm_b)
        if perm_diff <= obs_diff:
            count += 1
    p_value = count / n_perm
    return obs_diff, p_value


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir',   type=str, default='data/cache')
    parser.add_argument('--results_dir', type=str, default='results/nsga2')
    parser.add_argument('--output_dir',  type=str, default='results/analysis')
    parser.add_argument('--n_bootstrap', type=int, default=N_BOOTSTRAP)
    parser.add_argument('--pop_size',    type=int, default=100)
    parser.add_argument('--n_gen',       type=int, default=100)
    parser.add_argument('--n_workers',   type=int, default=40)
    args = parser.parse_args()

    print("="*80)
    print("PHASE 4: BASELINE COMPARISONS + STATISTICAL ANALYSIS")
    print("="*80)

    cache_dir   = ROOT / args.cache_dir
    results_dir = ROOT / args.results_dir
    output_dir  = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n📊 Loading cached predictions...")
    P_cache      = np.load(cache_dir / "P_cache.npy")
    y_true       = np.load(cache_dir / "y_true.npy")
    demographics = np.load(cache_dir / "demographics.npy", allow_pickle=True).astype(str)
    n_samples, n_models = P_cache.shape
    print(f"   Samples: {n_samples:,}  Models: {n_models}")
    print(f"   Positive rate: {y_true.mean():.3f}")
    print(f"   Gender: M={np.sum(demographics=='M'):,}  F={np.sum(demographics=='F'):,}")

    # ── Load NSGA-II knee point ───────────────────────────────────────────────
    print("\n📂 Loading NSGA-II results...")
    pareto_weights = np.load(results_dir / "pareto_weights.npy")
    pareto_fitness = np.load(results_dir / "pareto_fitness.npy")

    with open(results_dir / "summary.json") as f:
        nsga2_summary = json.load(f)

    # Find knee point (same logic as visualize_pareto.py)
    norm = (pareto_fitness - pareto_fitness.min(0)) / \
           (pareto_fitness.max(0) - pareto_fitness.min(0) + 1e-10)
    knee_idx    = int(np.argmin(np.sqrt(np.sum(norm**2, axis=1))))
    knee_weights = pareto_weights[knee_idx]
    print(f"   Pareto size: {len(pareto_weights)}")
    print(f"   Knee index:  {knee_idx}")

    # ── Build all baselines ───────────────────────────────────────────────────
    print("\n🔧 Building baselines...")

    baselines = {}   # name -> y_pred array

    # 1. Single models
    for i, name in enumerate(MODEL_NAMES[:n_models]):
        baselines[f"Single: {name}"] = P_cache[:, i]

    # 2. Equal-weight ensemble
    baselines["Ensemble: Equal weights"] = P_cache.mean(axis=1)

    # 3. Max-AUC greedy (single best model by AUC, full weight)
    single_aucs = [roc_auc_score(y_true, P_cache[:, i]) for i in range(n_models)]
    best_idx = int(np.argmax(single_aucs))
    baselines[f"Ensemble: Greedy best ({MODEL_NAMES[best_idx]})"] = P_cache[:, best_idx]
    print(f"   Best single model: {MODEL_NAMES[best_idx]} (AUC={single_aucs[best_idx]:.4f})")

    # 4. NSGA-II knee point (3-objective, our method)
    nsga2_pred = np.clip(P_cache @ knee_weights, 1e-7, 1 - 1e-7)
    baselines["Ensemble: NSGA-II 3-obj (ours)"] = nsga2_pred

    # 5. 2-objective NSGA-II (AUC + equity, no calibration) — ablation
    t2 = time.time()
    weights_2obj = run_2obj_nsga2(P_cache, y_true, demographics,
                                   pop_size=args.pop_size,
                                   n_gen=args.n_gen,
                                   n_workers=args.n_workers)
    pred_2obj = np.clip(P_cache @ weights_2obj, 1e-7, 1 - 1e-7)
    baselines["Ensemble: NSGA-II 2-obj (AUC+equity)"] = pred_2obj
    print(f"   2-obj run time: {(time.time()-t2)/60:.1f}m")

    # ── Evaluate all baselines ────────────────────────────────────────────────
    print(f"\n📈 Evaluating {len(baselines)} baselines with {args.n_bootstrap} bootstrap samples...")
    print("   (This may take a few minutes...)")

    rows = []
    t0 = time.time()

    for i, (name, y_pred) in enumerate(baselines.items()):
        t_start = time.time()
        y_pred = np.array(y_pred)

        # Point estimates
        auc, ace, gap = compute_all_metrics(y_pred, y_true, demographics)

        # Bootstrap CIs
        cis = bootstrap_ci(y_pred, y_true, demographics, n=args.n_bootstrap)

        rows.append({
            'Method':          name,
            'AUC':             round(auc,  4),
            'AUC_lo':          round(cis['auc'][1], 4),
            'AUC_hi':          round(cis['auc'][2], 4),
            'ACE':             round(ace,  4),
            'ACE_lo':          round(cis['ace'][1], 4),
            'ACE_hi':          round(cis['ace'][2], 4),
            'Equity_Gap':      round(gap,  4),
            'Gap_lo':          round(cis['gap'][1], 4),
            'Gap_hi':          round(cis['gap'][2], 4),
        })

        elapsed = time.time() - t_start
        print(f"   [{i+1}/{len(baselines)}] {name}")
        print(f"         AUC={auc:.4f}  ACE={ace:.4f}  Gap={gap:.4f}  ({elapsed:.0f}s)")

    print(f"\n   Total evaluation time: {(time.time()-t0)/60:.1f}m")

    # ── Save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)

    # Pretty-print table
    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80)
    print(f"\n{'Method':<45} {'AUC':>10} {'ACE':>10} {'Equity Gap':>12}")
    print("-"*80)
    for _, row in df.iterrows():
        marker = " ◀ OUR METHOD" if "3-obj" in row['Method'] else ""
        print(f"{row['Method']:<45} "
              f"{row['AUC']:.4f} [{row['AUC_lo']:.4f}-{row['AUC_hi']:.4f}]  "
              f"{row['ACE']:.4f}  "
              f"{row['Equity_Gap']:.4f}{marker}")
    print("="*80)

    # ── Significance test: NSGA-II 3-obj vs equal-weight ─────────────────────
    print("\n📊 Permutation significance test: NSGA-II 3-obj vs Equal-weight ensemble")
    y_nsga2  = baselines["Ensemble: NSGA-II 3-obj (ours)"]
    y_equal  = baselines["Ensemble: Equal weights"]

    ace_diff, ace_p  = permutation_test(y_nsga2, y_equal, y_true, demographics, 'ace')
    gap_diff, gap_p  = permutation_test(y_nsga2, y_equal, y_true, demographics, 'gap')

    print(f"   ACE improvement:         {-ace_diff:.4f}  (p={ace_p:.4f}{'  ✓ significant' if ace_p < 0.05 else '  ✗ not significant'})")
    print(f"   Equity gap improvement:  {-gap_diff:.4f}  (p={gap_p:.4f}{'  ✓ significant' if gap_p < 0.05 else '  ✗ not significant'})")

    # ── Save all outputs ──────────────────────────────────────────────────────
    csv_path = output_dir / "baseline_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved: {csv_path}")

    summary = {
        'n_samples':       int(n_samples),
        'n_models':        int(n_models),
        'n_bootstrap':     args.n_bootstrap,
        'best_single_model': {
            'name': MODEL_NAMES[best_idx],
            'auc':  round(single_aucs[best_idx], 4),
        },
        'nsga2_knee': {
            'auc':        float(round(-pareto_fitness[knee_idx, 0], 4)),
            'ace':        float(round( pareto_fitness[knee_idx, 1], 4)),
            'equity_gap': float(round( pareto_fitness[knee_idx, 2], 4)),
        },
        'significance': {
            'ace_improvement':        round(-ace_diff, 4),
            'ace_p_value':            round(ace_p,     4),
            'equity_gap_improvement': round(-gap_diff, 4),
            'equity_gap_p_value':     round(gap_p,     4),
        },
        'weights_2obj': weights_2obj.tolist(),
    }

    with open(output_dir / "baseline_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved: {output_dir / 'baseline_summary.json'}")

    np.save(output_dir / "weights_2obj_nsga2.npy", weights_2obj)
    print(f"💾 Saved: {output_dir / 'weights_2obj_nsga2.npy'}")

    print("\n" + "="*80)
    print("✅ PHASE 4 COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
