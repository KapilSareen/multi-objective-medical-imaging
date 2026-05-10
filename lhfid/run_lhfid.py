"""
LHFiD Evolution
Multi-objective optimization of ensemble weights using LHFiD algorithm
(upstream: https://github.com/mittalsukrit/LHFiD).

Uses modern pymoo API (>=0.6). Mirrors nsga2/run_nsga2.py structure but
runs LHFiD instead of NSGA-II.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import json
import time

from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

sys.path.insert(0, str(Path(__file__).parent.parent / "nsga2"))
from objectives import evaluate_ensemble

sys.path.insert(0, str(Path(__file__).parent))
from LHFiD import LHFID


# n_partitions=13 with 3 objectives -> C(15, 2) = 105 reference directions.
# This sets both pop_size and the structured reference grid to 105, matching
# NSGA-II's pop_size=100 closely while staying compatible with LHFiD's
# Das-Dennis grid requirement.
N_PARTITIONS = 13


class EnsembleProblem(Problem):
    """
    Pymoo Problem wrapping our 3-objective ensemble evaluation.

    Decision variables: 7 ensemble weights in [0, 1] (renormalized to a
        Dirichlet simplex inside _evaluate, matching the NSGA-II runner).
    Objectives:
        f1 = -AUC          (minimize  -> maximize AUC)
        f2 = ACE           (minimize)
        f3 = equity_gap    (minimize)
    """

    def __init__(self, P_cache, y_true, demographics):
        self.P_cache = P_cache
        self.y_true = y_true
        self.demographics = demographics

        n_models = P_cache.shape[1]
        super().__init__(
            n_var=n_models,
            n_obj=3,
            n_constr=0,
            xl=np.zeros(n_models),
            xu=np.ones(n_models),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate a batch of weight vectors."""
        F = np.zeros((X.shape[0], 3))

        for i, weights in enumerate(X):
            w = np.abs(weights)
            if w.sum() == 0:
                w = np.ones_like(w)
            w = w / w.sum()

            f1, f2, f3 = evaluate_ensemble(
                w, self.P_cache, self.y_true, self.demographics
            )
            F[i] = [f1, f2, f3]

        out["F"] = F


def find_knee_point(pareto_fitness):
    """Knee = solution closest to the ideal corner in normalized objective space."""
    normalized = (pareto_fitness - pareto_fitness.min(axis=0)) / (
        pareto_fitness.max(axis=0) - pareto_fitness.min(axis=0) + 1e-10
    )
    distances = np.sqrt(np.sum(normalized**2, axis=1))
    return int(np.argmin(distances))


def main():
    parser = argparse.ArgumentParser(description="LHFiD multi-objective optimization")
    parser.add_argument("--n_gen", type=int, default=100)
    parser.add_argument("--n_partitions", type=int, default=N_PARTITIONS,
                        help="Das-Dennis partitions; 13 -> 105 ref dirs / pop_size for 3 obj")
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--output_dir", type=str, default="results/lhfid")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 80)
    print("LHFiD MULTI-OBJECTIVE OPTIMIZATION")
    print("=" * 80)

    project_root = Path(__file__).parent.parent
    cache_dir = project_root / args.cache_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading cached predictions...")
    P_cache = np.load(cache_dir / "P_cache.npy")
    y_true = np.load(cache_dir / "y_true.npy")
    demographics = np.load(cache_dir / "demographics.npy", allow_pickle=True)
    demographics = demographics.astype(str)

    n_samples, n_models = P_cache.shape
    print(f"   Samples: {n_samples:,}")
    print(f"   Models:  {n_models}")

    # Build reference directions; this also defines pop_size for LHFiD.
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=args.n_partitions)
    pop_size = len(ref_dirs)

    print(f"\nSetting up LHFiD...")
    print(f"   n_partitions:        {args.n_partitions}")
    print(f"   Reference directions: {pop_size}")
    print(f"   Population size:      {pop_size}")
    print(f"   Generations:          {args.n_gen}")
    print(f"   Seed:                 {args.seed}")

    problem = EnsembleProblem(P_cache, y_true, demographics)

    # Configure LHFiD with explicit operators (matching upstream README example
    # and aligning eta/prob with NSGA-II for an apples-to-apples comparison).
    algorithm = LHFID(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        sampling=None,  # default FloatRandomSampling
        selection=RandomSelection(),
        crossover=SBX(prob=0.9, eta=20),
        mutation=PM(prob=1.0 / problem.n_var, eta=20),
        eliminate_duplicates=True,
    )

    print(f"\nStarting LHFiD evolution...")
    print("=" * 80)

    start_time = time.time()
    res = minimize(
        problem,
        algorithm,
        ("n_gen", args.n_gen),
        seed=args.seed,
        verbose=True,
    )
    elapsed = time.time() - start_time

    print("=" * 80)
    print(f"Optimization complete in {elapsed / 60:.1f} minutes")

    pareto_fitness = res.F
    pareto_weights_raw = res.X

    # Renormalize saved weights to the simplex so on-disk weights match
    # the ones used to compute the reported fitness.
    pareto_weights = np.zeros_like(pareto_weights_raw)
    for i, w in enumerate(pareto_weights_raw):
        w_abs = np.abs(w)
        if w_abs.sum() == 0:
            w_abs = np.ones_like(w_abs)
        pareto_weights[i] = w_abs / w_abs.sum()

    print(f"\nPareto front size: {len(pareto_fitness)}")

    knee_idx = find_knee_point(pareto_fitness)
    knee_w = pareto_weights[knee_idx]
    knee_f = pareto_fitness[knee_idx]
    print(f"Knee point: Solution {knee_idx}")
    print(f"   AUC:        {-knee_f[0]:.4f}")
    print(f"   ACE:        {knee_f[1]:.4f}")
    print(f"   Equity Gap: {knee_f[2]:.4f}")

    np.save(output_dir / "pareto_weights.npy", pareto_weights)
    np.save(output_dir / "pareto_fitness.npy", pareto_fitness)

    summary = {
        "algorithm": "LHFiD",
        "population_size": pop_size,
        "n_partitions": args.n_partitions,
        "ref_dirs": pop_size,
        "generations": args.n_gen,
        "pareto_size": len(pareto_fitness),
        "knee_index": knee_idx,
        "knee_metrics": {
            "auc": float(-knee_f[0]),
            "ace": float(knee_f[1]),
            "equity_gap": float(knee_f[2]),
        },
        "knee_weights": knee_w.tolist(),
        "final_stats": {
            "min": pareto_fitness.min(axis=0).tolist(),
            "avg": pareto_fitness.mean(axis=0).tolist(),
        },
        "total_time_minutes": elapsed / 60,
        "seed": args.seed,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved:")
    print(f"   {output_dir / 'pareto_weights.npy'}")
    print(f"   {output_dir / 'pareto_fitness.npy'}")
    print(f"   {output_dir / 'summary.json'}")

    print(f"\n{'=' * 80}")
    print("LHFiD COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
