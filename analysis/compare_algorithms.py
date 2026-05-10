"""
Compare NSGA-II and LHFiD Pareto fronts.
Produces overlay visualizations and numeric metrics (hypervolume, IGD, knee-point).
"""

import sys
from pathlib import Path
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_hypervolume(fitness, ref_point):
    """
    Compute hypervolume indicator using pymoo's HV.
    Falls back to hvwfg if pymoo HV is unavailable.
    """
    try:
        from pymoo.indicators.hv import HV
        hv = HV(ref_point=ref_point)
        return float(hv(fitness))
    except ImportError:
        try:
            import hvwfg
            return float(hvwfg.wfg(fitness.astype(float), ref_point.astype(float)))
        except ImportError:
            print("WARNING: Neither pymoo HV nor hvwfg available; skipping HV.")
            return None


def compute_igd(fitness, reference_front):
    """Compute Inverted Generational Distance."""
    from scipy.spatial.distance import cdist
    distances = cdist(reference_front, fitness).min(axis=1)
    return float(distances.mean())


def find_knee_point(pareto_fitness):
    """Find knee point via normalized Euclidean distance to ideal."""
    normalized = (pareto_fitness - pareto_fitness.min(axis=0)) / (
        pareto_fitness.max(axis=0) - pareto_fitness.min(axis=0) + 1e-10
    )
    distances = np.sqrt(np.sum(normalized**2, axis=1))
    return int(np.argmin(distances))


def create_3d_overlay(nsga2_fit, lhfid_fit, nsga2_knee, lhfid_knee):
    """Create 3D scatter with both Pareto fronts overlaid."""
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=-nsga2_fit[:, 0], y=nsga2_fit[:, 1], z=nsga2_fit[:, 2],
        mode="markers",
        marker=dict(size=5, color="royalblue", opacity=0.7),
        name="NSGA-II",
    ))
    fig.add_trace(go.Scatter3d(
        x=-lhfid_fit[:, 0], y=lhfid_fit[:, 1], z=lhfid_fit[:, 2],
        mode="markers",
        marker=dict(size=5, color="darkorange", opacity=0.7),
        name="LHFiD",
    ))

    fig.add_trace(go.Scatter3d(
        x=[-nsga2_fit[nsga2_knee, 0]],
        y=[nsga2_fit[nsga2_knee, 1]],
        z=[nsga2_fit[nsga2_knee, 2]],
        mode="markers",
        marker=dict(size=12, color="blue", symbol="diamond"),
        name="NSGA-II Knee",
    ))
    fig.add_trace(go.Scatter3d(
        x=[-lhfid_fit[lhfid_knee, 0]],
        y=[lhfid_fit[lhfid_knee, 1]],
        z=[lhfid_fit[lhfid_knee, 2]],
        mode="markers",
        marker=dict(size=12, color="red", symbol="diamond"),
        name="LHFiD Knee",
    ))

    fig.update_layout(
        title="Pareto Front Comparison: NSGA-II vs LHFiD",
        scene=dict(
            xaxis_title="AUC",
            yaxis_title="ACE (Calibration Error)",
            zaxis_title="Equity Gap",
        ),
        width=1000,
        height=800,
    )
    return fig


def create_2d_overlay(nsga2_fit, lhfid_fit, nsga2_knee, lhfid_knee):
    """Create 2D projection subplots comparing both fronts."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("AUC vs ACE", "AUC vs Equity Gap", "ACE vs Equity Gap"),
    )

    pairs = [
        (lambda f: -f[:, 0], lambda f: f[:, 1], "AUC", "ACE"),
        (lambda f: -f[:, 0], lambda f: f[:, 2], "AUC", "Equity Gap"),
        (lambda f: f[:, 1], lambda f: f[:, 2], "ACE", "Equity Gap"),
    ]

    for col, (xfn, yfn, xlabel, ylabel) in enumerate(pairs, start=1):
        # NSGA-II points
        fig.add_trace(go.Scatter(
            x=xfn(nsga2_fit), y=yfn(nsga2_fit),
            mode="markers", marker=dict(size=7, color="royalblue", opacity=0.7),
            name="NSGA-II" if col == 1 else None,
            showlegend=(col == 1),
        ), row=1, col=col)

        # LHFiD points
        fig.add_trace(go.Scatter(
            x=xfn(lhfid_fit), y=yfn(lhfid_fit),
            mode="markers", marker=dict(size=7, color="darkorange", opacity=0.7),
            name="LHFiD" if col == 1 else None,
            showlegend=(col == 1),
        ), row=1, col=col)

        # NSGA-II knee
        fig.add_trace(go.Scatter(
            x=[xfn(nsga2_fit)[nsga2_knee]], y=[yfn(nsga2_fit)[nsga2_knee]],
            mode="markers", marker=dict(size=14, color="blue", symbol="diamond"),
            name="NSGA-II Knee" if col == 1 else None,
            showlegend=(col == 1),
        ), row=1, col=col)

        # LHFiD knee
        fig.add_trace(go.Scatter(
            x=[xfn(lhfid_fit)[lhfid_knee]], y=[yfn(lhfid_fit)[lhfid_knee]],
            mode="markers", marker=dict(size=14, color="red", symbol="diamond"),
            name="LHFiD Knee" if col == 1 else None,
            showlegend=(col == 1),
        ), row=1, col=col)

        fig.update_xaxes(title_text=xlabel, row=1, col=col)
        fig.update_yaxes(title_text=ylabel, row=1, col=col)

    fig.update_layout(
        height=450, width=1500, title_text="2D Projections: NSGA-II vs LHFiD"
    )
    return fig


def main():
    project_root = Path(__file__).parent.parent
    nsga2_dir = project_root / "results" / "nsga2"
    lhfid_dir = project_root / "results" / "lhfid"
    output_dir = project_root / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ALGORITHM COMPARISON: NSGA-II vs LHFiD")
    print("=" * 80)

    # Load Pareto fronts
    nsga2_fit = np.load(nsga2_dir / "pareto_fitness.npy")
    nsga2_wts = np.load(nsga2_dir / "pareto_weights.npy")
    lhfid_fit = np.load(lhfid_dir / "pareto_fitness.npy")
    lhfid_wts = np.load(lhfid_dir / "pareto_weights.npy")

    # Load summaries
    with open(nsga2_dir / "summary.json") as f:
        nsga2_summary = json.load(f)
    with open(lhfid_dir / "summary.json") as f:
        lhfid_summary = json.load(f)

    print(f"\nNSGA-II Pareto size: {len(nsga2_fit)}")
    print(f"LHFiD Pareto size:   {len(lhfid_fit)}")

    # Knee points
    nsga2_knee = find_knee_point(nsga2_fit)
    lhfid_knee = find_knee_point(lhfid_fit)

    print(f"\nNSGA-II Knee (sol {nsga2_knee}):")
    print(f"   AUC={-nsga2_fit[nsga2_knee, 0]:.4f}  "
          f"ACE={nsga2_fit[nsga2_knee, 1]:.4f}  "
          f"Gap={nsga2_fit[nsga2_knee, 2]:.6f}")
    print(f"LHFiD Knee (sol {lhfid_knee}):")
    print(f"   AUC={-lhfid_fit[lhfid_knee, 0]:.4f}  "
          f"ACE={lhfid_fit[lhfid_knee, 1]:.4f}  "
          f"Gap={lhfid_fit[lhfid_knee, 2]:.6f}")

    # Hypervolume (reference point chosen to dominate all solutions)
    ref_point = np.array([-0.95, 0.05, 0.05])
    nsga2_hv = compute_hypervolume(nsga2_fit, ref_point)
    lhfid_hv = compute_hypervolume(lhfid_fit, ref_point)

    print(f"\nHypervolume (ref={ref_point.tolist()}):")
    print(f"   NSGA-II: {nsga2_hv}")
    print(f"   LHFiD:   {lhfid_hv}")

    # IGD using combined front as reference
    combined = np.vstack([nsga2_fit, lhfid_fit])
    nsga2_igd = compute_igd(nsga2_fit, combined)
    lhfid_igd = compute_igd(lhfid_fit, combined)

    print(f"\nIGD (combined reference front):")
    print(f"   NSGA-II: {nsga2_igd:.6f}")
    print(f"   LHFiD:   {lhfid_igd:.6f}")

    # Runtimes
    nsga2_time = nsga2_summary.get("total_time_minutes", None)
    lhfid_time = lhfid_summary.get("total_time_minutes", None)
    print(f"\nRuntime:")
    print(f"   NSGA-II: {nsga2_time:.1f} min" if nsga2_time else "   NSGA-II: N/A")
    print(f"   LHFiD:   {lhfid_time:.1f} min" if lhfid_time else "   LHFiD: N/A")

    # Build comparison summary
    comparison = {
        "nsga2": {
            "pareto_size": len(nsga2_fit),
            "hypervolume": nsga2_hv,
            "igd": nsga2_igd,
            "runtime_minutes": nsga2_time,
            "knee_index": nsga2_knee,
            "knee_auc": float(-nsga2_fit[nsga2_knee, 0]),
            "knee_ace": float(nsga2_fit[nsga2_knee, 1]),
            "knee_equity_gap": float(nsga2_fit[nsga2_knee, 2]),
        },
        "lhfid": {
            "pareto_size": len(lhfid_fit),
            "hypervolume": lhfid_hv,
            "igd": lhfid_igd,
            "runtime_minutes": lhfid_time,
            "knee_index": lhfid_knee,
            "knee_auc": float(-lhfid_fit[lhfid_knee, 0]),
            "knee_ace": float(lhfid_fit[lhfid_knee, 1]),
            "knee_equity_gap": float(lhfid_fit[lhfid_knee, 2]),
        },
    }

    with open(output_dir / "algorithm_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # CSV summary
    import csv
    csv_path = output_dir / "algorithm_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "NSGA-II", "LHFiD"])
        writer.writerow(["Pareto Size", len(nsga2_fit), len(lhfid_fit)])
        writer.writerow(["Hypervolume", nsga2_hv, lhfid_hv])
        writer.writerow(["IGD", f"{nsga2_igd:.6f}", f"{lhfid_igd:.6f}"])
        writer.writerow(["Runtime (min)", nsga2_time, lhfid_time])
        writer.writerow(["Knee AUC", f"{comparison['nsga2']['knee_auc']:.4f}",
                         f"{comparison['lhfid']['knee_auc']:.4f}"])
        writer.writerow(["Knee ACE", f"{comparison['nsga2']['knee_ace']:.4f}",
                         f"{comparison['lhfid']['knee_ace']:.4f}"])
        writer.writerow(["Knee Equity Gap", f"{comparison['nsga2']['knee_equity_gap']:.6f}",
                         f"{comparison['lhfid']['knee_equity_gap']:.6f}"])

    print(f"\nSaved: {output_dir / 'algorithm_comparison.json'}")
    print(f"Saved: {csv_path}")

    # Generate plots
    print("\nGenerating visualizations...")

    fig_3d = create_3d_overlay(nsga2_fit, lhfid_fit, nsga2_knee, lhfid_knee)
    fig_3d.write_html(output_dir / "pareto_compare_3d.html")
    print(f"Saved: {output_dir / 'pareto_compare_3d.html'}")

    fig_2d = create_2d_overlay(nsga2_fit, lhfid_fit, nsga2_knee, lhfid_knee)
    fig_2d.write_html(output_dir / "pareto_compare_2d.html")
    print(f"Saved: {output_dir / 'pareto_compare_2d.html'}")

    # Print final summary table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<20} {'NSGA-II':<15} {'LHFiD':<15} {'Winner':<10}")
    print("-" * 60)

    winner_hv = "NSGA-II" if (nsga2_hv or 0) > (lhfid_hv or 0) else "LHFiD"
    winner_igd = "NSGA-II" if nsga2_igd < lhfid_igd else "LHFiD"
    winner_auc = ("NSGA-II" if comparison["nsga2"]["knee_auc"] > comparison["lhfid"]["knee_auc"]
                  else "LHFiD")
    winner_ace = ("NSGA-II" if comparison["nsga2"]["knee_ace"] < comparison["lhfid"]["knee_ace"]
                  else "LHFiD")
    winner_gap = ("NSGA-II" if comparison["nsga2"]["knee_equity_gap"] < comparison["lhfid"]["knee_equity_gap"]
                  else "LHFiD")

    print(f"{'Hypervolume':<20} {nsga2_hv:<15} {lhfid_hv:<15} {winner_hv}")
    print(f"{'IGD':<20} {nsga2_igd:<15.6f} {lhfid_igd:<15.6f} {winner_igd}")
    print(f"{'Knee AUC':<20} {comparison['nsga2']['knee_auc']:<15.4f} "
          f"{comparison['lhfid']['knee_auc']:<15.4f} {winner_auc}")
    print(f"{'Knee ACE':<20} {comparison['nsga2']['knee_ace']:<15.4f} "
          f"{comparison['lhfid']['knee_ace']:<15.4f} {winner_ace}")
    print(f"{'Knee Equity Gap':<20} {comparison['nsga2']['knee_equity_gap']:<15.6f} "
          f"{comparison['lhfid']['knee_equity_gap']:<15.6f} {winner_gap}")
    print("=" * 80)


if __name__ == "__main__":
    main()
