"""
Phase 3: Pareto Front Visualization and Analysis
"""

import sys
from pathlib import Path
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def find_knee_point(pareto_fitness):
    """
    Find knee point using normalized Euclidean distance
    Knee point = maximum distance from ideal point
    """
    # Normalize objectives to [0, 1]
    normalized = (pareto_fitness - pareto_fitness.min(axis=0)) / \
                 (pareto_fitness.max(axis=0) - pareto_fitness.min(axis=0) + 1e-10)
    
    # Ideal point is (0, 0, 0) after normalization
    distances = np.sqrt(np.sum(normalized**2, axis=1))
    
    knee_idx = np.argmin(distances)
    return knee_idx


def create_3d_scatter(pareto_fitness, knee_idx, title="Pareto Front"):
    """Create 3D scatter plot of Pareto front"""
    fig = go.Figure()
    
    # All Pareto solutions
    fig.add_trace(go.Scatter3d(
        x=-pareto_fitness[:, 0],  # Convert back to positive AUC
        y=pareto_fitness[:, 1],
        z=pareto_fitness[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=np.arange(len(pareto_fitness)),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Solution #")
        ),
        text=[f"Solution {i}" for i in range(len(pareto_fitness))],
        name='Pareto Solutions'
    ))
    
    # Knee point
    fig.add_trace(go.Scatter3d(
        x=[-pareto_fitness[knee_idx, 0]],
        y=[pareto_fitness[knee_idx, 1]],
        z=[pareto_fitness[knee_idx, 2]],
        mode='markers',
        marker=dict(size=15, color='red', symbol='diamond'),
        text=['Knee Point'],
        name='Knee Point'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Utility (AUC)',
            yaxis_title='Trust Gap (ACE)',
            zaxis_title='Equity Gap (AUC Gap)',
        ),
        width=1000,
        height=800
    )
    
    return fig


def create_2d_projections(pareto_fitness, knee_idx):
    """Create 2D projections of all objective pairs"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('AUC vs ACE', 'AUC vs Equity', 'ACE vs Equity')
    )
    
    # AUC vs ACE
    fig.add_trace(
        go.Scatter(x=-pareto_fitness[:, 0], y=pareto_fitness[:, 1],
                  mode='markers', marker=dict(size=8, color='blue'),
                  name='Solutions'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[-pareto_fitness[knee_idx, 0]], 
                  y=[pareto_fitness[knee_idx, 1]],
                  mode='markers', marker=dict(size=15, color='red', symbol='diamond'),
                  name='Knee'),
        row=1, col=1
    )
    
    # AUC vs Equity
    fig.add_trace(
        go.Scatter(x=-pareto_fitness[:, 0], y=pareto_fitness[:, 2],
                  mode='markers', marker=dict(size=8, color='blue'),
                  showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=[-pareto_fitness[knee_idx, 0]], 
                  y=[pareto_fitness[knee_idx, 2]],
                  mode='markers', marker=dict(size=15, color='red', symbol='diamond'),
                  showlegend=False),
        row=1, col=2
    )
    
    # ACE vs Equity
    fig.add_trace(
        go.Scatter(x=pareto_fitness[:, 1], y=pareto_fitness[:, 2],
                  mode='markers', marker=dict(size=8, color='blue'),
                  showlegend=False),
        row=1, col=3
    )
    fig.add_trace(
        go.Scatter(x=[pareto_fitness[knee_idx, 1]], 
                  y=[pareto_fitness[knee_idx, 2]],
                  mode='markers', marker=dict(size=15, color='red', symbol='diamond'),
                  showlegend=False),
        row=1, col=3
    )
    
    fig.update_xaxes(title_text="AUC", row=1, col=1)
    fig.update_xaxes(title_text="AUC", row=1, col=2)
    fig.update_xaxes(title_text="ACE", row=1, col=3)
    
    fig.update_yaxes(title_text="ACE", row=1, col=1)
    fig.update_yaxes(title_text="Equity Gap", row=1, col=2)
    fig.update_yaxes(title_text="Equity Gap", row=1, col=3)
    
    fig.update_layout(height=400, width=1400, title_text="2D Projections")
    
    return fig


def analyze_weights(pareto_weights, knee_idx, model_names):
    """Analyze ensemble weights distribution"""
    knee_weights = pareto_weights[knee_idx]
    
    fig = go.Figure()
    
    # Bar chart of knee point weights
    fig.add_trace(go.Bar(
        x=model_names,
        y=knee_weights,
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title='Knee Point: Model Weights',
        xaxis_title='Model',
        yaxis_title='Weight',
        height=400,
        width=800
    )
    
    return fig


def main():
    print("="*80)
    print("PHASE 3: PARETO FRONT ANALYSIS")
    print("="*80)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results" / "nsga2"
    output_dir = project_root / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("\n📊 Loading NSGA-II results...")
    pareto_weights = np.load(results_dir / "pareto_weights.npy")
    pareto_fitness = np.load(results_dir / "pareto_fitness.npy")
    
    with open(results_dir / "summary.json", 'r') as f:
        summary = json.load(f)
    
    print(f"   Pareto solutions: {len(pareto_weights)}")
    print(f"   Generations: {summary['generations']}")
    print(f"   Total time: {summary['total_time_minutes']:.1f} min")
    
    # Find knee point
    print("\n🎯 Finding knee point...")
    knee_idx = find_knee_point(pareto_fitness)
    
    print(f"   Knee point: Solution #{knee_idx}")
    print(f"   AUC:        {-pareto_fitness[knee_idx, 0]:.4f}")
    print(f"   ACE:        {pareto_fitness[knee_idx, 1]:.4f}")
    print(f"   Equity Gap: {pareto_fitness[knee_idx, 2]:.4f}")
    
    # Model names
    model_names = ['DenseNet121', 'ResNet50', 'ResNet101', 
                   'EfficientNet-B4', 'VGG16', 'Inception-v3', 'MobileNetV2']
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    # 3D scatter
    fig_3d = create_3d_scatter(pareto_fitness, knee_idx)
    fig_3d.write_html(output_dir / "pareto_3d.html")
    print(f"   Saved: {output_dir / 'pareto_3d.html'}")
    
    # 2D projections
    fig_2d = create_2d_projections(pareto_fitness, knee_idx)
    fig_2d.write_html(output_dir / "pareto_2d.html")
    print(f"   Saved: {output_dir / 'pareto_2d.html'}")
    
    # Weights analysis
    fig_weights = analyze_weights(pareto_weights, knee_idx, model_names)
    fig_weights.write_html(output_dir / "knee_weights.html")
    print(f"   Saved: {output_dir / 'knee_weights.html'}")
    
    # Save knee point details
    knee_data = {
        'knee_index': int(knee_idx),
        'fitness': {
            'auc': float(-pareto_fitness[knee_idx, 0]),
            'ace': float(pareto_fitness[knee_idx, 1]),
            'equity_gap': float(pareto_fitness[knee_idx, 2])
        },
        'weights': {
            model_names[i]: float(pareto_weights[knee_idx, i])
            for i in range(len(model_names))
        }
    }
    
    with open(output_dir / "knee_point.json", 'w') as f:
        json.dump(knee_data, f, indent=2)
    print(f"   Saved: {output_dir / 'knee_point.json'}")
    
    # Create Pareto table
    pareto_df = pd.DataFrame({
        'Solution': range(len(pareto_fitness)),
        'AUC': -pareto_fitness[:, 0],
        'ACE': pareto_fitness[:, 1],
        'Equity_Gap': pareto_fitness[:, 2]
    })
    
    for i, name in enumerate(model_names):
        pareto_df[name] = pareto_weights[:, i]
    
    pareto_df.to_csv(output_dir / "pareto_solutions.csv", index=False)
    print(f"   Saved: {output_dir / 'pareto_solutions.csv'}")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\n📊 View results:")
    print(f"   3D Plot:  {output_dir / 'pareto_3d.html'}")
    print(f"   2D Plots: {output_dir / 'pareto_2d.html'}")
    print(f"   Weights:  {output_dir / 'knee_weights.html'}")
    print(f"   Data:     {output_dir / 'pareto_solutions.csv'}")
    print("="*80)


if __name__ == "__main__":
    main()
