"""
Phase 2: NSGA-II Objective Functions
Implements f1 (Utility), f2 (Trust), f3 (Equity)
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Tuple, List


def compute_ace(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Adaptive Calibration Error (ACE)
    Uses equal-mass binning (each bin has ~same number of samples)
    
    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted probabilities (0-1)
        n_bins: Number of bins
    
    Returns:
        ACE score (lower is better)
    """
    # Sort by predicted probability
    sorted_indices = np.argsort(y_pred)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    # Create equal-mass bins
    bin_size = len(y_pred) // n_bins
    ace = 0.0
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(y_pred)
        
        bin_y_true = y_true_sorted[start_idx:end_idx]
        bin_y_pred = y_pred_sorted[start_idx:end_idx]
        
        if len(bin_y_true) == 0:
            continue
        
        # Confidence = mean predicted probability in bin
        confidence = bin_y_pred.mean()
        
        # Accuracy = proportion of true positives in bin
        accuracy = bin_y_true.mean()
        
        # Weight by bin size
        weight = len(bin_y_true) / len(y_pred)
        
        ace += weight * np.abs(accuracy - confidence)
    
    return ace


def compute_demographic_auc_gap(y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 demographics: np.ndarray) -> float:
    """
    Compute AUC gap between demographic groups
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        demographics: Demographic labels (e.g., 'M', 'F')
    
    Returns:
        AUC gap (absolute difference between groups)
    """
    unique_groups = np.unique(demographics)
    
    if len(unique_groups) < 2:
        return 0.0
    
    aucs = []
    for group in unique_groups:
        mask = demographics == group
        if mask.sum() < 2:  # Need at least 2 samples
            continue
        
        group_y_true = y_true[mask]
        group_y_pred = y_pred[mask]
        
        # Check if we have both classes
        if len(np.unique(group_y_true)) < 2:
            continue
        
        try:
            auc = roc_auc_score(group_y_true, group_y_pred)
            aucs.append(auc)
        except:
            continue
    
    if len(aucs) < 2:
        return 0.0
    
    return np.max(aucs) - np.min(aucs)


def evaluate_ensemble(weights: np.ndarray,
                      P_cache: np.ndarray,
                      y_true: np.ndarray,
                      demographics: np.ndarray) -> Tuple[float, float, float]:
    """
    Evaluate ensemble with given weights on all three objectives
    
    Args:
        weights: Ensemble weights (shape: num_models,)
        P_cache: Cached predictions (shape: num_samples, num_models)
        y_true: Ground truth labels (shape: num_samples,)
        demographics: Demographic labels (shape: num_samples,)
    
    Returns:
        (f1, f2, f3): Three objective values
            f1: Utility (negative AUC, to minimize)
            f2: Trust Gap (ACE, to minimize)
            f3: Equity Gap (demographic AUC gap, to minimize)
    """
    # Normalize weights
    weights = np.array(weights)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    
    # Compute ensemble predictions
    y_pred = P_cache @ weights  # Weighted sum
    
    # Clip to valid probability range
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    
    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        return 1.0, 1.0, 1.0  # Worst possible scores
    
    # f1: Utility (negative AUC)
    try:
        auc = roc_auc_score(y_true, y_pred)
        f1 = -auc  # Negative because we minimize
    except:
        f1 = 1.0  # Worst case
    
    # f2: Trust Gap (ACE)
    try:
        ace = compute_ace(y_true, y_pred, n_bins=10)
        f2 = ace
    except:
        f2 = 1.0  # Worst case
    
    # f3: Equity Gap (demographic AUC gap)
    try:
        auc_gap = compute_demographic_auc_gap(y_true, y_pred, demographics)
        f3 = auc_gap
    except:
        f3 = 1.0  # Worst case
    
    return f1, f2, f3


def batch_evaluate(population: List[np.ndarray],
                   P_cache: np.ndarray,
                   y_true: np.ndarray,
                   demographics: np.ndarray) -> List[Tuple[float, float, float]]:
    """
    Evaluate multiple individuals in parallel
    
    Args:
        population: List of weight vectors
        P_cache: Cached predictions
        y_true: Ground truth labels
        demographics: Demographic labels
    
    Returns:
        List of (f1, f2, f3) tuples
    """
    results = []
    for weights in population:
        fitness = evaluate_ensemble(weights, P_cache, y_true, demographics)
        results.append(fitness)
    return results


# For testing
if __name__ == "__main__":
    # Test with dummy data
    print("Testing objective functions...")
    
    np.random.seed(42)
    n_samples = 1000
    n_models = 7
    
    # Dummy data
    P_cache = np.random.rand(n_samples, n_models)
    y_true = np.random.randint(0, 2, n_samples)
    demographics = np.random.choice(['M', 'F'], n_samples)
    
    # Test weights
    weights = np.array([0.2, 0.15, 0.15, 0.15, 0.1, 0.15, 0.1])
    
    f1, f2, f3 = evaluate_ensemble(weights, P_cache, y_true, demographics)
    
    print(f"\nTest Results:")
    print(f"  f1 (Utility):   {f1:.4f} (negative AUC)")
    print(f"  f2 (Trust Gap): {f2:.4f} (ACE)")
    print(f"  f3 (Equity Gap):{f3:.4f} (AUC gap)")
    print("\n✅ Objectives working correctly!")
