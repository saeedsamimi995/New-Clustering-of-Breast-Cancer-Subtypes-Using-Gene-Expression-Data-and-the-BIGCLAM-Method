"""
Advanced feature selection utilities.

Pipeline:
    1. Variance filtering (threshold defaults to mean variance).
    2. Correlation pruning to remove redundant genes.
    3. Laplacian Score ranking on a k-NN similarity graph.

The implementation follows the workflow requested by the user and is
designed to keep the number of informative genes in the 100–1,000 range
by default while remaining fully configurable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.neighbors import kneighbors_graph


@dataclass
class FeatureSelectionResult:
    """Container for feature selection outputs."""

    data: np.ndarray
    selected_indices: np.ndarray
    selected_gene_names: Optional[np.ndarray]
    variance_threshold: float
    correlation_threshold: float
    laplacian_scores: np.ndarray
    stage_counts: Dict[str, int]

    def to_dict(self) -> Dict:
        """Convert dataclass to plain dictionary."""
        return asdict(self)


def _resolve_threshold(values: np.ndarray, threshold) -> float:
    """Resolve numeric threshold with mean/median helpers."""
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return 0.0

    if isinstance(threshold, str):
        key = threshold.lower()
        if key == "mean":
            return float(np.mean(clean))
        if key == "median":
            return float(np.median(clean))
        raise ValueError(f"Unsupported threshold string: {threshold}")

    return float(threshold)


def _variance_filter(
    data: np.ndarray,
    threshold="mean",
    verbose: bool = True,
) -> (np.ndarray, np.ndarray, float, np.ndarray):
    """Remove genes with variance below threshold."""
    variances = np.var(data, axis=0)
    thresh_value = _resolve_threshold(variances, threshold)
    keep_mask = variances >= thresh_value

    if verbose:
        kept = int(np.sum(keep_mask))
        print(
            f"\n[Feature Selection] Variance filter: threshold={thresh_value:.4f} "
            f"-> kept {kept:,}/{len(variances):,} genes ({kept / len(variances) * 100:.1f}%)"
        )

    filtered = data[:, keep_mask]
    kept_indices = np.where(keep_mask)[0]
    return filtered, kept_indices, thresh_value, variances


def _correlation_prune(
    data: np.ndarray,
    original_indices: np.ndarray,
    threshold="mean",
    verbose: bool = True,
) -> (np.ndarray, np.ndarray, float):
    """Remove redundant genes using correlation groups."""
    if data.shape[1] == 0:
        return data, original_indices, 0.0

    corr_matrix = np.corrcoef(data, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    triu_indices = np.triu_indices_from(corr_matrix, k=1)
    abs_values = np.abs(corr_matrix[triu_indices])
    thresh_value = _resolve_threshold(abs_values, threshold)

    visited = set()
    kept_relative_indices: List[int] = []

    for gene_idx in range(corr_matrix.shape[0]):
        if gene_idx in visited:
            continue
        kept_relative_indices.append(gene_idx)
        correlated = np.where(np.abs(corr_matrix[gene_idx]) >= thresh_value)[0]
        visited.update(correlated.tolist())

    kept_relative_indices = np.array(kept_relative_indices, dtype=int)
    pruned_data = data[:, kept_relative_indices]
    pruned_indices = original_indices[kept_relative_indices]

    if verbose:
        print(
            f"[Feature Selection] Correlation pruning: threshold={thresh_value:.4f} "
            f"-> kept {len(kept_relative_indices):,}/{corr_matrix.shape[0]:,} genes"
        )

    return pruned_data, pruned_indices, thresh_value


def _compute_laplacian_scores(
    data: np.ndarray,
    k_neighbors: int = 5,
    verbose: bool = True,
) -> np.ndarray:
    """Compute Laplacian Scores for each gene."""
    n_samples, n_genes = data.shape

    if n_genes == 0:
        return np.array([])

    effective_k = max(1, min(k_neighbors, n_samples - 1))
    if effective_k < 1:
        return np.full(n_genes, np.inf)

    W_sparse = kneighbors_graph(
        data, n_neighbors=effective_k, mode="connectivity", include_self=False
    )
    # Symmetrize and densify for Laplacian operations.
    W = W_sparse.toarray()
    W = 0.5 * (W + W.T)
    D = np.sum(W, axis=1)
    L = np.diag(D) - W

    scores = np.zeros(n_genes, dtype=float)
    epsilon = 1e-10

    for idx in range(n_genes):
        f = data[:, idx]
        numerator = float(f.T @ (L @ f))
        denominator = float(np.dot(f * D, f))
        if math.isfinite(denominator) and denominator > epsilon:
            scores[idx] = numerator / denominator
        else:
            scores[idx] = np.inf

    if verbose:
        print("[Feature Selection] Laplacian scores computed.")

    return scores


def run_feature_selection(
    data: np.ndarray,
    gene_names: Optional[Iterable[str]] = None,
    variance_threshold="mean",
    correlation_threshold="mean",
    laplacian_neighbors: int = 5,
    num_selected_features: Optional[int] = None,
    verbose: bool = True,
) -> FeatureSelectionResult:
    """
    Execute the three-stage feature selection pipeline.

    Args:
        data: Expression matrix (samples × genes).
        gene_names: Optional iterable of gene identifiers.
        variance_threshold: Threshold for variance filtering.
        correlation_threshold: Threshold for correlation pruning.
        laplacian_neighbors: k for kNN graph in Laplacian Score.
        num_selected_features: Final number of genes. Defaults to
            clamp(100, min(1000, remaining_genes)).
        verbose: If True, prints progress information.
    """

    if data.ndim != 2:
        raise ValueError("data must be a 2D array (samples × genes)")

    if gene_names is not None:
        gene_names = np.asarray(gene_names)
        if gene_names.shape[0] != data.shape[1]:
            raise ValueError("gene_names length must match number of genes")

    initial_indices = np.arange(data.shape[1])

    # Stage 1: Variance filtering
    var_data, var_indices, var_thresh, variances = _variance_filter(
        data, threshold=variance_threshold, verbose=verbose
    )

    # Stage 2: Correlation pruning
    corr_data, corr_indices, corr_thresh = _correlation_prune(
        var_data, var_indices, threshold=correlation_threshold, verbose=verbose
    )

    # Stage 3: Laplacian score
    lap_scores = _compute_laplacian_scores(
        corr_data, k_neighbors=laplacian_neighbors, verbose=verbose
    )

    remaining_genes = corr_data.shape[1]
    if remaining_genes == 0:
        raise ValueError("No genes remain after variance/correlation filtering.")

    if num_selected_features is None:
        num_selected_features = max(100, min(1000, remaining_genes))

    num_selected_features = min(num_selected_features, remaining_genes)
    ranked_indices = np.argsort(lap_scores)
    selected_rel_indices = ranked_indices[:num_selected_features]
    selected_indices = corr_indices[selected_rel_indices]

    if verbose:
        print(
            f"[Feature Selection] Laplacian ranking: selected {len(selected_rel_indices):,} genes"
        )

    selected_data = data[:, selected_indices]
    selected_gene_names = (
        gene_names[selected_indices] if gene_names is not None else None
    )

    stage_counts = {
        "initial": data.shape[1],
        "post_variance": int(len(var_indices)),
        "post_correlation": int(len(corr_indices)),
        "final": int(len(selected_indices)),
    }

    return FeatureSelectionResult(
        data=selected_data,
        selected_indices=selected_indices,
        selected_gene_names=selected_gene_names,
        variance_threshold=var_thresh,
        correlation_threshold=corr_thresh,
        laplacian_scores=lap_scores[selected_rel_indices],
        stage_counts=stage_counts,
    )

