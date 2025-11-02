"""Analysis module for BIGCLAM project."""

from .cross_dataset_analysis import (
    analyze_cross_dataset_consistency,
    compute_community_centroids,
    compute_cross_dataset_correlation,
    find_matching_communities
)
from .parameter_sensitivity import (
    run_sensitivity_analysis,
    analyze_variance_threshold_sensitivity,
    analyze_similarity_threshold_sensitivity
)

__all__ = [
    'analyze_cross_dataset_consistency',
    'compute_community_centroids',
    'compute_cross_dataset_correlation',
    'find_matching_communities',
    'run_sensitivity_analysis',
    'analyze_variance_threshold_sensitivity',
    'analyze_similarity_threshold_sensitivity'
]

