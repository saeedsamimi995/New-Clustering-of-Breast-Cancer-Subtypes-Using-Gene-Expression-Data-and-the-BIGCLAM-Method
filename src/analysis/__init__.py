"""Analysis module for BIGCLAM project."""

from .cross_dataset_analysis import (
    analyze_cross_dataset_consistency,
    compute_community_centroids,
    compute_cross_dataset_correlation,
    find_matching_communities
)
from .parameter_grid_search import (
    run_grid_search,
    run_single_combination,
    create_grid_search_visualizations,
    clear_memory
)
from .comprehensive_method_comparison import (
    compare_all_methods
)
from .results_synthesis import (
    create_summary_table,
    generate_narrative,
    create_manuscript_results_section
)

__all__ = [
    'analyze_cross_dataset_consistency',
    'compute_community_centroids',
    'compute_cross_dataset_correlation',
    'find_matching_communities',
    'run_grid_search',
    'run_single_combination',
    'create_grid_search_visualizations',
    'clear_memory',
    'compare_all_methods',
    'create_summary_table',
    'generate_narrative',
    'create_manuscript_results_section'
]

