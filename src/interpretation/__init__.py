"""Interpretation module for BIGCLAM project."""

from .interpreters import (
    interpret_results,
    analyze_overlap,
    identify_border_samples
)
from .biological_interpretation import (
    biological_interpretation_pipeline,
    differential_expression_analysis,
    pathway_enrichment_analysis,
    cell_type_signature_analysis,
    interpret_cluster_biology
)

__all__ = [
    'interpret_results',
    'analyze_overlap',
    'identify_border_samples',
    'biological_interpretation_pipeline',
    'differential_expression_analysis',
    'pathway_enrichment_analysis',
    'cell_type_signature_analysis',
    'interpret_cluster_biology'
]

