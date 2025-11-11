"""
Feature selection package.

Provides multi-stage feature selection utilities that combine
variance filtering, correlation pruning, and Laplacian Score
ranking for high-dimensional gene expression data.
"""

from .feature_selection import run_feature_selection  # noqa: F401

__all__ = ["run_feature_selection"]

