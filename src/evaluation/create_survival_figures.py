"""
Standalone script to create survival analysis figures from existing results.
Can be run independently to generate visualization figures.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Import lifelines for Cox model reconstruction if needed
try:
    from lifelines import CoxPHFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("[Warning] lifelines not available. Some visualizations may be limited.")


def plot_hazard_ratio_forest_from_file(cox_summary_file, output_dir, dataset_name=None):
    """Create forest plot from saved Cox summary CSV."""
    if not cox_summary_file.exists():
        print(f"[Warning] Cox summary file not found: {cox_summary_file}")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = pd.read_csv(cox_summary_file, index_col=0)
    
    # Filter to cluster variables
    cluster_vars = [v for v in summary.index if 'cluster' in v.lower() or any(c.isdigit() for c in str(v))]
    
    if not cluster_vars:
        print("[Warning] No cluster variables found in Cox model")
        return
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(cluster_vars) * 0.8)))
    
    variables = []
    hr_values = []
    ci_lower = []
    ci_upper = []
    p_values = []
    
    # Check available columns
    available_cols = summary.columns.tolist()
    
    # Try different possible column names
    hr_col = None
    lower_col = None
    upper_col = None
    p_col = None
    
    for col in available_cols:
        if 'exp(coef)' in col.lower() or 'hazard' in col.lower() or 'hr' in col.lower():
            hr_col = col
        if 'lower' in col.lower() and '95' in col:
            lower_col = col
        if 'upper' in col.lower() and '95' in col:
            upper_col = col
        if col.lower() == 'p' or 'p-value' in col.lower() or 'pvalue' in col.lower():
            p_col = col
    
    if not hr_col or not lower_col or not upper_col or not p_col:
        print(f"[Warning] Missing required columns. Available: {available_cols}")
        print(f"  Looking for: HR, lower CI, upper CI, p-value")
        plt.close()
        return
    
    for var in cluster_vars:
        if var in summary.index:
            variables.append(str(var).replace('cluster_', 'Cluster ').replace('_', ' '))
            hr = summary.loc[var, hr_col]
            hr_lower = summary.loc[var, lower_col]
            hr_upper = summary.loc[var, upper_col]
            pval = summary.loc[var, p_col]
            
            hr_values.append(hr)
            ci_lower.append(hr_lower)
            ci_upper.append(hr_upper)
            p_values.append(pval)
    
    if not variables:
        plt.close()
        return
    
    y_pos = np.arange(len(variables))
    
    for i, (var, hr, lower, upper) in enumerate(zip(variables, hr_values, ci_lower, ci_upper)):
        color = '#d73027' if p_values[i] < 0.05 else '#4575b4'
        ax.plot([lower, upper], [i, i], color=color, linewidth=2, alpha=0.7)
        ax.scatter(hr, i, color=color, s=100, zorder=5, edgecolors='black', linewidth=1.5)
    
    ax.axvline(1, color='black', linestyle='--', linewidth=1, label='HR = 1.0')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=12)
    title = 'Hazard Ratio Forest Plot'
    if dataset_name:
        title += f' - {dataset_name.upper()}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()
    
    for i, (var, hr, pval) in enumerate(zip(variables, hr_values, p_values)):
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        ax.text(hr, i, f'  {sig}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    filename = "hazard_ratio_forest.png"
    if dataset_name:
        filename = f"{dataset_name}_hazard_ratio_forest.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"[Saved] {filename}")
    plt.close()


def plot_logrank_heatmap_from_file(logrank_file, output_dir, dataset_name=None):
    """Create heatmap from saved log-rank results CSV."""
    if not logrank_file.exists():
        print(f"[Warning] Log-rank file not found: {logrank_file}")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logrank_df = pd.read_csv(logrank_file)
    
    if len(logrank_df) == 0:
        print("[Warning] No log-rank test results available")
        return
    
    clusters = sorted(set(logrank_df['cluster_A'].unique()) | set(logrank_df['cluster_B'].unique()))
    n_clusters = len(clusters)
    
    p_matrix = np.ones((n_clusters, n_clusters))
    
    for _, row in logrank_df.iterrows():
        i = clusters.index(row['cluster_A'])
        j = clusters.index(row['cluster_B'])
        p_matrix[i, j] = row['p_value']
        p_matrix[j, i] = row['p_value']
    
    np.fill_diagonal(p_matrix, 1.0)
    
    fig, ax = plt.subplots(figsize=(max(6, n_clusters * 1.2), max(5, n_clusters * 1.0)))
    
    mask = np.triu(np.ones_like(p_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        p_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',
        vmin=0,
        vmax=0.05,
        center=0.025,
        square=True,
        mask=mask,
        cbar_kws={'label': 'p-value'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray',
        xticklabels=[f'C{c}' for c in clusters],
        yticklabels=[f'C{c}' for c in clusters]
    )
    
    title = "Log-Rank Test P-Values (Pairwise Comparisons)"
    if dataset_name:
        title += f" - {dataset_name.upper()}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    
    plt.tight_layout()
    filename = "logrank_heatmap.png"
    if dataset_name:
        filename = f"{dataset_name}_logrank_heatmap.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"[Saved] {filename}")
    plt.close()


def create_figures_from_results(dataset_name, results_dir='results/survival'):
    """
    Create all survival figures from existing results.
    
    Args:
        dataset_name: Dataset name ('tcga' or 'gse96058')
        results_dir: Base results directory
    """
    results_dir = Path(results_dir) / dataset_name
    output_dir = results_dir
    
    print(f"\n[Visualization] Creating survival figures for {dataset_name}...")
    
    # Create hazard ratio forest plot
    cox_file = results_dir / 'cox_summary.csv'
    if cox_file.exists():
        print("  Creating hazard ratio forest plot...")
        plot_hazard_ratio_forest_from_file(cox_file, output_dir, dataset_name)
    else:
        print(f"  [Skip] Cox summary not found: {cox_file}")
    
    # Create log-rank heatmap
    logrank_file = results_dir / 'logrank_results.csv'
    if logrank_file.exists():
        print("  Creating log-rank heatmap...")
        plot_logrank_heatmap_from_file(logrank_file, output_dir, dataset_name)
    else:
        print(f"  [Skip] Log-rank results not found: {logrank_file}")
    
    print(f"  [Complete] Figures saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create survival analysis figures from existing results')
    parser.add_argument('--dataset', type=str, default='both', choices=['tcga', 'gse96058', 'both'],
                       help='Dataset to visualize (default: both)')
    parser.add_argument('--results-dir', type=str, default='results/survival',
                       help='Results directory')
    
    args = parser.parse_args()
    
    datasets = ['tcga', 'gse96058'] if args.dataset == 'both' else [args.dataset]
    
    for dataset in datasets:
        create_figures_from_results(dataset, args.results_dir)

