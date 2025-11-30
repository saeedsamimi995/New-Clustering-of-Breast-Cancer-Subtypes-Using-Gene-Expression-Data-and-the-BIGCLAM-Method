"""
Survival Evaluation Module

Performs survival analysis on clustering results:
- Kaplan–Meier survival curves
- Log-rank test
- Cox proportional hazards model

Requirements:
    pip install lifelines
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import median_survival_times
from lifelines.statistics import proportional_hazard_test


def prepare_survival_dataframe(cluster_assignments, clinical_df,
                               id_col="sample_id", cluster_col="cluster"):
    """
    Merge predicted communities with clinical survival information.
    
    Args:
        cluster_assignments: DataFrame with [sample_id, cluster]
        clinical_df: Clinical dataframe with time and event columns
        id_col: name of sample identifier column
        cluster_col: name of predicted cluster col
    
    Returns:
        merged pandas DataFrame
    """
    # Validate required columns exist
    required_cols = ["OS_time", "OS_event"]
    missing_cols = [col for col in required_cols if col not in clinical_df.columns]
    if missing_cols:
        raise ValueError(f"Clinical dataframe missing required columns: {missing_cols}")
    
    if id_col not in clinical_df.columns:
        raise ValueError(f"Clinical dataframe missing ID column: {id_col}")
    
    # Debug: Check sample ID overlap before merge
    print(f"\n[Merge Debug]")
    print(f"  Cluster assignments: {len(cluster_assignments)} samples")
    print(f"  Clinical data: {len(clinical_df)} samples")
    
    # Show sample of IDs from both
    cluster_ids = set(cluster_assignments[id_col].astype(str))
    clinical_ids = set(clinical_df[id_col].astype(str))
    
    print(f"  Cluster ID sample (first 5): {list(cluster_ids)[:5]}")
    print(f"  Clinical ID sample (first 5): {list(clinical_ids)[:5]}")
    
    # Check overlap
    overlap = cluster_ids.intersection(clinical_ids)
    print(f"  Overlapping IDs: {len(overlap)}")
    
    if len(overlap) == 0:
        # Check if cluster IDs look like F1, F2, F3... (position-based for GSE96058)
        cluster_id_sample = list(cluster_ids)[0] if cluster_ids else ""
        if cluster_id_sample.startswith('F') and len(cluster_id_sample) > 1 and cluster_id_sample[1:].isdigit():
            print(f"  [Trying] Position-based matching for GSE96058 (F1, F2, F3...)...")
            # Match by position: F1 -> first row, F2 -> second row, etc.
            # Create position-based IDs for clinical data
            clinical_df_position = clinical_df.copy()
            clinical_df_position['position_id'] = ['F' + str(i+1) for i in range(len(clinical_df_position))]
            overlap_pos = cluster_ids.intersection(set(clinical_df_position['position_id'].astype(str)))
            print(f"  Position-based overlap: {len(overlap_pos)}")
            if len(overlap_pos) > 0:
                # Use position-based matching
                df = clinical_df_position.merge(cluster_assignments, left_on='position_id', right_on=id_col, how="inner")
                # Drop the temporary position_id column
                df = df.drop(columns=['position_id'], errors='ignore')
                print(f"  After position-based merge: {len(df)} samples")
            else:
                raise ValueError(f"No matching sample IDs found (tried direct and position-based). Cluster IDs: {list(cluster_ids)[:3]}, Clinical IDs: {list(clinical_ids)[:3]}")
        else:
            print(f"\n[ERROR] No matching sample IDs between clusters and clinical data!")
            print(f"  This suggests a sample ID format mismatch.")
            print(f"  Try checking:")
            print(f"    - Sample ID format in cluster assignments")
            print(f"    - Sample ID format in clinical data")
            print(f"    - Whether IDs need transformation (e.g., removing prefixes/suffixes)")
            raise ValueError(f"No matching sample IDs found. Cluster IDs: {list(cluster_ids)[:3]}, Clinical IDs: {list(clinical_ids)[:3]}")
    else:
        df = clinical_df.merge(cluster_assignments, on=id_col, how="inner")
        print(f"  After merge: {len(df)} samples")

    # Ensure correct dtypes
    df[cluster_col] = df[cluster_col].astype(int)

    # Drop rows with missing survival values
    initial_len = len(df)
    df = df.dropna(subset=["OS_time", "OS_event"])
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"[Warning] Dropped {dropped} rows with missing survival data")
    
    if len(df) == 0:
        raise ValueError("No samples remaining after merging and filtering. Check sample IDs and survival data.")

    return df


def plot_kaplan_meier(df, cluster_col="cluster", time_col="OS_time",
                      event_col="OS_event", output_dir="results/survival",
                      dataset_name=None, show_ci=True):
    """
    Generate Kaplan–Meier curves grouped by cluster with confidence intervals.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    km = KaplanMeierFitter()
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color palette for clusters
    clusters = sorted(df[cluster_col].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    
    for idx, group in enumerate(clusters):
        mask = df[cluster_col] == group
        n_samples = mask.sum()
        n_events = df.loc[mask, event_col].sum()
        
        km.fit(
            df.loc[mask, time_col],
            df.loc[mask, event_col],
            label=f"Cluster {group} (n={n_samples}, events={n_events})"
        )
        
        if show_ci:
            km.plot_survival_function(ax=ax, ci_show=True, color=colors[idx], linewidth=2.5)
        else:
            km.plot_survival_function(ax=ax, ci_show=False, color=colors[idx], linewidth=2.5)

    title = "Kaplan–Meier Survival by BIGCLAM Communities"
    if dataset_name:
        title += f" - {dataset_name.upper()}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Time (days)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.set_ylim([0, 1.05])

    filename = "km_survival.png"
    if dataset_name:
        filename = f"{dataset_name}_km_survival.png"

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"[Saved] Kaplan–Meier plot → {output_dir/filename}")
    plt.close()


def plot_hazard_ratio_forest(cph, output_dir="results/survival", dataset_name=None):
    """
    Create a forest plot of hazard ratios from Cox model.
    
    Args:
        cph: CoxPHFitter model object
        output_dir: Output directory
        dataset_name: Optional dataset name
    """
    if cph is None:
        print("[Warning] No Cox model available for forest plot")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get summary
    summary = cph.summary.copy()
    
    # Filter to cluster variables only (exclude adjustment variables for main plot)
    cluster_vars = [v for v in summary.index if 'cluster' in v.lower() or v.startswith('cluster')]
    if not cluster_vars:
        # Try to find cluster-related variables
        cluster_vars = [v for v in summary.index if any(c.isdigit() for c in str(v))]
    
    if not cluster_vars:
        print("[Warning] No cluster variables found in Cox model for forest plot")
        return
    
    # Find column names (lifelines uses different column names)
    available_cols = summary.columns.tolist()
    hr_col = None
    lower_col = None
    upper_col = None
    p_col = None
    
    for col in available_cols:
        col_lower = str(col).lower()
        if 'exp(coef)' in col_lower or 'hazard' in col_lower:
            hr_col = col
        if 'lower' in col_lower and '95' in str(col):
            lower_col = col
        if 'upper' in col_lower and '95' in str(col):
            upper_col = col
        if col_lower == 'p' or 'p-value' in col_lower:
            p_col = col
    
    if not hr_col or not lower_col or not upper_col or not p_col:
        print(f"[Warning] Missing required columns. Available: {available_cols}")
        return
    
    # Create forest plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(cluster_vars) * 0.8)))
    
    # Extract data
    variables = []
    hr_values = []
    ci_lower = []
    ci_upper = []
    p_values = []
    
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
    
    # Create forest plot
    y_pos = np.arange(len(variables))
    
    # Plot hazard ratios with confidence intervals
    for i, (var, hr, lower, upper) in enumerate(zip(variables, hr_values, ci_lower, ci_upper)):
        # Color based on significance
        color = '#d73027' if p_values[i] < 0.05 else '#4575b4'
        
        # Plot CI
        ax.plot([lower, upper], [i, i], color=color, linewidth=2, alpha=0.7)
        # Plot HR point
        ax.scatter(hr, i, color=color, s=100, zorder=5, edgecolors='black', linewidth=1.5)
    
    # Reference line at HR=1
    ax.axvline(1, color='black', linestyle='--', linewidth=1, label='HR = 1.0')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=12)
    ax.set_title(f'Hazard Ratio Forest Plot', fontsize=14, fontweight='bold', pad=15)
    if dataset_name:
        ax.set_title(f'Hazard Ratio Forest Plot - {dataset_name.upper()}', 
                    fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()
    
    # Add p-value annotations
    for i, (var, hr, pval) in enumerate(zip(variables, hr_values, p_values)):
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        ax.text(hr, i, f'  {sig}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    filename = "hazard_ratio_forest.png"
    if dataset_name:
        filename = f"{dataset_name}_hazard_ratio_forest.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"[Saved] Hazard ratio forest plot → {output_dir/filename}")
    plt.close()


def plot_logrank_heatmap(logrank_df, output_dir="results/survival", dataset_name=None):
    """
    Create a heatmap of log-rank test p-values between clusters.
    
    Args:
        logrank_df: DataFrame with log-rank test results
        output_dir: Output directory
        dataset_name: Optional dataset name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if logrank_df is None or len(logrank_df) == 0:
        print("[Warning] No log-rank test results available for heatmap")
        return
    
    # Create matrix of p-values
    clusters = sorted(set(logrank_df['cluster_A'].unique()) | set(logrank_df['cluster_B'].unique()))
    n_clusters = len(clusters)
    
    p_matrix = np.ones((n_clusters, n_clusters))
    
    for _, row in logrank_df.iterrows():
        i = clusters.index(row['cluster_A'])
        j = clusters.index(row['cluster_B'])
        p_matrix[i, j] = row['p_value']
        p_matrix[j, i] = row['p_value']  # Symmetric
    
    # Set diagonal to 1 (self-comparison)
    np.fill_diagonal(p_matrix, 1.0)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(6, n_clusters * 1.2), max(5, n_clusters * 1.0)))
    
    # Create mask for upper triangle (to avoid redundancy)
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
    print(f"[Saved] Log-rank heatmap → {output_dir/filename}")
    plt.close()


def create_survival_summary_figure(df, logrank_df, cph, cluster_col="cluster",
                                  time_col="OS_time", event_col="OS_event",
                                  output_dir="results/survival", dataset_name=None):
    """
    Create a comprehensive summary figure with multiple survival visualizations.
    
    Args:
        df: Merged survival dataframe
        logrank_df: Log-rank test results
        cph: Cox model object
        cluster_col: Cluster column name
        time_col: Time column name
        event_col: Event column name
        output_dir: Output directory
        dataset_name: Optional dataset name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Visualization] Creating survival summary figure for {dataset_name or 'dataset'}...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    clusters = sorted(df[cluster_col].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    
    # 1. Kaplan-Meier curves (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :])
    km = KaplanMeierFitter()
    
    for idx, group in enumerate(clusters):
        mask = df[cluster_col] == group
        n_samples = mask.sum()
        n_events = df.loc[mask, event_col].sum()
        
        km.fit(
            df.loc[mask, time_col],
            df.loc[mask, event_col],
            label=f"Cluster {group} (n={n_samples}, events={n_events})"
        )
        km.plot_survival_function(ax=ax1, ci_show=True, color=colors[idx], linewidth=2.5)
    
    ax1.set_title('Kaplan-Meier Survival Curves', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Time (days)', fontsize=10)
    ax1.set_ylabel('Survival Probability', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9, framealpha=0.9)
    ax1.set_ylim([0, 1.05])
    
    # Add number-at-risk table
    try:
        at_risk_df = add_number_at_risk_table(ax1, df, time_col, event_col, cluster_col, n_timepoints=5)
    except Exception as e:
        print(f"[Warning] Could not add number-at-risk table: {e}")
    
    # 2. Hazard ratio forest plot (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    if cph is not None:
        try:
            summary = cph.summary.copy()
            cluster_vars = [v for v in summary.index if 'cluster' in v.lower() or any(c.isdigit() for c in v)]
            
            if cluster_vars:
                # Find column names (lifelines uses different column names)
                available_cols = summary.columns.tolist()
                hr_col = None
                lower_col = None
                upper_col = None
                p_col = None
                
                for col in available_cols:
                    col_lower = str(col).lower()
                    if 'exp(coef)' in col_lower or 'hazard' in col_lower:
                        hr_col = col
                    if 'lower' in col_lower and '95' in str(col):
                        lower_col = col
                    if 'upper' in col_lower and '95' in str(col):
                        upper_col = col
                    if col_lower == 'p' or 'p-value' in col_lower:
                        p_col = col
                
                if hr_col and lower_col and upper_col and p_col:
                    variables = []
                    hr_values = []
                    ci_lower = []
                    ci_upper = []
                    p_values = []
                    
                    for var in cluster_vars[:5]:  # Limit to 5 for readability
                        if var in summary.index:
                            variables.append(var.replace('cluster_', 'C').replace('_', ' '))
                            hr_values.append(summary.loc[var, hr_col])
                            ci_lower.append(summary.loc[var, lower_col])
                            ci_upper.append(summary.loc[var, upper_col])
                            p_values.append(summary.loc[var, p_col])
                
                if variables:
                    y_pos = np.arange(len(variables))
                    for i, (hr, lower, upper) in enumerate(zip(hr_values, ci_lower, ci_upper)):
                        color = '#d73027' if p_values[i] < 0.05 else '#4575b4'
                        ax2.plot([lower, upper], [i, i], color=color, linewidth=2, alpha=0.7)
                        ax2.scatter(hr, i, color=color, s=80, zorder=5, edgecolors='black', linewidth=1)
                    
                    ax2.axvline(1, color='black', linestyle='--', linewidth=1)
                    ax2.set_yticks(y_pos)
                    ax2.set_yticklabels(variables, fontsize=9)
                    ax2.set_xlabel('Hazard Ratio', fontsize=10)
                    ax2.set_title('Hazard Ratios (Cox Model)', fontsize=11, fontweight='bold')
                    ax2.grid(True, alpha=0.3, axis='x')
                else:
                    ax2.text(0.5, 0.5, 'Cox model\ncolumns not found', ha='center', va='center', fontsize=11)
                    ax2.set_xticks([])
                    ax2.set_yticks([])
            else:
                ax2.text(0.5, 0.5, 'Cox model\nno cluster vars', ha='center', va='center', fontsize=11)
                ax2.set_xticks([])
                ax2.set_yticks([])
        except:
            ax2.text(0.5, 0.5, 'Cox model\nnot available', ha='center', va='center', fontsize=11)
            ax2.set_xticks([])
            ax2.set_yticks([])
    else:
        ax2.text(0.5, 0.5, 'Cox model\nnot available', ha='center', va='center', fontsize=11)
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    # 3. Log-rank p-values (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    if logrank_df is not None and len(logrank_df) > 0:
        # Create bar plot of p-values
        logrank_df_sorted = logrank_df.sort_values('p_value')
        comparisons = [f"C{int(row['cluster_A'])} vs C{int(row['cluster_B'])}" 
                      for _, row in logrank_df_sorted.iterrows()]
        p_vals = logrank_df_sorted['p_value'].values
        
        colors_bar = ['#d73027' if p < 0.05 else '#4575b4' for p in p_vals]
        bars = ax3.barh(range(len(comparisons)), p_vals, color=colors_bar, alpha=0.7)
        ax3.set_yticks(range(len(comparisons)))
        ax3.set_yticklabels(comparisons, fontsize=9)
        ax3.set_xlabel('p-value', fontsize=10)
        ax3.set_title('Log-Rank Test Results', fontsize=11, fontweight='bold')
        ax3.axvline(0.05, color='red', linestyle='--', linewidth=1, label='p=0.05')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='x')
    else:
        ax3.text(0.5, 0.5, 'Log-rank results\nnot available', ha='center', va='center', fontsize=11)
        ax3.set_xticks([])
        ax3.set_yticks([])
    
    # 4. Cluster sizes and events (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    cluster_sizes = df[cluster_col].value_counts().sort_index()
    cluster_events = df.groupby(cluster_col)[event_col].sum().sort_index()
    
    x = np.arange(len(clusters))
    width = 0.35
    
    ax4.bar(x - width/2, [cluster_sizes.get(c, 0) for c in clusters], 
           width, label='Total', color='steelblue', alpha=0.7)
    ax4.bar(x + width/2, [cluster_events.get(c, 0) for c in clusters], 
           width, label='Events', color='coral', alpha=0.7)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'C{c}' for c in clusters])
    ax4.set_ylabel('Count', fontsize=10)
    ax4.set_title('Cluster Sizes and Events', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Summary statistics (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Calculate summary stats
    total_samples = len(df)
    total_events = df[event_col].sum()
    median_survival = df[time_col].median()
    n_clusters = df[cluster_col].nunique()
    
    summary_text = f"""
    Dataset: {dataset_name.upper() if dataset_name else 'Dataset'}
    Total Samples: {total_samples}
    Total Events: {total_events}
    Median Survival: {median_survival:.0f} days
    Number of Clusters: {n_clusters}
    """
    
    if cph is not None:
        try:
            summary_text += f"\nCox Model: Fitted"
        except:
            summary_text += f"\nCox Model: Not available"
    
    if logrank_df is not None and len(logrank_df) > 0:
        sig_tests = (logrank_df['p_value'] < 0.05).sum()
        summary_text += f"\nSignificant Log-Rank Tests: {sig_tests}/{len(logrank_df)}"
    
    ax5.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax5.set_title('Summary Statistics', fontsize=11, fontweight='bold')
    
    plt.suptitle(f'{dataset_name.upper() if dataset_name else "Survival"} Analysis Summary', 
                fontsize=16, fontweight='bold', y=0.995)
    
    filename = "survival_summary.png"
    if dataset_name:
        filename = f"{dataset_name}_survival_summary.png"
    
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"[Saved] Survival summary figure → {output_dir/filename}")
    plt.close()


def test_proportional_hazards(cph, output_dir, dataset_name=None):
    """
    Test proportional hazards assumption using Schoenfeld residuals.
    
    Args:
        cph: Fitted CoxPHFitter object
        output_dir: Output directory for results
        dataset_name: Optional dataset name
    
    Returns:
        tuple: (ph_passed, ph_df) where ph_passed is bool and ph_df is DataFrame
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        ph_test = proportional_hazard_test(cph, cph.duration_col)
        
        # Save results
        ph_df = ph_test.summary
        ph_file = output_dir / f'{dataset_name}_ph_test.csv' if dataset_name else output_dir / 'ph_test.csv'
        ph_df.to_csv(ph_file)
        print(f"[Saved] Proportional hazards test → {ph_file}")
        
        # Check violations (p < 0.05 indicates violation)
        violations = ph_df[ph_df['p'] < 0.05]
        if len(violations) > 0:
            print(f"[Warning] PH assumption violated for: {violations.index.tolist()}")
            print(f"  Consider using stratified Cox or time-varying coefficients")
            return False, ph_df
        else:
            print("[OK] Proportional hazards assumption satisfied for all variables")
            return True, ph_df
    except Exception as e:
        print(f"[Warning] PH test failed: {e}")
        return None, None


def calculate_median_survival_with_ci(df, time_col="OS_time", event_col="OS_event",
                                     cluster_col="cluster", output_dir="results/survival",
                                     dataset_name=None):
    """
    Calculate median survival with 95% CI per cluster.
    
    Args:
        df: Survival dataframe
        time_col: Time column name
        event_col: Event column name
        cluster_col: Cluster column name
        output_dir: Output directory
        dataset_name: Optional dataset name
    
    Returns:
        DataFrame with median survival and CI per cluster
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    clusters = sorted(df[cluster_col].unique())
    
    for cluster in clusters:
        cluster_df = df[df[cluster_col] == cluster]
        
        if len(cluster_df) == 0:
            continue
        
        kmf = KaplanMeierFitter()
        kmf.fit(cluster_df[time_col], cluster_df[event_col])
        
        # Get median survival
        median_ci = median_survival_times(kmf.median_survival_time_)
        
        if len(median_ci) > 0:
            median_val = median_ci.iloc[0]
            ci = kmf.median_survival_time_.confidence_interval_
            
            results.append({
                'cluster': cluster,
                'median_survival_days': median_val if not pd.isna(median_val) else np.nan,
                'ci_lower_95': ci.iloc[0, 0] if len(ci) > 0 and not pd.isna(ci.iloc[0, 0]) else np.nan,
                'ci_upper_95': ci.iloc[0, 1] if len(ci) > 0 and not pd.isna(ci.iloc[0, 1]) else np.nan,
                'n_samples': len(cluster_df),
                'n_events': cluster_df[event_col].sum()
            })
        else:
            # If median not reached, use last observed time
            results.append({
                'cluster': cluster,
                'median_survival_days': np.nan,
                'ci_lower_95': np.nan,
                'ci_upper_95': np.nan,
                'n_samples': len(cluster_df),
                'n_events': cluster_df[event_col].sum()
            })
    
    results_df = pd.DataFrame(results)
    
    # Save
    median_file = output_dir / f'{dataset_name}_median_survival.csv' if dataset_name else output_dir / 'median_survival.csv'
    results_df.to_csv(median_file, index=False)
    print(f"[Saved] Median survival with CI → {median_file}")
    
    return results_df


def add_number_at_risk_table(ax, df, time_col="OS_time", event_col="OS_event",
                            cluster_col="cluster", n_timepoints=5):
    """
    Add number-at-risk table below KM plot.
    
    Args:
        ax: Matplotlib axis
        df: Survival dataframe
        time_col: Time column name
        event_col: Event column name
        cluster_col: Cluster column name
        n_timepoints: Number of timepoints to show
    
    Returns:
        DataFrame with number at risk at each timepoint
    """
    clusters = sorted(df[cluster_col].unique())
    max_time = df[time_col].max()
    timepoints = np.linspace(0, max_time, n_timepoints)
    
    at_risk_data = []
    table_data = []
    
    for cluster in clusters:
        cluster_df = df[df[cluster_col] == cluster]
        n_at_risk = []
        
        for t in timepoints:
            # Number at risk = those who haven't had event and haven't been censored before time t
            n = ((cluster_df[time_col] >= t) | 
                 ((cluster_df[time_col] < t) & (cluster_df[event_col] == 0))).sum()
            n_at_risk.append(int(n))
        
        at_risk_data.append({
            'cluster': cluster,
            **{f'timepoint_{i}': n for i, n in enumerate(n_at_risk)}
        })
        table_data.append([str(n) for n in n_at_risk])
    
    at_risk_df = pd.DataFrame(at_risk_data)
    
    # Add text table below plot (simpler approach)
    table_text = "Number at risk:\n"
    table_text += "Time: " + "  ".join([f"{int(t):>6}" for t in timepoints]) + "\n"
    for i, cluster in enumerate(clusters):
        table_text += f"C{cluster}: " + "  ".join([f"{n:>6}" for n in table_data[i]]) + "\n"
    
    # Add as text annotation
    ax.text(0.02, 0.02, table_text, transform=ax.transAxes,
           fontsize=8, verticalalignment='bottom', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return at_risk_df


def run_logrank(df, cluster_col="cluster", time_col="OS_time",
                event_col="OS_event", apply_fdr_correction=True):
    """
    Run pairwise log-rank tests between all clusters.
    
    Args:
        df: Survival dataframe
        cluster_col: Cluster column name
        time_col: Time column name
        event_col: Event column name
        apply_fdr_correction: If True, apply Benjamini-Hochberg FDR correction
    
    Returns:
        DataFrame with log-rank test results
    """
    clusters = sorted(df[cluster_col].unique())
    results = []

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            c1 = clusters[i]
            c2 = clusters[j]

            g1 = df[df[cluster_col] == c1]
            g2 = df[df[cluster_col] == c2]

            test = logrank_test(
                g1[time_col],
                g2[time_col],
                event_observed_A=g1[event_col],
                event_observed_B=g2[event_col]
            )

            results.append({
                "cluster_A": c1,
                "cluster_B": c2,
                "p_value": test.p_value,
                "p_value_adj": None,  # Will fill after
                "statistic": test.test_statistic
            })
    
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction if requested
    if apply_fdr_correction and len(results_df) > 0:
        try:
            from statsmodels.stats.multitest import multipletests  # type: ignore
            _, p_adj, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
            results_df['p_value_adj'] = p_adj
            print(f"[Info] Applied Benjamini-Hochberg FDR correction to log-rank tests")
        except ImportError:
            # Fallback to Bonferroni
            n_tests = len(results_df)
            results_df['p_value_adj'] = np.clip(results_df['p_value'] * n_tests, 0, 1)
            print(f"[Info] Applied Bonferroni correction to log-rank tests (statsmodels not available)")

    return results_df


def run_cox(df,
            duration_col="OS_time",
            event_col="OS_event",
            cluster_col="cluster",
            adjust_cols=None,
            output_dir="results/survival",
            dataset_name=None):
    """
    Multivariate Cox proportional hazards model.
    
    adjust_cols: list of clinical covariates
    
    Returns:
        CoxPHFitter object or None if model fails
    """
    # Start with required columns
    cox_df = df[[duration_col, event_col, cluster_col]].copy()
    
    # Track which adjustment variables are actually used
    used_adjust_cols = []

    if adjust_cols:
        for col in adjust_cols:
            if col in df.columns:
                # Check if column has sufficient non-null values
                non_null_count = df[col].notna().sum()
                total_count = len(df)
                
                if non_null_count < total_count * 0.5:  # Less than 50% non-null
                    print(f"[Warning] Adjustment column '{col}' has too many missing values ({non_null_count}/{total_count}), skipping")
                    continue
                
                cox_df[col] = df[col]
                used_adjust_cols.append(col)
            else:
                print(f"[Warning] Adjustment column '{col}' not found in data, skipping")

    # Check for missing values
    initial_len = len(cox_df)
    if cox_df.isna().any().any():
        print(f"[Info] Missing values in Cox model data:")
        for col in cox_df.columns:
            missing = cox_df[col].isna().sum()
            if missing > 0:
                print(f"  {col}: {missing} missing ({missing/initial_len*100:.1f}%)")
        
        print(f"[Info] Dropping rows with missing values...")
        cox_df = cox_df.dropna()
        print(f"  Samples before: {initial_len}")
        print(f"  Samples after: {len(cox_df)}")
        print(f"  Dropped: {initial_len - len(cox_df)} samples")

    if len(cox_df) == 0:
        raise ValueError("No valid data for Cox model after removing missing values")
    
    # Check minimum sample size (need at least 10 samples per variable)
    n_vars = 1 + len(used_adjust_cols)  # cluster + adjustment variables
    min_samples = n_vars * 10
    if len(cox_df) < min_samples:
        print(f"[Warning] Low sample size ({len(cox_df)}) for {n_vars} variables (recommended: {min_samples}+)")
        print(f"  Proceeding with caution...")

    # Check for events (need at least some events)
    n_events = cox_df[event_col].sum()
    if n_events < 5:
        raise ValueError(f"Insufficient events ({n_events}) for Cox model (need at least 5)")

    try:
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col=duration_col, event_col=event_col)
        
        if used_adjust_cols:
            print(f"[Info] Cox model fitted with adjustment variables: {used_adjust_cols}")
        else:
            print(f"[Info] Cox model fitted without adjustment variables (unadjusted)")
        
        return cph
    except Exception as e:
        # If adjusted model fails, try unadjusted model
        if used_adjust_cols:
            print(f"[Warning] Adjusted Cox model failed: {e}")
            print(f"[Info] Attempting unadjusted model (cluster only)...")
            try:
                cox_df_unadj = df[[duration_col, event_col, cluster_col]].copy()
                cox_df_unadj = cox_df_unadj.dropna()
                
                if len(cox_df_unadj) == 0:
                    raise ValueError("No valid data for unadjusted Cox model")
                
                n_events_unadj = cox_df_unadj[event_col].sum()
                if n_events_unadj < 5:
                    raise ValueError(f"Insufficient events ({n_events_unadj}) for unadjusted Cox model")
                
                cph = CoxPHFitter()
                cph.fit(cox_df_unadj, duration_col=duration_col, event_col=event_col)
                print(f"[Info] Unadjusted Cox model fitted successfully")
                return cph
            except Exception as e2:
                raise ValueError(f"Both adjusted and unadjusted Cox models failed. Last error: {e2}")
        else:
            raise


def survival_evaluation(cluster_assignments,
                        clinical_df,
                        output_dir="results/survival",
                        id_col="sample_id",
                        cluster_col="cluster",
                        adjust_cols=["age", "stage"],
                        dataset_name=None):
    """
    Full pipeline: Merge data → KM → Logrank → Cox
    
    Args:
        cluster_assignments (pd.DataFrame): cols = [sample_id, cluster]
        clinical_df: Clinical dataframe containing OS_time, OS_event
        output_dir: Output directory for results
        id_col: Name of sample identifier column
        cluster_col: Name of cluster column
        adjust_cols: List of adjustment variables for Cox model (default: ["age", "stage"])
                     If None, will attempt to use available variables
        dataset_name: Optional dataset name for output file naming
    
    Returns:
        tuple: (merged_df, logrank_df, cox_model)
    """
    df = prepare_survival_dataframe(cluster_assignments, clinical_df,
                                    id_col=id_col, cluster_col=cluster_col)

    print("\n=== SURVIVAL ANALYSIS ===")
    if dataset_name:
        print(f"Dataset: {dataset_name}")
    print(f"Patients available: {len(df)}")
    print(f"Number of clusters: {df[cluster_col].nunique()}")
    print(f"Cluster sizes: {df[cluster_col].value_counts().sort_index().to_dict()}")

    # Create dataset-specific output directory
    if dataset_name:
        dataset_output_dir = Path(output_dir) / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        dataset_output_dir = Path(output_dir)

    # KM Plot
    plot_kaplan_meier(df, cluster_col=cluster_col, 
                     output_dir=dataset_output_dir, dataset_name=dataset_name)

    # Calculate median survival with CI
    print("\n[Calculating] Median survival with 95% CI per cluster...")
    median_survival_df = calculate_median_survival_with_ci(
        df, cluster_col=cluster_col, 
        output_dir=dataset_output_dir, 
        dataset_name=dataset_name
    )
    print(median_survival_df)
    
    # Logrank test
    log_df = run_logrank(df, cluster_col=cluster_col, apply_fdr_correction=True)
    print("\nLog-rank pairwise results (with FDR correction):")
    print(log_df)
    
    # Save logrank results
    logrank_path = dataset_output_dir / "logrank_results.csv"
    log_df.to_csv(logrank_path, index=False)
    print(f"[Saved] Log-rank results → {logrank_path}")
    
    # Create log-rank heatmap
    try:
        plot_logrank_heatmap(log_df, output_dir=dataset_output_dir, dataset_name=dataset_name)
    except Exception as e:
        print(f"[Warning] Failed to create log-rank heatmap: {e}")

    # Determine adjustment columns - check what's actually available
    final_adjust_cols = None
    if adjust_cols is not None:
        # Filter to only include columns that exist and have sufficient data
        available_cols = []
        for col in adjust_cols:
            if col in df.columns:
                non_null_pct = df[col].notna().sum() / len(df) * 100
                if non_null_pct >= 50:  # At least 50% non-null
                    available_cols.append(col)
                else:
                    print(f"[Info] Skipping '{col}' for Cox model: only {non_null_pct:.1f}% non-null")
            else:
                print(f"[Info] Skipping '{col}' for Cox model: column not found")
        
        if available_cols:
            final_adjust_cols = available_cols
            print(f"[Info] Using adjustment variables: {final_adjust_cols}")
        else:
            print(f"[Info] No suitable adjustment variables available, using unadjusted model")
            final_adjust_cols = None
    else:
        final_adjust_cols = None

    # Cox model
    try:
        cph = run_cox(df, cluster_col=cluster_col, adjust_cols=final_adjust_cols,
                     output_dir=dataset_output_dir, dataset_name=dataset_name)
        if cph is not None:
            print("\nCox Model Summary:")
            print(cph.summary)

            # Save hazard table
            summary_path = dataset_output_dir / "cox_summary.csv"
            cph.summary.to_csv(summary_path)
            print(f"[Saved] Cox model → {summary_path}")
            
            # Test proportional hazards assumption
            print(f"\n[Testing] Proportional hazards assumption...")
            try:
                ph_passed, ph_df = test_proportional_hazards(cph, output_dir=dataset_output_dir, 
                                                             dataset_name=dataset_name)
                if ph_passed is False:
                    print(f"[Warning] PH assumption violated. Consider stratified Cox or time-varying coefficients.")
            except Exception as e:
                print(f"[Warning] Could not test PH assumption: {e}")
            
            # Create hazard ratio forest plot
            try:
                plot_hazard_ratio_forest(cph, output_dir=dataset_output_dir, dataset_name=dataset_name)
            except Exception as e:
                print(f"[Warning] Failed to create hazard ratio forest plot: {e}")
        else:
            print("[Warning] Cox model returned None")
            cph = None
    except Exception as e:
        print(f"[Error] Cox model failed: {e}")
        import traceback
        traceback.print_exc()
        cph = None
    
    # Create comprehensive summary figure
    try:
        create_survival_summary_figure(
            df, log_df, cph, 
            cluster_col=cluster_col,
            time_col="OS_time",
            event_col="OS_event",
            output_dir=dataset_output_dir,
            dataset_name=dataset_name
        )
    except Exception as e:
        print(f"[Warning] Failed to create survival summary figure: {e}")
        import traceback
        traceback.print_exc()

    return df, log_df, cph


def load_gse96058_clinical(clinical_file):
    """
    Load GSE96058 clinical data (transposed format) and convert to standard format.
    
    Args:
        clinical_file: Path to GSE96058 clinical data CSV (transposed format)
    
    Returns:
        DataFrame with samples as rows, features as columns
    """
    print(f"\n[Loading] GSE96058 clinical data from {clinical_file}...")
    
    # Load transposed data (features as rows, samples as columns)
    clinical_T = pd.read_csv(clinical_file, index_col=0)
    
    print(f"  Transposed shape: {clinical_T.shape}")
    print(f"  Rows (features): {list(clinical_T.index)}")
    print(f"  Columns (samples): {len(clinical_T.columns)} samples")
    
    # Transpose back to standard format (samples as rows, features as columns)
    clinical = clinical_T.T.reset_index()
    clinical = clinical.rename(columns={'index': 'sample_id'})
    
    print(f"  Converted shape: {clinical.shape}")
    print(f"  Columns: {list(clinical.columns)}")
    
    # Convert OS_time and OS_event to numeric
    if 'OS_time' in clinical.columns:
        clinical['OS_time'] = pd.to_numeric(clinical['OS_time'], errors='coerce')
    if 'OS_event' in clinical.columns:
        clinical['OS_event'] = pd.to_numeric(clinical['OS_event'], errors='coerce')
    if 'age' in clinical.columns:
        clinical['age'] = pd.to_numeric(clinical['age'], errors='coerce')
    
    return clinical


def load_tcga_clinical(clinical_file):
    """
    Load TCGA clinical data.
    
    Args:
        clinical_file: Path to TCGA clinical data TSV file
    
    Returns:
        DataFrame with samples as rows, features as columns
    """
    print(f"\n[Loading] TCGA clinical data from {clinical_file}...")
    
    # TCGA clinical data is in TSV format
    clinical = pd.read_csv(clinical_file, sep='\t', low_memory=False)
    
    print(f"  Shape: {clinical.shape}")
    print(f"  Columns: {list(clinical.columns)[:10]}...")
    
    # TCGA uses different column names - need to map them
    if 'sample_id' not in clinical.columns:
        # Try to find sample ID column
        for col in ['Sample ID', 'bcr_patient_barcode', 'Patient ID', 'sample']:
            if col in clinical.columns:
                clinical = clinical.rename(columns={col: 'sample_id'})
                break
    
    # TCGA sample IDs in clinical data are like "TCGA-A1-A0SB-01" (with dashes and -01 suffix)
    # But cluster assignments use "TCGA.A1.A0SB" (with dots, no suffix)
    # Convert clinical IDs to match cluster format
    if 'sample_id' in clinical.columns:
        # Remove -01, -02, etc. suffix and convert dashes to dots
        clinical['sample_id'] = clinical['sample_id'].astype(str).str.replace(r'-\d+$', '', regex=True)  # Remove -01 suffix
        clinical['sample_id'] = clinical['sample_id'].str.replace('-', '.')  # Convert dashes to dots
    
    # Map survival columns - TCGA uses "Overall Survival (Months)" and "Overall Survival Status"
    if 'OS_time' not in clinical.columns:
        for col in ['Overall Survival (Months)', 'OS.time', 'OS_time', 'OS Time', 
                    'overall_survival_days', 'days_to_death', 'Days to Death']:
            if col in clinical.columns:
                clinical = clinical.rename(columns={col: 'OS_time'})
                # Convert months to days (multiply by 30.44 average days per month)
                if col == 'Overall Survival (Months)':
                    clinical['OS_time'] = pd.to_numeric(clinical['OS_time'], errors='coerce') * 30.44
                break
    
    if 'OS_event' not in clinical.columns:
        for col in ['Overall Survival Status', 'OS', 'OS.event', 'OS_event', 'OS Event', 
                   'vital_status', 'Vital Status', 'overall_survival_event']:
            if col in clinical.columns:
                clinical = clinical.rename(columns={col: 'OS_event'})
                break
    
    # Convert OS_event to binary (1=event/dead, 0=censored/alive)
    if 'OS_event' in clinical.columns:
        if clinical['OS_event'].dtype == 'object':
            # Handle string values like "Dead", "Alive", "DECEASED", etc.
            clinical['OS_event'] = clinical['OS_event'].astype(str).str.upper()
            clinical['OS_event'] = clinical['OS_event'].apply(
                lambda x: 1 if any(word in str(x) for word in ['DEAD', 'DECEASED', '1', 'TRUE']) 
                else (0 if any(word in str(x) for word in ['ALIVE', 'LIVING', '0', 'FALSE']) else np.nan)
            )
        else:
            # Assume 1=event, 0=censored
            clinical['OS_event'] = pd.to_numeric(clinical['OS_event'], errors='coerce')
    
    # Convert OS_time to numeric (if not already converted from months)
    if 'OS_time' in clinical.columns:
        clinical['OS_time'] = pd.to_numeric(clinical['OS_time'], errors='coerce')
    
    # Map age column
    if 'age' not in clinical.columns:
        for col in ['Diagnosis Age', 'age', 'Age', 'AGE', 'age_at_initial_pathologic_diagnosis']:
            if col in clinical.columns:
                clinical = clinical.rename(columns={col: 'age'})
                break
    
    if 'age' in clinical.columns:
        clinical['age'] = pd.to_numeric(clinical['age'], errors='coerce')
    
    # Map stage column
    if 'stage' not in clinical.columns:
        for col in ['Converted Stage', 'Tumor Stage', 'stage', 'Stage', 'STAGE', 
                   'ajcc_pathologic_tumor_stage', 'pathologic_stage']:
            if col in clinical.columns:
                clinical = clinical.rename(columns={col: 'stage'})
                break
    
    return clinical


def create_dummy_clusters(clinical_df, n_clusters=5, random_seed=42):
    """
    Create dummy cluster assignments for testing purposes.
    
    Args:
        clinical_df: Clinical DataFrame with sample_id column
        n_clusters: Number of clusters to create
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns [sample_id, cluster]
    """
    print(f"\n[Creating] Dummy cluster assignments ({n_clusters} clusters)...")
    
    # Create random cluster assignments
    np.random.seed(random_seed)
    cluster_assignments = pd.DataFrame({
        'sample_id': clinical_df['sample_id'].values,
        'cluster': np.random.randint(0, n_clusters, size=len(clinical_df))
    })
    
    print(f"  Cluster distribution:")
    print(cluster_assignments['cluster'].value_counts().sort_index())
    
    return cluster_assignments


def load_bigclam_clusters(dataset_name, clustering_dir='data/clusterings', processed_dir='data/processed'):
    """
    Load real BIGCLAM cluster assignments from clustering results.
    
    Args:
        dataset_name: Dataset name ('tcga' or 'gse96058')
        clustering_dir: Directory containing cluster files
        processed_dir: Directory containing processed data files
    
    Returns:
        DataFrame with columns [sample_id, cluster] or None if files not found
    """
    import pickle
    
    # Map dataset names to file names
    dataset_file_map = {
        'tcga': 'tcga_brca_data',
        'gse96058': 'gse96058_data'
    }
    
    if dataset_name not in dataset_file_map:
        print(f"[Error] Unknown dataset: {dataset_name}")
        return None
    
    file_base = dataset_file_map[dataset_name]
    cluster_file = Path(clustering_dir) / f"{file_base}_communities.npy"
    target_file = Path(processed_dir) / f"{file_base}_targets.pkl"
    
    # Check if files exist
    if not cluster_file.exists():
        print(f"[Warning] Cluster file not found: {cluster_file}")
        return None
    
    if not target_file.exists():
        print(f"[Warning] Target file not found: {target_file}")
        return None
    
    print(f"\n[Loading] BIGCLAM clustering results...")
    print(f"  Cluster file: {cluster_file}")
    print(f"  Target file: {target_file}")
    
    try:
        # Load clusters
        communities = np.load(cluster_file)
        if communities.ndim == 2:
            print(f"  Converting 2D membership matrix to 1D assignments...")
            communities = np.argmax(communities, axis=1)
        communities = communities.flatten()
        
        print(f"  Loaded {len(communities)} cluster assignments")
        print(f"  Number of clusters: {len(set(communities))}")
        print(f"  Cluster distribution: {dict(pd.Series(communities).value_counts().sort_index())}")
        
        # Load targets to get sample names
        with open(target_file, 'rb') as f:
            targets_data = pickle.load(f)
        
        # Get sample names from targets
        sample_names = targets_data.get('sample_names', None)
        if sample_names is None:
            print(f"  [Warning] No sample_names in targets, using indices")
            sample_names = [f"sample_{i}" for i in range(len(communities))]
        else:
            # Ensure sample_names is a list/array
            if isinstance(sample_names, np.ndarray):
                sample_names = sample_names.tolist()
            if len(sample_names) != len(communities):
                print(f"  [Warning] Mismatch: {len(sample_names)} sample names vs {len(communities)} clusters")
                print(f"  Using indices instead")
                sample_names = [f"sample_{i}" for i in range(len(communities))]
        
        # Create cluster assignments DataFrame
        cluster_assignments = pd.DataFrame({
            'sample_id': sample_names,
            'cluster': communities
        })
        
        print(f"  ✓ Successfully loaded BIGCLAM clusters")
        return cluster_assignments
        
    except Exception as e:
        print(f"  [Error] Failed to load clusters: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    """
    Main function to run survival evaluation on TCGA and/or GSE96058 datasets.
    Uses real BIGCLAM clustering results if available, otherwise falls back to dummy clusters.
    Can be run directly: python src/evaluation/survival_evaluator.py
    """
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(
        description="Run survival evaluation on TCGA and/or GSE96058 datasets using BIGCLAM clusters"
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['tcga', 'gse96058', 'both'],
        default=['both'],
        help='Datasets to analyze (default: both)'
    )
    parser.add_argument(
        '--tcga-clinical',
        type=str,
        default='data/brca_tcga_pub_clinical_data.tsv',
        help='Path to TCGA clinical data file'
    )
    parser.add_argument(
        '--gse-clinical',
        type=str,
        default='data/GSE96058_clinical_data.csv',
        help='Path to GSE96058 clinical data file'
    )
    parser.add_argument(
        '--clustering-dir',
        type=str,
        default='data/clusterings',
        help='Directory containing BIGCLAM cluster files (*_communities.npy)'
    )
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed',
        help='Directory containing processed data files (*_targets.pkl)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/survival',
        help='Output directory for results'
    )
    parser.add_argument(
        '--use-dummy-clusters',
        action='store_true',
        help='Use dummy/random clusters instead of real BIGCLAM clusters (for testing)'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=5,
        help='Number of dummy clusters if using dummy clusters (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Expand 'both' to both datasets
    datasets_to_test = []
    if 'both' in args.datasets:
        datasets_to_test = ['tcga', 'gse96058']
    else:
        datasets_to_test = args.datasets
    
    print("="*80)
    print("SURVIVAL EVALUATION WITH BIGCLAM CLUSTERS")
    print("="*80)
    
    results = {}
    
    # Test each dataset
    for dataset_name in datasets_to_test:
        print("\n" + "="*80)
        print(f"ANALYZING: {dataset_name.upper()}")
        print("="*80)
        
        try:
            # Load clinical data
            if dataset_name == 'gse96058':
                clinical_file = Path(args.gse_clinical)
                if not clinical_file.exists():
                    print(f"✗ GSE96058 clinical file not found: {clinical_file}")
                    results[dataset_name] = None
                    continue
                clinical_df = load_gse96058_clinical(str(clinical_file))
            elif dataset_name == 'tcga':
                clinical_file = Path(args.tcga_clinical)
                if not clinical_file.exists():
                    print(f"✗ TCGA clinical file not found: {clinical_file}")
                    results[dataset_name] = None
                    continue
                clinical_df = load_tcga_clinical(str(clinical_file))
            else:
                print(f"✗ Unknown dataset: {dataset_name}")
                results[dataset_name] = None
                continue
            
            # Check required columns
            required_cols = ['sample_id', 'OS_time', 'OS_event']
            missing_cols = [col for col in required_cols if col not in clinical_df.columns]
            if missing_cols:
                print(f"\n❌ ERROR: Missing required columns: {missing_cols}")
                print(f"   Available columns: {list(clinical_df.columns)}")
                results[dataset_name] = False
                continue
            
            # Check data availability
            n_samples = len(clinical_df)
            n_with_time = clinical_df['OS_time'].notna().sum()
            n_with_event = clinical_df['OS_event'].notna().sum()
            n_complete = clinical_df[['OS_time', 'OS_event']].notna().all(axis=1).sum()
            
            print(f"\n[Data Summary]")
            print(f"  Total samples: {n_samples}")
            print(f"  Samples with OS_time: {n_with_time}")
            print(f"  Samples with OS_event: {n_with_event}")
            print(f"  Samples with both: {n_complete}")
            
            if n_complete == 0:
                print(f"\n❌ ERROR: No samples with complete survival data")
                results[dataset_name] = False
                continue
            
            # Load cluster assignments
            if args.use_dummy_clusters:
                print(f"\n[Using] Dummy clusters (--use-dummy-clusters flag set)")
                cluster_assignments = create_dummy_clusters(clinical_df, n_clusters=args.n_clusters)
            else:
                print(f"\n[Loading] Real BIGCLAM cluster assignments...")
                cluster_assignments = load_bigclam_clusters(
                    dataset_name, 
                    clustering_dir=args.clustering_dir,
                    processed_dir=args.processed_dir
                )
                
                if cluster_assignments is None:
                    print(f"[Warning] Could not load BIGCLAM clusters, falling back to dummy clusters")
                    cluster_assignments = create_dummy_clusters(clinical_df, n_clusters=args.n_clusters)
            
            # Determine adjustment columns
            adjust_cols = []
            if 'age' in clinical_df.columns:
                adjust_cols.append('age')
            if 'stage' in clinical_df.columns:
                adjust_cols.append('stage')
            if not adjust_cols:
                adjust_cols = None
            
            # Run survival evaluation
            print(f"\n[Running] Survival evaluation...")
            df, log_df, cph = survival_evaluation(
                cluster_assignments=cluster_assignments,
                clinical_df=clinical_df,
                output_dir=args.output_dir,
                id_col='sample_id',
                cluster_col='cluster',
                adjust_cols=adjust_cols,
                dataset_name=dataset_name
            )
            
            print(f"\n✅ SUCCESS: Survival evaluation completed for {dataset_name}")
            results[dataset_name] = True
            
        except Exception as e:
            print(f"\n❌ ERROR: Survival evaluation failed for {dataset_name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    for dataset, result in results.items():
        if result is None:
            print(f"  {dataset}: SKIPPED (file not found)")
        elif result:
            print(f"  {dataset}: ✅ COMPLETED")
        else:
            print(f"  {dataset}: ❌ FAILED")
    
    # Overall result
    if all(r is True for r in results.values() if r is not None):
        print("\n✅ ALL ANALYSES COMPLETED")
        exit(0)
    else:
        print("\n❌ SOME ANALYSES FAILED")
        exit(1)
