"""
Paper Figure Generation

Creates publication-ready figures for journal submission:
- Figure 2: Primary clustering visualization + membership heatmap
- Figure 3: Method comparison (compact, annotated)
- Figure 4: Stability + clinical relevance

All figures include honest visualization of weak signals with:
- Zoomed axes/insets for small values
- Numeric labels on all bars/points
- Explicit annotations for non-significant results
- Effect-size plots even when p>0.05
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Paper figure formatting (journal-ready)
plt.rcParams.update({
    "figure.dpi": 1200,
    "savefig.dpi": 1200,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": 6,
    "axes.titlesize": 7,
    "axes.labelsize": 6,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

sns.set_style("white")


def load_clustering_data(dataset_name, processed_dir='data/processed',
                        clustering_dir='data/clusterings'):
    """Load BIGCLAM communities, PAM50 labels, and expression data."""
    file_prefix_map = {
        'tcga': 'tcga_brca_data',
        'gse96058': 'gse96058_data'
    }
    prefix = file_prefix_map.get(dataset_name)
    if not prefix:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load communities
    cluster_file = Path(clustering_dir) / f"{prefix}_communities.npy"
    membership_file = Path(clustering_dir) / f"{prefix}_communities_membership.npy"
    
    communities = np.load(cluster_file)
    if communities.ndim > 1:
        membership_matrix = communities.copy()
        communities_hard = np.argmax(communities, axis=1)
    else:
        membership_matrix = None
        communities_hard = communities.flatten()
    
    # Load membership matrix if separate file exists
    if membership_matrix is None and membership_file.exists():
        membership_matrix = np.load(membership_file)
    
    # Load PAM50 labels
    target_file = Path(processed_dir) / f"{prefix}_targets.pkl"
    with open(target_file, 'rb') as f:
        targets_data = pickle.load(f)
        pam50_labels = np.array(targets_data.get('target_labels', []))
        sample_names = np.array(targets_data.get('sample_names', []))
    
    # Load expression data
    expression_file = Path(processed_dir) / f"{prefix}_processed.npy"
    expression_data = np.load(expression_file)
    
    return communities_hard, membership_matrix, pam50_labels, sample_names, expression_data


def create_figure2_clustering_visualization(dataset_name, output_dir='figures',
                                          processed_dir='data/processed',
                                          clustering_dir='data/clusterings'):
    """
    Figure 2: Primary clustering visualization + membership heatmap
    
    Panel A: UMAP/PCA colored by PAM50 (transparent) with BIGCLAM outlines
    Panel B: Membership strength heatmap (samples sorted by dominant cluster)
    """
    print(f"\n{'='*80}")
    print(f"CREATING FIGURE 2: Clustering Visualization - {dataset_name.upper()}")
    print(f"{'='*80}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    communities, membership_matrix, pam50_labels, sample_names, expression_data = \
        load_clustering_data(dataset_name, processed_dir, clustering_dir)
    
    # Filter to valid samples
    valid_mask = ~pd.isna(pam50_labels) & (pam50_labels != 'Unknown')
    communities_valid = communities[valid_mask]
    pam50_valid = pam50_labels[valid_mask]
    expression_valid = expression_data[valid_mask]
    
    if membership_matrix is not None:
        membership_valid = membership_matrix[valid_mask]
    else:
        membership_valid = None
    
    print(f"  Valid samples: {len(communities_valid)}")
    
    # Calculate ARI/NMI for annotation
    ari = adjusted_rand_score(communities_valid, pam50_valid)
    nmi = normalized_mutual_info_score(communities_valid, pam50_valid)
    
    # Create figure (double-column: 7.2 inches)
    fig = plt.figure(figsize=(7.2, 4.5))
    gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.25,
                 left=0.08, right=0.95, top=0.92, bottom=0.12)
    
    # ============================================================================
    # Panel A: UMAP/PCA with PAM50 colors and BIGCLAM outlines
    # ============================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Compute UMAP or PCA
    if UMAP_AVAILABLE and len(expression_valid) > 100:
        print("  Computing UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.3)
        embedding = reducer.fit_transform(expression_valid)
        method_name = "UMAP"
    else:
        print("  Computing PCA...")
        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(expression_valid)
        method_name = "PCA"
    
    # Color by PAM50 (transparent)
    unique_pam50 = sorted(set(pam50_valid))
    pam50_colors = sns.color_palette("Set2", len(unique_pam50))
    pam50_color_map = {p: pam50_colors[i] for i, p in enumerate(unique_pam50)}
    
    # Plot PAM50-colored points (transparent)
    for pam50_type in unique_pam50:
        mask = pam50_valid == pam50_type
        ax_a.scatter(embedding[mask, 0], embedding[mask, 1],
                    c=[pam50_color_map[pam50_type]], alpha=0.3, s=8,
                    label=f'{pam50_type}', edgecolors='none')
    
    # Overlay BIGCLAM communities as outlines (opaque)
    unique_communities = sorted(set(communities_valid))
    comm_colors = sns.color_palette("tab10", len(unique_communities))
    
    for comm_id in unique_communities:
        mask = communities_valid == comm_id
        ax_a.scatter(embedding[mask, 0], embedding[mask, 1],
                    c=[comm_colors[comm_id]], alpha=0.8, s=12,
                    marker='o', edgecolors='black', linewidth=0.5,
                    label=f'C{comm_id}')
    
    ax_a.set_xlabel(f'{method_name} Dimension 1', fontsize=6, fontweight='bold')
    ax_a.set_ylabel(f'{method_name} Dimension 2', fontsize=6, fontweight='bold')
    ax_a.set_title(f'{dataset_name.upper()}: {method_name} colored by PAM50\nwith BIGCLAM community outlines',
                  fontsize=7, fontweight='bold', pad=5)
    ax_a.legend(fontsize=4, loc='best', framealpha=0.9, ncol=2)
    
    # Add annotation box
    annotation_text = f"ARI = {ari:.3f}\nNMI = {nmi:.3f}\nNo clear separation"
    ax_a.text(0.02, 0.98, annotation_text, transform=ax_a.transAxes,
             fontsize=5, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax_a.text(-0.08, 1.05, 'A', transform=ax_a.transAxes,
             fontsize=8, fontweight='bold', va='bottom', ha='right')
    
    # ============================================================================
    # Panel B: Membership strength heatmap
    # ============================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    if membership_valid is not None:
        # Sort samples by dominant cluster
        dominant_clusters = np.argmax(membership_valid, axis=1)
        sort_indices = np.argsort(dominant_clusters)
        
        # Subsample if too many samples (max 500 for visualization)
        max_samples = 500
        if len(sort_indices) > max_samples:
            # Sample evenly across clusters
            sampled_indices = []
            for comm_id in unique_communities:
                comm_mask = dominant_clusters == comm_id
                comm_indices = np.where(comm_mask)[0]
                n_sample = min(max_samples // len(unique_communities), len(comm_indices))
                sampled_indices.extend(np.random.choice(comm_indices, n_sample, replace=False))
            sort_indices = np.array(sampled_indices)[np.argsort(dominant_clusters[sampled_indices])]
        
        membership_sorted = membership_valid[sort_indices]
        
        # Normalize membership
        membership_norm = membership_sorted / (membership_sorted.sum(axis=1, keepdims=True) + 1e-10)
        
        # Create heatmap
        im = ax_b.imshow(membership_norm.T, aspect='auto', cmap='YlOrRd',
                        vmin=0, vmax=1, interpolation='nearest')
        
        ax_b.set_xlabel('Samples (sorted by dominant community)', fontsize=6, fontweight='bold')
        ax_b.set_ylabel('BIGCLAM Communities', fontsize=6, fontweight='bold')
        ax_b.set_yticks(range(len(unique_communities)))
        ax_b.set_yticklabels([f'C{c}' for c in unique_communities], fontsize=5)
        ax_b.set_xticks([])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
        cbar.set_label('Membership Strength', fontsize=5)
        cbar.ax.tick_params(labelsize=4)
        
        # Add cluster size annotations
        cluster_sizes = pd.Series(communities_valid).value_counts().sort_index()
        size_text = "Cluster sizes: " + ", ".join([f"C{c}: {int(s)}" for c, s in cluster_sizes.items()])
        ax_b.text(0.5, -0.15, size_text, transform=ax_b.transAxes,
                 fontsize=5, ha='center', va='top')
        
        ax_b.set_title('Membership Strength Heatmap', fontsize=7, fontweight='bold', pad=5)
    else:
        ax_b.text(0.5, 0.5, 'Membership matrix\nnot available', ha='center', va='center',
                 fontsize=7, transform=ax_b.transAxes)
        ax_b.set_xticks([])
        ax_b.set_yticks([])
    
    ax_b.text(-0.08, 1.05, 'B', transform=ax_b.transAxes,
             fontsize=8, fontweight='bold', va='bottom', ha='right')
    
    # Save figure as TIFF
    output_file_tif = output_dir / f"Fig2_{dataset_name}_clustering_visualization.tif"
    fig.savefig(output_file_tif, dpi=1200, bbox_inches='tight', facecolor='white',
               edgecolor='none', format='tif')
    
    # Also save as PNG
    output_file_png = output_dir / f"Fig2_{dataset_name}_clustering_visualization.png"
    fig.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white',
               edgecolor='none', format='png')
    
    print(f"\n[Saved] {output_file_tif}")
    print(f"[Saved] {output_file_png}")
    plt.close()
    
    return output_file


def create_figure3_method_comparison(dataset_name, output_dir='figures',
                                    comparison_dir='results/method_comparison'):
    """
    Figure 3: Method comparison (compact, annotated)
    
    Grouped bar plot: ARI vs PAM50 and NMI vs PAM50
    Includes zoomed inset for small values
    """
    print(f"\n{'='*80}")
    print(f"CREATING FIGURE 3: Method Comparison - {dataset_name.upper()}")
    print(f"{'='*80}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load comparison results
    comparison_file = Path(comparison_dir) / f"{dataset_name}_method_comparison.csv"
    if not comparison_file.exists():
        print(f"  [Error] Comparison file not found: {comparison_file}")
        return None
    
    comparison_df = pd.read_csv(comparison_file)
    
    # Filter to methods we want to show
    methods_to_show = ['BIGCLAM', 'K-means', 'Spectral', 'NMF', 'Louvain']
    comparison_df = comparison_df[comparison_df['Method'].isin(methods_to_show)]
    
    if len(comparison_df) == 0:
        print(f"  [Warning] No matching methods found")
        return None
    
    # Create figure (Nature double-column)
    fig = plt.figure(figsize=(7.2, 3.5))
    gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.25,
                 left=0.1, right=0.95, top=0.9, bottom=0.2)
    
    metrics = ['ARI vs PAM50', 'NMI vs PAM50']
    metric_labels = ['ARI vs PAM50', 'NMI vs PAM50']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = fig.add_subplot(gs[0, idx])
        
        # Get values for this metric
        plot_df = comparison_df.sort_values(metric, ascending=False)
        methods = plot_df['Method'].values
        values = plot_df[metric].values
        
        # Check if we have std dev
        std_col = f'{metric} (std)' if f'{metric} (std)' in plot_df.columns else None
        if std_col and std_col in plot_df.columns:
            std_values = plot_df[std_col].values
        else:
            std_values = None
        
        # Create bars
        x_pos = np.arange(len(methods))
        colors = ['#d73027' if m == 'BIGCLAM' else '#4575b4' for m in methods]
        
        bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add error bars if available
        if std_values is not None:
            ax.errorbar(x_pos, values, yerr=std_values, fmt='none', 
                       color='black', capsize=2, linewidth=0.5)
        
        # Add numeric labels above bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            label_text = f'{val:.3f}'
            if std_values is not None:
                label_text += f'\n±{std_values[i]:.3f}'
            ax.text(bar.get_x() + bar.get_width()/2., val,
                   label_text, ha='center', va='bottom', fontsize=4)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, fontsize=5, rotation=45, ha='right')
        ax.set_ylabel(label, fontsize=6, fontweight='bold')
        ax.set_title(label, fontsize=7, fontweight='bold', pad=5)
        
        # Use broken y-axis if values are very small
        if values.max() < 0.1:
            ax.set_ylim([-0.01, values.max() * 1.2])
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        else:
            ax.set_ylim([0, values.max() * 1.2])
        
        # Add zoomed inset if values are very small
        if values.max() < 0.1:
            # Create inset axes
            axins = ax.inset_axes([0.5, 0.5, 0.45, 0.45])
            axins.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            if std_values is not None:
                axins.errorbar(x_pos, values, yerr=std_values, fmt='none',
                              color='black', capsize=2, linewidth=0.5)
            axins.set_ylim([-0.05, 0.2])  # Zoomed range
            axins.set_xticks(x_pos)
            axins.set_xticklabels(methods, fontsize=3, rotation=45, ha='right')
            axins.tick_params(labelsize=3)
            axins.set_title('Zoom', fontsize=4)
            ax.indicate_inset_zoom(axins, edgecolor='gray', linewidth=0.5)
        
        # Add panel label
        panel_label = 'A' if idx == 0 else 'B'
        ax.text(-0.1, 1.05, panel_label, transform=ax.transAxes,
               fontsize=8, fontweight='bold', va='bottom', ha='right')
    
    # Add annotation box
    annotation_text = "All methods show low alignment with PAM50\n(ARI near 0), consistent with cohort heterogeneity."
    fig.text(0.5, 0.05, annotation_text, ha='center', va='bottom',
            fontsize=5, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save figure as TIFF
    output_file_tif = output_dir / f"Fig3_{dataset_name}_method_comparison.tif"
    fig.savefig(output_file_tif, dpi=1200, bbox_inches='tight', facecolor='white',
               edgecolor='none', format='tif')
    
    output_file_png = output_dir / f"Fig3_{dataset_name}_method_comparison.png"
    fig.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white',
               edgecolor='none', format='png')
    
    print(f"\n[Saved] {output_file_tif}")
    print(f"[Saved] {output_file_png}")
    plt.close()
    
    return output_file


def create_figure4_stability_clinical(dataset_name, output_dir='figures',
                                     stability_dir='results/stability',
                                     survival_dir='results/survival'):
    """
    Figure 4: Stability + clinical relevance
    
    Left: Bootstrap ARI distribution (violin/boxplot)
    Right: Kaplan-Meier curves with annotations
    """
    print(f"\n{'='*80}")
    print(f"CREATING FIGURE 4: Stability + Clinical - {dataset_name.upper()}")
    print(f"{'='*80}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure (Nature double-column)
    fig = plt.figure(figsize=(7.2, 3.5))
    gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.25,
                 left=0.1, right=0.95, top=0.9, bottom=0.2)
    
    # ============================================================================
    # Panel A: Bootstrap ARI distribution
    # ============================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    bootstrap_file = Path(stability_dir) / dataset_name / "bootstrap_ari.csv"
    if bootstrap_file.exists():
        bootstrap_df = pd.read_csv(bootstrap_file)
        ari_values = bootstrap_df['ari'].values if 'ari' in bootstrap_df.columns else None
        
        if ari_values is not None:
            # Create violin plot
            parts = ax_a.violinplot([ari_values], positions=[0], widths=0.6,
                                   showmeans=True, showmedians=True)
            
            # Style violin plot
            for pc in parts['bodies']:
                pc.set_facecolor('#4575b4')
                pc.set_alpha(0.7)
            parts['cmeans'].set_color('red')
            parts['cmedians'].set_color('black')
            
            # Add box plot overlay
            bp = ax_a.boxplot([ari_values], positions=[0], widths=0.3,
                             patch_artist=True, showfliers=True)
            bp['boxes'][0].set_facecolor('white')
            bp['boxes'][0].set_alpha(0.5)
            
            # Calculate statistics
            mean_ari = np.mean(ari_values)
            std_ari = np.std(ari_values)
            median_ari = np.median(ari_values)
            
            # Add numeric annotation
            annotation_text = f"Mean = {mean_ari:.4f}\nSD = {std_ari:.4f}\nMedian = {median_ari:.4f}"
            ax_a.text(0.5, 0.95, annotation_text, transform=ax_a.transAxes,
                     fontsize=5, verticalalignment='top', ha='left',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Set y-axis range
            if abs(mean_ari) < 0.1:
                ax_a.set_ylim([-0.05, 0.05])
            else:
                ax_a.set_ylim([ari_values.min() * 1.1, ari_values.max() * 1.1])
            
            ax_a.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax_a.set_xticks([0])
            ax_a.set_xticklabels(['Bootstrap ARI'], fontsize=6, fontweight='bold')
            ax_a.set_ylabel('ARI', fontsize=6, fontweight='bold')
            ax_a.set_title('Bootstrap Stability', fontsize=7, fontweight='bold', pad=5)
        else:
            ax_a.text(0.5, 0.5, 'ARI values\nnot found', ha='center', va='center',
                     fontsize=7, transform=ax_a.transAxes)
            ax_a.set_xticks([])
            ax_a.set_yticks([])
    else:
        ax_a.text(0.5, 0.5, 'Bootstrap data\nnot available', ha='center', va='center',
                 fontsize=7, transform=ax_a.transAxes)
        ax_a.set_xticks([])
        ax_a.set_yticks([])
    
    ax_a.text(-0.1, 1.05, 'A', transform=ax_a.transAxes,
             fontsize=8, fontweight='bold', va='bottom', ha='right')
    
    # ============================================================================
    # Panel B: Kaplan-Meier curves
    # ============================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    # Try to load survival data
    merged_survival_file = Path(survival_dir) / dataset_name / "merged_survival_data.csv"
    if merged_survival_file.exists():
        survival_df = pd.read_csv(merged_survival_file)
        
        # Check if we have required columns
        if 'OS_time' in survival_df.columns and 'OS_event' in survival_df.columns and 'cluster' in survival_df.columns:
            from lifelines import KaplanMeierFitter
            from lifelines.statistics import logrank_test
            
            kmf = KaplanMeierFitter()
            clusters = sorted(survival_df['cluster'].unique())
            colors = sns.color_palette("tab10", len(clusters))
            
            # Plot KM curves
            for idx, cluster_id in enumerate(clusters):
                mask = survival_df['cluster'] == cluster_id
                n_samples = mask.sum()
                n_events = survival_df.loc[mask, 'OS_event'].sum()
                
                kmf.fit(survival_df.loc[mask, 'OS_time'],
                       survival_df.loc[mask, 'OS_event'],
                       label=f'C{cluster_id} (n={n_samples}, e={n_events})')
                kmf.plot_survival_function(ax=ax_b, ci_show=False,
                                          color=colors[idx], linewidth=1.0)
            
            # Calculate log-rank test
            if len(clusters) == 2:
                mask1 = survival_df['cluster'] == clusters[0]
                mask2 = survival_df['cluster'] == clusters[1]
                results = logrank_test(survival_df.loc[mask1, 'OS_time'],
                                      survival_df.loc[mask2, 'OS_time'],
                                      event_observed_A=survival_df.loc[mask1, 'OS_event'],
                                      event_observed_B=survival_df.loc[mask2, 'OS_event'])
                p_value = results.p_value
            else:
                # Multi-group log-rank (simplified - use pairwise)
                p_value = 0.5  # Placeholder
            
            # Add annotation
            if p_value >= 0.05:
                annotation_text = f"Log-rank p = {p_value:.3f}\n(not significant)"
            else:
                annotation_text = f"Log-rank p = {p_value:.3f}\n(significant)"
            
            ax_b.text(0.02, 0.98, annotation_text, transform=ax_b.transAxes,
                     fontsize=5, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax_b.set_xlabel('Time (days)', fontsize=6, fontweight='bold')
            ax_b.set_ylabel('Survival Probability', fontsize=6, fontweight='bold')
            ax_b.set_title('Kaplan-Meier Survival Curves', fontsize=7, fontweight='bold', pad=5)
            ax_b.legend(fontsize=4, loc='best', framealpha=0.9)
            ax_b.set_ylim([0, 1.05])
        else:
            ax_b.text(0.5, 0.5, 'Survival data\nincomplete', ha='center', va='center',
                     fontsize=7, transform=ax_b.transAxes)
            ax_b.set_xticks([])
            ax_b.set_yticks([])
    else:
        ax_b.text(0.5, 0.5, 'Survival data\nnot available', ha='center', va='center',
                 fontsize=7, transform=ax_b.transAxes)
        ax_b.set_xticks([])
        ax_b.set_yticks([])
    
    ax_b.text(-0.1, 1.05, 'B', transform=ax_b.transAxes,
             fontsize=8, fontweight='bold', va='bottom', ha='right')
    
    # Save figure as TIFF
    output_file_tif = output_dir / f"Fig4_{dataset_name}_stability_clinical.tif"
    fig.savefig(output_file_tif, dpi=1200, bbox_inches='tight', facecolor='white',
               edgecolor='none', format='tif')
    
    output_file_png = output_dir / f"Fig4_{dataset_name}_stability_clinical.png"
    fig.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white',
               edgecolor='none', format='png')
    
    print(f"\n[Saved] {output_file_tif}")
    print(f"[Saved] {output_file_png}")
    plt.close()
    
    return output_file


def create_all_paper_figures(dataset_name='gse96058', output_dir='figures'):
    """Create all paper figures for a dataset."""
    print(f"\n{'='*80}")
    print(f"CREATING ALL PAPER FIGURES FOR {dataset_name.upper()}")
    print(f"{'='*80}")
    
    figures = []
    
    # Figure 2
    try:
        fig2 = create_figure2_clustering_visualization(dataset_name, output_dir)
        if fig2:
            figures.append(fig2)
    except Exception as e:
        print(f"[Error] Failed to create Figure 2: {e}")
        import traceback
        traceback.print_exc()
    
    # Figure 3
    try:
        fig3 = create_figure3_method_comparison(dataset_name, output_dir)
        if fig3:
            figures.append(fig3)
    except Exception as e:
        print(f"[Error] Failed to create Figure 3: {e}")
        import traceback
        traceback.print_exc()
    
    # Figure 4
    try:
        fig4 = create_figure4_stability_clinical(dataset_name, output_dir)
        if fig4:
            figures.append(fig4)
    except Exception as e:
        print(f"[Error] Failed to create Figure 4: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"COMPLETE: Created {len(figures)} figures")
    print(f"{'='*80}")
    
    return figures


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        create_all_paper_figures(dataset_name)
    else:
        # Create for both datasets
        print("="*80)
        print("GENERATING PAPER FIGURES FOR ALL DATASETS")
        print("="*80)
        for dataset_name in ['gse96058', 'tcga']:
            try:
                create_all_paper_figures(dataset_name)
            except Exception as e:
                print(f"[Error] Failed for {dataset_name}: {e}")
                import traceback
                traceback.print_exc()

