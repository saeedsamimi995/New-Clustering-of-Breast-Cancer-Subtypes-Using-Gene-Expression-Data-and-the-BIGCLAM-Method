"""
Standalone script to create biological interpretation figures.
Can be run independently to generate visualization figures.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')


def create_biological_interpretation_figures(dataset_name, 
                                             output_dir='results/biological_interpretation',
                                             top_n_genes=10):
    """
    Create comprehensive visualization figures for biological interpretation results.
    
    Args:
        dataset_name: Name of dataset ('tcga' or 'gse96058')
        output_dir: Output directory
        top_n_genes: Number of top genes to show in bar plots
    """
    output_dir = Path(output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Visualization] Creating figures for {dataset_name}...")
    
    # Load results
    signature_scores_file = output_dir / 'signature_scores.csv'
    if not signature_scores_file.exists():
        print(f"  [Error] Signature scores file not found: {signature_scores_file}")
        return
    
    signature_scores = pd.read_csv(signature_scores_file)
    
    # Get list of clusters
    clusters = sorted(signature_scores['cluster'].unique())
    n_clusters = len(clusters)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # ============================================================================
    # FIGURE 1: Signature Scores Heatmap
    # ============================================================================
    print("  Creating signature scores heatmap...")
    
    # Pivot data for heatmap
    signature_pivot = signature_scores.pivot_table(
        index='signature', 
        columns='cluster', 
        values='difference',
        aggfunc='mean'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, n_clusters * 1.2), 6))
    
    # Create heatmap
    sns.heatmap(
        signature_pivot,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-3,
        vmax=3,
        cbar_kws={'label': 'Signature Score Difference'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_title(f'{dataset_name.upper()}: Cell-Type Signature Scores by Cluster', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Signature', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    heatmap_file = output_dir / f'{dataset_name}_signature_heatmap.png'
    plt.savefig(heatmap_file, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [Saved] {heatmap_file.name}")
    
    # ============================================================================
    # FIGURE 2: Top DE Genes per Cluster (Bar Plot)
    # ============================================================================
    print("  Creating top DE genes bar plots...")
    
    # Determine grid layout
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_clusters == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, cluster_id in enumerate(clusters):
        de_file = output_dir / f'cluster_{cluster_id}_differential_expression.csv'
        if not de_file.exists():
            if idx < len(axes):
                axes[idx].text(0.5, 0.5, f'Cluster {cluster_id}\nNo DE data', 
                              ha='center', va='center', fontsize=12)
                axes[idx].set_xticks([])
                axes[idx].set_yticks([])
            continue
        
        de_data = pd.read_csv(de_file)
        
        # Get top upregulated and downregulated
        top_up = de_data[de_data['log2fc'] > 0].nlargest(top_n_genes, 'log2fc')
        top_down = de_data[de_data['log2fc'] < 0].nsmallest(top_n_genes, 'log2fc')
        
        # Combine and sort
        top_genes = pd.concat([top_up, top_down]).sort_values('log2fc')
        
        if len(top_genes) == 0 or idx >= len(axes):
            if idx < len(axes):
                axes[idx].text(0.5, 0.5, f'Cluster {cluster_id}\nNo significant genes', 
                              ha='center', va='center', fontsize=12)
                axes[idx].set_xticks([])
                axes[idx].set_yticks([])
            continue
        
        # Create horizontal bar plot
        colors = ['#d73027' if x > 0 else '#4575b4' for x in top_genes['log2fc']]
        axes[idx].barh(range(len(top_genes)), top_genes['log2fc'], color=colors, alpha=0.7)
        axes[idx].set_yticks(range(len(top_genes)))
        axes[idx].set_yticklabels(top_genes['gene'], fontsize=8)
        axes[idx].set_xlabel('log2 Fold Change', fontsize=10)
        axes[idx].set_title(f'Cluster {cluster_id} (n={len(de_data)} genes)', 
                           fontsize=11, fontweight='bold')
        axes[idx].axvline(0, color='black', linestyle='--', linewidth=0.5)
        axes[idx].grid(axis='x', alpha=0.3)
    
    # Hide extra subplots
    for idx in range(len(clusters), len(axes)):
        if idx < len(axes):
            axes[idx].axis('off')
    
    plt.suptitle(f'{dataset_name.upper()}: Top Differential Genes by Cluster', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    de_plot_file = output_dir / f'{dataset_name}_top_de_genes.png'
    plt.savefig(de_plot_file, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [Saved] {de_plot_file.name}")
    
    # ============================================================================
    # FIGURE 3: Signature Scores Bar Plot (Grouped)
    # ============================================================================
    print("  Creating signature scores bar plot...")
    
    signatures = sorted(signature_scores['signature'].unique())
    n_signatures = len(signatures)
    
    if n_signatures > 0:
        fig, axes = plt.subplots(1, n_signatures, figsize=(4*n_signatures, 6))
        if n_signatures == 1:
            axes = [axes]
        
        for idx, sig in enumerate(signatures):
            sig_data = signature_scores[signature_scores['signature'] == sig].sort_values('cluster')
            
            colors = ['#2ecc71' if x else '#e74c3c' for x in sig_data['enriched']]
            bars = axes[idx].bar(range(len(sig_data)), sig_data['difference'], color=colors, alpha=0.7)
            
            axes[idx].set_xticks(range(len(sig_data)))
            axes[idx].set_xticklabels([f'C{c}' for c in sig_data['cluster']], rotation=45, ha='right')
            axes[idx].set_ylabel('Score Difference', fontsize=10)
            axes[idx].set_title(sig.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            axes[idx].axhline(0, color='black', linestyle='-', linewidth=0.5)
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add significance markers
            for i, (bar, pval) in enumerate(zip(bars, sig_data['pvalue'])):
                if pval < 0.001:
                    axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                  '***', ha='center', va='bottom' if bar.get_height() > 0 else 'top',
                                  fontsize=8, fontweight='bold')
                elif pval < 0.01:
                    axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                  '**', ha='center', va='bottom' if bar.get_height() > 0 else 'top',
                                  fontsize=8, fontweight='bold')
                elif pval < 0.05:
                    axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                  '*', ha='center', va='bottom' if bar.get_height() > 0 else 'top',
                                  fontsize=8, fontweight='bold')
        
        plt.suptitle(f'{dataset_name.upper()}: Signature Enrichment by Cluster', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        sig_bar_file = output_dir / f'{dataset_name}_signature_bars.png'
        plt.savefig(sig_bar_file, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    [Saved] {sig_bar_file.name}")
    
    # ============================================================================
    # FIGURE 4: Comprehensive Summary Figure
    # ============================================================================
    print("  Creating comprehensive summary figure...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Signature Heatmap (top, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :])
    sns.heatmap(
        signature_pivot,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-3,
        vmax=3,
        cbar_kws={'label': 'Score Difference'},
        ax=ax1,
        linewidths=0.5,
        linecolor='gray'
    )
    ax1.set_title('Cell-Type Signature Scores', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Cluster', fontsize=10)
    ax1.set_ylabel('Signature', fontsize=10)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    
    # 2. Top DE Genes for first cluster (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    if len(clusters) > 0:
        de_file = output_dir / f'cluster_{clusters[0]}_differential_expression.csv'
        if de_file.exists():
            de_data = pd.read_csv(de_file)
            de_data['abs_log2fc'] = de_data['log2fc'].abs()
            top_genes = de_data.nlargest(10, 'abs_log2fc')
            if len(top_genes) > 0:
                colors = ['#d73027' if x > 0 else '#4575b4' for x in top_genes['log2fc']]
                ax2.barh(range(len(top_genes)), top_genes['log2fc'], color=colors, alpha=0.7)
                ax2.set_yticks(range(len(top_genes)))
                ax2.set_yticklabels(top_genes['gene'], fontsize=9)
                ax2.set_xlabel('log2FC', fontsize=10)
                ax2.set_title(f'Top DE Genes: Cluster {clusters[0]}', fontsize=11, fontweight='bold')
                ax2.axvline(0, color='black', linestyle='--', linewidth=0.5)
                ax2.grid(axis='x', alpha=0.3)
    
    # 3. Signature enrichment summary (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    enriched_counts = signature_scores[signature_scores['enriched'] == True].groupby('signature').size()
    depleted_counts = signature_scores[signature_scores['enriched'] == False].groupby('signature').size()
    
    x = np.arange(len(signatures))
    width = 0.35
    
    ax3.bar(x - width/2, [enriched_counts.get(sig, 0) for sig in signatures], 
           width, label='Enriched', color='#2ecc71', alpha=0.7)
    ax3.bar(x + width/2, [depleted_counts.get(sig, 0) for sig in signatures], 
           width, label='Depleted', color='#e74c3c', alpha=0.7)
    
    ax3.set_xlabel('Signature', fontsize=10)
    ax3.set_ylabel('Number of Clusters', fontsize=10)
    ax3.set_title('Signature Enrichment Summary', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([s.replace('_', ' ').title() for s in signatures], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Top pathways (if available) or cluster sizes (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    # Try to load pathway data
    pathway_files = list(output_dir.glob('cluster_*_pathways_*.csv'))
    if pathway_files:
        # Count pathways per cluster
        pathway_counts = {}
        for pf in pathway_files:
            cluster_id = int(pf.stem.split('_')[1])
            try:
                pathway_data = pd.read_csv(pf)
                if len(pathway_data) > 0:
                    pathway_counts[cluster_id] = len(pathway_data[pathway_data['Adjusted P-value'] < 0.05])
                else:
                    pathway_counts[cluster_id] = 0
            except:
                pathway_counts[cluster_id] = 0
        
        if pathway_counts:
            cluster_ids = sorted(pathway_counts.keys())
            counts = [pathway_counts[c] for c in cluster_ids]
            ax4.bar(range(len(cluster_ids)), counts, color='steelblue', alpha=0.7)
            ax4.set_xticks(range(len(cluster_ids)))
            ax4.set_xticklabels([f'C{c}' for c in cluster_ids])
            ax4.set_ylabel('Significant Pathways', fontsize=10)
            ax4.set_title('Pathway Enrichment by Cluster', fontsize=11, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
    else:
        # Show cluster sizes from signature data
        cluster_sizes = []
        for cluster_id in clusters:
            cluster_data = signature_scores[signature_scores['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                # Try to estimate from first row (this is approximate)
                cluster_sizes.append(1)  # Placeholder
            else:
                cluster_sizes.append(0)
        
        if sum(cluster_sizes) > 0:
            ax4.bar(range(len(clusters)), cluster_sizes, color='steelblue', alpha=0.7)
            ax4.set_xticks(range(len(clusters)))
            ax4.set_xticklabels([f'C{c}' for c in clusters])
            ax4.set_ylabel('Relative Size', fontsize=10)
            ax4.set_title('Cluster Sizes', fontsize=11, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
    
    # 5. Summary statistics (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Calculate summary stats
    total_de_genes = 0
    total_clusters_with_de = 0
    for cluster_id in clusters:
        de_file = output_dir / f'cluster_{cluster_id}_differential_expression.csv'
        if de_file.exists():
            try:
                de_data = pd.read_csv(de_file)
                total_de_genes += len(de_data)
                total_clusters_with_de += 1
            except:
                pass
    
    summary_text = f"""
    Dataset: {dataset_name.upper()}
    Number of Clusters: {n_clusters}
    Clusters with DE data: {total_clusters_with_de}
    Total DE genes: {total_de_genes}
    Signatures analyzed: {len(signatures)}
    """
    
    ax5.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax5.set_title('Summary Statistics', fontsize=11, fontweight='bold')
    
    plt.suptitle(f'{dataset_name.upper()}: Biological Interpretation Summary', 
                fontsize=16, fontweight='bold', y=0.995)
    
    summary_file = output_dir / f'{dataset_name}_biological_summary.png'
    plt.savefig(summary_file, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [Saved] {summary_file.name}")
    
    print(f"  [Complete] All figures saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create biological interpretation figures')
    parser.add_argument('--dataset', type=str, default='both', choices=['tcga', 'gse96058', 'both'],
                       help='Dataset to visualize (default: both)')
    parser.add_argument('--output-dir', type=str, default='results/biological_interpretation',
                       help='Output directory')
    parser.add_argument('--top-n-genes', type=int, default=10,
                       help='Number of top genes to show (default: 10)')
    
    args = parser.parse_args()
    
    datasets = ['tcga', 'gse96058'] if args.dataset == 'both' else [args.dataset]
    
    for dataset in datasets:
        create_biological_interpretation_figures(dataset, args.output_dir, args.top_n_genes)

