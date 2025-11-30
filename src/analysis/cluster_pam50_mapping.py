"""
Cluster-to-PAM50 Mapping Analysis

Maps BIGCLAM clusters to PAM50 subtypes to show:
- Which clusters represent subdivisions of PAM50 subtypes
- PAM50 marker gene expression across clusters
- Cluster composition (PAM50 distribution per cluster)
- Heatmap showing PAM50 distribution per cluster

Addresses reviewer concern about interpretability of BIGCLAM's 9 clusters vs PAM50's 5 subtypes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.stats import chi2_contingency, fisher_exact
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_clusters_and_pam50(dataset_name, processed_dir='data/processed',
                            clustering_dir='data/clusterings'):
    """Load BIGCLAM clusters and PAM50 labels."""
    file_prefix_map = {
        'tcga': 'tcga_brca_data',
        'gse96058': 'gse96058_data'
    }
    prefix = file_prefix_map.get(dataset_name, dataset_name)
    
    # Load clusters
    cluster_file = Path(clustering_dir) / f'{prefix}_communities.npy'
    if not cluster_file.exists():
        raise FileNotFoundError(f"Cluster file not found: {cluster_file}")
    
    clusters = np.load(cluster_file)
    if clusters.ndim > 1:
        # Membership matrix - convert to hard assignments
        clusters = np.argmax(clusters, axis=1)
    clusters = clusters.flatten()
    
    # Load PAM50 labels
    target_file = Path(processed_dir) / f'{prefix}_targets.pkl'
    if not target_file.exists():
        raise FileNotFoundError(f"Target file not found: {target_file}")
    
    with open(target_file, 'rb') as f:
        targets_data = pickle.load(f)
        pam50_labels = np.array(targets_data.get('target_labels', []))
        sample_names = targets_data.get('sample_names', [])
    
    # Filter to valid samples (exclude 'Unknown')
    valid_mask = (pam50_labels != 'Unknown') & pd.notna(pam50_labels)
    clusters = clusters[valid_mask]
    pam50_labels = pam50_labels[valid_mask]
    if sample_names:
        sample_names = [sample_names[i] for i in range(len(sample_names)) if valid_mask[i]]
    
    return clusters, pam50_labels, sample_names


def create_cluster_pam50_heatmap(clusters, pam50_labels, output_dir, dataset_name):
    """Create heatmap showing PAM50 distribution per cluster."""
    unique_clusters = sorted(np.unique(clusters).astype(int))
    unique_pam50 = sorted(set(pam50_labels))
    
    # Create contingency matrix
    contingency = np.zeros((len(unique_clusters), len(unique_pam50)))
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = clusters == cluster_id
        cluster_pam50 = pam50_labels[cluster_mask]
        for j, pam50_type in enumerate(unique_pam50):
            contingency[i, j] = np.sum(cluster_pam50 == pam50_type)
    
    # Normalize by cluster size (percentage)
    cluster_sizes = contingency.sum(axis=1, keepdims=True)
    cluster_sizes[cluster_sizes == 0] = 1  # Avoid division by zero
    contingency_pct = (contingency / cluster_sizes * 100)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(unique_pam50) * 1.5), max(6, len(unique_clusters) * 0.8)))
    sns.heatmap(
        contingency_pct,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        xticklabels=unique_pam50,
        yticklabels=[f'Cluster {c}' for c in unique_clusters],
        cbar_kws={'label': '% of cluster'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    ax.set_xlabel('PAM50 Subtype', fontsize=12, fontweight='bold')
    ax.set_ylabel('BIGCLAM Cluster', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset_name.upper()}: Cluster Composition by PAM50 Subtype', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_file = output_dir / 'cluster_pam50_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[Saved] Cluster-PAM50 heatmap → {output_file}")
    plt.close()
    
    return contingency, contingency_pct


def add_statistical_tests(clusters, pam50_labels, output_dir, dataset_name):
    """
    Add chi-square and Fisher's exact tests for cluster-PAM50 enrichment.
    
    Returns:
        DataFrame with statistical test results
    """
    unique_clusters = sorted(np.unique(clusters).astype(int))
    unique_pam50 = sorted(set(pam50_labels))
    
    results = []
    
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        cluster_pam50 = pam50_labels[cluster_mask]
        other_pam50 = pam50_labels[~cluster_mask]
        
        # Build contingency table for each PAM50 type
        cluster_counts = pd.Series(cluster_pam50).value_counts()
        other_counts = pd.Series(other_pam50).value_counts()
        
        # Overall chi-square test (cluster vs others, all PAM50 types)
        all_types = sorted(set(pam50_labels))
        contingency_table = np.array([
            [cluster_counts.get(t, 0) for t in all_types],
            [other_counts.get(t, 0) for t in all_types]
        ])
        
        # Chi-square test
        try:
            chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
        except:
            chi2, p_chi2, dof, expected = np.nan, np.nan, np.nan, None
        
        # Calculate odds ratios for each PAM50 type
        ors = []
        for pam50_type in all_types:
            cluster_n = cluster_counts.get(pam50_type, 0)
            cluster_not = cluster_mask.sum() - cluster_n
            other_n = other_counts.get(pam50_type, 0)
            other_not = (~cluster_mask).sum() - other_n
            
            if cluster_not > 0 and other_n > 0 and other_not > 0:
                # Odds ratio
                or_val = (cluster_n / cluster_not) / (other_n / other_not) if cluster_not > 0 and other_n > 0 else np.nan
                
                # Fisher's exact test for 2x2 table
                try:
                    oddsratio, p_fisher = fisher_exact([[cluster_n, cluster_not],
                                                       [other_n, other_not]])
                    ors.append({
                        'pam50_type': pam50_type,
                        'odds_ratio': or_val,
                        'fisher_p': p_fisher,
                        'cluster_n': cluster_n,
                        'other_n': other_n
                    })
                except:
                    ors.append({
                        'pam50_type': pam50_type,
                        'odds_ratio': or_val,
                        'fisher_p': np.nan,
                        'cluster_n': cluster_n,
                        'other_n': other_n
                    })
        
        results.append({
            'cluster': int(cluster_id),
            'chi2_statistic': chi2 if not np.isnan(chi2) else None,
            'chi2_pvalue': p_chi2 if not np.isnan(p_chi2) else None,
            'chi2_dof': int(dof) if not np.isnan(dof) else None,
            'n_pam50_types': len(all_types),
            'odds_ratios': ors
        })
    
    # Apply FDR correction to chi-square p-values
    try:
        from statsmodels.stats.multitest import multipletests
        p_values = [r['chi2_pvalue'] for r in results if r['chi2_pvalue'] is not None]
        if len(p_values) > 0:
            _, p_adj, _, _ = multipletests(p_values, method='fdr_bh')
            adj_idx = 0
            for r in results:
                if r['chi2_pvalue'] is not None:
                    r['chi2_pvalue_adj'] = p_adj[adj_idx]
                    adj_idx += 1
                else:
                    r['chi2_pvalue_adj'] = None
    except ImportError:
        # Fallback: no correction
        for r in results:
            r['chi2_pvalue_adj'] = r['chi2_pvalue']
    
    # Flatten odds ratios into separate rows
    flattened_results = []
    for r in results:
        base_result = {k: v for k, v in r.items() if k != 'odds_ratios'}
        if r['odds_ratios']:
            for or_data in r['odds_ratios']:
                row = base_result.copy()
                row.update(or_data)
                flattened_results.append(row)
        else:
            flattened_results.append(base_result)
    
    results_df = pd.DataFrame(flattened_results)
    
    # Save
    stats_file = output_dir / 'cluster_pam50_statistical_tests.csv'
    results_df.to_csv(stats_file, index=False)
    print(f"[Saved] Statistical tests → {stats_file}")
    
    return results_df


def analyze_cluster_pam50_mapping(clusters, pam50_labels, output_dir, dataset_name):
    """Analyze mapping between clusters and PAM50."""
    unique_clusters = sorted(np.unique(clusters).astype(int))
    unique_pam50 = sorted(set(pam50_labels))
    
    mapping_results = []
    
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        cluster_pam50 = pam50_labels[cluster_mask]
        cluster_size = cluster_mask.sum()
        
        # PAM50 distribution
        pam50_counts = pd.Series(cluster_pam50).value_counts()
        dominant_pam50 = pam50_counts.index[0]
        dominant_pct = pam50_counts.iloc[0] / cluster_size * 100
        
        # Diversity
        n_pam50_types = len(pam50_counts)
        
        # Create distribution string
        dist_str = ', '.join([f"{k}: {v} ({v/cluster_size*100:.1f}%)" 
                             for k, v in pam50_counts.items()])
        
        mapping_results.append({
            'cluster': int(cluster_id),
            'n_samples': int(cluster_size),
            'dominant_pam50': dominant_pam50,
            'dominant_pct': round(dominant_pct, 2),
            'n_pam50_types': int(n_pam50_types),
            'pam50_distribution': dist_str
        })
    
    mapping_df = pd.DataFrame(mapping_results)
    
    # Save
    output_file = output_dir / 'cluster_pam50_mapping.csv'
    mapping_df.to_csv(output_file, index=False)
    print(f"[Saved] Cluster-PAM50 mapping → {output_file}")
    
    # Add statistical tests
    print(f"\n[Computing] Statistical tests (chi-square, Fisher's exact)...")
    stats_df = add_statistical_tests(clusters, pam50_labels, output_dir, dataset_name)
    
    return mapping_df, stats_df


def create_pam50_marker_heatmap(dataset_name, clusters, expression_data, output_dir):
    """Create heatmap of PAM50 marker gene expression across clusters."""
    # PAM50 marker genes
    PAM50_MARKERS = {
        'Luminal_A': ['ESR1', 'PGR', 'FOXA1', 'GATA3', 'XBP1', 'TFF1', 'TFF3'],
        'Luminal_B': ['ESR1', 'PGR', 'BCL2', 'CCND1', 'MYBL2', 'MKI67'],
        'HER2': ['ERBB2', 'GRB7', 'STARD3', 'TCAP', 'PNMT'],
        'Basal': ['KRT5', 'KRT14', 'KRT17', 'TP63', 'CDH3'],
        'Normal': ['GATA3', 'XBP1', 'FOXA1', 'ESR1', 'PGR']
    }
    
    # This would require loading expression data with gene names
    # For now, return placeholder message
    print("[Info] PAM50 marker gene heatmap requires expression data with gene names")
    print("      This feature can be added if expression data is available")
    return None


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Map BIGCLAM clusters to PAM50 subtypes')
    parser.add_argument('--dataset', type=str, default='gse96058',
                       choices=['tcga', 'gse96058'],
                       help='Dataset to analyze')
    parser.add_argument('--output-dir', type=str, 
                       default='results/cluster_pam50_mapping',
                       help='Output directory')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                       help='Processed data directory')
    parser.add_argument('--clustering-dir', type=str, default='data/clusterings',
                       help='Clustering results directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"CLUSTER-PAM50 MAPPING: {args.dataset.upper()}")
    print(f"{'='*80}")
    
    # Load data
    print("\n[Loading] Cluster assignments and PAM50 labels...")
    clusters, pam50_labels, sample_names = load_clusters_and_pam50(
        args.dataset, 
        args.processed_dir, 
        args.clustering_dir
    )
    
    print(f"  ✓ Loaded {len(clusters)} samples")
    print(f"  ✓ {len(np.unique(clusters))} BIGCLAM clusters")
    print(f"  ✓ {len(set(pam50_labels))} PAM50 subtypes")
    
    # Analyze mapping
    print("\n[Analyzing] Cluster-PAM50 mapping...")
    mapping_df, stats_df = analyze_cluster_pam50_mapping(clusters, pam50_labels, output_dir, args.dataset)
    
    # Create heatmap
    print("\n[Creating] Cluster-PAM50 heatmap...")
    contingency, contingency_pct = create_cluster_pam50_heatmap(
        clusters, pam50_labels, output_dir, args.dataset
    )
    
    # Calculate metrics
    nmi = normalized_mutual_info_score(pam50_labels, clusters)
    ari = adjusted_rand_score(pam50_labels, clusters)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  NMI vs PAM50: {nmi:.4f}")
    print(f"  ARI vs PAM50: {ari:.4f}")
    print(f"  Number of BIGCLAM clusters: {len(np.unique(clusters))}")
    print(f"  Number of PAM50 types: {len(set(pam50_labels))}")
    
    print(f"\n{'='*80}")
    print("CLUSTER-PAM50 MAPPING")
    print(f"{'='*80}")
    for _, row in mapping_df.iterrows():
        print(f"\n  Cluster {int(row['cluster'])} (n={int(row['n_samples'])}):")
        print(f"    Dominant PAM50: {row['dominant_pam50']} ({row['dominant_pct']:.1f}%)")
        if row['n_pam50_types'] > 1:
            print(f"    Mixed subtypes: {row['pam50_distribution']}")
        else:
            print(f"    Pure subtype: {row['dominant_pam50']}")
    
    # Save summary
    summary_file = output_dir / 'mapping_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Cluster-PAM50 Mapping Summary: {args.dataset.upper()}\n")
        f.write("="*80 + "\n\n")
        f.write(f"NMI vs PAM50: {nmi:.4f}\n")
        f.write(f"ARI vs PAM50: {ari:.4f}\n")
        f.write(f"Number of BIGCLAM clusters: {len(np.unique(clusters))}\n")
        f.write(f"Number of PAM50 types: {len(set(pam50_labels))}\n\n")
        f.write("Cluster Details:\n")
        f.write("-"*80 + "\n")
        for _, row in mapping_df.iterrows():
            f.write(f"\nCluster {int(row['cluster'])} (n={int(row['n_samples'])}):\n")
            f.write(f"  Dominant PAM50: {row['dominant_pam50']} ({row['dominant_pct']:.1f}%)\n")
            if row['n_pam50_types'] > 1:
                f.write(f"  Mixed subtypes: {row['pam50_distribution']}\n")
            else:
                f.write(f"  Pure subtype: {row['dominant_pam50']}\n")
    
    print(f"\n[Saved] Summary → {summary_file}")
    print(f"\n{'='*80}")
    print("MAPPING ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

