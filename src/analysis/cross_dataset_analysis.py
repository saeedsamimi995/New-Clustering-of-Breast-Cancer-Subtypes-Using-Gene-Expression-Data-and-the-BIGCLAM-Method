"""
Cross-Dataset Analysis Module

Computes correlations between community centroids across datasets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import pairwise_distances


def compute_community_centroids(expression_data, communities, membership_matrix):
    """
    Compute centroid (mean expression) for each community.
    
    Args:
        expression_data: Expression matrix (n_samples x n_features)
        communities: Cluster assignments
        membership_matrix: Membership matrix (n_samples x n_communities)
        
    Returns:
        numpy array: Centroids (n_communities x n_features)
    """
    print("\n[Computing Centroids]...")
    
    centroids = []
    community_ids = sorted(set(communities))
    
    for comm_id in community_ids:
        # Get samples in this community
        members = expression_data[communities == comm_id]
        centroid = members.mean(axis=0)
        centroids.append(centroid)
        print(f"    Community {comm_id}: {len(members)} samples")
    
    centroids = np.array(centroids)
    print(f"    Centroids shape: {centroids.shape}")
    
    return centroids, community_ids


def compute_cross_dataset_correlation(centroids_tcga, centroids_gse, method='pearson'):
    """
    Compute correlations between community centroids across datasets.
    
    Args:
        centroids_tcga: TCGA community centroids
        centroids_gse: GSE96058 community centroids
        method: Correlation method ('pearson', 'spearman', 'cosine')
        
    Returns:
        numpy array: Correlation matrix
    """
    print("\n[Cross-Dataset Correlation]...")
    print(f"    Method: {method}")
    print(f"    TCGA communities: {centroids_tcga.shape[0]}")
    print(f"    GSE96058 communities: {centroids_gse.shape[0]}")
    
    # Match gene dimensions (intersection of features)
    # Note: In practice, both should use same genes after preprocessing
    n_features = min(centroids_tcga.shape[1], centroids_gse.shape[1])
    if centroids_tcga.shape[1] != centroids_gse.shape[1]:
        print(f"    [WARNING] Feature dimension mismatch. Using first {n_features} features.")
    
    centroids_tcga = centroids_tcga[:, :n_features]
    centroids_gse = centroids_gse[:, :n_features]
    
    # Compute pairwise correlations
    if method == 'cosine':
        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        correlations = cosine_similarity(centroids_tcga, centroids_gse)
    elif method == 'pearson':
        # Pearson correlation
        correlations = np.corrcoef(centroids_tcga, centroids_gse)[:centroids_tcga.shape[0], 
                                                                   centroids_tcga.shape[0]:]
    else:
        # Spearman correlation
        from scipy.stats import spearmanr
        correlations = np.zeros((centroids_tcga.shape[0], centroids_gse.shape[0]))
        for i in range(centroids_tcga.shape[0]):
            for j in range(centroids_gse.shape[0]):
                correlations[i, j] = spearmanr(centroids_tcga[i], centroids_gse[j])[0]
    
    print(f"    Correlation matrix shape: {correlations.shape}")
    print(f"    Correlation range: [{correlations.min():.3f}, {correlations.max():.3f}]")
    
    return correlations


def find_matching_communities(correlation_matrix, threshold=0.7):
    """
    Find matching communities across datasets based on high correlation.
    
    Args:
        correlation_matrix: Cross-dataset correlation matrix
        threshold: Correlation threshold for matching
        
    Returns:
        list: List of (tcga_cluster, gse_cluster, correlation) tuples
    """
    print(f"\n[Finding Matching Communities] Threshold={threshold}...")
    
    matches = []
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            if correlation_matrix[i, j] >= threshold:
                matches.append((i, j, correlation_matrix[i, j]))
    
    # Sort by correlation
    matches.sort(key=lambda x: x[2], reverse=True)
    
    print(f"    Found {len(matches)} matches above threshold")
    for tcga_id, gse_id, corr in matches[:10]:  # Top 10
        print(f"    TCGA Community {tcga_id} ↔ GSE Community {gse_id}: {corr:.3f}")
    
    return matches


def create_correlation_heatmap(correlation_matrix, output_file, 
                               tcga_communities=None, gse_communities=None):
    """
    Create heatmap of cross-dataset correlations.
    
    Args:
        correlation_matrix: Correlation matrix
        output_file: Output file path
        tcga_communities: TCGA community labels
        gse_communities: GSE community labels
    """
    print(f"\n[Creating Correlation Heatmap]...")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame for better labeling
    if tcga_communities is None:
        tcga_communities = [f'TCGA_{i}' for i in range(correlation_matrix.shape[0])]
    if gse_communities is None:
        gse_communities = [f'GSE_{i}' for i in range(correlation_matrix.shape[1])]
    
    df_corr = pd.DataFrame(correlation_matrix, 
                          index=tcga_communities,
                          columns=gse_communities)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(tcga_communities)*0.5)))
    sns.heatmap(df_corr, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                center=0, vmin=-1, vmax=1, ax=ax,
                cbar_kws={'label': 'Correlation'})
    ax.set_title('Cross-Dataset Community Correlations')
    ax.set_xlabel('GSE96058 Communities')
    ax.set_ylabel('TCGA-BRCA Communities')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"    [Saved] {output_file}")
    plt.close()


def create_dendrogram_heatmap(centroids_tcga, centroids_gse, output_file,
                              tcga_communities=None, gse_communities=None):
    """
    Create dendrogram showing hierarchical clustering of community centroids.
    
    Args:
        centroids_tcga: TCGA centroids
        centroids_gse: GSE centroids
        output_file: Output file path
        tcga_communities: TCGA community labels
        gse_communities: GSE community labels
    """
    print(f"\n[Creating Dendrogram]...")
    
    # Combine centroids
    all_centroids = np.vstack([centroids_tcga, centroids_gse])
    
    # Compute distance matrix
    distances = pdist(all_centroids, metric='correlation')
    
    # Hierarchical clustering
    linkage_matrix = linkage(distances, method='ward')
    
    # Create labels
    if tcga_communities is None:
        tcga_communities = [f'TCGA_C{i}' for i in range(len(centroids_tcga))]
    if gse_communities is None:
        gse_communities = [f'GSE_C{i}' for i in range(len(centroids_gse))]
    
    labels = tcga_communities + gse_communities
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, max(8, len(labels)*0.3)))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, ax=ax)
    ax.set_title('Hierarchical Clustering of Community Centroids')
    ax.set_ylabel('Distance')
    plt.tight_layout()
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"    [Saved] {output_file}")
    plt.close()


def analyze_cross_dataset_consistency(processed_dir='data/processed',
                                     clustering_dir='data/clusterings',
                                     output_dir='results/cross_dataset'):
    """
    Complete cross-dataset analysis.
    
    Args:
        processed_dir: Directory with processed data
        clustering_dir: Directory with clustering results
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("CROSS-DATASET CONSISTENCY ANALYSIS")
    print("="*80)
    
    processed_dir = Path(processed_dir)
    clustering_dir = Path(clustering_dir)
    output_dir = Path(output_dir)
    
    # Load TCGA data
    # Note: Output files no longer include "_target_added" suffix since target is removed from processed data
    tcga_processed = processed_dir / "tcga_brca_data_processed.npy"
    tcga_communities = clustering_dir / "tcga_brca_data_communities.npy"
    tcga_membership = clustering_dir / "tcga_brca_data_communities_membership.npy"
    
    # Load GSE data
    # Note: Output files no longer include "_target_added" suffix since target is removed from processed data
    gse_processed = processed_dir / "gse96058_data_processed.npy"
    gse_communities = clustering_dir / "gse96058_data_communities.npy"
    gse_membership = clustering_dir / "gse96058_data_communities_membership.npy"
    
    # Check if files exist
    if not all([f.exists() for f in [tcga_processed, tcga_communities, tcga_membership,
                                     gse_processed, gse_communities, gse_membership]]):
        print("[ERROR] Required files not found. Please run preprocessing and clustering first.")
        return
    
    # Load data
    print("\n[Loading Data]...")
    tcga_expression = np.load(tcga_processed)
    tcga_communities = np.load(tcga_communities)
    tcga_membership = np.load(tcga_membership)
    
    gse_expression = np.load(gse_processed)
    gse_communities = np.load(gse_communities)
    gse_membership = np.load(gse_membership)
    
    # Compute centroids
    print("\n" + "-"*80)
    print("TCGA-BRCA")
    print("-"*80)
    tcga_centroids, tcga_ids = compute_community_centroids(tcga_expression, tcga_communities, tcga_membership)
    
    print("\n" + "-"*80)
    print("GSE96058")
    print("-"*80)
    gse_centroids, gse_ids = compute_community_centroids(gse_expression, gse_communities, gse_membership)
    
    # Compute correlations
    print("\n" + "-"*80)
    print("CORRELATION ANALYSIS")
    print("-"*80)
    correlation_matrix = compute_cross_dataset_correlation(tcga_centroids, gse_centroids, method='pearson')
    
    # Find matches
    matches = find_matching_communities(correlation_matrix, threshold=0.5)
    
    # Create visualizations
    print("\n" + "-"*80)
    print("VISUALIZATIONS")
    print("-"*80)
    create_correlation_heatmap(correlation_matrix, 
                               output_dir / "community_correlations.png",
                               tcga_communities=[f'TCGA_C{i}' for i in tcga_ids],
                               gse_communities=[f'GSE_C{i}' for i in gse_ids])
    
    create_dendrogram_heatmap(tcga_centroids, gse_centroids,
                             output_dir / "community_dendrogram.png",
                             tcga_communities=[f'TCGA_C{i}' for i in tcga_ids],
                             gse_communities=[f'GSE_C{i}' for i in gse_ids])
    
    # Save results
    results = {
        'correlation_matrix': correlation_matrix.tolist(),
        'matches': matches,
        'tcga_communities': tcga_ids,
        'gse_communities': gse_ids
    }
    
    with open(output_dir / "cross_dataset_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print("\n[Analysis Complete] Results saved to:", output_dir)
    print("\nInterpretation:")
    print("  → High correlations indicate cross-cohort consistency")
    print("  → Matching communities suggest generalizable molecular signatures")
    print("  → Dendrogram shows hierarchical relationships across datasets")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Cross-dataset consistency analysis')
    parser.add_argument('--processed_dir', type=str, default='data/processed', help='Processed data directory')
    parser.add_argument('--clustering_dir', type=str, default='data/clusterings', help='Clustering directory')
    parser.add_argument('--output_dir', type=str, default='results/cross_dataset', help='Output directory')
    
    args = parser.parse_args()
    
    analyze_cross_dataset_consistency(
        args.processed_dir,
        args.clustering_dir,
        args.output_dir
    )

