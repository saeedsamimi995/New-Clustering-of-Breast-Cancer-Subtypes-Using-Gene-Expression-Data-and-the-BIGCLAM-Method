"""
Visualization Module

Creates t-SNE/UMAP plots colored by clusters vs. targets and heatmaps.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[Note] UMAP not available. Install with: pip install umap-learn")


def create_tsne_plot(expression_data, communities, target_labels, dataset_name, 
                    output_dir='results/visualization', perplexity=30):
    """
    Create t-SNE visualization colored by clusters and target labels.
    
    Args:
        expression_data: Expression matrix (n_samples x n_features)
        communities: Cluster assignments
        target_labels: Ground truth labels
        dataset_name: Dataset name
        output_dir: Output directory
        perplexity: t-SNE perplexity parameter
    """
    print(f"\n[Visualization] Creating t-SNE plot for {dataset_name}...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Reduce dimensionality first with PCA for speed
    print("    Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=50, random_state=42)
    data_pca = pca.fit_transform(expression_data)
    
    print(f"    Computing t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    data_tsne = tsne.fit_transform(data_pca)
    
    # Encode labels for coloring
    le = LabelEncoder()
    target_encoded = le.fit_transform([str(lbl) for lbl in target_labels])
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Colored by clusters
    scatter1 = axes[0].scatter(data_tsne[:, 0], data_tsne[:, 1], 
                              c=communities, cmap='tab10', s=10, alpha=0.6)
    axes[0].set_title(f'{dataset_name}: BIGCLAM Clusters')
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Plot 2: Colored by target labels
    scatter2 = axes[1].scatter(data_tsne[:, 0], data_tsne[:, 1], 
                              c=target_encoded, cmap='tab20', s=10, alpha=0.6)
    axes[1].set_title(f'{dataset_name}: Target Labels')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    output_file = output_dir / f"{dataset_name.lower()}_tsne.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"    [Saved] {output_file}")
    plt.close()


def create_umap_plot(expression_data, communities, target_labels, dataset_name, 
                    output_dir='results/visualization', n_neighbors=15):
    """
    Create UMAP visualization colored by clusters and target labels.
    
    Args:
        expression_data: Expression matrix (n_samples x n_features)
        communities: Cluster assignments
        target_labels: Ground truth labels
        dataset_name: Dataset name
        output_dir: Output directory
        n_neighbors: UMAP n_neighbors parameter
    """
    if not UMAP_AVAILABLE:
        print(f"[Skipping] UMAP not available for {dataset_name}")
        return
    
    print(f"\n[Visualization] Creating UMAP plot for {dataset_name}...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Reduce dimensionality first with PCA
    print("    Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=50, random_state=42)
    data_pca = pca.fit_transform(expression_data)
    
    print(f"    Computing UMAP (n_neighbors={n_neighbors})...")
    umap_model = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
    data_umap = umap_model.fit_transform(data_pca)
    
    # Encode labels
    le = LabelEncoder()
    target_encoded = le.fit_transform([str(lbl) for lbl in target_labels])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Clusters
    scatter1 = axes[0].scatter(data_umap[:, 0], data_umap[:, 1], 
                              c=communities, cmap='tab10', s=10, alpha=0.6)
    axes[0].set_title(f'{dataset_name}: BIGCLAM Clusters (UMAP)')
    axes[0].set_xlabel('UMAP Dimension 1')
    axes[0].set_ylabel('UMAP Dimension 2')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Plot 2: Targets
    scatter2 = axes[1].scatter(data_umap[:, 0], data_umap[:, 1], 
                              c=target_encoded, cmap='tab20', s=10, alpha=0.6)
    axes[1].set_title(f'{dataset_name}: Target Labels (UMAP)')
    axes[1].set_xlabel('UMAP Dimension 1')
    axes[1].set_ylabel('UMAP Dimension 2')
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    output_file = output_dir / f"{dataset_name.lower()}_umap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"    [Saved] {output_file}")
    plt.close()


def create_membership_heatmap(membership_matrix, target_labels, dataset_name, 
                              output_dir='results/visualization'):
    """
    Create heatmap of membership scores vs. subtype distributions.
    
    Args:
        membership_matrix: Membership matrix (n_samples x n_communities)
        target_labels: Ground truth labels
        dataset_name: Dataset name
        output_dir: Output directory
    """
    print(f"\n[Visualization] Creating membership heatmap for {dataset_name}...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame for membership by label
    df_membership = pd.DataFrame(membership_matrix)
    df_membership['Label'] = target_labels
    
    # Group by label and compute mean membership
    label_means = df_membership.groupby('Label').mean()
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, max(6, len(label_means)*0.5)))
    sns.heatmap(label_means, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Mean Membership'})
    ax.set_title(f'{dataset_name}: Mean Membership Score by Label')
    ax.set_xlabel('Community')
    ax.set_ylabel('Target Label')
    
    plt.tight_layout()
    output_file = output_dir / f"{dataset_name.lower()}_membership_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"    [Saved] {output_file}")
    plt.close()


def create_all_visualizations(processed_dir='data/processed', 
                             clustering_dir='data/clusterings',
                             output_dir='results/visualization'):
    """
    Create all visualizations for all datasets.
    
    Args:
        processed_dir: Directory with processed expression data
        clustering_dir: Directory with clustering results
        output_dir: Output directory
    """
    processed_dir = Path(processed_dir)
    clustering_dir = Path(clustering_dir)
    
    # Find all processed files
    processed_files = list(processed_dir.glob('*_processed.npy'))
    
    if not processed_files:
        print(f"No processed files found in {processed_dir}")
        return
    
    for processed_file in processed_files:
        dataset_name = processed_file.stem.replace('_processed', '')
        
        print("\n" + "="*80)
        print(f"VISUALIZING: {dataset_name}")
        print("="*80)
        
        # Load expression data
        expression_data = np.load(processed_file)
        
        # Load communities
        clustering_file = clustering_dir / f"{dataset_name}_communities.npy"
        if not clustering_file.exists():
            print(f"    [SKIP] No clustering file found: {clustering_file}")
            continue
        
        communities = np.load(clustering_file)
        
        # Load targets
        target_file = processed_dir / f"{dataset_name}_targets.pkl"
        if not target_file.exists():
            print(f"    [SKIP] No target file found: {target_file}")
            continue
        
        import pickle
        with open(target_file, 'rb') as f:
            targets_data = pickle.load(f)
        
        target_labels = targets_data['target_labels']
        
        # Create visualizations
        create_tsne_plot(expression_data, communities, target_labels, dataset_name, output_dir)
        create_umap_plot(expression_data, communities, target_labels, dataset_name, output_dir)
        
        # Load membership matrix
        membership_file = clustering_dir / f"{dataset_name}_communities_membership.npy"
        if membership_file.exists():
            membership_matrix = np.load(membership_file)
            create_membership_heatmap(membership_matrix, target_labels, dataset_name, output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create visualizations')
    parser.add_argument('--processed_dir', type=str, default='data/processed', help='Processed data directory')
    parser.add_argument('--clustering_dir', type=str, default='data/clusterings', help='Clustering directory')
    parser.add_argument('--output_dir', type=str, default='results/visualization', help='Output directory')
    
    args = parser.parse_args()
    
    create_all_visualizations(
        args.processed_dir,
        args.clustering_dir,
        args.output_dir
    )

