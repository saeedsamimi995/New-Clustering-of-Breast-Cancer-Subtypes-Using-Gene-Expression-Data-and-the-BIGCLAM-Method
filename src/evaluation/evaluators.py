"""
Evaluation Module

Evaluates clustering results against ground truth labels.
Metrics: ARI, NMI, Purity, F1-score (macro)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    f1_score,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder


def calculate_purity(true_labels, predicted_labels):
    """
    Calculate cluster purity.
    
    Args:
        true_labels: True class labels
        predicted_labels: Predicted cluster assignments
        
    Returns:
        float: Purity score
    """
    # Compute contingency matrix
    contingency_matrix = pd.crosstab(pd.Series(true_labels), pd.Series(predicted_labels))
    
    # Sum max values in each column (each cluster's dominant class)
    return contingency_matrix.max(axis=0).sum() / len(true_labels)


def evaluate_clustering(communities, target_labels, dataset_name="Dataset"):
    """
    Evaluate clustering against ground truth labels.
    
    Args:
        communities: Cluster assignments (numpy array)
        target_labels: Ground truth labels (list or numpy array)
        dataset_name: Name of dataset for display
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*80)
    print(f"EVALUATION: {dataset_name}")
    print("="*80)
    
    # Convert to numpy array
    target_labels = np.array(target_labels)
    communities = np.array(communities)
    
    # Remove samples with missing labels
    valid_mask = np.array([lbl != 'Unknown' and pd.notna(lbl) for lbl in target_labels])
    
    if not valid_mask.any():
        print("[WARNING] No valid labels found for evaluation")
        return None
    
    communities_valid = communities[valid_mask]
    labels_valid = target_labels[valid_mask]
    
    print(f"\nSamples evaluated: {len(labels_valid)}/{len(target_labels)}")
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels_valid)
    
    # Calculate metrics
    ari = adjusted_rand_score(labels_encoded, communities_valid)
    nmi = normalized_mutual_info_score(labels_encoded, communities_valid)
    purity = calculate_purity(labels_valid, communities_valid)
    f1_macro = f1_score(labels_encoded, communities_valid, average='macro', zero_division=0)
    
    # Store results
    results = {
        'ari': ari,
        'nmi': nmi,
        'purity': purity,
        'f1_macro': f1_macro,
        'n_samples': len(labels_valid),
        'n_clusters': len(set(communities_valid))
    }
    
    # Display results
    print("\nMetrics:")
    print(f"  Adjusted Rand Index (ARI):   {ari:.4f}")
    print(f"  Normalized Mutual Info (NMI): {nmi:.4f}")
    print(f"  Purity:                        {purity:.4f}")
    print(f"  F1-score (macro):              {f1_macro:.4f}")
    
    # Confusion matrix
    print("\nCluster vs. Label Distribution:")
    result_df = pd.DataFrame({
        'Cluster': communities_valid,
        'Label': labels_valid
    })
    
    contingency = pd.crosstab(result_df['Label'], result_df['Cluster'], margins=True)
    print(contingency)
    
    # Calculate per-cluster statistics
    print("\nPer-Cluster Statistics:")
    for cluster_id in sorted(set(communities_valid)):
        cluster_mask = communities_valid == cluster_id
        cluster_labels = labels_valid[cluster_mask]
        label_counts = pd.Series(cluster_labels).value_counts()
        
        dominant_label = label_counts.index[0]
        dominant_count = label_counts.iloc[0]
        dominant_pct = dominant_count / len(cluster_labels) * 100
        
        print(f"  Cluster {cluster_id}: {len(cluster_labels)} samples")
        print(f"    Dominant label: {dominant_label} ({dominant_pct:.1f}%)")
        if len(label_counts) > 1:
            print(f"    Other labels: {dict(label_counts.iloc[1:3].items())}")
    
    return results, result_df


def create_confusion_matrix_heatmap(communities, target_labels, dataset_name, output_dir):
    """
    Create and save confusion matrix heatmap.
    
    Args:
        communities: Cluster assignments
        target_labels: Ground truth labels
        dataset_name: Dataset name
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter valid samples
    valid_mask = np.array([lbl != 'Unknown' and pd.notna(lbl) for lbl in target_labels])
    communities_valid = communities[valid_mask]
    labels_valid = np.array(target_labels)[valid_mask]
    
    # Create confusion matrix
    cm = confusion_matrix(labels_valid, communities_valid)
    
    # Normalize by row (true label distribution)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'{dataset_name}: Confusion Matrix (Counts)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Cluster')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1], 
                cbar_kws={'label': 'Proportion'})
    axes[1].set_title(f'{dataset_name}: Confusion Matrix (Proportions)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Cluster')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name.lower()}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"\n[Saved] Confusion matrix: {output_dir / dataset_name.lower()}_confusion_matrix.png")
    plt.close()


def create_cluster_distribution_plot(communities, target_labels, dataset_name, output_dir):
    """
    Create cluster-label distribution bar plot.
    
    Args:
        communities: Cluster assignments
        target_labels: Ground truth labels
        dataset_name: Dataset name
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter valid samples
    valid_mask = np.array([lbl != 'Unknown' and pd.notna(lbl) for lbl in target_labels])
    communities_valid = communities[valid_mask]
    labels_valid = np.array(target_labels)[valid_mask]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Cluster': communities_valid,
        'Label': labels_valid
    })
    
    # Pivot table for plotting
    pivot_table = pd.crosstab(df['Cluster'], df['Label'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_table.plot(kind='bar', ax=ax, stacked=True, colormap='tab10')
    ax.set_title(f'{dataset_name}: Cluster Distribution by Label')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Samples')
    ax.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name.lower()}_cluster_distribution.png", dpi=300, bbox_inches='tight')
    print(f"[Saved] Distribution plot: {output_dir / dataset_name.lower()}_cluster_distribution.png")
    plt.close()


def evaluate_all_datasets(clustering_dir='data/clusterings', targets_dir='data/processed',
                         output_dir='results/evaluation'):
    """
    Evaluate all clustering results.
    
    Args:
        clustering_dir: Directory with clustering results
        targets_dir: Directory with target labels
        output_dir: Output directory for plots
        
    Returns:
        dict: Evaluation results for all datasets
    """
    clustering_dir = Path(clustering_dir)
    targets_dir = Path(targets_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all clustering files
    clustering_files = list(clustering_dir.glob('*_communities.npy'))
    
    if not clustering_files:
        print(f"No clustering files found in {clustering_dir}")
        return {}
    
    results = {}
    
    for clustering_file in clustering_files:
        dataset_name = clustering_file.stem.replace('_communities', '')
        
        # Load communities
        communities = np.load(clustering_file)
        
        # Load targets
        target_file = targets_dir / f"{dataset_name}_targets.pkl"
        if not target_file.exists():
            print(f"\n[WARNING] Target file not found: {target_file}")
            continue
        
        with open(target_file, 'rb') as f:
            targets_data = pickle.load(f)
        
        target_labels = targets_data['target_labels']
        
        # Evaluate
        eval_results, result_df = evaluate_clustering(communities, target_labels, dataset_name)
        
        if eval_results:
            # Create visualizations
            create_confusion_matrix_heatmap(communities, target_labels, dataset_name, output_dir)
            create_cluster_distribution_plot(communities, target_labels, dataset_name, output_dir)
            
            results[dataset_name] = eval_results
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate clustering results')
    parser.add_argument('--clustering_dir', type=str, default='data/clusterings', help='Clustering directory')
    parser.add_argument('--targets_dir', type=str, default='data/processed', help='Targets directory')
    parser.add_argument('--output_dir', type=str, default='results/evaluation', help='Output directory')
    
    args = parser.parse_args()
    
    evaluate_all_datasets(
        args.clustering_dir,
        args.targets_dir,
        args.output_dir
    )

