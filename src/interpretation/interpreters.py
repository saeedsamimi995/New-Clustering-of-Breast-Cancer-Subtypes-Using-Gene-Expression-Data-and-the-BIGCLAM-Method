"""
Interpretation Module

Interprets clustering results and provides biological insights.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle


def interpret_results(ari, nmi, dataset_name, label_type):
    """
    Interpret clustering results based on metrics.
    
    Args:
        ari: Adjusted Rand Index
        nmi: Normalized Mutual Information
        dataset_name: Dataset name
        label_type: Type of labels (PAM50, Oncotree, etc.)
        
    Returns:
        dict: Interpretation results
    """
    print("\n" + "="*80)
    print(f"INTERPRETATION: {dataset_name} ({label_type})")
    print("="*80)
    
    # Thresholds for interpretation
    high_threshold = 0.5
    medium_threshold = 0.3
    
    interpretation = {
        'ari': ari,
        'nmi': nmi,
        'interpretation': {},
        'biological_significance': []
    }
    
    # Interpret based on metric levels
    if ari >= high_threshold and nmi >= high_threshold:
        interpretation['interpretation']['strength'] = 'HIGH'
        interpretation['interpretation']['meaning'] = 'Strong agreement between clusters and ground truth labels'
        
        if label_type == 'PAM50':
            interpretation['biological_significance'].append(
                '✅ Expression-driven BIGCLAM clusters captured true molecular patterns'
            )
            interpretation['biological_significance'].append(
                '✅ Gene expression profiles align well with PAM50 subtypes'
            )
        elif label_type == 'Oncotree':
            interpretation['biological_significance'].append(
                '✅ Clusters align with histological classification'
            )
        
    elif ari >= medium_threshold or nmi >= medium_threshold:
        interpretation['interpretation']['strength'] = 'MODERATE'
        interpretation['interpretation']['meaning'] = 'Moderate agreement, some clusters correspond to labels'
        
        if label_type == 'Oncotree':
            interpretation['biological_significance'].append(
                '⚠️  Histological labels partially reflect molecular community structure'
            )
        
    else:
        interpretation['interpretation']['strength'] = 'LOW'
        
        if label_type == 'Oncotree':
            interpretation['interpretation']['meaning'] = 'Histological labels do not reflect molecular community structure (expected)'
            interpretation['biological_significance'].append(
                '✅ Expected result: Oncotree is histological, not molecular'
            )
            interpretation['biological_significance'].append(
                '✅ Multiple PAM50 subtypes can share same Oncotree classification'
            )
        else:
            interpretation['interpretation']['meaning'] = 'Weak agreement between clusters and labels'
            interpretation['biological_significance'].append(
                '⚠️  Consider: Overlapping communities detected, samples may belong to multiple subtypes'
            )
    
    # Display interpretation
    print(f"\nMetrics:")
    print(f"  ARI: {ari:.4f}")
    print(f"  NMI: {nmi:.4f}")
    print(f"\nInterpretation:")
    print(f"  Strength: {interpretation['interpretation']['strength']}")
    print(f"  Meaning: {interpretation['interpretation']['meaning']}")
    print(f"\nBiological Significance:")
    for item in interpretation['biological_significance']:
        print(f"  {item}")
    
    return interpretation


def analyze_overlap(communities, target_labels, dataset_name):
    """
    Analyze overlapping communities and mixed subtypes.
    
    Args:
        communities: Cluster assignments (1D array or 2D membership matrix)
        target_labels: Ground truth labels
        dataset_name: Dataset name
        
    Returns:
        dict: Overlap analysis
    """
    print("\n" + "="*80)
    print(f"OVERLAP ANALYSIS: {dataset_name}")
    print("="*80)
    
    # Convert to numpy array and ensure 1D
    communities = np.array(communities)
    
    # Fix: If communities is 2D (membership matrix), convert to 1D (assignments)
    if communities.ndim == 2:
        print(f"[INFO] Converting 2D membership matrix to 1D community assignments...")
        communities = np.argmax(communities, axis=1)
    
    # Ensure 1D array
    communities = communities.flatten()
    
    # Get unique labels and clusters
    unique_labels = list(set(target_labels))
    unique_clusters = list(set(communities))
    
    # Analyze cluster composition
    print("\nCluster Composition:")
    overlap_results = {}
    
    for cluster_id in unique_clusters:
        cluster_mask = communities == cluster_id
        cluster_labels = np.array(target_labels)[cluster_mask]
        label_counts = pd.Series(cluster_labels).value_counts()
        
        # Calculate diversity
        num_unique_labels = len(label_counts)
        dominant_pct = label_counts.iloc[0] / len(cluster_labels) * 100
        
        overlap_results[cluster_id] = {
            'size': len(cluster_labels),
            'num_unique_labels': num_unique_labels,
            'dominant_label': label_counts.index[0],
            'dominant_pct': dominant_pct,
            'label_distribution': dict(label_counts)
        }
        
        print(f"\n  Cluster {cluster_id}: {len(cluster_labels)} samples")
        print(f"    Unique labels: {num_unique_labels}")
        print(f"    Dominant label: {label_counts.index[0]} ({dominant_pct:.1f}%)")
        
        if num_unique_labels > 1:
            print(f"    Label distribution: {dict(label_counts)}")
            print(f"    → Mixed subtype detected (biological overlap)")
    
    return overlap_results


def identify_border_samples(communities, target_labels, membership_matrix):
    """
    Identify samples on borders between subtypes (high membership in multiple communities).
    
    Args:
        communities: Hard cluster assignments (1D array or 2D membership matrix)
        target_labels: Ground truth labels
        membership_matrix: Membership strength matrix (if None, will extract from communities if 2D)
        
    Returns:
        dict: Border sample analysis
    """
    print("\n" + "="*80)
    print("BORDER SAMPLE ANALYSIS")
    print("="*80)
    
    # Convert to numpy arrays
    communities = np.array(communities)
    
    # Handle case where membership_matrix is None but communities is 2D
    if membership_matrix is None and communities.ndim == 2:
        membership_matrix = communities
        communities = np.argmax(communities, axis=1)
    elif membership_matrix is None:
        raise ValueError("membership_matrix is required when communities is 1D")
    
    membership_matrix = np.array(membership_matrix)
    
    # Ensure communities is 1D
    if communities.ndim > 1:
        communities = np.argmax(communities, axis=1) if communities.ndim == 2 else communities.flatten()
    else:
        communities = communities.flatten()
    
    # For each sample, find second-highest membership
    if membership_matrix.shape[1] < 2:
        # Need at least 2 communities for border analysis
        print("[WARNING] Need at least 2 communities for border analysis")
        return {
            'n_border_samples': 0,
            'border_samples': [],
            'membership_ratios': [],
            'border_labels': []
        }
    
    second_memberships = np.partition(membership_matrix, -2, axis=1)[:, -2]
    max_memberships = np.max(membership_matrix, axis=1)
    
    # Calculate membership ratio (second/max)
    membership_ratios = second_memberships / (max_memberships + 1e-10)
    
    # High ratio indicates border samples
    border_threshold = 0.3
    border_mask = membership_ratios > border_threshold
    
    n_border = border_mask.sum()
    n_total = len(communities)
    
    print(f"\nBorder samples (membership ratio > {border_threshold}):")
    print(f"  Total: {n_border}/{n_total} ({n_border/n_total*100:.1f}%)")
    
    if n_border > 0:
        border_labels = np.array(target_labels)[border_mask]
        label_dist = pd.Series(border_labels).value_counts()
        
        print(f"\nBorder sample label distribution:")
        for label, count in label_dist.items():
            print(f"  {label}: {count} ({count/len(border_labels)*100:.1f}%)")
        
        print(f"\nInterpretation:")
        print(f"  → These samples exhibit mixed membership")
        print(f"  → May represent transitional states between subtypes")
        print(f"  → Could indicate Luminal A ↔ Luminal B boundaries")
    
    return {
        'n_border_samples': int(n_border),
        'border_samples': np.where(border_mask)[0].tolist(),
        'membership_ratios': membership_ratios.tolist(),
        'border_labels': border_labels.tolist() if n_border > 0 else []
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Interpret clustering results')
    parser.add_argument('--clustering_dir', type=str, default='data/clusterings', help='Clustering directory')
    parser.add_argument('--targets_dir', type=str, default='data/processed', help='Targets directory')
    parser.add_argument('--evaluation_dir', type=str, default='results/evaluation', help='Evaluation results directory')
    
    args = parser.parse_args()
    
    # This would be called after evaluation to add interpretation
    print("Interpretation module - Use after running evaluators.py")

