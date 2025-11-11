"""
Baseline Comparison Module

Compares classification performance on:
1. Original data (no BIGCLAM filtering)
2. BIGCLAM-filtered data (using discovered clusters as features)

This validation analysis demonstrates whether BIGCLAM actually improves performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.classifiers.mlp_classifier import MLP, train_mlp
from src.classifiers.classifiers import split_data, encode_labels


def load_original_data(processed_file):
    """Load original preprocessed data (before BIGCLAM)."""
    data = np.load(processed_file)
    
    # Load targets
    target_file = processed_file.parent / processed_file.name.replace('_processed.npy', '_targets.pkl')
    with open(target_file, 'rb') as f:
        targets_data = pickle.load(f)
    
    return data, targets_data['target_labels']


def load_bigclam_data(clustering_file, processed_file):
    """Load BIGCLAM cluster assignments as features."""
    communities = np.load(clustering_file)
    
    # Convert to one-hot encoding (each cluster becomes a feature)
    n_samples = len(communities)
    n_clusters = len(set(communities))
    
    # Create one-hot encoding
    cluster_features = np.zeros((n_samples, n_clusters))
    for i, cluster in enumerate(communities):
        cluster_features[i, cluster] = 1.0
    
    # Also load original features for combined approach
    original_data, _ = load_original_data(processed_file)
    
    # Combine: original features + cluster membership
    combined_features = np.hstack([original_data, cluster_features])
    
    # Load targets
    target_file = processed_file.parent / processed_file.name.replace('_processed.npy', '_targets.pkl')
    with open(target_file, 'rb') as f:
        targets_data = pickle.load(f)
    
    return {
        'cluster_only': cluster_features,
        'original': original_data,
        'combined': combined_features
    }, targets_data['target_labels']


def compare_baselines(dataset_name, processed_dir='data/processed', 
                     clustering_dir='data/clusterings',
                     output_dir='results/baseline_comparison'):
    """
    Compare classification performance across different feature sets.
    
    Args:
        dataset_name: Name of dataset
        processed_dir: Directory with processed data
        clustering_dir: Directory with clustering results
        output_dir: Output directory for results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"BASELINE COMPARISON: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Load data
    processed_file = Path(processed_dir) / f"{dataset_name}_processed.npy"
    clustering_file = Path(clustering_dir) / f"{dataset_name}_communities.npy"
    
    if not processed_file.exists():
        print(f"[ERROR] Processed file not found: {processed_file}")
        return None
    
    if not clustering_file.exists():
        print(f"[ERROR] Clustering file not found: {clustering_file}")
        return None
    
    # Load original data
    print("[1/3] Loading original data...")
    X_original, y = load_original_data(processed_file)
    print(f"    Original data shape: {X_original.shape}")
    
    # Load BIGCLAM data
    print("[2/3] Loading BIGCLAM data...")
    bigclam_data, y_bigclam = load_bigclam_data(clustering_file, processed_file)
    print(f"    Cluster-only features: {bigclam_data['cluster_only'].shape}")
    print(f"    Combined features: {bigclam_data['combined'].shape}")
    
    # Encode labels
    y_encoded, label_encoder = encode_labels(y)
    
    # Split data (same splits for all comparisons)
    X_train_orig, X_valid_orig, X_test_orig, y_train, y_valid, y_test = split_data(
        X_original, y_encoded, test_size=0.2, valid_size=0.2, random_state=42
    )
    
    X_train_cluster, X_valid_cluster, X_test_cluster, _, _, _ = split_data(
        bigclam_data['cluster_only'], y_encoded, test_size=0.2, valid_size=0.2, random_state=42
    )
    
    X_train_combined, X_valid_combined, X_test_combined, _, _, _ = split_data(
        bigclam_data['combined'], y_encoded, test_size=0.2, valid_size=0.2, random_state=42
    )
    
    results = {}
    
    # 1. Original data (no BIGCLAM)
    print("\n[3/3] Training classifiers...")
    print("\n" + "-"*80)
    print("BASELINE 1: Original Data (No BIGCLAM)")
    print("-"*80)
    
    # SVM on original
    print("\n  Training SVM on original data...")
    start_time = time.time()
    svm_orig = SVC(kernel='rbf', C=0.1, gamma='scale', probability=True, random_state=42)
    svm_orig.fit(X_train_orig, y_train)
    svm_orig_time = time.time() - start_time
    
    y_pred_svm_orig = svm_orig.predict(X_test_orig)
    acc_svm_orig = accuracy_score(y_test, y_pred_svm_orig)
    
    print(f"    Accuracy: {acc_svm_orig:.4f}")
    print(f"    Training time: {svm_orig_time:.2f}s")
    
    # MLP on original
    print("\n  Training MLP on original data...")
    start_time = time.time()
    mlp_orig_results = train_mlp(
        X_train_orig, y_train, X_valid_orig, y_valid, X_test_orig, y_test,
        hidden_layers=[80, 50, 20],
        learning_rate=0.001,
        num_epochs=200,
        patience=10
    )
    mlp_orig_time = time.time() - start_time
    
    print(f"    Accuracy: {mlp_orig_results['test_accuracy']:.4f}")
    print(f"    Training time: {mlp_orig_time:.2f}s")
    
    results['original'] = {
        'svm': {'accuracy': acc_svm_orig, 'time': svm_orig_time},
        'mlp': {'accuracy': mlp_orig_results['test_accuracy'], 'time': mlp_orig_time}
    }
    
    # 2. Cluster-only features (BIGCLAM clusters as features)
    print("\n" + "-"*80)
    print("BASELINE 2: BIGCLAM Cluster Features Only")
    print("-"*80)
    
    # SVM on cluster features
    print("\n  Training SVM on cluster features...")
    start_time = time.time()
    svm_cluster = SVC(kernel='rbf', C=0.1, gamma='scale', probability=True, random_state=42)
    svm_cluster.fit(X_train_cluster, y_train)
    svm_cluster_time = time.time() - start_time
    
    y_pred_svm_cluster = svm_cluster.predict(X_test_cluster)
    acc_svm_cluster = accuracy_score(y_test, y_pred_svm_cluster)
    
    print(f"    Accuracy: {acc_svm_cluster:.4f}")
    print(f"    Training time: {svm_cluster_time:.2f}s")
    
    # MLP on cluster features
    print("\n  Training MLP on cluster features...")
    start_time = time.time()
    mlp_cluster_results = train_mlp(
        X_train_cluster, y_train, X_valid_cluster, y_valid, X_test_cluster, y_test,
        hidden_layers=[80, 50, 20],
        learning_rate=0.001,
        num_epochs=200,
        patience=10
    )
    mlp_cluster_time = time.time() - start_time
    
    print(f"    Accuracy: {mlp_cluster_results['test_accuracy']:.4f}")
    print(f"    Training time: {mlp_cluster_time:.2f}s")
    
    results['cluster_only'] = {
        'svm': {'accuracy': acc_svm_cluster, 'time': svm_cluster_time},
        'mlp': {'accuracy': mlp_cluster_results['test_accuracy'], 'time': mlp_cluster_time}
    }
    
    # 3. Combined features (original + cluster membership)
    print("\n" + "-"*80)
    print("BASELINE 3: Combined Features (Original + BIGCLAM Clusters)")
    print("-"*80)
    
    # SVM on combined
    print("\n  Training SVM on combined features...")
    start_time = time.time()
    svm_combined = SVC(kernel='rbf', C=0.1, gamma='scale', probability=True, random_state=42)
    svm_combined.fit(X_train_combined, y_train)
    svm_combined_time = time.time() - start_time
    
    y_pred_svm_combined = svm_combined.predict(X_test_combined)
    acc_svm_combined = accuracy_score(y_test, y_pred_svm_combined)
    
    print(f"    Accuracy: {acc_svm_combined:.4f}")
    print(f"    Training time: {svm_combined_time:.2f}s")
    
    # MLP on combined
    print("\n  Training MLP on combined features...")
    start_time = time.time()
    mlp_combined_results = train_mlp(
        X_train_combined, y_train, X_valid_combined, y_valid, X_test_combined, y_test,
        hidden_layers=[80, 50, 20],
        learning_rate=0.001,
        num_epochs=200,
        patience=10
    )
    mlp_combined_time = time.time() - start_time
    
    print(f"    Accuracy: {mlp_combined_results['test_accuracy']:.4f}")
    print(f"    Training time: {mlp_combined_time:.2f}s")
    
    results['combined'] = {
        'svm': {'accuracy': acc_svm_combined, 'time': svm_combined_time},
        'mlp': {'accuracy': mlp_combined_results['test_accuracy'], 'time': mlp_combined_time}
    }
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Baseline Comparison Results")
    print("="*80)
    
    summary_df = pd.DataFrame({
        'Feature Set': ['Original', 'Cluster Only', 'Combined'],
        'SVM Accuracy': [
            results['original']['svm']['accuracy'],
            results['cluster_only']['svm']['accuracy'],
            results['combined']['svm']['accuracy']
        ],
        'MLP Accuracy': [
            results['original']['mlp']['accuracy'],
            results['cluster_only']['mlp']['accuracy'],
            results['combined']['mlp']['accuracy']
        ],
        'SVM Time (s)': [
            results['original']['svm']['time'],
            results['cluster_only']['svm']['time'],
            results['combined']['svm']['time']
        ],
        'MLP Time (s)': [
            results['original']['mlp']['time'],
            results['cluster_only']['mlp']['time'],
            results['combined']['mlp']['time']
        ]
    })
    
    print("\n" + summary_df.to_string(index=False))
    
    # Calculate improvements
    print("\n" + "-"*80)
    print("IMPROVEMENT ANALYSIS")
    print("-"*80)
    
    svm_improvement = ((results['combined']['svm']['accuracy'] - results['original']['svm']['accuracy']) / 
                      results['original']['svm']['accuracy']) * 100
    mlp_improvement = ((results['combined']['mlp']['accuracy'] - results['original']['mlp']['accuracy']) / 
                      results['original']['mlp']['accuracy']) * 100
    
    print(f"\nSVM Improvement (Combined vs Original): {svm_improvement:+.2f}%")
    print(f"MLP Improvement (Combined vs Original): {mlp_improvement:+.2f}%")
    
    # Save results
    results['summary'] = summary_df
    results['improvements'] = {
        'svm': svm_improvement,
        'mlp': mlp_improvement
    }
    
    output_file = output_dir / f"{dataset_name}_baseline_comparison.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    summary_file = output_dir / f"{dataset_name}_baseline_comparison.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n[Saved] Results: {output_file}")
    print(f"[Saved] Summary: {summary_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare baseline methods vs BIGCLAM')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['tcga_brca_data', 'gse96058_data'],
                       help='Dataset to analyze')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                       help='Directory with processed data')
    parser.add_argument('--clustering_dir', type=str, default='data/clusterings',
                       help='Directory with clustering results')
    parser.add_argument('--output_dir', type=str, default='results/baseline_comparison',
                       help='Output directory')
    
    args = parser.parse_args()
    
    compare_baselines(
        args.dataset,
        args.processed_dir,
        args.clustering_dir,
        args.output_dir
    )

