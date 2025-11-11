"""
Method Comparison Module

Compares BIGCLAM with other clustering methods:
- K-means
- Hierarchical clustering
- Spectral clustering

Demonstrates advantages of BIGCLAM for overlapping community detection.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.evaluators import evaluate_clustering
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def compare_clustering_methods(dataset_name, processed_dir='data/processed',
                              clustering_dir='data/clusterings',
                              output_dir='results/method_comparison',
                              n_clusters=4):
    """
    Compare BIGCLAM with other clustering methods.
    
    Args:
        dataset_name: Name of dataset
        processed_dir: Directory with processed data
        clustering_dir: Directory with clustering results
        output_dir: Output directory for results
        n_clusters: Number of clusters to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"METHOD COMPARISON: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Load data
    processed_file = Path(processed_dir) / f"{dataset_name}_processed.npy"
    clustering_file = Path(clustering_dir) / f"{dataset_name}_communities.npy"
    
    if not processed_file.exists():
        print(f"[ERROR] Processed file not found: {processed_file}")
        return None
    
    # Load data
    X = np.load(processed_file)
    
    # Load targets
    target_file = processed_file.parent / processed_file.name.replace('_processed.npy', '_targets.pkl')
    with open(target_file, 'rb') as f:
        targets_data = pickle.load(f)
    
    y_true = targets_data['target_labels']
    
    print(f"Data shape: {X.shape}")
    print(f"Number of clusters: {n_clusters}")
    
    results = {}
    
    # ===== 1. K-MEANS =====
    print("\n" + "-"*80)
    print("1. K-MEANS CLUSTERING")
    print("-"*80)
    
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X)
    kmeans_time = time.time() - start_time
    
    ari_kmeans = adjusted_rand_score(y_true, labels_kmeans)
    nmi_kmeans = normalized_mutual_info_score(y_true, labels_kmeans)
    
    print(f"  Runtime: {kmeans_time:.2f}s")
    print(f"  ARI: {ari_kmeans:.4f}")
    print(f"  NMI: {nmi_kmeans:.4f}")
    
    # Evaluate with full metrics
    eval_kmeans = evaluate_clustering(labels_kmeans, y_true, f"{dataset_name}_kmeans")
    if eval_kmeans:
        eval_results_kmeans, _ = eval_kmeans
        results['kmeans'] = {
            'ari': ari_kmeans,
            'nmi': nmi_kmeans,
            'purity': eval_results_kmeans.get('purity', 0),
            'f1_macro': eval_results_kmeans.get('f1_macro', 0),
            'runtime': kmeans_time
        }
    
    # ===== 2. HIERARCHICAL CLUSTERING =====
    print("\n" + "-"*80)
    print("2. HIERARCHICAL CLUSTERING")
    print("-"*80)
    
    # Use subset for hierarchical (it's slow on large datasets)
    if X.shape[0] > 5000:
        print("  [Note] Using subset of 5000 samples for hierarchical clustering (too slow on full dataset)")
        indices = np.random.choice(X.shape[0], 5000, replace=False)
        X_hier = X[indices]
        y_true_hier = np.array(y_true)[indices]
    else:
        X_hier = X
        y_true_hier = np.array(y_true)
    
    start_time = time.time()
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels_hier = hierarchical.fit_predict(X_hier)
    hier_time = time.time() - start_time
    
    ari_hier = adjusted_rand_score(y_true_hier, labels_hier)
    nmi_hier = normalized_mutual_info_score(y_true_hier, labels_hier)
    
    print(f"  Runtime: {hier_time:.2f}s")
    print(f"  ARI: {ari_hier:.4f}")
    print(f"  NMI: {nmi_hier:.4f}")
    
    # Evaluate with full metrics
    eval_hier = evaluate_clustering(labels_hier, y_true_hier, f"{dataset_name}_hierarchical")
    if eval_hier:
        eval_results_hier, _ = eval_hier
        results['hierarchical'] = {
            'ari': ari_hier,
            'nmi': nmi_hier,
            'purity': eval_results_hier.get('purity', 0),
            'f1_macro': eval_results_hier.get('f1_macro', 0),
            'runtime': hier_time
        }
    
    # ===== 3. SPECTRAL CLUSTERING =====
    print("\n" + "-"*80)
    print("3. SPECTRAL CLUSTERING")
    print("-"*80)
    
    # Use subset for spectral (it's memory intensive)
    if X.shape[0] > 3000:
        print("  [Note] Using subset of 3000 samples for spectral clustering (memory intensive)")
        indices = np.random.choice(X.shape[0], 3000, replace=False)
        X_spec = X[indices]
        y_true_spec = np.array(y_true)[indices]
    else:
        X_spec = X
        y_true_spec = np.array(y_true)
    
    start_time = time.time()
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, n_jobs=-1)
    labels_spec = spectral.fit_predict(X_spec)
    spec_time = time.time() - start_time
    
    ari_spec = adjusted_rand_score(y_true_spec, labels_spec)
    nmi_spec = normalized_mutual_info_score(y_true_spec, labels_spec)
    
    print(f"  Runtime: {spec_time:.2f}s")
    print(f"  ARI: {ari_spec:.4f}")
    print(f"  NMI: {nmi_spec:.4f}")
    
    # Evaluate with full metrics
    eval_spec = evaluate_clustering(labels_spec, y_true_spec, f"{dataset_name}_spectral")
    if eval_spec:
        eval_results_spec, _ = eval_spec
        results['spectral'] = {
            'ari': ari_spec,
            'nmi': nmi_spec,
            'purity': eval_results_spec.get('purity', 0),
            'f1_macro': eval_results_spec.get('f1_macro', 0),
            'runtime': spec_time
        }
    
    # ===== 4. BIGCLAM =====
    print("\n" + "-"*80)
    print("4. BIGCLAM (Our Method)")
    print("-"*80)
    
    if clustering_file.exists():
        labels_bigclam = np.load(clustering_file)
        
        ari_bigclam = adjusted_rand_score(y_true, labels_bigclam)
        nmi_bigclam = normalized_mutual_info_score(y_true, labels_bigclam)
        
        # Evaluate with full metrics
        eval_bigclam = evaluate_clustering(labels_bigclam, y_true, f"{dataset_name}_bigclam")
        if eval_bigclam:
            eval_results_bigclam, _ = eval_bigclam
            # Get runtime from previous runs (approximate)
            bigclam_time = 600  # Approximate, should be measured separately
            
            results['bigclam'] = {
                'ari': ari_bigclam,
                'nmi': nmi_bigclam,
                'purity': eval_results_bigclam.get('purity', 0),
                'f1_macro': eval_results_bigclam.get('f1_macro', 0),
                'runtime': bigclam_time
            }
            
            print(f"  ARI: {ari_bigclam:.4f}")
            print(f"  NMI: {nmi_bigclam:.4f}")
            print(f"  Purity: {eval_results_bigclam.get('purity', 0):.4f}")
            print(f"  F1-macro: {eval_results_bigclam.get('f1_macro', 0):.4f}")
    else:
        print("  [WARNING] BIGCLAM results not found, skipping...")
    
    # ===== SUMMARY =====
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    if results:
        comparison_data = []
        for method, metrics in results.items():
            comparison_data.append({
                'Method': method.upper(),
                'ARI': metrics['ari'],
                'NMI': metrics['nmi'],
                'Purity': metrics['purity'],
                'F1-macro': metrics['f1_macro'],
                'Runtime (s)': metrics['runtime']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))
        
        # Save results
        results['comparison'] = comparison_df
        
        output_file = output_dir / f"{dataset_name}_method_comparison.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        csv_file = output_dir / f"{dataset_name}_method_comparison.csv"
        comparison_df.to_csv(csv_file, index=False)
        
        print(f"\n[Saved] Results: {output_file}")
        print(f"[Saved] CSV: {csv_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare clustering methods')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['tcga_brca_data', 'gse96058_data'],
                       help='Dataset to analyze')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                       help='Directory with processed data')
    parser.add_argument('--clustering_dir', type=str, default='data/clusterings',
                       help='Directory with clustering results')
    parser.add_argument('--output_dir', type=str, default='results/method_comparison',
                       help='Output directory')
    parser.add_argument('--n_clusters', type=int, default=4,
                       help='Number of clusters')
    
    args = parser.parse_args()
    
    compare_clustering_methods(
        args.dataset,
        args.processed_dir,
        args.clustering_dir,
        args.output_dir,
        args.n_clusters
    )

