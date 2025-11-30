"""
Comprehensive Method Comparison Module

Compares BIGCLAM with other clustering methods:
- K-means (centroid-based)
- Spectral clustering (graph-based)
- NMF (matrix factorization)
- HDBSCAN (density-based)
- Leiden/Louvain (graph-based)

Metrics:
- Silhouette Score
- Davies-Bouldin Index
- NMI vs PAM50
- ARI vs PAM50
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import NMF
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Optional dependencies
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("[Warning] hdbscan not available. Install with: pip install hdbscan")

try:
    import leidenalg
    import igraph as ig
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    print("[Warning] leidenalg/igraph not available. Install with: pip install leidenalg python-igraph")

try:
    import networkx as nx
    from networkx.algorithms import community
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("[Warning] networkx not available for Louvain. Install with: pip install networkx")


def load_data_for_comparison(dataset_name, processed_dir='data/processed', 
                            clustering_dir='data/clusterings'):
    """
    Load data and ground truth labels for comparison.
    
    Args:
        dataset_name: Dataset name ('tcga' or 'gse96058')
        processed_dir: Directory with processed data
        clustering_dir: Directory with BIGCLAM results
    
    Returns:
        tuple: (X, y_true, bigclam_labels, n_clusters)
    """
    # Map dataset names to file prefixes
    file_prefix_map = {
        'tcga': 'tcga_brca_data',
        'gse96058': 'gse96058_data'
    }
    
    prefix = file_prefix_map.get(dataset_name, dataset_name)
    
    # Load processed data
    processed_file = Path(processed_dir) / f"{prefix}_processed.npy"
    if not processed_file.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_file}")
    
    X = np.load(processed_file)
    print(f"  Loaded processed data: {X.shape}")
    
    # Load targets
    target_file = Path(processed_dir) / f"{prefix}_targets.pkl"
    if not target_file.exists():
        raise FileNotFoundError(f"Target file not found: {target_file}")
    
    with open(target_file, 'rb') as f:
        targets_data = pickle.load(f)
    
    y_true = targets_data.get('target_labels', None)
    if y_true is None:
        raise ValueError("No target_labels found in targets file")
    
    # Filter out 'Unknown' labels
    valid_mask = np.array([lbl != 'Unknown' and pd.notna(lbl) for lbl in y_true])
    X = X[valid_mask]
    y_true = np.array(y_true)[valid_mask]
    
    print(f"  Valid samples: {len(y_true)}")
    print(f"  Unique labels: {len(set(y_true))}")
    
    # Load BIGCLAM labels
    bigclam_file = Path(clustering_dir) / f"{prefix}_communities.npy"
    bigclam_labels = None
    if bigclam_file.exists():
        bigclam_labels = np.load(bigclam_file)
        if bigclam_labels.ndim > 1:
            bigclam_labels = np.argmax(bigclam_labels, axis=1)
        bigclam_labels = bigclam_labels.flatten().astype(int)
        bigclam_labels = bigclam_labels[valid_mask]
        print(f"  BIGCLAM clusters: {len(set(bigclam_labels))}")
    else:
        print(f"  [Warning] BIGCLAM file not found: {bigclam_file}")
    
    # Determine number of clusters from ground truth
    n_clusters = len(set(y_true))
    print(f"  Number of clusters (from PAM50): {n_clusters}")
    
    return X, y_true, bigclam_labels, n_clusters


def cluster_kmeans(X, n_clusters, random_state=42):
    """K-means clustering."""
    print("  Running K-means...")
    start_time = time.time()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X)
    
    runtime = time.time() - start_time
    print(f"    Runtime: {runtime:.2f}s")
    
    return labels, runtime


def cluster_spectral(X, n_clusters, random_state=42):
    """Spectral clustering."""
    print("  Running Spectral clustering...")
    start_time = time.time()
    
    # Use subset if too large (spectral is memory intensive)
    if X.shape[0] > 5000:
        print(f"    [Note] Using subset of 5000 samples (original: {X.shape[0]})")
        indices = np.random.choice(X.shape[0], 5000, replace=False)
        X_subset = X[indices]
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=random_state, n_jobs=-1)
        labels_subset = spectral.fit_predict(X_subset)
        
        # Map back to full dataset using nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_subset)
        _, nearest = nn.kneighbors(X)
        labels = labels_subset[nearest.flatten()]
    else:
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=random_state, n_jobs=-1)
        labels = spectral.fit_predict(X)
    
    runtime = time.time() - start_time
    print(f"    Runtime: {runtime:.2f}s")
    
    return labels, runtime


def cluster_nmf(X, n_clusters, random_state=42):
    """Non-negative Matrix Factorization clustering."""
    print("  Running NMF...")
    start_time = time.time()
    
    # NMF requires non-negative data
    # Scale to [0, 1] range
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Add small constant to ensure non-negative
    X_scaled = X_scaled + 1e-10
    
    # Run NMF
    nmf = NMF(n_components=n_clusters, random_state=random_state, max_iter=500)
    W = nmf.fit_transform(X_scaled)
    
    # Assign to cluster with highest weight
    labels = np.argmax(W, axis=1)
    
    runtime = time.time() - start_time
    print(f"    Runtime: {runtime:.2f}s")
    
    return labels, runtime


def cluster_hdbscan(X, min_cluster_size=None):
    """HDBSCAN density-based clustering."""
    if not HDBSCAN_AVAILABLE:
        print("  [Skip] HDBSCAN not available")
        return None, None
    
    print("  Running HDBSCAN...")
    start_time = time.time()
    
    if min_cluster_size is None:
        min_cluster_size = max(10, int(X.shape[0] * 0.01))  # 1% of samples, min 10
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5)
    labels = clusterer.fit_predict(X)
    
    # HDBSCAN can assign -1 for noise points, remap to positive integers
    unique_labels = np.unique(labels)
    if -1 in unique_labels:
        # Remap noise points to largest cluster ID + 1
        max_label = labels.max()
        labels[labels == -1] = max_label + 1
    
    runtime = time.time() - start_time
    print(f"    Runtime: {runtime:.2f}s")
    print(f"    Number of clusters found: {len(set(labels))}")
    
    return labels, runtime


def cluster_leiden(X, n_clusters=None, resolution=1.0):
    """Leiden algorithm for graph clustering."""
    if not LEIDEN_AVAILABLE:
        print("  [Skip] Leiden not available")
        return None, None
    
    print("  Running Leiden...")
    start_time = time.time()
    
    # Build k-nearest neighbor graph
    from sklearn.neighbors import kneighbors_graph
    k = min(15, X.shape[0] - 1)
    knn_graph = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)
    
    # Convert to igraph
    edges = np.array(knn_graph.nonzero()).T
    g = ig.Graph(edges=edges, directed=False)
    
    # Run Leiden
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, 
                                        resolution_parameter=resolution, seed=42)
    labels = np.array(partition.membership)
    
    runtime = time.time() - start_time
    print(f"    Runtime: {runtime:.2f}s")
    print(f"    Number of clusters found: {len(set(labels))}")
    
    return labels, runtime


def cluster_louvain(X, resolution=1.0):
    """Louvain algorithm for graph clustering."""
    if not LOUVAIN_AVAILABLE:
        print("  [Skip] Louvain not available")
        return None, None
    
    print("  Running Louvain...")
    start_time = time.time()
    
    # Build k-nearest neighbor graph
    from sklearn.neighbors import kneighbors_graph
    k = min(15, X.shape[0] - 1)
    knn_graph = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)
    
    # Convert to NetworkX
    G = nx.from_scipy_sparse_array(knn_graph)
    
    # Run Louvain
    communities = community.louvain_communities(G, resolution=resolution, seed=42)
    
    # Convert to labels
    labels = np.zeros(X.shape[0], dtype=int)
    for i, comm in enumerate(communities):
        for node in comm:
            labels[node] = i
    
    runtime = time.time() - start_time
    print(f"    Runtime: {runtime:.2f}s")
    print(f"    Number of clusters found: {len(set(labels))}")
    
    return labels, runtime


def calculate_metrics(X, labels, y_true):
    """
    Calculate all comparison metrics.
    
    Args:
        X: Feature matrix
        labels: Cluster assignments
        y_true: Ground truth labels (PAM50)
    
    Returns:
        dict: Metrics
    """
    if labels is None:
        return None
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_true)
    
    # Ensure labels are integers
    labels = labels.astype(int)
    
    # Remove any invalid labels
    valid_mask = (labels >= 0) & (labels < len(labels))
    if not valid_mask.all():
        print(f"    [Warning] {np.sum(~valid_mask)} invalid labels found, removing")
        labels = labels[valid_mask]
        X = X[valid_mask]
        y_encoded = y_encoded[valid_mask]
    
    metrics = {}
    
    # Silhouette Score
    try:
        # Use subset if too large (silhouette is O(n^2))
        if X.shape[0] > 5000:
            indices = np.random.choice(X.shape[0], 5000, replace=False)
            silhouette = silhouette_score(X[indices], labels[indices])
        else:
            silhouette = silhouette_score(X, labels)
        metrics['silhouette'] = silhouette
    except Exception as e:
        print(f"    [Warning] Silhouette score failed: {e}")
        metrics['silhouette'] = np.nan
    
    # Davies-Bouldin Index
    try:
        db_index = davies_bouldin_score(X, labels)
        metrics['davies_bouldin'] = db_index
    except Exception as e:
        print(f"    [Warning] Davies-Bouldin failed: {e}")
        metrics['davies_bouldin'] = np.nan
    
    # NMI vs PAM50
    try:
        nmi = normalized_mutual_info_score(y_encoded, labels)
        metrics['nmi'] = nmi
    except Exception as e:
        print(f"    [Warning] NMI failed: {e}")
        metrics['nmi'] = np.nan
    
    # ARI vs PAM50
    try:
        ari = adjusted_rand_score(y_encoded, labels)
        metrics['ari'] = ari
    except Exception as e:
        print(f"    [Warning] ARI failed: {e}")
        metrics['ari'] = np.nan
    
    return metrics


def compare_all_methods(dataset_name, processed_dir='data/processed',
                       clustering_dir='data/clusterings',
                       output_dir='results/method_comparison',
                       use_pca=True, n_components=50):
    """
    Compare BIGCLAM with all other clustering methods.
    
    Args:
        dataset_name: Dataset name ('tcga' or 'gse96058')
        processed_dir: Directory with processed data
        clustering_dir: Directory with BIGCLAM results
        output_dir: Output directory
        use_pca: Whether to use PCA for dimensionality reduction (for large datasets)
        n_components: Number of PCA components if use_pca=True
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"COMPREHENSIVE METHOD COMPARISON: {dataset_name.upper()}")
    print("="*80)
    
    # Load data
    print("\n[Loading] Data and ground truth...")
    X, y_true, bigclam_labels, n_clusters = load_data_for_comparison(
        dataset_name, processed_dir, clustering_dir
    )
    
    # Apply PCA if needed (for large datasets or high-dimensional data)
    if use_pca and X.shape[1] > 1000:
        print(f"\n[PCA] Reducing dimensions from {X.shape[1]} to {n_components}...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, random_state=42)
        X = pca.fit_transform(X)
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Standardize data (important for some methods)
    print("\n[Preprocessing] Standardizing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # ===== 1. K-MEANS =====
    print("\n" + "-"*80)
    print("1. K-MEANS (Centroid-based)")
    print("-"*80)
    labels_kmeans, runtime = cluster_kmeans(X_scaled, n_clusters)
    metrics_kmeans = calculate_metrics(X_scaled, labels_kmeans, y_true)
    if metrics_kmeans:
        metrics_kmeans['runtime'] = runtime
        metrics_kmeans['n_clusters'] = len(set(labels_kmeans))
        results['K-means'] = metrics_kmeans
        print(f"    Silhouette: {metrics_kmeans['silhouette']:.4f}")
        print(f"    Davies-Bouldin: {metrics_kmeans['davies_bouldin']:.4f}")
        print(f"    NMI vs PAM50: {metrics_kmeans['nmi']:.4f}")
        print(f"    ARI vs PAM50: {metrics_kmeans['ari']:.4f}")
    
    # ===== 2. SPECTRAL =====
    print("\n" + "-"*80)
    print("2. SPECTRAL CLUSTERING (Graph-based)")
    print("-"*80)
    labels_spectral, runtime = cluster_spectral(X_scaled, n_clusters)
    metrics_spectral = calculate_metrics(X_scaled, labels_spectral, y_true)
    if metrics_spectral:
        metrics_spectral['runtime'] = runtime
        metrics_spectral['n_clusters'] = len(set(labels_spectral))
        results['Spectral'] = metrics_spectral
        print(f"    Silhouette: {metrics_spectral['silhouette']:.4f}")
        print(f"    Davies-Bouldin: {metrics_spectral['davies_bouldin']:.4f}")
        print(f"    NMI vs PAM50: {metrics_spectral['nmi']:.4f}")
        print(f"    ARI vs PAM50: {metrics_spectral['ari']:.4f}")
    
    # ===== 3. NMF =====
    print("\n" + "-"*80)
    print("3. NMF (Matrix Factorization)")
    print("-"*80)
    labels_nmf, runtime = cluster_nmf(X_scaled, n_clusters)
    metrics_nmf = calculate_metrics(X_scaled, labels_nmf, y_true)
    if metrics_nmf:
        metrics_nmf['runtime'] = runtime
        metrics_nmf['n_clusters'] = len(set(labels_nmf))
        results['NMF'] = metrics_nmf
        print(f"    Silhouette: {metrics_nmf['silhouette']:.4f}")
        print(f"    Davies-Bouldin: {metrics_nmf['davies_bouldin']:.4f}")
        print(f"    NMI vs PAM50: {metrics_nmf['nmi']:.4f}")
        print(f"    ARI vs PAM50: {metrics_nmf['ari']:.4f}")
    
    # ===== 4. HDBSCAN =====
    print("\n" + "-"*80)
    print("4. HDBSCAN (Density-based)")
    print("-"*80)
    labels_hdbscan, runtime = cluster_hdbscan(X_scaled)
    if labels_hdbscan is not None:
        metrics_hdbscan = calculate_metrics(X_scaled, labels_hdbscan, y_true)
        if metrics_hdbscan:
            metrics_hdbscan['runtime'] = runtime
            metrics_hdbscan['n_clusters'] = len(set(labels_hdbscan))
            results['HDBSCAN'] = metrics_hdbscan
            print(f"    Silhouette: {metrics_hdbscan['silhouette']:.4f}")
            print(f"    Davies-Bouldin: {metrics_hdbscan['davies_bouldin']:.4f}")
            print(f"    NMI vs PAM50: {metrics_hdbscan['nmi']:.4f}")
            print(f"    ARI vs PAM50: {metrics_hdbscan['ari']:.4f}")
    
    # ===== 5. LEIDEN =====
    print("\n" + "-"*80)
    print("5. LEIDEN (Graph-based)")
    print("-"*80)
    labels_leiden, runtime = cluster_leiden(X_scaled)
    if labels_leiden is not None:
        metrics_leiden = calculate_metrics(X_scaled, labels_leiden, y_true)
        if metrics_leiden:
            metrics_leiden['runtime'] = runtime
            metrics_leiden['n_clusters'] = len(set(labels_leiden))
            results['Leiden'] = metrics_leiden
            print(f"    Silhouette: {metrics_leiden['silhouette']:.4f}")
            print(f"    Davies-Bouldin: {metrics_leiden['davies_bouldin']:.4f}")
            print(f"    NMI vs PAM50: {metrics_leiden['nmi']:.4f}")
            print(f"    ARI vs PAM50: {metrics_leiden['ari']:.4f}")
    
    # ===== 6. LOUVAIN =====
    print("\n" + "-"*80)
    print("6. LOUVAIN (Graph-based)")
    print("-"*80)
    labels_louvain, runtime = cluster_louvain(X_scaled)
    if labels_louvain is not None:
        metrics_louvain = calculate_metrics(X_scaled, labels_louvain, y_true)
        if metrics_louvain:
            metrics_louvain['runtime'] = runtime
            metrics_louvain['n_clusters'] = len(set(labels_louvain))
            results['Louvain'] = metrics_louvain
            print(f"    Silhouette: {metrics_louvain['silhouette']:.4f}")
            print(f"    Davies-Bouldin: {metrics_louvain['davies_bouldin']:.4f}")
            print(f"    NMI vs PAM50: {metrics_louvain['nmi']:.4f}")
            print(f"    ARI vs PAM50: {metrics_louvain['ari']:.4f}")
    
    # ===== 7. BIGCLAM =====
    print("\n" + "-"*80)
    print("7. BIGCLAM (Our Method)")
    print("-"*80)
    if bigclam_labels is not None:
        metrics_bigclam = calculate_metrics(X_scaled, bigclam_labels, y_true)
        if metrics_bigclam:
            metrics_bigclam['runtime'] = np.nan  # Runtime not tracked separately
            metrics_bigclam['n_clusters'] = len(set(bigclam_labels))
            results['BIGCLAM'] = metrics_bigclam
            print(f"    Silhouette: {metrics_bigclam['silhouette']:.4f}")
            print(f"    Davies-Bouldin: {metrics_bigclam['davies_bouldin']:.4f}")
            print(f"    NMI vs PAM50: {metrics_bigclam['nmi']:.4f}")
            print(f"    ARI vs PAM50: {metrics_bigclam['ari']:.4f}")
    else:
        print("  [Skip] BIGCLAM labels not available")
    
    # ===== SUMMARY =====
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    if results:
        # Create comparison DataFrame
        comparison_data = []
        for method, metrics in results.items():
            comparison_data.append({
                'Method': method,
                'Type': _get_method_type(method),
                'Silhouette': metrics.get('silhouette', np.nan),
                'Davies-Bouldin': metrics.get('davies_bouldin', np.nan),
                'NMI vs PAM50': metrics.get('nmi', np.nan),
                'ARI vs PAM50': metrics.get('ari', np.nan),
                'N_Clusters': metrics.get('n_clusters', np.nan),
                'Runtime (s)': metrics.get('runtime', np.nan)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))
        
        # Save results
        output_file = output_dir / f"{dataset_name}_method_comparison.csv"
        comparison_df.to_csv(output_file, index=False)
        print(f"\n[Saved] Comparison results: {output_file}")
        
        # Save detailed results
        results_file = output_dir / f"{dataset_name}_method_comparison.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump({
                'comparison_df': comparison_df,
                'detailed_results': results,
                'dataset_name': dataset_name,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_clusters_ground_truth': n_clusters
            }, f)
        print(f"[Saved] Detailed results: {results_file}")
        
        # Create visualizations
        try:
            create_comparison_figures(comparison_df, dataset_name, output_dir)
        except Exception as e:
            print(f"[Warning] Failed to create figures: {e}")
            import traceback
            traceback.print_exc()
        
        return comparison_df, results
    else:
        print("[Error] No results to compare")
        return None, None


def _get_method_type(method):
    """Get method type for categorization."""
    type_map = {
        'K-means': 'Centroid',
        'Spectral': 'Graph',
        'NMF': 'Matrix Factorization',
        'HDBSCAN': 'Density',
        'Leiden': 'Graph',
        'Louvain': 'Graph',
        'BIGCLAM': 'Graph'
    }
    return type_map.get(method, 'Unknown')


def create_comparison_figures(comparison_df, dataset_name, output_dir):
    """Create visualization figures for method comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[Visualization] Creating comparison figures...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # Filter out methods with NaN values for plotting
    plot_df = comparison_df.dropna(subset=['Silhouette', 'NMI vs PAM50', 'ARI vs PAM50'])
    
    if len(plot_df) == 0:
        print("  [Warning] No valid data for plotting")
        return
    
    # ===== FIGURE 1: Metrics Comparison Bar Plot =====
    print("  Creating metrics comparison bar plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['Silhouette', 'Davies-Bouldin', 'NMI vs PAM50', 'ARI vs PAM50']
    metric_titles = ['Silhouette Score', 'Davies-Bouldin Index', 'NMI vs PAM50', 'ARI vs PAM50']
    
    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Sort by metric value (descending, except for Davies-Bouldin which is lower is better)
        if metric == 'Davies-Bouldin':
            plot_df_sorted = plot_df.sort_values(metric, ascending=True)
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(plot_df_sorted)))
        else:
            plot_df_sorted = plot_df.sort_values(metric, ascending=False)
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(plot_df_sorted)))
        
        bars = ax.barh(range(len(plot_df_sorted)), plot_df_sorted[metric], color=colors, alpha=0.7)
        ax.set_yticks(range(len(plot_df_sorted)))
        ax.set_yticklabels(plot_df_sorted['Method'], fontsize=9)
        ax.set_xlabel(metric, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Highlight BIGCLAM
        if 'BIGCLAM' in plot_df_sorted['Method'].values:
            bigclam_idx = plot_df_sorted['Method'].tolist().index('BIGCLAM')
            bars[bigclam_idx].set_edgecolor('red')
            bars[bigclam_idx].set_linewidth(2)
    
    plt.suptitle(f'{dataset_name.upper()}: Clustering Method Comparison', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    fig_file = output_dir / f"{dataset_name}_method_comparison_metrics.png"
    plt.savefig(fig_file, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [Saved] {fig_file.name}")
    
    # ===== FIGURE 2: Comprehensive Summary Figure =====
    print("  Creating comprehensive summary figure...")
    
    fig = plt.figure(figsize=(16, 10))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Silhouette Score
    ax1 = fig.add_subplot(gs[0, 0])
    plot_df_sorted = plot_df.sort_values('Silhouette', ascending=False)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(plot_df_sorted)))
    bars = ax1.bar(range(len(plot_df_sorted)), plot_df_sorted['Silhouette'], color=colors, alpha=0.7)
    if 'BIGCLAM' in plot_df_sorted['Method'].values:
        bigclam_idx = plot_df_sorted['Method'].tolist().index('BIGCLAM')
        bars[bigclam_idx].set_edgecolor('red')
        bars[bigclam_idx].set_linewidth(2)
    ax1.set_xticks(range(len(plot_df_sorted)))
    ax1.set_xticklabels(plot_df_sorted['Method'], rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Silhouette Score', fontsize=10)
    ax1.set_title('Silhouette Score', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Davies-Bouldin Index
    ax2 = fig.add_subplot(gs[0, 1])
    plot_df_sorted = plot_df.sort_values('Davies-Bouldin', ascending=True)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(plot_df_sorted)))
    bars = ax2.bar(range(len(plot_df_sorted)), plot_df_sorted['Davies-Bouldin'], color=colors, alpha=0.7)
    if 'BIGCLAM' in plot_df_sorted['Method'].values:
        bigclam_idx = plot_df_sorted['Method'].tolist().index('BIGCLAM')
        bars[bigclam_idx].set_edgecolor('red')
        bars[bigclam_idx].set_linewidth(2)
    ax2.set_xticks(range(len(plot_df_sorted)))
    ax2.set_xticklabels(plot_df_sorted['Method'], rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Davies-Bouldin Index', fontsize=10)
    ax2.set_title('Davies-Bouldin Index (lower is better)', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. NMI vs PAM50
    ax3 = fig.add_subplot(gs[0, 2])
    plot_df_sorted = plot_df.sort_values('NMI vs PAM50', ascending=False)
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(plot_df_sorted)))
    bars = ax3.bar(range(len(plot_df_sorted)), plot_df_sorted['NMI vs PAM50'], color=colors, alpha=0.7)
    if 'BIGCLAM' in plot_df_sorted['Method'].values:
        bigclam_idx = plot_df_sorted['Method'].tolist().index('BIGCLAM')
        bars[bigclam_idx].set_edgecolor('red')
        bars[bigclam_idx].set_linewidth(2)
    ax3.set_xticks(range(len(plot_df_sorted)))
    ax3.set_xticklabels(plot_df_sorted['Method'], rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('NMI', fontsize=10)
    ax3.set_title('NMI vs PAM50', fontsize=11, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. ARI vs PAM50
    ax4 = fig.add_subplot(gs[1, 0])
    plot_df_sorted = plot_df.sort_values('ARI vs PAM50', ascending=False)
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(plot_df_sorted)))
    bars = ax4.bar(range(len(plot_df_sorted)), plot_df_sorted['ARI vs PAM50'], color=colors, alpha=0.7)
    if 'BIGCLAM' in plot_df_sorted['Method'].values:
        bigclam_idx = plot_df_sorted['Method'].tolist().index('BIGCLAM')
        bars[bigclam_idx].set_edgecolor('red')
        bars[bigclam_idx].set_linewidth(2)
    ax4.set_xticks(range(len(plot_df_sorted)))
    ax4.set_xticklabels(plot_df_sorted['Method'], rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('ARI', fontsize=10)
    ax4.set_title('ARI vs PAM50', fontsize=11, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Method types summary
    ax5 = fig.add_subplot(gs[1, 1])
    type_counts = plot_df['Type'].value_counts()
    ax5.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
    ax5.set_title('Method Types Distribution', fontsize=11, fontweight='bold')
    
    # 6. Summary statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    best_silhouette = plot_df.loc[plot_df['Silhouette'].idxmax(), 'Method']
    best_db = plot_df.loc[plot_df['Davies-Bouldin'].idxmin(), 'Method']
    best_nmi = plot_df.loc[plot_df['NMI vs PAM50'].idxmax(), 'Method']
    best_ari = plot_df.loc[plot_df['ARI vs PAM50'].idxmax(), 'Method']
    
    summary_text = f"""
    Dataset: {dataset_name.upper()}
    Methods compared: {len(plot_df)}
    
    Best Methods:
    Silhouette: {best_silhouette}
    Davies-Bouldin: {best_db}
    NMI vs PAM50: {best_nmi}
    ARI vs PAM50: {best_ari}
    
    BIGCLAM Performance:
    Silhouette: {plot_df[plot_df['Method']=='BIGCLAM']['Silhouette'].values[0]:.4f}
    NMI: {plot_df[plot_df['Method']=='BIGCLAM']['NMI vs PAM50'].values[0]:.4f}
    ARI: {plot_df[plot_df['Method']=='BIGCLAM']['ARI vs PAM50'].values[0]:.4f}
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.set_title('Summary Statistics', fontsize=11, fontweight='bold')
    
    plt.suptitle(f'{dataset_name.upper()}: Comprehensive Method Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    
    summary_file = output_dir / f"{dataset_name}_method_comparison_summary.png"
    plt.savefig(summary_file, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [Saved] {summary_file.name}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive clustering method comparison')
    parser.add_argument('--dataset', type=str, default='both', choices=['tcga', 'gse96058', 'both'],
                       help='Dataset to analyze (default: both)')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                       help='Directory with processed data')
    parser.add_argument('--clustering-dir', type=str, default='data/clusterings',
                       help='Directory with BIGCLAM results')
    parser.add_argument('--output-dir', type=str, default='results/method_comparison',
                       help='Output directory')
    parser.add_argument('--no-pca', action='store_true',
                       help='Disable PCA dimensionality reduction')
    
    args = parser.parse_args()
    
    datasets = ['tcga', 'gse96058'] if args.dataset == 'both' else [args.dataset]
    
    for dataset in datasets:
        try:
            compare_all_methods(
                dataset,
                processed_dir=args.processed_dir,
                clustering_dir=args.clustering_dir,
                output_dir=args.output_dir,
                use_pca=not args.no_pca
            )
        except Exception as e:
            print(f"\n[Error] Failed to compare methods for {dataset}: {e}")
            import traceback
            traceback.print_exc()

