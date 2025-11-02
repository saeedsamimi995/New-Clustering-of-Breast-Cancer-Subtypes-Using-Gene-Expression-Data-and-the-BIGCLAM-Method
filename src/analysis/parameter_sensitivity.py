"""
Parameter Sensitivity Analysis

Tests different parameter values and generates visualization plots showing:
- Variance threshold sensitivity
- Similarity threshold sensitivity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


def analyze_variance_threshold_sensitivity(data, threshold_range=None, output_dir='results/sensitivity'):
    """
    Analyze sensitivity to variance threshold parameter.
    
    Args:
        data: numpy array of shape (n_samples, n_features) - preprocessed expression data
        threshold_range: list of threshold values to test (default: 5-20)
        output_dir: Directory to save results
        
    Returns:
        dict: Analysis results
    """
    if threshold_range is None:
        threshold_range = list(range(5, 21))  # 5 to 20
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("VARIANCE THRESHOLD SENSITIVITY ANALYSIS")
    print("="*80)
    
    results = {
        'thresholds': [],
        'features_kept': [],
        'features_removed': [],
        'retention_rate': []
    }
    
    # Calculate feature variances
    feature_variances = np.var(data, axis=0)
    
    for threshold in threshold_range:
        var_threshold = VarianceThreshold(threshold=threshold)
        var_threshold.fit(data)
        selected_features = var_threshold.get_support()
        
        n_kept = selected_features.sum()
        n_total = len(selected_features)
        n_removed = n_total - n_kept
        retention_rate = n_kept / n_total * 100
        
        results['thresholds'].append(threshold)
        results['features_kept'].append(n_kept)
        results['features_removed'].append(n_removed)
        results['retention_rate'].append(retention_rate)
        
        print(f"Threshold {threshold:2d}: Kept {n_kept:5,}/{n_total:,} features ({retention_rate:5.1f}%)")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Features retained vs threshold
    axes[0, 0].plot(results['thresholds'], results['features_kept'], 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Variance Threshold', fontsize=12)
    axes[0, 0].set_ylabel('Features Retained', fontsize=12)
    axes[0, 0].set_title('Features Retained vs Variance Threshold', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=13, color='r', linestyle='--', label='Default (13)')
    axes[0, 0].legend()
    
    # Plot 2: Retention rate vs threshold
    axes[0, 1].plot(results['thresholds'], results['retention_rate'], 's-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Variance Threshold', fontsize=12)
    axes[0, 1].set_ylabel('Retention Rate (%)', fontsize=12)
    axes[0, 1].set_title('Feature Retention Rate vs Threshold', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=13, color='r', linestyle='--', label='Default (13)')
    axes[0, 1].legend()
    
    # Plot 3: Features removed vs threshold
    axes[1, 0].plot(results['thresholds'], results['features_removed'], '^-', linewidth=2, markersize=8, color='red')
    axes[1, 0].set_xlabel('Variance Threshold', fontsize=12)
    axes[1, 0].set_ylabel('Features Removed', fontsize=12)
    axes[1, 0].set_title('Features Removed vs Variance Threshold', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=13, color='r', linestyle='--', label='Default (13)')
    axes[1, 0].legend()
    
    # Plot 4: Variance distribution histogram
    axes[1, 1].hist(feature_variances, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=np.mean(feature_variances), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(feature_variances):.2f}')
    axes[1, 1].axvline(x=13, color='r', linestyle='--', linewidth=2, label='Default (13)')
    axes[1, 1].set_xlabel('Feature Variance', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Distribution of Feature Variances', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / 'variance_threshold_sensitivity.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[Saved] Variance threshold sensitivity plot: {output_file}")
    plt.close()
    
    # Find recommended threshold
    recommended = _recommend_variance_threshold(results, feature_variances)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'variance_threshold_sensitivity.csv', index=False)
    
    # Save recommendations
    with open(output_dir / 'variance_threshold_recommendations.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("VARIANCE THRESHOLD RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Recommended Threshold: {recommended['threshold']}\n")
        f.write(f"Reason: {recommended['reason']}\n\n")
        f.write("Analysis:\n")
        f.write(f"- Features retained: {recommended['features_kept']:,}\n")
        f.write(f"- Retention rate: {recommended['retention_rate']:.1f}%\n")
        f.write(f"- This threshold balances feature retention with noise removal\n\n")
        f.write("Alternative thresholds:\n")
        for alt in recommended.get('alternatives', []):
            f.write(f"  - {alt['threshold']}: {alt['reason']}\n")
    
    print("\n" + "="*80)
    print("VARIANCE THRESHOLD RECOMMENDATION")
    print("="*80)
    print(f"Recommended: {recommended['threshold']}")
    print(f"Reason: {recommended['reason']}")
    print(f"Features retained: {recommended['features_kept']:,} ({recommended['retention_rate']:.1f}%)")
    
    return results, recommended


def analyze_similarity_threshold_sensitivity(expression_data, threshold_range=None, output_dir='results/sensitivity'):
    """
    Analyze sensitivity to similarity threshold parameter.
    
    Args:
        expression_data: numpy array of shape (n_samples, n_features) - preprocessed expression data
        threshold_range: list of threshold values to test (default: 0.1 to 0.9, step 0.1)
        output_dir: Directory to save results
        
    Returns:
        dict: Analysis results
    """
    if threshold_range is None:
        threshold_range = [round(0.1 + i*0.1, 1) for i in range(9)]  # 0.1 to 0.9
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("SIMILARITY THRESHOLD SENSITIVITY ANALYSIS")
    print("="*80)
    
    N = expression_data.shape[0]
    print(f"Computing similarity matrix for {N} samples...")
    
    # Compute similarity matrix (this may take time for large datasets)
    if N > 5000:
        print("  Using chunked computation...")
        similarity_matrix = _compute_similarity_chunked(expression_data)
    else:
        print("  Computing full similarity matrix...")
        similarity_matrix = cosine_similarity(expression_data)
    
    # Calculate similarity distribution
    # Extract upper triangle (excluding diagonal) to avoid duplicates
    triu_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarity_values = similarity_matrix[triu_indices]
    
    results = {
        'thresholds': [],
        'n_edges': [],
        'graph_density': [],
        'avg_degree': [],
        'n_connected_components': []
    }
    
    for threshold in threshold_range:
        # Create adjacency matrix
        adjacency = (similarity_matrix > threshold).astype(int)
        np.fill_diagonal(adjacency, 0)  # Remove self-loops
        
        # Calculate graph statistics
        n_edges = (adjacency > 0).sum() // 2  # Divide by 2 for undirected graph
        graph_density = (n_edges * 2) / (N * (N - 1)) * 100  # Percentage
        
        # Average degree
        degrees = adjacency.sum(axis=1)
        avg_degree = degrees.mean()
        
        # Count connected components (simplified - approximate)
        # For large graphs, use sparse representation
        if N > 1000:
            adj_sparse = csr_matrix(adjacency)
            n_components = _count_connected_components_sparse(adj_sparse)
        else:
            n_components = _count_connected_components_dense(adjacency)
        
        results['thresholds'].append(threshold)
        results['n_edges'].append(n_edges)
        results['graph_density'].append(graph_density)
        results['avg_degree'].append(avg_degree)
        results['n_connected_components'].append(n_components)
        
        print(f"Threshold {threshold:.1f}: {n_edges:8,} edges ({graph_density:5.2f}% density), "
              f"avg degree: {avg_degree:.1f}, components: {n_components}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Number of edges vs threshold
    axes[0, 0].plot(results['thresholds'], results['n_edges'], 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Similarity Threshold', fontsize=12)
    axes[0, 0].set_ylabel('Number of Edges', fontsize=12)
    axes[0, 0].set_title('Graph Edges vs Similarity Threshold', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=0.4, color='r', linestyle='--', label='Default (0.4)')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Graph density vs threshold
    axes[0, 1].plot(results['thresholds'], results['graph_density'], 's-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Similarity Threshold', fontsize=12)
    axes[0, 1].set_ylabel('Graph Density (%)', fontsize=12)
    axes[0, 1].set_title('Graph Density vs Similarity Threshold', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=0.4, color='r', linestyle='--', label='Default (0.4)')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Average degree vs threshold
    axes[1, 0].plot(results['thresholds'], results['avg_degree'], '^-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Similarity Threshold', fontsize=12)
    axes[1, 0].set_ylabel('Average Node Degree', fontsize=12)
    axes[1, 0].set_title('Average Degree vs Similarity Threshold', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=0.4, color='r', linestyle='--', label='Default (0.4)')
    axes[1, 0].legend()
    
    # Plot 4: Similarity distribution histogram
    axes[1, 1].hist(similarity_values, bins=100, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=np.mean(similarity_values), color='g', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(similarity_values):.3f}')
    axes[1, 1].axvline(x=0.4, color='r', linestyle='--', linewidth=2, label='Default (0.4)')
    axes[1, 1].set_xlabel('Cosine Similarity', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Distribution of Pairwise Cosine Similarities', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / 'similarity_threshold_sensitivity.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[Saved] Similarity threshold sensitivity plot: {output_file}")
    plt.close()
    
    # Find recommended threshold
    recommended = _recommend_similarity_threshold(results, similarity_values)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'similarity_threshold_sensitivity.csv', index=False)
    
    # Save recommendations
    with open(output_dir / 'similarity_threshold_recommendations.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("SIMILARITY THRESHOLD RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Recommended Threshold: {recommended['threshold']}\n")
        f.write(f"Reason: {recommended['reason']}\n\n")
        f.write("Analysis:\n")
        f.write(f"- Graph density: {recommended['density']:.2f}%\n")
        f.write(f"- Average degree: {recommended['avg_degree']:.1f}\n")
        f.write(f"- Number of edges: {recommended['n_edges']:,}\n")
        f.write(f"- Connected components: {recommended['n_components']}\n\n")
        f.write("Guidelines:\n")
        f.write("- Good threshold: 0.3-0.5 (balanced connectivity)\n")
        f.write("- Too low (<0.3): Dense graph, many weak connections\n")
        f.write("- Too high (>0.6): Sparse graph, disconnected components\n\n")
        f.write("Alternative thresholds:\n")
        for alt in recommended.get('alternatives', []):
            f.write(f"  - {alt['threshold']}: {alt['reason']}\n")
    
    print("\n" + "="*80)
    print("SIMILARITY THRESHOLD RECOMMENDATION")
    print("="*80)
    print(f"Recommended: {recommended['threshold']}")
    print(f"Reason: {recommended['reason']}")
    print(f"Graph density: {recommended['density']:.2f}%")
    print(f"Average degree: {recommended['avg_degree']:.1f}")
    print(f"Connected components: {recommended['n_components']}")
    
    return results, recommended


def _compute_similarity_chunked(features, chunk_size=5000):
    """Compute cosine similarity in chunks for memory efficiency."""
    N = features.shape[0]
    similarity_matrix = np.zeros((N, N), dtype=np.float32)
    
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        for j in range(i, N, chunk_size):
            end_j = min(j + chunk_size, N)
            similarity_matrix[i:end_i, j:end_j] = cosine_similarity(
                features[i:end_i], 
                features[j:end_j]
            )
            if i != j:
                similarity_matrix[j:end_j, i:end_i] = similarity_matrix[i:end_i, j:end_j].T
    
    return similarity_matrix


def _count_connected_components_dense(adjacency):
    """Count connected components in dense adjacency matrix."""
    from scipy.sparse.csgraph import connected_components
    n_components, _ = connected_components(csgraph=adjacency, directed=False, return_labels=False)
    return n_components


def _count_connected_components_sparse(adj_sparse):
    """Count connected components in sparse adjacency matrix."""
    from scipy.sparse.csgraph import connected_components
    n_components, _ = connected_components(csgraph=adj_sparse, directed=False, return_labels=False)
    return n_components


def _recommend_variance_threshold(results, feature_variances):
    """
    Recommend optimal variance threshold based on analysis results.
    
    Criteria:
    - Retain 20-40% of features (balance between signal and noise)
    - Avoid removing too many features (<10% retention = too aggressive)
    - Avoid keeping too many features (>60% retention = too conservative)
    """
    df = pd.DataFrame(results)
    
    # Find threshold that retains 20-40% of features
    ideal_retention_range = (20, 40)
    
    candidates = []
    for idx, row in df.iterrows():
        retention = row['retention_rate']
        if ideal_retention_range[0] <= retention <= ideal_retention_range[1]:
            candidates.append({
                'threshold': row['thresholds'],
                'retention_rate': retention,
                'features_kept': row['features_kept'],
                'score': abs(retention - 30)  # Prefer close to 30%
            })
    
    if candidates:
        # Best candidate is closest to 30% retention
        best = min(candidates, key=lambda x: x['score'])
        threshold = best['threshold']
    else:
        # If no candidate in ideal range, find closest
        df['dist_from_30'] = abs(df['retention_rate'] - 30)
        best_idx = df['dist_from_30'].idxmin()
        threshold = df.loc[best_idx, 'thresholds']
        best = {
            'retention_rate': df.loc[best_idx, 'retention_rate'],
            'features_kept': int(df.loc[best_idx, 'features_kept'])
        }
    
    # Get values for recommended threshold
    row = df[df['thresholds'] == threshold].iloc[0]
    
    # Determine reason
    retention = row['retention_rate']
    if 20 <= retention <= 40:
        reason = f"Optimal retention rate ({retention:.1f}%) - balances signal preservation with noise removal"
    elif retention < 10:
        reason = f"Very aggressive filtering ({retention:.1f}% retention) - may remove important features"
    elif retention > 60:
        reason = f"Conservative filtering ({retention:.1f}% retention) - many low-variance features retained"
    else:
        reason = f"Balanced filtering ({retention:.1f}% retention) - acceptable compromise"
    
    # Find alternatives
    alternatives = []
    if threshold > min(df['thresholds']):
        lower_thresh = threshold - 2
        if lower_thresh in df['thresholds'].values:
            lower_row = df[df['thresholds'] == lower_thresh].iloc[0]
            alternatives.append({
                'threshold': lower_thresh,
                'reason': f"More conservative ({lower_row['retention_rate']:.1f}% retention)"
            })
    
    if threshold < max(df['thresholds']):
        higher_thresh = threshold + 2
        if higher_thresh in df['thresholds'].values:
            higher_row = df[df['thresholds'] == higher_thresh].iloc[0]
            alternatives.append({
                'threshold': higher_thresh,
                'reason': f"More aggressive ({higher_row['retention_rate']:.1f}% retention)"
            })
    
    return {
        'threshold': int(threshold),
        'retention_rate': retention,
        'features_kept': int(row['features_kept']),
        'reason': reason,
        'alternatives': alternatives
    }


def _recommend_similarity_threshold(results, similarity_values):
    """
    Recommend optimal similarity threshold based on analysis results.
    
    Criteria:
    - Graph density: 0.1-5% (not too sparse, not too dense)
    - Connected components: Should be 1 (fully connected graph)
    - Average degree: At least 2-5 connections per node
    - Density sweet spot: Around 1-2% is ideal for clustering
    """
    df = pd.DataFrame(results)
    
    candidates = []
    for idx, row in df.iterrows():
        density = row['graph_density']
        n_components = row['n_connected_components']
        avg_degree = row['avg_degree']
        
        # Score based on multiple criteria
        score = 0
        
        # Ideal density: 0.5-3%
        if 0.5 <= density <= 3.0:
            score += 3
        elif 0.1 <= density <= 5.0:
            score += 2
        elif 0.05 <= density <= 10.0:
            score += 1
        
        # Prefer fully connected (1 component)
        if n_components == 1:
            score += 3
        elif n_components <= 3:
            score += 2
        elif n_components <= 10:
            score += 1
        
        # Good average degree: 2-10
        if 2 <= avg_degree <= 10:
            score += 2
        elif 1 <= avg_degree <= 20:
            score += 1
        
        candidates.append({
            'threshold': row['thresholds'],
            'density': density,
            'n_components': n_components,
            'avg_degree': avg_degree,
            'n_edges': row['n_edges'],
            'score': score
        })
    
    # Find best candidate
    best = max(candidates, key=lambda x: x['score'])
    threshold = best['threshold']
    
    # Determine reason
    density = best['density']
    n_components = best['n_components']
    avg_degree = best['avg_degree']
    
    reasons_parts = []
    if 0.5 <= density <= 3.0:
        reasons_parts.append(f"ideal graph density ({density:.2f}%)")
    elif 0.1 <= density <= 5.0:
        reasons_parts.append(f"good graph density ({density:.2f}%)")
    else:
        reasons_parts.append(f"acceptable graph density ({density:.2f}%)")
    
    if n_components == 1:
        reasons_parts.append("fully connected graph")
    elif n_components <= 5:
        reasons_parts.append(f"mostly connected ({n_components} components)")
    else:
        reasons_parts.append(f"fragmented graph ({n_components} components)")
    
    if 2 <= avg_degree <= 10:
        reasons_parts.append(f"good connectivity (avg degree: {avg_degree:.1f})")
    
    reason = " - ".join(reasons_parts) if reasons_parts else "Balanced graph properties"
    
    # Find alternatives
    alternatives = []
    # More conservative (lower threshold, denser graph)
    if threshold > min([c['threshold'] for c in candidates]):
        lower_thresh = round(threshold - 0.1, 1)
        lower_cand = next((c for c in candidates if abs(c['threshold'] - lower_thresh) < 0.05), None)
        if lower_cand:
            alternatives.append({
                'threshold': lower_thresh,
                'reason': f"Denser graph ({lower_cand['density']:.2f}% density, {lower_cand['avg_degree']:.1f} avg degree)"
            })
    
    # More aggressive (higher threshold, sparser graph)
    if threshold < max([c['threshold'] for c in candidates]):
        higher_thresh = round(threshold + 0.1, 1)
        higher_cand = next((c for c in candidates if abs(c['threshold'] - higher_thresh) < 0.05), None)
        if higher_cand:
            alternatives.append({
                'threshold': higher_thresh,
                'reason': f"Sparser graph ({higher_cand['density']:.2f}% density, {higher_cand['avg_degree']:.1f} avg degree)"
            })
    
    return {
        'threshold': threshold,
        'density': density,
        'n_components': int(n_components),
        'avg_degree': avg_degree,
        'n_edges': int(best['n_edges']),
        'reason': reason,
        'alternatives': alternatives
    }


def run_sensitivity_analysis(processed_data_path=None, 
                            variance_threshold=True, 
                            similarity_threshold=True,
                            output_dir='results/sensitivity'):
    """
    Run complete sensitivity analysis.
    
    Args:
        processed_data_path: Path to processed data file (*_processed.npy)
        variance_threshold: Whether to analyze variance threshold sensitivity
        similarity_threshold: Whether to analyze similarity threshold sensitivity
        output_dir: Output directory
    """
    if processed_data_path is None:
        # Try to find processed data files
        processed_dir = Path('data/processed')
        processed_files = list(processed_dir.glob('*_processed.npy'))
        if not processed_files:
            print("Error: No processed data files found.")
            print("Please run preprocessing first or specify --data path")
            return
        processed_data_path = processed_files[0]
        print(f"Using processed data: {processed_data_path}")
    
    # Load processed data
    expression_data = np.load(processed_data_path)
    print(f"\nLoaded data: {expression_data.shape}")
    
    # Run analyses
    variance_recommended = None
    similarity_recommended = None
    
    if variance_threshold:
        variance_results, variance_recommended = analyze_variance_threshold_sensitivity(
            expression_data, 
            output_dir=output_dir
        )
    
    if similarity_threshold:
        similarity_results, similarity_recommended = analyze_similarity_threshold_sensitivity(
            expression_data,
            output_dir=output_dir
        )
    
    # Print summary
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    
    if variance_recommended:
        print(f"\n📊 Variance Threshold: Use {variance_recommended['threshold']} in config.yml")
        print(f"   Reason: {variance_recommended['reason']}")
    
    if similarity_recommended:
        print(f"\n📊 Similarity Threshold: Use {similarity_recommended['threshold']} in config.yml")
        print(f"   Reason: {similarity_recommended['reason']}")
    
    print("\n" + "="*80)
    print("HOW TO INTERPRET RESULTS")
    print("="*80)
    print("\n1. Check the visualization plots:")
    print("   - Look for the 'knee' or 'elbow' in the curves")
    print("   - Find thresholds where the rate of change slows down")
    print("   - Red dashed line shows current default value")
    print("\n2. Read recommendation files:")
    print("   - results/sensitivity/variance_threshold_recommendations.txt")
    print("   - results/sensitivity/similarity_threshold_recommendations.txt")
    print("\n3. Review CSV files for exact numbers:")
    print("   - results/sensitivity/variance_threshold_sensitivity.csv")
    print("   - results/sensitivity/similarity_threshold_sensitivity.csv")
    print("\n4. Update config.yml with recommended values")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Parameter sensitivity analysis')
    parser.add_argument('--data', type=str, default=None, 
                       help='Path to processed data file (*_processed.npy)')
    parser.add_argument('--variance_threshold', action='store_true', default=True,
                       help='Analyze variance threshold sensitivity')
    parser.add_argument('--similarity_threshold', action='store_true', default=True,
                       help='Analyze similarity threshold sensitivity')
    parser.add_argument('--no_variance', action='store_false', dest='variance_threshold',
                       help='Skip variance threshold analysis')
    parser.add_argument('--no_similarity', action='store_false', dest='similarity_threshold',
                       help='Skip similarity threshold analysis')
    parser.add_argument('--output_dir', type=str, default='results/sensitivity',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    run_sensitivity_analysis(
        processed_data_path=args.data,
        variance_threshold=args.variance_threshold,
        similarity_threshold=args.similarity_threshold,
        output_dir=args.output_dir
    )

