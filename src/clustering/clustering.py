"""
Clustering Module using BIGCLAM

Applies BIGCLAM clustering to similarity graphs and assigns labels to samples.
"""

import numpy as np
from pathlib import Path
import pickle
import sys

from ..bigclam.bigclam_model import train_bigclam


def cluster_data(adjacency, max_communities=10, iterations=100, lr=0.08):
    """
    Apply BIGCLAM clustering to adjacency matrix.
    
    Args:
        adjacency: Adjacency matrix (sparse or dense)
        max_communities: Maximum number of communities to search
        iterations: Number of optimization iterations
        lr: Learning rate
        
    Returns:
        tuple: (communities, membership_matrix, optimal_num_communities)
    """
    print("\n[Clustering] Applying BIGCLAM...")
    print(f"    Max communities: {max_communities}")
    print(f"    Iterations: {iterations}")
    print(f"    Learning rate: {lr}")
    
    # Train BIGCLAM
    F, best_num_communities = train_bigclam(
        adjacency,
        max_communities=max_communities,
        iterations=iterations,
        lr=lr
    )
    
    # Get community assignments
    communities = np.argmax(F, axis=1)
    
    print(f"\n[Results]")
    print(f"    Optimal communities (AIC): {best_num_communities}")
    print(f"    Actual communities found: {len(set(communities))}")
    print(f"    Community sizes: {dict(zip(*np.unique(communities, return_counts=True)))}")
    
    return communities, F, best_num_communities


def save_clustering_results(communities, membership, optimal_k, output_file):
    """
    Save clustering results.
    
    Args:
        communities: Cluster assignments
        membership: Membership matrix
        optimal_k: Optimal number of communities
        output_file: Output file path
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save numpy arrays
    np.save(output_file.with_suffix('.npy'), communities)
    np.save(str(output_file).replace('.npy', '_membership.npy'), membership)
    
    # Save metadata
    metadata = {
        'communities': communities.tolist(),
        'optimal_num_communities': int(optimal_k),
        'num_communities_found': int(len(set(communities))),
        'n_samples': int(len(communities))
    }
    
    with open(str(output_file).replace('.npy', '_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"    Saved results to: {output_file}")


def load_clustering_results(input_file):
    """
    Load clustering results.
    
    Args:
        input_file: Input file path
        
    Returns:
        dict: Clustering results
    """
    input_file = Path(input_file)
    
    communities = np.load(input_file)
    membership = np.load(str(input_file).replace('.npy', '_membership.npy'))
    
    with open(str(input_file).replace('.npy', '_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    return {
        'communities': communities,
        'membership': membership,
        'metadata': metadata
    }


def cluster_all_graphs(input_dir='data/graphs', output_dir='data/clusterings',
                      max_communities=10, iterations=100, lr=0.08):
    """
    Apply clustering to all graphs.
    
    Args:
        input_dir: Directory containing graphs
        output_dir: Directory to save clusterings
        max_communities: Maximum communities to search
        iterations: Iterations per community number
        lr: Learning rate
        
    Returns:
        dict: Clustering results for each dataset
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Find all graph files
    graph_files = list(input_dir.glob('*_adjacency.npz')) + list(input_dir.glob('*_adjacency.npy'))
    
    if not graph_files:
        print(f"No graph files found in {input_dir}")
        return results
    
    for graph_file in graph_files:
        print("\n" + "="*80)
        dataset_name = graph_file.stem.replace('_adjacency', '')
        print(f"CLUSTERING: {dataset_name}")
        print("="*80)
        
        # Load graph
        if graph_file.suffix == '.npz':
            from scipy.sparse import load_npz
            adjacency = load_npz(graph_file)
        else:
            adjacency = np.load(graph_file)
        
        # Cluster
        communities, membership, optimal_k = cluster_data(
            adjacency,
            max_communities=max_communities,
            iterations=iterations,
            lr=lr
        )
        
        # Save results
        output_file = output_dir / f"{dataset_name}_communities"
        save_clustering_results(communities, membership, optimal_k, output_file)
        
        results[dataset_name] = {
            'communities': communities,
            'membership': membership,
            'optimal_k': optimal_k,
            'adjacency': adjacency
        }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply BIGCLAM clustering to graphs')
    parser.add_argument('--input_dir', type=str, default='data/graphs', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='data/clusterings', help='Output directory')
    parser.add_argument('--max_communities', type=int, default=10, help='Max communities')
    parser.add_argument('--iterations', type=int, default=100, help='Iterations per community')
    parser.add_argument('--lr', type=float, default=0.08, help='Learning rate')
    
    args = parser.parse_args()
    
    cluster_all_graphs(
        args.input_dir,
        args.output_dir,
        max_communities=args.max_communities,
        iterations=args.iterations,
        lr=args.lr
    )

