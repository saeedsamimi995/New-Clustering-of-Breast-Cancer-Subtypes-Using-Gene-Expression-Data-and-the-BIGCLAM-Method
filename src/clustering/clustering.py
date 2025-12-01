"""
Clustering Module using BIGCLAM

Applies BIGCLAM clustering to similarity graphs and assigns labels to samples.
"""

import numpy as np
from pathlib import Path
import pickle
import time
import psutil
import os
import json
import sys

# Handle imports for both module and direct execution
try:
    from ..bigclam.bigclam_model import train_bigclam
except ImportError:
    # Running as script directly, use absolute import
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.bigclam.bigclam_model import train_bigclam


def cluster_data(adjacency, max_communities=10, min_communities=1, iterations=100, lr=0.08, criterion='BIC',
                 adaptive_lr=True, adaptive_iterations=True, early_stopping=True,
                 convergence_threshold=1e-6, patience=10, num_restarts=1):
    """
    Apply BIGCLAM clustering to adjacency matrix.
    
    Args:
        adjacency: Adjacency matrix (sparse or dense)
        max_communities: Maximum number of communities to search
        min_communities: Minimum number of communities to search (default: 1). 
                        Set to 5 or higher for finer-grained subtyping than PAM50.
        iterations: Base number of optimization iterations
        lr: Base learning rate
        criterion: Model selection criterion ('AIC' or 'BIC')
        adaptive_lr: Automatically adjust learning rate based on graph size
        adaptive_iterations: Automatically adjust iterations based on graph size and community number
        early_stopping: Enable early stopping when convergence detected
        convergence_threshold: Loss change threshold for convergence
        patience: Number of iterations without improvement before stopping
        num_restarts: Number of random restarts per community number
        
    Returns:
        tuple: (communities, membership_matrix, optimal_num_communities, runtime_info)
    """
    print("\n[Clustering] Applying BIGCLAM...")
    print(f"    Min communities: {min_communities}")
    print(f"    Max communities: {max_communities}")
    print(f"    Base iterations: {iterations}")
    print(f"    Base learning rate: {lr}")
    print(f"    Model selection: {criterion}")
    
    # Measure runtime and memory
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()
    
    # Train BIGCLAM
    F, best_num_communities = train_bigclam(
        adjacency,
        max_communities=max_communities,
        min_communities=min_communities,
        iterations=iterations,
        lr=lr,
        criterion=criterion,
        adaptive_lr=adaptive_lr,
        adaptive_iterations=adaptive_iterations,
        early_stopping=early_stopping,
        convergence_threshold=convergence_threshold,
        patience=patience,
        num_restarts=num_restarts
    )
    
    # Measure runtime and memory after
    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    runtime_seconds = end_time - start_time
    memory_used_mb = mem_after - mem_before
    
    # Get number of CPU cores used
    n_cores = os.cpu_count()
    
    print(f"\n[Performance]")
    print(f"    Runtime: {runtime_seconds:.2f} seconds ({runtime_seconds/60:.2f} minutes)")
    print(f"    Memory used: {memory_used_mb:.2f} MB ({memory_used_mb/1024:.2f} GB)")
    print(f"    Peak memory: {peak_memory:.2f} MB ({peak_memory/1024:.2f} GB)")
    print(f"    CPU cores available: {n_cores}")
    
    # Get community assignments
    communities = np.argmax(F, axis=1)
    
    print(f"\n[Results]")
    print(f"    Optimal communities ({criterion}): {best_num_communities}")
    print(f"    Actual communities found: {len(set(communities))}")
    print(f"    Community sizes: {dict(zip(*np.unique(communities, return_counts=True)))}")
    
    # Runtime info
    runtime_info = {
        'runtime_seconds': runtime_seconds,
        'runtime_minutes': runtime_seconds / 60,
        'memory_used_mb': memory_used_mb,
        'memory_used_gb': memory_used_mb / 1024,
        'peak_memory_mb': peak_memory,
        'peak_memory_gb': peak_memory / 1024,
        'n_samples': int(adjacency.shape[0]),
        'n_communities': int(best_num_communities),
        'n_cores': n_cores,
        'max_communities': max_communities,
        'iterations': iterations,
        'criterion': criterion
    }
    
    return communities, F, best_num_communities, runtime_info


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
    
    # Ensure communities is 1D (flatten if needed)
    if communities.ndim > 1:
        print(f"[WARNING] Communities is {communities.ndim}D, converting to 1D using argmax...")
        communities = np.argmax(communities, axis=1)
    communities = communities.flatten()
    
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
                      max_communities=10, min_communities=1, iterations=100, lr=0.08, criterion_dict=None,
                      adaptive_lr=True, adaptive_iterations=True, early_stopping=True,
                      convergence_threshold=1e-6, patience=10, num_restarts_dict=None,
                      dataset_specific_config=None):
    """
    Apply clustering to all graphs.
    
    Args:
        input_dir: Directory containing graphs
        output_dir: Directory to save clusterings
        max_communities: Maximum communities to search
        iterations: Iterations per community number
        lr: Base learning rate
        criterion_dict: Dictionary mapping dataset names to criterion ('AIC' or 'BIC')
                       If None, uses 'BIC' for all datasets
        adaptive_lr: Automatically adjust learning rate based on graph size
        adaptive_iterations: Automatically adjust iterations based on graph size
        early_stopping: Enable early stopping when convergence detected
        convergence_threshold: Loss change threshold for convergence
        patience: Number of iterations without improvement before stopping
        num_restarts_dict: Dictionary mapping dataset names to num_restarts
                          If None, uses num_restarts from dataset_specific_config or default
        dataset_specific_config: Dictionary with dataset-specific overrides
                                Format: {dataset_name: {iterations, learning_rate, num_restarts}}
        
    Returns:
        dict: Clustering results for each dataset
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if criterion_dict is None:
        criterion_dict = {}
    if num_restarts_dict is None:
        num_restarts_dict = {}
    if dataset_specific_config is None:
        dataset_specific_config = {}
    
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
        
        # Determine parameters for this dataset (with dataset-specific overrides)
        criterion = criterion_dict.get(dataset_name, criterion_dict.get('default', 'BIC'))
        
        # Get dataset-specific overrides
        dataset_config = dataset_specific_config.get(dataset_name, {})
        dataset_iterations = dataset_config.get('iterations', iterations)
        dataset_lr = dataset_config.get('learning_rate', lr)
        dataset_num_restarts = num_restarts_dict.get(dataset_name, dataset_config.get('num_restarts', 1))
        
        # Load graph
        if graph_file.suffix == '.npz':
            from scipy.sparse import load_npz
            adjacency = load_npz(graph_file)
        else:
            adjacency = np.load(graph_file)
        
        # Get min/max communities from dataset config or use defaults
        dataset_min_communities = dataset_config.get('min_communities', min_communities)
        dataset_max_communities = dataset_config.get('max_communities', max_communities)
        
        # Cluster
        communities, membership, optimal_k, runtime_info = cluster_data(
            adjacency,
            max_communities=dataset_max_communities,
            min_communities=dataset_min_communities,
            iterations=dataset_iterations,
            lr=dataset_lr,
            criterion=criterion,
            adaptive_lr=adaptive_lr,
            adaptive_iterations=adaptive_iterations,
            early_stopping=early_stopping,
            convergence_threshold=convergence_threshold,
            patience=patience,
            num_restarts=dataset_num_restarts
        )
        
        # Save results
        output_file = output_dir / f"{dataset_name}_communities"
        save_clustering_results(communities, membership, optimal_k, output_file)
        
        # Save runtime info
        runtime_file = output_dir / f"{dataset_name}_runtime_info.json"
        with open(runtime_file, 'w') as f:
            json.dump(runtime_info, f, indent=2)
        print(f"    Saved runtime info → {runtime_file}")
        
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

