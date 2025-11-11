"""
Computational Efficiency Benchmarking Module

Measures runtime and memory usage for each pipeline step.
Provides empirical evidence of computational efficiency.
"""

import time
import psutil
import os
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
import gc

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.data_preprocessing import preprocess_data
from src.graph.graph_construction import build_similarity_graph
from src.clustering.clustering import cluster_data


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def benchmark_preprocessing(input_file, output_dir='data/processed'):
    """Benchmark preprocessing step."""
    print("\n[Benchmark] Preprocessing...")
    start_time = time.time()
    start_memory = get_memory_usage()
    
    result = preprocess_data(
        input_file,
        output_dir=output_dir,
        variance_threshold="mean",
        apply_log2=True,
        apply_normalize=True
    )
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    runtime = end_time - start_time
    memory_used = end_memory - start_memory
    
    print(f"    Runtime: {runtime:.2f}s")
    print(f"    Memory: {memory_used:.2f} MB")
    
    return {
        'step': 'preprocessing',
        'runtime_seconds': runtime,
        'memory_mb': memory_used,
        'peak_memory_mb': end_memory
    }


def benchmark_graph_construction(processed_file, threshold=0.4):
    """Benchmark graph construction step."""
    print("\n[Benchmark] Graph Construction...")
    start_time = time.time()
    start_memory = get_memory_usage()
    
    data = np.load(processed_file)
    adjacency, similarity = build_similarity_graph(
        data,
        threshold=threshold,
        use_sparse=True
    )
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    runtime = end_time - start_time
    memory_used = end_memory - start_memory
    
    # Graph statistics
    if hasattr(adjacency, 'nnz'):
        n_edges = adjacency.nnz // 2
    else:
        n_edges = (adjacency > 0).sum() // 2
    
    n_nodes = data.shape[0]
    density = (n_edges * 2) / (n_nodes * (n_nodes - 1)) * 100 if n_nodes > 1 else 0
    
    print(f"    Runtime: {runtime:.2f}s")
    print(f"    Memory: {memory_used:.2f} MB")
    print(f"    Graph: {n_nodes} nodes, {n_edges} edges, {density:.2f}% density")
    
    del adjacency, similarity, data
    gc.collect()
    
    return {
        'step': 'graph_construction',
        'runtime_seconds': runtime,
        'memory_mb': memory_used,
        'peak_memory_mb': end_memory,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'density_percent': density
    }


def benchmark_clustering(adjacency_file, max_communities=10):
    """Benchmark BIGCLAM clustering step."""
    print("\n[Benchmark] BIGCLAM Clustering...")
    start_time = time.time()
    start_memory = get_memory_usage()
    
    import scipy.sparse as sp
    adjacency = sp.load_npz(adjacency_file)
    
    communities, membership, optimal_k = cluster_data(
        adjacency,
        max_communities=max_communities,
        iterations=100,
        lr=0.08,
        criterion='BIC',
        adaptive_lr=True,
        adaptive_iterations=True,
        early_stopping=True,
        num_restarts=1
    )
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    runtime = end_time - start_time
    memory_used = end_memory - start_memory
    
    n_communities = len(set(communities))
    
    print(f"    Runtime: {runtime:.2f}s")
    print(f"    Memory: {memory_used:.2f} MB")
    print(f"    Communities found: {n_communities} (optimal k: {optimal_k})")
    
    del adjacency, communities, membership
    gc.collect()
    
    return {
        'step': 'clustering',
        'runtime_seconds': runtime,
        'memory_mb': memory_used,
        'peak_memory_mb': end_memory,
        'n_communities': n_communities,
        'optimal_k': optimal_k
    }


def benchmark_full_pipeline(dataset_name, input_file, config_path='config/config.yml',
                           output_dir='results/benchmarks'):
    """
    Benchmark the complete pipeline.
    
    Args:
        dataset_name: Name of dataset
        input_file: Path to input CSV file
        config_path: Path to config file
        output_dir: Output directory for results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"COMPUTATIONAL BENCHMARK: {dataset_name}")
    print("="*80)
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    bigclam_config = config.get('bigclam', {})
    max_communities = bigclam_config.get('max_communities', 10)
    
    results = {
        'dataset': dataset_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'steps': []
    }
    
    # Step 1: Preprocessing
    prep_result = benchmark_preprocessing(input_file)
    results['steps'].append(prep_result)
    
    # Step 2: Graph Construction
    processed_file = Path('data/processed') / f"{dataset_name}_processed.npy"
    if not processed_file.exists():
        print(f"[ERROR] Processed file not found: {processed_file}")
        return None
    
    graph_result = benchmark_graph_construction(processed_file, threshold=0.4)
    results['steps'].append(graph_result)
    
    # Step 3: Clustering
    adjacency_file = Path('data/graphs') / f"{dataset_name}_adjacency.npz"
    if not adjacency_file.exists():
        print(f"[ERROR] Adjacency file not found: {adjacency_file}")
        return None
    
    cluster_result = benchmark_clustering(adjacency_file, max_communities=max_communities)
    results['steps'].append(cluster_result)
    
    # Summary
    total_runtime = sum(step['runtime_seconds'] for step in results['steps'])
    total_memory = max(step['peak_memory_mb'] for step in results['steps'])
    
    results['summary'] = {
        'total_runtime_seconds': total_runtime,
        'total_runtime_minutes': total_runtime / 60,
        'peak_memory_mb': total_memory,
        'peak_memory_gb': total_memory / 1024
    }
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal Runtime: {total_runtime:.2f}s ({total_runtime/60:.2f} minutes)")
    print(f"Peak Memory: {total_memory:.2f} MB ({total_memory/1024:.2f} GB)")
    
    print("\nBreakdown:")
    for step in results['steps']:
        print(f"  {step['step']:20s}: {step['runtime_seconds']:6.2f}s ({step['memory_mb']:6.2f} MB)")
    
    # Save results
    output_file = output_dir / f"{dataset_name}_benchmark.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save as CSV for easy viewing
    df = pd.DataFrame(results['steps'])
    csv_file = output_dir / f"{dataset_name}_benchmark.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"\n[Saved] Results: {output_file}")
    print(f"[Saved] CSV: {csv_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark computational efficiency')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['tcga_brca_data', 'gse96058_data'],
                       help='Dataset to benchmark')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--config', type=str, default='config/config.yml',
                       help='Config file path')
    parser.add_argument('--output_dir', type=str, default='results/benchmarks',
                       help='Output directory')
    
    args = parser.parse_args()
    
    benchmark_full_pipeline(
        args.dataset,
        args.input_file,
        args.config,
        args.output_dir
    )

