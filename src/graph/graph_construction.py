"""
Graph Construction Module

Builds similarity graphs from preprocessed expression data for BIGCLAM input.
- Computes cosine similarity between samples
- Creates adjacency matrix based on similarity threshold
- Memory-efficient with sparse matrices
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import gc


def calculate_similarity_chunked(features, chunk_size=5000):
    """
    Calculate cosine similarity in chunks to reduce memory usage.
    
    Args:
        features: Feature matrix (N x d)
        chunk_size: Number of samples to process at a time
        
    Returns:
        Similarity matrix (N x N)
    """
    N = features.shape[0]
    similarity_matrix = np.zeros((N, N), dtype=np.float32)
    
    print(f"    Computing similarity in chunks of {chunk_size}...")
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        for j in range(i, N, chunk_size):
            end_j = min(j + chunk_size, N)
            similarity_matrix[i:end_i, j:end_j] = cosine_similarity(
                features[i:end_i], 
                features[j:end_j]
            )
            # Copy upper triangle to lower triangle (symmetric)
            if i != j:
                similarity_matrix[j:end_j, i:end_i] = similarity_matrix[i:end_i, j:end_j].T
            
            if j % (chunk_size * 5) == 0:
                print(f"      Progress: {j}/{N}")
    
    return similarity_matrix


def build_similarity_graph(expression_data, threshold=0.4, use_sparse=True, chunk_size=None):
    """
    Build similarity graph from expression data.
    
    Args:
        expression_data: numpy array of shape (n_samples, n_features)
        threshold: Cosine similarity threshold for edges
        use_sparse: Whether to return sparse matrix
        chunk_size: Chunk size for similarity calculation (None for auto)
        
    Returns:
        tuple: (adjacency_matrix, similarity_matrix)
    """
    print("\n[Graph Construction]...")
    print(f"    Data shape: {expression_data.shape}")
    print(f"    Similarity threshold: {threshold}")
    
    N, d = expression_data.shape
    
    # Compute similarity
    if chunk_size is None:
        # Auto-detect chunk size based on memory
        chunk_size = min(5000, N)
    
    if N > 5000:
        print(f"    Using chunked similarity calculation...")
        similarity_matrix = calculate_similarity_chunked(expression_data, chunk_size)
    else:
        print(f"    Computing full similarity matrix...")
        similarity_matrix = cosine_similarity(expression_data)
    
    # Create adjacency matrix
    print(f"    Creating adjacency matrix from similarity...")
    adjacency = (similarity_matrix > threshold).astype(int)
    # Remove self-loops
    np.fill_diagonal(adjacency, 0)
    
    # Convert to sparse if requested
    if use_sparse and N > 1000:
        adjacency_sparse = csr_matrix(adjacency)
        print(f"    Converting to sparse matrix...")
        n_edges = adjacency_sparse.nnz
        density = n_edges / (N * N) * 100
        print(f"    Edges: {n_edges:,} ({density:.2f}% density)")
        adjacency = adjacency_sparse
    else:
        n_edges = (adjacency > 0).sum()
        print(f"    Edges: {n_edges:,} ({n_edges/(N*N)*100:.2f}% density)")
    
    gc.collect()
    
    return adjacency, similarity_matrix


def save_graph_data(adjacency, output_file):
    """
    Save graph adjacency matrix.
    
    Args:
        adjacency: Adjacency matrix (sparse or dense)
        output_file: Output file path
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if hasattr(adjacency, 'nnz'):
        # Sparse matrix
        import scipy.sparse
        scipy.sparse.save_npz(output_file, adjacency, compressed=True)
        print(f"    Saved sparse graph to: {output_file}")
    else:
        # Dense matrix
        np.save(output_file, adjacency)
        print(f"    Saved dense graph to: {output_file}")


def load_graph_data(input_file):
    """
    Load graph adjacency matrix.
    
    Args:
        input_file: Input file path
        
    Returns:
        Adjacency matrix
    """
    input_file = Path(input_file)
    
    if input_file.suffix == '.npz':
        import scipy.sparse
        adjacency = scipy.sparse.load_npz(input_file)
        print(f"    Loaded sparse graph from: {input_file}")
    else:
        adjacency = np.load(input_file)
        print(f"    Loaded dense graph from: {input_file}")
    
    return adjacency


def construct_graphs(input_dir='data/processed', output_dir='data/graphs', 
                    threshold=None, thresholds_dict=None, use_sparse=True):
    """
    Construct graphs for all processed datasets.
    
    Args:
        input_dir: Directory containing processed data
        output_dir: Directory to save graphs
        threshold: Single similarity threshold (used if thresholds_dict not provided)
        thresholds_dict: Dictionary mapping dataset names to thresholds (e.g., {'tcga_brca_data': 0.2, 'gse96058_data': 0.6})
        use_sparse: Whether to use sparse matrices
        
    Returns:
        dict: Graph data for each dataset
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Find all processed files
    processed_files = list(input_dir.glob('*_processed.npy'))
    
    if not processed_files:
        print(f"No processed files found in {input_dir}")
        return results
    
    for processed_file in processed_files:
        print("\n" + "="*80)
        dataset_name = processed_file.stem.replace('_processed', '')
        print(f"CONSTRUCTING GRAPH: {dataset_name}")
        print("="*80)
        
        # Determine threshold for this dataset
        if thresholds_dict and dataset_name in thresholds_dict:
            dataset_threshold = thresholds_dict[dataset_name]
            print(f"Using dataset-specific threshold: {dataset_threshold}")
        elif thresholds_dict and 'default' in thresholds_dict:
            dataset_threshold = thresholds_dict['default']
            print(f"Using default threshold: {dataset_threshold} (dataset '{dataset_name}' not in thresholds_dict)")
        elif threshold is not None:
            dataset_threshold = threshold
            print(f"Using provided threshold: {dataset_threshold}")
        else:
            dataset_threshold = 0.4  # Fallback default
            print(f"Using fallback default threshold: {dataset_threshold}")
        
        # Load expression data
        expression_data = np.load(processed_file)
        
        # Build graph
        adjacency, similarity = build_similarity_graph(
            expression_data, 
            threshold=dataset_threshold,
            use_sparse=use_sparse
        )
        
        # Save graph
        graph_file = output_dir / f"{dataset_name}_adjacency"
        save_graph_data(adjacency, graph_file)
        
        results[dataset_name] = {
            'adjacency': adjacency,
            'similarity': similarity,
            'n_samples': expression_data.shape[0]
        }
    
    return results


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Construct similarity graphs')
    parser.add_argument('--input_dir', type=str, default='data/processed', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='data/graphs', help='Output directory')
    parser.add_argument('--threshold', type=float, default=None, help='Single similarity threshold (overrides config)')
    parser.add_argument('--config', type=str, default='config/config.yml', help='Config file for dataset-specific thresholds')
    parser.add_argument('--use_sparse', action='store_true', default=True, help='Use sparse matrices')
    parser.add_argument('--no_sparse', action='store_false', dest='use_sparse', help='Use dense matrices')
    
    args = parser.parse_args()
    
    # Load thresholds from config if not provided
    thresholds_dict = None
    if args.threshold is None:
        if Path(args.config).exists():
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
                preprocessing_config = config.get('preprocessing', {})
                similarity_thresholds = preprocessing_config.get('similarity_thresholds', {})
                if similarity_thresholds:
                    thresholds_dict = similarity_thresholds
                    print("Loaded dataset-specific thresholds from config")
    
    construct_graphs(
        args.input_dir,
        args.output_dir,
        threshold=args.threshold,
        thresholds_dict=thresholds_dict,
        use_sparse=args.use_sparse
    )

