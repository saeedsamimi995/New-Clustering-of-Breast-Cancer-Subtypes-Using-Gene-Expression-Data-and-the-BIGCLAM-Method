"""Graph construction utilities for BIGCLAM with memory optimization."""

import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import gc


def create_adjacency_and_assignments(df, threshold=0.4, use_sparse=True, chunk_size=None):
    """
    Create adjacency matrix and community assignments from dataframe.
    Memory-efficient version with sparse matrices and optional chunking.
    
    Args:
        df (pd.DataFrame): Dataframe with features and optional community column.
        threshold (float): Similarity threshold for edge creation.
        use_sparse (bool): If True, use sparse matrix representation (recommended for >10K samples).
        chunk_size (int, optional): Chunk size for similarity calculation. If None, auto-detected.
        
    Returns:
        tuple: (A, p2c) where A is the adjacency matrix (sparse or dense) and p2c is list of community sets.
    """
    N = df.shape[0]
    num_features = df.shape[1] - 1  # last column is community
    
    features = df.iloc[:, :num_features].values
    
    # Memory-efficient similarity calculation with chunking for large datasets
    print(f"Calculating cosine similarity for {N} samples...")
    if chunk_size is None:
        # Auto-detect chunk size based on memory considerations
        # For 12GB RAM, process ~5000 samples at a time
        chunk_size = min(5000, N)
    
    if N > 5000 and chunk_size is not None:
        print(f"Using chunked similarity calculation (chunk_size={chunk_size})...")
        similarity_matrix = _calculate_similarity_chunked(features, chunk_size)
    else:
        similarity_matrix = cosine_similarity(features)
    
    # Create adjacency matrix based on similarity threshold
    if use_sparse and N > 1000:
        # Use sparse matrix for memory efficiency
        A = csr_matrix((similarity_matrix > threshold).astype(int) - np.eye(N).astype(int))
        print("Using sparse adjacency matrix representation")
    else:
        A = (similarity_matrix > threshold).astype(int) - np.eye(N).astype(int)
    
    # Clean up memory
    del similarity_matrix
    gc.collect()
    
    # Create community assignments from last column
    if df.shape[1] > num_features:
        community_assignments = df.iloc[:, -1].apply(lambda x: {int(x)}).tolist()
    else:
        # If no community column, assign each node to its own community
        community_assignments = [{i} for i in range(N)]
    
    return A, community_assignments


def _calculate_similarity_chunked(features, chunk_size):
    """
    Calculate cosine similarity in chunks to reduce memory usage.
    
    Args:
        features (np.ndarray): Feature matrix (N x d).
        chunk_size (int): Number of samples to process at a time.
        
    Returns:
        np.ndarray: Similarity matrix (N x N).
    """
    N = features.shape[0]
    similarity_matrix = np.zeros((N, N), dtype=np.float32)  # Use float32 to save memory
    
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        chunk_i = features[i:end_i]
        
        for j in range(0, N, chunk_size):
            end_j = min(j + chunk_size, N)
            chunk_j = features[j:end_j]
            
            chunk_sim = cosine_similarity(chunk_i, chunk_j)
            similarity_matrix[i:end_i, j:end_j] = chunk_sim
            
            print(f"  Processing chunk {i//chunk_size + 1}/{(N-1)//chunk_size + 1} x {j//chunk_size + 1}/{(N-1)//chunk_size + 1}...")
        
        # Free memory after each row chunk
        gc.collect()
    
    return similarity_matrix


def gen_json(A, p2c, F_argmax=None):
    """
    Generate JSON data for graph visualization.
    
    Args:
        A (np.ndarray, scipy.sparse matrix, or torch.Tensor): Adjacency matrix.
        p2c (list): List of community sets for each node.
        F_argmax (np.ndarray, optional): Assigned communities from BIGCLAM.
        
    Returns:
        dict: JSON-formatted graph data with nodes and links.
    """
    # Convert sparse matrix to dense for iteration
    from scipy.sparse import issparse
    if issparse(A):
        # For large sparse matrices, limit visualization to first 5000 nodes for memory
        max_nodes = 5000
        if A.shape[0] > max_nodes:
            print(f"Large graph detected ({A.shape[0]} nodes). Limiting visualization to {max_nodes} nodes.")
            A = A[:max_nodes, :max_nodes].toarray()
            p2c = p2c[:max_nodes]
            if F_argmax is not None:
                F_argmax = F_argmax[:max_nodes]
        else:
            A = A.toarray()
    
    # Convert torch tensor to numpy if needed
    if hasattr(A, 'cpu'):
        A = A.cpu().numpy()
    
    N = A.shape[0]
    data = {'nodes': [], 'links': []}
    
    for i in range(N):
        grp = ''.join(map(str, sorted(p2c[i])))
        node = {'id': str(i), 'group': str(grp)}
        if F_argmax is not None:
            node.update({'assigned': str(F_argmax[i])})
        data['nodes'].append(node)
        
        friends = np.where(A[i])[0]
        for friend in friends:
            inter = 2 - len(p2c[i].intersection(p2c[friend]))
            data['links'].append({
                'source': str(i),
                'target': str(friend),
                'value': str(inter * 5 + 1),
                'distance': str(inter * 15 + 1)
            })
    
    return data


def save_graph_data(A, p2c, save_path, adj_filename='adj.npy', p2c_filename='p2c.pkl'):
    """
    Save adjacency matrix and community assignments.
    Memory-efficient version supporting sparse matrices.
    
    Args:
        A (np.ndarray or scipy.sparse matrix): Adjacency matrix.
        p2c (list): Community assignments.
        save_path (str): Directory path to save files.
        adj_filename (str): Filename for adjacency matrix.
        p2c_filename (str): Filename for community assignments.
    """
    import os
    import pickle
    from scipy.sparse import issparse, save_npz
    
    # Create directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Make adjacency matrix symmetric
    if issparse(A):
        A_symmetric = A + A.T
        # Save sparse matrix efficiently
        save_npz(os.path.join(save_path, adj_filename.replace('.npy', '.npz')), A_symmetric)
        print("Saved sparse adjacency matrix")
    else:
        A_symmetric = A + A.T
        np.save(os.path.join(save_path, adj_filename), A_symmetric)
        print("Saved dense adjacency matrix")
    
    pickle.dump(p2c, open(os.path.join(save_path, p2c_filename), "wb"))
    
    print(f"Graph data saved to: {save_path}")

