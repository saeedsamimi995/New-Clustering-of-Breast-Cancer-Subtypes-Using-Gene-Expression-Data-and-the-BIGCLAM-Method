"""
Data Preprocessing Module

Handles:
1. Loading prepared data (genes x samples with target in last row)
2. Dropping target labels
3. Feature reduction via variance threshold
4. Log2 transformation
5. Z-score normalization across samples
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


def load_data_with_target(file_path):
    """
    Load gene expression data with target labels.
    
    Args:
        file_path: Path to CSV with genes as rows, samples as columns, target as last row
        
    Returns:
        tuple: (expression_data, target_labels, gene_names, sample_names)
    """
    print(f"\n[Loading] {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Get sample names (column headers, skip first which is gene names)
    sample_names = df.columns[1:].values
    
    # Get target labels from last row
    last_row = df.iloc[-1, :].values
    target_labels = last_row[1:]  # Skip first element which is row name
    
    # Get gene names (all rows except last)
    gene_names = df.iloc[:-1, 0].values
    
    # Get expression data (all rows except last, all columns except first)
    expression_data = df.iloc[:-1, 1:].values.T  # Transpose: samples x genes
    print(f"    Loaded: {expression_data.shape[0]} samples, {expression_data.shape[1]} genes")
    
    return expression_data, target_labels, gene_names, sample_names


def apply_variance_filter(data, threshold=13):
    """
    Filter features with variance below threshold.
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        threshold: variance threshold
        
    Returns:
        tuple: (filtered_data, selected_feature_indices)
    """
    print(f"\n[Variance Filter] Threshold={threshold}...")
    var_threshold = VarianceThreshold(threshold=threshold)
    data_filtered = var_threshold.fit_transform(data)
    selected_features = var_threshold.get_support()
    n_kept = selected_features.sum()
    n_total = len(selected_features)
    print(f"    Kept {n_kept:,}/{n_total:,} features ({n_kept/n_total*100:.1f}%)")
    
    return data_filtered, np.where(selected_features)[0]


def apply_log2_transform(data):
    """
    Apply log2(x+1) transformation to handle zeros.
    
    Args:
        data: numpy array
        
    Returns:
        numpy array: log2-transformed data
    """
    print("\n[Log2 Transform]...")
    data_transformed = np.log2(data + 1)
    print(f"    Values: min={data_transformed.min():.2f}, max={data_transformed.max():.2f}, mean={data_transformed.mean():.2f}")
    return data_transformed


def apply_zscore_normalize(data):
    """
    Normalize data across samples (rows) using z-score.
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        
    Returns:
        tuple: (normalized_data, scaler_object)
    """
    print("\n[Z-score Normalization]...")
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    print(f"    Mean: {data_normalized.mean():.6f}, Std: {data_normalized.std():.6f}")
    return data_normalized, scaler


def preprocess_data(input_file, output_dir='data/processed', variance_threshold=13, 
                   apply_log2=True, apply_normalize=True):
    """
    Complete preprocessing pipeline.
    
    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save processed data
        variance_threshold: Variance threshold for feature filtering
        apply_log2: Whether to apply log2 transformation
        apply_normalize: Whether to apply z-score normalization
        
    Returns:
        dict: Preprocessed data and metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    expression_data, target_labels, gene_names, sample_names = load_data_with_target(input_file)
    
    # Log2 transformation
    if apply_log2:
        expression_data = apply_log2_transform(expression_data)
    else:
        print("\n[Skipping] Log2 transformation")
    
    # Variance filtering
    expression_data, selected_features = apply_variance_filter(expression_data, variance_threshold)
    selected_gene_names = gene_names[selected_features]
    
    # Z-score normalization
    if apply_normalize:
        expression_data, scaler = apply_zscore_normalize(expression_data)
    else:
        scaler = None
        print("\n[Skipping] Z-score normalization")
    
    # Save results
    output_base = Path(input_file).stem
    np.save(output_dir / f'{output_base}_processed.npy', expression_data)
    with open(output_dir / f'{output_base}_targets.pkl', 'wb') as f:
        pickle.dump({
            'target_labels': target_labels,
            'gene_names': selected_gene_names.tolist(),
            'sample_names': sample_names.tolist(),
            'selected_features': selected_features
        }, f)
    
    print(f"\n[Saving] Processed data to {output_dir}/")
    print(f"    Expression matrix: {output_base}_processed.npy ({expression_data.shape})")
    print(f"    Metadata: {output_base}_targets.pkl")
    
    return {
        'expression_data': expression_data,
        'target_labels': target_labels,
        'gene_names': selected_gene_names,
        'sample_names': sample_names,
        'scaler': scaler,
        'selected_features': selected_features
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess gene expression data')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output_dir', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--variance_threshold', type=float, default=13, help='Variance threshold')
    parser.add_argument('--no_log2', action='store_true', help='Skip log2 transformation')
    parser.add_argument('--no_normalize', action='store_true', help='Skip z-score normalization')
    
    args = parser.parse_args()
    
    preprocess_data(
        args.input,
        args.output_dir,
        variance_threshold=args.variance_threshold,
        apply_log2=not args.no_log2,
        apply_normalize=not args.no_normalize
    )

