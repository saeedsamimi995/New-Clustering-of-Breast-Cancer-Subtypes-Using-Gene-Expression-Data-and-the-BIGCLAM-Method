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
import argparse
import yaml
import os
import sys
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


def load_data_with_target(file_path):
    """
    Load gene expression data with target labels.
    
    **IMPORTANT**: This function separates the target labels from expression data.
    The target (oncotree/PAM50) is stored in the last row of the input CSV and is 
    extracted separately. Expression data excludes the last row to ensure targets
    are not used for graph construction or clustering.
    
    Args:
        file_path: Path to CSV with genes as rows, samples as columns, target as last row
        
    Returns:
        tuple: (expression_data, target_labels, gene_names, sample_names)
               - expression_data: numpy array (n_samples, n_genes) WITHOUT target
               - target_labels: array of target labels (extracted from last row)
               - gene_names: array of gene identifiers
               - sample_names: array of sample identifiers
    """
    print(f"\n[Loading] {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Get sample names (column headers, skip first which is gene names)
    sample_names = df.columns[1:].values
    
    # Extract target labels from last row (oncotree/PAM50 row)
    last_row = df.iloc[-1, :].values
    target_labels = last_row[1:]  # Skip first element which is row name
    
    # Get gene names (all rows except last - target row is excluded)
    gene_names = df.iloc[:-1, 0].values
    
    # *** TARGET DROPPED HERE ***
    # Get expression data (all rows except last, all columns except first)
    # This ensures the target row (oncotree/PAM50) is NOT included in expression data
    # Expression data will be used for graph construction - targets must be excluded
    expression_df = df.iloc[:-1, 1:]  # Get expression rows and columns (exclude first column with gene names)
    
    # Convert to numeric, coercing any non-numeric values to NaN
    # This handles cases where pandas reads values as strings or mixed types
    expression_df = expression_df.apply(pd.to_numeric, errors='coerce')
    
    # Convert to numpy array and transpose: samples x genes
    expression_data = expression_df.values.T
    
    # Check for and report any NaN values (from non-numeric coercion)
    nan_count = np.isnan(expression_data).sum()
    if nan_count > 0:
        print(f"    [WARNING] Found {nan_count} NaN values (non-numeric entries converted to NaN)")
        print(f"             Filling NaN values with 0 for downstream processing...")
        expression_data = np.nan_to_num(expression_data, nan=0.0)
    
    # Ensure data is float type for numeric operations
    expression_data = expression_data.astype(np.float64)
    
    print(f"    Loaded: {expression_data.shape[0]} samples, {expression_data.shape[1]} genes")
    print(f"    Data type: {expression_data.dtype}")
    print(f"    Target labels extracted separately: {len(target_labels)} labels")
    
    return expression_data, target_labels, gene_names, sample_names


def apply_variance_filter(data, threshold="mean"):
    """
    Filter features with variance below threshold.
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        threshold: variance threshold. Can be:
                   - numeric value: use as fixed threshold
                   - "mean": use mean variance of all features as threshold
        
    Returns:
        tuple: (filtered_data, selected_feature_indices)
    """
    # Ensure data is valid (no NaN/Inf)
    if np.isnan(data).any() or np.isinf(data).any():
        print(f"    [WARNING] Found NaN/Inf in data, cleaning...")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate threshold if "mean" is specified
    if isinstance(threshold, str) and threshold.lower() == "mean":
        # Calculate variance for each feature (axis=0 means across samples)
        feature_variances = np.var(data, axis=0)
        
        # Remove NaN variances (from constant features or invalid data)
        valid_variances = feature_variances[np.isfinite(feature_variances)]
        
        if len(valid_variances) == 0:
            print(f"    [ERROR] No valid variances found! Using default threshold of 0.01...")
            threshold_value = 0.01
        else:
            threshold_value = np.mean(valid_variances)
            
            if not np.isfinite(threshold_value) or threshold_value < 0:
                print(f"    [WARNING] Mean variance is invalid ({threshold_value}), using median instead...")
                threshold_value = np.median(valid_variances)
                if not np.isfinite(threshold_value) or threshold_value < 0:
                    print(f"    [WARNING] Median variance also invalid, using default 0.01...")
                    threshold_value = 0.01
        
        print(f"\n[Variance Filter] Using mean variance as threshold...")
        print(f"    Mean variance: {threshold_value:.4f}")
        print(f"    Valid features: {len(valid_variances)}/{len(feature_variances)}")
    else:
        threshold_value = float(threshold)
        print(f"\n[Variance Filter] Threshold={threshold_value}...")
    
    var_threshold = VarianceThreshold(threshold=threshold_value)
    data_filtered = var_threshold.fit_transform(data)
    selected_features = var_threshold.get_support()
    n_kept = selected_features.sum()
    n_total = len(selected_features)
    print(f"    Kept {n_kept:,}/{n_total:,} features ({n_kept/n_total*100:.1f}%)")
    
    return data_filtered, np.where(selected_features)[0]


def apply_log2_transform(data):
    """
    Apply log2(x+1) transformation to handle zeros.
    
    **Note**: If data appears already log-transformed (has negatives, small range),
    the transformation will be skipped to avoid double transformation.
    
    Args:
        data: numpy array (must be numeric)
        
    Returns:
        numpy array: log2-transformed data (or original if already transformed)
    """
    print("\n[Log2 Transform]...")
    
    # Ensure data is numeric
    if not np.issubdtype(data.dtype, np.number):
        print(f"    [WARNING] Data type is {data.dtype}, converting to float64...")
        data = pd.to_numeric(data.flatten(), errors='coerce').reshape(data.shape)
        data = np.nan_to_num(data, nan=0.0)
        data = data.astype(np.float64)
    
    # Check for invalid values before transformation
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    negative_count = (data < 0).sum()
    
    if nan_count > 0:
        print(f"    [WARNING] Found {nan_count} NaN values, filling with 0...")
        data = np.nan_to_num(data, nan=0.0)
    
    if inf_count > 0:
        print(f"    [WARNING] Found {inf_count} infinite values, clipping to finite range...")
        data = np.where(np.isfinite(data), data, 0.0)
    
    # Check if data appears already log-transformed and/or normalized
    # GSE96058: "normalized and transformed" - likely already processed
    # Log-transformed expression data typically has:
    # - Small range (< 20-25) 
    # - Mean in log-space (typically 2-10 for log2) OR near 0 (if z-score normalized)
    # - May have negatives (if normalized after log transformation)
    # - Values typically don't exceed ~15-20 for log2 space OR can be negative (if normalized)
    # Raw count data typically has:
    # - Large range (can be 0 to millions)
    # - Mean much larger (> 100 for raw counts)
    # - No negatives (biologically impossible)
    
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    data_range = data_max - data_min
    data_mean_raw = np.nanmean(data)  # Raw mean (can be negative for normalized data)
    data_mean = np.abs(data_mean_raw)  # Use absolute mean to check magnitude
    data_std = np.nanstd(data)
    total_values = len(data.flatten())
    
    # Multiple indicators that data is already processed (log-transformed and/or normalized):
    indicator_1 = data_range < 25  # Small range typical of log-space or normalized
    indicator_2 = data_max < 20     # Max value reasonable for log-space
    indicator_3 = data_mean < 15    # Mean reasonable for log-space (or near 0 if normalized)
    indicator_4 = negative_count > total_values * 0.05  # Many negatives (normalized data)
    indicator_5 = (data_mean < 2) and (data_std < 5)  # Near zero mean, small std = z-score normalized
    
    # Check 1: Has negatives + small range = definitely processed (normalized log-space)
    has_negatives_and_small_range = (negative_count > 0) and (data_range < 25)
    
    # Check 2: Z-score normalized characteristics (mean ≈ 0, std ≈ 1, can have negatives)
    looks_zscore_normalized = (
        indicator_5 and 
        negative_count > 0 and
        data_range < 15  # Normalized data has smaller range
    )
    
    # Check 3: No negatives but looks like log-space (positive log-transformed values)
    no_negatives_but_log_space = (
        negative_count == 0 and 
        indicator_1 and 
        indicator_2 and 
        indicator_3
    )
    
    is_likely_log_transformed = has_negatives_and_small_range or looks_zscore_normalized or no_negatives_but_log_space
    
    if is_likely_log_transformed:
        print(f"    [SKIP] Data appears already processed (log-transformed and/or normalized):")
        print(f"             - Range: [{data_min:.2f}, {data_max:.2f}] (typical for processed data: < 25)")
        print(f"             - Mean: {np.nanmean(data):.2f}, Std: {data_std:.2f}")
        
        if looks_zscore_normalized:
            print(f"             - Characteristics: Z-score normalized (mean ≈ 0, std ≈ 1)")
            print(f"             - Has {negative_count} negative values ({negative_count/total_values*100:.1f}%)")
            print(f"               This is expected for normalized data (~50% should be negative)")
        elif negative_count > 0:
            print(f"             - Has {negative_count} negative values ({negative_count/total_values*100:.1f}%)")
            print(f"               (Likely normalized after log transformation)")
        else:
            print(f"             - No negatives (all values positive, typical for log2(RSEM))")
        
        print(f"             - Skipping log2 transformation to avoid double transformation")
        print(f"             - Only cleaning invalid values (NaN/Inf)")
        
        # Just clean NaN/Inf and return original (already processed) data
        # Preserve negatives - they're valid in normalized/log-space
        data_cleaned = np.nan_to_num(data, nan=0.0, posinf=data_max, neginf=data_min)
        print(f"    Final values: min={data_cleaned.min():.2f}, max={data_cleaned.max():.2f}, mean={data_cleaned.mean():.2f}")
        return data_cleaned
    
    # Data appears to be raw counts - proceed with log2 transformation
    if negative_count > 0:
        print(f"    [WARNING] Found {negative_count} negative values in raw data (min={data_min:.2f})")
        print(f"             Clipping negatives to 0 (biologically, expression cannot be negative)")
        data = np.maximum(data, 0.0)
    
    # Apply log2 transformation: log2(x+1) where x >= 0 ensures x+1 >= 1
    print(f"    Applying log2(x+1) transformation to raw expression data...")
    data_transformed = np.log2(data + 1)
    
    # Check for NaN/Inf after transformation and handle
    nan_after = np.isnan(data_transformed).sum()
    inf_after = np.isinf(data_transformed).sum()
    
    if nan_after > 0 or inf_after > 0:
        print(f"    [WARNING] Found {nan_after} NaN and {inf_after} Inf after log2, fixing...")
        data_transformed = np.nan_to_num(data_transformed, nan=0.0, posinf=50.0, neginf=0.0)
    
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


def preprocess_data(input_file, output_dir='data/processed', variance_threshold="mean", 
                   apply_log2=True, apply_normalize=True):
    """
    Complete preprocessing pipeline.
    
    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save processed data
        variance_threshold: Variance threshold for feature filtering.
                           Can be numeric value or "mean" to use mean variance
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
    # Output file names are derived from input file name, with "_target_added" removed:
    # - Input: data/tcga_brca_data_target_added.csv 
    #   Output: tcga_brca_data_processed.npy (expression data, NO target)
    #   Output: tcga_brca_data_targets.pkl (target labels only)
    # - Input: data/gse96058_data_target_added.csv
    #   Output: gse96058_data_processed.npy (expression data, NO target)
    #   Output: gse96058_data_targets.pkl (target labels only)
    # 
    # The "_target_added" suffix is removed from output names since the processed .npy file
    # does NOT contain targets (they are saved separately in .pkl file)
    input_base = Path(input_file).stem
    # Remove "_target_added" suffix if present to reflect that target is removed
    if input_base.endswith('_target_added'):
        output_base = input_base[:-13]  # Remove "_target_added" (13 characters)
    else:
        output_base = input_base
    
    output_expression_file = output_dir / f'{output_base}_processed.npy'
    output_metadata_file = output_dir / f'{output_base}_targets.pkl'
    
    # Save expression data (target-free, ready for graph construction)
    # Using .npy format (NumPy's native binary format) for several reasons:
    # 1. EFFICIENCY: Much faster I/O than CSV/text formats for large numerical arrays
    # 2. SIZE: Compressed binary format - significantly smaller file sizes than CSV
    # 3. PRECISION: Preserves exact numerical values (no text parsing errors)
    # 4. STRUCTURE: Maintains numpy array structure (shape, dtype) automatically
    # 5. STANDARD: Widely used in scientific computing for numerical data
    # 6. PERFORMANCE: Direct memory mapping possible for very large files
    # For example: A CSV with 1000 samples × 10000 genes might be ~200MB, 
    #             while .npy format could be ~80MB with faster load times
    np.save(output_expression_file, expression_data)
    with open(output_metadata_file, 'wb') as f:
        pickle.dump({
            'target_labels': target_labels,
            'gene_names': selected_gene_names.tolist(),
            'sample_names': sample_names.tolist(),
            'selected_features': selected_features
        }, f)
    
    print(f"\n[Saving] Processed data to {output_dir}/")
    print(f"    Expression matrix: {output_expression_file.name} ({expression_data.shape})")
    print(f"    Metadata: {output_metadata_file.name}")
    
    return {
        'expression_data': expression_data,
        'target_labels': target_labels,
        'gene_names': selected_gene_names,
        'sample_names': sample_names,
        'scaler': scaler,
        'selected_features': selected_features
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess TCGA BRCA and GSE96058 datasets for BIGCLAM pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yml',
        help='Path to configuration file (default: config/config.yml)'
    )
    parser.add_argument(
        '--dataset', 
        choices=['tcga', 'gse96058', 'both'],
        default='both',
        help='Which dataset(s) to process'
    )
    parser.add_argument('--input', type=str, default=None, 
                       help='Input CSV file path (overrides config if provided)')
    parser.add_argument('--output_dir', type=str, default='data/processed', 
                       help='Output directory')
    parser.add_argument('--variance_threshold', type=str, default=None, 
                       help='Variance threshold (numeric value or "mean" to use mean variance). Overrides config if provided.')
    parser.add_argument('--no_log2', action='store_true', help='Skip log2 transformation')
    parser.add_argument('--no_normalize', action='store_true', help='Skip z-score normalization')
    
    args = parser.parse_args()
    
    # If single input file is provided, use that mode
    if args.input:
        preprocess_data(
            args.input,
            args.output_dir,
            variance_threshold=args.variance_threshold or 'mean',
            apply_log2=not args.no_log2,
            apply_normalize=not args.no_normalize
        )
        sys.exit(0)
    
    # Otherwise, use config file mode
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Please create a configuration file or use the default: config/config.yml")
        print("Alternatively, use --input to process a single file")
        sys.exit(1)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get preprocessing parameters from config
    preprocessing_config = config.get('preprocessing', {})
    variance_threshold = args.variance_threshold or preprocessing_config.get('variance_threshold', 'mean')
    
    # Extract dataset paths from config
    if 'dataset_preparation' not in config:
        print("Error: 'dataset_preparation' section not found in config file")
        sys.exit(1)
    
    dataset_config = config['dataset_preparation']
    
    print("=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)
    
    # Process TCGA
    if args.dataset in ['tcga', 'both']:
        tcga_config = dataset_config.get('tcga', {})
        tcga_output = tcga_config.get('output')
        
        if tcga_output and Path(tcga_output).exists():
            print(f"\n[Processing TCGA BRCA]...")
            print(f"    Input: {tcga_output}")
            # Remove "_target_added" from output name since target is removed from .npy file
            output_base = Path(tcga_output).stem
            if output_base.endswith('_target_added'):
                output_base = output_base[:-13]
            print(f"    Output files will be:")
            print(f"      - {output_base}_processed.npy (expression data, target removed)")
            print(f"      - {output_base}_targets.pkl (target labels only)")
            preprocess_data(
                tcga_output,
                args.output_dir,
                variance_threshold=variance_threshold,
                apply_log2=not args.no_log2,
                apply_normalize=not args.no_normalize
            )
        else:
            print(f"\n⚠ TCGA output file not found. Skipping TCGA preprocessing.")
            print(f"   Expected: {tcga_output}")
            print(f"   Run data preparation first: python -m src.preprocessing.data_preparing --dataset tcga")
    
    # Process GSE96058
    if args.dataset in ['gse96058', 'both']:
        gse_config = dataset_config.get('gse96058', {})
        gse_output = gse_config.get('output')
        
        if gse_output and Path(gse_output).exists():
            print(f"\n[Processing GSE96058]...")
            print(f"    Input: {gse_output}")
            # Remove "_target_added" from output name since target is removed from .npy file
            output_base = Path(gse_output).stem
            if output_base.endswith('_target_added'):
                output_base = output_base[:-13]
            print(f"    Output files will be:")
            print(f"      - {output_base}_processed.npy (expression data, target removed)")
            print(f"      - {output_base}_targets.pkl (target labels only)")
            preprocess_data(
                gse_output,
                args.output_dir,
                variance_threshold=variance_threshold,
                apply_log2=not args.no_log2,
                apply_normalize=not args.no_normalize
            )
        else:
            print(f"\n⚠ GSE96058 output file not found. Skipping GSE96058 preprocessing.")
            print(f"   Expected: {gse_output}")
            print(f"   Run data preparation first: python -m src.preprocessing.data_preparing --dataset gse96058")
    
    print("\n" + "=" * 80)
    print("DATA PREPROCESSING COMPLETE")
    print("=" * 80)

