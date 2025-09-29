"""Data loading and preprocessing functions with memory optimization."""

import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import gc


def load_data(file_path, chunk_size=None):
    """
    Load gene expression data and prepare it for processing.
    Memory-efficient version with optional chunking for large files.
    
    Args:
        file_path (str): Path to the CSV file containing gene expression data.
        chunk_size (int, optional): Number of rows to read at a time. If None, loads entire file.
        
    Returns:
        tuple: (X, data) where X is the feature matrix and data is the full dataframe.
    """
    # For memory efficiency: determine if we need chunking
    if chunk_size is None:
        # Try to estimate file size
        file_size = os.path.getsize(file_path)
        # If file is larger than 500MB, use chunking
        if file_size > 500 * 1024 * 1024:
            chunk_size = 10000
            print(f"Large file detected ({file_size / (1024*1024):.2f} MB). Using chunked loading...")
    
    if chunk_size is not None:
        # Memory-efficient chunked loading
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
    else:
        # PAM50 column from clinical dataset has been added to genomic dataset as target
        df = pd.read_csv(file_path)
    
    # Making columns as features
    print("Transposing dataframe...")
    data = df.T
    
    del df
    gc.collect()
    
    # Removing PAM50 column from dataset (first row contains feature names)
    X = data.iloc[1:, :]
    
    return X, data


def apply_variance_threshold(X, data, threshold=13):
    """
    Apply variance threshold to reduce features.
    
    Args:
        X (pd.DataFrame): Feature matrix.
        data (pd.DataFrame): Full dataframe including feature names.
        threshold (float): Variance threshold value.
        
    Returns:
        tuple: (X_reduced, selected_feature_names) where X_reduced is the filtered
               feature matrix and selected_feature_names are the names of selected features.
    """
    # Get the names of the indices from the first row
    indices_names = data.iloc[0, :-1]
    
    # Initialize the VarianceThreshold object
    var_threshold = VarianceThreshold(threshold=threshold)
    
    # Fit the VarianceThreshold to data
    var_threshold.fit(X)
    
    # Get the indices of the features to keep
    selected_features = var_threshold.get_support(indices=True)
    
    # Extract the names of the selected features using the first row of 'data'
    selected_feature_names = indices_names.iloc[selected_features].values
    
    # Create a DataFrame to display the selected feature indices and their corresponding names
    df_selected_features = pd.DataFrame({
        "Selected Feature Index": selected_features,
        "Feature Name": selected_feature_names
    })
    
    # Subset the original data with the selected features
    X_reduced = X.iloc[:, selected_features]
    
    print("Original data shape:", X.shape)
    print("Reduced data shape:", X_reduced.shape)
    print(f"Selected {len(selected_features)} features out of {X.shape[1]}")
    
    return X_reduced, selected_feature_names, df_selected_features

