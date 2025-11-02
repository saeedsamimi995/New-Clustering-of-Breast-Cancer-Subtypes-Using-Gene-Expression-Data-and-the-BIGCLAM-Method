"""Data preprocessing including augmentation and splitting with memory optimization."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import gc


def augment_data(X, noise_mean=0.0, noise_std=0.1):
    """
    Augment data by adding random noise to balance classes.
    Memory-efficient version using batch augmentation.
    
    Args:
        X (pd.DataFrame): Input dataframe with features and labels (last column).
        noise_mean (float): Mean of noise distribution.
        noise_std (float): Standard deviation of noise distribution.
        
    Returns:
        pd.DataFrame: Augmented dataframe.
    """
    # Separate features and labels
    features = X.iloc[:, :-1].values
    labels = X.iloc[:, -1].values
    
    # Find the maximum number of samples among all classes
    max_samples = max(np.sum(labels == label) for label in set(labels))
    
    # Create a dictionary to store the indices of samples for each class
    class_indices = {label: np.where(labels == label)[0] for label in set(labels)}
    
    # Memory-efficient batch augmentation
    augmented_data_list = []
    
    # Augment each class to reach the maximum number of samples
    for label, indices in class_indices.items():
        # Calculate the number of augmentations needed for this class
        num_augmentations = max_samples - len(indices)
        
        if num_augmentations > 0:
            # Randomly sample existing examples from this class
            sampled_indices = np.random.choice(indices, num_augmentations, replace=True)
            
            # Batch augmentation for memory efficiency
            sampled_features = features[sampled_indices]
            noise = np.random.normal(noise_mean, noise_std, sampled_features.shape)
            augmented_features = sampled_features + noise
            augmented_labels = np.full((num_augmentations, 1), label)
            
            # Stack features and labels
            augmented_batch = np.hstack([augmented_features, augmented_labels])
            augmented_data_list.append(augmented_batch)
    
    # Convert the augmented data to a DataFrame
    if augmented_data_list:
        augmented_array = np.vstack(augmented_data_list)
        augmented_df = pd.DataFrame(augmented_array, columns=X.columns)
        # Concatenate the original and augmented data
        augmented_data = pd.concat([X, augmented_df], ignore_index=True)
        
        # Free memory
        del augmented_array, augmented_df
        gc.collect()
    else:
        augmented_data = X.copy()
    
    class_counts = augmented_data.iloc[:, -1].value_counts()
    print("Class counts after augmentation:\n", class_counts)
    
    return augmented_data


def split_data(augmented_data, test_size=0.2, valid_size=0.2, random_state=None):
    """
    Split data into train, validation, and test sets with stratified sampling.
    
    Args:
        augmented_data (pd.DataFrame): Dataframe with features and labels (last column).
        test_size (float): Proportion of data for test set.
        valid_size (float): Proportion of data for validation set.
        random_state (int, optional): Random state for reproducibility.
        
    Returns:
        tuple: (X_train, X_valid, X_test, y_train, y_valid, y_test) as numpy arrays.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize empty DataFrames for train, validation, and test sets
    X_train = pd.DataFrame()
    X_valid = pd.DataFrame()
    X_test = pd.DataFrame()
    
    # Iterate over each class
    for class_label in augmented_data.iloc[:, -1].unique():
        # Filter data for the current class
        class_data = augmented_data[augmented_data.iloc[:, -1] == class_label].copy()
        
        # Shuffle class data
        class_data = class_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Calculate the number of samples for each split
        total_samples = len(class_data)
        test_n = int(test_size * total_samples)
        valid_n = int(valid_size * total_samples)
        
        # Split the data
        test_data = class_data.iloc[:test_n]
        valid_data = class_data.iloc[test_n:test_n + valid_n]
        train_data = class_data.iloc[test_n + valid_n:]
        
        # Append to the respective sets
        X_test = pd.concat([X_test, test_data])
        X_valid = pd.concat([X_valid, valid_data])
        X_train = pd.concat([X_train, train_data])
    
    # Shuffle the datasets
    X_test = X_test.sample(frac=1, random_state=random_state).reset_index(drop=True)
    X_valid = X_valid.sample(frac=1, random_state=random_state).reset_index(drop=True)
    X_train = X_train.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Verify the splits
    print(f'Test set size: {X_test.shape}')
    print(f'Validation set size: {X_valid.shape}')
    print(f'Train set size: {X_train.shape}')
    
    # Define feature columns and target column for each set
    y_train = X_train.iloc[:, -1]
    y_valid = X_valid.iloc[:, -1]
    y_test = X_test.iloc[:, -1]
    
    X_train = X_train.iloc[:, :-1]
    X_valid = X_valid.iloc[:, :-1]
    X_test = X_test.iloc[:, :-1]
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def normalize_data(X_train, X_valid, X_test, feature_range=(-1, 1)):
    """
    Normalize data using MinMaxScaler.
    
    Args:
        X_train, X_valid, X_test: Training, validation, and test feature matrices.
        feature_range (tuple): Range for scaling.
        
    Returns:
        tuple: (X_train_norm, X_valid_norm, X_test_norm, scaler) normalized data and scaler.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    X_train_norm = scaler.fit_transform(X_train)
    X_valid_norm = scaler.transform(X_valid)
    X_test_norm = scaler.transform(X_test)
    
    return X_train_norm, X_valid_norm, X_test_norm, scaler


def encode_labels(y_train, y_valid, y_test):
    """
    One-hot encode labels.
    
    Args:
        y_train, y_valid, y_test: Label vectors.
        
    Returns:
        tuple: (y_train_onehot, y_valid_onehot, y_test_onehot, encoder) encoded labels and encoder.
    """
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train.values.reshape(-1, 1))
    y_valid_onehot = encoder.transform(y_valid.values.reshape(-1, 1))
    y_test_onehot = encoder.transform(y_test.values.reshape(-1, 1))
    
    return y_train_onehot, y_valid_onehot, y_test_onehot, encoder

