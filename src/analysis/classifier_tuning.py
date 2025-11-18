"""
Classifier Parameter Tuning Helper

Helps fine-tune MLP and SVM parameters for each dataset based on results.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import pickle
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

try:
    from ..classifiers.mlp_classifier import train_mlp
    from ..classifiers.classifiers import split_data, encode_labels
    MLP_AVAILABLE = True
except ImportError:
    MLP_AVAILABLE = False
    print("[Note] MLP classifier not available")


def tune_svm_parameters(X_train, y_train, X_valid, y_valid, X_test, y_test, 
                        param_grid=None):
    """
    Tune SVM parameters using grid search on validation set.
    
    Args:
        X_train, X_valid, X_test: Features
        y_train, y_valid, y_test: Labels
        param_grid: Parameter grid to search (default: common values)
        
    Returns:
        dict: Best parameters and results
    """
    if param_grid is None:
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear', 'poly']
        }
    
    print("\n[Tuning SVM] Grid search over parameters...")
    print(f"    Parameter combinations: {len(list(ParameterGrid(param_grid)))}")
    
    best_score = 0
    best_params = None
    best_model = None
    results = []
    
    # Calculate class weights for balanced training
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = dict(zip(classes, class_weights))
    
    for params in ParameterGrid(param_grid):
        svm = SVC(**params, probability=True, class_weight=class_weight, random_state=42)
        svm.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_valid_pred = svm.predict(X_valid)
        valid_acc = accuracy_score(y_valid, y_valid_pred)
        valid_f1 = f1_score(y_valid, y_valid_pred, average='macro')
        
        # Also check test set for reference (but don't use for selection)
        y_test_pred = svm.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='macro')
        
        results.append({
            'params': params,
            'valid_accuracy': valid_acc,
            'valid_f1': valid_f1,
            'test_accuracy': test_acc,
            'test_f1': test_f1
        })
        
        # Select best based on validation F1 (macro)
        if valid_f1 > best_score:
            best_score = valid_f1
            best_params = params
            best_model = svm
    
    print(f"\n[Best SVM Parameters]")
    print(f"    Validation F1: {best_score:.4f}")
    print(f"    Parameters: {best_params}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_model': best_model,
        'all_results': results
    }


def tune_mlp_parameters(X_train, y_train, X_valid, y_valid, X_test, y_test,
                        param_grid=None, dataset_name=None):
    """
    Tune MLP parameters using grid search on validation set.
    
    Args:
        X_train, X_valid, X_test: Features
        y_train, y_valid, y_test: Labels (will be one-hot encoded)
        param_grid: Parameter grid to search (default: dataset-specific values)
        dataset_name: Name of dataset (for dataset-specific parameter grids)
        
    Returns:
        dict: Best parameters and results
    """
    if not MLP_AVAILABLE:
        print("[ERROR] MLP classifier not available")
        return None
    
    if param_grid is None:
        # Determine input size
        input_size = X_train.shape[1]
        
        # Dataset-specific parameter grids
        if dataset_name == 'gse96058_data' or input_size > 500:
            # Deep architectures for large feature spaces (e.g., 843 features)
            param_grid = {
                'learning_rate': [0.001, 0.005, 0.01],
                'hidden_layers': [
                    [512, 256, 128, 64, 32, 16],  # Deep architecture
                    [512, 256, 128, 64],          # Medium-deep
                    [256, 128, 64, 32],           # Medium
                    [128, 64, 32],                # Shallow
                    [256, 128, 64],               # Alternative medium
                ],
                'dropout_rate': [0.2, 0.3, 0.4],
                'num_epochs': [10000]
            }
        else:
            # Standard architectures for smaller feature spaces (e.g., TCGA)
            param_grid = {
                'learning_rate': [0.001, 0.005, 0.01],
                'hidden_layers': [
                    [64, 32],
                    [64, 32, 16],
                    [100, 50],
                    [128, 64, 32],
                    [128, 64, 32, 16]  # Slightly deeper option
                ],
                'dropout_rate': [0.2, 0.3, 0.4],
                'num_epochs': [10000]
            }
    
    print("\n[Tuning MLP] Grid search over parameters...")
    print(f"    Parameter combinations: {len(list(ParameterGrid(param_grid)))}")
    
    # One-hot encode labels
    try:
        oe = OneHotEncoder(sparse_output=False)
    except TypeError:
        oe = OneHotEncoder(sparse=False)
    y_train_onehot = oe.fit_transform(y_train.reshape(-1, 1))
    y_valid_onehot = oe.transform(y_valid.reshape(-1, 1))
    y_test_onehot = oe.transform(y_test.reshape(-1, 1))
    
    best_score = 0
    best_params = None
    results = []
    
    # Augment training data for balanced classes
    print("\n[Augmenting training data for balanced classes...]")
    from src.analysis.augmentation_ablation import augment_data
    X_train_aug, y_train_aug = augment_data(X_train, y_train, noise_std=0.1)
    
    # One-hot encode augmented training data
    y_train_aug_onehot = oe.fit_transform(y_train_aug.reshape(-1, 1))
    
    for params in ParameterGrid(param_grid):
        print(f"\n  Testing: {params}")
        
        try:
            mlp_results = train_mlp(
                X_train_aug, y_train_aug_onehot, X_valid, y_valid_onehot, X_test, y_test_onehot,
                num_runs=3,  # Fewer runs for tuning
                num_epochs=params.get('num_epochs', 200),
                lr=params.get('learning_rate', 0.001),
                hidden_layers=tuple(params.get('hidden_layers', [80, 50, 20])),
                dropout_rate=params.get('dropout_rate', 0.3),
                weight_decay=0.0001,
                use_class_weights=True,  # Also use class weights
                lr_scheduler_factor=0.8,
                lr_scheduler_patience=20,
                lr_scheduler_min_lr=0.001,
                use_warm_restarts=False,  # Keep ReduceLROnPlateau for tuning
                warm_restart_T_0=100,
                warm_restart_T_mult=2,
                gradient_clip=1.0
            )
            
            # Get validation accuracy from best run
            best_run_idx = mlp_results.get('best_run_idx', 0)
            valid_cm = mlp_results['valid_cms'][best_run_idx]
            valid_acc = np.trace(valid_cm) / valid_cm.sum()
            
            test_cm = mlp_results['test_cms'][best_run_idx]
            test_acc = np.trace(test_cm) / test_cm.sum()
            
            results.append({
                'params': params,
                'valid_accuracy': valid_acc,
                'test_accuracy': test_acc
            })
            
            if valid_acc > best_score:
                best_score = valid_acc
                best_params = params
                
        except Exception as e:
            print(f"    [ERROR] Failed: {e}")
            continue
    
    print(f"\n[Best MLP Parameters]")
    print(f"    Validation Accuracy: {best_score:.4f}")
    print(f"    Parameters: {best_params}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results
    }


def tune_classifiers_for_dataset(dataset_name, processed_dir='data/processed',
                                 clustering_dir='data/clusterings',
                                 output_dir='results/classifier_tuning'):
    """
    Tune classifier parameters for a specific dataset.
    
    Args:
        dataset_name: Name of dataset (e.g., 'tcga_brca_data', 'gse96058_data')
        processed_dir: Directory with processed data
        clustering_dir: Directory with clustering results
        output_dir: Output directory for tuning results
    """
    processed_dir = Path(processed_dir)
    clustering_dir = Path(clustering_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"TUNING CLASSIFIERS: {dataset_name}")
    print("="*80)
    
    # Load data
    expression_file = processed_dir / f"{dataset_name}_processed.npy"
    clustering_file = clustering_dir / f"{dataset_name}_communities.npy"
    
    if not expression_file.exists():
        print(f"[ERROR] Expression file not found: {expression_file}")
        return None
    
    if not clustering_file.exists():
        print(f"[ERROR] Clustering file not found: {clustering_file}")
        return None
    
    expression_data = np.load(expression_file)
    communities = np.load(clustering_file)
    
    # Convert communities to 1D if needed
    if communities.ndim == 2:
        communities = np.argmax(communities, axis=1)
    communities = communities.flatten()
    
    # Encode labels
    communities_encoded, le = encode_labels(communities)
    
    X = expression_data
    y = communities_encoded
    
    # Split data
    from ..classifiers.classifiers import split_data
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        X, y, test_size=0.2, valid_size=0.2
    )
    
    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Communities: {len(set(communities))}")
    
    # Augment training data for balanced classes (used for both SVM and MLP)
    print("\n[Augmenting training data for balanced classes...]")
    from src.analysis.augmentation_ablation import augment_data
    X_train_aug, y_train_aug = augment_data(X_train, y_train, noise_std=0.1)
    
    # Show augmented distribution
    unique_aug, counts_aug = np.unique(y_train_aug, return_counts=True)
    print("\nAugmented class distribution:")
    for label, count in zip(unique_aug, counts_aug):
        print(f"  Class {label}: {count} samples ({count/len(y_train_aug)*100:.1f}%)")
    
    # Tune SVM
    print("\n" + "-"*80)
    print("SVM TUNING")
    print("-"*80)
    svm_results = tune_svm_parameters(X_train_aug, y_train_aug, X_valid, y_valid, X_test, y_test)
    
    # Tune MLP
    print("\n" + "-"*80)
    print("MLP TUNING")
    print("-"*80)
    mlp_results = tune_mlp_parameters(X_train, y_train, X_valid, y_valid, X_test, y_test,
                                     dataset_name=dataset_name)
    
    # Save results
    results = {
        'dataset': dataset_name,
        'svm': svm_results,
        'mlp': mlp_results,
        'n_communities': len(set(communities))
    }
    
    with open(output_dir / f"{dataset_name}_tuning_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    # Create summary
    summary = {
        'dataset': dataset_name,
        'n_communities': len(set(communities)),
        'best_svm_params': svm_results['best_params'] if svm_results else None,
        'best_svm_score': svm_results['best_score'] if svm_results else None,
        'best_mlp_params': mlp_results['best_params'] if mlp_results else None,
        'best_mlp_score': mlp_results['best_score'] if mlp_results else None
    }
    
    # Save as YAML for easy config update
    summary_yaml = {
        'svm': svm_results['best_params'] if svm_results else {},
        'mlp': mlp_results['best_params'] if mlp_results else {}
    }
    
    with open(output_dir / f"{dataset_name}_best_params.yaml", 'w') as f:
        yaml.dump(summary_yaml, f, default_flow_style=False)
    
    print(f"\n[Saved] Tuning results to: {output_dir}")
    print(f"        Best parameters saved as YAML for easy config update")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune classifier parameters')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['tcga_brca_data', 'gse96058_data'],
                       help='Dataset to tune')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                       help='Processed data directory')
    parser.add_argument('--clustering_dir', type=str, default='data/clusterings',
                       help='Clustering directory')
    parser.add_argument('--output_dir', type=str, default='results/classifier_tuning',
                       help='Output directory')
    
    args = parser.parse_args()
    
    tune_classifiers_for_dataset(
        args.dataset,
        args.processed_dir,
        args.clustering_dir,
        args.output_dir
    )

