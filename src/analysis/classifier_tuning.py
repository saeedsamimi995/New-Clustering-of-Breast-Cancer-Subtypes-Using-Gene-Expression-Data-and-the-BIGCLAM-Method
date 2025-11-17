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
    
    for params in ParameterGrid(param_grid):
        svm = SVC(**params, probability=True, random_state=42)
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
                        param_grid=None):
    """
    Tune MLP parameters using grid search on validation set.
    
    Args:
        X_train, X_valid, X_test: Features
        y_train, y_valid, y_test: Labels (will be one-hot encoded)
        param_grid: Parameter grid to search (default: common values)
        
    Returns:
        dict: Best parameters and results
    """
    if not MLP_AVAILABLE:
        print("[ERROR] MLP classifier not available")
        return None
    
    if param_grid is None:
        param_grid = {
            'learning_rate': [0.0001, 0.001, 0.01],
            'hidden_layers': [
                [64, 32],
                [80, 50, 20],
                [100, 50],
                [128, 64, 32]
            ],
            'dropout_rate': [0.2, 0.3, 0.4],
            'num_epochs': [150, 200, 250]
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
    
    for params in ParameterGrid(param_grid):
        print(f"\n  Testing: {params}")
        
        try:
            mlp_results = train_mlp(
                X_train, y_train_onehot, X_valid, y_valid_onehot, X_test, y_test_onehot,
                num_runs=3,  # Fewer runs for tuning
                num_epochs=params.get('num_epochs', 200),
                lr=params.get('learning_rate', 0.001),
                hidden_layers=tuple(params.get('hidden_layers', [80, 50, 20])),
                dropout_rate=params.get('dropout_rate', 0.3),
                patience=10,
                weight_decay=0.0001
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
    
    # Tune SVM
    print("\n" + "-"*80)
    print("SVM TUNING")
    print("-"*80)
    svm_results = tune_svm_parameters(X_train, y_train, X_valid, y_valid, X_test, y_test)
    
    # Tune MLP
    print("\n" + "-"*80)
    print("MLP TUNING")
    print("-"*80)
    mlp_results = tune_mlp_parameters(X_train, y_train, X_valid, y_valid, X_test, y_test)
    
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

