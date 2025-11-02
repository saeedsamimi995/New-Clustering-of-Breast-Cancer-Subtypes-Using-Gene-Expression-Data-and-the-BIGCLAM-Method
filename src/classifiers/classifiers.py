"""
Classification Module for Validation

Uses MLP and SVM to validate clustering by predicting labels from BIGCLAM community assignments.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from .svm_classifier import train_svm

try:
    from .mlp_classifier import train_mlp
    MLP_AVAILABLE = True
except ImportError:
    MLP_AVAILABLE = False
    print("[Note] MLP classifier not available")


def encode_labels(target_labels):
    """
    Encode labels to numerical format.
    
    Args:
        target_labels: List of string labels
        
    Returns:
        tuple: (encoded_labels, label_encoder)
    """
    le = LabelEncoder()
    encoded = le.fit_transform([str(lbl) for lbl in target_labels])
    return encoded, le


def split_data(X, y, test_size=0.2, valid_size=0.2, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Features
        y: Labels
        test_size: Test set proportion
        valid_size: Validation set proportion
        random_state: Random seed
        
    Returns:
        tuple: (X_train, X_valid, X_test, y_train, y_valid, y_test)
    """
    # First split: train+valid vs test
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs valid
    valid_ratio = valid_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=valid_ratio, 
        random_state=random_state, stratify=y_train_valid
    )
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def train_mlp_classifier(X_train, y_train, X_valid, y_valid, X_test, y_test, **params):
    """
    Train and evaluate MLP classifier.
    
    Args:
        X_train, X_valid, X_test: Features
        y_train, y_valid, y_test: Labels
        params: MLP parameters
        
    Returns:
        dict: Results
    """
    if not MLP_AVAILABLE:
        print("[ERROR] MLP classifier not available")
        return None
    
    print("\n[Training MLP]...")
    
    # One-hot encode labels
    oe = OneHotEncoder(sparse=False)
    y_train_onehot = oe.fit_transform(y_train.reshape(-1, 1))
    y_valid_onehot = oe.transform(y_valid.reshape(-1, 1))
    y_test_onehot = oe.transform(y_test.reshape(-1, 1))
    
    # Default parameters
    num_runs = params.get('num_runs', 10)
    num_epochs = params.get('num_epochs', 200)
    lr = params.get('lr', 0.001)
    hidden_layers = params.get('hidden_layers', (80, 50, 20))
    
    print(f"    Parameters: runs={num_runs}, epochs={num_epochs}, lr={lr}")
    
    try:
        results = train_mlp(
            X_train, y_train_onehot, X_valid, y_valid_onehot, X_test, y_test_onehot,
            num_runs=num_runs,
            num_epochs=num_epochs,
            lr=lr,
            hidden_layers=hidden_layers
        )
        
        train_cm, valid_cm, test_cm = results[0], results[1], results[2]
        
        # Calculate accuracies
        train_acc = np.trace(train_cm) / train_cm.sum()
        valid_acc = np.trace(valid_cm) / valid_cm.sum()
        test_acc = np.trace(test_cm) / test_cm.sum()
        
        print(f"\n[MLP Results]")
        print(f"    Train Accuracy: {train_acc:.4f}")
        print(f"    Valid Accuracy: {valid_acc:.4f}")
        print(f"    Test Accuracy: {test_acc:.4f}")
        
        return {
            'model': 'MLP',
            'train_accuracy': train_acc,
            'valid_accuracy': valid_acc,
            'test_accuracy': test_acc,
            'train_cm': train_cm,
            'valid_cm': valid_cm,
            'test_cm': test_cm,
            'results': results
        }
        
    except Exception as e:
        print(f"[ERROR] MLP training failed: {e}")
        return None


def create_classification_plots(svm_results, mlp_results, label_encoder, output_dir, dataset_name):
    """
    Create confusion matrices and ROC curves for classifiers.
    
    Args:
        svm_results: SVM results dict
        mlp_results: MLP results dict
        label_encoder: Label encoder
        output_dir: Output directory
        dataset_name: Dataset name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    label_names = label_encoder.classes_
    n_classes = len(label_names)
    
    # Plot confusion matrices
    for classifier_name, results in [('SVM', svm_results), ('MLP', mlp_results)]:
        if results is None:
            continue
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (split, cm) in enumerate([('Train', results['train_cm']), 
                                           ('Valid', results['valid_cm']),
                                           ('Test', results['test_cm'])]):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=label_names[:n_classes], 
                       yticklabels=label_names[:n_classes])
            axes[idx].set_title(f'{classifier_name} - {split}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name.lower()}_{classifier_name.lower()}_confusion_matrix.png", 
                   dpi=300, bbox_inches='tight')
        print(f"    [Saved] {dataset_name.lower()}_{classifier_name.lower()}_confusion_matrix.png")
        plt.close()


def validate_clustering_with_classifiers(processed_dir='data/processed',
                                        clustering_dir='data/clusterings',
                                        output_dir='results/classification',
                                        mlp_params=None, svm_params=None):
    """
    Validate clustering by training classifiers to predict labels from communities.
    
    Args:
        processed_dir: Directory with processed data
        clustering_dir: Directory with clustering results
        output_dir: Output directory
        mlp_params: MLP parameters
        svm_params: SVM parameters
    """
    processed_dir = Path(processed_dir)
    clustering_dir = Path(clustering_dir)
    
    # Find all clustering files
    clustering_files = list(clustering_dir.glob('*_communities.npy'))
    
    if not clustering_files:
        print(f"No clustering files found in {clustering_dir}")
        return
    
    # Default parameters
    if mlp_params is None:
        mlp_params = {'num_runs': 10, 'num_epochs': 200, 'lr': 0.001, 'hidden_layers': (80, 50, 20)}
    if svm_params is None:
        svm_params = {'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale'}
    
    for clustering_file in clustering_files:
        dataset_name = clustering_file.stem.replace('_communities', '')
        
        print("\n" + "="*80)
        print(f"VALIDATING: {dataset_name}")
        print("="*80)
        
        # Load communities
        communities = np.load(clustering_file)
        
        # Load targets
        target_file = processed_dir / f"{dataset_name}_targets.pkl"
        if not target_file.exists():
            print(f"[SKIP] No target file: {target_file}")
            continue
        
        with open(target_file, 'rb') as f:
            targets_data = pickle.load(f)
        
        target_labels = targets_data['target_labels']
        
        # Encode labels
        labels_encoded, le = encode_labels(target_labels)
        
        # Use communities as features
        # Convert communities to one-hot encoding
        from sklearn.preprocessing import OneHotEncoder
        oe = OneHotEncoder(sparse=False)
        communities_onehot = oe.fit_transform(communities.reshape(-1, 1))
        
        # Use community memberships if available
        membership_file = clustering_dir / f"{dataset_name}_communities_membership.npy"
        if membership_file.exists():
            membership_matrix = np.load(membership_file)
            X = membership_matrix  # Use membership matrix as features
            print(f"    Using membership matrix as features: {membership_matrix.shape}")
        else:
            X = communities_onehot
            print(f"    Using one-hot encoded communities as features: {X.shape}")
        
        # Split data
        X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
            X, labels_encoded, test_size=0.2, valid_size=0.2
        )
        
        # Train classifiers
        svm_results = train_svm(X_train, y_train, X_valid, y_valid, X_test, y_test, **svm_params)
        mlp_results = train_mlp_classifier(X_train, y_train, X_valid, y_valid, X_test, y_test, **mlp_params)
        
        # Create plots
        create_classification_plots(svm_results, mlp_results, le, output_dir, dataset_name)
        
        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / f"{dataset_name}_classification_results.pkl", 'wb') as f:
            pickle.dump({
                'svm': svm_results,
                'mlp': mlp_results,
                'label_encoder': le
            }, f)
        
        print(f"[Saved] Classification results to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate clustering with classifiers')
    parser.add_argument('--processed_dir', type=str, default='data/processed', help='Processed data directory')
    parser.add_argument('--clustering_dir', type=str, default='data/clusterings', help='Clustering directory')
    parser.add_argument('--output_dir', type=str, default='results/classification', help='Output directory')
    
    args = parser.parse_args()
    
    validate_clustering_with_classifiers(
        args.processed_dir,
        args.clustering_dir,
        args.output_dir
    )

