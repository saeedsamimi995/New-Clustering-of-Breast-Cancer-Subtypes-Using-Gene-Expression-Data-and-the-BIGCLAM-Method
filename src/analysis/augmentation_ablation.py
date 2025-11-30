"""
Data Augmentation Ablation Study

Compares classification performance with and without data augmentation.
Evaluates the impact of augmentation on model performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.classifiers.mlp_classifier import train_mlp
from src.classifiers.classifiers import split_data, encode_labels
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


def augment_data(X, y, noise_std=0.1):
    """
    Augment data by adding Gaussian noise.
    
    Args:
        X: Feature matrix
        y: Labels
        noise_std: Standard deviation of noise
        
    Returns:
        X_augmented, y_augmented
    """
    # Find max samples per class
    unique_labels, counts = np.unique(y, return_counts=True)
    #max_samples = counts.max()
    max_samples = int(counts.mean() + counts.std())
    
    X_augmented = [X]
    y_augmented = [y]
    
    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        n_current = len(label_indices)
        n_needed = max_samples - n_current
        
        if n_needed > 0:
            # Sample with replacement
            sampled_indices = np.random.choice(label_indices, n_needed, replace=True)
            
            # Add noise
            alpha = 0.2
            lam = np.random.beta(alpha, alpha)
            idx2 = np.random.choice(label_indices, n_needed, replace=True)
            X_new = lam * X[sampled_indices] + (1-lam) * X[idx2]
            y_new = y[sampled_indices]
            
            X_augmented.append(X_new)
            y_augmented.append(y_new)
    
    X_final = np.vstack(X_augmented)
    y_final = np.hstack(y_augmented)
    
    return X_final, y_final


def compare_with_without_augmentation(dataset_name, processed_dir='data/processed',
                                     clustering_dir='data/clusterings',
                                     output_dir='results/augmentation_ablation'):
    """
    Compare classification performance with and without augmentation.
    
    Args:
        dataset_name: Name of dataset
        processed_dir: Directory with processed data
        clustering_dir: Directory with clustering results
        output_dir: Output directory for results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"AUGMENTATION ABLATION STUDY: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Load data
    processed_file = Path(processed_dir) / f"{dataset_name}_processed.npy"
    clustering_file = Path(clustering_dir) / f"{dataset_name}_communities.npy"
    
    if not processed_file.exists():
        print(f"[ERROR] Processed file not found: {processed_file}")
        return None
    
    if not clustering_file.exists():
        print(f"[ERROR] Clustering file not found: {clustering_file}")
        return None
    
    # Load data
    X = np.load(processed_file)
    
    # Load targets
    target_file = processed_file.parent / processed_file.name.replace('_processed.npy', '_targets.pkl')
    with open(target_file, 'rb') as f:
        targets_data = pickle.load(f)
    
    y = targets_data['target_labels']
    
    # Encode labels
    y_encoded, label_encoder = encode_labels(y)
    
    # Show original class distribution
    unique, counts = np.unique(y_encoded, return_counts=True)
    print("Original Class Distribution:")
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({count/len(y_encoded)*100:.1f}%)")
    
    results = {}
    
    # ===== WITHOUT AUGMENTATION =====
    print("\n" + "-"*80)
    print("TRAINING WITHOUT AUGMENTATION")
    print("-"*80)
    
    X_train_no_aug, X_valid_no_aug, X_test_no_aug, y_train_no_aug, y_valid_no_aug, y_test_no_aug = split_data(
        X, y_encoded, test_size=0.2, valid_size=0.2, random_state=42
    )
    
    # Show training distribution
    unique_train, counts_train = np.unique(y_train_no_aug, return_counts=True)
    print("\nTraining Set Distribution (No Augmentation):")
    for label, count in zip(unique_train, counts_train):
        print(f"  Class {label}: {count} samples ({count/len(y_train_no_aug)*100:.1f}%)")
    
    # SVM without augmentation
    print("\n  Training SVM (no augmentation)...")
    start_time = time.time()
    svm_no_aug = SVC(kernel='rbf', C=0.1, gamma='scale', probability=True, random_state=42)
    svm_no_aug.fit(X_train_no_aug, y_train_no_aug)
    y_pred_svm_no_aug = svm_no_aug.predict(X_test_no_aug)
    svm_no_aug_time = time.time() - start_time
    
    acc_svm_no_aug = accuracy_score(y_test_no_aug, y_pred_svm_no_aug)
    f1_svm_no_aug = f1_score(y_test_no_aug, y_pred_svm_no_aug, average='macro')
    
    print(f"    Accuracy: {acc_svm_no_aug:.4f}")
    print(f"    F1-macro: {f1_svm_no_aug:.4f}")
    print(f"    Time: {svm_no_aug_time:.2f}s")
    
    # MLP without augmentation
    print("\n  Training MLP (no augmentation)...")
    start_time = time.time()
    mlp_no_aug_results = train_mlp(
        X_train_no_aug, y_train_no_aug, X_valid_no_aug, y_valid_no_aug,
        X_test_no_aug, y_test_no_aug,
        hidden_layers=[64, 32, 16],
        learning_rate=0.01,
        num_epochs=10000,
        patience=10
    )
    mlp_no_aug_time = time.time() - start_time
    
    print(f"    Accuracy: {mlp_no_aug_results['test_accuracy']:.4f}")
    print(f"    F1-macro: {mlp_no_aug_results.get('test_f1', 0):.4f}")
    print(f"    Time: {mlp_no_aug_time:.2f}s")
    
    results['no_augmentation'] = {
        'svm': {
            'accuracy': acc_svm_no_aug,
            'f1_macro': f1_svm_no_aug,
            'time': svm_no_aug_time
        },
        'mlp': {
            'accuracy': mlp_no_aug_results['test_accuracy'],
            'f1_macro': mlp_no_aug_results.get('test_f1', 0),
            'time': mlp_no_aug_time
        }
    }
    
    # ===== WITH AUGMENTATION =====
    print("\n" + "-"*80)
    print("TRAINING WITH AUGMENTATION")
    print("-"*80)
    
    # Augment training data only
    print("\n  Augmenting training data...")
    X_train_aug, y_train_aug = augment_data(X_train_no_aug, y_train_no_aug, noise_std=0.1)
    
    # Show augmented distribution
    unique_aug, counts_aug = np.unique(y_train_aug, return_counts=True)
    print("\nTraining Set Distribution (With Augmentation):")
    for label, count in zip(unique_aug, counts_aug):
        print(f"  Class {label}: {count} samples ({count/len(y_train_aug)*100:.1f}%)")
    
    # SVM with augmentation
    print("\n  Training SVM (with augmentation)...")
    start_time = time.time()
    svm_aug = SVC(kernel='rbf', C=0.1, gamma='scale', probability=True, random_state=42)
    svm_aug.fit(X_train_aug, y_train_aug)
    y_pred_svm_aug = svm_aug.predict(X_test_no_aug)  # Test on original test set
    svm_aug_time = time.time() - start_time
    
    acc_svm_aug = accuracy_score(y_test_no_aug, y_pred_svm_aug)
    f1_svm_aug = f1_score(y_test_no_aug, y_pred_svm_aug, average='macro')
    
    print(f"    Accuracy: {acc_svm_aug:.4f}")
    print(f"    F1-macro: {f1_svm_aug:.4f}")
    print(f"    Time: {svm_aug_time:.2f}s")
    
    # MLP with augmentation
    print("\n  Training MLP (with augmentation)...")
    start_time = time.time()
    mlp_aug_results = train_mlp(
        X_train_aug, y_train_aug, X_valid_no_aug, y_valid_no_aug,
        X_test_no_aug, y_test_no_aug,
        hidden_layers=[80, 50, 20],
        learning_rate=0.001,
        num_epochs=200,
        patience=10
    )
    mlp_aug_time = time.time() - start_time
    
    print(f"    Accuracy: {mlp_aug_results['test_accuracy']:.4f}")
    print(f"    F1-macro: {mlp_aug_results.get('test_f1', 0):.4f}")
    print(f"    Time: {mlp_aug_time:.2f}s")
    
    results['with_augmentation'] = {
        'svm': {
            'accuracy': acc_svm_aug,
            'f1_macro': f1_svm_aug,
            'time': svm_aug_time
        },
        'mlp': {
            'accuracy': mlp_aug_results['test_accuracy'],
            'f1_macro': mlp_aug_results.get('test_f1', 0),
            'time': mlp_aug_time
        }
    }
    
    # ===== COMPARISON =====
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        'Method': ['SVM', 'MLP'],
        'No Augmentation Accuracy': [
            results['no_augmentation']['svm']['accuracy'],
            results['no_augmentation']['mlp']['accuracy']
        ],
        'With Augmentation Accuracy': [
            results['with_augmentation']['svm']['accuracy'],
            results['with_augmentation']['mlp']['accuracy']
        ],
        'Improvement': [
            results['with_augmentation']['svm']['accuracy'] - results['no_augmentation']['svm']['accuracy'],
            results['with_augmentation']['mlp']['accuracy'] - results['no_augmentation']['mlp']['accuracy']
        ],
        'Improvement %': [
            ((results['with_augmentation']['svm']['accuracy'] - results['no_augmentation']['svm']['accuracy']) / 
             results['no_augmentation']['svm']['accuracy']) * 100,
            ((results['with_augmentation']['mlp']['accuracy'] - results['no_augmentation']['mlp']['accuracy']) / 
             results['no_augmentation']['mlp']['accuracy']) * 100
        ]
    })
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Save results
    results['comparison'] = comparison_df
    
    output_file = output_dir / f"{dataset_name}_augmentation_ablation.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    csv_file = output_dir / f"{dataset_name}_augmentation_ablation.csv"
    comparison_df.to_csv(csv_file, index=False)
    
    print(f"\n[Saved] Results: {output_file}")
    print(f"[Saved] CSV: {csv_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ablation study for data augmentation')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['tcga_brca_data', 'gse96058_data'],
                       help='Dataset to analyze')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                       help='Directory with processed data')
    parser.add_argument('--clustering_dir', type=str, default='data/clusterings',
                       help='Directory with clustering results')
    parser.add_argument('--output_dir', type=str, default='results/augmentation_ablation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    compare_with_without_augmentation(
        args.dataset,
        args.processed_dir,
        args.clustering_dir,
        args.output_dir
    )

