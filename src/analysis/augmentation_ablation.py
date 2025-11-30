"""
Data Augmentation Ablation Study

Compares classification performance with and without data augmentation.
Uses SMOTE (Synthetic Minority Oversampling Technique) for biologically plausible augmentation.
Evaluates the impact of augmentation on model performance and validates distribution preservation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys
import time
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.classifiers.mlp_classifier import train_mlp
from src.classifiers.classifiers import split_data, encode_labels
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARNING] imbalanced-learn not available. Install with: pip install imbalanced-learn")


def validate_distribution_preservation(X_original, X_augmented, feature_indices=None, alpha=0.05):
    """
    Validate that augmentation preserves the original data distribution.
    
    Uses Kolmogorov-Smirnov test to compare distributions of original vs augmented data.
    
    Args:
        X_original: Original feature matrix
        X_augmented: Augmented feature matrix (includes original + synthetic)
        feature_indices: Indices of features to test (None = test all, or sample subset)
        alpha: Significance level for KS test
        
    Returns:
        dict: Validation results including p-values and pass/fail status
    """
    # Extract only synthetic samples (augmented data minus original)
    n_original = len(X_original)
    X_synthetic = X_augmented[n_original:]
    
    if len(X_synthetic) == 0:
        return {'status': 'no_synthetic_samples', 'p_values': [], 'passed': True}
    
    # Sample features if too many (test up to 100 features for efficiency)
    if feature_indices is None:
        n_features = X_original.shape[1]
        if n_features > 100:
            feature_indices = np.random.choice(n_features, 100, replace=False)
        else:
            feature_indices = np.arange(n_features)
    
    p_values = []
    passed_features = 0
    
    for feat_idx in feature_indices:
        # KS test: compare distribution of original vs synthetic for this feature
        ks_stat, p_value = stats.ks_2samp(
            X_original[:, feat_idx],
            X_synthetic[:, feat_idx]
        )
        p_values.append(p_value)
        
        # Feature passes if p > alpha (distributions are not significantly different)
        if p_value > alpha:
            passed_features += 1
    
    # Overall validation: at least 80% of features should preserve distribution
    pass_rate = passed_features / len(feature_indices)
    passed = pass_rate >= 0.80
    
    return {
        'status': 'passed' if passed else 'failed',
        'p_values': p_values,
        'mean_p_value': np.mean(p_values),
        'median_p_value': np.median(p_values),
        'pass_rate': pass_rate,
        'passed_features': passed_features,
        'total_features_tested': len(feature_indices),
        'passed': passed
    }


def augment_data(X, y, method='smote', k_neighbors=5, random_state=42, noise_std=None):
    """
    Augment data using SMOTE (Synthetic Minority Oversampling Technique).
    
    SMOTE generates synthetic samples by interpolating between existing samples
    in the feature space, which is more biologically plausible than Gaussian noise
    for gene expression data.
    
    Args:
        X: Feature matrix (training data only)
        y: Labels (training data only)
        method: Augmentation method ('smote' or 'smote_nc' for nominal/continuous)
        k_neighbors: Number of nearest neighbors for SMOTE
        random_state: Random seed for reproducibility
        noise_std: DEPRECATED - kept for backward compatibility, ignored
        
    Returns:
        X_augmented, y_augmented: Augmented data (includes original + synthetic)
        validation_results: Distribution validation results
    """
    if not SMOTE_AVAILABLE:
        raise ImportError("imbalanced-learn is required for SMOTE. Install with: pip install imbalanced-learn")
    
    # Store original data
    X_original = X.copy()
    y_original = y.copy()
    
    # Find target sample size (mean + 1 std, or max if very imbalanced)
    unique_labels, counts = np.unique(y, return_counts=True)
    mean_count = counts.mean()
    std_count = counts.std()
    max_samples = int(mean_count + std_count)
    
    # Ensure we don't oversample too much (cap at 2x the mean)
    max_samples = min(max_samples, int(mean_count * 2))
    
    print(f"  Target samples per class: {max_samples}")
    print(f"  Current class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"    Class {label}: {count} samples")
    
    # Calculate desired sampling strategy
    # We want each class to have at least max_samples samples
    sampling_strategy_dict = {}
    for label, count in zip(unique_labels, counts):
        if count < max_samples:
            sampling_strategy_dict[label] = max_samples
        else:
            # Don't oversample classes that already have enough samples
            sampling_strategy_dict[label] = count
    
    # Ensure k < min class size
    min_class_size = min(counts)
    k_neighbors = min(k_neighbors, min_class_size - 1) if min_class_size > 1 else 1
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy_dict,
        k_neighbors=k_neighbors,
        random_state=random_state
    )
    
    try:
        X_augmented, y_augmented = smote.fit_resample(X, y)
    except ValueError as e:
        # Fallback: if SMOTE fails (e.g., too few samples), return original
        print(f"  [WARNING] SMOTE failed: {e}")
        print("  [FALLBACK] Using original data without augmentation")
        return X_original, y_original, {'status': 'smote_failed', 'passed': False}
    
    # Validate distribution preservation
    validation_results = validate_distribution_preservation(X_original, X_augmented)
    
    # Show results
    unique_aug, counts_aug = np.unique(y_augmented, return_counts=True)
    print(f"\n  Augmented class distribution:")
    for label, count in zip(unique_aug, counts_aug):
        print(f"    Class {label}: {count} samples")
    
    print(f"\n  Distribution validation:")
    print(f"    Status: {validation_results['status']}")
    print(f"    Pass rate: {validation_results['pass_rate']:.2%}")
    print(f"    Mean p-value: {validation_results['mean_p_value']:.4f}")
    print(f"    Median p-value: {validation_results['median_p_value']:.4f}")
    
    if not validation_results['passed']:
        print(f"  [WARNING] Distribution validation failed. Augmentation may distort data.")
    
    return X_augmented, y_augmented, validation_results


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
    
    # Augment training data only (NOT validation or test)
    print("\n  Augmenting training data using SMOTE...")
    print("  [IMPORTANT] Augmentation applied ONLY to training data, not validation/test")
    
    X_train_aug, y_train_aug, validation_results = augment_data(
        X_train_no_aug, y_train_no_aug, 
        method='smote', 
        k_neighbors=5, 
        random_state=42
    )
    
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
        },
        'validation_results': validation_results  # Distribution validation
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
    
    # Print distribution validation summary
    if 'validation_results' in results['with_augmentation']:
        val_results = results['with_augmentation']['validation_results']
        print(f"\nDistribution Validation:")
        print(f"  Status: {val_results['status']}")
        print(f"  Pass rate: {val_results['pass_rate']:.2%}")
        print(f"  Mean p-value: {val_results['mean_p_value']:.4f}")
    
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
