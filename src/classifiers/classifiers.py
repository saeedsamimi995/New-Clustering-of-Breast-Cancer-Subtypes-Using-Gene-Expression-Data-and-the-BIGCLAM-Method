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
    
    # Check if augmentation is enabled
    use_augmentation = params.get('use_augmentation', True)
    noise_std = params.get('augmentation_noise_std', 0.1)
    use_class_weights = params.get('use_class_weights', True)
    
    if use_augmentation:
        print("\n[Augmenting training data to balance classes...]")
        from src.analysis.augmentation_ablation import augment_data
        
        # Show original distribution
        unique_orig, counts_orig = np.unique(y_train, return_counts=True)
        print("Original class distribution:")
        for label, count in zip(unique_orig, counts_orig):
            print(f"  Class {label}: {count} samples ({count/len(y_train)*100:.1f}%)")
        
        # Augment training data
        X_train_aug, y_train_aug = augment_data(X_train, y_train, noise_std=noise_std)
        
        # Show augmented distribution
        unique_aug, counts_aug = np.unique(y_train_aug, return_counts=True)
        print("\nAugmented class distribution:")
        for label, count in zip(unique_aug, counts_aug):
            print(f"  Class {label}: {count} samples ({count/len(y_train_aug)*100:.1f}%)")
        
        # Use augmented data for training
        X_train_final = X_train_aug
        y_train_final = y_train_aug
    else:
        X_train_final = X_train
        y_train_final = y_train
    
    # One-hot encode labels (use augmented data if augmentation was used)
    # Use sparse_output for newer sklearn versions, fallback to sparse for older versions
    try:
        oe = OneHotEncoder(sparse_output=False)
    except TypeError:
        # Fallback for older sklearn versions
        oe = OneHotEncoder(sparse=False)
    y_train_onehot = oe.fit_transform(y_train_final.reshape(-1, 1))
    y_valid_onehot = oe.transform(y_valid.reshape(-1, 1))
    y_test_onehot = oe.transform(y_test.reshape(-1, 1))
    
    # Default parameters (handle both 'lr' and 'learning_rate')
    num_runs = params.get('num_runs', 10)
    num_epochs = params.get('num_epochs', 200)
    lr = params.get('lr', params.get('learning_rate', 0.001))
    hidden_layers = tuple(params.get('hidden_layers', [80, 50, 20]))  # Convert list to tuple
    weight_decay = params.get('weight_decay', 0.0001)
    dropout_rate = params.get('dropout_rate', 0.3)
    lr_scheduler_factor = params.get('lr_scheduler_factor', 0.8)
    lr_scheduler_patience = params.get('lr_scheduler_patience', 20)
    lr_scheduler_min_lr = params.get('lr_scheduler_min_lr', 0.001)
    use_warm_restarts = params.get('use_warm_restarts', False)
    warm_restart_T_0 = params.get('warm_restart_T_0', 50)
    warm_restart_T_mult = params.get('warm_restart_T_mult', 2)
    gradient_clip = params.get('gradient_clip', 1.0)
    
    print(f"\n    Parameters: runs={num_runs}, epochs={num_epochs}, lr={lr}")
    print(f"                hidden_layers={hidden_layers}, dropout={dropout_rate}, weight_decay={weight_decay}")
    print(f"                use_augmentation={use_augmentation}, use_class_weights={use_class_weights}")
    if use_warm_restarts:
        print(f"                lr_scheduler: CosineAnnealingWarmRestarts (T_0={warm_restart_T_0}, T_mult={warm_restart_T_mult}, eta_min={lr_scheduler_min_lr})")
    else:
        print(f"                lr_scheduler: ReduceLROnPlateau (factor={lr_scheduler_factor}, patience={lr_scheduler_patience}, min_lr={lr_scheduler_min_lr})")
    print(f"                gradient_clip: {gradient_clip}")
    
    try:
        results = train_mlp(
            X_train_final, y_train_onehot, X_valid, y_valid_onehot, X_test, y_test_onehot,
            num_runs=num_runs,
            num_epochs=num_epochs,
            lr=lr,
            hidden_layers=hidden_layers,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            use_class_weights=use_class_weights,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_min_lr=lr_scheduler_min_lr,
            use_warm_restarts=use_warm_restarts,
            warm_restart_T_0=warm_restart_T_0,
            warm_restart_T_mult=warm_restart_T_mult,
            gradient_clip=gradient_clip
        )
        
        # Get best run results
        best_run_idx = results.get('best_run_idx', 0)
        train_cm = results['train_cms'][best_run_idx]
        valid_cm = results['valid_cms'][best_run_idx]
        test_cm = results['test_cms'][best_run_idx]
        
        # Get probability outputs (softmax probabilities)
        test_outputs = results['best_test_outputs']
        # Convert to numpy if it's a torch tensor
        try:
            import torch
            if hasattr(test_outputs, 'numpy'):  # Check if it's a torch tensor
                test_proba = test_outputs.numpy()
            else:
                test_proba = np.array(test_outputs) if test_outputs is not None else None
        except (ImportError, AttributeError):
            # If torch not available or not a tensor, assume numpy array
            test_proba = np.array(test_outputs) if test_outputs is not None else None
        
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
            'test_proba': test_proba,
            'y_test': y_test,
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
        
        # Confusion matrices
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (split, cm) in enumerate([('Train', results['train_cm']), 
                                           ('Valid', results['valid_cm']),
                                           ('Test', results['test_cm'])]):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=label_names[:n_classes], 
                       yticklabels=label_names[:n_classes],
                       cbar_kws={'label': 'Count'})
            axes[idx].set_title(f'{classifier_name} - {split} Set', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted Community', fontsize=10)
            axes[idx].set_ylabel('True Community', fontsize=10)
        
        plt.suptitle(f'{classifier_name} Confusion Matrices - {dataset_name}', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name.lower()}_{classifier_name.lower()}_confusion_matrix.tiff", 
                   dpi=300, bbox_inches='tight', format='tiff')
        print(f"    [Saved] {dataset_name.lower()}_{classifier_name.lower()}_confusion_matrix.tiff")
        plt.close()
        
        # ROC curves (only if we have probability predictions)
        if 'y_test_proba' in results or 'test_proba' in results or 'y_test' in results:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get probabilities
            if 'y_test_proba' in results:
                y_test_proba = results['y_test_proba']
                y_test = results.get('y_test', None)
            elif 'test_proba' in results:
                y_test_proba = results['test_proba']
                y_test = results.get('y_test', None)
            else:
                y_test_proba = None
                y_test = None
            
            # For MLP, test_proba might be numpy array
            if y_test_proba is not None:
                if hasattr(y_test_proba, 'numpy'):
                    y_test_proba = y_test_proba.numpy()
                y_test_proba = np.array(y_test_proba)
            
            if y_test_proba is not None and y_test is not None:
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import roc_curve, auc, roc_auc_score
                
                # Binarize labels for multi-class ROC
                y_test_bin = label_binarize(y_test, classes=range(n_classes))
                
                if n_classes == 2:
                    # Binary classification
                    fpr, tpr, _ = roc_curve(y_test, y_test_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
                else:
                    # Multi-class: Compute ROC for each class
                    for i in range(n_classes):
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, lw=2, 
                               label=f'Community {label_names[i]} (AUC = {roc_auc:.3f})')
                    
                    # Compute macro-average ROC
                    fpr_macro, tpr_macro, _ = roc_curve(y_test_bin.ravel(), y_test_proba.ravel())
                    roc_auc_macro = auc(fpr_macro, tpr_macro)
                    ax.plot(fpr_macro, tpr_macro, lw=2, linestyle='--',
                           label=f'Macro-average (AUC = {roc_auc_macro:.3f})')
                
                ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
                ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
                ax.set_title(f'{classifier_name} ROC Curves - {dataset_name}', 
                           fontsize=14, fontweight='bold')
                ax.legend(loc="lower right", fontsize=10)
                ax.grid(alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / f"{dataset_name.lower()}_{classifier_name.lower()}_roc_curve.tiff", 
                           dpi=300, bbox_inches='tight', format='tiff')
                print(f"    [Saved] {dataset_name.lower()}_{classifier_name.lower()}_roc_curve.tiff")
                plt.close()
            
            # Store AUC in results
            if y_test_proba is not None and y_test is not None:
                try:
                    if n_classes == 2:
                        results['test_auc'] = roc_auc_score(y_test, y_test_proba[:, 1])
                    else:
                        results['test_auc_macro'] = roc_auc_score(y_test, y_test_proba, 
                                                                  multi_class='ovr', average='macro')
                        results['test_auc_weighted'] = roc_auc_score(y_test, y_test_proba, 
                                                                     multi_class='ovr', average='weighted')
                except:
                    pass


def validate_clustering_with_classifiers(processed_dir='data/processed',
                                        clustering_dir='data/clusterings',
                                        output_dir='results/classification',
                                        mlp_params=None, svm_params=None,
                                        dataset_specific_params=None):
    """
    Validate clustering by training classifiers to predict communities from expression data.
    
    Uses expression data as features and clustering results (communities) as targets.
    This validates that the discovered communities are learnable from expression profiles.
    
    Args:
        processed_dir: Directory with processed data
        clustering_dir: Directory with clustering results
        output_dir: Output directory
        mlp_params: Default MLP parameters (used if dataset-specific not provided)
        svm_params: Default SVM parameters (used if dataset-specific not provided)
        dataset_specific_params: Dict mapping dataset names to their specific parameters
                               Format: {dataset_name: {'mlp': {...}, 'svm': {...}}}
    """
    processed_dir = Path(processed_dir)
    clustering_dir = Path(clustering_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all clustering files
    clustering_files = list(clustering_dir.glob('*_communities.npy'))
    
    if not clustering_files:
        print(f"No clustering files found in {clustering_dir}")
        return
    
    # Default parameters (fallback)
    default_mlp_params = {'num_runs': 10, 'num_epochs': 200, 'lr': 0.001, 'hidden_layers': (80, 50, 20)}
    default_svm_params = {'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale'}
    
    if mlp_params is None:
        mlp_params = default_mlp_params
    if svm_params is None:
        svm_params = default_svm_params
    if dataset_specific_params is None:
        dataset_specific_params = {}
    
    for clustering_file in clustering_files:
        dataset_name = clustering_file.stem.replace('_communities', '')
        
        print("\n" + "="*80)
        print(f"VALIDATING: {dataset_name}")
        print("="*80)
        print("Using expression data as features to predict BIGCLAM communities")
        
        # Load expression data (features)
        expression_file = processed_dir / f"{dataset_name}_processed.npy"
        if not expression_file.exists():
            print(f"[SKIP] No expression file: {expression_file}")
            continue
        
        expression_data = np.load(expression_file)
        print(f"    Expression data shape: {expression_data.shape}")
        
        # Load communities (targets)
        communities = np.load(clustering_file)
        
        # Fix: If communities is 2D (membership matrix), convert to 1D (assignments)
        if communities.ndim == 2:
            print(f"[INFO] Converting 2D membership matrix to 1D community assignments...")
            communities = np.argmax(communities, axis=1)
        
        # Ensure 1D array
        communities = communities.flatten()
        
        # Validate sizes match
        if len(communities) != expression_data.shape[0]:
            print(f"[ERROR] Size mismatch: communities={len(communities)}, expression_data={expression_data.shape[0]}")
            print(f"        Skipping {dataset_name}...")
            continue
        
        print(f"    Communities shape: {communities.shape}")
        print(f"    Number of communities: {len(set(communities))}")
        
        # Encode communities as labels
        communities_encoded, le = encode_labels(communities)
        
        # Use expression data as features (X), communities as targets (y)
        X = expression_data
        y = communities_encoded
        
        print(f"    Features (expression): {X.shape}")
        print(f"    Targets (communities): {y.shape}")
        
        # Get dataset-specific parameters if available
        if dataset_name in dataset_specific_params:
            ds_params = dataset_specific_params[dataset_name]
            dataset_mlp_params = ds_params.get('mlp', mlp_params)
            dataset_svm_params = ds_params.get('svm', svm_params)
            print(f"\n[Using dataset-specific parameters for {dataset_name}]")
            print(f"    MLP: {dataset_mlp_params}")
            print(f"    SVM: {dataset_svm_params}")
        else:
            dataset_mlp_params = mlp_params
            dataset_svm_params = svm_params
            print(f"\n[Using default parameters for {dataset_name}]")
        
        # Convert MLP params format (handle both 'lr' and 'learning_rate')
        if 'learning_rate' in dataset_mlp_params and 'lr' not in dataset_mlp_params:
            dataset_mlp_params = dataset_mlp_params.copy()
            dataset_mlp_params['lr'] = dataset_mlp_params.pop('learning_rate')
        
        # Split data
        X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
            X, y, test_size=0.2, valid_size=0.2
        )
        
        # Check if augmentation is enabled for SVM (use same setting as MLP)
        use_augmentation = dataset_mlp_params.get('use_augmentation', True)
        noise_std = dataset_mlp_params.get('augmentation_noise_std', 0.1)
        
        if use_augmentation:
            print("\n[Augmenting training data for SVM...]")
            from src.analysis.augmentation_ablation import augment_data
            
            # Augment training data for SVM
            X_train_svm, y_train_svm = augment_data(X_train, y_train, noise_std=noise_std)
        else:
            X_train_svm = X_train
            y_train_svm = y_train
        
        # Train classifiers with dataset-specific parameters
        svm_results = train_svm(X_train_svm, y_train_svm, X_valid, y_valid, X_test, y_test, **dataset_svm_params)
        mlp_results = train_mlp_classifier(X_train, y_train, X_valid, y_valid, X_test, y_test, **dataset_mlp_params)
        
        # Add probability predictions for ROC curves (use same model from train_svm)
        if svm_results and 'model' not in svm_results:
            # If model wasn't returned, create a new one for probabilities
            from sklearn.svm import SVC
            from sklearn.utils.class_weight import compute_class_weight
            use_class_weights = dataset_svm_params.get('use_class_weights', True)
            class_weight = None
            if use_class_weights:
                classes = np.unique(y_train_svm)
                class_weights = compute_class_weight('balanced', classes=classes, y=y_train_svm)
                class_weight = dict(zip(classes, class_weights))
            
            svm_model = SVC(kernel=dataset_svm_params.get('kernel', 'rbf'), 
                           C=dataset_svm_params.get('C', 0.1),
                           gamma=dataset_svm_params.get('gamma', 'scale'),
                           class_weight=class_weight,
                           probability=True, random_state=42)
            svm_model.fit(X_train_svm, y_train_svm)
            svm_results['y_test_proba'] = svm_model.predict_proba(X_test)
            svm_results['y_test'] = y_test
        
        # Create plots (confusion matrices and ROC curves)
        create_classification_plots(svm_results, mlp_results, le, output_dir, dataset_name)
        
        # Save results
        with open(output_dir / f"{dataset_name}_classification_results.pkl", 'wb') as f:
            pickle.dump({
                'svm': svm_results,
                'mlp': mlp_results,
                'label_encoder': le,
                'n_communities': len(set(communities))
            }, f)
        
        print(f"\n[Saved] Classification results to: {output_dir}")
        print(f"        Confusion matrices and ROC curves saved as .tiff files")


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

