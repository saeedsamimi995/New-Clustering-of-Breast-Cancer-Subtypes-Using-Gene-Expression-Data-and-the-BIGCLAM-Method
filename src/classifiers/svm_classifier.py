"""SVM classifier implementation."""

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_curve, roc_auc_score


def train_svm(X_train, y_train, X_valid, y_valid, X_test, y_test,
              num_runs=10, kernel='rbf', C=0.1, gamma='scale', probability=True,
              models_dir='models'):
    """
    Train SVM classifier with multiple runs.
    
    Args:
        X_train, X_valid, X_test: Feature matrices.
        y_train, y_valid, y_test: Label vectors.
        num_runs (int): Number of training runs.
        kernel (str): SVM kernel type.
        C (float): Regularization parameter.
        gamma (str or float): Kernel coefficient.
        probability (bool): Whether to enable probability estimates.
        models_dir (str): Directory to save models.
        
    Returns:
        dict: Results dictionary with confusion matrices, ROC curves, and metrics.
    """
    import os
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize storage for results
    conf_matrices_train = []
    conf_matrices_valid = []
    conf_matrices_test = []
    roc_curves_train = []
    roc_curves_valid = []
    roc_curves_test = []
    auc_train = []
    auc_valid = []
    auc_test = []

    # Metrics storage
    tp_train, tn_train, fp_train, fn_train = [], [], [], []
    tp_valid, tn_valid, fp_valid, fn_valid = [], [], [], []
    tp_test, tn_test, fp_test, fn_test = [], [], [], []
    sensitivity_train, specificity_train, mcc_train = [], [], []
    sensitivity_valid, specificity_valid, mcc_valid = [], [], []
    sensitivity_test, specificity_test, mcc_test = [], [], []

    for run in range(num_runs):
        print(f'SVM Run {run + 1}/{num_runs}')
        
        # Initialize SVM classifier
        svm_classifier = SVC(kernel=kernel, C=C, gamma=gamma, probability=probability)

        # Train the SVM classifier
        svm_classifier.fit(X_train, y_train)

        # Save the model
        joblib.dump(svm_classifier, f'{models_dir}/svm_classifier_run_{run}.joblib')

        # Predict on all sets
        y_pred_train = svm_classifier.predict(X_train)
        y_pred_valid = svm_classifier.predict(X_valid)
        y_pred_test = svm_classifier.predict(X_test)

        # Evaluate the model
        conf_matrix_train = confusion_matrix(y_train, y_pred_train)
        conf_matrix_valid = confusion_matrix(y_valid, y_pred_valid)
        conf_matrix_test = confusion_matrix(y_test, y_pred_test)

        conf_matrices_train.append(conf_matrix_train)
        conf_matrices_valid.append(conf_matrix_valid)
        conf_matrices_test.append(conf_matrix_test)

        # Calculate metrics for train set
        tp = np.diag(conf_matrix_train)
        fp = conf_matrix_train.sum(axis=0) - tp
        fn = conf_matrix_train.sum(axis=1) - tp
        tn = conf_matrix_train.sum() - (fp + fn + tp)
        tp_train.append(tp.sum())
        tn_train.append(tn.sum())
        fp_train.append(fp.sum())
        fn_train.append(fn.sum())
        sensitivity_train.append(tp.sum() / (tp.sum() + fn.sum() + 1e-10))
        specificity_train.append(tn.sum() / (tn.sum() + fp.sum() + 1e-10))
        mcc_train.append(matthews_corrcoef(y_train, y_pred_train))

        # Calculate metrics for valid set
        tp = np.diag(conf_matrix_valid)
        fp = conf_matrix_valid.sum(axis=0) - tp
        fn = conf_matrix_valid.sum(axis=1) - tp
        tn = conf_matrix_valid.sum() - (fp + fn + tp)
        tp_valid.append(tp.sum())
        tn_valid.append(tn.sum())
        fp_valid.append(fp.sum())
        fn_valid.append(fn.sum())
        sensitivity_valid.append(tp.sum() / (tp.sum() + fn.sum() + 1e-10))
        specificity_valid.append(tn.sum() / (tn.sum() + fp.sum() + 1e-10))
        mcc_valid.append(matthews_corrcoef(y_valid, y_pred_valid))

        # Calculate metrics for test set
        tp = np.diag(conf_matrix_test)
        fp = conf_matrix_test.sum(axis=0) - tp
        fn = conf_matrix_test.sum(axis=1) - tp
        tn = conf_matrix_test.sum() - (fp + fn + tp)
        tp_test.append(tp.sum())
        tn_test.append(tn.sum())
        fp_test.append(fp.sum())
        fn_test.append(fn.sum())
        sensitivity_test.append(tp.sum() / (tp.sum() + fn.sum() + 1e-10))
        specificity_test.append(tn.sum() / (tn.sum() + fp.sum() + 1e-10))
        mcc_test.append(matthews_corrcoef(y_test, y_pred_test))

        # Obtain predicted probabilities
        train_probs = svm_classifier.decision_function(X_train)
        valid_probs = svm_classifier.decision_function(X_valid)
        test_probs = svm_classifier.decision_function(X_test)

        # ROC curves for each class
        n_classes = len(np.unique(y_train))
        roc_curves_train.append([
            roc_curve((y_train == i).astype(int), train_probs[:, i]) 
            for i in range(n_classes)
        ])
        roc_curves_valid.append([
            roc_curve((y_valid == i).astype(int), valid_probs[:, i]) 
            for i in range(n_classes)
        ])
        roc_curves_test.append([
            roc_curve((y_test == i).astype(int), test_probs[:, i]) 
            for i in range(n_classes)
        ])

        # AUC for each class
        auc_train.append([
            roc_auc_score((y_train == i).astype(int), train_probs[:, i]) 
            for i in range(n_classes)
        ])
        auc_valid.append([
            roc_auc_score((y_valid == i).astype(int), valid_probs[:, i]) 
            for i in range(n_classes)
        ])
        auc_test.append([
            roc_auc_score((y_test == i).astype(int), test_probs[:, i]) 
            for i in range(n_classes)
        ])

    return {
        'conf_matrices_train': conf_matrices_train,
        'conf_matrices_valid': conf_matrices_valid,
        'conf_matrices_test': conf_matrices_test,
        'roc_curves_train': roc_curves_train,
        'roc_curves_valid': roc_curves_valid,
        'roc_curves_test': roc_curves_test,
        'auc_train': auc_train,
        'auc_valid': auc_valid,
        'auc_test': auc_test,
        'train_metrics': {
            'tp': np.mean(tp_train), 'tn': np.mean(tn_train),
            'fp': np.mean(fp_train), 'fn': np.mean(fn_train),
            'sensitivity': np.mean(sensitivity_train),
            'specificity': np.mean(specificity_train),
            'mcc': np.mean(mcc_train)
        },
        'valid_metrics': {
            'tp': np.mean(tp_valid), 'tn': np.mean(tn_valid),
            'fp': np.mean(fp_valid), 'fn': np.mean(fn_valid),
            'sensitivity': np.mean(sensitivity_valid),
            'specificity': np.mean(specificity_valid),
            'mcc': np.mean(mcc_valid)
        },
        'test_metrics': {
            'tp': np.mean(tp_test), 'tn': np.mean(tn_test),
            'fp': np.mean(fp_test), 'fn': np.mean(fn_test),
            'sensitivity': np.mean(sensitivity_test),
            'specificity': np.mean(specificity_test),
            'mcc': np.mean(mcc_test)
        }
    }

