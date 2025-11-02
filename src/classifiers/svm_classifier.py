"""SVM classifier implementation."""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def train_svm(X_train, y_train, X_valid, y_valid, X_test, y_test, **params):
    """
    Train and evaluate SVM classifier.
    
    Args:
        X_train, X_valid, X_test: Features
        y_train, y_valid, y_test: Labels
        params: SVM parameters (kernel, C, gamma, etc.)
        
    Returns:
        dict: Results including accuracy, F1, confusion matrices
    """
    print("\n[Training SVM]...")
    
    # Default parameters
    kernel = params.get('kernel', 'rbf')
    C = params.get('C', 0.1)
    gamma = params.get('gamma', 'scale')
    
    print(f"    Parameters: kernel={kernel}, C={C}, gamma={gamma}")
    
    # Train SVM
    svm = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = svm.predict(X_train)
    y_valid_pred = svm.predict(X_valid)
    y_test_pred = svm.predict(X_test)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    valid_acc = accuracy_score(y_valid, y_valid_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    valid_f1 = f1_score(y_valid, y_valid_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    # Confusion matrices
    train_cm = confusion_matrix(y_train, y_train_pred)
    valid_cm = confusion_matrix(y_valid, y_valid_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    results = {
        'model': 'SVM',
        'train_accuracy': train_acc,
        'valid_accuracy': valid_acc,
        'test_accuracy': test_acc,
        'train_f1': train_f1,
        'valid_f1': valid_f1,
        'test_f1': test_f1,
        'train_cm': train_cm,
        'valid_cm': valid_cm,
        'test_cm': test_cm,
        'y_train_pred': y_train_pred,
        'y_valid_pred': y_valid_pred,
        'y_test_pred': y_test_pred
    }
    
    print(f"\n[SVM Results]")
    print(f"    Train Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
    print(f"    Valid Accuracy: {valid_acc:.4f}, F1: {valid_f1:.4f}")
    print(f"    Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    return results

