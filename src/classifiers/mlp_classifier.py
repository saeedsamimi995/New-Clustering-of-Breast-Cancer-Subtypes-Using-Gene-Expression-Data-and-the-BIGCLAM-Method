"""Multi-Layer Perceptron classifier implementation."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix


class MLP(nn.Module):
    """Multi-Layer Perceptron with batch normalization and dropout."""
    
    def __init__(self, input_size, h1, h2, h3, output_size, dropout_rate=0.3):
        """
        Initialize MLP model.
        
        Args:
            input_size (int): Size of input features.
            h1 (int): Size of first hidden layer.
            h2 (int): Size of second hidden layer.
            h3 (int): Size of third hidden layer.
            output_size (int): Number of output classes.
            dropout_rate (float): Dropout rate.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU()

        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(h3, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward pass."""
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leakyrelu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.softmax(x)
        return x


def calculate_metrics(cm):
    """
    Extract metrics from confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix.
        
    Returns:
        tuple: (TP, TN, FP, FN, sensitivity, specificity, mcc)
    """
    if cm.shape == (2, 2):
        TP = cm[1, 1]
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]

        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        
        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc = ((TP * TN) - (FP * FN)) / denominator if denominator != 0 else 0

        return TP, TN, FP, FN, sensitivity, specificity, mcc
    else:
        # For multi-class, calculate metrics per class and average
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (FP + FN + TP)
        
        sensitivity = TP / (TP + FN + 1e-10)
        specificity = TN / (TN + FP + 1e-10)
        
        # Average over classes
        return TP.sum(), TN.sum(), FP.sum(), FN.sum(), sensitivity.mean(), specificity.mean(), 0


def train_and_evaluate(X_train, y_train_onehot, X_valid, y_valid_onehot, X_test, y_test_onehot,
                       num_epochs=200, lr=0.001, patience=10, weight_decay=1e-4, 
                       dropout_rate=0.3, hidden_layers=(80, 50, 20), save_path='models/best_mlp_model.pth'):
    """
    Train and evaluate MLP model with early stopping.
    
    Args:
        X_train, X_valid, X_test: Training, validation, and test features.
        y_train_onehot, y_valid_onehot, y_test_onehot: One-hot encoded labels.
        num_epochs (int): Maximum number of training epochs.
        lr (float): Learning rate.
        patience (int): Early stopping patience.
        weight_decay (float): Weight decay for regularization.
        dropout_rate (float): Dropout rate.
        hidden_layers (tuple): Sizes of hidden layers.
        save_path (str): Path to save best model.
        
    Returns:
        tuple: Training results including confusion matrices, outputs, and metrics.
    """
    input_size = len(X_train[0])
    h1, h2, h3 = hidden_layers
    output_size = len(y_train_onehot[0])

    model = MLP(input_size, h1, h2, h3, output_size, dropout_rate=dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_valid_loss = float('inf')
    patience_counter = 0
    train_errors = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train, dtype=torch.float))
        loss = criterion(outputs, torch.argmax(torch.tensor(y_train_onehot, dtype=torch.float), dim=1))
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            valid_outputs = model(torch.tensor(X_valid, dtype=torch.float))
            valid_loss = criterion(valid_outputs, torch.argmax(torch.tensor(y_valid_onehot, dtype=torch.float), dim=1))

        # Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        train_errors.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Valid Loss: {valid_loss.item():.4f}')

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        scheduler.step(valid_loss)

    # Load the best model
    model.load_state_dict(torch.load(save_path))

    # Testing the model
    with torch.no_grad():
        model.eval()
        train_outputs = model(torch.tensor(X_train, dtype=torch.float))
        valid_outputs = model(torch.tensor(X_valid, dtype=torch.float))
        test_outputs = model(torch.tensor(X_test, dtype=torch.float))

    # Get predicted labels
    train_preds = torch.argmax(train_outputs, dim=1).numpy()
    valid_preds = torch.argmax(valid_outputs, dim=1).numpy()
    test_preds = torch.argmax(test_outputs, dim=1).numpy()

    # Calculate confusion matrices
    train_cm = confusion_matrix(np.argmax(y_train_onehot, axis=1), train_preds)
    valid_cm = confusion_matrix(np.argmax(y_valid_onehot, axis=1), valid_preds)
    test_cm = confusion_matrix(np.argmax(y_test_onehot, axis=1), test_preds)

    # Calculate metrics
    train_metrics = calculate_metrics(train_cm)
    valid_metrics = calculate_metrics(valid_cm)
    test_metrics = calculate_metrics(test_cm)

    return (train_cm, valid_cm, test_cm, train_errors, train_outputs, 
            valid_outputs, test_outputs, train_metrics, valid_metrics, test_metrics)


def train_mlp(X_train, y_train_onehot, X_valid, y_valid_onehot, X_test, y_test_onehot,
              num_runs=10, num_epochs=200, lr=0.001, patience=10, weight_decay=1e-4,
              dropout_rate=0.3, hidden_layers=(80, 50, 20), models_dir='models'):
    """
    Train MLP model with multiple runs and return best results.
    
    Args:
        X_train, X_valid, X_test: Feature matrices.
        y_train_onehot, y_valid_onehot, y_test_onehot: One-hot encoded labels.
        num_runs (int): Number of training runs.
        Other args: Model hyperparameters.
        
    Returns:
        dict: Results dictionary with best run results and statistics.
    """
    import os
    os.makedirs(models_dir, exist_ok=True)
    
    train_cms, valid_cms, test_cms = [], [], []
    train_errors_list, train_outputs_list = [], []
    valid_outputs_list, test_outputs_list = [], []
    train_metrics_list, valid_metrics_list, test_metrics_list = [], [], []

    for run in range(num_runs):
        print(f'MLP Run {run + 1}/{num_runs}')
        save_path = f'{models_dir}/best_mlp_model_run_{run}.pth'
        
        train_cm, valid_cm, test_cm, train_errors, train_outputs, valid_outputs, test_outputs, \
        train_metrics, valid_metrics, test_metrics = train_and_evaluate(
            X_train, y_train_onehot, X_valid, y_valid_onehot, X_test, y_test_onehot,
            num_epochs=num_epochs, lr=lr, patience=patience, weight_decay=weight_decay,
            dropout_rate=dropout_rate, hidden_layers=hidden_layers, save_path=save_path
        )
        
        train_cms.append(train_cm)
        valid_cms.append(valid_cm)
        test_cms.append(test_cm)
        train_errors_list.append(train_errors)
        train_outputs_list.append(train_outputs)
        valid_outputs_list.append(valid_outputs)
        test_outputs_list.append(test_outputs)
        train_metrics_list.append(train_metrics)
        valid_metrics_list.append(valid_metrics)
        test_metrics_list.append(test_metrics)

    # Find the run with the lowest final validation error
    best_run_idx = np.argmin([errors[-1] for errors in train_errors_list])
    
    return {
        'train_cms': train_cms,
        'valid_cms': valid_cms,
        'test_cms': test_cms,
        'train_errors_list': train_errors_list,
        'train_outputs_list': train_outputs_list,
        'valid_outputs_list': valid_outputs_list,
        'test_outputs_list': test_outputs_list,
        'train_metrics_list': train_metrics_list,
        'valid_metrics_list': valid_metrics_list,
        'test_metrics_list': test_metrics_list,
        'best_run_idx': best_run_idx,
        'best_train_outputs': train_outputs_list[best_run_idx],
        'best_valid_outputs': valid_outputs_list[best_run_idx],
        'best_test_outputs': test_outputs_list[best_run_idx],
        'best_train_metrics': train_metrics_list[best_run_idx],
        'best_valid_metrics': valid_metrics_list[best_run_idx],
        'best_test_metrics': test_metrics_list[best_run_idx]
    }

