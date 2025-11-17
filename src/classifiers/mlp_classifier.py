"""Multi-Layer Perceptron classifier implementation."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix


class MLP(nn.Module):
    """Multi-Layer Perceptron with batch normalization and dropout.
    
    Supports arbitrary number of hidden layers for flexible architectures.
    Example: [512, 256, 128, 64, 32, 16] for 6 hidden layers.
    """
    
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.3):
        """
        Initialize MLP model with dynamic number of hidden layers.
        
        Args:
            input_size (int): Size of input features.
            hidden_layers (list or tuple): Sizes of hidden layers (supports any number of layers).
                Example: [512, 256, 128, 64, 32, 16] for 6 hidden layers.
            output_size (int): Number of output classes.
            dropout_rate (float): Dropout rate.
        """
        super(MLP, self).__init__()
        self.hidden_layers = hidden_layers
        self.num_hidden = len(hidden_layers)
        self.dropout_rate = dropout_rate
        
        if self.num_hidden == 0:
            raise ValueError("At least one hidden layer is required")
        
        # Create layers dynamically using ModuleList
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # Input to first hidden layer
        self.fc_layers.append(nn.Linear(input_size, hidden_layers[0]))
        self.bn_layers.append(nn.BatchNorm1d(hidden_layers[0]))
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers (from first to last)
        for i in range(len(hidden_layers) - 1):
            self.fc_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.bn_layers.append(nn.BatchNorm1d(hidden_layers[i + 1]))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Output layer (from last hidden layer to output)
        self.fc_out = nn.Linear(hidden_layers[-1], output_size)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward pass through all hidden layers."""
        # Pass through all hidden layers
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)
            x = self.bn_layers[i](x)
            x = self.leakyrelu(x)
            x = self.dropout_layers[i](x)
        
        # Output layer
        x = self.fc_out(x)
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
                       num_epochs=200, lr=0.001, min_loss_change=1e-6, weight_decay=1e-4, 
                       dropout_rate=0.3, hidden_layers=(80, 50, 20), save_path='models/best_mlp_model.pth',
                       use_class_weights=True):
    """
    Train and evaluate MLP model with early stopping based on loss change.
    
    Args:
        X_train, X_valid, X_test: Training, validation, and test features.
        y_train_onehot, y_valid_onehot, y_test_onehot: One-hot encoded labels.
        num_epochs (int): Maximum number of training epochs.
        lr (float): Learning rate.
        min_loss_change (float): Minimum change in validation loss to continue training.
            If loss change is less than this value, training stops.
        weight_decay (float): Weight decay for regularization.
        dropout_rate (float): Dropout rate.
        hidden_layers (tuple): Sizes of hidden layers.
        save_path (str): Path to save best model.
        use_class_weights (bool): Whether to use class weights for imbalanced classes.
        
    Returns:
        tuple: Training results including confusion matrices, outputs, and metrics.
    """
    input_size = len(X_train[0])
    # Ensure hidden_layers is a list/tuple
    if isinstance(hidden_layers, (list, tuple)):
        hidden_layers = tuple(hidden_layers)
    else:
        raise ValueError(f"hidden_layers must be a list or tuple, got {type(hidden_layers)}")
    
    # Validate that we have at least one hidden layer
    if len(hidden_layers) == 0:
        raise ValueError("At least one hidden layer is required")
    
    # Validate all layer sizes are positive integers
    for i, size in enumerate(hidden_layers):
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"Hidden layer {i} size must be a positive integer, got {size}")
    
    output_size = len(y_train_onehot[0])

    model = MLP(input_size, hidden_layers, output_size, dropout_rate=dropout_rate)
    
    # Calculate class weights for imbalanced classes
    if use_class_weights:
        # Get class frequencies from training labels
        y_train_labels = np.argmax(y_train_onehot, axis=1)
        class_counts = np.bincount(y_train_labels, minlength=output_size)
        total_samples = len(y_train_labels)
        num_classes = len(class_counts)
        
        # Calculate inverse frequency weights (more weight to rare classes)
        # Add small epsilon to avoid division by zero
        class_weights = total_samples / (num_classes * class_counts + 1e-6)
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * num_classes
        
        # Convert to torch tensor
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        print(f"    Class weights: {dict(zip(range(num_classes), class_weights.round(3)))}")
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Remove verbose parameter for compatibility with newer PyTorch versions
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_valid_loss = float('inf')
    prev_valid_loss = float('inf')
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
            valid_loss_value = valid_loss.item()

        # Save best model
        if valid_loss_value < best_valid_loss:
            best_valid_loss = valid_loss_value
            torch.save(model.state_dict(), save_path)

        # Early stopping based on loss change
        loss_change = None
        if epoch > 0:  # Need at least 2 epochs to compute change
            loss_change = abs(prev_valid_loss - valid_loss_value)
            if loss_change < min_loss_change:
                print(f"Early stopping triggered: loss change ({loss_change:.6f}) < threshold ({min_loss_change})")
                break

        train_errors.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            loss_change_str = f"{loss_change:.6f}" if loss_change is not None else "N/A"
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Valid Loss: {valid_loss_value:.4f}, Loss Change: {loss_change_str}')

        # Update previous loss after checking and printing
        prev_valid_loss = valid_loss_value

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
              num_runs=10, num_epochs=200, lr=0.001, min_loss_change=1e-6, weight_decay=1e-4,
              dropout_rate=0.3, hidden_layers=(80, 50, 20), models_dir='models',
              use_class_weights=True):
    """
    Train MLP model with multiple runs and return best results.
    
    Args:
        X_train, X_valid, X_test: Feature matrices.
        y_train_onehot, y_valid_onehot, y_test_onehot: One-hot encoded labels.
        num_runs (int): Number of training runs.
        min_loss_change (float): Minimum change in validation loss to continue training.
        use_class_weights (bool): Whether to use class weights for imbalanced classes.
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
            num_epochs=num_epochs, lr=lr, min_loss_change=min_loss_change, weight_decay=weight_decay,
            dropout_rate=dropout_rate, hidden_layers=hidden_layers, save_path=save_path,
            use_class_weights=use_class_weights
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

