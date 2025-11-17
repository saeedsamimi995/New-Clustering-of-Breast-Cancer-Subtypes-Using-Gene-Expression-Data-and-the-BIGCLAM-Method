# Classifier Fine-Tuning Guide

This guide explains how to fine-tune MLP and SVM classifiers for each dataset (TCGA and GSE96058).

## Overview

The classification system now supports **dataset-specific parameters** for both MLP and SVM models. This allows you to fine-tune each classifier independently based on the characteristics of each dataset.

## Current Setup

### Configuration Structure

The `config/config.yml` file now has dataset-specific classifier parameters:

```yaml
classifiers:
  # Default parameters (fallback)
  default:
    mlp:
      num_runs: 10
      num_epochs: 200
      learning_rate: 0.001
      hidden_layers: [80, 50, 20]
      # ... other parameters
    svm:
      kernel: "rbf"
      C: 0.1
      gamma: "scale"
  
  # Dataset-specific parameters
  dataset_specific:
    tcga_brca_data:
      mlp: { ... }
      svm: { ... }
    gse96058_data:
      mlp: { ... }
      svm: { ... }
```

## How It Works

1. **Default Parameters**: Used as fallback if dataset-specific parameters are not provided
2. **Dataset-Specific Parameters**: Automatically applied when processing each dataset
3. **Automatic Selection**: The system automatically selects the right parameters for each dataset

## Fine-Tuning Process

### Step 1: Run Classification with Current Parameters

```bash
python run_all.py --steps classify
```

This will train both MLP and SVM for each dataset using the current parameters in `config.yml`.

### Step 2: Review Results

Check the classification results:
- `results/classification/tcga_brca_data_classification_results.pkl`
- `results/classification/gse96058_data_classification_results.pkl`
- Confusion matrices: `*_confusion_matrix.tiff`
- ROC curves: `*_roc_curve.tiff`

Look for:
- **Test accuracy**: Should be > 0.7 for good performance
- **Validation vs Test gap**: Large gap indicates overfitting
- **Confusion matrix patterns**: Check which classes are confused

### Step 3: Tune Parameters (Optional)

Use the tuning script to find optimal parameters:

```bash
# Tune for TCGA
python -m src.analysis.classifier_tuning --dataset tcga_brca_data

# Tune for GSE96058
python -m src.analysis.classifier_tuning --dataset gse96058_data
```

This will:
- Test multiple parameter combinations
- Find best parameters based on validation set
- Save results to `results/classifier_tuning/`
- Generate YAML file with best parameters for easy config update

### Step 4: Update Config with Best Parameters

After tuning, update `config/config.yml` with the best parameters:

```yaml
dataset_specific:
  tcga_brca_data:
    mlp:
      learning_rate: 0.001  # From tuning results
      hidden_layers: [100, 50]  # From tuning results
      # ... other tuned parameters
    svm:
      C: 1.0  # From tuning results
      gamma: 0.01  # From tuning results
```

### Step 5: Re-run Classification

```bash
python run_all.py --steps classify
```

## Parameter Tuning Guidelines

### SVM Parameters

**C (Regularization parameter)**:
- **Low (0.01-0.1)**: More regularization, simpler model, less overfitting
- **High (1.0-10.0)**: Less regularization, complex model, may overfit
- **Try**: 0.01, 0.1, 1.0, 10.0

**gamma (RBF kernel parameter)**:
- **"scale"**: Default, good starting point
- **"auto"**: 1/n_features
- **Low (0.001-0.01)**: Wider influence, smoother decision boundary
- **High (0.1-1.0)**: Narrow influence, more complex boundaries
- **Try**: "scale", "auto", 0.001, 0.01, 0.1

**kernel**:
- **"rbf"**: Good for non-linear problems (default)
- **"linear"**: Good for linear problems, faster
- **"poly"**: Polynomial kernel, rarely better than RBF

### MLP Parameters

**learning_rate**:
- **Low (0.0001)**: Slower convergence, more stable
- **Medium (0.001)**: Good default
- **High (0.01)**: Faster but may overshoot
- **Try**: 0.0001, 0.001, 0.01

**hidden_layers**:
- **Smaller**: [64, 32] - Less capacity, less overfitting
- **Medium**: [80, 50, 20] - Good default
- **Larger**: [128, 64, 32] - More capacity, may overfit
- **Try**: [64, 32], [80, 50, 20], [100, 50], [128, 64, 32]

**dropout_rate**:
- **Low (0.2)**: Less regularization
- **Medium (0.3)**: Good default
- **High (0.4-0.5)**: More regularization, prevents overfitting
- **Try**: 0.2, 0.3, 0.4

**num_epochs**:
- **150**: May underfit
- **200**: Good default
- **250-300**: May overfit if not enough regularization
- **Try**: 150, 200, 250

## Interpreting Results

### Good Performance Indicators

- **Test accuracy > 0.7**: Good classification
- **Validation accuracy ≈ Test accuracy**: No overfitting
- **Balanced confusion matrix**: All classes predicted well
- **High AUC (> 0.8)**: Good ROC performance

### Problem Indicators

- **Test accuracy < 0.5**: Poor performance, may need more features or different approach
- **Large gap between validation and test**: Overfitting, reduce model complexity
- **Unbalanced confusion matrix**: Some classes poorly predicted, may need class balancing
- **Low AUC (< 0.6)**: Poor discrimination, may need feature engineering

## Quick Reference

### Current Models

- **TCGA-BRCA**: 1,093 samples, 5 communities
- **GSE96058**: 3,409 samples, 6 communities

### Expected Performance

- **TCGA**: May have lower accuracy due to smaller dataset and more communities
- **GSE96058**: Should have higher accuracy due to larger dataset

### Fine-Tuning Priority

1. **Start with SVM**: Faster to tune, good baseline
2. **Then tune MLP**: More parameters, takes longer
3. **Compare results**: Choose best model per dataset

## Example Workflow

```bash
# 1. Run initial classification
python run_all.py --steps classify

# 2. Check results
ls -lh results/classification/*.pkl

# 3. If results are poor, tune parameters
python -m src.analysis.classifier_tuning --dataset tcga_brca_data
python -m src.analysis.classifier_tuning --dataset gse96058_data

# 4. Update config.yml with best parameters from tuning results
# (Check results/classifier_tuning/*_best_params.yaml)

# 5. Re-run classification
python run_all.py --steps classify

# 6. Compare results and iterate if needed
```

## Notes

- Tuning can take a long time (hours) depending on parameter grid size
- Start with smaller parameter grids for quick testing
- Use validation set for parameter selection, test set only for final evaluation
- Save tuning results for future reference

