# Detailed Methodology Documentation

This document provides comprehensive methodological details to address reviewer concerns and ensure reproducibility.

## Table of Contents

1. [Data Description](#data-description)
2. [Feature Selection](#feature-selection)
3. [Graph Construction](#graph-construction)
4. [BIGCLAM Implementation](#bigclam-implementation)
5. [Data Augmentation](#data-augmentation)
6. [Classifier Specifications](#classifier-specifications)
7. [Evaluation Metrics](#evaluation-metrics)

## Data Description

### Dataset: GSE96058

- **Source**: Gene Expression Omnibus (GEO)
- **Total Features**: 30,865
  - ~20,000 protein-coding genes
  - ~10,865 additional features (non-coding RNAs, pseudogenes, etc.)
- **Samples**: Breast cancer patients with PAM50 subtype labels

### Feature Types

The 30,865 features include:
- Protein-coding genes (majority)
- Non-coding RNAs (lncRNAs, microRNAs)
- Pseudogenes
- Other annotated genomic features

Reference: See GEO annotation files for complete feature breakdown.

## Feature Selection

### Variance Threshold Method

**Method**: VarianceThreshold from scikit-learn

**Threshold Selection**: 13

**Justification**:
1. We tested threshold values from 5-20 (see `parameter_sensitivity.py`)
2. Threshold of 13 was selected based on:
   - Retaining sufficient features for downstream analysis
   - Removing low-variance/noise features
   - Maintaining biological signal

**Important Note**: 
Reviewer concern about scaling variance to mean (coefficient of variation) is valid. The current implementation uses absolute variance. Future iterations should consider:

```python
# Coefficient of Variation (CV) approach
CV = std / mean
# Filter features with CV below threshold
```

**Equation**:
```
Feature i is retained if: Var(X_i) > threshold
```

**Results**:
- Original features: 30,865
- Retained after threshold=13: [Run analysis to get exact number]

See `results/sensitivity/variance_threshold_sensitivity.png` for detailed analysis.

## Graph Construction

### Similarity Calculation

**Method**: Cosine Similarity

**Formula**:
```
similarity(i, j) = (X_i · X_j) / (||X_i|| × ||X_j||)
```

### Adjacency Matrix Construction

**Threshold**: 0.4

**Selection Process**:
1. Tested thresholds from 0.1 to 0.9 (see sensitivity analysis)
2. Threshold 0.4 selected based on:
   - Graph connectivity (not too sparse, not too dense)
   - Community structure preservation
   - Computational efficiency

**Distribution Analysis**:
See `results/sensitivity/similarity_threshold_sensitivity.png` for similarity distribution and graph statistics at different thresholds.

**Equation**:
```
A_{ij} = 1 if similarity(i, j) > threshold, else 0
A_{ii} = 0  (no self-loops)
```

## BIGCLAM Implementation

### Algorithm Details

**Reference**: Yang & Leskovec (2013). "Overlapping Community Detection at Scale: A Nonnegative Matrix Factorization Approach"

### Model

BIGCLAM models the probability of an edge between nodes i and j as:

```
P(A_{ij} = 1) = 1 - exp(-F_i · F_j^T)
```

where F is the membership strength matrix (N × C):
- N: number of nodes
- C: number of communities
- F_{ic}: membership strength of node i in community c

### Optimization

**Objective**: Maximize log-likelihood

```
L = Σ_{(i,j) in edges} log(1 - exp(-F_i · F_j^T)) - Σ_{(i,j) not in edges} F_i · F_j^T
```

**Optimizer**: Adam
- Learning rate: 0.08
- Constraints: F ≥ ε (small positive value for numerical stability)

**Model Selection**: AIC (Akaike Information Criterion)

```
AIC = -2 × log_likelihood + 2 × k
where k = N × C (number of parameters)
```

### Community Assignment

Nodes are assigned to the community with maximum membership strength:

```
Community(i) = argmax_c F_{ic}
```

## Data Augmentation

### Method

**Purpose**: Balance class distribution for training classifiers

**Technique**: Gaussian noise injection

**Implementation**:
```python
X_augmented = X_original + ε
where ε ~ N(0, σ^2), σ = 0.1
```

### Impact Analysis

To address reviewer concerns, we provide ablation studies:

1. **Without augmentation**: Direct training on original imbalanced data
2. **With augmentation**: Training on balanced augmented data

**Results**: See `results/ablation/augmentation_impact.json`

**Justification**: 
- Original class distribution: [Provide actual distribution]
- After augmentation: Balanced distribution
- Impact on performance: [Provide comparison metrics]

## Classifier Specifications

### MLP (Multi-Layer Perceptron)

**Architecture**:
- Input layer: Number of features
- Hidden layer 1: 80 neurons
- Hidden layer 2: 50 neurons  
- Hidden layer 3: 20 neurons
- Output layer: Number of classes

**Activation Functions**:
- Hidden layers: LeakyReLU (α = 0.01)
- Output layer: Softmax

**Regularization**:
- Dropout rate: 0.3 (applied after each hidden layer)
- Batch Normalization: After each fully connected layer
- Weight decay: 1e-4

**Training**:
- Optimizer: Adam
- Learning rate: 0.001
- Learning rate scheduling: ReduceLROnPlateau (factor=0.5, patience=5)
- Early stopping: Patience = 10 epochs
- Loss function: CrossEntropyLoss
- Batch size: Full batch (all training samples)

**Hyperparameters**:
```yaml
hidden_layers: [80, 50, 20]
dropout_rate: 0.3
learning_rate: 0.001
weight_decay: 1e-4
num_epochs: 200
patience: 10
```

### SVM (Support Vector Machine)

**Specifications**:
- Kernel: RBF (Radial Basis Function)
- C parameter: 0.1
- Gamma: 'scale' (1 / (n_features × X.var()))
- Probability estimates: Enabled (for ROC curves)

**Justification**: 
- RBF kernel suitable for non-linear gene expression patterns
- C=0.1 provides regularization for high-dimensional data
- Gamma='scale' adapts to feature variance

## Evaluation Metrics

### Confusion Matrix

**Calculation**:
Standard sklearn confusion_matrix function:
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
```

**Error Estimation**:
Confusion matrices are computed across multiple runs (n=10), then averaged:
```
CM_mean = mean(CM_1, CM_2, ..., CM_10)
CM_std = std(CM_1, CM_2, ..., CM_10)
```

**Reported Values**: Integer counts (no decimals)
- If decimals appear in results, they represent mean across runs
- Should be rounded for final reporting

### Metrics Derived from Confusion Matrix

For each class c:

**True Positives (TP)**: `TP_c = CM[c, c]`

**False Positives (FP)**: `FP_c = Σ_{j≠c} CM[j, c]`

**False Negatives (FN)**: `FN_c = Σ_{i≠c} CM[c, i]`

**True Negatives (TN)**: `TN_c = Total - TP_c - FP_c - FN_c`

**Sensitivity (Recall)**: `TP_c / (TP_c + FN_c)`

**Specificity**: `TN_c / (TN_c + FP_c)`

**MCC (Matthews Correlation Coefficient)**:
```
MCC = (TP × TN - FP × FN) / sqrt((TP + FP) × (TP + FN) × (TN + FP) × (TN + FN))
```

### ROC Curves

**Method**: 
- One-vs-rest approach for multi-class
- Probability estimates from classifiers
- Area Under Curve (AUC) calculated for each class

## Computational Requirements

### Hardware

**Minimum**:
- CPU: 4 cores
- RAM: 8 GB
- GPU: Optional (CPU fallback available)

**Recommended**:
- CPU: 8+ cores
- RAM: 16+ GB
- GPU: CUDA-capable (NVIDIA, 4GB+ VRAM)

### Runtime Benchmarks

See `results/benchmark_results.json` for detailed timing:
- Data loading: ~X seconds
- Feature selection: ~X seconds
- Graph construction: ~X seconds
- BIGCLAM training: ~X seconds (varies with max_communities)
- Classifier training: ~X seconds (MLP), ~X seconds (SVM)

**Total Pipeline**: ~X minutes

## Reproducibility

### Random Seeds

Set random seeds for reproducibility:
- NumPy: `np.random.seed(42)`
- PyTorch: `torch.manual_seed(42)`
- Data splitting: `random_state=42`

### Version Information

All package versions specified in `requirements.txt`

### Configuration

All hyperparameters stored in `config/config.yaml`

## Addressing Reviewer Concerns

### R1: Why classifier needed with BIGCLAM?

BIGCLAM identifies communities (clusters) in the gene expression network. The classifiers (MLP/SVM) then use these community assignments as labels for:
1. **Validation**: Verify that communities correspond to meaningful biological subtypes
2. **Prediction**: Enable classification of new samples based on learned patterns

**Comparison**: See `baseline_comparison.py` for results comparing:
- Classifiers on original data (without BIGCLAM)
- Classifiers on BIGCLAM-clustered data

### R1: Community label column

The community label is the cluster assignment from BIGCLAM. It represents which community (subtype) each sample belongs to based on the network structure.

### R2: Data augmentation impact

See ablation study results in `results/ablation/` showing performance with/without augmentation.

### R2: Literature review

We acknowledge this limitation. Future versions will include:
- Recent network-based cancer classification methods
- Comparison with state-of-the-art approaches
- Discussion of BIGCLAM advantages vs alternatives

## References

1. Yang, J., & Leskovec, J. (2013). Overlapping community detection at scale: a nonnegative matrix factorization approach. WSDM.
2. Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. JMLR.
3. Paszke et al. (2019). PyTorch: An imperative style, high-performance deep learning library. NeurIPS.

