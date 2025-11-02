# Detailed Methodology Documentation

This document provides comprehensive methodological details to address reviewer concerns and ensure reproducibility.

## Table of Contents

1. [Data Description](#data-description)
2. [Preprocessing Pipeline](#preprocessing-pipeline)
3. [Feature Selection](#feature-selection)
4. [Graph Construction](#graph-construction)
5. [BIGCLAM Implementation](#bigclam-implementation)
6. [Data Augmentation](#data-augmentation)
7. [Classifier Specifications](#classifier-specifications)
8. [Evaluation Metrics](#evaluation-metrics)

## Data Description

### Dataset 1: TCGA-BRCA (The Cancer Genome Atlas Breast Invasive Carcinoma)

- **Source**: The Cancer Genome Atlas Research Network
- **Total Features**: Variable (typically 20,000+ genes)
- **Samples**: 1,093 breast cancer patients
- **Labels**: Oncotree histological subtypes (IDC, ILC, MDLC, Mixed, etc.)
- **Data Format**: RNA-seq expression data (log2-transformed RSEM values)
- **Access**: 
  - Genomic Data Commons (GDC): https://portal.gdc.cancer.gov/projects/TCGA-BRCA
  - TCGA Research Network: https://www.cancer.gov/tcga

**Citation**: Cancer Genome Atlas Network. (2012). Comprehensive molecular portraits of human breast tumours. *Nature*, 490(7418), 61-70.

### Dataset 2: GSE96058

- **Source**: Gene Expression Omnibus (GEO)
- **Total Features**: 30,865
  - ~20,000 protein-coding genes
  - ~10,865 additional features (non-coding RNAs, pseudogenes, etc.)
- **Samples**: 3,273 breast cancer patients with 136 replicates
- **Labels**: PAM50 molecular subtypes (LumA, LumB, Her2, Basal, Normal)
- **Data Format**: Transcript-level expression (normalized and transformed)
- **Access**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058

**Citation**: Brueffer, C., et al. (2018). Clinical value of RNA sequencing-based classifiers for prediction of the five conventional breast cancer biomarkers. *JCO Precision Oncology*, 2, 1-18.

### Feature Types

The gene expression features include:
- Protein-coding genes (majority)
- Non-coding RNAs (lncRNAs, microRNAs)
- Pseudogenes
- Other annotated genomic features

**Note**: The exact number of features may vary between datasets due to different sequencing platforms and annotation versions.

## Preprocessing Pipeline

The preprocessing pipeline consists of sequential steps applied to raw gene expression data:

### Step 1: Log2 Transformation

**Purpose**: Normalize expression values to reduce skewness and stabilize variance.

**Implementation**:
```python
X_log2 = log2(X + 1)
```

Where:
- `X`: Original expression values
- `+1`: Pseudocount to handle zero values
- Result: Log2-transformed expression matrix

**Rationale**:
- Gene expression data typically follows log-normal distribution
- Log transformation compresses dynamic range
- Makes data more suitable for downstream statistical analysis

### Step 2: Variance-Based Feature Selection

**Purpose**: Remove low-variance features (see [Feature Selection](#feature-selection) section for details).

**Implementation**: VarianceThreshold filter with dynamic or fixed threshold.

**Output**: Reduced feature set retaining only informative genes.

### Step 3: Z-Score Normalization

**Purpose**: Standardize features to zero mean and unit variance across samples.

**Implementation**:
```python
X_normalized = (X - mean(X, axis=0)) / std(X, axis=0)
```

**Rationale**:
- Removes sample-specific batch effects
- Ensures all features are on same scale
- Critical for similarity calculations and machine learning algorithms

**Pipeline Order**:
1. Log2 transformation
2. Variance filtering
3. Z-score normalization

This order ensures:
- Variance calculated on log-transformed data (more meaningful)
- Normalization applied after feature selection (computational efficiency)
- Final data ready for graph construction and clustering

## Feature Selection

### Variance Threshold Method

**Method**: VarianceThreshold from scikit-learn

**Purpose**: Remove low-variance features that contribute little to downstream analysis, reducing noise and computational burden.

**Threshold Selection Methods**:

The pipeline supports two approaches for threshold selection:

1. **Dynamic Threshold ("mean")**:
   - Calculates the mean variance across all features
   - Uses this mean value as the threshold: `threshold = mean(Var(X_i)) for all i`
   - Advantages:
     - Adapts automatically to dataset characteristics
     - No manual tuning required
     - Works well across different datasets
   - Implementation:
     ```python
     feature_variances = np.var(data, axis=0)
     threshold = np.mean(feature_variances)
     ```

2. **Fixed Numeric Threshold**:
   - User-specified numeric value (e.g., 13)
   - Requires prior knowledge or sensitivity analysis
   - Recommended range: 5-20 for typical gene expression data

**Threshold Determination Process**:

To determine an optimal threshold value, use the sensitivity analysis script:

```bash
python -m src.analysis.parameter_sensitivity --variance_threshold
```

This script:
- Tests threshold values from 5-20 (configurable range)
- Measures feature retention rates for each threshold
- Generates visualization plots showing:
  - Features retained vs threshold (look for "knee" in curve)
  - Retention rate percentage (ideal: 20-40%)
  - Variance distribution across features
- Provides automated recommendations based on:
  - Retention rate: 20-40% is ideal
  - Biological signal preservation
  - Downstream analysis requirements

**Selection Criteria**:
- **Ideal retention**: 20-40% of features
- **Too aggressive** (<10% retention): Removes potentially useful features
- **Too conservative** (>60% retention): Keeps too many noisy/low-signal features
- **Biological consideration**: Adjust based on expected number of relevant genes

**Equation**:
```
Feature i is retained if: Var(X_i) > threshold

Where threshold = {
    mean(Var(X_j)) for all j, if "mean" specified
    user_value, if numeric value provided
}
```

**Current Configuration**: 
- Default: `"mean"` (dynamic calculation)
- Alternative: Numeric value (e.g., `13`) if sensitivity analysis indicates better performance

**Results**: See `results/sensitivity/variance_threshold_sensitivity.png` and recommendation files for detailed analysis.

**Future Improvement Note**: 
A reviewer concern about scaling variance to mean (coefficient of variation) is valid. The current implementation uses absolute variance. Future iterations could consider:

```python
# Coefficient of Variation (CV) approach
CV_i = std(X_i) / mean(X_i)
# Filter features with CV below threshold
```

This would normalize variance relative to expression level, potentially improving feature selection for highly variable genes.

## Graph Construction

### Similarity Calculation

**Method**: Cosine Similarity

**Purpose**: Measure expression profile similarity between samples to construct a sample-similarity graph for community detection.

**Formula**:
```
similarity(i, j) = (X_i · X_j) / (||X_i|| × ||X_j||)
```

Where:
- `X_i`, `X_j`: Expression vectors for samples i and j
- `·`: Dot product
- `||X||`: L2 norm (Euclidean norm)

**Properties**:
- Range: [-1, 1] (typically [0, 1] for normalized expression data)
- Scale-invariant: Normalized by vector magnitude
- Suitable for high-dimensional gene expression data

### Adjacency Matrix Construction

**Purpose**: Convert pairwise similarities into a binary adjacency matrix representing edges in the similarity graph.

**Threshold Selection**:

The similarity threshold determines which sample pairs are connected by edges. The optimal threshold is determined through sensitivity analysis:

**Determination Process**:

To find the optimal similarity threshold, use the sensitivity analysis script:

```bash
python -m src.analysis.parameter_sensitivity --similarity_threshold
```

This script:
- Tests threshold values from 0.1 to 0.9 (configurable range)
- For each threshold, calculates:
  - Number of edges created
  - Graph density (percentage of possible edges)
  - Number of connected components
  - Average degree (connections per node)
  - Similarity value distribution
- Generates visualization plots showing:
  - Edge count vs threshold (log scale)
  - Graph density vs threshold (ideal: 0.5-3%)
  - Average degree vs threshold (ideal: 2-10 connections)
  - Similarity distribution histogram
- Provides automated recommendations based on:
  - **Graph density**: 0.5-3% is ideal for clustering
  - **Connectivity**: Should be 1 connected component (fully connected graph)
  - **Average degree**: 2-10 connections per node indicates good connectivity
  - **Distribution**: Avoid thresholds in sparse regions of similarity histogram

**Selection Criteria**:

- **Recommended range**: 0.3-0.5 for typical gene expression data
- **Ideal graph density**: 0.5-3%
  - Too sparse (<0.1%): Graph has disconnected components, clustering may fail
  - Too dense (>5%): Many weak/noisy connections, computational overhead
- **Connectivity requirement**: Must be 1 connected component (fully connected)
- **Average degree**: 2-10 connections per node is ideal
  - Too low (<1): Nodes are isolated
  - Too high (>20): Many weak connections, noise

**Current Configuration**: 
- Default: `0.4` (selected based on typical gene expression data characteristics)
- Can be adjusted based on sensitivity analysis recommendations

**Equation**:
```
A_{ij} = {
    1, if similarity(i, j) > threshold
    0, otherwise
}
A_{ii} = 0  (no self-loops)
A_{ji} = A_{ij}  (undirected graph, symmetric matrix)
```

**Graph Properties**:
- **Undirected**: Symmetric adjacency matrix
- **Unweighted**: Binary edges (1 if connected, 0 otherwise)
- **Sparse**: Most similarity values below threshold, efficient sparse matrix storage

**Distribution Analysis**:
See `results/sensitivity/similarity_threshold_sensitivity.png` for:
- Similarity distribution across all sample pairs
- Graph statistics at different thresholds
- Automated threshold recommendations with justifications

## BIGCLAM Implementation

### Algorithm Details

**Reference**: Yang & Leskovec (2013). "Overlapping Community Detection at Scale: A Nonnegative Matrix Factorization Approach"

**Purpose**: Detect overlapping communities in sample-similarity graphs, identifying molecular subtypes with shared characteristics.

### Model

BIGCLAM models the probability of an edge between nodes i and j as:

```
P(A_{ij} = 1) = 1 - exp(-F_i · F_j^T)
```

where:
- `F`: Membership strength matrix (N × C)
  - N: number of nodes (samples)
  - C: number of communities (subtypes)
- `F_{ic}`: Membership strength of node i in community c (≥ 0)
- Higher `F_{ic}` values indicate stronger affiliation with community c

**Key Feature**: Supports **overlapping communities** - nodes can belong to multiple communities with different strengths, capturing biological complexity where samples may exhibit characteristics of multiple subtypes.

### Optimization

**Objective**: Maximize log-likelihood of observed graph structure

```
L = Σ_{(i,j) in edges} log(1 - exp(-F_i · F_j^T)) - Σ_{(i,j) not in edges} F_i · F_j^T
```

**Interpretation**:
- First term: Maximize probability of existing edges
- Second term: Minimize probability of non-edges
- Balances fitting observed connections while avoiding overfitting

**Optimizer**: Adam (Adaptive Moment Estimation)
- Learning rate: 0.08 (configured in `config/config.yml`)
- Constraints: `F_{ic} ≥ ε` (small positive value, typically 1e-6, for numerical stability)
- Iterations: 100 per community number (configurable)

### Automatic Model Selection

**Method**: AIC (Akaike Information Criterion)

The pipeline **automatically determines** the optimal number of communities by testing values from 1 to `max_communities` (default: 10) and selecting the model with the lowest AIC.

```
AIC = -2 × log_likelihood + 2 × k
```

where:
- `log_likelihood`: Model fit quality
- `k = N × C`: Number of parameters (membership strength values)
- Lower AIC = better model (better fit with complexity penalty)

**Selection Process**:
1. For each candidate number of communities C ∈ {1, 2, ..., max_communities}:
   - Initialize membership matrix F (N × C)
   - Optimize F using Adam optimizer (100 iterations)
   - Calculate log-likelihood on observed graph
   - Compute AIC = -2 × log_likelihood + 2 × (N × C)
   
2. Select C* with minimum AIC:
   ```
   C* = argmin_{C} AIC(C)
   ```

3. Report optimal communities and final membership matrix

**Configuration**:
- `max_communities`: Maximum number of communities to test (default: 10)
- Model automatically finds optimal within this range
- **Note**: `num_communities` parameter is not used - the model determines this automatically

**Advantages of AIC**:
- Balances model fit with model complexity
- Prevents overfitting (penalizes too many communities)
- Prevents underfitting (penalizes too few communities)
- Data-driven selection, no manual tuning required

### Community Assignment

After optimization, nodes are assigned to communities based on membership strength:

**Hard Assignment** (for discrete labels):
```
Community(i) = argmax_c F_{ic}
```

**Soft Assignment** (for overlapping communities):
- Node i has membership strength `F_{ic}` in each community c
- Nodes can belong to multiple communities if multiple `F_{ic}` values are high
- Threshold can be applied: Node belongs to community c if `F_{ic} > threshold`

**Output**:
- `*_communities.npy`: Hard community assignments (N × 1 array)
- `*_membership.npy`: Full membership strength matrix (N × C array)

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

All hyperparameters stored in `config/config.yml`

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

