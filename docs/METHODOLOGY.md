# Detailed Methodology Documentation

This document provides comprehensive methodological details to ensure reproducibility and validation.

## Table of Contents

1. [Data Description](#data-description)
2. [Preprocessing Pipeline](#preprocessing-pipeline)
3. [Feature Selection](#feature-selection)
4. [Graph Construction](#graph-construction)
5. [BIGCLAM Implementation](#bigclam-implementation)
6. [Data Augmentation](#data-augmentation)
7. [Classifier Specifications](#classifier-specifications)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Survival Analysis](#survival-analysis)
10. [Comprehensive Method Comparison](#comprehensive-method-comparison)
11. [Cluster-to-PAM50 Mapping](#cluster-to-pam50-mapping)
12. [Additional Validation Analyses](#additional-validation-analyses)

## Data Description

### Dataset 1: TCGA-BRCA (The Cancer Genome Atlas Breast Invasive Carcinoma)

- **Source**: The Cancer Genome Atlas Research Network
- **Total Features**: Variable (typically 20,000+ genes)
- **Samples**: 1,093 breast cancer patients
- **Labels**: Oncotree histological subtypes (IDC, ILC, MDLC, Mixed, etc.)
  - **Note**: Oncotree represents histological classification (tissue morphology)
  - **For molecular clustering validation**: PAM50 is used as ground truth (see GSE96058)
  - **Oncotree usage**: External validation only, not ground truth for molecular clustering
- **Data Format**: RNA-seq expression data (log2-transformed RSEM values)
- **Access**: 
  - Genomic Data Commons (GDC): https://portal.gdc.cancer.gov/projects/TCGA-BRCA
  - TCGA Research Network: https://www.cancer.gov/tcga

**Citation**: Cancer Genome Atlas Network. (2012). Comprehensive molecular portraits of human breast tumours. *Nature*, 490(7418), 61-70.

### Dataset 2: GSE96058

- **Source**: Gene Expression Omnibus (GEO)
- **Total Features**: 30,865
  - **~20,000 protein-coding genes**: Standard protein-coding transcripts
  - **~10,865 additional features**:
    - Non-coding RNAs (lncRNAs, microRNAs, snoRNAs)
    - Pseudogenes
    - Processed transcripts
    - Other RNA species
- **Samples**: 3,273 breast cancer patients with 136 replicates
- **Labels**: PAM50 molecular subtypes (LumA, LumB, Her2, Basal, Normal)
  - **PAM50**: Gene expression-based molecular classification system
  - **Used as primary ground truth** for molecular clustering validation in this study
  - **Distinct from Oncotree**: PAM50 reflects molecular characteristics, not histological morphology
- **Data Format**: Transcript-level expression (normalized and transformed)
- **Access**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058

**Citation**: Brueffer, C., et al. (2018). Clinical value of RNA sequencing-based classifiers for prediction of the five conventional breast cancer biomarkers. *JCO Precision Oncology*, 2, 1-18.

### Feature Types

The gene expression features include:
- **Protein-coding genes** (~20,000): Standard protein-coding transcripts, majority of features
- **Non-coding RNAs** (~5,000-7,000): 
  - Long non-coding RNAs (lncRNAs)
  - MicroRNAs (miRNAs)
  - Small nucleolar RNAs (snoRNAs)
  - Other regulatory RNAs
- **Pseudogenes** (~3,000-5,000): Processed and unprocessed pseudogenes
- **Other transcripts**: Processed transcripts, antisense transcripts, etc.

**After intersection with TCGA-BRCA**: 19,842 common features (primarily protein-coding genes)
- Pseudogenes
- Other annotated genomic features

**Note**: The exact number of features may vary between datasets due to different sequencing platforms and annotation versions.

## Preprocessing Pipeline

The preprocessing pipeline consists of sequential steps applied to raw gene expression data:

### Step 1: Variance Filtering (Pre-Log2)

**Purpose**: Remove extremely low-variance genes before any transformation so downstream steps operate on a denser, higher-signal matrix.

**Implementation**:
```python
variances = np.var(X, axis=0)
threshold = variances.mean()
keep_mask = variances >= threshold
X_filtered = X[:, keep_mask]
```

**Rationale**:
- Operating on raw values retains the original dynamic range for the variance estimate.
- Reduces I/O and compute for the rest of the pipeline.

### Step 2: Log2 Transformation

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
1. Variance filtering (raw space)
2. Log2 transformation
3. Z-score normalization

This order ensures:
- Downstream steps operate on a reduced, higher-signal matrix
- Log2 is applied only once, avoiding double transforms on already processed files
- Normalization is the final step before graph construction

## Feature Selection

### Stage 1: Mean-Based Variance Filter

- **Method**: Variance of each gene computed across samples.
- **Threshold**: Dataset-wide mean variance (computed per run; no user override).
- **Retain rule**: Keep genes with `Var(gene) ≥ mean(Var(all genes))`.
- **Outcome**: Removes flat/constant genes while adapting to dataset-specific dispersion.
- **Execution order**: Applied immediately after loading raw counts, *before* log2 or z-score steps, so later preprocessing operates on the reduced matrix.

### Stage 2 onward: (Not applied)

After the variance filter, no additional pruning or ranking is performed. The reduced matrix flows directly into log2/z-score normalization and downstream steps. This keeps the pipeline simple and ensures that every dataset uses the same, reproducible mean-variance heuristic as its sole feature-selection criterion.

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

The similarity threshold determines which sample pairs are connected by edges. The optimal threshold is determined through the similarity-only grid search:

**Determination Process**:

To find the optimal similarity threshold, run:

```bash
python src/analysis/parameter_grid_search.py --dataset tcga --similarity_range 0.1 0.9 0.05
```

This routine:
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
- Defaults come from the latest grid search (see `config.yml`)
- Re-run the grid search whenever preprocessing or feature-selection logic changes

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
See `results/grid_search/*_grid_search_overview.png` for:
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

To validate the augmentation strategy, we provide ablation studies:

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
Confusion matrices are computed across multiple runs (n=10 for MLP, n=10 for SVM), then averaged:
```
CM_mean = mean(CM_1, CM_2, ..., CM_10)
CM_std = std(CM_1, CM_2, ..., CM_10)
```

**Why Multiple Runs?**
- MLP training involves random initialization
- Multiple runs ensure robustness of results
- Standard deviation captures variability

**Reported Values**: 
- **Integer counts** in final tables (rounded from mean)
- **Decimal values** in intermediate results represent mean across runs
- **Standard deviations** available in detailed results files
- **Physical meaning**: Each cell represents number of samples, so final reporting uses integers

**Example**:
- If 10 runs give: [2080, 2081, 2079, 2082, 2080, 2081, 2079, 2080, 2081, 2080]
- Mean = 2080.3, Std = 0.95
- **Reported**: 2080 (rounded integer)
- **In detailed results**: 2080.3 ± 0.95 (shows variability)

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

## Survival Analysis

### Overview

Survival analysis evaluates whether BIGCLAM-identified clusters have prognostic significance by assessing differences in overall survival (OS) between clusters. This analysis validates that the discovered molecular subtypes are clinically meaningful and associated with patient outcomes.

### Data Requirements

**Required Clinical Variables:**
- `OS_time`: Overall survival time (in days)
- `OS_event`: Event indicator (1 = death, 0 = censored)
- `sample_id`: Sample identifier for merging with cluster assignments

**Optional Adjustment Variables:**
- `age`: Age at diagnosis (continuous)
- `stage`: Disease stage (categorical, e.g., I, II, III, IV)

**Data Sources:**
- **TCGA**: `data/brca_tcga_pub_clinical_data.tsv`
- **GSE96058**: `data/GSE96058_clinical_data.csv` (transposed format)

### Sample ID Matching

**TCGA Dataset:**
- Cluster assignments use normalized IDs (e.g., `TCGA.A1.A0SB`)
- Clinical data uses TCGA barcodes (e.g., `TCGA-A1-A0SB-01`)
- **Matching strategy**: Normalize IDs by:
  1. Converting dashes to dots
  2. Removing suffix (e.g., `-01`, `-02`)
  3. Matching normalized forms

**GSE96058 Dataset:**
- Cluster assignments use position-based IDs (e.g., `F1`, `F2`, `F3...`)
- Clinical data uses GEO sample IDs (e.g., `GSM...`)
- **Matching strategy**: Position-based matching
  - `F1` → first row in clinical data
  - `F2` → second row in clinical data
  - Assumes same sample order in both datasets

### Methods

#### 1. Kaplan-Meier Survival Curves

**Purpose**: Visualize survival probability over time for each cluster.

**Method**: Non-parametric estimation of survival function

**Formula**:
```
S(t) = ∏_{i: t_i ≤ t} (1 - d_i / n_i)
```

Where:
- `S(t)`: Survival probability at time t
- `d_i`: Number of events at time t_i
- `n_i`: Number at risk at time t_i

**Implementation**: `lifelines.KaplanMeierFitter`

**Output**: 
- Survival curves for each cluster
- Median survival times
- Number at risk table
- File: `{dataset}_km_survival.png`

#### 2. Log-Rank Test

**Purpose**: Test for statistically significant differences in survival distributions between clusters.

**Method**: Non-parametric hypothesis test

**Null Hypothesis (H₀)**: All clusters have identical survival distributions

**Alternative Hypothesis (H₁)**: At least one cluster has a different survival distribution

**Test Statistic**:
```
χ² = Σ (O_i - E_i)² / E_i
```

Where:
- `O_i`: Observed number of events in group i
- `E_i`: Expected number of events in group i (under H₀)

**Pairwise Comparisons**: 
- Performed for all cluster pairs
- P-values adjusted for multiple comparisons (Bonferroni correction)

**Implementation**: `lifelines.statistics.logrank_test`

**Output**:
- Pairwise p-values between clusters
- Heatmap visualization
- File: `{dataset}_logrank_results.csv`, `{dataset}_logrank_heatmap.png`

**Interpretation**:
- `p < 0.05`: Significant difference in survival between clusters
- `p ≥ 0.05`: No significant difference

#### 3. Cox Proportional Hazards Model

**Purpose**: Quantify the association between cluster membership and survival, adjusting for clinical covariates.

**Model Type**: Multivariate regression model

**Hazard Function**:
```
h(t|X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ... + βₖXₖ)
```

Where:
- `h(t|X)`: Hazard at time t given covariates X
- `h₀(t)`: Baseline hazard function
- `βᵢ`: Regression coefficients
- `Xᵢ`: Covariates (cluster membership, age, stage, etc.)

**Hazard Ratio (HR)**:
```
HR = exp(β)
```

**Interpretation**:
- `HR > 1`: Increased hazard (worse survival) compared to reference
- `HR < 1`: Decreased hazard (better survival) compared to reference
- `HR = 1`: No difference from reference

**Model Specifications**:
- **Reference cluster**: Typically cluster with best survival or largest sample size
- **Adjustment variables**: Age, stage (if available and ≥50% non-null)
- **Fallback**: Unadjusted model (cluster only) if adjusted model fails

**Robustness Checks**:
- Minimum sample size: 10 samples per variable
- Minimum events: 5 events required
- Missing data: Variables with <50% non-null values are excluded
- Automatic fallback: Unadjusted model if adjusted model fails

**Implementation**: `lifelines.CoxPHFitter`

**Output**:
- Hazard ratios with 95% confidence intervals
- P-values for each cluster
- Model summary statistics
- File: `{dataset}_cox_summary.csv`

**Visualization**:
- Forest plot of hazard ratios
- File: `{dataset}_hazard_ratio_forest.png`

### Comprehensive Survival Summary Figure

**Purpose**: Combine all survival analyses into a single publication-ready figure.

**Components**:
1. **Kaplan-Meier curves** (top panel, full width)
   - Survival curves for all clusters
   - Median survival times
   - Number at risk table

2. **Log-rank test heatmap** (middle left)
   - Pairwise p-values between clusters
   - Color-coded by significance

3. **Hazard ratio forest plot** (middle right)
   - HR with 95% CI for each cluster
   - Reference line at HR=1.0
   - Significance markers (*, **, ***)

4. **Cox model summary table** (bottom)
   - HR, CI, p-values for all clusters
   - Adjustment variables included

**Output**: `{dataset}_survival_summary.png`

### Implementation Details

**Library**: `lifelines` (Python survival analysis library)

**Installation**:
```bash
pip install lifelines
```

**Usage**:
```bash
# Run survival evaluation
python src/evaluation/survival_evaluator.py --datasets both

# Or via run_all.py
python run_all.py --steps survival
```

**Input Files**:
- Cluster assignments: `data/clusterings/{dataset}_communities.npy`
- Sample names: `data/processed/{dataset}_targets.pkl`
- Clinical data: `data/{dataset}_clinical_data.{tsv|csv}`

**Output Directory**: `results/survival/{dataset}/`

**Output Files**:
- `{dataset}_km_survival.png`: Kaplan-Meier curves
- `{dataset}_logrank_results.csv`: Log-rank test results
- `{dataset}_logrank_heatmap.png`: Log-rank p-value heatmap
- `{dataset}_cox_summary.csv`: Cox model summary
- `{dataset}_hazard_ratio_forest.png`: HR forest plot
- `{dataset}_survival_summary.png`: Comprehensive summary figure

### Interpretation Guidelines

**Kaplan-Meier Curves**:
- **Separation**: Clear separation indicates prognostic differences
- **Median survival**: Time at which 50% of patients have died
- **Censoring**: Vertical ticks indicate censored observations

**Log-Rank Test**:
- **Overall significance**: Tests if any clusters differ
- **Pairwise comparisons**: Identifies which specific clusters differ
- **Multiple testing**: Bonferroni correction applied

**Cox Model**:
- **Hazard Ratio**: Magnitude of survival difference
- **Confidence Intervals**: Precision of HR estimate
- **P-value**: Statistical significance
- **Adjustment**: Controls for confounding variables (age, stage)

**Clinical Significance**:
- Clusters with significantly different survival (p < 0.05) are clinically meaningful
- HR > 1.5 or < 0.67 indicates substantial prognostic difference
- Consistent results across datasets strengthen validity

### Limitations

1. **Sample size**: Small clusters may have insufficient power
2. **Censoring**: High censoring rates reduce statistical power
3. **Follow-up time**: Short follow-up may miss late events
4. **Missing data**: Adjustment variables may have missing values
5. **Proportional hazards assumption**: Cox model assumes constant HR over time

### Validation

**Robustness Checks**:
- Automatic fallback to unadjusted model if adjusted model fails
- Minimum sample size and event requirements
- Missing data handling (exclude variables with <50% non-null)
- Sample ID matching validation (reports overlap before merge)

**Quality Control**:
- Reports number of samples with complete survival data
- Validates required columns exist
- Checks for sufficient events (≥5)
- Warns about low sample size relative to number of variables

## Comprehensive Method Comparison

### Overview

To validate BIGCLAM's performance, we compared it against five state-of-the-art clustering methods using multiple evaluation metrics. This comprehensive comparison demonstrates BIGCLAM's competitive performance and unique advantages for breast cancer subtype discovery.

### Compared Methods

1. **K-means** (Centroid-based)
   - Type: Centroid-based partitioning
   - Implementation: `sklearn.cluster.KMeans`
   - Parameters: `n_clusters=5` (matching PAM50 subtypes), `random_state=42`
   - Assumptions: Spherical clusters, non-overlapping

2. **Spectral Clustering** (Graph-based)
   - Type: Graph-based spectral decomposition
   - Implementation: `sklearn.cluster.SpectralClustering`
   - Parameters: `n_clusters=5`, `affinity='rbf'`, `gamma=1.0`
   - Assumptions: Non-linear cluster boundaries, graph structure

3. **NMF** (Non-negative Matrix Factorization)
   - Type: Matrix factorization
   - Implementation: `sklearn.decomposition.NMF`
   - Parameters: `n_components=5`, `random_state=42`
   - Assumptions: Non-negative data, additive parts

4. **HDBSCAN** (Density-based) - Optional
   - Type: Hierarchical density-based clustering
   - Implementation: `hdbscan.HDBSCAN` (if available)
   - Parameters: `min_cluster_size=10`, `min_samples=5`
   - Assumptions: Variable density clusters, noise handling

5. **Leiden/Louvain** (Graph-based community detection) - Optional
   - Type: Modularity-based community detection
   - Implementation: `leidenalg` (Leiden) or `networkx.algorithms.community` (Louvain)
   - Parameters: Resolution parameter optimized
   - Assumptions: Community structure in graphs

6. **BIGCLAM** (Our Method)
   - Type: Overlapping community detection via non-negative matrix factorization
   - Implementation: Custom BIGCLAM model
   - Parameters: Automatically determined via AIC
   - Assumptions: Overlapping communities, network structure

### Evaluation Metrics

We evaluated all methods using four complementary metrics:

1. **Silhouette Score** (Internal validation)
   - Range: [-1, 1], higher is better
   - Measures: Cohesion within clusters vs separation between clusters
   - Formula: `s(i) = (b(i) - a(i)) / max(a(i), b(i))`
     - `a(i)`: Average distance to points in same cluster
     - `b(i)`: Average distance to points in nearest other cluster

2. **Davies-Bouldin Index** (Internal validation)
   - Range: [0, ∞), lower is better
   - Measures: Average similarity ratio of clusters
   - Formula: `DB = (1/k) × Σ max_{j≠i} ((σ_i + σ_j) / d(c_i, c_j))`
     - `σ_i`: Average distance within cluster i
     - `d(c_i, c_j)`: Distance between cluster centers

3. **Normalized Mutual Information (NMI) vs PAM50** (External validation)
   - Range: [0, 1], higher is better
   - Measures: Agreement between clusters and PAM50 ground truth
   - Formula: `NMI = I(X;Y) / sqrt(H(X) × H(Y))`
     - `I(X;Y)`: Mutual information
     - `H(X)`, `H(Y)`: Entropy

4. **Adjusted Rand Index (ARI) vs PAM50** (External validation)
   - Range: [-1, 1], higher is better (0 = random)
   - Measures: Pairwise agreement between clusters and PAM50
   - Formula: `ARI = (RI - E[RI]) / (max(RI) - E[RI])`
     - `RI`: Rand Index
     - `E[RI]`: Expected Rand Index

### Results Summary

#### TCGA Dataset (521 samples, 5 PAM50 subtypes)

| Method | Type | Silhouette | Davies-Bouldin | NMI vs PAM50 | ARI vs PAM50 | N_Clusters | Runtime (s) |
|--------|------|------------|----------------|--------------|--------------|------------|-------------|
| **K-means** | Centroid | 0.0032 | 5.42 | **0.2570** | **0.1894** | 5 | 0.10 |
| **BIGCLAM** | Graph | 0.0045 | 6.14 | **0.2370** | **0.2189** | 3 | - |
| Spectral | Graph | **0.1678** | **0.66** | 0.0224 | 0.0071 | 5 | 7.74 |
| NMF | Matrix Factorization | -0.0437 | 6.31 | 0.0984 | 0.0702 | 4 | 0.11 |
| Louvain | Graph | 0.0064 | 5.57 | 0.0926 | 0.0471 | 8 | 0.17 |

**Key Findings for TCGA:**
- **BIGCLAM ranks 2nd** in PAM50 alignment (NMI=0.237, ARI=0.219), close to K-means (NMI=0.257, ARI=0.189)
- BIGCLAM finds 3 clusters vs 5 PAM50 subtypes, suggesting it may merge related subtypes (e.g., LumA+LumB)
- Spectral clustering has best internal structure (Silhouette=0.168, DB=0.66) but poor PAM50 alignment
- K-means performs best overall for PAM50 alignment but assumes non-overlapping clusters

#### GSE96058 Dataset (3,409 samples, 5 PAM50 subtypes)

| Method | Type | Silhouette | Davies-Bouldin | NMI vs PAM50 | ARI vs PAM50 | N_Clusters | Runtime (s) |
|--------|------|------------|----------------|--------------|--------------|------------|-------------|
| **K-means** | Centroid | **0.0119** | 10.27 | **0.0271** | **0.0125** | 5 | 1.96 |
| NMF | Matrix Factorization | -0.0184 | 16.16 | 0.0214 | 0.0159 | 5 | 3.94 |
| Spectral | Graph | -0.1066 | **3.97** | 0.0042 | 0.0022 | 5 | 1.57 |
| Louvain | Graph | -0.0158 | 13.78 | 0.0097 | -0.0000 | 18 | 1.25 |
| **BIGCLAM** | Graph | -0.0431 | 20.14 | 0.0070 | -0.0018 | 9 | - |

**Key Findings for GSE96058:**
- **K-means performs best** overall (NMI=0.027, ARI=0.013), though alignment is lower than TCGA
- BIGCLAM shows over-clustering (9 clusters vs 5 PAM50) and negative ARI, indicating poor alignment
- All methods show lower PAM50 alignment on GSE96058 compared to TCGA, suggesting dataset-specific challenges
- Spectral clustering has best internal structure (DB=3.97) but poor PAM50 alignment

### Interpretation

1. **BIGCLAM Performance on TCGA:**
   - Competitive with K-means for PAM50 alignment (2nd place)
   - Discovers 3 clusters that may represent broader biological groupings
   - May capture novel subtype relationships beyond standard PAM50 classification

2. **BIGCLAM Performance on GSE96058:**
   - Requires parameter tuning or preprocessing adjustments
   - Over-clustering suggests threshold optimization needed
   - Dataset may have different characteristics requiring method-specific tuning

3. **Method-Specific Insights:**
   - **K-means**: Best PAM50 alignment on both datasets, fast, but assumes non-overlapping clusters
   - **Spectral**: Best internal structure metrics but poor PAM50 alignment, suggesting it captures different structure
   - **BIGCLAM**: Competitive on TCGA, unique advantage of overlapping communities, automatically determines cluster number

### Advantages of BIGCLAM

1. **Overlapping Communities**: Unlike K-means and Spectral, BIGCLAM allows samples to belong to multiple subtypes, capturing biological complexity
2. **Automatic Cluster Selection**: AIC-based model selection eliminates manual parameter tuning
3. **Network-Based**: Leverages graph structure for community detection, suitable for expression similarity networks
4. **Competitive Performance**: On TCGA, BIGCLAM ranks 2nd in PAM50 alignment while discovering potentially novel groupings

### Implementation Details

**Preprocessing for Comparison:**
- All methods use the same standardized data (z-score normalized)
- StandardScaler applied to ensure fair comparison
- Same samples and features used across all methods

**Runtime Considerations:**
- K-means: Fastest (0.10s for TCGA, 1.96s for GSE96058)
- BIGCLAM: Runtime not measured (uses pre-computed clusters)
- Spectral: Slowest (7.74s for TCGA) due to eigenvalue decomposition

**Cluster Number Selection:**
- K-means, Spectral, NMF: Fixed at 5 (matching PAM50)
- BIGCLAM: Automatically determined via AIC (3 for TCGA, 9 for GSE96058)
- Louvain: Automatically determined via modularity optimization

### Files and Outputs

**Results Location**: `results/method_comparison/`

**Output Files:**
- `{dataset}_method_comparison.csv`: Summary table with all metrics
- `{dataset}_method_comparison.pkl`: Detailed results including cluster assignments
- `{dataset}_method_comparison_metrics.png`: Bar plots comparing metrics
- `{dataset}_method_comparison_summary.png`: Comprehensive visualization

**Usage:**
```bash
# Run comprehensive method comparison
python src/analysis/comprehensive_method_comparison.py --dataset tcga_brca_data
python src/analysis/comprehensive_method_comparison.py --dataset gse96058_data

# Or via run_all.py
python run_all.py --steps method_comparison
```

## Cluster-to-PAM50 Mapping

### Purpose

Cluster-to-PAM50 mapping analysis addresses reviewer concerns about interpretability of BIGCLAM clusters, especially when the number of clusters differs from the 5-class PAM50 system (e.g., 9 clusters on GSE96058 vs 5 PAM50 subtypes).

### Method

For each BIGCLAM cluster, we analyze:
1. **PAM50 distribution**: Count and percentage of each PAM50 subtype within the cluster
2. **Dominant PAM50 subtype**: The PAM50 type most represented in each cluster
3. **Cluster purity**: Whether clusters are pure (single PAM50 type) or mixed (multiple PAM50 types)
4. **Visualization**: Heatmaps showing PAM50 distribution per cluster

### Implementation

**Script**: `src/analysis/cluster_pam50_mapping.py`

**Usage**:
```bash
# Run for specific dataset
python src/analysis/cluster_pam50_mapping.py --dataset gse96058
python src/analysis/cluster_pam50_mapping.py --dataset tcga

# Or via run_all.py
python run_all.py --steps cluster_pam50_mapping
```

**Output Files**:
- `results/cluster_pam50_mapping/{dataset}/cluster_pam50_mapping.csv`: Detailed mapping table
- `results/cluster_pam50_mapping/{dataset}/cluster_pam50_heatmap.png`: Visualization heatmap
- `results/cluster_pam50_mapping/{dataset}/mapping_summary.txt`: Summary statistics

### Interpretation

**Pure Clusters**:
- Single dominant PAM50 subtype (>80% of samples)
- Suggests cluster represents a distinct molecular subtype
- Example: Cluster X = 95% Luminal A

**Mixed Clusters**:
- Multiple PAM50 subtypes represented
- May indicate:
  - Transition states between subtypes
  - Intermediate phenotypes
  - Biological overlap between subtypes
- Example: Cluster Y = 60% Luminal A, 30% Luminal B, 10% HER2

**Sub-subtype Discovery**:
- When BIGCLAM finds more clusters than PAM50 types (e.g., 9 vs 5)
- Suggests finer molecular substructure within PAM50 subtypes
- Example: Multiple clusters mapping to "Luminal A" may represent Luminal A1, Luminal A2, etc.

### Ground Truth Clarification

**PAM50 vs Oncotree**:
- **PAM50**: Gene expression-based molecular subtypes (Luminal A, Luminal B, HER2-enriched, Basal-like, Normal)
  - Used as primary ground truth for molecular clustering validation
  - Reflects intrinsic molecular characteristics
- **Oncotree**: Histological subtypes (IDC, ILC, MDLC, etc.)
  - Used in TCGA dataset
  - Represents tissue morphology, not molecular classification
  - Treated as external validation, not ground truth for molecular clustering

**For this study**:
- PAM50 is the primary ground truth for molecular alignment
- Oncotree labels are used for external validation (histological classification)
- BIGCLAM clusters are validated against PAM50, not Oncotree

---

## Additional Validation Analyses

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

