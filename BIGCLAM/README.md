# BIGCLAM Breast Cancer Subtype Clustering

A modular pipeline for detecting **overlapping molecular subtypes** in breast cancer gene expression data using **BIGCLAM** (Big Community Affiliation Model). Analyzes TCGA-BRCA and GSE96058 datasets to understand the relationship between histological and molecular classifications.

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Structure](#pipeline-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Details](#module-details)
- [Expected Results](#expected-results)

## Overview

This project implements a modular pipeline for clustering breast cancer subtypes with BIGCLAM to detect **overlapping communities**:

**Datasets:**
- **TCGA-BRCA**: 1,093 samples with Oncotree histological labels (IDC, ILC, MDLC, etc.)
- **GSE96058**: 3,273 samples with PAM50 molecular labels (LumA, LumB, Her2, Basal, Normal)

**Key Features:**
- ✅ Overlapping community detection
- ✅ Memory-efficient sparse matrix operations
- ✅ GPU-accelerated BIGCLAM
- ✅ Comprehensive evaluation metrics (ARI, NMI, Purity, F1)
- ✅ Rich visualizations (t-SNE, UMAP, heatmaps, confusion matrices)
- ✅ Cross-dataset validation
- ✅ MLP/SVM classification validation

## Pipeline Structure

The pipeline consists of **9 modular Python scripts**:

```
BIGCLAM/
├── data_preparing.py          # Step 1: Load and prepare raw TCGA/GSE data
├── data_preprocessing.py      # Step 2: Log2, z-score, variance filtering
├── graph_construction.py      # Step 3: Build similarity graphs
├── clustering.py               # Step 4: Apply BIGCLAM clustering
├── evaluators.py               # Step 5: Evaluate against targets (ARI, NMI, Purity, F1)
├── visualizers.py              # Step 6: t-SNE, UMAP, heatmaps
├── classifiers.py              # Step 7: MLP/SVM validation
├── interpreters.py             # Step 8: Interpret results
├── cross_dataset_analysis.py   # Step 9: Cross-cohort consistency
├── run_all.py                  # Orchestrates all steps
├── data_preparing.py           # Alternative data prep script
└── config/
    └── config.yaml             # Configuration parameters
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional but recommended)
- 8GB+ RAM
- 10GB+ disk space

### Setup

```bash
# Clone repository
cd BIGCLAM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install UMAP for additional visualizations
pip install umap-learn
```

## Quick Start

### Option 1: Run Complete Pipeline

```bash
# Run all steps
python run_all.py --config config/config.yaml

# Or run specific steps
python run_all.py --steps preprocess graph cluster evaluate visualize cross_dataset
```

### Option 2: Run Individual Modules

```bash
# 1. Prepare data (if using raw files)
python data_preparing.py --config config/config.yaml --dataset both

# 2. Preprocess
python data_preprocessing.py --input data/tcga_brca_data_target_added.csv
python data_preprocessing.py --input data/gse96058_data_target_added.csv

# 3. Build graphs
python graph_construction.py --input_dir data/processed --threshold 0.4

# 4. Cluster
python clustering.py --input_dir data/graphs --max_communities 10

# 5. Evaluate
python evaluators.py

# 6. Visualize
python visualizers.py

# 7. Validate with classifiers
python classifiers.py

# 8. Cross-dataset analysis
python cross_dataset_analysis.py
```

## Module Details

### 1. `data_preparing.py`
- Loads raw TCGA BRCA and GSE96058 datasets
- Adds target labels (Oncotree for TCGA, PAM50 for GSE)
- Matches clinical and expression data
- Exports to `*_target_added.csv` format

### 2. `data_preprocessing.py`
- **Log2 transformation**: Handles zeros with `log2(x+1)`
- **Variance filtering**: Removes low-variance genes (threshold=13)
- **Z-score normalization**: Across samples
- Outputs: `*_processed.npy` and `*_targets.pkl`

### 3. `graph_construction.py`
- **Cosine similarity**: Between samples
- **Threshold filtering**: Edges for similarity > 0.4
- **Sparse matrices**: Memory-efficient for large datasets
- Outputs: `*_adjacency.npz` graphs

### 4. `clustering.py`
- **BIGCLAM**: Overlapping community detection
- **AIC model selection**: Optimal number of communities
- **GPU acceleration**: PyTorch implementation
- Outputs: `*_communities.npy` and `*_membership.npy`

### 5. `evaluators.py`
- **Metrics**: ARI, NMI, Purity, F1-score (macro)
- **Confusion matrices**: Cluster vs. label distributions
- **Per-cluster statistics**: Dominant labels and diversity
- Outputs: `*_confusion_matrix.png`, `*_cluster_distribution.png`

### 6. `visualizers.py`
- **t-SNE plots**: Colored by clusters vs. labels
- **UMAP plots**: Alternative dimensionality reduction
- **Membership heatmaps**: Community affinity by label
- Outputs: `*_tsne.png`, `*_umap.png`, `*_membership_heatmap.png`

### 7. `classifiers.py`
- **MLP**: Multi-layer perceptron with early stopping
- **SVM**: RBF kernel with probability estimates
- **Validation**: Predicts labels from community assignments
- Outputs: Classification accuracy, confusion matrices, ROC curves

### 8. `interpreters.py`
- **Result interpretation**: Based on ARI/NMI thresholds
- **Overlap analysis**: Mixed subtypes and diversity
- **Border samples**: High membership in multiple communities
- Biological insights and explanations

### 9. `cross_dataset_analysis.py`
- **Centroid computation**: Mean expression per community
- **Correlation analysis**: Between TCGA and GSE communities
- **Matching**: Find corresponding clusters across datasets
- Outputs: Correlation heatmaps, dendrograms

## Expected Results

### Evaluation Metrics

| Dataset | Metric | Interpretation |
|---------|--------|----------------|
| **GSE96058** | ARI with PAM50: **> 0.5** | ✅ Captured molecular patterns |
| **TCGA-BRCA** | ARI with Oncotree: **< 0.3** | ✅ Expected: histological ≠ molecular |
| Both | Purity: **> 0.6** | Dominant subtype per cluster |
| Both | F1 (macro): **> 0.5** | Balanced classification performance |

### Visualizations

- **t-SNE/UMAP**: Clear separation of PAM50 subtypes in GSE96058
- **Confusion matrices**: Asymmetric patterns for TCGA (IDC split across PAM50)
- **Membership heatmaps**: Gradient patterns indicate borders
- **Cross-dataset correlations**: Moderate-high correlations suggest consistency

### Biological Insights

1. **PAM50 Alignment**: BIGCLAM clusters align well with PAM50 molecular subtypes
2. **Oncotree Divergence**: Histological classification (Oncotree) poorly reflects molecular structure
3. **Overlapping Communities**: Border samples show mixed membership
4. **Cross-Cohort Consistency**: Similar molecular signatures across datasets

## Configuration

Edit `config/config.yaml`:

```yaml
preprocessing:
  variance_threshold: 13      # Remove low-variance genes
  similarity_threshold: 0.4   # Graph edge threshold

bigclam:
  max_communities: 10          # Search range
  iterations: 100              # Per community
  learning_rate: 0.08          # Optimization LR

classifiers:
  mlp:
    num_runs: 10
    num_epochs: 200
    hidden_layers: [80, 50, 20]
  svm:
    kernel: "rbf"
    C: 0.1
```

## Output Structure

```
results/
├── evaluation/
│   ├── tcga_brca_confusion_matrix.png
│   ├── gse96058_confusion_matrix.png
│   └── *_cluster_distribution.png
├── visualization/
│   ├── *_tsne.png
│   ├── *_umap.png
│   └── *_membership_heatmap.png
├── classification/
│   ├── svm_confusion_matrix.png
│   └── mlp_confusion_matrix.png
└── cross_dataset/
    ├── community_correlations.png
    └── community_dendrogram.png
```

## Citations

- **BIGCLAM**: Yang & Leskovec (2013). "Overlapping Community Detection at Scale: A Nonnegative Matrix Factorization Approach"
- **PAM50**: Parker et al. (2009). "Supervised Risk Predictor of Breast Cancer Based on Intrinsic Subtypes"
- **TCGA BRCA**: Cancer Genome Atlas Network (2012)
- **GSE96058**: Brueffer et al. (2018)

## License

MIT License - See LICENSE file for details.
