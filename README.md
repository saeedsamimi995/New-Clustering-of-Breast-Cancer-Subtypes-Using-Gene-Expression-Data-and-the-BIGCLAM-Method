# BIGCLAM Breast Cancer Subtype Clustering

A modular pipeline for detecting **overlapping molecular subtypes** in breast cancer gene expression data using **BIGCLAM** (Big Community Affiliation Model). Analyzes TCGA-BRCA and GSE96058 datasets to understand the relationship between histological and molecular classifications.

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Structure](#pipeline-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Details](#module-details)
- [Expected Results](#expected-results)
- [Troubleshooting](#troubleshooting)
- [Configuration](#configuration)
- [Output Structure](#output-structure)
- [Citations](#citations)
- [Documentation](#documentation)

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

**Documentation:**
- For detailed scientific methodology, parameter selection, and reproducibility details, see [docs/METHODOLOGY.md](docs/METHODOLOGY.md)
- For pipeline structure and advanced usage, see [STRUCTURE.md](STRUCTURE.md)

## Pipeline Structure

The pipeline consists of **9 modular Python modules** organized in `src/`:

```
BIGCLAM/
├── run_all.py                  # Main orchestration script
├── src/
│   ├── preprocessing/          # Steps 1-2: Data preparation & preprocessing
│   │   ├── data_preparing.py
│   │   └── data_preprocessing.py
│   ├── graph/                  # Step 3: Graph construction
│   │   └── graph_construction.py
│   ├── clustering/             # Step 4: BIGCLAM clustering
│   │   └── clustering.py
│   ├── bigclam/                # BIGCLAM model
│   │   └── bigclam_model.py
│   ├── evaluation/             # Step 5: Evaluation metrics
│   │   └── evaluators.py
│   ├── visualization/          # Step 6: Visualizations
│   │   └── visualizers.py
│   ├── classifiers/            # Step 7: Classification validation
│   │   ├── classifiers.py
│   │   └── mlp_classifier.py
│   ├── interpretation/         # Step 8: Result interpretation
│   │   └── interpreters.py
│   └── analysis/               # Step 9: Cross-dataset analysis
│       └── cross_dataset_analysis.py
└── config/
    └── config.yml             # Configuration parameters
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

### Complete Pipeline (Recommended)

```bash
# Activate virtual environment (if using one)
source venv/bin/activate

# Run all steps
python run_all.py --config config/config.yml
```

That's it! The pipeline will:
1. Preprocess TCGA-BRCA and GSE96058
2. Build similarity graphs
3. Apply BIGCLAM clustering
4. Evaluate against PAM50/Oncotree labels
5. Generate visualizations
6. Validate with MLP/SVM
7. Analyze cross-dataset consistency

### Run Specific Steps

```bash
# Run only preprocessing and graph construction
python run_all.py --steps prepare preprocess graph

# Run evaluation and visualization
python run_all.py --steps evaluate visualize

# Skip clustering if you already have results
python run_all.py --steps evaluate visualize --skip_clustering

# Or run specific steps
python run_all.py --steps preprocess graph cluster evaluate visualize cross_dataset
```

### View Results

After running, check:
- `results/evaluation/` - ARI, NMI, confusion matrices
- `results/visualization/` - t-SNE, UMAP plots
- `results/classification/` - Classifier results
- `results/cross_dataset/` - Correlation heatmaps

### Expected Runtime

- Preprocessing: ~2-5 minutes per dataset
- Graph construction: ~5-10 minutes per dataset  
- Clustering: ~10-20 minutes per dataset (GPU recommended)
- Evaluation/Visualization: ~5 minutes
- **Total: ~1-2 hours for complete pipeline**

## Module Details

### 1. `src/preprocessing/data_preparing.py`
- Loads raw TCGA BRCA and GSE96058 datasets
- Adds target labels (Oncotree for TCGA, PAM50 for GSE)
- Matches clinical and expression data
- Exports to `*_target_added.csv` format

### 2. `src/preprocessing/data_preprocessing.py`
- **Log2 transformation**: Handles zeros with `log2(x+1)`
- **Variance filtering**: Removes low-variance genes (uses mean variance by default)
- **Z-score normalization**: Across samples
- Outputs: `*_processed.npy` and `*_targets.pkl`

### 3. `src/graph/graph_construction.py`
- **Cosine similarity**: Between samples
- **Threshold filtering**: Edges for similarity > 0.4
- **Sparse matrices**: Memory-efficient for large datasets
- Outputs: `*_adjacency.npz` graphs

### 4. `src/clustering/clustering.py`
- **BIGCLAM**: Overlapping community detection
- **AIC model selection**: Automatically finds optimal number of communities (tests 1 to max_communities)
- **GPU acceleration**: PyTorch implementation
- Outputs: `*_communities.npy` and `*_membership.npy`

### 5. `src/evaluation/evaluators.py`
- **Metrics**: ARI, NMI, Purity, F1-score (macro)
- **Confusion matrices**: Cluster vs. label distributions
- **Per-cluster statistics**: Dominant labels and diversity
- Outputs: `*_confusion_matrix.png`, `*_cluster_distribution.png`

### 6. `src/visualization/visualizers.py`
- **t-SNE plots**: Colored by clusters vs. labels
- **UMAP plots**: Alternative dimensionality reduction
- **Membership heatmaps**: Community affinity by label
- Outputs: `*_tsne.png`, `*_umap.png`, `*_membership_heatmap.png`

### 7. `src/classifiers/classifiers.py`
- **MLP**: Multi-layer perceptron with early stopping
- **SVM**: RBF kernel with probability estimates
- **Validation**: Predicts labels from community assignments
- Outputs: Classification accuracy, confusion matrices, ROC curves

### 8. `src/interpretation/interpreters.py`
- **Result interpretation**: Based on ARI/NMI thresholds
- **Overlap analysis**: Mixed subtypes and diversity
- **Border samples**: High membership in multiple communities
- Biological insights and explanations

### 9. `src/analysis/cross_dataset_analysis.py`
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

## Troubleshooting

**Memory issues**: Process datasets separately or reduce thresholds  
**Slow performance**: Use GPU or reduce iterations  
**Import errors**: Make sure virtual environment is activated and dependencies are installed

For more detailed troubleshooting, see [STRUCTURE.md](STRUCTURE.md#troubleshooting).

## Configuration

Edit `config/config.yml`:

```yaml
preprocessing:
  variance_threshold: "mean"  # Use mean variance as threshold, or numeric value for fixed threshold
  similarity_threshold: 0.4   # Graph edge threshold

bigclam:
  max_communities: 10          # Maximum communities to search (optimal found automatically via AIC)
  iterations: 100              # Optimization iterations per community number
  learning_rate: 0.08          # Adam optimizer learning rate

classifiers:
  mlp:
    num_runs: 10
    num_epochs: 200
    hidden_layers: [80, 50, 20]
  svm:
    kernel: "rbf"
    C: 0.1
```

### Parameter Selection Guide

**Finding optimal thresholds**: Use the sensitivity analysis script to determine the best `variance_threshold` and `similarity_threshold` values:

```bash
# Run sensitivity analysis (auto-detects processed data)
python -m src.analysis.parameter_sensitivity

# Or specify data file
python -m src.analysis.parameter_sensitivity --data data/processed/your_data_processed.npy
```

**What it does:**
- Tests multiple threshold values and measures their impact
- Generates plots and recommendations
- Saves results in `results/sensitivity/`

**How to choose:**

**Variance Threshold:**
- **Recommended range**: 5-20 (or use `"mean"` for automatic)
- **Ideal retention**: 20-40% of features
- **Too low (<10% retention)**: Removes too many potentially useful features
- **Too high (>60% retention)**: Keeps too many noisy/low-signal features

**Similarity Threshold:**
- **Recommended range**: 0.3-0.5
- **Ideal graph density**: 0.5-3%
- **Must be fully connected**: Should have 1 connected component
- **Average degree**: 2-10 connections per node is ideal
- **Too low**: Graph too sparse, disconnected components
- **Too high**: Graph too dense, many weak/noisy connections

The script prints recommendations directly and saves detailed analysis in `results/sensitivity/` with plots and CSV data.

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

## Documentation

### Methodology

**[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** provides comprehensive scientific documentation:
- **Dataset descriptions**: TCGA-BRCA and GSE96058 with citations and access links
- **Preprocessing pipeline**: Log2 transformation, variance filtering, z-score normalization
- **Feature selection**: Variance threshold methods (dynamic "mean" vs fixed values) with sensitivity analysis
- **Graph construction**: Similarity calculation, adjacency matrix construction, threshold selection
- **BIGCLAM implementation**: Algorithm details, automatic AIC-based model selection, optimization
- **Classifier specifications**: MLP and SVM architectures and hyperparameters
- **Evaluation metrics**: Detailed metric calculations and interpretations
- **Reproducibility**: Random seeds, version information, configuration details

### Pipeline Structure

**[STRUCTURE.md](STRUCTURE.md)** provides:
- Detailed pipeline structure and module dependencies
- Advanced usage examples
- Troubleshooting guide
- Configuration reference

## License

MIT License - See LICENSE file for details.
