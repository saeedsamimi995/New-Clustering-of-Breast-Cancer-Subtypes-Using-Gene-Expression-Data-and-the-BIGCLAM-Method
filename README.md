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
- ✅ Adaptive model selection (AIC/BIC based on dataset size)
- ✅ Comprehensive evaluation metrics (ARI, NMI, Purity, F1, MCC)
- ✅ Rich visualizations (t-SNE, UMAP, heatmaps, confusion matrices)
- ✅ Cross-dataset validation
- ✅ MLP/SVM classification validation
- ✅ Parameter grid search (144 combinations per dataset)
- ✅ Baseline comparison (original vs BIGCLAM features)
- ✅ Computational benchmarking (runtime and memory)
- ✅ Data augmentation ablation study
- ✅ Comprehensive method comparison (BIGCLAM vs K-means, Spectral, NMF, HDBSCAN, Leiden/Louvain)
- ✅ Cluster-to-PAM50 mapping analysis
- ✅ Coefficient of variation option for feature selection

**Documentation:**
- For detailed scientific methodology, parameter selection, and reproducibility details, see [docs/METHODOLOGY.md](docs/METHODOLOGY.md)
- For pipeline structure and advanced usage, see [STRUCTURE.md](STRUCTURE.md)

## Pipeline Structure

The pipeline consists of **13 modular Python modules** organized in `src/`:

**Note**: This pipeline expects prepared CSV files (`*_target_added.csv`) to already exist.
Data preparation is a separate step (see [Data Preparation](#data-preparation) below).

```
BIGCLAM/
├── run_all.py                  # Main orchestration script
├── src/
│   ├── preprocessing/          # Step 1: Data preprocessing
│   │   └── data_preprocessing.py
│   ├── graph/                  # Step 2: Graph construction
│   │   └── graph_construction.py
│   ├── clustering/             # Step 3: BIGCLAM clustering
│   │   └── clustering.py
│   ├── bigclam/                # BIGCLAM model
│   │   └── bigclam_model.py
│   ├── evaluation/             # Step 4: Evaluation metrics
│   │   └── evaluators.py
│   ├── visualization/          # Step 5: Visualizations
│   │   └── visualizers.py
│   ├── classifiers/            # Step 6: Classification validation
│   │   ├── classifiers.py
│   │   └── mlp_classifier.py
│   ├── interpretation/         # Step 7: Result interpretation
│   │   └── interpreters.py
│   └── analysis/               # Step 8: Cross-dataset analysis & grid search
│       ├── cross_dataset_analysis.py
│       └── parameter_grid_search.py  # Parameter optimization
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

### Data Preparation (One-Time Setup)

**Note**: This pipeline expects prepared CSV files (`*_target_added.csv`) to already exist.
If you need to prepare data from raw files, run this separately:

```bash
# Activate virtual environment (if using one)
source venv/bin/activate

# Prepare data from raw files (one-time setup)
python -m src.preprocessing.data_preparing --config config/config.yml --dataset both
```

This creates:
- `data/tcga_brca_data_target_added.csv`
- `data/gse96058_data_target_added.csv`

### Complete Pipeline (Recommended)

```bash
# Run all steps (expects prepared CSV files to exist)
python run_all.py --config config/config.yml
```

The pipeline will:
1. Preprocess TCGA-BRCA and GSE96058 (from prepared CSV files)
2. Build similarity graphs
3. Apply BIGCLAM clustering
4. Evaluate against PAM50/Oncotree labels
5. Generate visualizations
6. Validate with MLP/SVM
7. Analyze cross-dataset consistency

### Parameter Grid Search

Test different variance and similarity thresholds to find optimal parameters:

```bash
# Run grid search for both datasets
python run_all.py --steps grid_search

# This will:
# - Auto-generate parameter ranges from start/end/step in config.yml
#   (e.g., variance: 0.5-15.0 step 0.5 = 30 values, similarity: 0.1-0.9 step 0.05 = 17 values)
# - Test all combinations (510 per dataset: 30 × 17)
# - Run full pipeline for each combination
# - Generate paper-ready visualizations
# - Recommend best configuration
```

Results are saved to `results/grid_search/` with PNG visualizations.

### Run Specific Steps

```bash
# Run only preprocessing and graph construction
python run_all.py --steps preprocess graph

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

### Additional Validation Analyses

The pipeline includes comprehensive validation analyses to ensure methodological rigor:

#### 1. Baseline Comparison
Compares classification performance on:
- **Original data** (no BIGCLAM filtering)
- **BIGCLAM cluster features only** (using discovered clusters as features)
- **Combined features** (original + cluster membership)

**Purpose**: Demonstrates whether BIGCLAM actually improves classification performance.

```bash
python src/analysis/baseline_comparison.py --dataset gse96058_data
```

**Output**: `results/baseline_comparison/{dataset}_baseline_comparison.csv`
- Accuracy comparison across feature sets
- Improvement percentages
- Training time comparison

#### 2. Computational Benchmarking
Measures runtime and memory usage for each pipeline step:
- Preprocessing time and memory
- Graph construction time and memory
- BIGCLAM clustering time and memory
- Total pipeline efficiency

```bash
python src/analysis/computational_benchmark.py \
    --dataset gse96058_data \
    --input_file data/gse96058_data_target_added.csv
```

**Output**: `results/benchmarks/{dataset}_benchmark.json` and `.csv`
- Runtime per step (seconds)
- Memory usage (MB/GB)
- Peak memory consumption

#### 3. Data Augmentation Ablation Study
Compares performance with and without data augmentation:
- **Without augmentation**: Training on original imbalanced data
- **With augmentation**: Training on balanced augmented data using SMOTE (Synthetic Minority Oversampling Technique)

**Purpose**: Validates the impact of augmentation on model performance and justifies the augmentation strategy. SMOTE generates synthetic samples by interpolating between existing samples in feature space, which is more biologically plausible than Gaussian noise for gene expression data.

```bash
python src/analysis/augmentation_ablation.py --dataset gse96058_data
```

**Output**: `results/augmentation_ablation/{dataset}_augmentation_ablation.csv`
- Performance difference with/without augmentation
- Impact on imbalanced classes
- Distribution validation results (Kolmogorov-Smirnov tests)
- Justification for augmentation approach

#### 4. Cluster-to-PAM50 Mapping
Maps BIGCLAM clusters to PAM50 molecular subtypes to address interpretability concerns, especially when BIGCLAM finds different numbers of clusters than the 5-class PAM50 system:

**Purpose**: 
- Understand which PAM50 subtypes each BIGCLAM cluster represents
- Identify sub-subtypes (e.g., when 9 clusters map to 5 PAM50 types)
- Distinguish pure vs mixed clusters
- Clarify ground truth usage (PAM50 for molecular, Oncotree for histological)

**Features**:
- PAM50 distribution per cluster (percentage composition)
- Dominant PAM50 subtype identification
- Cluster purity analysis (pure vs mixed)
- Heatmap visualizations

```bash
# Run cluster-to-PAM50 mapping
python src/analysis/cluster_pam50_mapping.py --dataset gse96058
python src/analysis/cluster_pam50_mapping.py --dataset tcga

# Or via run_all.py
python run_all.py --steps cluster_pam50_mapping
```

**Output**: `results/cluster_pam50_mapping/{dataset}/`
- `cluster_pam50_mapping.csv`: Detailed mapping table
- `cluster_pam50_heatmap.png`: Visualization heatmap
- `mapping_summary.txt`: Summary statistics

**Key Findings**:
- **GSE96058**: BIGCLAM's 9 clusters represent subdivisions of PAM50 subtypes
- **Mixed clusters**: Some clusters contain multiple PAM50 types, suggesting transition states
- **Sub-subtype discovery**: Multiple clusters mapping to same PAM50 type may represent finer molecular substructure

#### 5. Comprehensive Method Comparison
Compares BIGCLAM with five state-of-the-art clustering methods:
- **K-means** (Centroid-based): Fast, assumes spherical clusters
- **Spectral Clustering** (Graph-based): Captures non-linear boundaries
- **NMF** (Matrix Factorization): Non-negative matrix decomposition
- **HDBSCAN** (Density-based): Variable density, noise handling (optional)
- **Leiden/Louvain** (Graph-based): Modularity-based community detection (optional)
- **BIGCLAM** (Our Method): Overlapping community detection

**Metrics Evaluated:**
- **Silhouette Score**: Internal cluster quality (cohesion vs separation)
- **Davies-Bouldin Index**: Internal cluster quality (lower is better)
- **NMI vs PAM50**: Agreement with ground truth subtypes
- **ARI vs PAM50**: Pairwise agreement with ground truth

**Purpose**: Validates BIGCLAM's competitive performance and demonstrates its unique advantages for overlapping community detection.

```bash
# Run comprehensive method comparison
python src/analysis/comprehensive_method_comparison.py --dataset tcga_brca_data
python src/analysis/comprehensive_method_comparison.py --dataset gse96058_data

# Or via run_all.py
python run_all.py --steps method_comparison
```

**Output**: `results/method_comparison/{dataset}_method_comparison.csv`
- Complete metrics table (Silhouette, Davies-Bouldin, NMI, ARI)
- Runtime comparison
- Cluster number comparison
- Comprehensive visualization figures

**Key Results:**
- **TCGA**: BIGCLAM ranks 2nd in PAM50 alignment (NMI=0.237, ARI=0.219), close to K-means
- **GSE96058**: K-means performs best overall; BIGCLAM requires parameter tuning
- BIGCLAM's unique advantage: Overlapping communities and automatic cluster selection

**Note**: Ground truth clarification:
- **PAM50** (molecular subtypes): Used as primary ground truth for molecular clustering validation
- **Oncotree** (histological subtypes): Used in TCGA, treated as external validation, not ground truth for molecular clustering

#### Run All Validation Analyses

```bash
# Run all analyses for one dataset
python run_additional_analyses.py --dataset gse96058_data

# Run for both datasets
python run_additional_analyses.py --dataset both

# Skip specific analyses
python run_additional_analyses.py --dataset gse96058_data --skip_benchmark
```

#### Additional Features

- **Mean-variance feature selection** runs immediately after loading raw expression values, trimming flat genes before any transformation.

**Error Estimation**: 
Confusion matrices are computed across multiple runs (n=10) and averaged. Decimal values represent means across runs; final reporting uses rounded integers.

**Feature Types Clarification**:
- GSE96058: ~20,000 protein-coding genes + ~10,865 non-coding RNAs, pseudogenes
- TCGA-BRCA: ~20,000 protein-coding genes
- Common genes after intersection: 19,842

**Results Locations**:
- `results/baseline_comparison/` - Comparison of original vs BIGCLAM features
- `results/benchmarks/` - Runtime and memory usage
- `results/augmentation_ablation/` - Impact of data augmentation
- `results/method_comparison/` - Comprehensive comparison with K-means, Spectral, NMF, HDBSCAN, Leiden/Louvain

## Module Details

### Data Preparation (Separate Step)

`src/preprocessing/data_preparing.py` is run separately to prepare data from raw files:
- Loads raw TCGA BRCA and GSE96058 datasets
- Adds target labels (Oncotree for TCGA, PAM50 for GSE)
- Matches clinical and expression data
- Exports to `*_target_added.csv` format

**Note**: This step is separate from the main pipeline. The prepared CSV files should already exist in the repository.

### 1. `src/preprocessing/data_preprocessing.py`
- **Variance filter first**: Raw expression matrices are trimmed using the dataset mean variance *before* any other preprocessing. This drops flat genes while the data are still in their original scale.
- **Log2 transformation**: Handles zeros with `log2(x+1)` on the reduced matrix.
- **Z-score normalization**: Across samples (skipped automatically when input already looks normalized)
- Outputs: `*_processed.npy` and `*_targets.pkl`

### 2. `src/graph/graph_construction.py`
- **Cosine similarity**: Between samples
- **Threshold filtering**: Edges for similarity > 0.4
- **Sparse matrices**: Memory-efficient for large datasets
- Outputs: `*_adjacency.npz` graphs

### 3. `src/clustering/clustering.py`
- **BIGCLAM**: Overlapping community detection
- **Adaptive model selection**: Automatically chooses AIC or BIC based on dataset size
  - **BIC** for smaller datasets (n < 2000): Stronger penalty, prevents overfitting
  - **AIC** for larger datasets (n ≥ 2000): Lighter penalty, captures complex patterns
  - Tests communities from 1 to max_communities, selects optimal
- **GPU acceleration**: PyTorch implementation
- Outputs: `*_communities.npy` and `*_membership.npy`

### 4. `src/evaluation/evaluators.py`
- **Metrics**: ARI, NMI, Purity, F1-score (macro)
- **Confusion matrices**: Cluster vs. label distributions
- **Per-cluster statistics**: Dominant labels and diversity
- Outputs: `*_confusion_matrix.png`, `*_cluster_distribution.png`

### 5. `src/visualization/visualizers.py`
- **t-SNE plots**: Colored by clusters vs. labels
- **UMAP plots**: Alternative dimensionality reduction
- **Membership heatmaps**: Community affinity by label
- Outputs: `*_tsne.png`, `*_umap.png`, `*_membership_heatmap.png`

### 6. `src/classifiers/classifiers.py`
- **MLP**: Multi-layer perceptron with early stopping
- **SVM**: RBF kernel with probability estimates
- **Validation**: Predicts labels from community assignments
- Outputs: Classification accuracy, confusion matrices, ROC curves

### 7. `src/interpretation/interpreters.py`
- **Result interpretation**: Based on ARI/NMI thresholds
- **Overlap analysis**: Mixed subtypes and diversity
- **Border samples**: High membership in multiple communities
- Biological insights and explanations

### 8. `src/analysis/cross_dataset_analysis.py`
- **Centroid computation**: Mean expression per community
- **Correlation analysis**: Between TCGA and GSE communities
- **Matching**: Find corresponding clusters across datasets
- Outputs: Correlation heatmaps, dendrograms

- **Grid search**: Automatically generates similarity ranges from start/end/step in `config.yml`
- **Full pipeline evaluation**: Runs preprocessing → graph → cluster → evaluate for each similarity threshold (17 per dataset by default)
- **Paper-ready visualizations**: Generates high-resolution PNG heatmaps and plots
- **Automated recommendations**: Identifies best parameter configuration based on composite scoring
- Outputs: `results/grid_search/*.png` visualizations and CSV results

### 10. `src/analysis/baseline_comparison.py`
- **Baseline validation**: Compares classification performance on original data vs BIGCLAM-filtered data
- **Feature sets tested**: Original data, cluster-only features, combined (original + clusters)
- **Classifiers**: SVM and MLP evaluated on each feature set
- **Purpose**: Demonstrates whether BIGCLAM improves classification performance
- Outputs: `results/baseline_comparison/{dataset}_baseline_comparison.csv`

### 11. `src/analysis/computational_benchmark.py`
- **Runtime measurement**: Time per pipeline step (preprocessing, graph construction, clustering)
- **Memory profiling**: Peak and per-step memory usage (MB/GB)
- **Efficiency analysis**: Total pipeline time and resource requirements
- **Purpose**: Provides empirical evidence of computational efficiency
- Outputs: `results/benchmarks/{dataset}_benchmark.json` and `.csv`

### 12. `src/analysis/augmentation_ablation.py`
- **Ablation study**: Compares performance with and without data augmentation
- **Augmentation method**: SMOTE (Synthetic Minority Oversampling Technique) to balance classes
- **Impact analysis**: Performance difference, class balance effects
- **Purpose**: Validates augmentation strategy and quantifies its impact
- Outputs: `results/augmentation_ablation/{dataset}_augmentation_ablation.csv`

### 13. `src/analysis/comprehensive_method_comparison.py`
- **Comprehensive method comparison**: BIGCLAM vs K-means, Spectral, NMF, HDBSCAN, Leiden/Louvain
- **Metrics evaluated**: Silhouette Score, Davies-Bouldin, NMI vs PAM50, ARI vs PAM50, Runtime
- **Purpose**: Validates BIGCLAM's competitive performance and demonstrates unique advantages for overlapping community detection
- Outputs: 
  - `results/method_comparison/{dataset}_method_comparison.csv` (summary table)
  - `results/method_comparison/{dataset}_method_comparison.pkl` (detailed results)
  - `results/method_comparison/{dataset}_method_comparison_metrics.png` (bar plots)
  - `results/method_comparison/{dataset}_method_comparison_summary.png` (comprehensive figure)

### 14. `src/analysis/cluster_pam50_mapping.py`
- **Cluster-to-PAM50 mapping**: Maps BIGCLAM clusters to PAM50 molecular subtypes
- **Features**: PAM50 distribution per cluster, dominant subtype identification, cluster purity analysis
- **Purpose**: Addresses interpretability concerns when BIGCLAM finds different numbers of clusters than PAM50 (e.g., 9 vs 5)
- **Ground truth clarification**: PAM50 (molecular) vs Oncotree (histological) usage
- Outputs:
  - `results/cluster_pam50_mapping/{dataset}/cluster_pam50_mapping.csv` (detailed mapping table)
  - `results/cluster_pam50_mapping/{dataset}/cluster_pam50_heatmap.png` (visualization heatmap)
  - `results/cluster_pam50_mapping/{dataset}/mapping_summary.txt` (summary statistics)

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
  variance_threshold_mode: "mean"  # Fixed mean-based threshold (ignored if changed)
  similarity_thresholds:
    tcga_brca_data: 0.2  # Dataset-specific similarity thresholds
    gse96058_data: 0.6
    default: 0.4

grid_search:  # Parameter grid search ranges (auto-generated from start/end/step)
  tcga:
    similarity_start: 0.1
    similarity_end: 0.9
    similarity_step: 0.05
  gse96058:
    similarity_start: 0.1
    similarity_end: 0.9
    similarity_step: 0.05

bigclam:
  max_communities: 10          # Maximum communities to search (optimal found automatically via BIC)
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

**Method 1: Similarity Grid Search (Recommended for Papers)**

Use the grid search to test multiple parameter combinations and generate paper-ready visualizations:

```bash
# Run grid search (tests all similarity values from config)
python run_all.py --steps grid_search
```

**What it does:**
- Automatically generates similarity ranges from start/end/step values in config (e.g., 0.1 to 0.9 step 0.05 = 17 values)
- Tests each similarity threshold with the fixed mean-based variance filter (17 runs per dataset)
- Runs full pipeline (preprocess → graph → cluster → evaluate) for each combination
- Generates comprehensive similarity profiles and summary plots
- Recommends best configuration based on ARI, NMI, Purity, and F1 scores
- Creates paper-ready PNG visualizations (300 DPI)

Results saved to `results/grid_search/`:
- `{dataset}_grid_search_overview.png` - Comprehensive similarity summary
- `{dataset}_{metric}_profile.png` - Individual metric profiles
- `{dataset}_grid_search_results.csv` - Full results table

**Method 2: Sensitivity Analysis**

Use the similarity-only grid search for detailed threshold analysis:

```bash
# Run similarity sweep (auto-detects processed data via config)
python src/analysis/parameter_grid_search.py --dataset tcga

# Or specify a custom range
python src/analysis/parameter_grid_search.py --dataset tcga --similarity_range 0.1 0.2 0.3
```

**What it does:**
- Runs preprocessing → graph → clustering → evaluation for each similarity value
- Aggregates metrics (ARI, NMI, Purity, F1) and graph stats
- Saves CSV + publication-grade plots in `results/grid_search/`

**How to choose:**

**Similarity Threshold:**
- **Recommended range**: 0.3-0.5
- **Ideal graph density**: 0.5-3%
- **Must be fully connected**: Should have 1 connected component
- **Average degree**: 2-10 connections per node is ideal
- **Too low**: Graph too sparse, disconnected components
- **Too high**: Graph too dense, many weak/noisy connections

The script prints recommendations directly and saves detailed analysis in `results/grid_search/` with plots and CSV data.

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
├── grid_search/              # Parameter optimization results
│   ├── *_grid_search_overview.png
│   ├── *_ari_profile.png
│   ├── *_nmi_profile.png
│   └── *_grid_search_results.csv
├── baseline_comparison/      # Baseline validation results
│   ├── *_baseline_comparison.pkl
│   └── *_baseline_comparison.csv
├── benchmarks/               # Computational efficiency results
│   ├── *_benchmark.json
│   └── *_benchmark.csv
├── augmentation_ablation/   # Augmentation impact study
│   ├── *_augmentation_ablation.pkl
│   └── *_augmentation_ablation.csv
├── method_comparison/       # Comprehensive method comparison results
│   ├── *_method_comparison.pkl
│   ├── *_method_comparison.csv
│   ├── *_method_comparison_metrics.png
│   └── *_method_comparison_summary.png
├── cluster_pam50_mapping/  # Cluster-to-PAM50 mapping analysis
│   ├── {dataset}/
│   │   ├── cluster_pam50_mapping.csv
│   │   ├── cluster_pam50_heatmap.png
│   │   └── mapping_summary.txt
└── cross_dataset/
    ├── community_correlations.png
    └── community_dendrogram.png
```

## Data and Code Availability

**Code Repository**:
- GitHub: [Repository URL - to be added]
- DOI: [Zenodo DOI - to be added]

**Processed Data and Results**:
- Supplementary data: [Zenodo DOI - to be added]
- Includes: Processed expression matrices, cluster assignments, evaluation metrics, survival analysis results

**Raw Data Access**:
- **TCGA-BRCA**: 
  - Genomic Data Commons (GDC): https://portal.gdc.cancer.gov/projects/TCGA-BRCA
  - Access via GDC Data Portal or GDC API
  - Clinical data: `data/brca_tcga_pub_clinical_data.tsv`
  
- **GSE96058**:
  - Gene Expression Omnibus (GEO): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058
  - Access via GEO website or GEOquery R package
  - Clinical metadata: `data/GSE96058_clinical_data.csv`

**Reproducibility**:
- All analysis scripts are provided in `src/`
- Configuration files: `config/config.yml`
- Exact commands to reproduce: See [docs/METHODOLOGY.md](docs/METHODOLOGY.md)
- Runtime and memory requirements: See `data/clusterings/{dataset}_runtime_info.json`

**Environment**:
- Python 3.8+
- Dependencies: See `requirements.txt`
- Optional: Dockerfile or environment.yml (to be added)

## Citations

- **BIGCLAM**: Yang & Leskovec (2013). "Overlapping Community Detection at Scale: A Nonnegative Matrix Factorization Approach"
- **PAM50**: Parker et al. (2009). "Supervised Risk Predictor of Breast Cancer Based on Intrinsic Subtypes"
- **TCGA BRCA**: Cancer Genome Atlas Network (2012)
- **GSE96058**: Brueffer et al. (2018)

## Documentation

### Methodology

- **Dataset descriptions**: TCGA-BRCA and GSE96058 with citations and access links
- **Feature types**: Detailed breakdown of ~20,000 protein-coding genes and ~10,865 non-coding RNAs/pseudogenes
- **Preprocessing pipeline**: Variance filtering (mean variance), log2 transformation, z-score normalization
- **Feature selection**: Mean-based variance filter applied globally before any other steps
- **Graph construction**: Similarity calculation, adjacency matrix construction, threshold selection
- **Adaptive model selection**: AIC/BIC selection based on dataset size
- **Error estimation**: Confusion matrix averaging across multiple runs
- **Data augmentation**: SMOTE-based augmentation methodology and impact analysis with distribution validation
- **Cluster-to-PAM50 mapping**: Interpretability analysis and ground truth clarification (PAM50 vs Oncotree)

### Additional Validation Analyses

**[docs/ADDITIONAL_ANALYSES_SUMMARY.md](docs/ADDITIONAL_ANALYSES_SUMMARY.md)** provides detailed documentation for all validation analyses:
- Baseline comparison methodology
- Computational benchmarking procedures
- Augmentation ablation study design
- Comprehensive method comparison framework (K-means, Spectral, NMF, HDBSCAN, Leiden/Louvain)

**[docs/ANALYSES_RESPONSES.md](docs/ANALYSES_RESPONSES.md)** addresses methodological considerations:
- Feature selection justification
- Parameter optimization rationale
- Validation framework details

**[docs/LITERATURE_UPDATE.md](docs/LITERATURE_UPDATE.md)** contains recent references (2020-2024):
- Network-based clustering in cancer
- Recent breast cancer subtyping advances
- Graph clustering methods
- Machine learning in cancer genomics

### Pipeline Structure

**[STRUCTURE.md](STRUCTURE.md)** provides:
- Detailed pipeline structure and module dependencies
- Advanced usage examples
- Troubleshooting guide
- Configuration reference

## License

MIT License - See LICENSE file for details.
