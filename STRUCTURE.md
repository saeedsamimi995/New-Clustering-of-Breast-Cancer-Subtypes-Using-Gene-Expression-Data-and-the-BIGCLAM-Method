# Pipeline Structure Guide

## Overview

The BIGCLAM pipeline is organized into **8 modular scripts** within the `src/` directory. Modules can be run independently or together via `run_all.py`.

**Note**: This pipeline expects prepared CSV files (`*_target_added.csv`) to already exist.
Data preparation is a separate step (see [Data Preparation](#data-preparation) below).

## Project Structure

```
project_root/
├── run_all.py              # Main pipeline orchestration
├── setup.py                # Package configuration
├── requirements.txt        # Dependencies
├── config/
│   └── config.yml        # Configuration
├── data/                  # Input/output data (prepared CSV files expected here)
└── src/                   # Source code modules
    ├── preprocessing/     # Step 1: Data preprocessing
    ├── graph/              # Step 2: Graph construction
    ├── clustering/         # Step 3: BIGCLAM clustering
    ├── bigclam/            # BIGCLAM model implementation
    ├── evaluation/         # Step 4: Evaluation metrics
    ├── visualization/      # Step 5: Visualizations
    ├── classifiers/        # Step 6: Classification validation
    ├── interpretation/     # Step 7: Result interpretation
    └── analysis/           # Step 8: Cross-dataset analysis & grid search
```

## Data Preparation

**Separate step** - Run once to prepare data from raw files:

```bash
python -m src.preprocessing.data_preparing --config config/config.yml --dataset both
```

This creates:
- `data/tcga_brca_data_target_added.csv`
- `data/gse96058_data_target_added.csv`

These prepared files are expected to exist before running the main pipeline.

## Module Flow

```
[Prepared CSV files must exist]
    data/tcga_brca_data_target_added.csv
    data/gse96058_data_target_added.csv

[1] src/preprocessing/data_preprocessing.py
    ↓
    Log2 + Z-score + Variance filter
    ↓
    tcga_brca_data_processed.npy
    gse96058_data_processed.npy

[2] src/graph/graph_construction.py
    ↓
    Cosine similarity → Adjacency matrix
    ↓
    *_adjacency.npz

[3] src/clustering/clustering.py
    ↓
    BIGCLAM → Communities
    ↓
    *_communities.npy
    *_membership.npy

[4] src/evaluation/evaluators.py
    ↓
    ARI, NMI, Purity, F1
    + Confusion matrices

[5] src/visualization/visualizers.py
    ↓
    t-SNE, UMAP, Heatmaps

[6] src/classifiers/classifiers.py
    ↓
    MLP + SVM validation

[7] src/interpretation/interpreters.py
    ↓
    Biological interpretation

[8] src/analysis/cross_dataset_analysis.py
    ↓
    Cross-cohort correlations

[9] src/analysis/parameter_grid_search.py (NEW!)
    ↓
    Grid search over parameter combinations
    ↓
    Paper-ready visualizations + recommendations
```

## Usage Examples

### Complete Pipeline
```bash
# Assumes prepared CSV files exist
python run_all.py --config config/config.yml
```

### Parameter Grid Search
```bash
# Test multiple parameter combinations
python run_all.py --steps grid_search

# This will:
# - Auto-generate parameter ranges from start/end/step in config
#   (e.g., variance: 0.5-15.0 step 0.5 = 30 values, similarity: 0.1-0.9 step 0.05 = 17 values)
# - Test all combinations (510 per dataset: 30 × 17)
# - Run full pipeline for each combination
# - Generate paper-ready visualizations
# - Output best configuration recommendation
```

### Step-by-Step (via run_all.py - Recommended)
```bash
# All steps via main script
python run_all.py --config config/config.yml

# Or specific steps only
python run_all.py --steps preprocess graph cluster evaluate

# Grid search for parameter optimization
python run_all.py --steps grid_search
```

### Step-by-Step (Direct Module Execution)
```bash
# Data Preparation (separate step, one-time setup)
python -m src.preprocessing.data_preparing --config config/config.yml --dataset both

# Step 1: Preprocess
python -m src.preprocessing.data_preprocessing --input data/tcga_brca_data_target_added.csv

# Step 2: Build graphs
python -m src.graph.graph_construction --input_dir data/processed --threshold 0.4

# Step 3: Cluster
python -m src.clustering.clustering --input_dir data/graphs --max_communities 10

# Step 4: Evaluate
python -m src.evaluation.evaluators --clustering_dir data/clusterings --targets_dir data/processed

# Step 5: Visualize
python -m src.visualization.visualizers

# Step 6: Classify
python -m src.classifiers.classifiers

# Step 7: Cross-dataset
python -m src.analysis.cross_dataset_analysis

# Step 8: Grid search (parameter optimization)
python -m src.analysis.parameter_grid_search --dataset tcga --config config/config.yml
```

### Custom Runs
```bash
# Only clustering and evaluation
python run_all.py --steps preprocess graph cluster evaluate

# Skip classification
python run_all.py --skip_classification

# Run only cross-dataset analysis (requires existing clusterings)
python run_all.py --steps cross_dataset --skip_clustering

# Grid search for parameter optimization
python run_all.py --steps grid_search
```

## Input/Output Files

### Input Files
- `data/tcga_brca_data_target_added.csv` (genes × samples, target in last row)
- `data/gse96058_data_target_added.csv` (genes × samples, target in last row)

### Intermediate Files
- `data/processed/*_processed.npy` - Processed expression matrices
- `data/processed/*_targets.pkl` - Target labels and metadata
- `data/graphs/*_adjacency.npz` - Similarity graphs
- `data/clusterings/*_communities.npy` - Cluster assignments
- `data/clusterings/*_membership.npy` - Membership matrices

### Output Files
- `results/evaluation/*.png` - Confusion matrices, distributions
- `results/visualization/*.png` - t-SNE, UMAP, heatmaps
- `results/classification/*.png` - Classifier results
- `results/cross_dataset/*.png` - Cross-dataset plots

## Configuration

All parameters are in `config/config.yml`:

```yaml
preprocessing:
  variance_thresholds:
    tcga_brca_data: "percentile_75"  # Dataset-specific variance thresholds
    gse96058_data: "percentile_75"
    default: "mean"
  similarity_thresholds:
    tcga_brca_data: 0.2  # Dataset-specific similarity thresholds
    gse96058_data: 0.6
    default: 0.4

grid_search:  # Parameter grid search ranges (auto-generated from start/end/step)
  tcga:
    variance_start: 0.5
    variance_end: 15.0
    variance_step: 0.5
    similarity_start: 0.1
    similarity_end: 0.9
    similarity_step: 0.05
  gse96058:
    variance_start: 0.5
    variance_end: 15.0
    variance_step: 0.5
    similarity_start: 0.1
    similarity_end: 0.9
    similarity_step: 0.05

bigclam:
  max_communities: 10          # Maximum communities to search (optimal found automatically via BIC)
  iterations: 100              # Optimization iterations per community number
  learning_rate: 0.08          # Adam optimizer learning rate

classifiers:
  mlp:
    num_runs: 10               # Multiple runs
    num_epochs: 200
    hidden_layers: [80, 50, 20]
  svm:
    kernel: "rbf"
    C: 0.1
    gamma: "scale"
```

## Quick Commands

```bash
# Complete pipeline (recommended - expects prepared CSV files)
python run_all.py

# Grid search for parameter optimization
python run_all.py --steps grid_search

# Individual modules via run_all.py
python run_all.py --steps preprocess graph cluster evaluate

# With custom config
python run_all.py --config config/config.yml

# Skip expensive steps if results exist
python run_all.py --skip_clustering --skip_classification

# Data preparation (separate step, one-time setup)
python -m src.preprocessing.data_preparing --config config/config.yml --dataset both
```

## Module Dependencies

```
run_all.py
├── src/preprocessing/data_preprocessing.py (needs: *_target_added.csv files - separate step)
├── src/graph/graph_construction.py (needs: data_preprocessing.py output)
├── src/clustering/clustering.py (needs: graph_construction.py output)
│   └── src/bigclam/bigclam_model.py (used by clustering)
├── src/evaluation/evaluators.py (needs: clustering.py + data_preprocessing.py output)
├── src/visualization/visualizers.py (needs: clustering.py + data_preprocessing.py output)
├── src/classifiers/classifiers.py (needs: clustering.py + data_preprocessing.py output)
│   └── src/classifiers/mlp_classifier.py (used by classifiers)
├── src/interpretation/interpreters.py (needs: evaluators.py output)
└── src/analysis/
    ├── cross_dataset_analysis.py (needs: clustering.py + data_preprocessing.py output)
    └── parameter_grid_search.py (runs full pipeline: preprocess → graph → cluster → evaluate)

Separate data preparation:
└── src/preprocessing/data_preparing.py (independent, creates *_target_added.csv files)
```

## Parallel Execution

Each dataset (TCGA, GSE96058) is processed independently, so you can:
1. Run preprocessing in parallel
2. Build graphs in parallel
3. Cluster in parallel
4. Evaluate and visualize separately

## Troubleshooting

### Out of Memory
- Reduce `similarity_threshold` in graph construction
- Use `--no_sparse` to disable sparse matrices (may be slower)
- Process datasets separately

### Slow Performance
- Reduce `max_communities` in clustering
- Reduce `iterations` in clustering
- Use GPU for BIGCLAM training
- Skip UMAP if not essential

### Missing Files
- Check paths in `config/config.yml`
- **Prepared CSV files**: Ensure `*_target_added.csv` files exist in `data/` directory
  - If missing, run data preparation: `python -m src.preprocessing.data_preparing`
- Verify output directories exist

