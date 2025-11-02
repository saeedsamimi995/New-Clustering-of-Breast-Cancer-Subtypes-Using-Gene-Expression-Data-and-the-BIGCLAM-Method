# Pipeline Structure Guide

## Overview

The BIGCLAM pipeline is organized into **9 modular scripts** within the `src/` directory. Modules can be run independently or together via `run_all.py`.

## Project Structure

```
project_root/
├── run_all.py              # Main pipeline orchestration
├── setup.py                # Package configuration
├── requirements.txt        # Dependencies
├── config/
│   └── config.yml        # Configuration
├── data/                  # Input/output data
└── src/                   # Source code modules
    ├── preprocessing/     # Steps 1-2: Data preparation & preprocessing
    ├── graph/              # Step 3: Graph construction
    ├── clustering/         # Step 4: BIGCLAM clustering
    ├── bigclam/            # BIGCLAM model implementation
    ├── evaluation/         # Step 5: Evaluation metrics
    ├── visualization/      # Step 6: Visualizations
    ├── classifiers/        # Step 7: Classification validation
    ├── interpretation/     # Step 8: Result interpretation
    └── analysis/           # Step 9: Cross-dataset analysis
```

## Module Flow

```
[1] src/preprocessing/data_preparing.py
    ↓
    raw data → tcga_brca_data_target_added.csv
            → gse96058_data_target_added.csv

[2] src/preprocessing/data_preprocessing.py
    ↓
    Log2 + Z-score + Variance filter
    ↓
    tcga_brca_data_target_added_processed.npy
    gse96058_data_target_added_processed.npy

[3] src/graph/graph_construction.py
    ↓
    Cosine similarity → Adjacency matrix
    ↓
    *_adjacency.npz

[4] src/clustering/clustering.py
    ↓
    BIGCLAM → Communities
    ↓
    *_communities.npy
    *_membership.npy

[5] src/evaluation/evaluators.py
    ↓
    ARI, NMI, Purity, F1
    + Confusion matrices

[6] src/visualization/visualizers.py
    ↓
    t-SNE, UMAP, Heatmaps

[7] src/classifiers/classifiers.py
    ↓
    MLP + SVM validation

[8] src/interpretation/interpreters.py
    ↓
    Biological interpretation

[9] src/analysis/cross_dataset_analysis.py
    ↓
    Cross-cohort correlations
```

## Usage Examples

### Complete Pipeline
```bash
python run_all.py --config config/config.yml
```

### Step-by-Step (via run_all.py - Recommended)
```bash
# All steps via main script
python run_all.py --config config/config.yml

# Or specific steps only
python run_all.py --steps prepare preprocess graph cluster evaluate
```

### Step-by-Step (Direct Module Execution)
```bash
# Step 1: Prepare data
python -m src.preprocessing.data_preparing --config config/config.yml --dataset both

# Step 2: Preprocess
python -m src.preprocessing.data_preprocessing --input data/tcga_brca_data_target_added.csv

# Step 3: Build graphs
python -m src.graph.graph_construction --input_dir data/processed --threshold 0.4

# Step 4: Cluster
python -m src.clustering.clustering --input_dir data/graphs --max_communities 10

# Step 5: Evaluate
python -m src.evaluation.evaluators --clustering_dir data/clusterings --targets_dir data/processed

# Step 6: Visualize
python -m src.visualization.visualizers

# Step 7: Classify
python -m src.classifiers.classifiers

# Step 8: Cross-dataset
python -m src.analysis.cross_dataset_analysis
```

### Custom Runs
```bash
# Only clustering and evaluation
python run_all.py --steps preprocess graph cluster evaluate

# Skip classification
python run_all.py --skip_classification

# Run only cross-dataset analysis (requires existing clusterings)
python run_all.py --steps cross_dataset --skip_clustering
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
  variance_threshold: "mean"  # Use mean variance as threshold, or numeric value for fixed threshold
  similarity_threshold: 0.4   # Graph construction

bigclam:
  max_communities: 10          # Maximum communities to search (optimal found automatically via AIC)
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
# Complete pipeline (recommended)
python run_all.py

# Individual modules via run_all.py
python run_all.py --steps prepare preprocess graph cluster evaluate

# With custom config
python run_all.py --config config/config.yml

# Skip expensive steps if results exist
python run_all.py --skip_clustering --skip_classification
```

## Module Dependencies

```
run_all.py
├── src/preprocessing/data_preparing.py (independent)
├── src/preprocessing/data_preprocessing.py (needs: data_preparing.py output)
├── src/graph/graph_construction.py (needs: data_preprocessing.py output)
├── src/clustering/clustering.py (needs: graph_construction.py output)
│   └── src/bigclam/bigclam_model.py (used by clustering)
├── src/evaluation/evaluators.py (needs: clustering.py + data_preprocessing.py output)
├── src/visualization/visualizers.py (needs: clustering.py + data_preprocessing.py output)
├── src/classifiers/classifiers.py (needs: clustering.py + data_preprocessing.py output)
│   └── src/classifiers/mlp_classifier.py (used by classifiers)
├── src/interpretation/interpreters.py (needs: evaluators.py output)
└── src/analysis/cross_dataset_analysis.py (needs: clustering.py + data_preprocessing.py output)
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
- Run `python data_preparing.py` first if raw data exists
- Verify output directories exist

