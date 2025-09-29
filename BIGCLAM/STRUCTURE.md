# Pipeline Structure Guide

## Overview

The BIGCLAM pipeline is organized into **9 modular scripts** that can be run independently or together.

## Module Flow

```
[1] data_preparing.py
    ↓
    raw data → tcga_brca_data_target_added.csv
            → gse96058_data_target_added.csv

[2] data_preprocessing.py
    ↓
    Log2 + Z-score + Variance filter
    ↓
    tcga_brca_data_target_added_processed.npy
    gse96058_data_target_added_processed.npy

[3] graph_construction.py
    ↓
    Cosine similarity → Adjacency matrix
    ↓
    *_adjacency.npz

[4] clustering.py
    ↓
    BIGCLAM → Communities
    ↓
    *_communities.npy
    *_membership.npy

[5] evaluators.py
    ↓
    ARI, NMI, Purity, F1
    + Confusion matrices

[6] visualizers.py
    ↓
    t-SNE, UMAP, Heatmaps

[7] classifiers.py
    ↓
    MLP + SVM validation

[8] interpreters.py
    ↓
    Biological interpretation

[9] cross_dataset_analysis.py
    ↓
    Cross-cohort correlations
```

## Usage Examples

### Complete Pipeline
```bash
python run_all.py --config config/config.yaml
```

### Step-by-Step
```bash
# Step 1: Prepare data
python data_preparing.py --config config/config.yaml --dataset both

# Step 2: Preprocess
python data_preprocessing.py --input data/tcga_brca_data_target_added.csv
python data_preprocessing.py --input data/gse96058_data_target_added.csv

# Step 3: Build graphs
python graph_construction.py --input_dir data/processed --threshold 0.4

# Step 4: Cluster
python clustering.py --input_dir data/graphs --max_communities 10

# Step 5: Evaluate
python evaluators.py --clustering_dir data/clusterings --targets_dir data/processed

# Step 6: Visualize
python visualizers.py

# Step 7: Classify
python classifiers.py

# Step 8: Cross-dataset
python cross_dataset_analysis.py
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

All parameters are in `config/config.yaml`:

```yaml
preprocessing:
  variance_threshold: 13      # Feature filtering
  similarity_threshold: 0.4   # Graph construction

bigclam:
  max_communities: 10          # Community search range
  iterations: 100              # Optimization steps
  learning_rate: 0.08          # Adam LR

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
# Complete pipeline
python run_all.py

# Individual modules
python data_preparing.py --dataset both
python data_preprocessing.py --input data/gse96058_data_target_added.csv
python graph_construction.py
python clustering.py
python evaluators.py
python visualizers.py
python classifiers.py
python cross_dataset_analysis.py

# With custom config
python run_all.py --config config/config.yaml

# Skip expensive steps if results exist
python run_all.py --skip_clustering
```

## Module Dependencies

```
run_all.py
├── data_preparing.py (independent)
├── data_preprocessing.py (needs: data_preparing.py output)
├── graph_construction.py (needs: data_preprocessing.py output)
├── clustering.py (needs: graph_construction.py output)
├── evaluators.py (needs: clustering.py + data_preprocessing.py output)
├── visualizers.py (needs: clustering.py + data_preprocessing.py output)
├── classifiers.py (needs: clustering.py + data_preprocessing.py output)
├── interpreters.py (needs: evaluators.py output)
└── cross_dataset_analysis.py (needs: clustering.py + data_preprocessing.py output)
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
- Check paths in `config/config.yaml`
- Run `python data_preparing.py` first if raw data exists
- Verify output directories exist

