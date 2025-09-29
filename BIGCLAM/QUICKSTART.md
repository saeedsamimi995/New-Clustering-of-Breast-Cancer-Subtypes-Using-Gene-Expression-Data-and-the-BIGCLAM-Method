# Quick Start Guide

## Complete Pipeline (Recommended)

```bash
# Activate virtual environment
source venv/bin/activate

# Run all steps
python run_all.py --config config/config.yaml
```

That's it! The pipeline will:
1. Preprocess TCGA-BRCA and GSE96058
2. Build similarity graphs
3. Apply BIGCLAM clustering
4. Evaluate against PAM50/Oncotree labels
5. Generate visualizations
6. Validate with MLP/SVM
7. Analyze cross-dataset consistency

## Individual Steps

If you want to run steps separately:

```bash
# 1. Preprocess data
python data_preprocessing.py --input data/tcga_brca_data_target_added.csv
python data_preprocessing.py --input data/gse96058_data_target_added.csv

# 2. Build graphs
python graph_construction.py

# 3. Cluster
python clustering.py

# 4. Evaluate
python evaluators.py

# 5. Visualize
python visualizers.py

# 6. Classify
python classifiers.py

# 7. Cross-dataset analysis
python cross_dataset_analysis.py
```

## View Results

After running, check:
- `results/evaluation/` - ARI, NMI, confusion matrices
- `results/visualization/` - t-SNE, UMAP plots
- `results/cross_dataset/` - Correlation heatmaps

## Configuration

Edit `config/config.yaml` to adjust:
- `variance_threshold`: Remove low-variance genes (default: 13)
- `similarity_threshold`: Graph edge threshold (default: 0.4)
- `max_communities`: BIGCLAM search range (default: 10)
- `iterations`: Optimization steps (default: 100)

## Troubleshooting

**Memory issues**: Process datasets separately or reduce thresholds  
**Slow performance**: Use GPU or reduce iterations  
**Import errors**: Make sure `source venv/bin/activate` was run

## Expected Runtime

- Preprocessing: ~2-5 minutes per dataset
- Graph construction: ~5-10 minutes per dataset  
- Clustering: ~10-20 minutes per dataset (GPU recommended)
- Evaluation/Visualization: ~5 minutes
- Total: ~1-2 hours for complete pipeline
