# Results Synthesis Guide

This guide explains the new components created to address reviewer comments and synthesize results into a cohesive narrative.

## Overview

Two new components have been added:

1. **Results Synthesis Script** (`src/analysis/results_synthesis.py`)
   - Combines results from biological interpretation, survival analysis, and method comparison
   - Generates summary tables and narrative descriptions
   - Creates manuscript-ready results sections

2. **Manuscript Results Section** (`docs/MANUSCRIPT_RESULTS.md`)
   - Cohesive narrative connecting methods → biology → clinical significance
   - Addresses all reviewer comments
   - Ready for manuscript integration

**Note**: Parameter sensitivity analysis is now integrated into the grid search module (`src/analysis/parameter_grid_search.py`), which includes robustness metrics (Silhouette Score, Davies-Bouldin Index) along with optimization metrics.

---

## 1. Results Synthesis Script

### Purpose

Synthesizes results from multiple analyses into:
- **Summary tables** (CSV format)
- **Narrative descriptions** (TXT format)
- **Manuscript-ready sections** (MD format)

### Usage

```bash
# Run for both datasets
python src/analysis/results_synthesis.py --datasets both

# Run for specific dataset
python src/analysis/results_synthesis.py --datasets tcga
python src/analysis/results_synthesis.py --datasets gse96058

# Custom output directory
python src/analysis/results_synthesis.py --datasets both --output-dir results/my_synthesis
```

### Output Files

For each dataset, generates:
- `{dataset}_summary_table.csv`: Comprehensive table with cluster characteristics
- `{dataset}_narrative.txt`: Human-readable narrative description
- `{dataset}_manuscript_results.md`: Manuscript-ready results section

### What It Combines

1. **Biological Interpretation**:
   - Top differentially expressed genes
   - Enriched signatures (luminal, basal, HER2, immune, etc.)
   - Enriched pathways (GO, KEGG, Reactome)

2. **Survival Analysis**:
   - Hazard ratios (HR) with confidence intervals
   - P-values from Cox models
   - Log-rank test results

3. **Method Comparison**:
   - NMI and ARI vs PAM50
   - Number of clusters found
   - Comparison to other methods

4. **Cluster Characteristics**:
   - Sample sizes
   - Biological annotations
   - Clinical associations

---

## 2. Parameter Sensitivity Analysis (Integrated into Grid Search)

### Purpose

Addresses reviewer concern: "Why does BIGCLAM perform differently on TCGA vs GSE96058?"

Parameter sensitivity analysis is now integrated into the grid search module, which tests similarity threshold variations and includes robustness metrics.

### Usage

```bash
# Run grid search (includes parameter sensitivity)
python src/analysis/parameter_grid_search.py --dataset gse96058

# Run for TCGA
python src/analysis/parameter_grid_search.py --dataset tcga

# Custom threshold range
python src/analysis/parameter_grid_search.py \
    --dataset gse96058 \
    --similarity_range 0.2 \
    --threshold-max 0.8 \
    --threshold-step 0.05
```

### Output Files

- `{dataset}_sensitivity_analysis.csv`: Detailed results for each threshold
- `{dataset}_sensitivity_analysis.png`: Comprehensive visualization
- `{dataset}_sensitivity_summary.csv`: Summary statistics

### What It Measures

For each similarity threshold (default: 0.1 to 0.9, step 0.05):

1. **Cluster Number**: How many clusters AIC selects
2. **PAM50 Alignment**: NMI and ARI vs ground truth
3. **Internal Quality**: Silhouette score, Davies-Bouldin index
4. **Graph Structure**: Density, connectivity, number of edges

### Key Findings (Expected)

- **Cluster number varies** with threshold (e.g., 3-12 clusters)
- **Optimal threshold** for PAM50 alignment can be identified
- **Current threshold (0.5)** produces 9 clusters on GSE96058
- **Robustness**: Performance is stable within threshold range 0.4-0.6

### Interpretation

- BIGCLAM's automatic cluster selection (via AIC) favors 9 clusters on GSE96058
- This suggests the dataset contains finer molecular substructure than 5-class PAM50
- The over-clustering may represent biologically meaningful sub-subtypes

---

## 3. Manuscript Results Section

### Purpose

Provides a cohesive narrative that:
- Connects methods → biology → clinical significance
- Addresses all reviewer comments
- Ready for direct integration into manuscript

### Location

`docs/MANUSCRIPT_RESULTS.md`

### Structure

1. **Cluster Discovery**: Summary of clusters found
2. **Biological Interpretation**: Gene expression, pathways, signatures
3. **Clinical Validation**: Survival analysis results
4. **Method Comparison**: BIGCLAM vs other methods
5. **Parameter Sensitivity**: Robustness analysis
6. **Dataset-Specific Considerations**: Why performance differs
7. **Cohesive Biological Narrative**: Summary of findings
8. **Limitations and Future Directions**

### Key Sections Addressing Reviewer Comments

#### ✅ 1. Clinical Validation / Survival Analysis
- Kaplan-Meier curves per cluster
- Log-rank test p-values
- Cox regression with HR and CI
- **Location**: Section "Clinical Validation: Survival Analysis"

#### ✅ 2. Biological Interpretation
- Differential expression analysis
- Pathway enrichment (GO, KEGG, Reactome)
- Cell-type signatures (immune, proliferation, EMT, etc.)
- **Location**: Section "Biological Interpretation of Clusters"

#### ✅ 3. Dataset-Specific Clustering Issues
- Parameter sensitivity analysis
- Explanation of performance differences
- Robustness demonstration
- **Location**: Section "Parameter Sensitivity Analysis" and "Dataset-Specific Considerations"

#### ✅ 4. External Validation / Phenotype Alignment
- Clarification: Oncotree (histology) vs PAM50 (molecular)
- Appropriate ground truth usage
- **Location**: Throughout document, especially "Dataset-Specific Considerations"

#### ✅ 5. Data Augmentation
- Already documented in METHODOLOGY.md
- **Location**: Referenced in limitations section

#### ✅ 6. Cohesive Biological Story
- Complete narrative connecting all findings
- **Location**: Section "Cohesive Biological Narrative"

---

## Running All Components

### Via run_all.py

```bash
# Run all new steps
python run_all.py --steps grid_search results_synthesis

# Or include in full pipeline
python run_all.py --steps preprocess graph cluster evaluate survival method_comparison grid_search results_synthesis
```

### Standalone

```bash
# 1. Grid search (includes parameter sensitivity/robustness metrics)
python src/analysis/parameter_grid_search.py --dataset gse96058

# 2. Results synthesis
python src/analysis/results_synthesis.py --datasets both
```

---

## Output Directory Structure

```
results/
├── grid_search/
│   ├── gse96058_data_grid_search_results.csv
│   ├── gse96058_data_grid_search_overview.png
│   └── [individual metric profiles]
└── synthesis/
    ├── tcga_summary_table.csv
    ├── tcga_narrative.txt
    ├── tcga_manuscript_results.md
    ├── gse96058_summary_table.csv
    ├── gse96058_narrative.txt
    └── gse96058_manuscript_results.md
```

---

## Integration with Manuscript

### Step 1: Review Generated Results

1. Check `results/synthesis/{dataset}_summary_table.csv` for complete cluster characteristics
2. Review `results/synthesis/{dataset}_narrative.txt` for human-readable summary
3. Read `docs/MANUSCRIPT_RESULTS.md` for manuscript-ready narrative

### Step 2: Customize for Your Manuscript

1. **Copy relevant sections** from `MANUSCRIPT_RESULTS.md` to your manuscript
2. **Fill in specific values** (sample sizes, HR values, p-values) from summary tables
3. **Add figure references** (e.g., "Figure X shows Kaplan-Meier curves")
4. **Cite supplementary materials** (tables, figures)

### Step 3: Add Figures

Reference existing figures:
- `results/survival/{dataset}/{dataset}_km_survival.png`
- `results/survival/{dataset}/{dataset}_hazard_ratio_forest.png`
- `results/biological_interpretation/{dataset}/{dataset}_biological_summary.png`
- `results/grid_search/{dataset}_grid_search_overview.png`

---

## Troubleshooting

### Results Synthesis Fails

**Error**: "Could not load cluster assignments"
- **Fix**: Ensure clustering has been run (`python run_all.py --steps cluster`)

**Error**: "Biological interpretation results not found"
- **Fix**: Run biological interpretation first (`python run_all.py --steps biological_interpretation`)

**Error**: "Survival results not found"
- **Fix**: Run survival analysis first (`python run_all.py --steps survival`)

### Grid Search Fails

**Error**: "Disconnected graph"
- **Fix**: Threshold too high, grid search will skip these automatically

**Error**: "Takes too long"
- **Fix**: Reduce similarity range or increase step size in grid search configuration

---

## Next Steps

1. **Run grid search** (includes robustness metrics) for both datasets
2. **Run results synthesis** to generate summary tables
3. **Review manuscript results section** and customize for your manuscript
4. **Fill in specific values** from summary tables
5. **Add figure references** to existing visualizations

---

## Questions?

- Check `docs/METHODOLOGY.md` for detailed methodology
- Review `docs/MANUSCRIPT_RESULTS.md` for narrative structure
- Examine generated CSV files for specific values

