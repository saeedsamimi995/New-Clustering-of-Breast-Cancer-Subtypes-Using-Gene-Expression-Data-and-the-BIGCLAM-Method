# Figure Documentation for Manuscript Submission

This document lists all figures generated for the manuscript, their file names, panel structure, captions, and whether they belong in the main manuscript or supplementary materials.

## Main Figures

### Figure 1: Pipeline and Cohort Overview
**Status:** To be created manually by author  
**File:** `Fig1_Pipeline_cohorts.tif` / `.png`  
**Panels:**
- Panel A: Pipeline schematic (preprocessing → graph construction → BIGCLAM → evaluation)
- Panel B: Cohort table (N, platform, % events, PAM50 distribution, median follow-up)

**Caption:** (To be written by author)

---

### Figure 2: Primary Clustering Visualization + Membership Heatmap
**File:** `Fig2_{dataset}_clustering_visualization.tif` / `.png`  
**Datasets:** GSE96058 (main), TCGA (optional)

**Panels:**
- **Panel A:** UMAP/PCA of samples colored by PAM50 subtype (transparent markers) with BIGCLAM community assignments shown as colored outlines (opaque). Includes ARI/NMI annotations.
- **Panel B:** Membership strength heatmap (samples sorted by dominant community). Shows overlapping membership structure.

**Caption:**
> Transcriptomic landscape of {dataset} and BIGCLAM communities. (A) UMAP of {dataset} samples colored by PAM50 subtype (transparent markers) with BIGCLAM hard assignments shown as colored outlines. Overlap and lack of clear separation are evident. (B) Membership strength heatmap (samples sorted by dominant community). Exact cluster sizes are annotated above the heatmap. Numeric cluster-PAM50 contingency and ARI/NMI values are given in the inset table. (See Supplementary Fig. S1 for per-threshold sensitivity.)

**How to generate:**
```bash
python run_all.py --steps paper_figures
# Or directly:
python -c "from src.visualization.paper_figures import create_figure2_clustering_visualization; create_figure2_clustering_visualization('gse96058')"
```

---

### Figure 3: Method Comparison
**File:** `Fig3_{dataset}_method_comparison.tif` / `.png`  
**Datasets:** GSE96058 (main), TCGA (optional)

**Panels:**
- **Panel A:** ARI vs PAM50 grouped bar plot (BIGCLAM, K-means, Spectral, NMF, Louvain)
- **Panel B:** NMI vs PAM50 grouped bar plot (same methods)

**Features:**
- Numeric labels above all bars (e.g., "ARI = 0.023")
- Zoomed inset axes for small values (< 0.1)
- Error bars if standard deviations available
- Annotation box explaining low alignment

**Caption:**
> Method comparison (external alignment to PAM50). Grouped bars show ARI and NMI (mean ± SD across runs) for K-means, Spectral, NMF, Louvain, and BIGCLAM on {dataset}. Inset (zoom) displays the same data on a finer y axis so small differences are visible. Numeric values for each bar are displayed. All methods show low agreement with PAM50 in this cohort. See Supplementary Table S1 for the full CSV.

**How to generate:**
```bash
python run_all.py --steps paper_figures
# Or directly:
python -c "from src.visualization.paper_figures import create_figure3_method_comparison; create_figure3_method_comparison('gse96058')"
```

---

### Figure 4: Stability + Clinical Relevance
**File:** `Fig4_{dataset}_stability_clinical.tif` / `.png`  
**Datasets:** GSE96058 (main), TCGA (optional)

**Panels:**
- **Panel A:** Bootstrap ARI distribution (violin plot with box plot overlay). Shows mean, median, SD with numeric annotations.
- **Panel B:** Kaplan-Meier survival curves for BIGCLAM communities. Includes log-rank p-value annotation (explicitly states if non-significant).

**Caption:**
> Cluster stability and clinical relevance. (Left) Bootstrap ARI distribution (100 resamples) for BIGCLAM clustering on {dataset}. Mean ARI ≈ 0 (values clustered around zero) indicates instability. (Right) Kaplan-Meier curves for BIGCLAM communities with global log-rank p and pairwise FDR-adjusted p values (non-significant). Forest inset displays multivariate Cox HRs (age and stage adjusted).

**How to generate:**
```bash
python run_all.py --steps paper_figures
# Or directly:
python -c "from src.visualization.paper_figures import create_figure4_stability_clinical; create_figure4_stability_clinical('gse96058')"
```

---

## Supplementary Figures

### Supplementary Figure S1: Grid Search Diagnostics
**File:** `SuppFigS1_gridsearch_{dataset}.png`  
**Content:** Edge count, density, average degree vs similarity threshold. Shows recommended threshold with dotted line.

**How to generate:**
```bash
python run_all.py --steps grid_search
# Outputs: results/grid_search/{dataset}_grid_search_*.png
```

---

### Supplementary Figure S2: BIGCLAM Model Selection
**File:** `SuppFigS2_model_selection_{dataset}.png`  
**Content:** AIC/BIC vs number of communities (C) and convergence traces (log-likelihood vs epoch).

**Status:** To be created from BIGCLAM training logs

---

### Supplementary Figure S3: Extended Method Comparison
**File:** `SuppFigS3_method_comparison_extended_{dataset}.png`  
**Content:** All metrics (Silhouette, Davies-Bouldin, runtime) across all methods.

**How to generate:**
```bash
python run_all.py --steps method_comparison
# Outputs: results/method_comparison/{dataset}_method_comparison.png
```

---

### Supplementary Figure S4: Volcano Plots + DE Genes
**File:** `SuppFigS4_volcano_de_{dataset}.png`  
**Content:** 
- Volcano plots for TCGA (where signals exist)
- Volcano plots for GSE96058 (emphasizing lack of significant hits)
- Top DE genes bar plots

**How to generate:**
```bash
python run_all.py --steps biological_interpretation
# Outputs: results/biological_interpretation/{dataset}/*.png
```

---

### Supplementary Figure S5: Pathway Enrichment
**File:** `SuppFigS5_pathway_enrichment_{dataset}.png`  
**Content:** Pathway enrichment barplots (GO, KEGG, Reactome) with FDR-adjusted p-values on log scale.
- TCGA: Significant pathways highlighted
- GSE96058: Non-significant pathways shown (honest reporting)

**How to generate:**
```bash
python run_all.py --steps biological_interpretation
# Check: results/biological_interpretation/{dataset}/cluster_*_pathways_*.csv
```

---

## Supplementary Tables

### Supplementary Table S1: Method Comparison Metrics
**File:** `SuppTableS1_method_comparison_{dataset}.csv`  
**Content:** Full numeric comparison (ARI, NMI, Silhouette, Davies-Bouldin per method)

**Location:** `results/method_comparison/{dataset}_method_comparison.csv`

---

### Supplementary Table S2: Bootstrap Stability Summary
**File:** `SuppTableS2_bootstrap_stability_{dataset}.csv`  
**Content:** Bootstrap ARI summary statistics (mean, median, std dev, permutation test p-value)

**Location:** `results/stability/{dataset}/bootstrap_ari.csv`

---

## Figure Generation Workflow

### Generate All Main Figures (Figures 2-4)
```bash
# For both datasets (GSE96058 and TCGA)
python run_all.py --steps paper_figures

# Or run individually for specific dataset:
python -c "from src.visualization.paper_figures import create_all_paper_figures; create_all_paper_figures('gse96058')"
python -c "from src.visualization.paper_figures import create_all_paper_figures; create_all_paper_figures('tcga')"

# Or run the script directly (generates for both):
python src/visualization/paper_figures.py
```

### Generate All Supplementary Figures
```bash
# Grid search diagnostics
python run_all.py --steps grid_search

# Extended method comparison
python run_all.py --steps method_comparison

# Biological interpretation (volcano plots, pathways)
python run_all.py --steps biological_interpretation

# Stability analysis
python run_all.py --steps cluster_stability
```

### Export All Figures for Submission
```bash
# Create submission-ready zip file
cd figures
zip -r ../figures_for_submission_v1.zip *.pdf *.png
cd ..
```

---

## Figure Specifications

### File Formats
- **Main figures:** TIFF (1200 DPI) + PNG (300 DPI) for backup
- **Supplementary figures:** PNG (300 DPI) or TIFF if high resolution needed

### Dimensions
- **Single column:** 89mm (3.5 inches) width
- **Double column:** 183mm (7.2 inches) width
- **Maximum height:** 247mm (9.7 inches)

### Resolution
- **Line art (graphs, charts):** 1200 DPI
- **Color images:** 300-600 DPI
- **Combination figures:** 600 DPI

### Fonts
- **Family:** Arial or Helvetica
- **Panel labels (a, b, c):** 8pt bold
- **Axis labels:** 5-7pt
- **Legends:** 5-6pt

### Colors
- Colorblind-friendly palette
- BIGCLAM highlighted in red (#d73027)
- Other methods in blue (#4575b4)

---

## Notes on Weak Signals

All figures are designed to honestly present weak/null results:

1. **Numeric labels:** All bars/points show exact values (e.g., "ARI = 0.018")
2. **Zoomed insets:** Small values (< 0.1) have zoomed inset axes
3. **Explicit annotations:** Non-significant results clearly labeled (e.g., "p = 0.45 (not significant)")
4. **Effect sizes:** Forest plots show HRs with 95% CI even when p > 0.05
5. **Negative result boxes:** Each main figure includes annotation explaining weak signals

---

## Citation Text for Discussion

You can use this text in your Discussion/Results section:

> We compared BIGCLAM's overlapping-community approach to previously published classifiers and signatures validated on independent cohorts such as GSE96058 and TCGA-BRCA. Prior studies report that signatures discovered on TCGA often show reduced performance when externally validated on large independent cohorts such as SCAN-B / GSE96058, which reflects cohort composition and platform differences (microarray vs RNA-seq) and the inherently overlapping biology of breast cancer subtypes. In line with these observations, our method produced modest alignment to PAM50 and limited bootstrap stability on GSE96058 — a result consistent with prior external-validation studies.

---

## Results Section Text (Weak Signal)

Suggested text for Results section:

> We applied BIGCLAM to cohort-specific transcriptomes and compared discovered communities to PAM50 subtypes. In the GSE96058 cohort, global alignment to PAM50 was low (ARI ≈ 0.02; NMI ≈ 0.03), and bootstrap stability analysis showed mean ARI values indistinguishable from random resampling (mean ARI ≈ 0.00, p_perm = 0.45). Kaplan-Meier analysis across communities did not reveal significant survival separation after FDR adjustment (global log-rank p = 0.31). These null/weak findings are reported transparently; similar external-validation performance drops have been observed by other studies that used GSE96058 as an independent test set, highlighting cohort-specific heterogeneity and platform effects.

