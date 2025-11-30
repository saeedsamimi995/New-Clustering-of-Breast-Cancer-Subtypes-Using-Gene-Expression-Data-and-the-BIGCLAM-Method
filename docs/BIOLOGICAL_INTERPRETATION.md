# Biological Interpretation of BIGCLAM Clusters

This module provides comprehensive biological interpretation of BIGCLAM-discovered clusters through:

1. **Differential Gene Expression Analysis**
2. **Pathway Enrichment Analysis** (GO, KEGG, Reactome)
3. **Cell-Type Signature Analysis**

## Features

### 1. Differential Expression Analysis

For each cluster, identifies genes that are:
- **Upregulated**: Higher expression in cluster vs. all other samples
- **Downregulated**: Lower expression in cluster vs. all other samples

**Output:**
- `cluster_{N}_differential_expression.csv`: All significant DE genes with:
  - Log2 fold change
  - P-values (adjusted for multiple testing)
  - Mean expression in cluster vs. others

**Example:**
```
Cluster 1
Highly expressed: FOXA1, ESR1, PGR, GATA3, TFF1
```

### 2. Pathway Enrichment Analysis

Uses Enrichr API (via `gseapy`) to identify enriched pathways:
- **GO Biological Processes**: Gene Ontology biological processes
- **KEGG Pathways**: Kyoto Encyclopedia of Genes and Genomes
- **Reactome**: Reactome pathway database

**Output:**
- `cluster_{N}_pathways_{database}.csv`: Enriched pathways with:
  - Pathway name
  - Adjusted P-value
  - Genes in pathway
  - Overlap statistics

**Example:**
```
Cluster 1
Enriched pathways: 
  - Estrogen receptor signaling pathway (GO:0030520)
  - Hormone-mediated signaling pathway (GO:0009755)
  - Breast cancer pathway (KEGG:05224)
```

### 3. Cell-Type Signature Analysis

Calculates signature scores for breast cancer-relevant gene sets:

- **Luminal/Hormonal**: ESR1, PGR, FOXA1, GATA3, etc.
- **Basal-like**: KRT5, KRT14, TP63, etc.
- **HER2-enriched**: ERBB2, GRB7, etc.
- **Immune Infiltration**: CD8A, CD3D, CXCL9, IFNG, etc.
- **Proliferation**: MKI67, TOP2A, CCNB1, etc.
- **EMT**: VIM, FN1, SNAI1, TWIST1, etc.
- **Angiogenesis**: VEGFA, FLT1, KDR, etc.

**Output:**
- `signature_scores.csv`: Signature scores for each cluster with:
  - Mean signature score in cluster
  - Mean signature score in others
  - Statistical significance (p-value)
  - Enrichment status

**Example:**
```
Cluster 1
Signatures: luminal_hormonal, proliferation
  - High ESR1/PGR expression → Luminal subtype
  - Moderate proliferation → Luminal B-like
```

## Usage

### Command Line

```bash
# Run for single dataset
python3 src/interpretation/biological_interpretation.py --dataset tcga

# Run for both datasets
python3 src/interpretation/biological_interpretation.py --dataset both

# Custom thresholds
python3 src/interpretation/biological_interpretation.py \
    --dataset tcga \
    --log2fc-threshold 1.5 \
    --pvalue-threshold 0.01
```

### Integrated Pipeline

```bash
# Run as part of full pipeline
python3 run_all.py --steps biological_interpretation

# Or include in full run
python3 run_all.py
```

## Output Structure

```
results/biological_interpretation/
├── tcga/
│   ├── cluster_0_differential_expression.csv
│   ├── cluster_0_pathways_GO_Biological_Process_2021.csv
│   ├── cluster_0_pathways_KEGG_2021_Human.csv
│   ├── cluster_0_pathways_Reactome_2022.csv
│   ├── cluster_1_differential_expression.csv
│   ├── ...
│   ├── signature_scores.csv
│   ├── biological_interpretation_summary.txt
│   └── biological_interpretation_results.pkl
└── gse96058/
    └── (same structure)
```

## Interpretation Format

The module generates interpretations in the format:

```
Cluster 1
Highly expressed: FOXA1, ESR1, PGR, GATA3, TFF1
Enriched pathways: Estrogen receptor signaling, Hormone-mediated signaling
Signatures: luminal_hormonal, proliferation
→ Corresponds to luminal tumors with hormone receptor expression
```

## Dependencies

### Required
- `numpy`
- `pandas`
- `scipy`

### Optional (for pathway enrichment)
- `gseapy`: For pathway enrichment via Enrichr API
  ```bash
  pip install gseapy
  ```
- `statsmodels`: For FDR correction (alternative to basic Bonferroni)
  ```bash
  pip install statsmodels
  ```

**Note:** The module works without these optional dependencies, but pathway enrichment will be skipped if `gseapy` is not available.

## Example Results

### Cluster 1 (Luminal-like)
- **Highly expressed**: FOXA1, ESR1, PGR, GATA3, TFF1, TFF3
- **Enriched pathways**: 
  - Estrogen receptor signaling pathway
  - Hormone-mediated signaling pathway
  - Breast cancer pathway
- **Signatures**: luminal_hormonal ↑, proliferation ↑
- **Interpretation**: Luminal B subtype with high hormone receptor expression and moderate proliferation

### Cluster 3 (Immune-enriched)
- **Highly expressed**: CXCL9, CD8A, CD3D, IFNG, GZMA, GZMB
- **Enriched pathways**:
  - T cell activation
  - Immune response
  - Cytokine signaling
- **Signatures**: immune_infiltration ↑, proliferation ↓
- **Interpretation**: Immunogenic subtype with high T-cell infiltration

### Cluster 5 (Basal-like)
- **Highly expressed**: KRT5, KRT14, TP63, CDH3, TRIM29
- **Enriched pathways**:
  - Keratinization
  - Cell adhesion
  - Basal cell differentiation
- **Signatures**: basal_like ↑, luminal_hormonal ↓
- **Interpretation**: Basal-like subtype with low hormone receptor expression

## Integration with Manuscript

The biological interpretation results can be directly used in the manuscript:

1. **Results Section**: Report top DE genes, enriched pathways, and signature scores for each cluster
2. **Discussion Section**: Interpret biological meaning of each cluster
3. **Figures**: Create heatmaps of signature scores, pathway enrichment plots
4. **Tables**: Summary tables of DE genes and pathways per cluster

## Notes

- **Gene name matching**: The module handles gene name variations (e.g., "ESR1.1" → "ESR1")
- **Multiple testing**: P-values are adjusted using FDR (Benjamini-Hochberg) when `statsmodels` is available
- **Signature scoring**: Uses mean expression of signature genes (can be extended to GSVA or ssGSEA)
- **Pathway databases**: Uses Enrichr API which requires internet connection

