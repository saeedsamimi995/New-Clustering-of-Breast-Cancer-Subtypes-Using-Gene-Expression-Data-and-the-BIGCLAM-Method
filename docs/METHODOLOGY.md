# Methodology

## Cohorts and Ground Truth Subtyping

Two independent breast cancer cohorts were analyzed. TCGA-BRCA (n = 1,093) included RNA-seq expression profiles annotated using Oncotree histologic subtypes¹. GSE96058 (n = 3,273) included Affymetrix microarray profiles with clinically assigned PAM50 molecular subtypes². PAM50 labels were used as the primary reference for molecular cluster validation, whereas Oncotree assignments provided external validation of histological consistency. To avoid cross-platform bias, RNA-seq and microarray datasets were analyzed independently with cohort-specific normalization.

## Expression Normalization and Feature Filtering

Expression matrices were log2(+1) transformed to approximate a variance-stabilizing distribution (log2 transformation provides variance stabilization comparable to VST for normalized count data). Genes falling below the mean cross-sample variance were excluded to reduce high-dimensional noise. Z-score normalization¹⁰ was applied per gene across samples, performed independently within each cohort to avoid information leakage.

## Graph Construction from Transcriptomic Similarity

Pairwise cosine similarity was computed between samples using normalized gene expression. Binary adjacency matrices were constructed by thresholding similarity values. Thresholds were optimized via grid search (increments of 0.01), selecting the minimum value ensuring a single connected component while preserving graph sparsity. Threshold stability was verified across ±0.05 intervals (Supplementary Figure X). All graphs were undirected and loop-free.

## Overlapping Community Detection

Breast tumors exhibit mixed molecular programs (e.g., HER2/Luminal hybrids), motivating overlapping clustering. Non-overlapping methods (e.g., K-means, spectral clustering) assign each sample to a single subtype, obscuring hybrid phenotypes that reflect clinical heterogeneity. We applied BIGCLAM³ (Big Community Affiliation Model), which parameterizes community memberships via a non-negative matrix \(\mathbf{F} \in \mathbb{R}_{\geq 0}^{N \times C}\). The probability of an edge is:

\[
P(A_{uv} = 1 | \mathbf{F}) = 1 - \exp(-\mathbf{F}_u^T \mathbf{F}_v)
\]

The optimal \(\mathbf{F}\) maximizes the BIGCLAM log-likelihood:

\[
\mathcal{L}(\mathbf{F}) = \sum_{(u,v) \in E} \log\left(1 - \exp(-\mathbf{F}_u^T \mathbf{F}_v)\right) - \sum_{(u,v) \notin E} \mathbf{F}_u^T \mathbf{F}_v
\]

Number of communities \(C \in [1, 10]\) was estimated using AIC¹¹ or BIC¹²:

\[
\text{AIC} = -2\mathcal{L} + 2k, \quad \text{BIC} = -2\mathcal{L} + k\log N
\]

with \(k = NC\). BIC was used for TCGA (higher sample size, stability), AIC for GSE96058 (smaller parameter-effective ratio). Model selection curves and convergence diagnostics are shown in Supplementary Figure Y. For downstream analyses, samples were assigned to their dominant community:

\[
c(u) = \arg\max_{c} F_{uc}
\]

## Survival Modeling

Kaplan–Meier curves⁴ were generated for each community. Survival distributions were compared using two-sided log-rank tests (α = 0.05). Cox proportional hazards models⁵ estimated hazard ratios adjusting for age and stage (included when ≥50% completeness). Schoenfeld residuals were used to test proportionality; models violating assumptions were stratified. Pairwise log-rank p-values were adjusted via Benjamini–Hochberg FDR¹³. Median survival and 95% CI were computed with Greenwood estimators.

## Differential Gene Expression

For each community, differential expression was measured against all remaining samples using two-sample t-tests on normalized log2 values. Effect sizes were reported as log2 fold-change. Significant genes were defined as \(|\text{log2FC}| \geq 1\) with FDR < 0.05¹³.

## Pathway and Functional Enrichment

Genes were ranked by log2 fold-change and analyzed using GSEA⁶ (weighted statistic, p = 1) with 1,000 phenotype permutations. Gene sets from GO Biological Process⁷, KEGG⁸, and Reactome⁹ were tested. Multiple hypothesis testing used Benjamini–Hochberg FDR correction¹³.

## Computational Implementation

BIGCLAM optimization was implemented in PyTorch with Adam optimizer (lr = 0.08) and early stopping based on log-likelihood stabilization (convergence threshold: ΔlogL < 10⁻⁵ over 10 consecutive epochs). Hyperparameter sensitivity analysis and optimization trajectories are provided in Supplementary Figure Z. Analyses were executed on an 8-core CPU workstation (≥16GB RAM).

**Note on Data Augmentation**: No data augmentation was applied to expression data used for BIGCLAM clustering, survival analysis, differential expression, or pathway enrichment. All core analyses were performed on original, unmodified expression profiles to preserve biological signal integrity. Augmentation (if used) was limited to optional classifier validation and was evaluated with and without augmentation to assess impact (see Supplementary Methods).

## References

1. Cancer Genome Atlas Network. Comprehensive molecular portraits of human breast tumours. *Nature* 490, 61-70 (2012).
2. Parker, J. S. et al. Supervised risk predictor of breast cancer based on intrinsic subtypes. *J. Clin. Oncol.* 27, 1160-1167 (2009).
3. Yang, J. & Leskovec, J. Overlapping community detection at scale: a nonnegative matrix factorization approach. *Proc. 6th ACM Int. Conf. Web Search Data Mining* 587-596 (2013).
4. Kaplan, E. L. & Meier, P. Nonparametric estimation from incomplete observations. *J. Am. Stat. Assoc.* 53, 457-481 (1958).
5. Cox, D. R. Regression models and life-tables. *J. R. Stat. Soc. B* 34, 187-202 (1972).
6. Subramanian, A. et al. Gene set enrichment analysis: a knowledge-based approach for interpreting genome-wide expression profiles. *Proc. Natl. Acad. Sci. USA* 102, 15545-15550 (2005).
7. Ashburner, M. et al. Gene ontology: tool for the unification of biology. *Nat. Genet.* 25, 25-29 (2000).
8. Kanehisa, M. & Goto, S. KEGG: kyoto encyclopedia of genes and genomes. *Nucleic Acids Res.* 28, 27-30 (2000).
9. Jassal, B. et al. The reactome pathway knowledgebase. *Nucleic Acids Res.* 48, D498-D503 (2020).
10. Snedecor, G. W. & Cochran, W. G. *Statistical Methods* (Iowa State University Press, 1989).
11. Akaike, H. Information theory and an extension of the maximum likelihood principle. *Proc. 2nd Int. Symp. Information Theory* 267-281 (1973).
12. Schwarz, G. Estimating the dimension of a model. *Ann. Stat.* 6, 461-464 (1978).
13. Benjamini, Y. & Hochberg, Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *J. R. Stat. Soc. B* 57, 289-300 (1995).
