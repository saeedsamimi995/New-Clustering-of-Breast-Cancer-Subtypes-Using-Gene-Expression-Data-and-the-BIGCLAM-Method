# Survival Analysis Results Summary

## Overview
This document summarizes the survival analysis results for both TCGA-BRCA and GSE96058 datasets using BIGCLAM clustering communities.

---

## GSE96058 Dataset Results

### Sample Information
- **Total samples analyzed:** 3,409
- **Number of clusters:** 5
- **Cluster sizes:**
  - Cluster 0: 697 samples
  - Cluster 1: 655 samples
  - Cluster 2: 652 samples
  - Cluster 3: 703 samples
  - Cluster 4: 702 samples

### Kaplan-Meier Survival Curves
**Key Observations:**
- All clusters start with 100% survival probability at time 0
- **Cluster 0 (Blue, n=697)** and **Cluster 1 (Orange, n=655)** show the **best survival outcomes**
  - At 2,500 days: ~87% and ~86% survival probability, respectively
- **Cluster 4 (Purple, n=702)** shows the **worst survival outcome**
  - At 2,500 days: ~80% survival probability
- **Cluster 2 (Green, n=652)** and **Cluster 3 (Red, n=703)** show intermediate survival
  - At 2,500 days: ~82% and ~85% survival probability, respectively
- Visual differences are present but relatively modest

### Log-Rank Test Results
**Statistical Significance:**
- **All pairwise comparisons show p-values > 0.05** (ranging from 0.38 to 0.97)
- **Conclusion:** No statistically significant differences in survival between clusters
- This suggests that while visual differences exist in the KM curves, they are not statistically significant

**Pairwise Comparisons:**
- Cluster 0 vs 1: p=0.88 (not significant)
- Cluster 0 vs 2: p=0.84 (not significant)
- Cluster 0 vs 3: p=0.74 (not significant)
- Cluster 0 vs 4: p=0.56 (not significant)
- Cluster 1 vs 2: p=0.97 (not significant)
- Cluster 1 vs 3: p=0.87 (not significant)
- Cluster 1 vs 4: p=0.48 (not significant)
- Cluster 2 vs 3: p=0.90 (not significant)
- Cluster 2 vs 4: p=0.48 (not significant)
- Cluster 3 vs 4: p=0.38 (not significant)

### Cox Proportional Hazards Model
**Results:**
- **Cluster variable:** 
  - Coefficient: -0.0041
  - Hazard Ratio (exp(coef)): 0.996
  - p-value: 0.912 (NOT significant)
  - **Interpretation:** Cluster assignment does not significantly predict survival risk

- **Age variable:**
  - Coefficient: 0.0729
  - Hazard Ratio: 1.076
  - p-value: 4.37e-50 (HIGHLY significant)
  - **Interpretation:** Age is a strong predictor of survival - each year of age increases hazard by ~7.6%

**Key Findings:**
1. **Cluster membership is NOT a significant predictor** of survival in GSE96058
2. **Age is a highly significant predictor** of survival
3. The BIGCLAM clusters do not capture survival-relevant biological differences in this dataset

---

## TCGA-BRCA Dataset Results

### Sample Information
- **Total samples analyzed:** 818 (5 clusters)
- **Cluster sizes:**
  - Cluster 0: 175 samples
  - Cluster 1: 157 samples
  - Cluster 2: 148 samples
  - Cluster 3: 171 samples
  - Cluster 4: 167 samples

### Kaplan-Meier Survival Curves
**Key Observations:**
- All clusters start with 100% survival probability at time 0
- **Cluster 3 (Red, n=171)** shows the **worst survival outcome**
  - Drops to ~40% survival by 4,000 days
  - Maintains around 40% survival through 7,000 days
- **Cluster 0 (Blue, n=175)** and **Cluster 2 (Green, n=148)** show **better survival outcomes**
  - Drop to ~20-25% survival by 4,000 days
  - Maintain similar levels thereafter
- **Cluster 1 (Orange, n=157)** and **Cluster 4 (Purple, n=167)** show intermediate survival
  - Cluster 1: ~35% survival at 4,000 days
  - Cluster 4: ~25% survival at 4,000 days
- Curves begin to diverge noticeably after 1,000-2,000 days
- **Note:** Lower survival probabilities compared to GSE96058, likely due to more aggressive disease or longer follow-up

### Log-Rank Test Results
**Statistical Significance:**
- **All pairwise comparisons show p-values > 0.05** (ranging from 0.55 to 0.99)
- **Conclusion:** No statistically significant differences in survival between clusters
- Despite visual differences in KM curves, statistical tests do not support significant differences

**Pairwise Comparisons:**
- Cluster 0 vs 1: p=0.92 (not significant)
- Cluster 0 vs 2: p=0.98 (not significant)
- Cluster 0 vs 3: p=0.55 (not significant)
- Cluster 0 vs 4: p=0.76 (not significant)
- Cluster 1 vs 2: p=0.91 (not significant)
- Cluster 1 vs 3: p=0.74 (not significant)
- Cluster 1 vs 4: p=0.99 (not significant)
- Cluster 2 vs 3: p=0.71 (not significant)
- Cluster 2 vs 4: p=0.81 (not significant)
- Cluster 3 vs 4: p=0.73 (not significant)

### Cox Proportional Hazards Model
**Status: FAILED** ❌

**Why TCGA doesn't have cox_summary.csv:**

The Cox model likely failed for one or more of the following reasons:

1. **Missing adjustment variables:** The model attempts to adjust for `age` and `stage`, but:
   - TCGA clinical data may have missing values for these variables
   - After dropping rows with missing values, insufficient samples may remain
   - The `stage` variable may not be available or properly mapped

2. **Data quality issues:**
   - Missing values in age/stage columns
   - All values being the same (no variation)
   - Convergence problems in the Cox model fitting

3. **Sample size after filtering:**
   - If too many samples are dropped due to missing age/stage data, the model may fail
   - Minimum sample size requirements not met

**Recommendation:** Check the TCGA clinical data for:
- Availability of `age` and `stage` columns
- Missing value patterns
- Consider running Cox model without adjustment variables or with only available variables

---

## Overall Conclusions

### GSE96058:
1. ✅ **Visual differences** in survival curves exist between clusters
2. ❌ **No statistical significance** - clusters do not significantly predict survival
3. ✅ **Age is a strong predictor** of survival (highly significant)
4. ❌ **BIGCLAM clusters** do not capture survival-relevant biological differences

### TCGA-BRCA:
1. ✅ **Visual differences** in survival curves exist, with Cluster 3 showing worse outcomes
2. ❌ **No statistical significance** - clusters do not significantly predict survival
3. ❌ **Cox model failed** - likely due to missing adjustment variables (age/stage)
4. ⚠️ **Need to investigate** why Cox model failed and fix data issues

### Key Insights:
1. **Both datasets show similar patterns:**
   - Visual differences in KM curves
   - No statistical significance in log-rank tests
   - BIGCLAM clusters may not be capturing survival-relevant subtypes

2. **Possible explanations:**
   - Clusters may be based on gene expression patterns not directly related to survival
   - Sample sizes may be insufficient to detect differences
   - Survival may be influenced by factors not captured in the clustering (treatment, comorbidities, etc.)
   - The clustering may need refinement or different parameters

3. **Recommendations:**
   - Investigate why TCGA Cox model failed
   - Consider adjusting clustering parameters or methods
   - Explore whether clusters align with known PAM50 subtypes
   - Consider multivariate analysis with additional clinical variables

---

## Technical Notes

### Log-Rank Test Interpretation:
- p < 0.05: Statistically significant difference in survival
- p ≥ 0.05: No statistically significant difference
- All tests here show p ≥ 0.05, indicating no significant differences

### Cox Model Interpretation:
- Hazard Ratio (HR) > 1: Increased risk
- HR < 1: Decreased risk
- HR = 1: No effect
- For GSE96058: Cluster HR = 0.996 (essentially no effect, not significant)

### Missing Cox Model for TCGA:
The Cox model is wrapped in a try-except block. If it fails, an error message is printed but execution continues. Check the console output when running the survival analysis to see the specific error message.

