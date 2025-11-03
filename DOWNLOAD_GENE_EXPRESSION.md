# Download TCGA BRCA Gene Expression Data

## Option 1: cBioPortal (Same Source as Clinical Data)

### Steps:
1. **Go to cBioPortal TCGA BRCA Study:**
   - URL: https://www.cbioportal.org/study/summary?id=brca_tcga_pub

2. **Download Gene Expression Data:**
   - Click on **"Download"** tab
   - Scroll to **"Gene Expression"** section
   - Look for options like:
     - **"RNA Seq V2 RSEM"** (Recommended)
     - **"RNA Seq V2 RSEM (normalized)"**
     - **"mRNA Expression (microarray)"**
   
3. **Select Download Format:**
   - Choose **"Tab-separated values (.txt)"** or **"CSV"**
   - Click **"Download"**

4. **Save the file:**
   - Save as: `data/Human__TCGA_BRCA__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct.gz`
   - Or any name, then update `config.yml` with the correct path

---

## Option 2: GDC Data Portal (Official TCGA Source)

### Steps:
1. **Go to GDC Data Portal:**
   - URL: https://portal.gdc.cancer.gov/

2. **Search for TCGA-BRCA:**
   - Click **"Repository"** tab
   - Search: `TCGA-BRCA`
   - Select **"TCGA-BRCA"** project

3. **Filter for Gene Expression:**
   - Go to **"Files"** tab
   - Filter by:
     - **Data Category**: `Transcriptome Profiling`
     - **Data Type**: `Gene Expression Quantification`
     - **Workflow Type**: `HTSeq - FPKM` or `STAR - Counts`

4. **Download:**
   - Select files or use **"Manifest"** to download multiple files
   - Files will be in `.tsv` or `.txt` format
   - May need to merge multiple files if samples are in separate files

---

## Option 3: Firehose Legacy Data (Pre-processed)

### Steps:
1. **Go to Broad Firehose:**
   - URL: https://gdac.broadinstitute.org/

2. **Navigate to BRCA:**
   - Click on **"BRCA"** dataset
   - Look for **"mRNA expression"** section
   - Download **"BRCA.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt"**
   - This is already normalized and ready to use

---

## File Format Requirements

Your gene expression file should have:
- **Genes as rows** (or columns - script can transpose)
- **Samples as columns** (TCGA sample IDs like `TCGA-XX-XXXX-01A-01R-XXXX-07`)
- **Expression values** (FPKM, RSEM, or normalized values)

### Expected Sample ID Format:
```
TCGA-A1-A0SB-01A-11R-A115-07
TCGA-A1-A0SD-01A-11R-A115-07
TCGA-A1-A0SE-01A-11R-A115-07
```

---

## After Download

1. **Save to data folder:**
   ```bash
   # If downloaded as .gz file, keep as is
   # If downloaded as .txt/.tsv, you may need to compress or rename
   mv downloaded_file.txt data/Human__TCGA_BRCA__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct.gz
   ```

2. **Update config.yml if needed:**
   ```yaml
   dataset_preparation:
     tcga:
       clinical: "data/brca_tcga_pub_clinical_data.tsv"
       expression: "data/your_downloaded_expression_file.tsv"  # Update this
       output: "data/tcga_brca_data_target_added.csv"
   ```

3. **Verify file format:**
   ```bash
   # Check first few lines
   head -5 data/your_expression_file.tsv
   
   # Check file size (should be large, ~50MB+)
   ls -lh data/your_expression_file.tsv
   ```

---

## Quick Links

- **cBioPortal BRCA**: https://www.cbioportal.org/study/summary?id=brca_tcga_pub
- **GDC Portal**: https://portal.gdc.cancer.gov/
- **Broad Firehose**: https://gdac.broadinstitute.org/

---

## Recommended Source

**cBioPortal** is recommended since:
- ✅ Same source as your clinical data
- ✅ Pre-processed and normalized
- ✅ Easy to download
- ✅ Multiple format options

After downloading, the gene expression data should be ready to use with `data_preparing.py`.

