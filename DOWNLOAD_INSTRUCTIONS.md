# Download TCGA BRCA Gene Expression Data

Since the old Firehose links are broken, here are **working methods** to download the gene expression data:

## ✅ METHOD 1: UCSC Xena Browser (EASIEST - RECOMMENDED)

**Step-by-step:**

1. **Go to UCSC Xena Browser:**
   - URL: https://xenabrowser.net/datapages/

2. **Search for TCGA BRCA:**
   - In the search box, type: **"TCGA Breast Cancer (BRCA)"**
   - Or search: **"TCGA.BRCA"**

3. **Find the Gene Expression Dataset:**
   - Look for: **"TCGA.BRCA.sampleMap/HiSeqV2_PANCAN"**
   - This is the RNA-seq gene expression data

4. **Download:**
   - Click on the dataset
   - Click the **"Download"** button
   - Save as: `data/TCGA_BRCA_HiSeqV2.tsv`

5. **Verify the download:**
   ```bash
   # Check file size (should be ~50MB+)
   ls -lh data/TCGA_BRCA_HiSeqV2.tsv
   
   # Check first few lines
   head -5 data/TCGA_BRCA_HiSeqV2.tsv
   ```

---

## ✅ METHOD 2: Zenodo Repository (Pre-processed)

1. **Go to Zenodo:**
   - URL: https://zenodo.org/records/4309168

2. **Download:**
   - Click **"Download"** button
   - Extract the file
   - Save to `data/` folder

---

## ✅ METHOD 3: GDC Data Portal (Official Source)

1. **Go to GDC Portal:**
   - URL: https://portal.gdc.cancer.gov/repository

2. **Search:**
   - Type: **TCGA-BRCA**

3. **Filter:**
   - **Data Category**: `Transcriptome Profiling`
   - **Data Type**: `Gene Expression Quantification`
   - **Workflow Type**: `HTSeq - FPKM-UQ` or `STAR - Counts`

4. **Download:**
   - Add files to cart
   - Download using GDC Data Transfer Tool (requires installation)

**Note:** GDC downloads individual sample files that may need to be merged.

---

## After Download

1. **Save the file** to your `data/` folder

2. **Tell me the filename**, and I'll:
   - Update `config.yml` with the correct path
   - Verify the file format
   - Test it with your data preparation script

---

## Quick Links

- **UCSC Xena**: https://xenabrowser.net/datapages/
- **Zenodo**: https://zenodo.org/records/4309168
- **GDC Portal**: https://portal.gdc.cancer.gov/repository

---

## Recommended

**Use UCSC Xena Browser (Method 1)** because:
- ✅ Simple web interface
- ✅ One-click download
- ✅ Pre-processed and ready to use
- ✅ Compatible with your clinical data

