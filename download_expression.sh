#!/bin/bash
# Script to download TCGA BRCA gene expression data from UCSC Xena Browser

echo "=========================================="
echo "TCGA BRCA Gene Expression Data Download"
echo "=========================================="
echo ""
echo "This script will help you download from UCSC Xena Browser"
echo ""

# Create data directory if it doesn't exist
mkdir -p data

echo "METHOD 1: UCSC Xena Browser (RECOMMENDED - Easiest)"
echo "---------------------------------------------------"
echo ""
echo "1. Open your web browser and go to:"
echo "   https://xenabrowser.net/datapages/"
echo ""
echo "2. In the search box, type: 'TCGA Breast Cancer (BRCA)'"
echo ""
echo "3. Look for dataset: 'TCGA.BRCA.sampleMap/HiSeqV2_PANCAN'"
echo "   OR search for: 'TCGA.BRCA.sampleMap'"
echo ""
echo "4. Click on the dataset, then click 'Download' button"
echo ""
echo "5. Save the file to: data/TCGA_BRCA_HiSeqV2.tsv"
echo ""
echo "6. After download, run this script again with the file path"
echo ""
echo "=========================================="
echo ""
echo "METHOD 2: Zenodo (Pre-processed)"
echo "---------------------------------"
echo ""
echo "1. Go to: https://zenodo.org/records/4309168"
echo ""
echo "2. Click 'Download' for the BRCA gene expression file"
echo ""
echo "3. Extract and save to data/ folder"
echo ""
echo "=========================================="
echo ""
echo "METHOD 3: GDC Data Portal (Official, but complex)"
echo "---------------------------------------------------"
echo ""
echo "1. Go to: https://portal.gdc.cancer.gov/repository"
echo ""
echo "2. Search for: TCGA-BRCA"
echo ""
echo "3. Filter:"
echo "   - Data Category: Transcriptome Profiling"
echo "   - Data Type: Gene Expression Quantification"
echo "   - Workflow Type: HTSeq - FPKM-UQ"
echo ""
echo "4. Add files to cart and download"
echo ""
echo "=========================================="
echo ""
echo "After downloading, please:"
echo "  1. Save the file to: data/BRCA_gene_expression.tsv (or similar)"
echo "  2. Tell me the filename, and I'll update config.yml"
echo ""

# Check if a file was provided as argument
if [ -n "$1" ] && [ -f "$1" ]; then
    echo "✓ Found file: $1"
    echo ""
    echo "Checking file format..."
    
    # Check first few lines
    head -3 "$1"
    
    echo ""
    echo "Would you like me to:"
    echo "  1. Check the file format compatibility"
    echo "  2. Update config.yml with this file path"
    echo ""
else
    echo ""
    echo "💡 TIP: If you already downloaded the file, provide the path:"
    echo "   ./download_expression.sh /path/to/downloaded/file.tsv"
fi

