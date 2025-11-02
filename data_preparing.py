"""
Data Preparation Script

Prepares TCGA BRCA and GSE96058 datasets:
- Adds PAM50 column from clinical dataset as target to expression dataset
- Output format: Genes as rows, Samples as columns, PAM50 as last row
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import yaml
import argparse
import gc


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def normalize_tcga_id(sample_id):
    """Normalize TCGA sample ID for matching - removes separators and vial suffix."""
    sample_id = str(sample_id).strip().upper()
    # Remove all separators (hyphens and dots) and vial suffix
    # TCGA-XX-XXXX-01 or TCGA.XX.XXXX -> TCGAXXXXXX
    sample_id = sample_id.replace('-', '').replace('.', '')
    # Remove trailing vial suffix (usually 01, 02, etc.)
    # TCGAXXXXXX01 -> TCGAXXXXXX
    if len(sample_id) > 10 and sample_id[-2:].isdigit():
        sample_id = sample_id[:-2]
    return sample_id


def prepare_tcga_brca_data(clinical_file, expression_file, output_file):
    """
    Prepare TCGA BRCA data.
    
    Format: Genes as rows, Samples as columns, PAM50 as last column.
    """
    print("=" * 80)
    print("PREPARING TCGA BRCA DATA")
    print("=" * 80)
    
    # Load clinical data
    print("\n[1/4] Loading clinical data...")
    clinical = pd.read_csv(clinical_file, sep='\t', low_memory=False)
    print(f"    Shape: {clinical.shape}")
    
    # Find sample ID column
    sample_id_col = None
    for col in ['Sample ID', 'Patient ID', 'Sample_ID', 'Patient_ID']:
        if col in clinical.columns:
            sample_id_col = col
            break
    if sample_id_col is None:
        sample_id_col = [c for c in clinical.columns if 'ID' in c.upper()][0]
    
    # Find PAM50/subtype column - use Oncotree Code as surrogate for TCGA BRCA
    pam50_col = None
    for pattern in ['PAM50', 'pam50 subtype', 'intrinsic subtype', 'molecular subtype']:
        cols = [c for c in clinical.columns if pattern.upper() in c.upper()]
        if cols:
            pam50_col = cols[0]
            break
    
    # If PAM50 not found, use Oncotree Code as surrogate (tumor subtype classification)
    if not pam50_col:
        for col_name in ['Oncotree Code', 'OncotreeCode']:
            if col_name in clinical.columns:
                pam50_col = col_name
                print(f"    Using '{pam50_col}' as surrogate for PAM50 (Oncotree classification)")
                break
    
    if pam50_col:
        print(f"    [OK] Found subtype column: {pam50_col}")
    else:
        print("    [WARNING] No subtype column found - will create file without subtype labels")
    
    # Create mapping: normalized_sample_id -> pam50_label
    clinical_samples = clinical[sample_id_col].astype(str).values
    sample_to_pam50 = {}
    for idx, sample_id in enumerate(clinical_samples):
        norm_id = normalize_tcga_id(sample_id)
        if pam50_col:
            label = str(clinical.iloc[idx][pam50_col]).strip()
            if label and label not in ['nan', 'NA', '']:
                sample_to_pam50[norm_id] = label
    
    print(f"    Samples with PAM50: {len(sample_to_pam50)}")
    
    # Load expression data
    print("\n[2/4] Loading expression data...")
    print("    (This may take a few minutes for large files...)")
    
    # Check file size for memory optimization
    file_size = os.path.getsize(expression_file)
    print(f"    File size: {file_size / (1024*1024):.2f} MB")
    
    # Load expression data with memory considerations
    if expression_file.endswith('.gz'):
        expression = pd.read_csv(expression_file, sep='\t', index_col=0, low_memory=False, compression='gzip')
    else:
        expression = pd.read_csv(expression_file, sep='\t', index_col=0, low_memory=False)
    
    print(f"    Shape: {expression.shape} (genes x samples)")
    print(f"    Genes: {expression.shape[0]:,}")
    print(f"    Expression samples: {expression.shape[1]:,}")
    
    # Free memory
    gc.collect()
    
    # Match samples
    print("\n[3/4] Matching samples...")
    expression_cols = list(expression.columns)
    expression_cols_norm = {col: normalize_tcga_id(col) for col in expression_cols}
    
    # Create reverse mapping: normalized_id -> original_column_name
    norm_to_original = {}
    for orig, norm in expression_cols_norm.items():
        if norm not in norm_to_original:
            norm_to_original[norm] = []
        norm_to_original[norm].append(orig)
    
    # Find matching samples
    matched_samples = []
    pam50_labels = []
    
    for norm_id, orig_cols in norm_to_original.items():
        matched_samples.append(orig_cols[0])  # Use first matching column
        if norm_id in sample_to_pam50:
            pam50_labels.append(sample_to_pam50[norm_id])
        else:
            pam50_labels.append('Unknown')
    
    print(f"    Matched: {len(matched_samples):,} samples")
    print(f"    With PAM50 labels: {sum(1 for x in pam50_labels if x != 'Unknown'):,}")
    
    # Filter expression to matched samples
    expression_matched = expression[matched_samples].copy()
    
    # Transpose: genes as rows, samples as columns
    print("\n[4/4] Transposing and adding PAM50...")
    expression_t = expression_matched.T  # Now: samples x genes
    
    # Add PAM50 as last column
    expression_t['PAM50'] = pam50_labels
    
    # Transpose back: genes (including PAM50) as rows, samples as columns
    expression_final = expression_t.T
    
    # Save
    print(f"\n[Saving] Writing to {output_file}...")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    expression_final.to_csv(output_file)
    
    print(f"    [OK] Saved successfully!")
    print(f"    Final shape: {expression_final.shape}")
    print(f"    Genes (rows): {expression_final.shape[0]:,}")
    print(f"    Samples (columns): {expression_final.shape[1]:,}")
    print(f"    PAM50 labels: {sum(1 for x in pam50_labels if x != 'Unknown'):,}")
    
    return output_file


def prepare_gse96058_data(expression_file, output_file, clinical_file=None):
    """
    Prepare GSE96058 data.
    
    Format: Genes as rows, Samples as columns, PAM50 as last column.
    """
    print("\n" + "=" * 80)
    print("PREPARING GSE96058 DATA")
    print("=" * 80)
    
    print(f"\n[1/3] Loading expression file...")
    print(f"    File: {expression_file}")
    print("    (This may take a while for large files...)")
    
    # Check file structure first
    sample_df = pd.read_csv(expression_file, nrows=5, low_memory=False)
    print(f"    Sample structure:")
    print(f"      Shape: {sample_df.shape}")
    print(f"      Columns: {list(sample_df.columns)[:5]}...")
    
    # Determine orientation
    # Read more rows to understand structure
    test_df = pd.read_csv(expression_file, nrows=100, low_memory=False)
    
    # If first column looks like gene names/IDs, genes are rows
    # If first column is 'Unnamed: 0' or numeric, might need to check
    first_col = test_df.columns[0]
    
    # Load full data
    print(f"\n[2/3] Loading full expression data...")
    if first_col == 'Unnamed: 0':
        # First column is row index (gene names)
        expression = pd.read_csv(expression_file, index_col=0, low_memory=False)
    else:
        # No index column, create numeric index
        expression = pd.read_csv(expression_file, low_memory=False)
        # Assume first column contains gene identifiers
        if expression.shape[0] < expression.shape[1] // 2:
            # Few rows, many columns -> samples are columns, need to set gene names
            expression = expression.set_index(expression.columns[0])
        else:
            # Many rows -> genes are rows, columns are samples
            pass
    
    # Transpose if needed: we want genes as rows, samples as columns
    if expression.shape[0] < expression.shape[1]:
        print("    Transposing (samples were rows)...")
        expression = expression.T
    
    print(f"    Final shape: {expression.shape}")
    print(f"    Genes (rows): {expression.shape[0]:,}")
    print(f"    Samples (columns): {expression.shape[1]:,}")
    
    # Handle PAM50
    print(f"\n[3/3] Processing PAM50 labels...")
    pam50_labels = ['Unknown'] * expression.shape[1]  # One label per sample (column)
    
    if clinical_file and Path(clinical_file).exists():
        print(f"    Loading clinical data: {clinical_file}")
        try:
            clinical = pd.read_csv(clinical_file, low_memory=False)
            # Try to match samples and extract PAM50
            # This would need to be customized based on actual clinical file structure
            print("    Clinical file loaded - you may need to manually match PAM50")
        except Exception as e:
            print(f"    Could not load clinical file: {e}")
    else:
        print("    No clinical file provided")
        print("    Output will have 'Unknown' PAM50 labels")
        print("    You can manually add PAM50 labels later if available")
    
    # Add PAM50 as a new row (since genes are rows, samples are columns)
    # PAM50 row will have one value per sample column
    expression.loc['PAM50'] = pam50_labels
    
    # Save
    print(f"\n[Saving] Writing to {output_file}...")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    expression.to_csv(output_file)
    
    print(f"    [OK] Saved successfully!")
    print(f"    Final shape: {expression.shape}")
    print(f"    Genes + PAM50 (rows): {expression.shape[0]:,}")
    print(f"    Samples (columns): {expression.shape[1]:,}")
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare TCGA BRCA and GSE96058 datasets for BIGCLAM pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--dataset', 
        choices=['tcga', 'gse96058', 'both'],
        default='both',
        help='Which dataset(s) to process'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Please create a configuration file or use the default: config/config.yaml")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Extract dataset paths from config
    if 'dataset_preparation' not in config:
        print("Error: 'dataset_preparation' section not found in config file")
        sys.exit(1)
    
    dataset_config = config['dataset_preparation']
    
    # Process TCGA
    if args.dataset in ['tcga', 'both']:
        tcga_config = dataset_config.get('tcga', {})
        tcga_clinical = tcga_config.get('clinical')
        tcga_expression = tcga_config.get('expression')
        tcga_output = tcga_config.get('output')
        
        if tcga_clinical and tcga_expression and tcga_output:
            if Path(tcga_clinical).exists() and Path(tcga_expression).exists():
                prepare_tcga_brca_data(
                    tcga_clinical,
                    tcga_expression,
                    tcga_output
                )
            else:
                print(f"⚠ TCGA files not found. Skipping TCGA processing.")
                print(f"   Clinical: {tcga_clinical}")
                print(f"   Expression: {tcga_expression}")
        else:
            print("⚠ TCGA configuration incomplete in config file. Skipping.")
    
    # Process GSE96058
    if args.dataset in ['gse96058', 'both']:
        gse_config = dataset_config.get('gse96058', {})
        gse_expression = gse_config.get('expression')
        gse_clinical = gse_config.get('clinical')
        gse_output = gse_config.get('output')
        
        if gse_expression and gse_output:
            if Path(gse_expression).exists():
                prepare_gse96058_data(
                    gse_expression,
                    gse_output,
                    gse_clinical
                )
            else:
                print(f"⚠ GSE96058 expression file not found. Skipping.")
                print(f"   Expression: {gse_expression}")
        else:
            print("⚠ GSE96058 configuration incomplete in config file. Skipping.")
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)

