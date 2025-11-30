"""
Prepare GSE96058 data from GEO Series Matrix files.

This script:
1. Extracts expression data from GSE96058-GPL11154 and GSE96058-GPL18573 series matrix files
2. Combines them into a single expression matrix
3. Extracts clinical data (PAM50, survival, age, etc.) from metadata
4. Creates gse96058_data.csv (expression only)
5. Creates gse96058_data_target_added.csv (expression + PAM50)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import sys
import os


def parse_geo_series_matrix(file_path):
    """
    Parse a GEO Series Matrix file to extract:
    - Expression data (after !series_matrix_table_begin)
    - Sample metadata (Sample_characteristics_ch1 rows)
    
    Returns:
        tuple: (expression_df, metadata_dict)
            - expression_df: DataFrame with genes as rows, samples as columns
            - metadata_dict: dict mapping metadata field names to lists of values per sample
    """
    print(f"\n[Parsing] {file_path}...")
    
    # Find the start of the data table
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find where the data table starts
    table_start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == '!series_matrix_table_begin':
            table_start_idx = i + 1
            break
    
    if table_start_idx is None:
        raise ValueError("Could not find !series_matrix_table_begin in file")
    
    # Find where the data table ends
    table_end_idx = None
    for i in range(table_start_idx, len(lines)):
        if lines[i].strip() == '!series_matrix_table_end':
            table_end_idx = i
            break
    
    if table_end_idx is None:
        raise ValueError("Could not find !series_matrix_table_end in file")
    
    # Read the expression data table
    print(f"    Reading expression data (lines {table_start_idx+1} to {table_end_idx})...")
    expression_df = pd.read_csv(
        file_path,
        sep='\t',
        skiprows=table_start_idx,
        nrows=table_end_idx - table_start_idx - 1,
        low_memory=False
    )
    
    # First column should be ID_REF (gene identifiers)
    if expression_df.columns[0] != 'ID_REF':
        raise ValueError(f"Expected first column to be 'ID_REF', got '{expression_df.columns[0]}'")
    
    # Set ID_REF as index
    expression_df = expression_df.set_index('ID_REF')
    
    # Remove quotes from column names (sample IDs)
    expression_df.columns = [col.strip('"') for col in expression_df.columns]
    
    print(f"    Expression data shape: {expression_df.shape}")
    print(f"    Genes: {expression_df.shape[0]:,}, Samples: {expression_df.shape[1]:,}")
    
    # Extract metadata from lines before table_start
    print(f"    Extracting metadata from header...")
    metadata_dict = {}
    sample_ids = list(expression_df.columns)
    
    for i, line in enumerate(lines[:table_start_idx]):
        if line.startswith('Sample_characteristics_ch1'):
            # Parse the metadata line
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            
            # Extract field name and values
            # Format: "field_name: value1"	"field_name: value2" ...
            field_name = None
            values = []
            
            for j, part in enumerate(parts[1:], 1):  # Skip first column (field name)
                if j > len(sample_ids):
                    break
                
                part = part.strip('"')
                # Extract value after ": "
                if ': ' in part:
                    if field_name is None:
                        # Extract field name from first value
                        field_name = part.split(': ')[0]
                    # Extract value
                    value = part.split(': ', 1)[1]
                    values.append(value)
                else:
                    values.append(part)
            
            if field_name and len(values) == len(sample_ids):
                metadata_dict[field_name] = values
                print(f"      Found metadata: {field_name} ({len(values)} values)")
    
    return expression_df, metadata_dict


def combine_gse96058_datasets(file1_path, file2_path, output_file):
    """
    Combine two GSE96058 datasets (GPL11154 and GPL18573) into one.
    
    Args:
        file1_path: Path to GSE96058-GPL11154_series_matrix.txt
        file2_path: Path to GSE96058-GPL18573_series_matrix.txt
        output_file: Path to save combined expression data (gse96058_data.csv)
    
    Returns:
        tuple: (combined_expression_df, combined_metadata_dict)
    """
    print("\n" + "=" * 80)
    print("COMBINING GSE96058 DATASETS")
    print("=" * 80)
    
    # Parse both files
    expr1, meta1 = parse_geo_series_matrix(file1_path)
    expr2, meta2 = parse_geo_series_matrix(file2_path)
    
    print(f"\n[Combining] Expression data...")
    print(f"    GPL11154: {expr1.shape[0]:,} genes × {expr1.shape[1]:,} samples")
    print(f"    GPL18573: {expr2.shape[0]:,} genes × {expr2.shape[1]:,} samples")
    
    # Find common genes (intersection)
    common_genes = expr1.index.intersection(expr2.index)
    print(f"    Common genes: {len(common_genes):,}")
    
    if len(common_genes) == 0:
        raise ValueError("No common genes found between the two platforms!")
    
    # Combine expression data (only common genes)
    combined_expr = pd.concat([
        expr1.loc[common_genes],
        expr2.loc[common_genes]
    ], axis=1)
    
    print(f"    Combined: {combined_expr.shape[0]:,} genes × {combined_expr.shape[1]:,} samples")
    
    # Combine metadata
    print(f"\n[Combining] Metadata...")
    combined_metadata = {}
    
    # Get all unique metadata fields
    all_fields = set(meta1.keys()) | set(meta2.keys())
    
    for field in all_fields:
        values1 = meta1.get(field, [None] * expr1.shape[1])
        values2 = meta2.get(field, [None] * expr2.shape[1])
        combined_metadata[field] = values1 + values2
        print(f"      {field}: {len(combined_metadata[field])} values")
    
    # Save combined expression data
    print(f"\n[Saving] Combined expression data to {output_file}...")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    combined_expr.to_csv(output_file)
    print(f"    [OK] Saved {combined_expr.shape[0]:,} genes × {combined_expr.shape[1]:,} samples")
    
    return combined_expr, combined_metadata


def create_target_added_file(expression_file, metadata_dict, output_file):
    """
    Add PAM50 and clinical data as rows to the expression data.
    
    Format: Genes as rows, Samples as columns, PAM50 and clinical data as additional rows.
    
    Args:
        expression_file: Path to gse96058_data.csv
        metadata_dict: Dictionary of metadata fields and values
        output_file: Path to save gse96058_data_target_added.csv
    """
    print("\n" + "=" * 80)
    print("ADDING TARGETS AND CLINICAL DATA")
    print("=" * 80)
    
    # Load expression data
    print(f"\n[Loading] Expression data from {expression_file}...")
    expression = pd.read_csv(expression_file, index_col=0, low_memory=False)
    print(f"    Shape: {expression.shape}")
    
    # Ensure metadata values match number of samples
    n_samples = expression.shape[1]
    print(f"\n[Processing] Metadata for {n_samples} samples...")
    
    # Extract and add PAM50
    pam50_values = metadata_dict.get('pam50 subtype', [None] * n_samples)
    if len(pam50_values) != n_samples:
        print(f"    [Warning] PAM50 values count ({len(pam50_values)}) != samples ({n_samples})")
        if len(pam50_values) < n_samples:
            pam50_values.extend([None] * (n_samples - len(pam50_values)))
        else:
            pam50_values = pam50_values[:n_samples]
    
    # Clean PAM50 values (remove "pam50 subtype: " prefix if present)
    pam50_clean = []
    for val in pam50_values:
        if val is None:
            pam50_clean.append('Unknown')
        elif isinstance(val, str) and 'pam50 subtype: ' in val.lower():
            pam50_clean.append(val.split(': ', 1)[1] if ': ' in val else val)
        else:
            pam50_clean.append(str(val) if val is not None else 'Unknown')
    
    expression.loc['PAM50'] = pam50_clean
    print(f"    Added PAM50: {len([v for v in pam50_clean if v != 'Unknown'])} non-unknown values")
    
    # Add other clinical metadata as rows
    clinical_fields = {
        'age at diagnosis': 'age',
        'overall survival days': 'OS_time',
        'overall survival event': 'OS_event',
        'tumor size': 'tumor_size',
        'lymph node status': 'lymph_node_status',
        'nhg': 'nhg',
        'er status': 'er_status',
        'her2 status': 'her2_status',
        'ki67 status': 'ki67_status'
    }
    
    for metadata_field, row_name in clinical_fields.items():
        values = metadata_dict.get(metadata_field, [None] * n_samples)
        if len(values) != n_samples:
            if len(values) < n_samples:
                values.extend([None] * (n_samples - len(values)))
            else:
                values = values[:n_samples]
        
        # Clean values (remove field name prefix if present)
        clean_values = []
        for val in values:
            if val is None:
                clean_values.append('NA')
            elif isinstance(val, str) and f'{metadata_field}: ' in val.lower():
                clean_values.append(val.split(': ', 1)[1] if ': ' in val else val)
            else:
                clean_values.append(str(val) if val is not None else 'NA')
        
        expression.loc[row_name] = clean_values
        non_na_count = len([v for v in clean_values if v != 'NA' and v != ''])
        print(f"    Added {row_name}: {non_na_count} non-NA values")
    
    # Save
    print(f"\n[Saving] Target-added data to {output_file}...")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    expression.to_csv(output_file)
    print(f"    [OK] Saved successfully!")
    print(f"    Final shape: {expression.shape}")
    print(f"    Genes + targets (rows): {expression.shape[0]:,}")
    print(f"    Samples (columns): {expression.shape[1]:,}")
    
    return output_file


if __name__ == "__main__":
    import argparse
    import os
    import yaml
    
    parser = argparse.ArgumentParser(
        description="Prepare GSE96058 data from GEO Series Matrix files"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yml',
        help='Path to config file'
    )
    parser.add_argument(
        '--file1',
        type=str,
        default='data/GSE96058-GPL11154_series_matrix.txt',
        help='Path to GSE96058-GPL11154 series matrix file'
    )
    parser.add_argument(
        '--file2',
        type=str,
        default='data/GSE96058-GPL18573_series_matrix.txt',
        help='Path to GSE96058-GPL18573 series matrix file'
    )
    parser.add_argument(
        '--output-data',
        type=str,
        default='data/gse96058_data.csv',
        help='Output path for combined expression data'
    )
    parser.add_argument(
        '--output-targets',
        type=str,
        default='data/gse96058_data_target_added.csv',
        help='Output path for expression data with targets'
    )
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not Path(args.file1).exists():
        print(f"❌ Error: File not found: {args.file1}")
        sys.exit(1)
    
    if not Path(args.file2).exists():
        print(f"❌ Error: File not found: {args.file2}")
        sys.exit(1)
    
    try:
        # Step 1: Combine datasets
        combined_expr, combined_metadata = combine_gse96058_datasets(
            args.file1,
            args.file2,
            args.output_data
        )
        
        # Step 2: Add targets and clinical data
        create_target_added_file(
            args.output_data,
            combined_metadata,
            args.output_targets
        )
        
        print("\n" + "=" * 80)
        print("✅ GSE96058 DATA PREPARATION COMPLETE")
        print("=" * 80)
        print(f"\nOutput files:")
        print(f"  - Expression data: {args.output_data}")
        print(f"  - Expression + targets: {args.output_targets}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

