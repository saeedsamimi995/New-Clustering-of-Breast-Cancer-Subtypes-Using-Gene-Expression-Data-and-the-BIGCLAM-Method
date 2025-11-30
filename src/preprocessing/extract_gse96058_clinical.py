"""
Extract clinical data from GSE96058 GEO Series Matrix metadata files.

This script extracts PAM50, survival, and other clinical data from the metadata
and saves it in a format that can be used for survival analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os


def extract_metadata_from_geo_file(file_path):
    """
    Extract metadata from a GEO Series Matrix file.
    
    Returns:
        dict: Dictionary mapping metadata field names to lists of values per sample
    """
    print(f"\n[Extracting] Metadata from {file_path}...")
    
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
    
    # Read the header row to get sample IDs
    header_line = lines[table_start_idx].strip()
    header_parts = header_line.split('\t')
    sample_ids = [col.strip('"') for col in header_parts[1:]]  # Skip ID_REF
    
    print(f"    Found {len(sample_ids)} samples")
    
    # Extract metadata from lines before table_start
    metadata_dict = {}
    
    for i, line in enumerate(lines[:table_start_idx]):
        if line.startswith('Sample_characteristics_ch1'):
            # Parse the metadata line
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            
            # Extract field name and values
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
                print(f"      Found: {field_name} ({len(values)} values)")
    
    return metadata_dict, sample_ids


def create_clinical_dataframe(metadata_dict, sample_ids):
    """
    Create a clinical DataFrame from metadata dictionary.
    """
    print(f"\n[Creating] Clinical DataFrame...")
    
    # Create DataFrame with sample IDs
    clinical_df = pd.DataFrame({'sample_id': sample_ids})
    
    # Map metadata fields to clinical column names
    field_mapping = {
        'pam50 subtype': 'PAM50',
        'age at diagnosis': 'age',
        'overall survival days': 'OS_time',
        'overall survival event': 'OS_event',
        'tumor size': 'tumor_size',
        'lymph node status': 'lymph_node_status',
        'nhg': 'nhg',
        'er status': 'er_status',
        'her2 status': 'her2_status',
        'ki67 status': 'ki67_status',
        'pgr status': 'pgr_status'
    }
    
    for metadata_field, col_name in field_mapping.items():
        if metadata_field in metadata_dict:
            values = metadata_dict[metadata_field]
            # Clean values (remove field name prefix if present)
            clean_values = []
            for val in values:
                if val is None or val == '':
                    clean_values.append('NA')
                elif isinstance(val, str) and f'{metadata_field}: ' in val.lower():
                    clean_values.append(val.split(': ', 1)[1] if ': ' in val else val)
                else:
                    clean_values.append(str(val) if val is not None else 'NA')
            
            clinical_df[col_name] = clean_values
            non_na_count = len([v for v in clean_values if v != 'NA' and v != ''])
            print(f"      Added {col_name}: {non_na_count} non-NA values")
    
    return clinical_df


def combine_clinical_data(file1_path, file2_path, output_file):
    """
    Combine clinical data from two GSE96058 files.
    """
    print("\n" + "=" * 80)
    print("EXTRACTING GSE96058 CLINICAL DATA")
    print("=" * 80)
    
    # Extract metadata from both files
    meta1, samples1 = extract_metadata_from_geo_file(file1_path)
    meta2, samples2 = extract_metadata_from_geo_file(file2_path)
    
    # Create clinical DataFrames
    clinical1 = create_clinical_dataframe(meta1, samples1)
    clinical2 = create_clinical_dataframe(meta2, samples2)
    
    # Combine
    print(f"\n[Combining] Clinical data...")
    print(f"    GPL11154: {len(clinical1)} samples")
    print(f"    GPL18573: {len(clinical2)} samples")
    
    combined_clinical = pd.concat([clinical1, clinical2], ignore_index=True)
    print(f"    Combined: {len(combined_clinical)} samples")
    
    # Transpose: set sample_id as index, then transpose
    print(f"\n[Transposing] Clinical data...")
    combined_clinical_T = combined_clinical.set_index('sample_id').T
    print(f"    Original shape: {combined_clinical.shape}")
    print(f"    Transposed shape: {combined_clinical_T.shape}")
    print(f"    Rows (features): {list(combined_clinical_T.index)}")
    print(f"    Columns (samples): {len(combined_clinical_T.columns)} samples")
    
    # Save transposed version
    print(f"\n[Saving] Transposed clinical data to {output_file}...")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    combined_clinical_T.to_csv(output_file)
    print(f"    [OK] Saved successfully!")
    print(f"    Final shape: {combined_clinical_T.shape}")
    
    return combined_clinical_T


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract clinical data from GSE96058 GEO Series Matrix files"
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
        '--output',
        type=str,
        default='data/GSE96058_clinical_data.csv',
        help='Output path for clinical data CSV'
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
        # Extract and combine clinical data
        combined_clinical = combine_clinical_data(
            args.file1,
            args.file2,
            args.output
        )
        
        print("\n" + "=" * 80)
        print("✅ CLINICAL DATA EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"\nOutput file: {args.output}")
        print(f"\nNote: These GEO Series Matrix files contain only metadata.")
        print("      The actual expression data is not included in these files.")
        print("      You will need to download the expression data separately from GEO.")
        print("      The clinical data extracted here can be used for survival analysis")
        print("      once you have the expression data.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



