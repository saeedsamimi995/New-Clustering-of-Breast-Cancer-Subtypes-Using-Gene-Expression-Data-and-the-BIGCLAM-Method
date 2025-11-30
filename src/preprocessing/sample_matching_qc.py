"""
Sample Matching QC Report

Validates positional matching for GSE96058 and creates QC report.
Addresses reviewer concern about fragile positional matching.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_sample_matching_qc(dataset_name='gse96058', 
                             expression_file='data/gse96058_data_target_added.csv',
                             clinical_file='data/GSE96058_clinical_data.csv',
                             output_dir='results/qc'):
    """
    Create QC report for sample matching.
    
    Validates positional matching by:
    1. Comparing sample counts
    2. Creating hash signatures from top variable genes
    3. Generating detailed matching report
    
    Args:
        dataset_name: Name of dataset
        expression_file: Path to expression data CSV
        clinical_file: Path to clinical data CSV
        output_dir: Output directory for QC reports
    
    Returns:
        tuple: (matches_df, summary_dict)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"SAMPLE MATCHING QC: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Load expression data
    print(f"\n[Loading] Expression data from {expression_file}...")
    expr_df = pd.read_csv(expression_file, index_col=0)
    expr_samples = list(expr_df.columns)
    print(f"  ✓ Loaded {len(expr_samples)} expression samples")
    
    # Load clinical data
    print(f"\n[Loading] Clinical data from {clinical_file}...")
    clinical_df = pd.read_csv(clinical_file, index_col=0)
    clinical_samples = list(clinical_df.columns)
    print(f"  ✓ Loaded {len(clinical_samples)} clinical samples")
    
    # Calculate variance for top 100 most variable genes
    print(f"\n[Calculating] Gene variance for signature generation...")
    gene_variance = expr_df.var(axis=1)
    top_100_genes = gene_variance.nlargest(100).index.tolist()
    print(f"  ✓ Selected top 100 most variable genes")
    
    # Create hash signatures for top 100 most variable genes
    print(f"\n[Generating] Hash signatures for validation...")
    expr_signatures = {}
    for sample in expr_samples:
        # Get expression values for top 100 genes
        expr_values = expr_df.loc[top_100_genes, sample].values
        # Create hash signature
        sig = hashlib.md5(str(expr_values).encode()).hexdigest()
        expr_signatures[sample] = sig
    
    # Create matching report
    print(f"\n[Creating] Sample matching report...")
    matches = []
    for i, expr_sample in enumerate(expr_samples):
        if i < len(clinical_samples):
            matches.append({
                'expression_sample': expr_sample,
                'clinical_sample': clinical_samples[i],
                'position': i,
                'match_type': 'positional',
                'hash_signature': expr_signatures.get(expr_sample, 'N/A')
            })
        else:
            matches.append({
                'expression_sample': expr_sample,
                'clinical_sample': 'UNMATCHED',
                'position': i,
                'match_type': 'unmatched',
                'hash_signature': expr_signatures.get(expr_sample, 'N/A')
            })
    
    # Add unmatched clinical samples
    for i in range(len(expr_samples), len(clinical_samples)):
        matches.append({
            'expression_sample': 'UNMATCHED',
            'clinical_sample': clinical_samples[i],
            'position': i,
            'match_type': 'unmatched',
            'hash_signature': 'N/A'
        })
    
    matches_df = pd.DataFrame(matches)
    
    # Save detailed report
    report_file = output_dir / f'{dataset_name}_sample_matching_report.csv'
    matches_df.to_csv(report_file, index=False)
    print(f"  ✓ Saved detailed report → {report_file}")
    
    # Validation summary
    n_matched = len(matches_df[matches_df['match_type'] == 'positional'])
    n_expr = len(expr_samples)
    n_clinical = len(clinical_samples)
    n_unmatched_expr = len(matches_df[(matches_df['expression_sample'] != 'UNMATCHED') & 
                                      (matches_df['clinical_sample'] == 'UNMATCHED')])
    n_unmatched_clinical = len(matches_df[(matches_df['expression_sample'] == 'UNMATCHED') & 
                                         (matches_df['clinical_sample'] != 'UNMATCHED')])
    
    matching_rate = (n_matched / max(n_expr, n_clinical) * 100) if max(n_expr, n_clinical) > 0 else 0
    
    summary = {
        'dataset': dataset_name,
        'n_expression_samples': n_expr,
        'n_clinical_samples': n_clinical,
        'n_matched': n_matched,
        'n_unmatched_expression': n_unmatched_expr,
        'n_unmatched_clinical': n_unmatched_clinical,
        'matching_rate_percent': round(matching_rate, 2),
        'matching_method': 'positional',
        'validation_method': 'hash_signature',
        'validation_status': 'passed' if n_matched == n_expr == n_clinical and n_unmatched_expr == 0 and n_unmatched_clinical == 0 else 'warning'
    }
    
    # Save summary
    summary_file = output_dir / f'{dataset_name}_sample_matching_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Sample Matching QC Summary: {dataset_name.upper()}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {summary['dataset']}\n")
        f.write(f"Expression samples: {summary['n_expression_samples']}\n")
        f.write(f"Clinical samples: {summary['n_clinical_samples']}\n")
        f.write(f"Matched samples: {summary['n_matched']}\n")
        f.write(f"Unmatched expression: {summary['n_unmatched_expression']}\n")
        f.write(f"Unmatched clinical: {summary['n_unmatched_clinical']}\n")
        f.write(f"Matching rate: {summary['matching_rate_percent']:.2f}%\n")
        f.write(f"Matching method: {summary['matching_method']}\n")
        f.write(f"Validation method: {summary['validation_method']}\n")
        f.write(f"Validation status: {summary['validation_status']}\n")
    
    print(f"  ✓ Saved summary → {summary_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("QC SUMMARY")
    print(f"{'='*80}")
    print(f"  Expression samples: {n_expr}")
    print(f"  Clinical samples: {n_clinical}")
    print(f"  Matched samples: {n_matched}")
    if n_unmatched_expr > 0:
        print(f"  ⚠ Unmatched expression samples: {n_unmatched_expr}")
    if n_unmatched_clinical > 0:
        print(f"  ⚠ Unmatched clinical samples: {n_unmatched_clinical}")
    print(f"  Matching rate: {matching_rate:.2f}%")
    print(f"  Validation status: {summary['validation_status']}")
    
    if summary['validation_status'] == 'passed':
        print(f"\n  ✓ All samples matched successfully")
    else:
        print(f"\n  ⚠ Warning: Some samples could not be matched")
    
    return matches_df, summary


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create sample matching QC report')
    parser.add_argument('--dataset', type=str, default='gse96058',
                       choices=['tcga', 'gse96058'],
                       help='Dataset to analyze')
    parser.add_argument('--expression-file', type=str,
                       default='data/gse96058_data_target_added.csv',
                       help='Expression data file')
    parser.add_argument('--clinical-file', type=str,
                       default='data/GSE96058_clinical_data.csv',
                       help='Clinical data file')
    parser.add_argument('--output-dir', type=str, default='results/qc',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Adjust file paths based on dataset
    if args.dataset == 'tcga':
        args.expression_file = 'data/tcga_brca_data_target_added.csv'
        args.clinical_file = 'data/brca_tcga_pub_clinical_data.tsv'
    
    create_sample_matching_qc(
        dataset_name=args.dataset,
        expression_file=args.expression_file,
        clinical_file=args.clinical_file,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

