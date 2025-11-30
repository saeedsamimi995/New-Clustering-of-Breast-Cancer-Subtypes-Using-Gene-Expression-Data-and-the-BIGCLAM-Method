"""
Create Integrated Biological + Clinical Narrative

Combines results from:
- Biological interpretation (DE genes, pathways, signatures)
- Survival analysis (HR, log-rank p-values, KM curves)
- Cluster-to-PAM50 mapping
- Method comparison (NMI, ARI vs PAM50)

Creates comprehensive summary tables and narrative descriptions that connect
all analyses into a cohesive story.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.results_synthesis import (
    load_biological_interpretation,
    load_survival_results,
    load_method_comparison,
    load_cluster_assignments
)
from src.analysis.cluster_pam50_mapping import load_clusters_and_pam50


def create_comprehensive_summary_table(dataset_name: str, output_dir: str = 'results/synthesis') -> pd.DataFrame:
    """Create comprehensive summary table with all requested fields."""
    print(f"\n{'='*80}")
    print(f"CREATING COMPREHENSIVE SUMMARY: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all data sources
    print("\n[Loading] Data sources...")
    bio_data = load_biological_interpretation(dataset_name)
    surv_data = load_survival_results(dataset_name)
    method_comp = load_method_comparison(dataset_name)
    clusters, sample_names = load_cluster_assignments(dataset_name)
    
    if clusters is None:
        print(f"[Error] Could not load cluster assignments")
        return None
    
    # Convert clusters to integer array
    if isinstance(clusters, np.ndarray):
        if clusters.ndim > 1:
            clusters = np.argmax(clusters, axis=1)
        clusters = clusters.flatten().astype(int)
    
    # Load PAM50 mapping
    try:
        pam50_clusters, pam50_labels, _ = load_clusters_and_pam50(dataset_name)
        # Align with main clusters
        if len(pam50_clusters) == len(clusters):
            clusters = pam50_clusters.astype(int)
    except:
        print("[Warning] Could not load PAM50 mapping, using main clusters")
        pam50_labels = None
    
    unique_clusters = sorted(np.unique(clusters).tolist())
    print(f"  ✓ {len(unique_clusters)} clusters")
    
    # Build comprehensive table
    print("\n[Building] Comprehensive summary table...")
    summary_rows = []
    
    for cluster_id in unique_clusters:
        cluster_id_int = int(cluster_id)
        cluster_mask = clusters == cluster_id_int
        n_samples = cluster_mask.sum()
        
        row = {
            'cluster': cluster_id_int,
            'n_samples': int(n_samples),
            'top_de_genes': '',
            'enriched_pathways': '',
            'enriched_signatures': '',
            'median_os_days': np.nan,
            'hr': np.nan,
            'hr_95ci': '',
            'logrank_p': np.nan,
            'dominant_pam50': '',
            'pam50_distribution': ''
        }
        
        # Biological interpretation
        if 'detailed' in bio_data and bio_data['detailed']:
            detailed = bio_data['detailed']
            
            # DE genes
            if 'de_results' in detailed and cluster_id_int in detailed['de_results']:
                de_sig = detailed['de_results'][cluster_id_int].get('significant', pd.DataFrame())
                if len(de_sig) > 0:
                    top_genes = de_sig.head(5)['gene'].tolist()
                    row['top_de_genes'] = ', '.join(top_genes)
            
            # Signatures
            if 'signature_results' in detailed and cluster_id_int in detailed['signature_results']:
                sigs = detailed['signature_results'][cluster_id_int]
                enriched = [sig for sig, data in sigs.items() if data.get('enriched', False)]
                row['enriched_signatures'] = ', '.join(enriched[:3])  # Top 3
            
            # Pathways
            if 'pathway_results' in detailed and cluster_id_int in detailed['pathway_results']:
                pathways = detailed['pathway_results'][cluster_id_int]
                pathway_names = []
                for db_name, pathway_df in pathways.items():
                    if len(pathway_df) > 0:
                        pathway_names.extend(pathway_df.head(2)['Term'].tolist())
                row['enriched_pathways'] = ', '.join(pathway_names[:3])  # Top 3
        
        # Survival stats
        if surv_data.get('cox') is not None:
            cox_df = surv_data['cox']
            # Try to find cluster in Cox model
            for idx in cox_df.index:
                if str(cluster_id_int) in str(idx) or 'cluster' in str(idx).lower():
                    try:
                        hr_col = [c for c in cox_df.columns if 'exp(coef)' in c.lower() or 'hazard' in c.lower()]
                        lower_col = [c for c in cox_df.columns if 'lower' in c.lower() and '95' in c.lower()]
                        upper_col = [c for c in cox_df.columns if 'upper' in c.lower() and '95' in c.lower()]
                        p_col = [c for c in cox_df.columns if c.lower() == 'p' or 'p-value' in c.lower()]
                        
                        if hr_col:
                            row['hr'] = cox_df.loc[idx, hr_col[0]]
                        if lower_col and upper_col:
                            row['hr_95ci'] = f"[{cox_df.loc[idx, lower_col[0]]:.2f}-{cox_df.loc[idx, upper_col[0]]:.2f}]"
                        if p_col:
                            row['hr_pvalue'] = cox_df.loc[idx, p_col[0]]
                    except:
                        pass
                    break
        
        # PAM50 mapping
        if pam50_labels is not None:
            cluster_pam50 = pam50_labels[cluster_mask]
            pam50_counts = pd.Series(cluster_pam50).value_counts()
            if len(pam50_counts) > 0:
                row['dominant_pam50'] = pam50_counts.index[0]
                if len(pam50_counts) > 1:
                    dist_str = ', '.join([f"{k}({v})" for k, v in pam50_counts.head(3).items()])
                    row['pam50_distribution'] = dist_str
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Add overall metrics
    if method_comp is not None:
        bigclam_row = method_comp[method_comp['Method'] == 'BIGCLAM']
        if len(bigclam_row) > 0:
            summary_df['overall_nmi'] = bigclam_row.iloc[0]['NMI vs PAM50']
            summary_df['overall_ari'] = bigclam_row.iloc[0]['ARI vs PAM50']
    
    # Save
    summary_file = output_dir / f'{dataset_name}_comprehensive_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\n[Saved] Comprehensive summary → {summary_file}")
    
    return summary_df


def generate_integrated_narrative(dataset_name: str, summary_df: pd.DataFrame,
                                  output_dir: str = 'results/synthesis') -> str:
    """Generate integrated narrative connecting all analyses."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    narrative = []
    narrative.append(f"# Integrated Results Narrative: {dataset_name.upper()}\n")
    narrative.append("="*80 + "\n\n")
    
    # Overview
    n_clusters = len(summary_df)
    total_samples = summary_df['n_samples'].sum()
    narrative.append(f"## Overview\n")
    narrative.append(f"BIGCLAM identified **{n_clusters} distinct molecular subtypes** across **{total_samples} breast cancer samples**.\n\n")
    
    # Overall performance
    if 'overall_nmi' in summary_df.columns and pd.notna(summary_df['overall_nmi'].iloc[0]):
        nmi = summary_df['overall_nmi'].iloc[0]
        ari = summary_df['overall_ari'].iloc[0]
        narrative.append(f"**PAM50 Alignment**: NMI={nmi:.3f}, ARI={ari:.3f}\n\n")
    
    # Cluster-by-cluster narrative
    narrative.append("## Cluster Characteristics\n\n")
    
    for _, row in summary_df.iterrows():
        cluster_id = int(row['cluster'])
        n_samples = int(row['n_samples'])
        
        narrative.append(f"### Cluster {cluster_id} (n={n_samples})\n\n")
        
        # Biological characteristics
        if pd.notna(row['top_de_genes']) and row['top_de_genes']:
            narrative.append(f"**Highly Expressed Genes**: {row['top_de_genes']}\n\n")
        
        if pd.notna(row['enriched_signatures']) and row['enriched_signatures']:
            narrative.append(f"**Enriched Signatures**: {row['enriched_signatures']}\n\n")
        
        if pd.notna(row['enriched_pathways']) and row['enriched_pathways']:
            narrative.append(f"**Enriched Pathways**: {row['enriched_pathways']}\n\n")
        
        # PAM50 mapping
        if pd.notna(row['dominant_pam50']) and row['dominant_pam50']:
            narrative.append(f"**PAM50 Alignment**: Dominant subtype = {row['dominant_pam50']}")
            if pd.notna(row['pam50_distribution']) and row['pam50_distribution']:
                narrative.append(f" (also contains: {row['pam50_distribution']})")
            narrative.append("\n\n")
        
        # Survival
        if pd.notna(row['hr']) and pd.notna(row['hr_pvalue']):
            hr = row['hr']
            pval = row['hr_pvalue']
            ci = row['hr_95ci'] if pd.notna(row['hr_95ci']) else ''
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            direction = "worse" if hr > 1 else "better"
            narrative.append(f"**Survival**: HR={hr:.2f} {ci} (p={pval:.3f} {sig}) - {direction} prognosis\n\n")
        
        narrative.append("---\n\n")
    
    narrative_text = ''.join(narrative)
    
    # Save
    narrative_file = output_dir / f'{dataset_name}_integrated_narrative.md'
    with open(narrative_file, 'w') as f:
        f.write(narrative_text)
    
    print(f"[Saved] Integrated narrative → {narrative_file}")
    
    return narrative_text


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create integrated narrative from all analyses')
    parser.add_argument('--datasets', nargs='+', choices=['tcga', 'gse96058', 'both'],
                       default=['both'], help='Datasets to process')
    parser.add_argument('--output-dir', type=str, default='results/synthesis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    datasets = []
    if 'both' in args.datasets:
        datasets = ['tcga', 'gse96058']
    else:
        datasets = args.datasets
    
    for dataset_name in datasets:
        try:
            print(f"\n{'='*80}")
            print(f"PROCESSING: {dataset_name.upper()}")
            print(f"{'='*80}")
            
            # Create comprehensive summary
            summary_df = create_comprehensive_summary_table(dataset_name, args.output_dir)
            
            if summary_df is not None:
                # Generate integrated narrative
                generate_integrated_narrative(dataset_name, summary_df, args.output_dir)
        except Exception as e:
            print(f"\n[Error] Failed to process {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("INTEGRATED NARRATIVE COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

