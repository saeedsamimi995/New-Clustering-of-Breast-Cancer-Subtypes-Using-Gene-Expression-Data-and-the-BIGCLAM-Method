"""
Results Synthesis Script

Generates comprehensive summary tables and narrative descriptions by combining:
- Biological interpretation results
- Survival analysis results
- Method comparison results
- Cluster characteristics

Outputs:
- Summary table (CSV)
- Narrative description (TXT)
- Manuscript-ready results section (MD)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_biological_interpretation(dataset_name: str, results_dir: str = 'results/biological_interpretation') -> Dict:
    """Load biological interpretation results."""
    bio_dir = Path(results_dir) / dataset_name
    
    results = {
        'summary_file': bio_dir / 'biological_interpretation_summary.txt',
        'detailed_file': bio_dir / 'biological_interpretation_results.pkl',
        'signature_file': bio_dir / 'signature_scores.csv'
    }
    
    bio_data = {}
    
    # Load summary text
    if results['summary_file'].exists():
        with open(results['summary_file'], 'r') as f:
            bio_data['summary_text'] = f.read()
    
    # Load detailed results
    if results['detailed_file'].exists():
        with open(results['detailed_file'], 'rb') as f:
            bio_data['detailed'] = pickle.load(f)
    
    # Load signature scores
    if results['signature_file'].exists():
        bio_data['signatures'] = pd.read_csv(results['signature_file'])
    
    return bio_data


def load_survival_results(dataset_name: str, results_dir: str = 'results/survival') -> Dict:
    """Load survival analysis results."""
    surv_dir = Path(results_dir) / dataset_name
    
    results = {}
    
    # Load Cox model summary
    cox_file = surv_dir / 'cox_summary.csv'
    if cox_file.exists():
        try:
            cox_df = pd.read_csv(cox_file, index_col=0)
            results['cox'] = cox_df
        except:
            results['cox'] = None
    else:
        results['cox'] = None
    
    # Load log-rank results
    logrank_file = surv_dir / 'logrank_results.csv'
    if logrank_file.exists():
        results['logrank'] = logrank_file
    else:
        results['logrank'] = None
    
    return results


def load_method_comparison(dataset_name: str, results_dir: str = 'results/method_comparison') -> Optional[pd.DataFrame]:
    """Load method comparison results."""
    comp_file = Path(results_dir) / f'{dataset_name}_method_comparison.csv'
    if comp_file.exists():
        return pd.read_csv(comp_file)
    return None


def load_cluster_assignments(dataset_name: str, clustering_dir: str = 'data/clusterings',
                            processed_dir: str = 'data/processed') -> Tuple[np.ndarray, List]:
    """Load cluster assignments and sample names."""
    import pickle
    
    # Map dataset names
    file_prefix_map = {
        'tcga': 'tcga_brca_data',
        'gse96058': 'gse96058_data'
    }
    prefix = file_prefix_map.get(dataset_name, dataset_name)
    
    # Load clusters
    cluster_file = Path(clustering_dir) / f'{prefix}_communities.npy'
    if not cluster_file.exists():
        return None, None
    
    clusters = np.load(cluster_file)
    
    # Load sample names
    target_file = Path(processed_dir) / f'{prefix}_targets.pkl'
    if not target_file.exists():
        return clusters, None
    
    with open(target_file, 'rb') as f:
        targets_data = pickle.load(f)
        sample_names = targets_data.get('sample_names', [])
    
    return clusters, sample_names


def extract_cluster_characteristics(bio_data: Dict, clusters: np.ndarray) -> pd.DataFrame:
    """Extract cluster characteristics from biological interpretation."""
    # Handle numpy array - ensure integer cluster IDs
    if isinstance(clusters, np.ndarray):
        if clusters.ndim > 1:
            # Membership matrix - convert to hard assignments
            clusters = np.argmax(clusters, axis=1)
        unique_clusters = sorted(np.unique(clusters).astype(int).tolist())
    else:
        unique_clusters = sorted(set(int(c) for c in clusters))
    
    cluster_info = []
    
    for cluster_id in unique_clusters:
        # Ensure cluster_id is integer for comparison
        cluster_id_int = int(cluster_id)
        info = {
            'cluster': cluster_id_int,
            'n_samples': int((clusters == cluster_id_int).sum()),
            'top_genes': [],
            'enriched_signatures': [],
            'enriched_pathways': []
        }
        
        # Extract from detailed results
        if 'detailed' in bio_data and bio_data['detailed']:
            detailed = bio_data['detailed']
            
            # DE genes
            if 'de_results' in detailed and cluster_id_int in detailed['de_results']:
                de_sig = detailed['de_results'][cluster_id_int].get('significant', pd.DataFrame())
                if len(de_sig) > 0:
                    info['top_genes'] = de_sig.head(5)['gene'].tolist()
                    info['n_de_genes'] = len(de_sig)
            
            # Signatures
            if 'signature_results' in detailed and cluster_id_int in detailed['signature_results']:
                sigs = detailed['signature_results'][cluster_id_int]
                enriched = [sig for sig, data in sigs.items() if data.get('enriched', False)]
                info['enriched_signatures'] = enriched
            
            # Pathways
            if 'pathway_results' in detailed and cluster_id_int in detailed['pathway_results']:
                pathways = detailed['pathway_results'][cluster_id_int]
                pathway_names = []
                for db_name, pathway_df in pathways.items():
                    if len(pathway_df) > 0:
                        pathway_names.extend(pathway_df.head(3)['Term'].tolist())
                info['enriched_pathways'] = pathway_names[:5]  # Top 5
        
        cluster_info.append(info)
    
    return pd.DataFrame(cluster_info)


def extract_survival_stats(surv_data: Dict, clusters: np.ndarray) -> pd.DataFrame:
    """Extract survival statistics per cluster."""
    # Handle numpy array - ensure integer cluster IDs
    if isinstance(clusters, np.ndarray):
        if clusters.ndim > 1:
            clusters = np.argmax(clusters, axis=1)
        unique_clusters = sorted(np.unique(clusters).astype(int).tolist())
    else:
        unique_clusters = sorted(set(int(c) for c in clusters))
    
    surv_stats = []
    
    # Extract from Cox model summary
    cox_df = None
    if 'cox' in surv_data and surv_data['cox'] is not None:
        cox_df = surv_data['cox']
    
    # Extract from log-rank results
    logrank_df = None
    if 'logrank' in surv_data:
        logrank_path = surv_data['logrank']
        if isinstance(logrank_path, str) or isinstance(logrank_path, Path):
            if Path(logrank_path).exists():
                logrank_df = pd.read_csv(logrank_path)
    
    # Ensure clusters is numpy array for comparison
    if not isinstance(clusters, np.ndarray):
        clusters = np.array(clusters)
    
    # Extract per-cluster stats
    for cluster_id in unique_clusters:
        cluster_id_int = int(cluster_id)
        cluster_mask = clusters == cluster_id_int
        
        # Initialize with defaults
        hr = np.nan
        hr_ci_lower = np.nan
        hr_ci_upper = np.nan
        hr_pvalue = np.nan
        
        # Try to extract HR from Cox model
        if cox_df is not None and len(cox_df) > 0:
            # Look for cluster variable in Cox summary
            # Column names may vary: 'exp(coef)', 'exp(coef) lower 95%', etc.
            cluster_var = None
            for idx in cox_df.index:
                if 'cluster' in str(idx).lower() or str(cluster_id_int) in str(idx):
                    cluster_var = idx
                    break
            
            if cluster_var is not None:
                try:
                    # Try different possible column names
                    hr_col = None
                    lower_col = None
                    upper_col = None
                    p_col = None
                    
                    for col in cox_df.columns:
                        col_lower = col.lower()
                        if 'exp(coef)' in col_lower or 'hazard' in col_lower:
                            hr_col = col
                        if 'lower' in col_lower and '95' in col_lower:
                            lower_col = col
                        if 'upper' in col_lower and '95' in col_lower:
                            upper_col = col
                        if col_lower == 'p' or 'p-value' in col_lower:
                            p_col = col
                    
                    if hr_col:
                        hr = cox_df.loc[cluster_var, hr_col]
                    if lower_col:
                        hr_ci_lower = cox_df.loc[cluster_var, lower_col]
                    if upper_col:
                        hr_ci_upper = cox_df.loc[cluster_var, upper_col]
                    if p_col:
                        hr_pvalue = cox_df.loc[cluster_var, p_col]
                except:
                    pass
        
        surv_stats.append({
            'cluster': cluster_id_int,
            'n_samples': int(cluster_mask.sum()),
            'hr': hr if pd.notna(hr) else np.nan,
            'hr_ci_lower': hr_ci_lower if pd.notna(hr_ci_lower) else np.nan,
            'hr_ci_upper': hr_ci_upper if pd.notna(hr_ci_upper) else np.nan,
            'hr_pvalue': hr_pvalue if pd.notna(hr_pvalue) else np.nan
        })
    
    return pd.DataFrame(surv_stats)


def create_summary_table(dataset_name: str, output_dir: str = 'results/synthesis') -> pd.DataFrame:
    """Create comprehensive summary table."""
    print(f"\n{'='*80}")
    print(f"SYNTHESIZING RESULTS: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Load all data
    print("\n[Loading] Data sources...")
    bio_data = load_biological_interpretation(dataset_name)
    surv_data = load_survival_results(dataset_name)
    method_comp = load_method_comparison(dataset_name)
    clusters, sample_names = load_cluster_assignments(dataset_name)
    
    if clusters is None:
        print(f"[Error] Could not load cluster assignments for {dataset_name}")
        return None
    
    # Convert to list if numpy array for set() operation
    if isinstance(clusters, np.ndarray):
        unique_clusters = len(np.unique(clusters))
    else:
        unique_clusters = len(set(clusters))
    
    print(f"  ✓ Loaded {unique_clusters} clusters")
    
    # Extract characteristics
    print("\n[Extracting] Cluster characteristics...")
    cluster_chars = extract_cluster_characteristics(bio_data, clusters)
    surv_stats = extract_survival_stats(surv_data, clusters)
    
    # Merge
    summary = cluster_chars.merge(surv_stats, on='cluster', how='outer', suffixes=('', '_surv'))
    
    # Add method comparison info
    if method_comp is not None:
        bigclam_row = method_comp[method_comp['Method'] == 'BIGCLAM']
        if len(bigclam_row) > 0:
            summary['nmi_vs_pam50'] = bigclam_row.iloc[0]['NMI vs PAM50']
            summary['ari_vs_pam50'] = bigclam_row.iloc[0]['ARI vs PAM50']
            summary['n_clusters_found'] = bigclam_row.iloc[0]['N_Clusters']
    
    # Format for readability
    summary['top_genes_str'] = summary['top_genes'].apply(lambda x: ', '.join(x[:5]) if isinstance(x, list) else '')
    summary['signatures_str'] = summary['enriched_signatures'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
    summary['pathways_str'] = summary['enriched_pathways'].apply(lambda x: ', '.join(x[:3]) if isinstance(x, list) else '')
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_dir / f'{dataset_name}_summary_table.csv'
    summary.to_csv(summary_file, index=False)
    print(f"\n[Saved] Summary table → {summary_file}")
    
    return summary


def generate_narrative(dataset_name: str, summary_df: pd.DataFrame, 
                      output_dir: str = 'results/synthesis') -> str:
    """Generate narrative description of results."""
    narrative = []
    
    narrative.append(f"# Results Summary: {dataset_name.upper()}\n")
    narrative.append(f"BIGCLAM identified {len(summary_df)} distinct molecular subtypes.\n")
    
    for _, row in summary_df.iterrows():
        cluster_id = int(row['cluster'])
        n_samples = int(row['n_samples'])
        
        narrative.append(f"\n## Cluster {cluster_id} (n={n_samples})")
        
        # Biological characteristics
        if pd.notna(row['top_genes_str']) and row['top_genes_str']:
            narrative.append(f"\n**Highly Expressed Genes**: {row['top_genes_str']}")
        
        if pd.notna(row['signatures_str']) and row['signatures_str']:
            narrative.append(f"\n**Enriched Signatures**: {row['signatures_str']}")
        
        if pd.notna(row['pathways_str']) and row['pathways_str']:
            narrative.append(f"\n**Enriched Pathways**: {row['pathways_str']}")
        
        # Survival
        if pd.notna(row['hr']):
            hr = row['hr']
            ci_lower = row['hr_ci_lower']
            ci_upper = row['hr_ci_upper']
            pval = row['hr_pvalue']
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            narrative.append(f"\n**Survival**: HR={hr:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f}, p={pval:.3f} {sig})")
    
    # Overall performance
    if 'nmi_vs_pam50' in summary_df.columns and pd.notna(summary_df['nmi_vs_pam50'].iloc[0]):
        nmi = summary_df['nmi_vs_pam50'].iloc[0]
        ari = summary_df['ari_vs_pam50'].iloc[0]
        narrative.append(f"\n\n## Overall Performance")
        narrative.append(f"**PAM50 Alignment**: NMI={nmi:.3f}, ARI={ari:.3f}")
    
    narrative_text = '\n'.join(narrative)
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    narrative_file = output_dir / f'{dataset_name}_narrative.txt'
    with open(narrative_file, 'w') as f:
        f.write(narrative_text)
    
    print(f"[Saved] Narrative → {narrative_file}")
    
    return narrative_text


def create_manuscript_results_section(dataset_name: str, summary_df: pd.DataFrame,
                                     output_dir: str = 'results/synthesis') -> str:
    """Create manuscript-ready results section."""
    manuscript = []
    
    manuscript.append(f"# Results: {dataset_name.upper()}\n")
    
    # Introduction
    n_clusters = len(summary_df)
    total_samples = summary_df['n_samples'].sum()
    manuscript.append(f"BIGCLAM identified {n_clusters} distinct molecular subtypes across {total_samples} breast cancer samples.\n")
    
    # Cluster-by-cluster description
    manuscript.append("## Cluster Characteristics\n")
    
    for _, row in summary_df.iterrows():
        cluster_id = int(row['cluster'])
        n_samples = int(row['n_samples'])
        
        manuscript.append(f"**Cluster {cluster_id}** (n={n_samples})")
        
        # Biological interpretation
        bio_desc = []
        if pd.notna(row['signatures_str']) and row['signatures_str']:
            bio_desc.append(f"exhibited {row['signatures_str']} signatures")
        
        if pd.notna(row['pathways_str']) and row['pathways_str']:
            bio_desc.append(f"with enrichment in {row['pathways_str']}")
        
        if bio_desc:
            manuscript.append(" ".join(bio_desc) + ".")
        
        # Survival
        if pd.notna(row['hr']):
            hr = row['hr']
            pval = row['hr_pvalue']
            if pval < 0.05:
                direction = "worse" if hr > 1 else "better"
                manuscript.append(f"Survival analysis revealed {direction} prognosis (HR={hr:.2f}, p={pval:.3f}).")
        
        manuscript.append("")
    
    # Overall performance
    if 'nmi_vs_pam50' in summary_df.columns and pd.notna(summary_df['nmi_vs_pam50'].iloc[0]):
        nmi = summary_df['nmi_vs_pam50'].iloc[0]
        ari = summary_df['ari_vs_pam50'].iloc[0]
        manuscript.append("## Comparison to PAM50\n")
        manuscript.append(f"BIGCLAM clusters showed moderate alignment with PAM50 molecular subtypes (NMI={nmi:.3f}, ARI={ari:.3f}).")
    
    manuscript_text = '\n'.join(manuscript)
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manuscript_file = output_dir / f'{dataset_name}_manuscript_results.md'
    with open(manuscript_file, 'w') as f:
        f.write(manuscript_text)
    
    print(f"[Saved] Manuscript section → {manuscript_file}")
    
    return manuscript_text


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Synthesize results from all analyses')
    parser.add_argument('--datasets', nargs='+', choices=['tcga', 'gse96058', 'both'],
                       default=['both'], help='Datasets to synthesize')
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
            # Create summary table
            summary_df = create_summary_table(dataset_name, args.output_dir)
            
            if summary_df is not None:
                # Generate narrative
                generate_narrative(dataset_name, summary_df, args.output_dir)
                
                # Create manuscript section
                create_manuscript_results_section(dataset_name, summary_df, args.output_dir)
        except Exception as e:
            print(f"\n[Error] Failed to synthesize {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("SYNTHESIS COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

