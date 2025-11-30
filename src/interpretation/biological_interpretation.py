"""
Biological Interpretation Module

Performs comprehensive biological interpretation of BIGCLAM clusters:
1. Differential gene expression analysis
2. Pathway enrichment (GO, KEGG, Reactome)
3. Cell-type signature analysis (immune, proliferation, EMT, angiogenesis, luminal/hormonal)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy import stats
from scipy.stats import ttest_ind
import warnings
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.preprocessing.data_preprocessing import load_data_with_target

# Try to import enrichment libraries (optional dependencies)
try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False
    print("[Warning] gseapy not available. Pathway enrichment will use alternative methods.")

try:
    from statsmodels.stats.multitest import multipletests  # type: ignore
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    multipletests = None  # Placeholder to avoid NameError
    print("[Warning] statsmodels not available. Will use basic FDR correction.")


# Breast cancer signature gene sets
BREAST_CANCER_SIGNATURES = {
    'luminal_hormonal': [
        'ESR1', 'PGR', 'FOXA1', 'GATA3', 'XBP1', 'TFF1', 'TFF3', 
        'AGR2', 'AGR3', 'CA12', 'NAT1', 'BCL2', 'CCND1', 'FGFR1'
    ],
    'basal_like': [
        'KRT5', 'KRT14', 'KRT17', 'TP63', 'CDH3', 'TRIM29', 
        'FOXC1', 'ANLN', 'BUB1', 'CCNB1', 'CDC20', 'CENPF'
    ],
    'her2_enriched': [
        'ERBB2', 'GRB7', 'STARD3', 'TCAP', 'PNMT', 'PGAP3', 
        'ERBB2', 'GRB7', 'MIEN1', 'IKZF3', 'RPS6KB1'
    ],
    'immune_infiltration': [
        'CD8A', 'CD8B', 'CD3D', 'CD3E', 'CD3G', 'CD2', 'CD5', 
        'CD7', 'CD4', 'CD19', 'CD20', 'CD79A', 'CD79B',
        'CXCL9', 'CXCL10', 'CXCL11', 'CXCL13', 'CCL5', 'CCL19',
        'IFNG', 'GZMA', 'GZMB', 'PRF1', 'TBX21', 'STAT1'
    ],
    'proliferation': [
        'MKI67', 'TOP2A', 'CCNB1', 'CCNB2', 'CCNA2', 'CDK1', 
        'BUB1', 'BUB1B', 'CENPF', 'CENPE', 'AURKA', 'AURKB',
        'PLK1', 'CDC20', 'CDCA8', 'NUSAP1', 'KIF2C', 'KIF11'
    ],
    'emt': [
        'VIM', 'FN1', 'CDH2', 'SNAI1', 'SNAI2', 'TWIST1', 'TWIST2',
        'ZEB1', 'ZEB2', 'MMP2', 'MMP9', 'TGFB1', 'TGFB2', 'TGFB3',
        'COL1A1', 'COL1A2', 'COL3A1', 'ACTA2', 'TAGLN'
    ],
    'angiogenesis': [
        'VEGFA', 'VEGFB', 'VEGFC', 'VEGFD', 'PGF', 'FLT1', 'KDR',
        'FLT4', 'ANGPT1', 'ANGPT2', 'TEK', 'PDGFA', 'PDGFB', 'PDGFC',
        'FGF2', 'FGF1', 'THBS1', 'THBS2', 'SPARC', 'COL4A1', 'COL4A2'
    ]
}


def load_original_expression_data(dataset_name, data_dir='data'):
    """
    Load original expression data from CSV (before feature selection).
    This ensures we have all genes including breast cancer markers.
    
    Args:
        dataset_name: Name of dataset ('tcga' or 'gse96058')
        data_dir: Directory with original data files
    
    Returns:
        tuple: (expression_data, gene_names, sample_names)
    """
    from src.preprocessing.data_preprocessing import load_data_with_target
    
    # Map dataset name to file prefixes
    file_prefix_map = {
        'tcga': 'tcga_brca_data',
        'gse96058': 'gse96058_data'
    }
    prefix = file_prefix_map.get(dataset_name, dataset_name)
    
    # Load from original CSV file
    csv_file = Path(data_dir) / f"{prefix}_target_added.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(f"Original data file not found: {csv_file}")
    
    print(f"  Loading original data from {csv_file}...")
    expression_data, target_labels, gene_names, sample_names = load_data_with_target(str(csv_file))
    
    # Apply log2 transformation if not already done (check if values are in log space)
    # Original data is typically in raw counts or RSEM values, need log2
    if expression_data.max() > 100:  # Likely not log-transformed
        print(f"  Applying log2 transformation...")
        expression_data = np.log2(expression_data + 1)
    
    print(f"  Original expression data: {expression_data.shape}")
    print(f"  Gene names: {len(gene_names)}")
    print(f"  Sample names: {len(sample_names)}")
    
    return expression_data, gene_names, sample_names


def load_expression_and_clusters(dataset_name, processed_dir='data/processed', 
                                 clustering_dir='data/clusterings',
                                 use_original_data=True):
    """
    Load expression data and cluster assignments.
    
    Args:
        dataset_name: Name of dataset ('tcga' or 'gse96058')
        processed_dir: Directory with processed data
        clustering_dir: Directory with cluster assignments
        use_original_data: If True, load from original CSV (all genes). 
                          If False, use processed data (feature-selected genes).
    
    Returns:
        tuple: (expression_data, gene_names, sample_names, cluster_assignments)
    """
    print(f"\n[Loading] Expression data and clusters for {dataset_name}...")
    
    # Map dataset name to file prefixes
    file_prefix_map = {
        'tcga': 'tcga_brca_data',
        'gse96058': 'gse96058_data'
    }
    prefix = file_prefix_map.get(dataset_name, dataset_name)
    
    # Load expression data
    if use_original_data:
        # Load original data (all genes) for biological interpretation
        expression_data, gene_names, sample_names = load_original_expression_data(
            dataset_name, data_dir='data'
        )
    else:
        # Load processed data (feature-selected genes)
        processed_file = Path(processed_dir) / f"{prefix}_processed.npy"
        targets_file = Path(processed_dir) / f"{prefix}_targets.pkl"
        
        if not processed_file.exists():
            raise FileNotFoundError(f"Processed data not found: {processed_file}")
        if not targets_file.exists():
            raise FileNotFoundError(f"Targets file not found: {targets_file}")
        
        expression_data = np.load(processed_file)
        
        with open(targets_file, 'rb') as f:
            targets_data = pickle.load(f)
        
        gene_names = targets_data.get('gene_names', None)
        sample_names = targets_data.get('sample_names', None)
    
    # Load cluster assignments
    cluster_file = Path(clustering_dir) / f"{prefix}_communities.npy"
    if not cluster_file.exists():
        raise FileNotFoundError(f"Cluster file not found: {cluster_file}")
    
    clusters = np.load(cluster_file)
    if clusters.ndim == 2:
        clusters = np.argmax(clusters, axis=1)
    clusters = clusters.flatten()
    
    # Ensure cluster assignments match sample count
    if len(clusters) != len(sample_names):
        print(f"  [Warning] Cluster count ({len(clusters)}) != sample count ({len(sample_names)})")
        print(f"  Using first {min(len(clusters), len(sample_names))} samples/clusters")
        min_len = min(len(clusters), len(sample_names))
        clusters = clusters[:min_len]
        sample_names = sample_names[:min_len]
        expression_data = expression_data[:min_len, :]
    
    print(f"  Expression data: {expression_data.shape}")
    print(f"  Gene names: {len(gene_names) if gene_names is not None else 'N/A'}")
    print(f"  Sample names: {len(sample_names) if sample_names is not None else 'N/A'}")
    print(f"  Clusters: {len(set(clusters))} unique clusters")
    
    return expression_data, gene_names, sample_names, clusters


def differential_expression_analysis(expression_data, gene_names, clusters, 
                                    cluster_id, method='ttest', 
                                    log2_fold_change_threshold=1.0,
                                    pvalue_threshold=0.05):
    """
    Perform differential expression analysis for a specific cluster vs all others.
    
    Args:
        expression_data: numpy array (n_samples, n_genes)
        gene_names: array of gene identifiers
        clusters: array of cluster assignments
        cluster_id: ID of cluster to analyze
        method: 'ttest' or 'wilcoxon'
        log2_fold_change_threshold: Minimum log2 fold change
        pvalue_threshold: P-value threshold
    
    Returns:
        DataFrame with DE results
    """
    print(f"\n[Differential Expression] Cluster {cluster_id} vs others...")
    
    # Split into cluster and others
    cluster_mask = clusters == cluster_id
    other_mask = ~cluster_mask
    
    cluster_expr = expression_data[cluster_mask, :]
    other_expr = expression_data[other_mask, :]
    
    n_cluster = cluster_expr.shape[0]
    n_other = other_expr.shape[0]
    
    print(f"  Cluster {cluster_id}: {n_cluster} samples")
    print(f"  Others: {n_other} samples")
    
    if n_cluster < 3 or n_other < 3:
        print(f"  [Warning] Insufficient samples for DE analysis")
        return pd.DataFrame()
    
    # Calculate mean expression
    cluster_mean = np.mean(cluster_expr, axis=0)
    other_mean = np.mean(other_expr, axis=0)
    
    # Log2 fold change
    log2fc = cluster_mean - other_mean
    
    # Statistical test
    if method == 'ttest':
        pvalues = []
        for i in range(expression_data.shape[1]):
            try:
                t_stat, p_val = ttest_ind(cluster_expr[:, i], other_expr[:, i])
                pvalues.append(p_val)
            except:
                pvalues.append(1.0)
        pvalues = np.array(pvalues)
    elif method == 'wilcoxon':
        from scipy.stats import ranksums
        pvalues = []
        for i in range(expression_data.shape[1]):
            try:
                stat, p_val = ranksums(cluster_expr[:, i], other_expr[:, i])
                pvalues.append(p_val)
            except:
                pvalues.append(1.0)
        pvalues = np.array(pvalues)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # FDR correction
    if STATSMODELS_AVAILABLE and multipletests is not None:
        _, pvalues_adj, _, _ = multipletests(pvalues, method='fdr_bh')
    else:
        # Simple Bonferroni correction (fallback when statsmodels not available)
        pvalues_adj = pvalues * len(pvalues)
        pvalues_adj = np.clip(pvalues_adj, 0, 1)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'gene': gene_names,
        'cluster_mean': cluster_mean,
        'other_mean': other_mean,
        'log2fc': log2fc,
        'pvalue': pvalues,
        'padj': pvalues_adj
    })
    
    # Filter significant genes
    significant = (np.abs(results['log2fc']) >= log2_fold_change_threshold) & \
                  (results['padj'] < pvalue_threshold)
    
    results_sig = results[significant].copy()
    results_sig = results_sig.sort_values('log2fc', ascending=False)
    
    print(f"  Significant genes (|log2FC| >= {log2_fold_change_threshold}, padj < {pvalue_threshold}): {len(results_sig)}")
    
    if len(results_sig) > 0:
        print(f"\n  Top 10 upregulated:")
        top_up = results_sig.head(10)
        for _, row in top_up.iterrows():
            print(f"    {row['gene']}: log2FC={row['log2fc']:.2f}, padj={row['padj']:.2e}")
        
        print(f"\n  Top 10 downregulated:")
        top_down = results_sig.tail(10)
        for _, row in top_down.iterrows():
            print(f"    {row['gene']}: log2FC={row['log2fc']:.2f}, padj={row['padj']:.2e}")
    
    return results, results_sig


def pathway_enrichment_analysis(de_genes, gene_names, dataset_name='human',
                                databases=['GO_Biological_Process_2021', 
                                          'KEGG_2021_Human',
                                          'Reactome_2022'],
                                output_dir='results/biological_interpretation'):
    """
    Perform pathway enrichment analysis using gseapy.
    
    Args:
        de_genes: List of differentially expressed gene names
        gene_names: All gene names in dataset
        dataset_name: Organism name for gseapy
        databases: List of enrichment databases
        output_dir: Output directory
    
    Returns:
        dict: Enrichment results for each database
    """
    print(f"\n[Pathway Enrichment] Analyzing {len(de_genes)} DE genes...")
    
    if not GSEAPY_AVAILABLE:
        print("  [Warning] gseapy not available. Skipping pathway enrichment.")
        print("  Install with: pip install gseapy")
        return {}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    enrichment_results = {}
    
    for db in databases:
        try:
            print(f"\n  Database: {db}...")
            enr = gp.enrichr(gene_list=de_genes,
                           gene_sets=db,
                           organism=dataset_name,
                           outdir=None,
                           cutoff=0.05,
                           verbose=False)
            
            if enr is not None and hasattr(enr, 'res2d') and len(enr.res2d) > 0:
                results_df = enr.res2d
                enrichment_results[db] = results_df
                print(f"    Found {len(results_df)} enriched pathways")
                
                if len(results_df) > 0:
                    print(f"    Top 5 pathways:")
                    for i, row in results_df.head(5).iterrows():
                        print(f"      {row.get('Term', 'N/A')}: padj={row.get('Adjusted P-value', 'N/A')}")
            else:
                print(f"    No significant pathways found")
                enrichment_results[db] = pd.DataFrame()
                
        except Exception as e:
            print(f"    [Error] Failed to run enrichment for {db}: {e}")
            enrichment_results[db] = pd.DataFrame()
    
    return enrichment_results


def calculate_signature_scores(expression_data, gene_names, signature_name, signature_genes):
    """
    Calculate signature scores for a gene set.
    
    Args:
        expression_data: numpy array (n_samples, n_genes)
        gene_names: array of gene identifiers
        signature_name: Name of signature
        signature_genes: List of genes in signature
    
    Returns:
        array: Signature scores for each sample
    """
    # Normalize gene names (uppercase, remove version numbers, handle various formats)
    def normalize_gene_name(gene):
        """Normalize gene name for matching."""
        if not isinstance(gene, str):
            gene = str(gene)
        # Remove version numbers (e.g., "ESR1.1" -> "ESR1")
        gene = gene.split('.')[0]
        # Remove any whitespace
        gene = gene.strip()
        # Convert to uppercase
        gene = gene.upper()
        # Handle special cases (e.g., "ERBB2" vs "HER2")
        gene_aliases = {
            'HER2': 'ERBB2',
            'HER-2': 'ERBB2',
        }
        return gene_aliases.get(gene, gene)
    
    gene_names_clean = [normalize_gene_name(g) for g in gene_names]
    signature_genes_clean = [normalize_gene_name(g) for g in signature_genes]
    
    # Find matching genes (try exact match first, then partial match)
    matching_indices = []
    matched_genes = []
    unmatched_genes = []
    
    for sig_gene in signature_genes_clean:
        if sig_gene in gene_names_clean:
            idx = gene_names_clean.index(sig_gene)
            matching_indices.append(idx)
            matched_genes.append(sig_gene)
        else:
            # Try partial match (gene name contains signature gene or vice versa)
            found = False
            for i, gene_name in enumerate(gene_names_clean):
                if sig_gene in gene_name or gene_name in sig_gene:
                    if i not in matching_indices:  # Avoid duplicates
                        matching_indices.append(i)
                        matched_genes.append(f"{sig_gene} (matched: {gene_names[i]})")
                        found = True
                        break
            if not found:
                unmatched_genes.append(sig_gene)
    
    if len(matching_indices) == 0:
        print(f"    [Warning] No matching genes found for {signature_name}")
        print(f"      Looking for: {signature_genes[:5]}...")
        print(f"      Sample gene names: {gene_names[:5] if len(gene_names) > 0 else 'N/A'}...")
        return np.zeros(expression_data.shape[0])
    
    # Calculate mean expression of signature genes
    signature_expr = expression_data[:, matching_indices]
    signature_scores = np.mean(signature_expr, axis=1)
    
    if len(unmatched_genes) > 0:
        print(f"    [Info] {signature_name}: {len(matched_genes)}/{len(signature_genes)} genes matched")
        if len(unmatched_genes) <= 5:
            print(f"      Unmatched: {unmatched_genes}")
    
    return signature_scores


def cell_type_signature_analysis(expression_data, gene_names, clusters, 
                                output_dir='results/biological_interpretation'):
    """
    Analyze cell-type and functional signatures for each cluster.
    
    Args:
        expression_data: numpy array (n_samples, n_genes)
        gene_names: array of gene identifiers
        clusters: array of cluster assignments
        output_dir: Output directory
    
    Returns:
        dict: Signature scores for each cluster
    """
    print(f"\n[Cell-Type Signatures] Analyzing {len(BREAST_CANCER_SIGNATURES)} signatures...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    unique_clusters = sorted(set(clusters))
    signature_results = {}
    
    # Calculate signature scores for all samples
    all_signature_scores = {}
    for sig_name, sig_genes in BREAST_CANCER_SIGNATURES.items():
        scores = calculate_signature_scores(expression_data, gene_names, sig_name, sig_genes)
        all_signature_scores[sig_name] = scores
    
    # Analyze per cluster
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        cluster_results = {}
        
        print(f"\n  Cluster {cluster_id} (n={cluster_mask.sum()}):")
        
        for sig_name, scores in all_signature_scores.items():
            cluster_scores = scores[cluster_mask]
            other_scores = scores[~cluster_mask]
            
            cluster_mean = np.mean(cluster_scores)
            other_mean = np.mean(other_scores)
            
            # Statistical test
            try:
                t_stat, p_val = ttest_ind(cluster_scores, other_scores)
            except:
                p_val = 1.0
            
            cluster_results[sig_name] = {
                'cluster_mean': cluster_mean,
                'other_mean': other_mean,
                'difference': cluster_mean - other_mean,
                'pvalue': p_val,
                'enriched': cluster_mean > other_mean and p_val < 0.05
            }
            
            status = "↑ ENRICHED" if cluster_results[sig_name]['enriched'] else ""
            if cluster_mean < other_mean and p_val < 0.05:
                status = "↓ DEPLETED"
            
            print(f"    {sig_name:20s}: {cluster_mean:6.3f} vs {other_mean:6.3f} (p={p_val:.4f}) {status}")
        
        signature_results[cluster_id] = cluster_results
    
    # Save results
    results_df = []
    for cluster_id, cluster_results in signature_results.items():
        for sig_name, sig_data in cluster_results.items():
            results_df.append({
                'cluster': cluster_id,
                'signature': sig_name,
                'cluster_mean': sig_data['cluster_mean'],
                'other_mean': sig_data['other_mean'],
                'difference': sig_data['difference'],
                'pvalue': sig_data['pvalue'],
                'enriched': sig_data['enriched']
            })
    
    results_df = pd.DataFrame(results_df)
    output_file = output_dir / 'signature_scores.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n  Saved signature scores to {output_file}")
    
    return signature_results, results_df


def interpret_cluster_biology(cluster_id, de_results, pathway_results, signature_results):
    """
    Generate biological interpretation for a cluster.
    
    Args:
        cluster_id: Cluster ID
        de_results: Differential expression results DataFrame
        pathway_results: Pathway enrichment results dict
        signature_results: Signature analysis results dict
    
    Returns:
        dict: Biological interpretation
    """
    interpretation = {
        'cluster_id': cluster_id,
        'highly_expressed_genes': [],
        'enriched_pathways': [],
        'cell_type_signatures': [],
        'biological_interpretation': ''
    }
    
    # Top upregulated genes
    if len(de_results) > 0:
        top_genes = de_results.head(10)['gene'].tolist()
        interpretation['highly_expressed_genes'] = top_genes
    
    # Enriched pathways
    for db_name, pathway_df in pathway_results.items():
        if len(pathway_df) > 0:
            top_pathways = pathway_df.head(5)['Term'].tolist()
            interpretation['enriched_pathways'].extend(top_pathways)
    
    # Cell-type signatures
    if cluster_id in signature_results:
        cluster_sigs = signature_results[cluster_id]
        enriched_sigs = [sig for sig, data in cluster_sigs.items() 
                        if data.get('enriched', False)]
        interpretation['cell_type_signatures'] = enriched_sigs
    
    # Generate text interpretation
    text_parts = [f"Cluster {cluster_id}:"]
    
    if interpretation['highly_expressed_genes']:
        top_5 = ', '.join(interpretation['highly_expressed_genes'][:5])
        text_parts.append(f"Highly expressed: {top_5}")
    
    if interpretation['enriched_pathways']:
        top_3 = ', '.join(interpretation['enriched_pathways'][:3])
        text_parts.append(f"Enriched pathways: {top_3}")
    
    if interpretation['cell_type_signatures']:
        sigs = ', '.join(interpretation['cell_type_signatures'])
        text_parts.append(f"Signatures: {sigs}")
    
    interpretation['biological_interpretation'] = '\n'.join(text_parts)
    
    return interpretation


def create_biological_interpretation_figures(dataset_name, 
                                             output_dir='results/biological_interpretation',
                                             top_n_genes=10):
    """
    Create comprehensive visualization figures for biological interpretation results.
    
    Args:
        dataset_name: Name of dataset ('tcga' or 'gse96058')
        output_dir: Output directory
        top_n_genes: Number of top genes to show in bar plots
    """
    output_dir = Path(output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Visualization] Creating figures for {dataset_name}...")
    
    # Load results
    signature_scores_file = output_dir / 'signature_scores.csv'
    if not signature_scores_file.exists():
        print(f"  [Error] Signature scores file not found: {signature_scores_file}")
        return
    
    signature_scores = pd.read_csv(signature_scores_file)
    
    # Get list of clusters
    clusters = sorted(signature_scores['cluster'].unique())
    n_clusters = len(clusters)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # ============================================================================
    # FIGURE 1: Signature Scores Heatmap
    # ============================================================================
    print("  Creating signature scores heatmap...")
    
    # Pivot data for heatmap
    signature_pivot = signature_scores.pivot_table(
        index='signature', 
        columns='cluster', 
        values='difference',
        aggfunc='mean'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, n_clusters * 1.2), 6))
    
    # Create heatmap
    sns.heatmap(
        signature_pivot,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-3,
        vmax=3,
        cbar_kws={'label': 'Signature Score Difference'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_title(f'{dataset_name.upper()}: Cell-Type Signature Scores by Cluster', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Signature', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    heatmap_file = output_dir / f'{dataset_name}_signature_heatmap.png'
    plt.savefig(heatmap_file, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [Saved] {heatmap_file.name}")
    
    # ============================================================================
    # FIGURE 2: Top DE Genes per Cluster (Bar Plot)
    # ============================================================================
    print("  Creating top DE genes bar plots...")
    
    # Determine grid layout
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_clusters == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, cluster_id in enumerate(clusters):
        de_file = output_dir / f'cluster_{cluster_id}_differential_expression.csv'
        if not de_file.exists():
            if idx < len(axes):
                axes[idx].text(0.5, 0.5, f'Cluster {cluster_id}\nNo DE data', 
                              ha='center', va='center', fontsize=12)
                axes[idx].set_xticks([])
                axes[idx].set_yticks([])
            continue
        
        de_data = pd.read_csv(de_file)
        
        # Get top upregulated and downregulated
        top_up = de_data[de_data['log2fc'] > 0].nlargest(top_n_genes, 'log2fc')
        top_down = de_data[de_data['log2fc'] < 0].nsmallest(top_n_genes, 'log2fc')
        
        # Combine and sort
        top_genes = pd.concat([top_up, top_down]).sort_values('log2fc')
        
        if len(top_genes) == 0 or idx >= len(axes):
            if idx < len(axes):
                axes[idx].text(0.5, 0.5, f'Cluster {cluster_id}\nNo significant genes', 
                              ha='center', va='center', fontsize=12)
                axes[idx].set_xticks([])
                axes[idx].set_yticks([])
            continue
        
        # Create horizontal bar plot
        colors = ['#d73027' if x > 0 else '#4575b4' for x in top_genes['log2fc']]
        axes[idx].barh(range(len(top_genes)), top_genes['log2fc'], color=colors, alpha=0.7)
        axes[idx].set_yticks(range(len(top_genes)))
        axes[idx].set_yticklabels(top_genes['gene'], fontsize=8)
        axes[idx].set_xlabel('log2 Fold Change', fontsize=10)
        axes[idx].set_title(f'Cluster {cluster_id} (n={len(de_data)} genes)', 
                           fontsize=11, fontweight='bold')
        axes[idx].axvline(0, color='black', linestyle='--', linewidth=0.5)
        axes[idx].grid(axis='x', alpha=0.3)
    
    # Hide extra subplots
    for idx in range(len(clusters), len(axes)):
        if idx < len(axes):
            axes[idx].axis('off')
    
    plt.suptitle(f'{dataset_name.upper()}: Top Differential Genes by Cluster', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    de_plot_file = output_dir / f'{dataset_name}_top_de_genes.png'
    plt.savefig(de_plot_file, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [Saved] {de_plot_file.name}")
    
    # ============================================================================
    # FIGURE 3: Signature Scores Bar Plot (Grouped)
    # ============================================================================
    print("  Creating signature scores bar plot...")
    
    signatures = sorted(signature_scores['signature'].unique())
    n_signatures = len(signatures)
    
    if n_signatures > 0:
        fig, axes = plt.subplots(1, n_signatures, figsize=(4*n_signatures, 6))
        if n_signatures == 1:
            axes = [axes]
        
        for idx, sig in enumerate(signatures):
            sig_data = signature_scores[signature_scores['signature'] == sig].sort_values('cluster')
            
            colors = ['#2ecc71' if x else '#e74c3c' for x in sig_data['enriched']]
            bars = axes[idx].bar(range(len(sig_data)), sig_data['difference'], color=colors, alpha=0.7)
            
            axes[idx].set_xticks(range(len(sig_data)))
            axes[idx].set_xticklabels([f'C{c}' for c in sig_data['cluster']], rotation=45, ha='right')
            axes[idx].set_ylabel('Score Difference', fontsize=10)
            axes[idx].set_title(sig.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            axes[idx].axhline(0, color='black', linestyle='-', linewidth=0.5)
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add significance markers
            for i, (bar, pval) in enumerate(zip(bars, sig_data['pvalue'])):
                if pval < 0.001:
                    axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                  '***', ha='center', va='bottom' if bar.get_height() > 0 else 'top',
                                  fontsize=8, fontweight='bold')
                elif pval < 0.01:
                    axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                  '**', ha='center', va='bottom' if bar.get_height() > 0 else 'top',
                                  fontsize=8, fontweight='bold')
                elif pval < 0.05:
                    axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                  '*', ha='center', va='bottom' if bar.get_height() > 0 else 'top',
                                  fontsize=8, fontweight='bold')
        
        plt.suptitle(f'{dataset_name.upper()}: Signature Enrichment by Cluster', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        sig_bar_file = output_dir / f'{dataset_name}_signature_bars.png'
        plt.savefig(sig_bar_file, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    [Saved] {sig_bar_file.name}")
    
    # ============================================================================
    # FIGURE 4: Comprehensive Summary Figure
    # ============================================================================
    print("  Creating comprehensive summary figure...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Signature Heatmap (top, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :])
    sns.heatmap(
        signature_pivot,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-3,
        vmax=3,
        cbar_kws={'label': 'Score Difference'},
        ax=ax1,
        linewidths=0.5,
        linecolor='gray'
    )
    ax1.set_title('Cell-Type Signature Scores', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Cluster', fontsize=10)
    ax1.set_ylabel('Signature', fontsize=10)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    
    # 2. Top DE Genes for first cluster (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    if len(clusters) > 0:
        de_file = output_dir / f'cluster_{clusters[0]}_differential_expression.csv'
        if de_file.exists():
            de_data = pd.read_csv(de_file)
            de_data['abs_log2fc'] = de_data['log2fc'].abs()
            top_genes = de_data.nlargest(10, 'abs_log2fc')
            if len(top_genes) > 0:
                colors = ['#d73027' if x > 0 else '#4575b4' for x in top_genes['log2fc']]
                ax2.barh(range(len(top_genes)), top_genes['log2fc'], color=colors, alpha=0.7)
                ax2.set_yticks(range(len(top_genes)))
                ax2.set_yticklabels(top_genes['gene'], fontsize=9)
                ax2.set_xlabel('log2FC', fontsize=10)
                ax2.set_title(f'Top DE Genes: Cluster {clusters[0]}', fontsize=11, fontweight='bold')
                ax2.axvline(0, color='black', linestyle='--', linewidth=0.5)
                ax2.grid(axis='x', alpha=0.3)
    
    # 3. Signature enrichment summary (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    enriched_counts = signature_scores[signature_scores['enriched'] == True].groupby('signature').size()
    depleted_counts = signature_scores[signature_scores['enriched'] == False].groupby('signature').size()
    
    x = np.arange(len(signatures))
    width = 0.35
    
    ax3.bar(x - width/2, [enriched_counts.get(sig, 0) for sig in signatures], 
           width, label='Enriched', color='#2ecc71', alpha=0.7)
    ax3.bar(x + width/2, [depleted_counts.get(sig, 0) for sig in signatures], 
           width, label='Depleted', color='#e74c3c', alpha=0.7)
    
    ax3.set_xlabel('Signature', fontsize=10)
    ax3.set_ylabel('Number of Clusters', fontsize=10)
    ax3.set_title('Signature Enrichment Summary', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([s.replace('_', ' ').title() for s in signatures], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Top pathways (if available) or cluster sizes (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    # Try to load pathway data
    pathway_files = list(output_dir.glob('cluster_*_pathways_*.csv'))
    if pathway_files:
        # Count pathways per cluster
        pathway_counts = {}
        for pf in pathway_files:
            cluster_id = int(pf.stem.split('_')[1])
            try:
                pathway_data = pd.read_csv(pf)
                if len(pathway_data) > 0:
                    pathway_counts[cluster_id] = len(pathway_data[pathway_data['Adjusted P-value'] < 0.05])
                else:
                    pathway_counts[cluster_id] = 0
            except:
                pathway_counts[cluster_id] = 0
        
        if pathway_counts:
            cluster_ids = sorted(pathway_counts.keys())
            counts = [pathway_counts[c] for c in cluster_ids]
            ax4.bar(range(len(cluster_ids)), counts, color='steelblue', alpha=0.7)
            ax4.set_xticks(range(len(cluster_ids)))
            ax4.set_xticklabels([f'C{c}' for c in cluster_ids])
            ax4.set_ylabel('Significant Pathways', fontsize=10)
            ax4.set_title('Pathway Enrichment by Cluster', fontsize=11, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
    else:
        # Show cluster sizes from signature data
        cluster_sizes = []
        for cluster_id in clusters:
            cluster_data = signature_scores[signature_scores['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                # Try to estimate from first row (this is approximate)
                cluster_sizes.append(1)  # Placeholder
            else:
                cluster_sizes.append(0)
        
        if sum(cluster_sizes) > 0:
            ax4.bar(range(len(clusters)), cluster_sizes, color='steelblue', alpha=0.7)
            ax4.set_xticks(range(len(clusters)))
            ax4.set_xticklabels([f'C{c}' for c in clusters])
            ax4.set_ylabel('Relative Size', fontsize=10)
            ax4.set_title('Cluster Sizes', fontsize=11, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
    
    # 5. Summary statistics (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Calculate summary stats
    total_de_genes = 0
    total_clusters_with_de = 0
    for cluster_id in clusters:
        de_file = output_dir / f'cluster_{cluster_id}_differential_expression.csv'
        if de_file.exists():
            try:
                de_data = pd.read_csv(de_file)
                total_de_genes += len(de_data)
                total_clusters_with_de += 1
            except:
                pass
    
    summary_text = f"""
    Dataset: {dataset_name.upper()}
    Number of Clusters: {n_clusters}
    Clusters with DE data: {total_clusters_with_de}
    Total DE genes: {total_de_genes}
    Signatures analyzed: {len(signatures)}
    """
    
    ax5.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax5.set_title('Summary Statistics', fontsize=11, fontweight='bold')
    
    plt.suptitle(f'{dataset_name.upper()}: Biological Interpretation Summary', 
                fontsize=16, fontweight='bold', y=0.995)
    
    summary_file = output_dir / f'{dataset_name}_biological_summary.png'
    plt.savefig(summary_file, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [Saved] {summary_file.name}")
    
    print(f"  [Complete] All figures saved to {output_dir}")


def biological_interpretation_pipeline(dataset_name, 
                                     processed_dir='data/processed',
                                     clustering_dir='data/clusterings',
                                     output_dir='results/biological_interpretation',
                                     log2fc_threshold=1.0,
                                     pvalue_threshold=0.05,
                                     use_original_data=True):
    """
    Complete biological interpretation pipeline.
    
    Args:
        dataset_name: 'tcga' or 'gse96058'
        processed_dir: Directory with processed data
        clustering_dir: Directory with cluster assignments
        output_dir: Output directory
        log2fc_threshold: Log2 fold change threshold for DE
        pvalue_threshold: P-value threshold for DE
    
    Returns:
        dict: Complete interpretation results
    """
    print("\n" + "="*80)
    print(f"BIOLOGICAL INTERPRETATION: {dataset_name.upper()}")
    print("="*80)
    
    output_dir = Path(output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    expression_data, gene_names, sample_names, clusters = load_expression_and_clusters(
        dataset_name, processed_dir, clustering_dir, use_original_data=use_original_data
    )
    
    unique_clusters = sorted(set(clusters))
    all_interpretations = {}
    
    # 1. Differential expression analysis
    print("\n" + "-"*80)
    print("1. DIFFERENTIAL GENE EXPRESSION ANALYSIS")
    print("-"*80)
    
    de_results_all = {}
    for cluster_id in unique_clusters:
        de_full, de_sig = differential_expression_analysis(
            expression_data, gene_names, clusters, cluster_id,
            log2_fold_change_threshold=log2fc_threshold,
            pvalue_threshold=pvalue_threshold
        )
        
        de_results_all[cluster_id] = {
            'full': de_full,
            'significant': de_sig
        }
        
        # Save DE results
        if len(de_sig) > 0:
            de_file = output_dir / f"cluster_{cluster_id}_differential_expression.csv"
            de_sig.to_csv(de_file, index=False)
            print(f"  Saved DE results to {de_file}")
    
    # 2. Pathway enrichment
    print("\n" + "-"*80)
    print("2. PATHWAY ENRICHMENT ANALYSIS")
    print("-"*80)
    
    pathway_results_all = {}
    for cluster_id in unique_clusters:
        de_sig = de_results_all[cluster_id]['significant']
        
        if len(de_sig) > 0:
            # Get upregulated genes (log2FC > threshold)
            up_genes = de_sig[de_sig['log2fc'] > log2fc_threshold]['gene'].tolist()
            
            if len(up_genes) > 0:
                pathway_results = pathway_enrichment_analysis(
                    up_genes, gene_names, 
                    dataset_name='human',
                    output_dir=output_dir
                )
                pathway_results_all[cluster_id] = pathway_results
                
                # Save pathway results
                for db_name, pathway_df in pathway_results.items():
                    if len(pathway_df) > 0:
                        pathway_file = output_dir / f"cluster_{cluster_id}_pathways_{db_name}.csv"
                        pathway_df.to_csv(pathway_file, index=False)
            else:
                print(f"  Cluster {cluster_id}: No upregulated genes for pathway enrichment")
                pathway_results_all[cluster_id] = {}
        else:
            print(f"  Cluster {cluster_id}: No significant DE genes")
            pathway_results_all[cluster_id] = {}
    
    # 3. Cell-type signature analysis
    print("\n" + "-"*80)
    print("3. CELL-TYPE SIGNATURE ANALYSIS")
    print("-"*80)
    
    signature_results, signature_df = cell_type_signature_analysis(
        expression_data, gene_names, clusters, output_dir
    )
    
    # 4. Generate interpretations
    print("\n" + "-"*80)
    print("4. BIOLOGICAL INTERPRETATION SUMMARY")
    print("-"*80)
    
    for cluster_id in unique_clusters:
        de_sig = de_results_all[cluster_id]['significant']
        pathway_res = pathway_results_all.get(cluster_id, {})
        sig_res = signature_results.get(cluster_id, {})
        
        interpretation = interpret_cluster_biology(cluster_id, de_sig, pathway_res, signature_results)
        all_interpretations[cluster_id] = interpretation
        
        print(f"\n{interpretation['biological_interpretation']}")
    
    # Save summary
    summary_file = output_dir / 'biological_interpretation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"BIOLOGICAL INTERPRETATION SUMMARY: {dataset_name.upper()}\n")
        f.write("="*80 + "\n\n")
        
        for cluster_id in unique_clusters:
            interp = all_interpretations[cluster_id]
            f.write(f"\n{interp['biological_interpretation']}\n")
            f.write("-"*80 + "\n")
    
    print(f"\n  Saved summary to {summary_file}")
    
    # Save detailed results
    results_file = output_dir / 'biological_interpretation_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump({
            'de_results': de_results_all,
            'pathway_results': pathway_results_all,
            'signature_results': signature_results,
            'interpretations': all_interpretations
        }, f)
    
    print(f"  Saved detailed results to {results_file}")
    
    # Create visualization figures
    try:
        create_biological_interpretation_figures(dataset_name, output_dir)
    except Exception as e:
        print(f"  [Warning] Failed to create figures: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        'de_results': de_results_all,
        'pathway_results': pathway_results_all,
        'signature_results': signature_results,
        'interpretations': all_interpretations
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Biological interpretation of BIGCLAM clusters')
    parser.add_argument('--dataset', type=str, default='both', choices=['tcga', 'gse96058', 'both'],
                       help='Dataset to analyze (default: both)')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                       help='Directory with processed data')
    parser.add_argument('--clustering-dir', type=str, default='data/clusterings',
                       help='Directory with cluster assignments')
    parser.add_argument('--output-dir', type=str, default='results/biological_interpretation',
                       help='Output directory')
    parser.add_argument('--log2fc-threshold', type=float, default=1.0,
                       help='Log2 fold change threshold for DE')
    parser.add_argument('--pvalue-threshold', type=float, default=0.05,
                       help='P-value threshold for DE')
    parser.add_argument('--use-processed-data', action='store_true',
                       help='Use processed data (feature-selected) instead of original data')
    
    args = parser.parse_args()
    
    datasets = ['tcga', 'gse96058'] if args.dataset == 'both' else [args.dataset]
    
    for dataset in datasets:
        biological_interpretation_pipeline(
            dataset,
            processed_dir=args.processed_dir,
            clustering_dir=args.clustering_dir,
            output_dir=args.output_dir,
            log2fc_threshold=args.log2fc_threshold,
            pvalue_threshold=args.pvalue_threshold,
            use_original_data=not args.use_processed_data
        )

