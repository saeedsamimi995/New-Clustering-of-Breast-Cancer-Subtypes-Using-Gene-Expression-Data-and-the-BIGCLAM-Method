"""
Parameter Grid Search Module

Sweeps cosine similarity thresholds (variance fixed to dataset mean),
evaluates clustering performance, and generates paper-ready visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import yaml
import sys
import tempfile
import shutil
import gc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing import preprocess_data
from src.graph.graph_construction import build_similarity_graph
from src.clustering.clustering import cluster_data
from src.evaluation.evaluators import evaluate_clustering


def generate_precise_range(start, end, step):
    """
    Generate a range of values with precise decimal steps, avoiding floating point errors.
    
    Uses integer arithmetic internally to avoid precision issues with np.arange().
    
    Args:
        start: Starting value
        end: Ending value (inclusive)
        step: Step size
        
    Returns:
        numpy array: Array of values from start to end with step spacing
    """
    # Convert to integers to avoid floating point errors
    # Find the number of decimal places in step
    step_str = f"{step:.10f}".rstrip('0')
    if '.' in step_str:
        decimals = len(step_str.split('.')[1])
    else:
        decimals = 0
    
    # Scale everything to integers
    scale = 10 ** decimals
    start_int = int(start * scale)
    end_int = int(end * scale)
    step_int = int(step * scale)
    
    # Generate range as integers
    values_int = np.arange(start_int, end_int + step_int, step_int)
    
    # Convert back to floats
    values = values_int / scale
    
    # Round to handle any remaining floating point issues
    return np.round(values, decimals=decimals)


def clear_memory():
    """
    Aggressively clear memory by forcing garbage collection and clearing caches.
    Useful before switching between datasets (e.g., TCGA → GSE).
    """
    # Clear matplotlib caches
    plt.close('all')
    
    # Clear any module-level caches if they exist
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    
    # Force garbage collection
    gc.collect()
    gc.collect()  # Call twice to handle cyclic references


def run_single_combination(
    preprocessed_data,
    target_labels,
    similarity_threshold,
    temp_dir,
    bigclam_config,
    dataset_name,
    original_feature_count,
    variance_threshold_value,
    correlation_threshold_value,
    laplacian_neighbors,
    requested_feature_count,
    variance_threshold="mean"
):
    """
    Run full pipeline for a single parameter combination.
    
    Args:
        preprocessed_data: Preprocessed expression data (after log2 and normalization, before variance filtering)
        target_labels: Target labels array
        variance_threshold: Reserved for logging; automatically set to dataset mean
        similarity_threshold: Similarity threshold for graph construction
        temp_dir: Temporary directory for intermediate files
        bigclam_config: BIGCLAM configuration dict
        dataset_name: Name of dataset
        
    Returns:
        dict: Results including metrics and metadata
    """
    results = {
        'variance_threshold': float(variance_threshold_value),
        'correlation_threshold': float(correlation_threshold_value),
        'laplacian_neighbors': int(laplacian_neighbors),
        'requested_features': requested_feature_count,
        'similarity_threshold': similarity_threshold,
        'success': False,
        'error': None
    }
    
    try:
        expression_data = preprocessed_data.copy()
        n_features = expression_data.shape[1]
        
        # Step 2: Build graph with specific similarity threshold
        temp_graph_dir = Path(temp_dir) / 'graphs'
        temp_graph_dir.mkdir(parents=True, exist_ok=True)
        
        adjacency, similarity = build_similarity_graph(
            expression_data,
            threshold=float(similarity_threshold),
            use_sparse=True
        )
        
        # Calculate graph statistics
        if hasattr(adjacency, 'nnz'):
            # Sparse matrix
            n_edges = adjacency.nnz // 2  # Divide by 2 because adjacency is symmetric
            n_nodes = expression_data.shape[0]
            density = (n_edges * 2) / (n_nodes * (n_nodes - 1)) * 100 if n_nodes > 1 else 0
        else:
            # Dense matrix
            n_edges = (adjacency > 0).sum() // 2  # Divide by 2 because adjacency is symmetric
            n_nodes = expression_data.shape[0]
            density = (n_edges * 2) / (n_nodes * (n_nodes - 1)) * 100 if n_nodes > 1 else 0
        
        # Step 3: Cluster
        max_communities = bigclam_config.get('max_communities', 10)
        iterations = bigclam_config.get('iterations', 100)
        lr = bigclam_config.get('learning_rate', 0.08)
        
        # Get optimization parameters from config
        adaptive_lr = bigclam_config.get('adaptive_lr', True)
        adaptive_iterations = bigclam_config.get('adaptive_iterations', True)
        early_stopping = bigclam_config.get('early_stopping', True)
        convergence_threshold = float(bigclam_config.get('convergence_threshold', 1e-6))
        patience = int(bigclam_config.get('patience', 10))
        
        # Get criterion (AIC/BIC) for this dataset
        criterion_config = bigclam_config.get('model_selection_criterion', {})
        criterion = criterion_config.get(dataset_name, criterion_config.get('default', 'BIC'))
        
        # Get num_restarts for this dataset (from dataset_specific or default)
        dataset_specific_config = bigclam_config.get('dataset_specific', {})
        dataset_config = dataset_specific_config.get(dataset_name, {})
        num_restarts = int(dataset_config.get('num_restarts', bigclam_config.get('num_restarts', 1)))
        
        communities, membership, optimal_k = cluster_data(
            adjacency,
            max_communities=max_communities,
            iterations=iterations,
            lr=lr,
            criterion=criterion,
            adaptive_lr=adaptive_lr,
            adaptive_iterations=adaptive_iterations,
            early_stopping=early_stopping,
            convergence_threshold=convergence_threshold,
            patience=patience,
            num_restarts=num_restarts
        )
        
        n_communities = len(set(communities))
        
        # Step 4: Evaluate
        eval_output = evaluate_clustering(
            communities,
            target_labels,
            f"{dataset_name}_var{variance_threshold}_sim{similarity_threshold}"
        )
        
        # Handle return format: evaluate_clustering returns (results_dict, result_df) tuple
        if eval_output is None:
            results['error'] = 'Evaluation failed - no valid labels'
        else:
            eval_results, _ = eval_output
            
            if eval_results:
                results.update({
                    'success': True,
                    'ari': eval_results['ari'],
                    'nmi': eval_results['nmi'],
                    'purity': eval_results['purity'],
                    'f1_macro': eval_results['f1_macro'],
                    'n_communities': n_communities,
                    'optimal_k': optimal_k,
                    'n_features': n_features,
                    'graph_density': density,
                    'n_edges': n_edges,
                    'n_samples': len(communities)
                })
            else:
                results['error'] = 'Evaluation failed - no valid labels'
        
        # Explicit memory cleanup for large arrays (don't delete preprocessed_data or target_labels - they're shared)
        del expression_data  # This is the variance-filtered copy, safe to delete
        del adjacency, similarity
        del communities, membership
        if 'eval_results' in locals():
            del eval_results
        if 'eval_output' in locals():
            del eval_output
        if 'selected_features' in locals():
            del selected_features
            
    except Exception as e:
        results['error'] = str(e)
        print(f"    [ERROR] {e}")
        # Cleanup even on error - try to delete any variables that might exist
        # (don't delete preprocessed_data or target_labels - they're shared)
        try:
            if 'expression_data' in locals():
                del expression_data
            if 'adjacency' in locals():
                del adjacency
            if 'similarity' in locals():
                del similarity
            if 'communities' in locals():
                del communities
            if 'membership' in locals():
                del membership
            if 'selected_features' in locals():
                del selected_features
        except:
            pass  # Ignore errors if variables don't exist
    
    # Force garbage collection
    gc.collect()
    
    return results


def run_grid_search(dataset_name, input_file, similarity_range,
                   config_path='config/config.yml', output_dir='results/grid_search'):
    """
    Run grid search over variance and similarity thresholds.
    Supports resuming from previous runs by checking for existing results.
    
    NOTE: This function starts from data_preprocessing.py. It takes prepared CSV files
          (*_target_added.csv) as input. It does NOT access raw clinical/expression files.
          Pipeline: *_target_added.csv -> preprocess_data() -> graph -> cluster -> evaluate
    
    Args:
        dataset_name: Name of dataset (e.g., 'tcga_brca_data')
        input_file: Path to prepared CSV file (*_target_added.csv from data_preparing.py output)
                    This is NOT raw clinical/expression files - those are only used in data_preparing.py
        similarity_range: List of similarity thresholds to test
        config_path: Path to config file
        output_dir: Directory to save results
        
    Returns:
        dict: Grid search results and recommendations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    bigclam_config = config.get('bigclam', {})
    preprocessing_config = config.get('preprocessing', {})
    correlation_threshold_mode = preprocessing_config.get('correlation_threshold_mode', 'mean')
    laplacian_neighbors_cfg = int(preprocessing_config.get('laplacian_neighbors', 5))
    num_selected_features_cfg = preprocessing_config.get('num_selected_features', None)
    if num_selected_features_cfg is not None:
        num_selected_features_cfg = int(num_selected_features_cfg)
    
    # Check for existing results file for resume functionality
    results_file = output_dir / f'{dataset_name}_grid_search_results.csv'
    completed_similarities = set()
    
    if results_file.exists():
        print(f"\n[INFO] Found existing results file: {results_file}")
        try:
            existing_df = pd.read_csv(results_file)
            if len(existing_df) > 0:
                # Create set of completed combinations (only successful ones - retry failed ones)
                # Round to 6 decimal places to handle float precision issues
                for _, row in existing_df.iterrows():
                    if pd.notna(row.get('similarity_threshold')):
                        is_successful = row.get('success', False)
                        if is_successful:
                            sim_val = round(float(row['similarity_threshold']), 6)
                            completed_similarities.add(sim_val)
                print(f"[INFO] Found {len(completed_similarities)} successfully completed similarity thresholds")
                
                # Count failed combinations that will be retried
                failed_count = len(existing_df) - len(completed_similarities)
                if failed_count > 0:
                    print(f"[INFO] Found {failed_count} failed combinations - will retry them")
                
                if len(completed_similarities) > 0:
                    sample = list(completed_similarities)[:5]
                    print(f"[INFO] Completed similarity sample: {sample}")
                print(f"[INFO] Will resume from where it left off...")
                
                # Only keep successful results - remove failed ones so they can be retried
                successful_df = existing_df[existing_df.get('success', False) == True]
                all_results = successful_df.to_dict('records') if len(successful_df) > 0 else []
            else:
                all_results = []
        except Exception as e:
            print(f"[WARNING] Could not read existing results file: {e}")
            print(f"[INFO] Starting fresh grid search...")
            all_results = []
    else:
        all_results = []
    
    print("\n" + "="*80)
    print(f"PARAMETER GRID SEARCH: {dataset_name}")
    print("="*80)
    print(f"\nVariance threshold: dataset mean (fixed)")
    print(f"Similarity thresholds: {similarity_range}")
    total_combinations = len(similarity_range)
    
    completed_in_range = sum(
        1 for sim_thresh in similarity_range
        if round(float(sim_thresh), 6) in completed_similarities
    )
    
    remaining = total_combinations - completed_in_range
    
    print(f"Total combinations in current range: {total_combinations}")
    print(f"Completed in current range: {completed_in_range}")
    print(f"Remaining to process: {remaining}")
    if completed_in_range > 0:
        print(f"Progress: {completed_in_range/total_combinations*100:.1f}%")
    
    print("\nThis will take some time...\n")
    
    # Load data once before the loop (this is the expensive operation)
    print("\n[INFO] Loading data once for all combinations...")
    from src.preprocessing.data_preprocessing import (
        load_data_with_target,
        apply_log2_transform,
        apply_zscore_normalize,
        apply_mean_variance_filter,
        apply_correlation_pruning,
        apply_laplacian_score_selection
    )
    
    expression_data_raw, target_labels, gene_names, sample_names = load_data_with_target(input_file)
    original_feature_count = expression_data_raw.shape[1]
    
    print("[INFO] Applying variance filter before preprocessing...")
    expression_data_var, gene_names_filtered, _, variance_thresh = apply_mean_variance_filter(
        expression_data_raw,
        gene_names=gene_names,
        verbose=True
    )
    gene_names = gene_names_filtered if gene_names_filtered is not None else gene_names
    
    expression_data_corr, gene_names_corr, _, corr_thresh = apply_correlation_pruning(
        expression_data_var,
        gene_names=gene_names,
        threshold=correlation_threshold_mode,
        verbose=True
    )
    gene_names = gene_names_corr if gene_names_corr is not None else gene_names
    
    expression_data_selected, gene_names_selected, lap_indices, lap_scores = apply_laplacian_score_selection(
        expression_data_corr,
        gene_names=gene_names,
        k_neighbors=laplacian_neighbors_cfg,
        num_features=num_selected_features_cfg,
        verbose=True
    )
    gene_names = gene_names_selected if gene_names_selected is not None else gene_names
    requested_feature_count = (
        int(num_selected_features_cfg)
        if num_selected_features_cfg
        else int(expression_data_selected.shape[1])
    )
    
    # Apply log2 transformation (parameter-independent, do once)
    print("[INFO] Applying log2 transformation...")
    expression_data_preprocessed, is_already_normalized = apply_log2_transform(expression_data_selected)
    
    # Apply z-score normalization (parameter-independent, do once)
    if not is_already_normalized:
        print("[INFO] Applying z-score normalization...")
        expression_data_preprocessed, _ = apply_zscore_normalize(expression_data_preprocessed)
    else:
        print("[INFO] Data already normalized, skipping z-score normalization")
    
    print(f"[INFO] Preprocessed data shape: {expression_data_preprocessed.shape}")
    print(f"[INFO] Data loaded and preprocessed. Starting grid search...\n")
    
    # Clean up raw data to save memory
    del expression_data_raw, expression_data_var, expression_data_corr, expression_data_selected
    gc.collect()
    
    # Create temporary directory for intermediate files
    temp_dir = Path(output_dir) / f'temp_{dataset_name}'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Run grid search
    combination_count = 0
    
    for sim_thresh in similarity_range:
        sim_rounded = round(float(sim_thresh), 6)
        
        if sim_rounded in completed_similarities:
            print(f"\n[SKIP] Already completed: Similarity={sim_thresh}")
            continue
        
        combination_count += 1
        print(f"\n{'='*80}")
        print(f"Testing [{combination_count}/{remaining}]: Similarity={sim_thresh}")
        print(f"{'='*80}")
        
        result = run_single_combination(
            expression_data_preprocessed,  # Preprocessed data (log2 + normalized)
            target_labels,  # Target labels
            sim_thresh,
            temp_dir,
            bigclam_config,
            dataset_name,
            original_feature_count,
            variance_thresh,
            corr_thresh,
            laplacian_neighbors_cfg,
            requested_feature_count
        )
        result['dataset'] = dataset_name
        all_results.append(result)
        
        # Save incrementally after each combination to enable resume
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_file, index=False)
        
        if result['success']:
            print(f"  ✓ Success: {result['n_communities']} communities, ARI={result['ari']:.3f}, NMI={result['nmi']:.3f}")
        else:
            print(f"  ✗ Failed: {result['error']}")
        
        # Clean up temp files for this combination
        for subdir in ['processed', 'graphs', 'clusterings']:
            temp_subdir = temp_dir / subdir
            if temp_subdir.exists():
                shutil.rmtree(temp_subdir)
        
        # Aggressive memory cleanup after each combination
        del result
        gc.collect()
    
    # Convert to DataFrame (use existing if we resumed)
    results_df = pd.DataFrame(all_results)
    
    # Filter successful runs
    successful_df = results_df[results_df['success'] == True].copy()
    
    if len(successful_df) == 0:
        print("\n[ERROR] No successful runs! Check error messages above.")
        return None
    
    # Save final results (already saved incrementally, but save again to ensure consistency)
    results_df.to_csv(results_file, index=False)
    
    # Find best combination
    # Score = weighted combination of metrics
    successful_df['composite_score'] = (
        0.4 * successful_df['ari'] +
        0.3 * successful_df['nmi'] +
        0.2 * successful_df['purity'] +
        0.1 * successful_df['f1_macro']
    )
    
    # Prefer configurations with appropriate number of communities (dataset-specific)
    successful_df['community_bonus'] = 0.0
    if dataset_name == 'gse96058_data':
        # GSE96058: Prefer 5-8 communities (target 7, closer to PAM50's 5 than 4)
        successful_df.loc[(successful_df['n_communities'] >= 5) & (successful_df['n_communities'] <= 8), 'community_bonus'] = 0.1
    else:
        # TCGA-BRCA: Prefer 5+ communities (breast cancer subtypes)
        successful_df.loc[(successful_df['n_communities'] >= 5), 'community_bonus'] = 0.1
    successful_df['final_score'] = successful_df['composite_score'] + successful_df['community_bonus']
    
    best_idx = successful_df['final_score'].idxmax()
    best_config = successful_df.loc[best_idx]
    
    # Generate visualizations
    create_grid_search_visualizations(successful_df, dataset_name, output_dir, best_config)
    
    # Print recommendations
    print("\n" + "="*80)
    print(f"GRID SEARCH RESULTS: {dataset_name}")
    print("="*80)
    print(f"\n📊 Best Configuration:")
    print(f"   Variance threshold: {best_config['variance_threshold']}")
    print(f"   Similarity threshold: {best_config['similarity_threshold']}")
    print(f"\n   Metrics:")
    print(f"     - ARI: {best_config['ari']:.4f}")
    print(f"     - NMI: {best_config['nmi']:.4f}")
    print(f"     - Purity: {best_config['purity']:.4f}")
    print(f"     - F1-score: {best_config['f1_macro']:.4f}")
    print(f"     - Communities: {int(best_config['n_communities'])} (optimal: {int(best_config['optimal_k'])})")
    print(f"     - Features kept: {int(best_config['n_features']):,}")
    print(f"     - Graph density: {best_config['graph_density']:.2f}%")
    
    # Clean up temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # Final memory cleanup after grid search complete
    clear_memory()
    
    return {
        'results_df': results_df,
        'successful_df': successful_df,
        'best_config': best_config.to_dict(),
        'output_dir': output_dir
    }


def create_grid_search_visualizations(df, dataset_name, output_dir, best_config):
    """
    Create similarity-only visualizations for grid search results.
    """
    output_dir = Path(output_dir)
    
    df = df.copy()
    df['similarity_threshold'] = df['similarity_threshold'].astype(float)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg = (
        df.groupby('similarity_threshold')[numeric_cols]
        .mean()
        .sort_index()
    )
    agg['composite'] = (
        0.4 * agg['ari'] +
        0.3 * agg['nmi'] +
        0.2 * agg['purity'] +
        0.1 * agg['f1_macro']
    )
    
    best_sim = best_config['similarity_threshold']
    best_var = best_config['variance_threshold']
    
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Clustering metrics overview
    ax1 = fig.add_subplot(gs[0, 0])
    colors = {
        'ari': '#d73027',
        'nmi': '#fc8d59',
        'purity': '#4575b4',
        'f1_macro': '#1a9850'
    }
    for metric, color in colors.items():
        ax1.plot(
            agg.index,
            agg[metric],
            marker='o',
            linewidth=2,
            markersize=8,
            color=color,
            label=metric.upper()
        )
        ax1.scatter(
            best_sim,
            best_config[metric],
            color=color,
            edgecolor='black',
            s=120,
            zorder=5
        )
    ax1.axvline(best_sim, color='black', linestyle='--', linewidth=1.5, label='Selected threshold')
    ax1.set_xlabel('Similarity Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title(f'{dataset_name}: Clustering Metrics vs. Similarity', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Composite score profile
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        agg.index,
        agg['composite'],
        marker='s',
        linewidth=2.5,
        color='#542788',
        label='Composite score'
    )
    ax2.scatter(best_sim, agg.loc[best_sim, 'composite'], color='#ff7f00', s=150, marker='*', zorder=6, label='Best')
    ax2.axvline(best_sim, color='black', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Similarity Threshold')
    ax2.set_ylabel('Composite Score')
    ax2.set_title('Composite Score (weighted ARI/NMI/Purity/F1)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Features retained
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(
        agg.index,
        agg['n_features'],
        marker='o',
        linewidth=2,
        color='#2b8cbe'
    )
    ax3.axvline(best_sim, color='black', linestyle='--', linewidth=1.5)
    ax3.set_xlabel('Similarity Threshold')
    ax3.set_ylabel('Features Kept')
    ax3.set_title('Selected Features vs. Similarity', fontsize=12, fontweight='bold')
    ax3.ticklabel_format(style='plain', axis='y')
    ax3.grid(True, alpha=0.3)
    
    # Graph density
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(
        agg.index,
        agg['graph_density'],
        marker='o',
        linewidth=2,
        color='#31a354'
    )
    ax4.axvline(best_sim, color='black', linestyle='--', linewidth=1.5)
    ax4.set_xlabel('Similarity Threshold')
    ax4.set_ylabel('Graph Density (%)')
    ax4.set_title('Graph Density vs. Similarity', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Communities vs similarity
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(
        agg.index,
        agg['n_communities'],
        marker='o',
        linewidth=2,
        color='#756bb1',
        label='Communities found'
    )
    ax5.plot(
        agg.index,
        agg['optimal_k'],
        marker='s',
        linewidth=2,
        linestyle='--',
        color='#de2d26',
        label='Optimal k (BIGCLAM)'
    )
    ax5.axvline(best_sim, color='black', linestyle=':', linewidth=1.5, label='Selected threshold')
    ax5.set_xlabel('Similarity Threshold')
    ax5.set_ylabel('Communities')
    ax5.set_title('Community Count vs. Similarity', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Summary text
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    summary_text = f"""
BEST CONFIGURATION – {dataset_name.upper()}
{'='*70}
Fixed Variance Threshold (dataset mean): {best_var:.4f}
Selected Similarity Threshold: {best_sim:.3f}

Clustering Metrics:
  • ARI:    {best_config['ari']:.4f}
  • NMI:    {best_config['nmi']:.4f}
  • Purity: {best_config['purity']:.4f}
  • F1macro:{best_config['f1_macro']:.4f}

Graph/Feature Stats:
  • Features kept: {int(best_config['n_features']):,}
  • Graph density: {best_config['graph_density']:.2f}%
  • Communities:   {int(best_config['n_communities'])} (optimal k = {int(best_config['optimal_k'])})
"""
    ax6.text(0, 0.95, summary_text, fontsize=11, fontfamily='monospace', va='top')
    
    plt.suptitle(f'{dataset_name}: Similarity Grid Search Summary', fontsize=16, fontweight='bold')
    output_path = output_dir / f'{dataset_name}_grid_search_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[Saved] Grid search visualization: {output_path}")
    plt.close(fig)
    
    # Additional single-metric plots
    create_individual_heatmaps(df, dataset_name, output_dir, best_config)


def create_individual_heatmaps(df, dataset_name, output_dir, best_config):
    """Create individual high-quality line plots for each metric (paper-ready)."""
    output_dir = Path(output_dir)
    df = df.copy()
    df['similarity_threshold'] = df['similarity_threshold'].astype(float)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metrics = ['ari', 'nmi', 'purity', 'f1_macro']
    metric_names = [
        'Adjusted Rand Index (ARI)',
        'Normalized Mutual Information (NMI)',
        'Purity',
        'F1-Score (Macro)'
    ]
    palette = ['#d73027', '#fc8d59', '#4575b4', '#1a9850']
    
    agg = (
        df.groupby('similarity_threshold')[numeric_cols]
        .mean()
        .sort_index()
    )
    best_sim = best_config['similarity_threshold']
    
    for metric, metric_name, color in zip(metrics, metric_names, palette):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            agg.index,
            agg[metric],
            marker='o',
            linewidth=2,
            color=color,
            label=metric_name
        )
        ax.scatter(best_sim, best_config[metric], color='black', s=120, marker='*', zorder=5, label='Selected')
        ax.axvline(best_sim, color='black', linestyle='--', linewidth=1.2)
        ax.set_xlabel('Similarity Threshold')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{dataset_name}: {metric_name} vs. Similarity', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        output_file = output_dir / f'{dataset_name}_{metric}_profile.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [Saved] {metric_name} profile: {output_file.name}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Parameter grid search for optimal thresholds')
    parser.add_argument('--dataset', type=str, required=True, choices=['tcga', 'gse96058'],
                       help='Dataset to test')
    parser.add_argument('--config', type=str, default='config/config.yml',
                       help='Config file path')
    parser.add_argument('--similarity_range', type=float, nargs='+', default=None,
                       help='Similarity thresholds to test (default: auto-range)')
    parser.add_argument('--output_dir', type=str, default='results/grid_search',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load config to get input file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_config = config['dataset_preparation']
    grid_search_config = config.get('grid_search', {})
    
    if args.dataset == 'tcga':
        input_file = dataset_config['tcga']['output']
        dataset_name = 'tcga_brca_data'
        if args.similarity_range is None:
            tcga_grid = grid_search_config.get('tcga', {})
            similarity_range = list(generate_precise_range(
                tcga_grid.get('similarity_start', 0.1),
                tcga_grid.get('similarity_end', 0.9),
                tcga_grid.get('similarity_step', 0.05)
            ))
        else:
            similarity_range = args.similarity_range
    else:  # gse96058
        input_file = dataset_config['gse96058']['output']
        dataset_name = 'gse96058_data'
        if args.similarity_range is None:
            gse_grid = grid_search_config.get('gse96058', {})
            similarity_range = list(generate_precise_range(
                gse_grid.get('similarity_start', 0.1),
                gse_grid.get('similarity_end', 0.9),
                gse_grid.get('similarity_step', 0.05)
            ))
        else:
            similarity_range = args.similarity_range
    
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        print(f"Run data preparation first: python -m src.preprocessing.data_preparing --dataset {args.dataset}")
        sys.exit(1)
    
    # Run grid search
    results = run_grid_search(
        dataset_name,
        input_file,
        similarity_range,
        args.config,
        args.output_dir
    )
    
    if results:
        print(f"\n✅ Grid search complete! Results saved to: {results['output_dir']}")

