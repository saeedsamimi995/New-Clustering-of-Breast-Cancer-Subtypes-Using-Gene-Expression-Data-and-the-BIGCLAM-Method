"""
Parameter Grid Search Module

Tests different combinations of variance and similarity thresholds,
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
from itertools import product

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


def run_single_combination(input_file, variance_threshold, similarity_threshold,
                          temp_dir, bigclam_config, dataset_name):
    """
    Run full pipeline for a single parameter combination.
    
    NOTE: input_file must be a prepared CSV file (*_target_added.csv) from data_preparing.py.
          This file has targets in the last row and is ready for data_preprocessing.py.
          It does NOT use raw clinical/expression files.
    
    Args:
        input_file: Path to prepared CSV file (with _target_added suffix) - NOT raw clinical/expression files
        variance_threshold: Numeric variance threshold for feature filtering
        similarity_threshold: Similarity threshold for graph construction
        temp_dir: Temporary directory for intermediate files
        bigclam_config: BIGCLAM configuration dict
        dataset_name: Name of dataset
        
    Returns:
        dict: Results including metrics and metadata
    """
    results = {
        'variance_threshold': variance_threshold,
        'similarity_threshold': similarity_threshold,
        'success': False,
        'error': None
    }
    
    try:
        # Step 1: Preprocess with specific variance threshold
        # input_file is the prepared CSV (*_target_added.csv) - ready for data_preprocessing.py
        # This starts the pipeline from preprocessing, NOT from data preparation
        temp_processed_dir = Path(temp_dir) / 'processed'
        temp_processed_dir.mkdir(parents=True, exist_ok=True)
        
        preprocess_result = preprocess_data(
            input_file,  # This is the prepared CSV with targets, not raw files
            output_dir=str(temp_processed_dir),
            variance_threshold=float(variance_threshold),
            apply_log2=True,
            apply_normalize=True
        )
        
        expression_data = preprocess_result['expression_data']
        target_labels = preprocess_result['target_labels']
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
        
        # Explicit memory cleanup for large arrays
        del expression_data, target_labels
        del adjacency, similarity
        del communities, membership
        if 'eval_results' in locals():
            del eval_results
        if 'eval_output' in locals():
            del eval_output
            
    except Exception as e:
        results['error'] = str(e)
        print(f"    [ERROR] {e}")
        # Cleanup even on error - try to delete any variables that might exist
        try:
            if 'expression_data' in locals():
                del expression_data
            if 'target_labels' in locals():
                del target_labels
            if 'adjacency' in locals():
                del adjacency
            if 'similarity' in locals():
                del similarity
            if 'communities' in locals():
                del communities
            if 'membership' in locals():
                del membership
        except:
            pass  # Ignore errors if variables don't exist
    
    # Force garbage collection
    gc.collect()
    
    return results


def run_grid_search(dataset_name, input_file, variance_range, similarity_range,
                   config_path='config/config.yml', output_dir='results/grid_search'):
    """
    Run grid search over variance and similarity thresholds.
    
    NOTE: This function starts from data_preprocessing.py. It takes prepared CSV files
          (*_target_added.csv) as input. It does NOT access raw clinical/expression files.
          Pipeline: *_target_added.csv -> preprocess_data() -> graph -> cluster -> evaluate
    
    Args:
        dataset_name: Name of dataset (e.g., 'tcga_brca_data')
        input_file: Path to prepared CSV file (*_target_added.csv from data_preparing.py output)
                    This is NOT raw clinical/expression files - those are only used in data_preparing.py
        variance_range: List of variance thresholds to test (numeric values)
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
    
    print("\n" + "="*80)
    print(f"PARAMETER GRID SEARCH: {dataset_name}")
    print("="*80)
    print(f"\nVariance thresholds: {variance_range}")
    print(f"Similarity thresholds: {similarity_range}")
    print(f"Total combinations: {len(variance_range) * len(similarity_range)}")
    print("\nThis will take some time...\n")
    
    # Create temporary directory for intermediate files
    temp_dir = Path(output_dir) / f'temp_{dataset_name}'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Run grid search
    all_results = []
    
    for var_thresh, sim_thresh in product(variance_range, similarity_range):
        print(f"\n{'='*80}")
        print(f"Testing: Variance={var_thresh}, Similarity={sim_thresh}")
        print(f"{'='*80}")
        
        result = run_single_combination(
            input_file,
            var_thresh,
            sim_thresh,
            temp_dir,
            bigclam_config,
            dataset_name
        )
        result['dataset'] = dataset_name
        all_results.append(result)
        
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
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Filter successful runs
    successful_df = results_df[results_df['success'] == True].copy()
    
    if len(successful_df) == 0:
        print("\n[ERROR] No successful runs! Check error messages above.")
        return None
    
    # Save results
    results_df.to_csv(output_dir / f'{dataset_name}_grid_search_results.csv', index=False)
    
    # Find best combination
    # Score = weighted combination of metrics
    successful_df['composite_score'] = (
        0.4 * successful_df['ari'] +
        0.3 * successful_df['nmi'] +
        0.2 * successful_df['purity'] +
        0.1 * successful_df['f1_macro']
    )
    
    # Prefer configurations with 4-5 communities (breast cancer subtypes)
    successful_df['community_bonus'] = 0
    successful_df.loc[(successful_df['n_communities'] >= 4) & (successful_df['n_communities'] <= 5), 'community_bonus'] = 0.1
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
    Create paper-ready visualizations for grid search results.
    """
    output_dir = Path(output_dir)
    
    # Pivot tables for heatmaps
    ari_pivot = df.pivot_table(values='ari', index='variance_threshold', 
                               columns='similarity_threshold', aggfunc='mean')
    nmi_pivot = df.pivot_table(values='nmi', index='variance_threshold', 
                               columns='similarity_threshold', aggfunc='mean')
    purity_pivot = df.pivot_table(values='purity', index='variance_threshold', 
                                 columns='similarity_threshold', aggfunc='mean')
    communities_pivot = df.pivot_table(values='n_communities', index='variance_threshold', 
                                       columns='similarity_threshold', aggfunc='mean')
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Heatmap 1: ARI
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(ari_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'ARI'})
    ax1.set_title(f'{dataset_name}: Adjusted Rand Index (ARI)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Similarity Threshold')
    ax1.set_ylabel('Variance Threshold')
    # Mark best
    best_var = best_config['variance_threshold']
    best_sim = best_config['similarity_threshold']
    ax1.scatter([list(ari_pivot.columns).index(best_sim)], 
                [list(ari_pivot.index).index(best_var)], 
                color='red', s=200, marker='*', zorder=5, label='Best')
    ax1.legend()
    
    # Heatmap 2: NMI
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(nmi_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'NMI'})
    ax2.set_title(f'{dataset_name}: Normalized Mutual Information (NMI)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Similarity Threshold')
    ax2.set_ylabel('Variance Threshold')
    ax2.scatter([list(nmi_pivot.columns).index(best_sim)], 
                [list(nmi_pivot.index).index(best_var)], 
                color='red', s=200, marker='*', zorder=5)
    
    # Heatmap 3: Purity
    ax3 = fig.add_subplot(gs[0, 2])
    sns.heatmap(purity_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Purity'})
    ax3.set_title(f'{dataset_name}: Purity', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Similarity Threshold')
    ax3.set_ylabel('Variance Threshold')
    ax3.scatter([list(purity_pivot.columns).index(best_sim)], 
                [list(purity_pivot.index).index(best_var)], 
                color='red', s=200, marker='*', zorder=5)
    
    # Heatmap 4: Number of Communities
    ax4 = fig.add_subplot(gs[1, 0])
    sns.heatmap(communities_pivot, annot=True, fmt='.0f', cmap='viridis', ax=ax4, cbar_kws={'label': 'Communities'})
    ax4.set_title(f'{dataset_name}: Number of Communities Found', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Similarity Threshold')
    ax4.set_ylabel('Variance Threshold')
    ax4.scatter([list(communities_pivot.columns).index(best_sim)], 
                [list(communities_pivot.index).index(best_var)], 
                color='red', s=200, marker='*', zorder=5)
    
    # Heatmap 5: Features Kept
    features_pivot = df.pivot_table(values='n_features', index='variance_threshold', 
                                   columns='similarity_threshold', aggfunc='mean')
    ax5 = fig.add_subplot(gs[1, 1])
    sns.heatmap(features_pivot, annot=True, fmt='.0f', cmap='Blues', ax=ax5, cbar_kws={'label': 'Features'})
    ax5.set_title(f'{dataset_name}: Features Kept', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Similarity Threshold')
    ax5.set_ylabel('Variance Threshold')
    
    # Heatmap 6: Graph Density
    density_pivot = df.pivot_table(values='graph_density', index='variance_threshold', 
                                   columns='similarity_threshold', aggfunc='mean')
    ax6 = fig.add_subplot(gs[1, 2])
    sns.heatmap(density_pivot, annot=True, fmt='.2f', cmap='Greens', ax=ax6, cbar_kws={'label': 'Density (%)'})
    ax6.set_title(f'{dataset_name}: Graph Density (%)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Similarity Threshold')
    ax6.set_ylabel('Variance Threshold')
    
    # Line plots showing effect of each parameter
    # Effect of variance threshold (averaged over similarity)
    ax7 = fig.add_subplot(gs[2, 0])
    var_effect = df.groupby('variance_threshold').agg({
        'ari': 'mean',
        'nmi': 'mean',
        'purity': 'mean'
    })
    ax7.plot(var_effect.index, var_effect['ari'], 'o-', label='ARI', linewidth=2, markersize=8)
    ax7.plot(var_effect.index, var_effect['nmi'], 's-', label='NMI', linewidth=2, markersize=8)
    ax7.plot(var_effect.index, var_effect['purity'], '^-', label='Purity', linewidth=2, markersize=8)
    ax7.axvline(x=best_var, color='r', linestyle='--', linewidth=2, label='Best')
    ax7.set_xlabel('Variance Threshold')
    ax7.set_ylabel('Metric Score')
    ax7.set_title('Effect of Variance Threshold (avg over similarity)', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Effect of similarity threshold (averaged over variance)
    ax8 = fig.add_subplot(gs[2, 1])
    sim_effect = df.groupby('similarity_threshold').agg({
        'ari': 'mean',
        'nmi': 'mean',
        'purity': 'mean'
    })
    ax8.plot(sim_effect.index, sim_effect['ari'], 'o-', label='ARI', linewidth=2, markersize=8)
    ax8.plot(sim_effect.index, sim_effect['nmi'], 's-', label='NMI', linewidth=2, markersize=8)
    ax8.plot(sim_effect.index, sim_effect['purity'], '^-', label='Purity', linewidth=2, markersize=8)
    ax8.axvline(x=best_sim, color='r', linestyle='--', linewidth=2, label='Best')
    ax8.set_xlabel('Similarity Threshold')
    ax8.set_ylabel('Metric Score')
    ax8.set_title('Effect of Similarity Threshold (avg over variance)', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Communities found vs parameters
    ax9 = fig.add_subplot(gs[2, 2])
    communities_var = df.groupby('variance_threshold')['n_communities'].mean()
    communities_sim = df.groupby('similarity_threshold')['n_communities'].mean()
    ax9_twin = ax9.twinx()
    line1 = ax9.plot(communities_var.index, communities_var.values, 'o-', 
                     color='blue', linewidth=2, markersize=8, label='By Variance')
    line2 = ax9_twin.plot(communities_sim.index, communities_sim.values, 's-', 
                         color='orange', linewidth=2, markersize=8, label='By Similarity')
    ax9.set_xlabel('Parameter Value')
    ax9.set_ylabel('Communities (by Variance)', color='blue')
    ax9_twin.set_ylabel('Communities (by Similarity)', color='orange')
    ax9.set_title('Number of Communities vs Parameters', fontsize=12, fontweight='bold')
    ax9.tick_params(axis='y', labelcolor='blue')
    ax9_twin.tick_params(axis='y', labelcolor='orange')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax9.legend(lines, labels, loc='upper left')
    ax9.grid(True, alpha=0.3)
    
    # Best configuration summary table
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis('off')
    summary_text = f"""
BEST CONFIGURATION FOR {dataset_name.upper()}
{'='*80}
Variance Threshold: {best_config['variance_threshold']}
Similarity Threshold: {best_config['similarity_threshold']}

Performance Metrics:
  - Adjusted Rand Index (ARI):   {best_config['ari']:.4f}
  - Normalized Mutual Info (NMI): {best_config['nmi']:.4f}
  - Purity:                        {best_config['purity']:.4f}
  - F1-score (macro):              {best_config['f1_macro']:.4f}

Clustering Results:
  - Communities Found: {int(best_config['n_communities'])}
  - Optimal K (BIC):   {int(best_config['optimal_k'])}
  - Features Kept:     {int(best_config['n_features']):,}
  - Graph Density:     {best_config['graph_density']:.2f}%
  - Graph Edges:       {int(best_config['n_edges']):,}
"""
    ax10.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
              verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Parameter Grid Search Results: {dataset_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_file = output_dir / f'{dataset_name}_parameter_grid_search.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[Saved] Grid search visualization: {output_file}")
    plt.close()
    
    # Clear pivot tables before creating individual heatmaps
    del ari_pivot, nmi_pivot, purity_pivot, communities_pivot, features_pivot, density_pivot
    gc.collect()
    
    # Create individual metric heatmaps (for paper)
    create_individual_heatmaps(df, dataset_name, output_dir, best_config)


def create_individual_heatmaps(df, dataset_name, output_dir, best_config):
    """Create individual high-quality heatmaps for each metric (paper-ready)."""
    
    metrics = ['ari', 'nmi', 'purity', 'f1_macro']
    metric_names = ['Adjusted Rand Index (ARI)', 'Normalized Mutual Information (NMI)', 
                    'Purity', 'F1-Score (Macro)']
    
    for metric, metric_name in zip(metrics, metric_names):
        pivot = df.pivot_table(values=metric, index='variance_threshold', 
                              columns='similarity_threshold', aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, 
                   cbar_kws={'label': metric_name}, linewidths=0.5, linecolor='gray')
        ax.set_title(f'{dataset_name}: {metric_name}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Similarity Threshold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Variance Threshold', fontsize=12, fontweight='bold')
        
        # Mark best configuration
        best_var = best_config['variance_threshold']
        best_sim = best_config['similarity_threshold']
        ax.scatter([list(pivot.columns).index(best_sim)], 
                  [list(pivot.index).index(best_var)], 
                  color='red', s=300, marker='*', zorder=5, 
                  edgecolors='black', linewidths=2, label='Best Configuration')
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        output_file = output_dir / f'{dataset_name}_{metric}_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Saved] {metric_name} heatmap: {output_file.name}")
        
        # Clean up pivot table after each heatmap
        del pivot
        gc.collect()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Parameter grid search for optimal thresholds')
    parser.add_argument('--dataset', type=str, required=True, choices=['tcga', 'gse96058'],
                       help='Dataset to test')
    parser.add_argument('--config', type=str, default='config/config.yml',
                       help='Config file path')
    parser.add_argument('--variance_range', type=float, nargs='+', default=None,
                       help='Variance thresholds to test (default: auto-range)')
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
        # Read from config if not provided via command line
        if args.variance_range is None:
            tcga_grid = grid_search_config.get('tcga', {})
            variance_range = list(generate_precise_range(
                tcga_grid.get('variance_start', 0.5),
                tcga_grid.get('variance_end', 15.0),
                tcga_grid.get('variance_step', 0.5)
            ))
        else:
            variance_range = args.variance_range
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
        # Read from config if not provided via command line
        if args.variance_range is None:
            gse_grid = grid_search_config.get('gse96058', {})
            variance_range = list(generate_precise_range(
                gse_grid.get('variance_start', 0.5),
                gse_grid.get('variance_end', 15.0),
                gse_grid.get('variance_step', 0.5)
            ))
        else:
            variance_range = args.variance_range
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
        variance_range,
        similarity_range,
        args.config,
        args.output_dir
    )
    
    if results:
        print(f"\n✅ Grid search complete! Results saved to: {results['output_dir']}")

