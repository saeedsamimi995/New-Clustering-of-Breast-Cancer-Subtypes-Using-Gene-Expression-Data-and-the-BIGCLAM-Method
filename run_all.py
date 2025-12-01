"""
Main Orchestration Script

Runs the complete BIGCLAM pipeline:
1. Data preprocessing
2. Graph construction
3. Clustering
4. Evaluation
5. Visualization
6. Interpretation
7. Cross-dataset analysis
8. Classification validation

NOTE: This script expects prepared CSV files (*_target_added.csv) to already exist.
      Data preparation is a separate step (use data_preparing.py if needed).
"""

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import pickle

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.preprocessing import preprocess_data
from src.graph import construct_graphs
from src.clustering import cluster_all_graphs
from src.evaluation import evaluate_all_datasets, survival_evaluation
from src.evaluation.survival_evaluator import load_tcga_clinical, load_gse96058_clinical
from src.visualization import create_all_visualizations
from src.interpretation import interpret_results, analyze_overlap, biological_interpretation_pipeline
from src.analysis import analyze_cross_dataset_consistency, run_grid_search
from src.analysis.comprehensive_method_comparison import compare_all_methods
from src.classifiers import validate_clustering_with_classifiers
from src.preprocessing.sample_matching_qc import create_sample_matching_qc
from src.analysis.cluster_stability import run_cluster_stability_analysis
from src.analysis.parameter_grid_search import generate_precise_range

def load_config(config_path='config/config.yml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_optimal_similarity_from_grid_search(dataset_name_key, short_name, config, config_path, output_dir='results/grid_search'):
    """
    Run or reuse parameter grid search to obtain the best similarity threshold
    for a given dataset, based on ARI/NMI/purity/F1 and community-count heuristics.
    
    Args:
        dataset_name_key: Long dataset name used in filenames, e.g. 'tcga_brca_data'
        short_name: Short dataset key used in config, e.g. 'tcga' or 'gse96058'
        config: Loaded YAML config dict
        config_path: Path to config file (string)
        output_dir: Directory where grid search results are stored
    
    Returns:
        float or None: Selected similarity threshold, or None if grid search fails
    """
    dataset_prep_cfg = config.get('dataset_preparation', {})
    grid_cfg = config.get('grid_search', {})
    
    ds_prep = dataset_prep_cfg.get(short_name, {})
    input_file = ds_prep.get('output')
    if not input_file or not Path(input_file).exists():
        print(f"[WARNING] Grid search skipped for {dataset_name_key}: input file not found ({input_file})")
        return None
    
    ds_grid = grid_cfg.get(short_name, {})
    sim_range = list(generate_precise_range(
        ds_grid.get('similarity_start', 0.1),
        ds_grid.get('similarity_end', 0.9),
        ds_grid.get('similarity_step', 0.05)
    ))
    
    print(f"\n[Grid Search] Auto-running for {dataset_name_key} to select similarity threshold...")
    results = run_grid_search(
        dataset_name_key,
        input_file,
        sim_range,
        config_path,
        output_dir
    )
    
    if results and isinstance(results, dict):
        best_cfg = results.get('best_config', {})
        if 'similarity_threshold' in best_cfg:
            best_sim = float(best_cfg['similarity_threshold'])
            print(f"[Grid Search] Selected similarity for {dataset_name_key}: {best_sim}")
            return best_sim
    
    print(f"[WARNING] Grid search did not return a best similarity for {dataset_name_key}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Complete BIGCLAM pipeline for breast cancer subtype clustering'
    )
    parser.add_argument('--config', type=str, default='config/config.yml', help='Config file')
    parser.add_argument('--skip_clustering', action='store_true', help='Skip clustering (use existing results)')
    parser.add_argument('--skip_classification', action='store_true', help='Skip classification validation')
    parser.add_argument('--steps', nargs='+', choices=[
        'preprocess', 'graph', 'cluster', 'evaluate', 
        'visualize', 'interpret', 'biological_interpretation', 'cross_dataset', 'classify', 'grid_search', 'survival', 'method_comparison',
        'cluster_pam50_mapping', 'results_synthesis', 'sample_matching_qc', 'cluster_stability'
    ], help='Run specific steps only')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("\n" + "="*80)
    print("BIGCLAM BREAST CANCER SUBTYPE CLUSTERING - COMPLETE PIPELINE")
    print("="*80)
    
    # Determine steps to run
    if args.steps:
        steps_to_run = args.steps
    else:
        steps_to_run = ['preprocess', 'graph', 'cluster', 'evaluate', 
                       'visualize', 'interpret', 'biological_interpretation', 'cross_dataset', 'classify', 'survival', 'method_comparison',
                       'cluster_pam50_mapping', 'results_synthesis', 'sample_matching_qc', 'cluster_stability']
    
    # Special case: grid_search runs its own pipeline (starts from data_preprocessing.py)
    # NOTE: Grid search ONLY uses prepared CSV files (*_target_added.csv) as input.
    #       It expects prepared CSV files to already exist (data preparation is a separate step).
    #       Grid search starts with preprocessing: it takes the prepared CSV and runs preprocess_data()
    if 'grid_search' in steps_to_run:
        print("\n" + "="*80)
        print("RUNNING PARAMETER GRID SEARCH")
        print("="*80)
        print("\nNOTE: Grid search uses ONLY prepared CSV files (*_target_added.csv)")
        print("      It does NOT access raw clinical/expression files.")
        print("      Pipeline: *_target_added.csv -> preprocess_data() -> graph -> cluster -> evaluate")
        print("      Data preparation (data_preparing.py) is NOT run here.\n")
        
        dataset_config = config.get('dataset_preparation', {})
        grid_search_config = config.get('grid_search', {})
        
        # TCGA BRCA Grid Search
        # Use ONLY the prepared CSV file (output from data_preparing.py)
        # This is the file with targets already added: tcga_brca_data_target_added.csv
        tcga_config = dataset_config.get('tcga', {})
        tcga_input_file = tcga_config.get('output') if tcga_config else None  # This is the *_target_added.csv file
        
        if tcga_input_file and Path(tcga_input_file).exists():
            print("\n[Grid Search] TCGA-BRCA")
            print(f"    Input file (prepared CSV with targets): {tcga_input_file}")
            print(f"    This file will be processed by data_preprocessing.py")
            
            # Generate similarity sweep from config (variance is fixed to dataset mean)
            tcga_grid = grid_search_config.get('tcga', {})
            from src.analysis.parameter_grid_search import generate_precise_range
            tcga_sim_range = list(generate_precise_range(
                tcga_grid.get('similarity_start', 0.1),
                tcga_grid.get('similarity_end', 0.9),
                tcga_grid.get('similarity_step', 0.05)
            ))
            
            run_grid_search(
                'tcga_brca_data',
                tcga_input_file,  # This is the prepared CSV, not raw clinical/expression files
                tcga_sim_range,
                args.config,
                'results/grid_search'
            )
            
            # Clear memory before loading GSE data (important for limited RAM)
            print("\n[Memory] Clearing TCGA data from RAM before loading GSE data...")
            from src.analysis.parameter_grid_search import clear_memory
            clear_memory()
            print("[Memory] Cleanup complete. Ready for GSE96058.\n")
        else:
            print(f"\n[SKIP] TCGA-BRCA grid search - prepared CSV file not found:")
            print(f"       Expected: {tcga_input_file}")
            print(f"       This should be a prepared CSV file (with _target_added suffix)")
            print(f"       Run data preparation separately if needed: python -m src.preprocessing.data_preparing")
        
        # GSE96058 Grid Search
        # Use ONLY the prepared CSV file (output from data_preparing.py)
        # This is the file with targets already added: gse96058_data_target_added.csv
        gse_config = dataset_config.get('gse96058', {})
        gse_input_file = gse_config.get('output') if gse_config else None  # This is the *_target_added.csv file
        
        if gse_input_file and Path(gse_input_file).exists():
            print("\n[Grid Search] GSE96058")
            print(f"    Input file (prepared CSV with targets): {gse_input_file}")
            print(f"    This file will be processed by data_preprocessing.py")
            
            # Generate similarity sweep from config (variance is fixed to dataset mean)
            gse_grid = grid_search_config.get('gse96058', {})
            from src.analysis.parameter_grid_search import generate_precise_range
            gse_sim_range = list(generate_precise_range(
                gse_grid.get('similarity_start', 0.1),
                gse_grid.get('similarity_end', 0.9),
                gse_grid.get('similarity_step', 0.05)
            ))
            
            run_grid_search(
                'gse96058_data',
                gse_input_file,  # This is the prepared CSV, not raw clinical/expression files
                gse_sim_range,
                args.config,
                'results/grid_search'
            )
        else:
            print(f"\n[SKIP] GSE96058 grid search - prepared CSV file not found:")
            print(f"       Expected: {gse_input_file}")
            print(f"       This should be a prepared CSV file (with _target_added suffix)")
            print(f"       Run data preparation separately if needed: python -m src.preprocessing.data_preparing")
        
        print("\n" + "="*80)
        print("GRID SEARCH COMPLETE!")
        print("="*80)
        print("\nResults saved to: results/grid_search/")
        print("Check PNG files for paper-ready visualizations.")
        print("="*80)
        return  # Exit after grid search
    
    # Step 1: Data Preprocessing
    if 'preprocess' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 1: DATA PREPROCESSING")
        print("="*80)
        print("\nNOTE: This step expects prepared CSV files (*_target_added.csv) to exist.")
        print("      If files are missing, run data preparation separately.\n")
        
        preprocessing_config = config.get('preprocessing', {})
        print("\nVariance filtering note: threshold is automatically set to the")
        print("dataset-wide mean; config overrides are ignored after the latest update.")
        
        # Process TCGA BRCA
        tcga_output = config['dataset_preparation']['tcga']['output']
        if Path(tcga_output).exists():
            preprocess_data(tcga_output, output_dir='data/processed')
        
        # Process GSE96058
        gse_output = config['dataset_preparation']['gse96058']['output']
        if Path(gse_output).exists():
            preprocess_data(gse_output, output_dir='data/processed')
    
    # Step 2: Graph Construction (now driven by parameter_grid_search)
    if 'graph' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 2: GRAPH CONSTRUCTION")
        print("="*80)
        
        # Reload config to ensure we pick up any recent edits on disk
        graph_config = load_config(args.config)
        preprocessing_config = graph_config.get('preprocessing', {})
        
        # Prefer automatic similarity selection via parameter_grid_search
        auto_similarity = graph_config.get('grid_search', {}).get('use_in_run_all', True)
        
        thresholds_dict = {}
        if auto_similarity:
            # Use grid search to pick best similarity per dataset (if possible)
            tcga_sim = get_optimal_similarity_from_grid_search(
                dataset_name_key='tcga_brca_data',
                short_name='tcga',
                config=graph_config,
                config_path=args.config,
                output_dir='results/grid_search'
            )
            if tcga_sim is not None:
                thresholds_dict['tcga_brca_data'] = tcga_sim
            
            gse_sim = get_optimal_similarity_from_grid_search(
                dataset_name_key='gse96058_data',
                short_name='gse96058',
                config=graph_config,
                config_path=args.config,
                output_dir='results/grid_search'
            )
            if gse_sim is not None:
                thresholds_dict['gse96058_data'] = gse_sim
        
        # If auto mode is disabled or failed, fall back to config thresholds
        if not thresholds_dict:
            similarity_thresholds = preprocessing_config.get('similarity_thresholds', {})
            if similarity_thresholds:
                # Work with a copy and cast to float (handles YAML strings/decimals)
                similarity_thresholds = {
                    dataset: float(value)
                    for dataset, value in similarity_thresholds.items()
                }
                thresholds_dict = {
                    k: v for k, v in similarity_thresholds.items() if k != 'default'
                }
        
        if thresholds_dict:
            config_path = Path(args.config).resolve()
            print(f"Using dataset-specific similarity thresholds (from grid search / config, {config_path}):")
            for dataset, thresh in thresholds_dict.items():
                print(f"  {dataset}: {thresh}")
            construct_graphs(
                input_dir='data/processed',
                output_dir='data/graphs',
                thresholds_dict=thresholds_dict,
                use_sparse=True
            )
        else:
            # Fallback to single threshold (for backwards compatibility)
            similarity_threshold = preprocessing_config.get('similarity_threshold', 0.4)
            print(f"Using single similarity threshold: {similarity_threshold}")
            construct_graphs(
                input_dir='data/processed',
                output_dir='data/graphs',
                threshold=similarity_threshold,
                use_sparse=True
            )
    
    # Step 3: Clustering
    if 'cluster' in steps_to_run and not args.skip_clustering:
        print("\n" + "="*80)
        print("STEP 3: CLUSTERING WITH BIGCLAM")
        print("="*80)
        
        bigclam_config = config.get('bigclam', {})
        max_communities = bigclam_config.get('max_communities', 10)
        min_communities = bigclam_config.get('min_communities', 3)  # Default to 3 for major molecular programs
        iterations = bigclam_config.get('iterations', 100)
        lr = bigclam_config.get('learning_rate', 0.08)
        
        # Get optimization parameters
        adaptive_lr = bigclam_config.get('adaptive_lr', True)
        adaptive_iterations = bigclam_config.get('adaptive_iterations', True)
        early_stopping = bigclam_config.get('early_stopping', True)
        convergence_threshold = float(bigclam_config.get('convergence_threshold', 1e-6))
        patience = int(bigclam_config.get('patience', 10))
        default_num_restarts = int(bigclam_config.get('num_restarts', 1))
        
        # Get dataset-specific model selection criteria (AIC/BIC)
        criterion_config = bigclam_config.get('model_selection_criterion', {})
        criterion_dict = {}
        if criterion_config:
            print("\nModel selection criteria:")
            for dataset, criterion in criterion_config.items():
                if dataset != 'default':
                    criterion_dict[dataset] = criterion
                    print(f"  {dataset}: {criterion}")
            if 'default' in criterion_config:
                criterion_dict['default'] = criterion_config['default']
                print(f"  default: {criterion_config['default']}")
        
        # Get dataset-specific overrides and num_restarts
        dataset_specific_config = bigclam_config.get('dataset_specific', {})
        num_restarts_dict = {}
        if dataset_specific_config:
            print("\nDataset-specific configurations:")
            for dataset, ds_config in dataset_specific_config.items():
                if 'num_restarts' in ds_config:
                    num_restarts_dict[dataset] = int(ds_config['num_restarts'])
                    print(f"  {dataset}: num_restarts={ds_config['num_restarts']}")
                if 'min_communities' in ds_config:
                    print(f"  {dataset}: min_communities={ds_config['min_communities']}")
        
        print("\nOptimization settings:")
        print(f"  Min communities: {min_communities} (for finer-grained subtyping than PAM50)")
        print(f"  Max communities: {max_communities}")
        print(f"  Adaptive LR: {adaptive_lr}")
        print(f"  Adaptive iterations: {adaptive_iterations}")
        print(f"  Early stopping: {early_stopping} (patience={patience}, threshold={convergence_threshold})")
        print(f"  Default num_restarts: {default_num_restarts}")
        
        cluster_all_graphs(input_dir='data/graphs', output_dir='data/clusterings',
                         max_communities=max_communities, min_communities=min_communities,
                         iterations=iterations, lr=lr,
                         criterion_dict=criterion_dict if criterion_dict else None,
                         adaptive_lr=adaptive_lr, adaptive_iterations=adaptive_iterations,
                         early_stopping=early_stopping, convergence_threshold=convergence_threshold,
                         patience=patience, num_restarts_dict=num_restarts_dict if num_restarts_dict else None,
                         dataset_specific_config=dataset_specific_config if dataset_specific_config else None)
    
    # Step 4: Classification Validation (using clustering results as targets)
    if 'classify' in steps_to_run and not args.skip_classification:
        print("\n" + "="*80)
        print("STEP 4: CLASSIFICATION VALIDATION")
        print("="*80)
        print("Training SVM and MLP to predict BIGCLAM communities from expression data")
        print("Using dataset-specific parameters for fine-tuning")
        
        classify_config = config.get('classifiers', {})
        
        # Get default parameters
        default_config = classify_config.get('default', {})
        default_mlp_params = default_config.get('mlp', {})
        default_svm_params = default_config.get('svm', {})
        
        # Get dataset-specific parameters
        dataset_specific_config = classify_config.get('dataset_specific', {})
        dataset_specific_params = {}
        
        for dataset_name, ds_config in dataset_specific_config.items():
            dataset_specific_params[dataset_name] = {
                'mlp': ds_config.get('mlp', default_mlp_params),
                'svm': ds_config.get('svm', default_svm_params)
            }
        
        if dataset_specific_params:
            print("\nDataset-specific classifier configurations:")
            for dataset, params in dataset_specific_params.items():
                print(f"  {dataset}:")
                print(f"    MLP: hidden_layers={params['mlp'].get('hidden_layers', 'N/A')}, "
                      f"lr={params['mlp'].get('learning_rate', 'N/A')}, "
                      f"epochs={params['mlp'].get('num_epochs', 'N/A')}")
                print(f"    SVM: C={params['svm'].get('C', 'N/A')}, "
                      f"gamma={params['svm'].get('gamma', 'N/A')}, "
                      f"kernel={params['svm'].get('kernel', 'N/A')}")
        
        validate_clustering_with_classifiers(processed_dir='data/processed',
                                           clustering_dir='data/clusterings',
                                           output_dir='results/classification',
                                           mlp_params=default_mlp_params,
                                           svm_params=default_svm_params,
                                           dataset_specific_params=dataset_specific_params)
    
    # Step 5: Evaluation
    if 'evaluate' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 5: EVALUATION")
        print("="*80)
        
        evaluate_all_datasets(clustering_dir='data/clusterings',
                            targets_dir='data/processed',
                            output_dir='results/evaluation')
    
    # Step 5.1: Survival Analysis
    if 'survival' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 5.1: SURVIVAL ANALYSIS")
        print("="*80)

        clustering_dir = Path('data/clusterings')
        processed_dir = Path('data/processed')
        survival_output_dir = Path('results/survival')
        survival_output_dir.mkdir(parents=True, exist_ok=True)

        # Helper function to normalize TCGA sample IDs
        def normalize_tcga_id(sid):
            """Normalize TCGA sample ID by removing suffixes like -01, -02, etc."""
            sid = str(sid).strip().upper()
            sid = sid.replace('-', '').replace('.', '')
            if len(sid) > 10 and sid[-2:].isdigit():
                sid = sid[:-2]
            return sid
        
        # Helper function to load and prepare clinical data using survival_evaluator functions
        def load_clinical_data(clinical_file_path, sample_names, dataset_type='tcga'):
            """
            Load clinical data from file using survival_evaluator functions.
            
            Args:
                clinical_file_path: Path to clinical data file
                sample_names: List of sample names to match
                dataset_type: 'tcga' or 'gse96058'
            
            Returns DataFrame with columns: sample_id, OS_time, OS_event, and optionally age, stage
            """
            if not Path(clinical_file_path).exists():
                return None
            
            try:
                # Use the appropriate loader based on dataset type
                if dataset_type == 'gse96058':
                    clinical_df = load_gse96058_clinical(clinical_file_path)
                elif dataset_type == 'tcga':
                    clinical_df = load_tcga_clinical(clinical_file_path)
                    # For TCGA, we need to match sample names using normalized IDs
                    if 'sample_id' in clinical_df.columns:
                        # Normalize sample IDs for matching
                        clinical_df['normalized_id'] = clinical_df['sample_id'].astype(str).apply(normalize_tcga_id)
                        
                        # Create normalized IDs from sample_names
                        sample_df = pd.DataFrame({
                            'sample_id': sample_names,
                            'normalized_id': [normalize_tcga_id(sid) for sid in sample_names]
                        })
                        
                        # Merge to match samples
                        clinical_df = sample_df.merge(
                            clinical_df,
                            on='normalized_id',
                            how='left',
                            suffixes=('', '_clinical')
                        )
                        
                        # Use original sample_id, drop duplicate
                        if 'sample_id_clinical' in clinical_df.columns:
                            clinical_df = clinical_df.drop(columns=['sample_id_clinical'])
                        clinical_df = clinical_df.drop(columns=['normalized_id'], errors='ignore')
                else:
                    print(f"    [Error] Unknown dataset type: {dataset_type}")
                    return None
                
                # Check required columns
                if 'OS_time' not in clinical_df.columns or 'OS_event' not in clinical_df.columns:
                    print(f"    [Warning] Missing survival columns in loaded data")
                    print(f"      Available columns: {list(clinical_df.columns)}")
                    return None
                
                print(f"    Survival data prepared: {len(clinical_df)} samples")
                print(f"    Samples with OS_time: {clinical_df['OS_time'].notna().sum()}")
                print(f"    Samples with OS_event: {clinical_df['OS_event'].notna().sum()}")
                
                return clinical_df
                
            except Exception as e:
                print(f"    [Error] Failed to load clinical data: {e}")
                import traceback
                traceback.print_exc()
                return None

        # Loop over datasets - only process those with clinical data available
        # Check which datasets have clinical files available
        dataset_config = config.get('dataset_preparation', {})
        datasets_to_process = []
        
        # Check TCGA
        tcga_clinical = dataset_config.get('tcga', {}).get('clinical')
        if tcga_clinical and Path(tcga_clinical).exists():
            datasets_to_process.append('tcga_brca_data')
        else:
            print(f"[INFO] TCGA clinical file not found, skipping TCGA survival analysis")
            print(f"       Expected: {tcga_clinical}")
        
        # Check GSE96058 (optional - skip if not available)
        gse_clinical = dataset_config.get('gse96058', {}).get('clinical')
        if gse_clinical and Path(gse_clinical).exists():
            datasets_to_process.append('gse96058_data')
        else:
            print(f"[INFO] GSE96058 clinical file not found, skipping GSE96058 survival analysis")
            if gse_clinical:
                print(f"       Expected: {gse_clinical}")
            else:
                print(f"       No clinical file path configured for GSE96058")
        
        if not datasets_to_process:
            print("\n[WARNING] No datasets with clinical data available for survival analysis")
            print("          Survival analysis requires clinical files with OS_time and OS_event columns")
            print("          Skipping survival analysis step")
            return
        
        print(f"\n[INFO] Processing survival analysis for: {datasets_to_process}")
        
        for dataset_name in datasets_to_process:
            cluster_file = clustering_dir / f"{dataset_name}_communities.npy"
            target_file = processed_dir / f"{dataset_name}_targets.pkl"

            if not cluster_file.exists() or not target_file.exists():
                print(f"[SKIP] Survival analysis for {dataset_name}: Missing clustering or targets")
                continue

            print(f"\n[Survival] Processing dataset: {dataset_name}")

            # Load clusters
            communities = np.load(cluster_file)
            if communities.ndim == 2:
                communities = np.argmax(communities, axis=1)
            communities = communities.flatten()

            # Load targets to get sample names
            with open(target_file, 'rb') as f:
                targets_data = pickle.load(f)
            
            # Get sample names from targets
            sample_names = targets_data.get('sample_names', None)
            if sample_names is None:
                print(f"[Warning] No sample_names in {target_file}, using indices")
                sample_names = [f"sample_{i}" for i in range(len(communities))]
            else:
                # Ensure sample_names is a list/array
                if isinstance(sample_names, np.ndarray):
                    sample_names = sample_names.tolist()
                if len(sample_names) != len(communities):
                    print(f"[Warning] Mismatch: {len(sample_names)} sample names vs {len(communities)} clusters")
                    print(f"  Using indices instead")
                    sample_names = [f"sample_{i}" for i in range(len(communities))]

            cluster_assignments = pd.DataFrame({
                'sample_id': sample_names,
                'cluster': communities
            })

            # Try to load clinical data from targets first
            clinical_df = targets_data.get('clinical', None)
            
            # If not in targets, load from original clinical file
            if clinical_df is None:
                print(f"    Clinical data not in targets.pkl, loading from original file...")
                dataset_config = config.get('dataset_preparation', {})
                
                if dataset_name == 'tcga_brca_data':
                    clinical_file = dataset_config.get('tcga', {}).get('clinical')
                    dataset_type = 'tcga'
                elif dataset_name == 'gse96058_data':
                    clinical_file = dataset_config.get('gse96058', {}).get('clinical')
                    dataset_type = 'gse96058'
                else:
                    clinical_file = None
                    dataset_type = None
                
                if clinical_file and Path(clinical_file).exists():
                    clinical_df = load_clinical_data(clinical_file, sample_names, dataset_type=dataset_type)
                else:
                    if clinical_file:
                        print(f"    [SKIP] Clinical file not found: {clinical_file}")
                    else:
                        print(f"    [SKIP] No clinical file path in config for {dataset_name}")
                    print(f"    Survival analysis requires clinical data with OS_time and OS_event columns")
                    continue

            if clinical_df is None:
                print(f"[SKIP] Could not load clinical data for {dataset_name}")
                continue

            # Ensure clinical_df is a DataFrame
            if not isinstance(clinical_df, pd.DataFrame):
                print(f"[SKIP] Clinical data is not a DataFrame")
                continue

            # Check if sample_id column exists in clinical data
            if 'sample_id' not in clinical_df.columns:
                print(f"[SKIP] Clinical data missing 'sample_id' column")
                print(f"       Available columns: {list(clinical_df.columns)}")
                continue

            # Run survival pipeline
            try:
                df, log_df, cph = survival_evaluation(
                    cluster_assignments=cluster_assignments,
                    clinical_df=clinical_df,
                    output_dir=str(survival_output_dir),
                    id_col='sample_id',
                    cluster_col='cluster',
                    adjust_cols=['age', 'stage'],
                    dataset_name=dataset_name
                )
                print(f"[OK] Survival analysis complete for {dataset_name}")
            except Exception as e:
                print(f"[ERROR] Survival analysis failed for {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Step 6: Visualization
    if 'visualize' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 6: VISUALIZATION")
        print("="*80)
        
        create_all_visualizations(processed_dir='data/processed',
                                clustering_dir='data/clusterings',
                                output_dir='results/visualization')
    
    # Step 7: Interpretation
    if 'interpret' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 7: INTERPRETATION")
        print("="*80)
        
        clustering_dir = Path('data/clusterings')
        processed_dir = Path('data/processed')
        interpretation_output_dir = Path('results/interpretation')
        interpretation_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all clustering files
        clustering_files = list(clustering_dir.glob('*_communities.npy'))
        
        for clustering_file in clustering_files:
            dataset_name = clustering_file.stem.replace('_communities', '')
            
            print("\n" + "-"*80)
            print(f"INTERPRETING: {dataset_name}")
            print("-"*80)
            
            # Load communities
            communities = np.load(clustering_file)
            
            # Fix: If communities is 2D (membership matrix), convert to 1D (assignments)
            if communities.ndim == 2:
                print(f"[INFO] Converting 2D membership matrix to 1D community assignments...")
                communities = np.argmax(communities, axis=1)
            
            communities = communities.flatten()
            
            # Load targets
            target_file = processed_dir / f"{dataset_name}_targets.pkl"
            if not target_file.exists():
                print(f"[SKIP] No target file: {target_file}")
                continue
            
            with open(target_file, 'rb') as f:
                targets_data = pickle.load(f)
            
            target_labels = targets_data['target_labels']
            label_type = targets_data.get('label_type', 'PAM50')
            
            # Get evaluation metrics if available
            evaluation_file = Path('results/evaluation') / f"{dataset_name}_evaluation_results.pkl"
            ari, nmi = None, None
            if evaluation_file.exists():
                try:
                    with open(evaluation_file, 'rb') as f:
                        eval_results = pickle.load(f)
                        ari = eval_results.get('ari', None)
                        nmi = eval_results.get('nmi', None)
                except:
                    pass
            
            # Interpret results
            if ari is not None and nmi is not None:
                interpretation = interpret_results(ari, nmi, dataset_name, label_type)
            else:
                print(f"[INFO] Evaluation metrics not available, skipping metric-based interpretation")
                interpretation = None
            
            # Analyze overlap
            overlap_results = analyze_overlap(communities, target_labels, dataset_name)
            
            # Save interpretation results
            interpretation_data = {
                'dataset_name': dataset_name,
                'label_type': label_type,
                'ari': ari,
                'nmi': nmi,
                'interpretation': interpretation,
                'overlap_analysis': overlap_results,
                'n_communities': len(set(communities)),
                'n_samples': len(communities)
            }
            
            with open(interpretation_output_dir / f"{dataset_name}_interpretation.pkl", 'wb') as f:
                pickle.dump(interpretation_data, f)
            
            print(f"\n[Saved] Interpretation results to: {interpretation_output_dir / f'{dataset_name}_interpretation.pkl'}")
    
    # Step 8: Biological Interpretation
    if 'biological_interpretation' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 8: BIOLOGICAL INTERPRETATION")
        print("="*80)
        
        dataset_config = config.get('dataset_preparation', {})
        
        # Run for each dataset
        for dataset_name in ['tcga', 'gse96058']:
            dataset_info = dataset_config.get(dataset_name, {})
            if not dataset_info:
                print(f"\n[Skip] {dataset_name}: No configuration found")
                continue
            
            print(f"\n[Running] Biological interpretation for {dataset_name}...")
            try:
                # Use slightly more permissive thresholds for GSE96058 to capture
                # trend-level biology in the validation cohort
                if dataset_name == 'gse96058':
                    log2fc_thr = 0.8
                    p_thr = 0.10
                else:
                    log2fc_thr = 1.0
                    p_thr = 0.05

                biological_interpretation_pipeline(
                    dataset_name=dataset_name,
                    processed_dir='data/processed',
                    clustering_dir='data/clusterings',
                    output_dir='results/biological_interpretation',
                    log2fc_threshold=log2fc_thr,
                    pvalue_threshold=p_thr,
                    use_original_data=True  # Use original data to include all genes
                )
            except Exception as e:
                print(f"[Error] Failed to run biological interpretation for {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Step 9: Cross-Dataset Analysis
    if 'cross_dataset' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 8: CROSS-DATASET CONSISTENCY ANALYSIS")
        print("="*80)
        
        analyze_cross_dataset_consistency(processed_dir='data/processed',
                                        clustering_dir='data/clusterings',
                                        output_dir='results/cross_dataset')
    
    # Step 10: Method Comparison
    if 'method_comparison' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 10: COMPREHENSIVE METHOD COMPARISON")
        print("="*80)
        
        dataset_config = config.get('dataset_preparation', {})
        
        # Run for each dataset
        for dataset_name in ['tcga', 'gse96058']:
            dataset_info = dataset_config.get(dataset_name, {})
            if not dataset_info:
                print(f"\n[Skip] {dataset_name}: No configuration found")
                continue
            
            print(f"\n[Running] Method comparison for {dataset_name}...")
            try:
                compare_all_methods(
                    dataset_name=dataset_name,
                    processed_dir='data/processed',
                    clustering_dir='data/clusterings',
                    output_dir='results/method_comparison',
                    use_pca=True,
                    n_components=50
                )
            except Exception as e:
                print(f"[Error] Failed to run method comparison for {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Step 11: Cluster-to-PAM50 Mapping
    if 'cluster_pam50_mapping' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 11: CLUSTER-TO-PAM50 MAPPING")
        print("="*80)
        
        print("\n[Running] Cluster-to-PAM50 mapping analysis...")
        try:
            import subprocess
            import sys
            
            for dataset_name in ['tcga', 'gse96058']:
                print(f"\n[Processing] {dataset_name}...")
                result = subprocess.run(
                    [sys.executable, 'src/analysis/cluster_pam50_mapping.py', 
                     '--dataset', dataset_name],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(result.stdout)
                else:
                    print(f"[Warning] Cluster-PAM50 mapping failed for {dataset_name}: {result.stderr}")
        except Exception as e:
            print(f"[Error] Failed to run cluster-PAM50 mapping: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 13: Sample Matching QC
    if 'sample_matching_qc' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 13: SAMPLE MATCHING QUALITY CONTROL")
        print("="*80)
        
        for dataset_name in ['tcga', 'gse96058']:
            print(f"\n[Running] Sample matching QC for {dataset_name}...")
            try:
                if dataset_name == 'tcga':
                    expr_file = 'data/tcga_brca_data_target_added.csv'
                    clinical_file = 'data/brca_tcga_pub_clinical_data.tsv'
                else:
                    expr_file = 'data/gse96058_data_target_added.csv'
                    clinical_file = 'data/GSE96058_clinical_data.csv'
                
                create_sample_matching_qc(
                    dataset_name=dataset_name,
                    expression_file=expr_file,
                    clinical_file=clinical_file,
                    output_dir='results/qc'
                )
            except Exception as e:
                print(f"[Error] Failed to run sample matching QC for {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Step 14: Cluster Stability Analysis
    if 'cluster_stability' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 14: CLUSTER STABILITY ANALYSIS")
        print("="*80)
        
        for dataset_name in ['tcga', 'gse96058']:
            print(f"\n[Running] Cluster stability analysis for {dataset_name}...")
            try:
                run_cluster_stability_analysis(
                    dataset_name=dataset_name,
                    processed_dir='data/processed',
                    clustering_dir='data/clusterings',
                    output_dir='results/stability',
                    n_bootstrap=100,
                    n_permutations=1000
                )
            except Exception as e:
                print(f"[Error] Failed to run cluster stability analysis for {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Step 15: Results Synthesis
    if 'results_synthesis' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 15: RESULTS SYNTHESIS")
        print("="*80)
        
        print("\n[Running] Synthesizing results from all analyses...")
        try:
            from src.analysis.results_synthesis import create_summary_table, generate_narrative, create_manuscript_results_section
            
            for dataset_name in ['tcga', 'gse96058']:
                print(f"\n[Processing] {dataset_name}...")
                summary_df = create_summary_table(dataset_name, 'results/synthesis')
                if summary_df is not None:
                    generate_narrative(dataset_name, summary_df, 'results/synthesis')
                    create_manuscript_results_section(dataset_name, summary_df, 'results/synthesis')
        except Exception as e:
            print(f"[Error] Failed to synthesize results: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nResults Summary:")
    print("  📊 Evaluation:     results/evaluation/")
    print("  📈 Visualizations: results/visualization/")
    print("  🤖 Classification: results/classification/")
    print("  🔍 Interpretation: results/interpretation/")
    print("  🧬 Biological:     results/biological_interpretation/")
    print("  🔗 Cross-dataset:  results/cross_dataset/")
    print("  💊 Survival:       results/survival/")
    print("  📊 Method Comparison: results/method_comparison/")
    print("  🗺️  Cluster-PAM50 Mapping: results/cluster_pam50_mapping/")
    print("  📋 Results Synthesis: results/synthesis/")
    print("  🔍 Sample Matching QC: results/qc/")
    print("  🔬 Cluster Stability: results/stability/")
    print("\nKey findings:")
    print("  • Check ARI/NMI scores in evaluation output")
    print("  • View t-SNE/UMAP plots to see cluster separation")
    print("  • Review confusion matrices and ROC curves for classification performance")
    print("  • Check interpretation results for biological significance")
    print("  • Review differential expression and pathway enrichment results")
    print("  • Examine cell-type signature scores for each cluster")
    print("  • Examine cross-dataset correlations for consistency")
    print("  • Review survival curves and Cox models for prognostic significance")
    print("  • Compare BIGCLAM with other clustering methods (K-means, Spectral, NMF, HDBSCAN, Leiden/Louvain)")
    print("="*80)


if __name__ == "__main__":
    main()

