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
import numpy as np
import pickle

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.preprocessing import preprocess_data
from src.graph import construct_graphs
from src.clustering import cluster_all_graphs
from src.evaluation import evaluate_all_datasets
from src.visualization import create_all_visualizations
from src.interpretation import interpret_results, analyze_overlap
from src.analysis import analyze_cross_dataset_consistency, run_grid_search
from src.classifiers import validate_clustering_with_classifiers


def load_config(config_path='config/config.yml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Complete BIGCLAM pipeline for breast cancer subtype clustering'
    )
    parser.add_argument('--config', type=str, default='config/config.yml', help='Config file')
    parser.add_argument('--skip_clustering', action='store_true', help='Skip clustering (use existing results)')
    parser.add_argument('--skip_classification', action='store_true', help='Skip classification validation')
    parser.add_argument('--steps', nargs='+', choices=[
        'preprocess', 'graph', 'cluster', 'evaluate', 
        'visualize', 'interpret', 'cross_dataset', 'classify', 'grid_search'
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
                       'visualize', 'interpret', 'cross_dataset', 'classify']
    
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
    
    # Step 2: Graph Construction
    if 'graph' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 2: GRAPH CONSTRUCTION")
        print("="*80)
        
        # Reload config to ensure we pick up any recent edits on disk
        graph_config = load_config(args.config)
        preprocessing_config = graph_config.get('preprocessing', {})
        
        # Use dataset-specific thresholds if available, otherwise fallback to single threshold
        similarity_thresholds = preprocessing_config.get('similarity_thresholds', {})
        if similarity_thresholds:
            # Work with a copy and cast to float (handles YAML strings/decimals)
            similarity_thresholds = {
                dataset: float(value)
                for dataset, value in similarity_thresholds.items()
            }
        if similarity_thresholds:
            config_path = Path(args.config).resolve()
            print(f"Using dataset-specific similarity thresholds (from {config_path}):")
            for dataset, thresh in similarity_thresholds.items():
                if dataset != 'default':
                    print(f"  {dataset}: {thresh}")
            construct_graphs(input_dir='data/processed', output_dir='data/graphs',
                            thresholds_dict=similarity_thresholds, use_sparse=True)
        else:
            # Fallback to single threshold (for backwards compatibility)
            similarity_threshold = preprocessing_config.get('similarity_threshold', 0.4)
            print(f"Using single similarity threshold: {similarity_threshold}")
            construct_graphs(input_dir='data/processed', output_dir='data/graphs',
                            threshold=similarity_threshold, use_sparse=True)
    
    # Step 3: Clustering
    if 'cluster' in steps_to_run and not args.skip_clustering:
        print("\n" + "="*80)
        print("STEP 3: CLUSTERING WITH BIGCLAM")
        print("="*80)
        
        bigclam_config = config.get('bigclam', {})
        max_communities = bigclam_config.get('max_communities', 10)
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
        
        print("\nOptimization settings:")
        print(f"  Adaptive LR: {adaptive_lr}")
        print(f"  Adaptive iterations: {adaptive_iterations}")
        print(f"  Early stopping: {early_stopping} (patience={patience}, threshold={convergence_threshold})")
        print(f"  Default num_restarts: {default_num_restarts}")
        
        cluster_all_graphs(input_dir='data/graphs', output_dir='data/clusterings',
                         max_communities=max_communities, iterations=iterations, lr=lr,
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
    
    # Step 8: Cross-Dataset Analysis
    if 'cross_dataset' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 8: CROSS-DATASET CONSISTENCY ANALYSIS")
        print("="*80)
        
        analyze_cross_dataset_consistency(processed_dir='data/processed',
                                        clustering_dir='data/clusterings',
                                        output_dir='results/cross_dataset')
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nResults Summary:")
    print("  📊 Evaluation:     results/evaluation/")
    print("  📈 Visualizations: results/visualization/")
    print("  🤖 Classification: results/classification/")
    print("  🔍 Interpretation: results/interpretation/")
    print("  🔗 Cross-dataset:  results/cross_dataset/")
    print("\nKey findings:")
    print("  • Check ARI/NMI scores in evaluation output")
    print("  • View t-SNE/UMAP plots to see cluster separation")
    print("  • Review confusion matrices and ROC curves for classification performance")
    print("  • Check interpretation results for biological significance")
    print("  • Examine cross-dataset correlations for consistency")
    print("="*80)


if __name__ == "__main__":
    main()

