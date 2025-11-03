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
            
            # Generate ranges from start/end/step
            tcga_grid = grid_search_config.get('tcga', {})
            # Use precise range generation to avoid floating point errors
            from src.analysis.parameter_grid_search import generate_precise_range
            tcga_var_range = list(generate_precise_range(
                tcga_grid.get('variance_start', 0.5),
                tcga_grid.get('variance_end', 15.0),
                tcga_grid.get('variance_step', 0.5)
            ))
            tcga_sim_range = list(generate_precise_range(
                tcga_grid.get('similarity_start', 0.1),
                tcga_grid.get('similarity_end', 0.9),
                tcga_grid.get('similarity_step', 0.05)
            ))
            
            run_grid_search(
                'tcga_brca_data',
                tcga_input_file,  # This is the prepared CSV, not raw clinical/expression files
                tcga_var_range,
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
            
            # Generate ranges from start/end/step
            gse_grid = grid_search_config.get('gse96058', {})
            # Use precise range generation to avoid floating point errors
            from src.analysis.parameter_grid_search import generate_precise_range
            gse_var_range = list(generate_precise_range(
                gse_grid.get('variance_start', 0.5),
                gse_grid.get('variance_end', 15.0),
                gse_grid.get('variance_step', 0.5)
            ))
            gse_sim_range = list(generate_precise_range(
                gse_grid.get('similarity_start', 0.1),
                gse_grid.get('similarity_end', 0.9),
                gse_grid.get('similarity_step', 0.05)
            ))
            
            run_grid_search(
                'gse96058_data',
                gse_input_file,  # This is the prepared CSV, not raw clinical/expression files
                gse_var_range,
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
        
        # Use dataset-specific variance thresholds if available, otherwise fallback
        variance_thresholds = preprocessing_config.get('variance_thresholds', {})
        
        # Process TCGA BRCA
        tcga_output = config['dataset_preparation']['tcga']['output']
        if Path(tcga_output).exists():
            # Determine threshold for TCGA
            if variance_thresholds and 'tcga_brca_data' in variance_thresholds:
                tcga_threshold = variance_thresholds['tcga_brca_data']
                print(f"\nUsing dataset-specific variance threshold for TCGA-BRCA: {tcga_threshold}")
            elif variance_thresholds and 'default' in variance_thresholds:
                tcga_threshold = variance_thresholds['default']
                print(f"\nUsing default variance threshold for TCGA-BRCA: {tcga_threshold}")
            else:
                tcga_threshold = preprocessing_config.get('variance_threshold', 'mean')
                print(f"\nUsing fallback variance threshold for TCGA-BRCA: {tcga_threshold}")
            
            preprocess_data(tcga_output, output_dir='data/processed', 
                          variance_threshold=tcga_threshold)
        
        # Process GSE96058
        gse_output = config['dataset_preparation']['gse96058']['output']
        if Path(gse_output).exists():
            # Determine threshold for GSE96058
            if variance_thresholds and 'gse96058_data' in variance_thresholds:
                gse_threshold = variance_thresholds['gse96058_data']
                print(f"\nUsing dataset-specific variance threshold for GSE96058: {gse_threshold}")
            elif variance_thresholds and 'default' in variance_thresholds:
                gse_threshold = variance_thresholds['default']
                print(f"\nUsing default variance threshold for GSE96058: {gse_threshold}")
            else:
                gse_threshold = preprocessing_config.get('variance_threshold', 'mean')
                print(f"\nUsing fallback variance threshold for GSE96058: {gse_threshold}")
            
            preprocess_data(gse_output, output_dir='data/processed',
                          variance_threshold=gse_threshold)
    
    # Step 2: Graph Construction
    if 'graph' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 2: GRAPH CONSTRUCTION")
        print("="*80)
        
        preprocessing_config = config.get('preprocessing', {})
        
        # Use dataset-specific thresholds if available, otherwise fallback to single threshold
        similarity_thresholds = preprocessing_config.get('similarity_thresholds', {})
        if similarity_thresholds:
            print("Using dataset-specific similarity thresholds:")
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
        
        cluster_all_graphs(input_dir='data/graphs', output_dir='data/clusterings',
                         max_communities=max_communities, iterations=iterations, lr=lr)
    
    # Step 4: Evaluation
    if 'evaluate' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 4: EVALUATION")
        print("="*80)
        
        evaluate_all_datasets(clustering_dir='data/clusterings',
                            targets_dir='data/processed',
                            output_dir='results/evaluation')
    
    # Step 5: Visualization
    if 'visualize' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 5: VISUALIZATION")
        print("="*80)
        
        create_all_visualizations(processed_dir='data/processed',
                                clustering_dir='data/clusterings',
                                output_dir='results/visualization')
    
    # Step 6: Classification Validation
    if 'classify' in steps_to_run and not args.skip_classification:
        print("\n" + "="*80)
        print("STEP 6: CLASSIFICATION VALIDATION")
        print("="*80)
        
        classify_config = config.get('classifiers', {})
        mlp_params = classify_config.get('mlp', {})
        svm_params = classify_config.get('svm', {})
        
        validate_clustering_with_classifiers(processed_dir='data/processed',
                                           clustering_dir='data/clusterings',
                                           output_dir='results/classification',
                                           mlp_params=mlp_params, svm_params=svm_params)
    
    # Step 7: Interpretation
    if 'interpret' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 7: INTERPRETATION")
        print("="*80)
        print("Interpretation results are included in evaluation output above.")
        print("See results/evaluation/ for detailed analysis.")
    
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
    print("  🔗 Cross-dataset:  results/cross_dataset/")
    print("\nKey findings:")
    print("  • Check ARI/NMI scores in evaluation output")
    print("  • View t-SNE/UMAP plots to see cluster separation")
    print("  • Review confusion matrices for label correspondence")
    print("  • Examine cross-dataset correlations for consistency")
    print("="*80)


if __name__ == "__main__":
    main()

