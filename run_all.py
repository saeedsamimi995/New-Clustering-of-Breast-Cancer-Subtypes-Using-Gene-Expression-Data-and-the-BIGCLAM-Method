"""
Main Orchestration Script

Runs the complete BIGCLAM pipeline:
1. Data preparation (if needed)
2. Data preprocessing
3. Graph construction
4. Clustering
5. Evaluation
6. Visualization
7. Interpretation
8. Cross-dataset analysis
9. Classification validation
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.preprocessing import prepare_tcga_brca_data, prepare_gse96058_data, preprocess_data
from src.graph import construct_graphs
from src.clustering import cluster_all_graphs
from src.evaluation import evaluate_all_datasets
from src.visualization import create_all_visualizations
from src.interpretation import interpret_results, analyze_overlap
from src.analysis import analyze_cross_dataset_consistency
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
    parser.add_argument('--skip_prep', action='store_true', help='Skip data preparation')
    parser.add_argument('--skip_clustering', action='store_true', help='Skip clustering (use existing results)')
    parser.add_argument('--skip_classification', action='store_true', help='Skip classification validation')
    parser.add_argument('--steps', nargs='+', choices=[
        'prepare', 'preprocess', 'graph', 'cluster', 'evaluate', 
        'visualize', 'interpret', 'cross_dataset', 'classify'
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
        steps_to_run = ['prepare', 'preprocess', 'graph', 'cluster', 'evaluate', 
                       'visualize', 'interpret', 'cross_dataset', 'classify']
    
    # Step 1: Data Preparation
    if 'prepare' in steps_to_run and not args.skip_prep:
        print("\n" + "="*80)
        print("STEP 1: DATA PREPARATION")
        print("="*80)
        
        dataset_config = config.get('dataset_preparation', {})
        
        # Prepare TCGA BRCA
        tcga_config = dataset_config.get('tcga', {})
        if tcga_config:
            prepare_tcga_brca_data(
                tcga_config.get('clinical'),
                tcga_config.get('expression'),
                tcga_config.get('output')
            )
        
        # Prepare GSE96058
        gse_config = dataset_config.get('gse96058', {})
        if gse_config:
            prepare_gse96058_data(
                gse_config.get('expression'),
                gse_config.get('output'),
                gse_config.get('clinical')
            )
    
    # Step 2: Data Preprocessing
    if 'preprocess' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 2: DATA PREPROCESSING")
        print("="*80)
        
        preprocessing_config = config.get('preprocessing', {})
        variance_threshold = preprocessing_config.get('variance_threshold', 'mean')
        
        # Process TCGA BRCA
        tcga_output = config['dataset_preparation']['tcga']['output']
        if Path(tcga_output).exists():
            preprocess_data(tcga_output, output_dir='data/processed', 
                          variance_threshold=variance_threshold)
        
        # Process GSE96058
        gse_output = config['dataset_preparation']['gse96058']['output']
        if Path(gse_output).exists():
            preprocess_data(gse_output, output_dir='data/processed',
                          variance_threshold=variance_threshold)
    
    # Step 3: Graph Construction
    if 'graph' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 3: GRAPH CONSTRUCTION")
        print("="*80)
        
        preprocessing_config = config.get('preprocessing', {})
        similarity_threshold = preprocessing_config.get('similarity_threshold', 0.4)
        
        construct_graphs(input_dir='data/processed', output_dir='data/graphs',
                        threshold=similarity_threshold, use_sparse=True)
    
    # Step 4: Clustering
    if 'cluster' in steps_to_run and not args.skip_clustering:
        print("\n" + "="*80)
        print("STEP 4: CLUSTERING WITH BIGCLAM")
        print("="*80)
        
        bigclam_config = config.get('bigclam', {})
        max_communities = bigclam_config.get('max_communities', 10)
        iterations = bigclam_config.get('iterations', 100)
        lr = bigclam_config.get('learning_rate', 0.08)
        
        cluster_all_graphs(input_dir='data/graphs', output_dir='data/clusterings',
                         max_communities=max_communities, iterations=iterations, lr=lr)
    
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
    
    # Step 7: Classification Validation
    if 'classify' in steps_to_run and not args.skip_classification:
        print("\n" + "="*80)
        print("STEP 7: CLASSIFICATION VALIDATION")
        print("="*80)
        
        classify_config = config.get('classifiers', {})
        mlp_params = classify_config.get('mlp', {})
        svm_params = classify_config.get('svm', {})
        
        validate_clustering_with_classifiers(processed_dir='data/processed',
                                           clustering_dir='data/clusterings',
                                           output_dir='results/classification',
                                           mlp_params=mlp_params, svm_params=svm_params)
    
    # Step 8: Interpretation
    if 'interpret' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 8: INTERPRETATION")
        print("="*80)
        print("Interpretation results are included in evaluation output above.")
        print("See results/evaluation/ for detailed analysis.")
    
    # Step 9: Cross-Dataset Analysis
    if 'cross_dataset' in steps_to_run:
        print("\n" + "="*80)
        print("STEP 9: CROSS-DATASET CONSISTENCY ANALYSIS")
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

