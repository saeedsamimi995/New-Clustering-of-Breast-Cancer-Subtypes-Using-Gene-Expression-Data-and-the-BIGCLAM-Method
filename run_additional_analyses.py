"""
Master script to run all additional validation analyses.

This script runs comprehensive validation analyses:
1. Baseline comparison
2. Computational benchmarking
3. Augmentation ablation
4. Method comparison
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_analysis(script_name, args_list):
    """Run an analysis script and handle errors."""
    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print(f"{'='*80}\n")
    
    cmd = [sys.executable, script_name] + args_list
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with return code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Run all additional validation analyses')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['tcga_brca_data', 'gse96058_data', 'both'],
                       help='Dataset to analyze')
    parser.add_argument('--skip_baseline', action='store_true',
                       help='Skip baseline comparison')
    parser.add_argument('--skip_benchmark', action='store_true',
                       help='Skip computational benchmarking')
    parser.add_argument('--skip_augmentation', action='store_true',
                       help='Skip augmentation ablation')
    parser.add_argument('--skip_methods', action='store_true',
                       help='Skip method comparison')
    
    args = parser.parse_args()
    
    datasets = []
    if args.dataset == 'both':
        datasets = ['tcga_brca_data', 'gse96058_data']
    else:
        datasets = [args.dataset]
    
    print("\n" + "="*80)
    print("ADDITIONAL VALIDATION ANALYSES")
    print("="*80)
    print(f"\nDatasets: {', '.join(datasets)}")
    print(f"Total analyses: {len(datasets) * 4}")
    
    success_count = 0
    total_count = 0
    
    for dataset in datasets:
        print(f"\n\n{'#'*80}")
        print(f"PROCESSING: {dataset}")
        print(f"{'#'*80}\n")
        
        # 1. Baseline Comparison
        if not args.skip_baseline:
            total_count += 1
            if run_analysis('src/analysis/baseline_comparison.py', [
                '--dataset', dataset,
                '--processed_dir', 'data/processed',
                '--clustering_dir', 'data/clusterings',
                '--output_dir', 'results/baseline_comparison'
            ]):
                success_count += 1
        
        # 2. Computational Benchmarking
        if not args.skip_benchmark:
            total_count += 1
            # Determine input file
            if 'tcga' in dataset:
                input_file = 'data/tcga_brca_data_target_added.csv'
            else:
                input_file = 'data/gse96058_data_target_added.csv'
            
            if Path(input_file).exists():
                if run_analysis('src/analysis/computational_benchmark.py', [
                    '--dataset', dataset,
                    '--input_file', input_file,
                    '--config', 'config/config.yml',
                    '--output_dir', 'results/benchmarks'
                ]):
                    success_count += 1
            else:
                print(f"[WARNING] Input file not found: {input_file}")
                print("         Skipping computational benchmark...")
        
        # 3. Augmentation Ablation
        if not args.skip_augmentation:
            total_count += 1
            if run_analysis('src/analysis/augmentation_ablation.py', [
                '--dataset', dataset,
                '--processed_dir', 'data/processed',
                '--clustering_dir', 'data/clusterings',
                '--output_dir', 'results/augmentation_ablation'
            ]):
                success_count += 1
        
        # 4. Method Comparison
        if not args.skip_methods:
            total_count += 1
            if run_analysis('src/analysis/method_comparison.py', [
                '--dataset', dataset,
                '--processed_dir', 'data/processed',
                '--clustering_dir', 'data/clusterings',
                '--output_dir', 'results/method_comparison',
                '--n_clusters', '4'
            ]):
                success_count += 1
    
    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Completed: {success_count}/{total_count} analyses")
    
    if success_count == total_count:
        print("\n✅ All analyses completed successfully!")
        print("\nResults saved in:")
        print("  - results/baseline_comparison/")
        print("  - results/benchmarks/")
        print("  - results/augmentation_ablation/")
        print("  - results/method_comparison/")
    else:
        print(f"\n⚠️  {total_count - success_count} analyses failed")
        print("Check error messages above for details")


if __name__ == "__main__":
    main()

