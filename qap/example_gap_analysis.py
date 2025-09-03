#!/usr/bin/env python3
"""
Comprehensive example script for analyzing QAP gap curves.
This script demonstrates how to use the new gap curve plotting functions
and provides examples for different analysis scenarios.
"""

import sys
import os
import glob
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from read_qap_log import (
    read_gap_curves_from_json,
    read_gap_curves_from_old_format,
    plot_gap_curves_for_run,
    plot_best_run_gap_curves_for_problem,
    plot_average_gap_curves_across_problems,
    plot_problem_gap_comparison
)

def find_gap_curve_files():
    """
    Find all gap curve JSON files in the train_be directory.
    """
    pattern = "log/gap_curves_*.json"
    files = glob.glob(pattern)
    return files

def analyze_single_problem(gap_results, problem_name):
    """
    Analyze gap curves for a single problem.
    """
    print(f"\n{'='*60}")
    print(f"ANALYSIS FOR PROBLEM: {problem_name}")
    print(f"{'='*60}")
    
    if problem_name not in gap_results:
        print(f"Problem {problem_name} not found in results.")
        return
    
    runs = gap_results[problem_name]
    print(f"Number of runs: {len(runs)}")
    
    # Find best run
    best_run_idx = None
    best_final_gap = float('inf')
    
    for run_idx, (epochs, gaps) in runs.items():
        if len(gaps) > 0:
            final_gap = gaps[-1]
            print(f"  Run {run_idx}: Final gap = {final_gap:.2f}%")
            if final_gap < best_final_gap:
                best_final_gap = final_gap
                best_run_idx = run_idx
    
    print(f"Best run: {best_run_idx} (Final gap: {best_final_gap:.2f}%)")
    
    # Plot best run
    plot_best_run_gap_curves_for_problem(gap_results, problem_name)

def analyze_multiple_problems(gap_results, problem_names=None):
    """
    Analyze gap curves across multiple problems.
    """
    if problem_names is None:
        problem_names = list(gap_results.keys())
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS ACROSS {len(problem_names)} PROBLEMS")
    print(f"{'='*60}")
    
    # Print summary for each problem
    for problem_name in problem_names:
        if problem_name in gap_results:
            runs = gap_results[problem_name]
            best_final_gap = float('inf')
            for run_idx, (epochs, gaps) in runs.items():
                if len(gaps) > 0:
                    final_gap = gaps[-1]
                    if final_gap < best_final_gap:
                        best_final_gap = final_gap
            print(f"{problem_name}: Best final gap = {best_final_gap:.2f}%")
    
    # Plot average across problems
    plot_average_gap_curves_across_problems(gap_results, problem_names)
    
    # Plot comparison
    plot_problem_gap_comparison(gap_results, problem_names)

def compare_algorithms():
    """
    Compare different algorithms by finding their gap curve files.
    """
    print(f"\n{'='*60}")
    print("ALGORITHM COMPARISON")
    print(f"{'='*60}")
    
    # Find all gap curve files
    files = find_gap_curve_files()
    
    if not files:
        print("No gap curve files found in log/ directory.")
        return
    
    print(f"Found {len(files)} gap curve files:")
    for file in files:
        print(f"  {file}")
    
    # Group files by algorithm
    algorithm_files = {}
    for file in files:
        # Extract algorithm name from filename
        filename = os.path.basename(file)
        parts = filename.split('_')
        if len(parts) >= 3:
            alg_name = parts[2]  # e.g., "gd+k+5"
            if alg_name not in algorithm_files:
                algorithm_files[alg_name] = []
            algorithm_files[alg_name].append(file)
    
    print(f"\nAlgorithms found:")
    for alg, files_list in algorithm_files.items():
        print(f"  {alg}: {len(files_list)} files")
    
    # For demonstration, analyze the first algorithm
    if algorithm_files:
        first_alg = list(algorithm_files.keys())[0]
        first_file = algorithm_files[first_alg][0]
        print(f"\nAnalyzing algorithm: {first_alg}")
        print(f"File: {first_file}")
        
        try:
            gap_results = read_gap_curves_from_json(first_file)
            if len(gap_results) > 0:
                first_problem = list(gap_results.keys())[0]
                analyze_single_problem(gap_results, first_problem)
        except Exception as e:
            print(f"Error reading file {first_file}: {e}")

def main():
    """
    Main function demonstrating gap curve analysis.
    """
    print("QAP Gap Curve Analysis")
    print("=" * 60)
    
    # Find gap curve files
    files = find_gap_curve_files()
    
    if not files:
        print("No gap curve files found.")
        print("Please run test_grad_be_qap.py first to generate gap curve data.")
        return
    
    # Use the first file found
    json_file = files[0]
    print(f"Using file: {json_file}")
    
    # Try to read the file
    try:
        gap_results = read_gap_curves_from_json(json_file)
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        print("Trying old format...")
        try:
            gap_results = read_gap_curves_from_old_format(json_file)
        except Exception as e2:
            print(f"Error reading old format: {e2}")
            return
    
    print(f"Successfully loaded gap curves for {len(gap_results)} problems")
    
    # List available problems
    problem_names = list(gap_results.keys())
    print(f"Available problems: {problem_names}")
    
    # Analyze first problem
    if len(problem_names) > 0:
        analyze_single_problem(gap_results, problem_names[0])
    
    # Analyze multiple problems
    if len(problem_names) >= 2:
        analyze_multiple_problems(gap_results, problem_names[:2])
    
    # Compare algorithms
    compare_algorithms()
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 