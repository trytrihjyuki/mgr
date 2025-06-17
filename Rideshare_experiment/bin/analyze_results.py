#!/usr/bin/env python3
"""
Simple analysis script for ride-hailing experiment results.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from typing import Dict, List

def load_detailed_results(file_path: str) -> Dict:
    """Load detailed results from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        return {}

def analyze_experiment(data: Dict) -> Dict:
    """Analyze experiment data and return summary statistics."""
    if not data or 'iterations' not in data:
        return {}
    
    iterations = data['iterations']
    if not iterations:
        return {}
    
    # Extract metrics for each method
    methods = set()
    for iteration in iterations:
        if 'results' in iteration:
            methods.update(iteration['results'].keys())
    
    analysis = {
        'experiment_info': data.get('parameters', {}),
        'total_iterations': len(iterations),
        'methods': {}
    }
    
    for method in methods:
        objectives = []
        solve_times = []
        successful_iterations = 0
        
        for iteration in iterations:
            if 'results' in iteration and method in iteration['results']:
                result = iteration['results'][method]
                if isinstance(result, dict):
                    if 'objective_value' in result:
                        objectives.append(result['objective_value'])
                    if 'solve_time' in result:
                        solve_times.append(result['solve_time'])
                    if result.get('status') == 'optimal':
                        successful_iterations += 1
        
        if objectives:
            analysis['methods'][method] = {
                'avg_objective': np.mean(objectives),
                'std_objective': np.std(objectives),
                'min_objective': np.min(objectives),
                'max_objective': np.max(objectives),
                'avg_solve_time': np.mean(solve_times) if solve_times else 0,
                'std_solve_time': np.std(solve_times) if solve_times else 0,
                'success_rate': successful_iterations / len(iterations) if iterations else 0,
                'num_iterations': len(objectives)
            }
    
    return analysis

def print_analysis(analysis: Dict):
    """Print analysis results in a readable format."""
    if not analysis:
        print("No data to analyze.")
        return
    
    print("=" * 50)
    print("EXPERIMENT ANALYSIS")
    print("=" * 50)
    
    # Experiment info
    if 'experiment_info' in analysis:
        info = analysis['experiment_info']
        print(f"Place: {info.get('place', 'Unknown')}")
        print(f"Day: {info.get('day', 'Unknown')}")
        print(f"Time Interval: {info.get('time_interval', 'Unknown')}{info.get('time_unit', '')}")
        print(f"Total Iterations: {analysis.get('total_iterations', 0)}")
        print()
    
    # Results by method
    if 'methods' in analysis:
        for method, stats in analysis['methods'].items():
            print(f"Method: {method}")
            print("-" * 30)
            print(f"Average Objective Value: {stats['avg_objective']:.4f} ± {stats['std_objective']:.4f}")
            print(f"Objective Range: [{stats['min_objective']:.4f}, {stats['max_objective']:.4f}]")
            print(f"Average Solve Time: {stats['avg_solve_time']:.4f}s ± {stats['std_solve_time']:.4f}s")
            print(f"Success Rate: {stats['success_rate']:.2%}")
            print(f"Valid Iterations: {stats['num_iterations']}")
            print()

def create_plots(analysis: Dict, output_dir: str = None):
    """Create plots from analysis data."""
    if not analysis or 'methods' not in analysis:
        print("No data available for plotting.")
        return
    
    methods = list(analysis['methods'].keys())
    if not methods:
        return
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Objective values
    objectives = [analysis['methods'][m]['avg_objective'] for m in methods]
    obj_stds = [analysis['methods'][m]['std_objective'] for m in methods]
    
    ax1.bar(methods, objectives, yerr=obj_stds, capsize=5)
    ax1.set_title('Average Objective Value by Method')
    ax1.set_ylabel('Objective Value')
    ax1.tick_params(axis='x', rotation=45)
    
    # Solve times
    solve_times = [analysis['methods'][m]['avg_solve_time'] for m in methods]
    time_stds = [analysis['methods'][m]['std_solve_time'] for m in methods]
    
    ax2.bar(methods, solve_times, yerr=time_stds, capsize=5, color='orange')
    ax2.set_title('Average Solve Time by Method')
    ax2.set_ylabel('Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Success rates
    success_rates = [analysis['methods'][m]['success_rate'] * 100 for m in methods]
    
    ax3.bar(methods, success_rates, color='green')
    ax3.set_title('Success Rate by Method')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_ylim(0, 100)
    ax3.tick_params(axis='x', rotation=45)
    
    # Objective ranges
    mins = [analysis['methods'][m]['min_objective'] for m in methods]
    maxs = [analysis['methods'][m]['max_objective'] for m in methods]
    avgs = objectives
    
    x_pos = range(len(methods))
    ax4.scatter(x_pos, mins, label='Min', color='red', alpha=0.7)
    ax4.scatter(x_pos, maxs, label='Max', color='blue', alpha=0.7)
    ax4.scatter(x_pos, avgs, label='Avg', color='black', s=100)
    
    for i, method in enumerate(methods):
        ax4.plot([i, i], [mins[i], maxs[i]], 'k-', alpha=0.3)
    
    ax4.set_title('Objective Value Ranges')
    ax4.set_ylabel('Objective Value')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(methods, rotation=45)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        plot_file = output_path / "analysis_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
    
    plt.show()

def compare_experiments(file_paths: List[str]):
    """Compare multiple experiments."""
    print("=" * 50)
    print("EXPERIMENT COMPARISON")
    print("=" * 50)
    
    all_analyses = []
    
    for file_path in file_paths:
        data = load_detailed_results(file_path)
        analysis = analyze_experiment(data)
        if analysis:
            analysis['file_path'] = file_path
            all_analyses.append(analysis)
    
    if not all_analyses:
        print("No valid experiments to compare.")
        return
    
    # Create comparison table
    comparison_data = []
    
    for analysis in all_analyses:
        info = analysis.get('experiment_info', {})
        row = {
            'Experiment': f"{info.get('place', 'Unknown')}_{info.get('day', 'X')}_{info.get('time_interval', 'X')}{info.get('time_unit', '')}",
            'Iterations': analysis.get('total_iterations', 0)
        }
        
        # Add method statistics
        for method, stats in analysis.get('methods', {}).items():
            row[f'{method}_Objective'] = f"{stats['avg_objective']:.2f}"
            row[f'{method}_Time'] = f"{stats['avg_solve_time']:.4f}s"
            row[f'{method}_Success'] = f"{stats['success_rate']:.1%}"
        
        comparison_data.append(row)
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
    
def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze ride-hailing experiment results')
    parser.add_argument('files', nargs='+', help='JSON result files to analyze')
    parser.add_argument('--compare', action='store_true', help='Compare multiple experiments')
    parser.add_argument('--plot', action='store_true', help='Create plots')
    parser.add_argument('--output-dir', help='Directory to save plots')
    
    args = parser.parse_args()
    
    if args.compare and len(args.files) > 1:
        compare_experiments(args.files)
    else:
        # Analyze single experiment
        if len(args.files) != 1:
            print("Please provide exactly one file for single experiment analysis")
            sys.exit(1)
        
        data = load_detailed_results(args.files[0])
        analysis = analyze_experiment(data)
        print_analysis(analysis)
        
        if args.plot:
            create_plots(analysis, args.output_dir)

if __name__ == "__main__":
    main() 