#!/usr/bin/env python3
"""
Aggregate and analyze results from multiple experiment runs.
Creates comprehensive summaries and comparisons across different configurations.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_experiment_results(results_dir="results"):
    """Load all experiment results from the results directory."""
    results_path = Path(results_dir)
    
    # Find all detailed result files
    detailed_files = list((results_path / "detailed").glob("detailed_*.json"))
    summary_files = list((results_path / "summary").glob("summary_*.csv"))
    
    experiments = []
    
    # Load detailed JSON results
    for file_path in detailed_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract metadata from filename or content
            exp_info = {
                'file_path': str(file_path),
                'experiment_name': data.get('experiment_name', ''),
                'parameters': data.get('parameters', {}),
                'iterations': data.get('iterations', []),
                'total_iterations': len(data.get('iterations', [])),
                'kpis': data.get('kpis', {})
            }
            
            experiments.append(exp_info)
            
        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {e}")
    
    return experiments

def extract_performance_metrics(experiments):
    """Extract performance metrics from experiment data."""
    performance_data = []
    
    for exp in experiments:
        exp_name = exp['experiment_name']
        params = exp['parameters']
        iterations = exp['iterations']
        
        # Extract experiment metadata
        base_info = {
            'experiment_type': 'PL' if 'PL' in exp_name else 'Sigmoid' if 'Sigmoid' in exp_name else 'Unknown',
            'place': params.get('place', 'Unknown'),
            'day': params.get('day', 0),
            'vehicle_type': extract_vehicle_type(exp_name),
            'year_month': f"{params.get('year', 0)}-{params.get('month', 0):02d}",
            'total_iterations': len(iterations)
        }
        
        # Aggregate method performance across iterations
        method_performance = {}
        
        for iteration in iterations:
            results = iteration.get('results', {})
            
            for method, method_data in results.items():
                if method not in method_performance:
                    method_performance[method] = {
                        'objectives': [],
                        'solve_times': [],
                        'success_count': 0
                    }
                
                if isinstance(method_data, dict):
                    if 'objective_value' in method_data:
                        method_performance[method]['objectives'].append(method_data['objective_value'])
                    
                    solve_time = method_data.get('solve_time', method_data.get('computation_time', 0))
                    method_performance[method]['solve_times'].append(solve_time)
                    method_performance[method]['success_count'] += 1
        
        # Calculate statistics for each method
        for method, perf_data in method_performance.items():
            if perf_data['objectives']:
                record = base_info.copy()
                record.update({
                    'method': method,
                    'avg_objective': np.mean(perf_data['objectives']),
                    'std_objective': np.std(perf_data['objectives']),
                    'min_objective': np.min(perf_data['objectives']),
                    'max_objective': np.max(perf_data['objectives']),
                    'avg_solve_time': np.mean(perf_data['solve_times']),
                    'std_solve_time': np.std(perf_data['solve_times']),
                    'success_rate': perf_data['success_count'] / len(iterations),
                    'iterations_with_method': len(perf_data['objectives'])
                })
                
                performance_data.append(record)
    
    return pd.DataFrame(performance_data)

def extract_vehicle_type(experiment_name):
    """Extract vehicle type from experiment name."""
    name_lower = experiment_name.lower()
    if 'yellow' in name_lower:
        return 'yellow'
    elif 'green' in name_lower:
        return 'green'
    elif 'fhv' in name_lower:
        return 'fhv'
    elif 'fhvhv' in name_lower:
        return 'fhvhv'
    else:
        return 'unknown'

def create_summary_tables(df):
    """Create summary tables for different groupings."""
    summaries = {}
    
    if df.empty:
        return summaries
    
    # Summary by experiment type
    if 'experiment_type' in df.columns:
        summaries['by_experiment_type'] = df.groupby('experiment_type').agg({
            'avg_objective': ['mean', 'std', 'count'],
            'avg_solve_time': ['mean', 'std'],
            'success_rate': 'mean'
        }).round(4)
    
    # Summary by vehicle type
    if 'vehicle_type' in df.columns:
        summaries['by_vehicle_type'] = df.groupby('vehicle_type').agg({
            'avg_objective': ['mean', 'std', 'count'],
            'avg_solve_time': ['mean', 'std'],
            'success_rate': 'mean'
        }).round(4)
    
    # Summary by method
    if 'method' in df.columns:
        summaries['by_method'] = df.groupby('method').agg({
            'avg_objective': ['mean', 'std', 'count'],
            'avg_solve_time': ['mean', 'std'],
            'success_rate': 'mean'
        }).round(4)
    
    # Summary by place
    if 'place' in df.columns:
        summaries['by_place'] = df.groupby('place').agg({
            'avg_objective': ['mean', 'std', 'count'],
            'avg_solve_time': ['mean', 'std'],
            'success_rate': 'mean'
        }).round(4)
    
    return summaries

def create_visualizations(df, output_dir):
    """Create visualization plots."""
    if df.empty:
        print("⚠️  No data available for visualization")
        return
    
    plt.style.use('default')
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # 1. Objective values by experiment type
    if 'experiment_type' in df.columns and 'avg_objective' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='experiment_type', y='avg_objective')
        plt.title('Objective Values by Experiment Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(fig_dir / "objectives_by_experiment_type.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Solve times by method
    if 'method' in df.columns and 'avg_solve_time' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='method', y='avg_solve_time')
        plt.title('Solve Times by Method')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(fig_dir / "solve_times_by_method.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Performance comparison heatmap
    if 'vehicle_type' in df.columns and 'experiment_type' in df.columns:
        pivot_data = df.pivot_table(
            values='avg_objective', 
            index='vehicle_type', 
            columns='experiment_type',
            aggfunc='mean'
        )
        
        if not pivot_data.empty:
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.2f')
            plt.title('Average Objective Value: Vehicle Type vs Experiment Type')
            plt.tight_layout()
            plt.savefig(fig_dir / "performance_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"📊 Visualizations saved to: {fig_dir}")

def main():
    """Main aggregation function."""
    parser = argparse.ArgumentParser(description='Aggregate Experiment Results')
    
    parser.add_argument('--results-dir', default='results',
                       help='Results directory (default: results)')
    
    parser.add_argument('--output-dir', default='results/aggregated',
                       help='Output directory for aggregated results')
    
    parser.add_argument('--create-plots', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    print("📊 Experiment Results Aggregator")
    print("=" * 40)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load experiment results
    print("📂 Loading experiment results...")
    experiments = load_experiment_results(args.results_dir)
    
    if not experiments:
        print("❌ No experiment results found")
        return 1
    
    print(f"✅ Loaded {len(experiments)} experiments")
    
    # Extract performance metrics
    print("🔍 Extracting performance metrics...")
    df = extract_performance_metrics(experiments)
    
    if df.empty:
        print("❌ No performance data could be extracted")
        return 1
    
    print(f"✅ Extracted metrics for {len(df)} experiment-method combinations")
    
    # Create summary tables
    print("📋 Creating summary tables...")
    summaries = create_summary_tables(df)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed data
    df.to_csv(output_dir / f'detailed_performance_{timestamp}.csv', index=False)
    
    # Save summary tables
    summary_file = output_dir / f'summary_tables_{timestamp}.txt'
    with open(summary_file, 'w') as f:
        f.write("EXPERIMENT RESULTS SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total experiments: {len(experiments)}\n")
        f.write(f"Total performance records: {len(df)}\n\n")
        
        for name, summary_df in summaries.items():
            f.write(f"\n{name.upper()}\n")
            f.write("-" * len(name) + "\n")
            f.write(str(summary_df))
            f.write("\n\n")
    
    # Create visualizations
    if args.create_plots:
        print("📊 Creating visualizations...")
        create_visualizations(df, args.output_dir)
    
    # Final summary
    print(f"\n🎉 Aggregation Complete!")
    print(f"   Experiments processed: {len(experiments)}")
    print(f"   Performance records: {len(df)}")
    print(f"   Summary tables: {len(summaries)}")
    print(f"   Results saved to: {output_dir}")
    
    # Show quick stats
    if not df.empty:
        print(f"\n📈 Quick Stats:")
        if 'avg_objective' in df.columns:
            print(f"   Avg objective value: {df['avg_objective'].mean():.2f} ± {df['avg_objective'].std():.2f}")
        if 'avg_solve_time' in df.columns:
            print(f"   Avg solve time: {df['avg_solve_time'].mean():.3f}s ± {df['avg_solve_time'].std():.3f}s")
        if 'success_rate' in df.columns:
            print(f"   Avg success rate: {df['success_rate'].mean():.1%}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 