#!/usr/bin/env python3
"""
Parallel experiment runner for multiple datasets and configurations.
Runs experiments across different vehicle types, years, months, and boroughs.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import logging
import os

def run_single_experiment(config):
    """
    Run a single experiment configuration.
    
    Args:
        config: Dictionary with experiment configuration
        
    Returns:
        Dictionary with experiment results
    """
    try:
        # Prepare command based on experiment type
        script_name = f"experiment_{config['experiment_type']}_refactored.py"
        
        cmd = [
            'python3', script_name,
            config['borough'],
            str(config['day']),
            str(config['time_interval']),
            config['time_unit'],
            str(config['simulation_range'])
        ]
        
        # Run experiment
        start_time = datetime.now()
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd='bin',
            timeout=3600  # 1 hour timeout
        )
        end_time = datetime.now()
        
        return {
            'config': config,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': (end_time - start_time).total_seconds(),
            'return_code': result.returncode,
            'success': result.returncode == 0,
            'stdout_preview': result.stdout[:500] if result.stdout else '',
            'stderr_preview': result.stderr[:500] if result.stderr else ''
        }
        
    except subprocess.TimeoutExpired:
        return {
            'config': config,
            'success': False,
            'error': 'Timeout (> 1 hour)',
            'return_code': -1
        }
    except Exception as e:
        return {
            'config': config,
            'success': False,
            'error': str(e),
            'return_code': -1
        }

def generate_experiment_configs(vehicle_types, years, months, days, 
                              experiment_types, boroughs, time_interval=30, 
                              time_unit='s', simulation_range=5):
    """Generate all experiment configurations."""
    configs = []
    
    for vehicle_type in vehicle_types:
        for year in years:
            for month in months:
                # Check if data file exists (try both parquet and csv, both locations)
                parquet_file = Path(f"data/{vehicle_type}_tripdata_{year}-{month:02d}.parquet")
                csv_file = Path(f"data/{vehicle_type}_tripdata_{year}-{month:02d}.csv")
                csv_subdir = Path(f"data/csv/{vehicle_type}_tripdata_{year}-{month:02d}.csv")
                
                if not (parquet_file.exists() or csv_file.exists() or csv_subdir.exists()):
                    print(f"⚠️  Skipping {vehicle_type} {year}-{month:02d}: data file not found")
                    continue
                
                for day in days:
                    for exp_type in experiment_types:
                        for borough in boroughs:
                            config = {
                                'vehicle_type': vehicle_type,
                                'year': year,
                                'month': month,
                                'day': day,
                                'experiment_type': exp_type,
                                'borough': borough,
                                'time_interval': time_interval,
                                'time_unit': time_unit,
                                'simulation_range': simulation_range,
                                'id': f"{vehicle_type}_{year}_{month:02d}_{day:02d}_{exp_type}_{borough}"
                            }
                            configs.append(config)
    
    return configs

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Parallel Experiment Runner')
    
    # Data selection
    parser.add_argument('--vehicle-types', nargs='+',
                       choices=['yellow', 'green', 'fhv', 'fhvhv'],
                       default=['yellow'],
                       help='Vehicle types to experiment with')
    
    parser.add_argument('--years', nargs='+', type=int,
                       default=[2019],
                       help='Years to experiment with')
    
    parser.add_argument('--months', nargs='+', type=int,
                       default=[10],
                       help='Months to experiment with')
    
    parser.add_argument('--days', nargs='+', type=int,
                       default=[6],
                       help='Days to experiment with')
    
    # Experiment configuration
    parser.add_argument('--experiment-types', nargs='+',
                       choices=['PL', 'Sigmoid'],
                       default=['PL'],
                       help='Experiment types to run')
    
    parser.add_argument('--boroughs', nargs='+',
                       choices=['Manhattan', 'Queens', 'Bronx', 'Brooklyn'],
                       default=['Manhattan'],
                       help='Boroughs to experiment with')
    
    parser.add_argument('--time-interval', type=int, default=30,
                       help='Time interval for simulation')
    
    parser.add_argument('--time-unit', choices=['s', 'm'], default='s',
                       help='Time unit for simulation')
    
    parser.add_argument('--simulation-range', type=int, default=5,
                       help='Number of simulation iterations')
    
    # Execution configuration
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of parallel workers')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show experiment configurations without running')
    
    args = parser.parse_args()
    
    print("🧪 Parallel Experiment Runner")
    print("=" * 40)
    print(f"Vehicle types: {args.vehicle_types}")
    print(f"Years: {args.years}")
    print(f"Months: {args.months}")
    print(f"Days: {args.days}")
    print(f"Experiment types: {args.experiment_types}")
    print(f"Boroughs: {args.boroughs}")
    print(f"Max workers: {args.max_workers}")
    print()
    
    # Generate experiment configurations
    configs = generate_experiment_configs(
        vehicle_types=args.vehicle_types,
        years=args.years,
        months=args.months,
        days=args.days,
        experiment_types=args.experiment_types,
        boroughs=args.boroughs,
        time_interval=args.time_interval,
        time_unit=args.time_unit,
        simulation_range=args.simulation_range
    )
    
    if not configs:
        print("❌ No valid experiment configurations generated")
        return 1
    
    print(f"📋 Generated {len(configs)} experiment configurations")
    
    if args.dry_run:
        print("\n🔍 Experiment configurations (dry run):")
        for i, config in enumerate(configs, 1):
            print(f"  {i:2d}. {config['id']}")
        print(f"\n✅ Dry run complete - {len(configs)} experiments would be executed")
        return 0
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    (results_dir / "parallel_runs").mkdir(exist_ok=True)

    # Run experiments in parallel
    print(f"\n🚀 Starting parallel execution...")
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    results = []
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all experiments
        future_to_config = {
            executor.submit(run_single_experiment, config): config
            for config in configs
        }
        
        # Process completed experiments
        for i, future in enumerate(as_completed(future_to_config), 1):
            config = future_to_config[future]
            
            try:
                result = future.result()
                results.append(result)
                
                if result['success']:
                    successful += 1
                    status = "✅"
                else:
                    failed += 1
                    status = "❌"
                
                print(f"  {status} ({i:2d}/{len(configs)}) {config['id']} "
                      f"[{result.get('duration_seconds', 0):.1f}s]")
                
            except Exception as e:
                failed += 1
                print(f"  ❌ ({i:2d}/{len(configs)}) {config['id']} - Exception: {e}")
                results.append({
                    'config': config,
                    'success': False,
                    'error': str(e)
                })
    
    # Create summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiment_args': vars(args),
        'total_experiments': len(configs),
        'successful': successful,
        'failed': failed,
        'success_rate': successful / len(configs) if configs else 0,
        'results': results
    }
    
    # Save summary
    summary_file = results_dir / "parallel_runs" / f"run_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print final summary
    print(f"\n📊 Execution Summary:")
    print(f"   Total experiments: {len(configs)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {summary['success_rate']:.1%}")
    print(f"   Summary saved to: {summary_file}")
    
    if failed > 0:
        print(f"\n⚠️  {failed} experiments failed. Check the summary file for details.")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 