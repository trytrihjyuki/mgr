#!/usr/bin/env python3
"""
Parallel experiment runner for ride-hailing pricing experiments.
Supports multiple datasets, years, months, and vehicle types.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import subprocess
import os

from data_manager import NYCTaxiDataManager, DatasetInfo
from benchmark_utils import create_benchmark_summary

class ExperimentRunner:
    """
    Manages parallel execution of ride-hailing pricing experiments.
    """
    
    def __init__(self, data_dir: Path = Path("../data"), 
                 results_dir: Path = Path("../results"),
                 max_workers: int = 4):
        """
        Initialize experiment runner.
        
        Args:
            data_dir: Directory containing data files
            results_dir: Directory to store experiment results
            max_workers: Maximum number of parallel experiment processes
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.max_workers = max_workers
        
        # Create results subdirectories
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "experiments").mkdir(exist_ok=True)
        (self.results_dir / "aggregated").mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_experiment_configs(self, vehicle_types: List[str], years: List[int], 
                                  months: List[int], days: List[int],
                                  experiment_types: List[str],
                                  boroughs: List[str] = None) -> List[Dict[str, Any]]:
        """
        Generate experiment configurations.
        
        Args:
            vehicle_types: List of vehicle types
            years: List of years
            months: List of months
            days: List of days
            experiment_types: List of experiment types ('PL', 'Sigmoid')
            boroughs: List of boroughs (if None, use all)
            
        Returns:
            List of experiment configuration dictionaries
        """
        if boroughs is None:
            boroughs = ['Manhattan', 'Queens', 'Bronx', 'Brooklyn']
        
        configs = []
        
        for vehicle_type in vehicle_types:
            for year in years:
                for month in months:
                    for day in days:
                        for exp_type in experiment_types:
                            for borough in boroughs:
                                # Check if data file exists
                                csv_file = (self.data_dir / "csv" / 
                                          f"{vehicle_type}_tripdata_{year}-{month:02d}.csv")
                                
                                if csv_file.exists():
                                    config = {
                                        'vehicle_type': vehicle_type,
                                        'year': year,
                                        'month': month,
                                        'day': day,
                                        'experiment_type': exp_type,
                                        'borough': borough,
                                        'data_file': str(csv_file),
                                        'config_id': f"{vehicle_type}_{year}_{month:02d}_{day:02d}_{exp_type}_{borough}"
                                    }
                                    configs.append(config)
                                else:
                                    self.logger.warning(f"Data file not found: {csv_file}")
        
        return configs
    
    def run_single_experiment(self, config: Dict[str, Any], 
                            time_interval: int = 30, time_unit: str = 's',
                            simulation_range: int = 120) -> Dict[str, Any]:
        """
        Run a single experiment configuration.
        
        Args:
            config: Experiment configuration
            time_interval: Time interval for simulation
            time_unit: Time unit ('s' for seconds, 'm' for minutes)
            simulation_range: Number of simulation iterations
            
        Returns:
            Experiment result dictionary
        """
        try:
            exp_type = config['experiment_type']
            script_name = f"experiment_{exp_type}_refactored.py"
            
            # Prepare command
            cmd = [
                'python3', script_name,
                config['borough'],
                str(config['day']),
                str(time_interval),
                time_unit,
                str(simulation_range),
                '--vehicle-type', config['vehicle_type'],
                '--year', str(config['year']),
                '--month', str(config['month'])
            ]
            
            self.logger.info(f"🚀 Starting: {config['config_id']}")
            
            # Run experiment
            start_time = datetime.now()
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=self.data_dir.parent / "bin")
            end_time = datetime.now()
            
            # Prepare result
            experiment_result = {
                'config': config,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
            
            if result.returncode == 0:
                self.logger.info(f"✅ Completed: {config['config_id']}")
            else:
                self.logger.error(f"❌ Failed: {config['config_id']} (code: {result.returncode})")
                if result.stderr:
                    self.logger.error(f"   Error: {result.stderr[:200]}...")
            
            return experiment_result
            
        except Exception as e:
            self.logger.error(f"❌ Exception in {config['config_id']}: {e}")
            return {
                'config': config,
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': 0,
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'success': False
            }
    
    def run_experiments_parallel(self, configs: List[Dict[str, Any]], 
                                **experiment_kwargs) -> Dict[str, Any]:
        """
        Run multiple experiments in parallel.
        
        Args:
            configs: List of experiment configurations
            **experiment_kwargs: Additional arguments for run_single_experiment
            
        Returns:
            Summary of all experiment results
        """
        self.logger.info(f"🏃 Starting {len(configs)} experiments with {self.max_workers} workers...")
        
        results = []
        successful = 0
        failed = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all experiments
            future_to_config = {
                executor.submit(self.run_single_experiment, config, **experiment_kwargs): config
                for config in configs
            }
            
            # Process completed experiments
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        successful += 1
                    else:
                        failed += 1
                        
                except Exception as e:
                    self.logger.error(f"❌ Future exception for {config['config_id']}: {e}")
                    failed += 1
                    results.append({
                        'config': config,
                        'success': False,
                        'error': str(e)
                    })
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(configs),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(configs) if configs else 0,
            'results': results
        }
        
        # Save summary
        summary_file = self.results_dir / "experiments" / f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"📊 Experiment Summary:")
        self.logger.info(f"   Total: {len(configs)}")
        self.logger.info(f"   Successful: {successful}")
        self.logger.info(f"   Failed: {failed}")
        self.logger.info(f"   Success Rate: {summary['success_rate']:.1%}")
        self.logger.info(f"   Summary saved to: {summary_file}")
        
        return summary
    
    def aggregate_results(self, pattern: str = "detailed_*.json") -> Dict[str, Any]:
        """
        Aggregate results from multiple experiments.
        
        Args:
            pattern: File pattern to match for aggregation
            
        Returns:
            Aggregated results dictionary
        """
        self.logger.info(f"📊 Aggregating results with pattern: {pattern}")
        
        # Find all result files
        detailed_dir = self.results_dir / "detailed"
        if not detailed_dir.exists():
            self.logger.warning("No detailed results directory found")
            return {}
        
        result_files = list(detailed_dir.glob(pattern))
        self.logger.info(f"Found {len(result_files)} result files")
        
        aggregated = {
            'timestamp': datetime.now().isoformat(),
            'source_files': [str(f) for f in result_files],
            'experiments': [],
            'summary_by_method': {},
            'summary_by_vehicle_type': {},
            'summary_by_borough': {},
            'summary_by_year_month': {}
        }
        
        # Load and aggregate results
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    
                # Extract experiment info
                exp_info = data.get('parameters', {})
                iterations = data.get('iterations', [])
                
                if iterations:
                    # Calculate summary statistics for this experiment
                    exp_summary = self._calculate_experiment_summary(data)
                    exp_summary['experiment_info'] = exp_info
                    exp_summary['source_file'] = str(result_file)
                    
                    aggregated['experiments'].append(exp_summary)
                    
            except Exception as e:
                self.logger.error(f"Error processing {result_file}: {e}")
        
        # Calculate aggregated summaries
        if aggregated['experiments']:
            aggregated['summary_by_method'] = self._aggregate_by_field('method', aggregated['experiments'])
            aggregated['summary_by_vehicle_type'] = self._aggregate_by_field('vehicle_type', aggregated['experiments'])
            aggregated['summary_by_borough'] = self._aggregate_by_field('place', aggregated['experiments'])
            aggregated['summary_by_year_month'] = self._aggregate_by_field('year_month', aggregated['experiments'])
        
        # Save aggregated results
        agg_file = self.results_dir / "aggregated" / f"aggregated_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(agg_file, 'w') as f:
            json.dump(aggregated, f, indent=2, default=str)
        
        self.logger.info(f"📊 Aggregated {len(aggregated['experiments'])} experiments")
        self.logger.info(f"   Results saved to: {agg_file}")
        
        return aggregated
    
    def _calculate_experiment_summary(self, experiment_data: Dict) -> Dict:
        """Calculate summary statistics for a single experiment."""
        iterations = experiment_data.get('iterations', [])
        exp_info = experiment_data.get('parameters', {})
        
        summary = {
            'experiment_name': experiment_data.get('experiment_name', 'unknown'),
            'place': exp_info.get('place', 'unknown'),
            'day': exp_info.get('day', 0),
            'year_month': f"{exp_info.get('year', 0)}-{exp_info.get('month', 0):02d}",
            'total_iterations': len(iterations),
            'methods': {}
        }
        
        # Extract vehicle type from experiment name or file pattern
        exp_name = experiment_data.get('experiment_name', '')
        if 'PL' in exp_name:
            summary['experiment_type'] = 'PL'
        elif 'Sigmoid' in exp_name:
            summary['experiment_type'] = 'Sigmoid'
        else:
            summary['experiment_type'] = 'unknown'
        
        # Aggregate method results
        method_results = {}
        for iteration in iterations:
            results = iteration.get('results', {})
            for method, data in results.items():
                if method not in method_results:
                    method_results[method] = {'objectives': [], 'times': []}
                
                if isinstance(data, dict):
                    if 'objective_value' in data:
                        method_results[method]['objectives'].append(data['objective_value'])
                    if 'solve_time' in data or 'computation_time' in data:
                        time_val = data.get('solve_time', data.get('computation_time', 0))
                        method_results[method]['times'].append(time_val)
        
        # Calculate statistics for each method
        for method, data in method_results.items():
            objectives = data['objectives']
            times = data['times']
            
            if objectives:
                import numpy as np
                summary['methods'][method] = {
                    'avg_objective': float(np.mean(objectives)),
                    'std_objective': float(np.std(objectives)),
                    'min_objective': float(np.min(objectives)),
                    'max_objective': float(np.max(objectives)),
                    'avg_time': float(np.mean(times)) if times else 0,
                    'std_time': float(np.std(times)) if times else 0,
                    'num_iterations': len(objectives)
                }
        
        return summary
    
    def _aggregate_by_field(self, field: str, experiments: List[Dict]) -> Dict:
        """Aggregate experiments by a specific field."""
        import numpy as np
        
        aggregated = {}
        
        for exp in experiments:
            # Get field value
            if field == 'method':
                # Special case: aggregate by method across all experiments
                for method, method_data in exp.get('methods', {}).items():
                    if method not in aggregated:
                        aggregated[method] = {'objectives': [], 'times': []}
                    
                    aggregated[method]['objectives'].append(method_data['avg_objective'])
                    aggregated[method]['times'].append(method_data['avg_time'])
            else:
                # Regular field aggregation
                field_value = exp.get(field, 'unknown')
                if field_value not in aggregated:
                    aggregated[field_value] = {'experiments': [], 'objectives': [], 'times': []}
                
                aggregated[field_value]['experiments'].append(exp)
                
                # Aggregate primary method results
                methods = exp.get('methods', {})
                if methods:
                    primary_method = list(methods.keys())[0]  # Use first method as primary
                    method_data = methods[primary_method]
                    aggregated[field_value]['objectives'].append(method_data['avg_objective'])
                    aggregated[field_value]['times'].append(method_data['avg_time'])
        
        # Calculate statistics for each group
        for group, data in aggregated.items():
            if 'objectives' in data and data['objectives']:
                objectives = data['objectives']
                times = data['times']
                
                data['stats'] = {
                    'count': len(objectives),
                    'avg_objective': float(np.mean(objectives)),
                    'std_objective': float(np.std(objectives)),
                    'avg_time': float(np.mean(times)) if times else 0,
                    'std_time': float(np.std(times)) if times else 0
                }
        
        return aggregated

def main():
    """Command line interface for experiment runner."""
    parser = argparse.ArgumentParser(description='Parallel Experiment Runner')
    
    parser.add_argument('--vehicle-types', nargs='+',
                       choices=['yellow', 'green', 'fhv', 'fhvhv'],
                       default=['yellow', 'green'],
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
    
    parser.add_argument('--experiment-types', nargs='+',
                       choices=['PL', 'Sigmoid'],
                       default=['PL', 'Sigmoid'],
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
    
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of parallel workers')
    
    parser.add_argument('--aggregate-only', action='store_true',
                       help='Only aggregate existing results')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ExperimentRunner(max_workers=args.max_workers)
    
    if args.aggregate_only:
        # Only run aggregation
        runner.aggregate_results()
        return
    
    # Generate experiment configurations
    configs = runner.generate_experiment_configs(
        vehicle_types=args.vehicle_types,
        years=args.years,
        months=args.months,
        days=args.days,
        experiment_types=args.experiment_types,
        boroughs=args.boroughs
    )
    
    if not configs:
        print("❌ No experiment configurations generated")
        return
    
    print(f"🧪 Experiment Runner")
    print(f"   Configurations: {len(configs)}")
    print(f"   Max workers: {args.max_workers}")
    print()
    
    # Run experiments
    summary = runner.run_experiments_parallel(
        configs=configs,
        time_interval=args.time_interval,
        time_unit=args.time_unit,
        simulation_range=args.simulation_range
    )
    
    # Aggregate results
    if summary['successful'] > 0:
        print("\n📊 Aggregating results...")
        runner.aggregate_results()

if __name__ == "__main__":
    main() 