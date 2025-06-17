"""
Benchmark utilities for ride-hailing experiments.
Handles logging, data storage, and KPI tracking.
"""

import csv
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

class BenchmarkLogger:
    """
    Handles logging and benchmarking for ride-hailing experiments.
    """
    
    def __init__(self, experiment_name: str, output_dir: str = "../results"):
        """
        Initialize benchmark logger.
        
        Args:
            experiment_name: Name of the experiment (e.g., 'PL', 'Sigmoid')
            output_dir: Directory to store results
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "detailed").mkdir(exist_ok=True)
        (self.output_dir / "summary").mkdir(exist_ok=True)
        (self.output_dir / "benchmarks").mkdir(exist_ok=True)
        
        self.setup_logging()
        self.experiment_data = []
        self.iteration_results = []
        self.kpis = {}
        
    def setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / "logs" / f"{self.experiment_name}_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(f"{self.experiment_name}_experiment")
        self.logger.info(f"Starting {self.experiment_name} experiment")
        
    def log_iteration_start(self, iteration: int, params: Dict[str, Any]):
        """Log the start of an iteration."""
        self.logger.info(f"=== Iteration {iteration + 1} ===")
        self.logger.info(f"Parameters: {params}")
        
    def log_iteration_result(self, iteration: int, results: Dict[str, Any]):
        """Log and store results from a single iteration."""
        self.logger.info(f"Iteration {iteration + 1} completed:")
        
        # Log key metrics
        for method, data in results.items():
            if isinstance(data, dict) and 'objective_value' in data:
                obj_val = data['objective_value']
                solve_time = data.get('solve_time', 0)
                self.logger.info(f"  {method}: objective={obj_val:.4f}, time={solve_time:.3f}s")
        
        # Store detailed results
        iteration_data = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        self.iteration_results.append(iteration_data)
        
    def store_detailed_results(self, place: str, day: int, time_interval: int, time_unit: str):
        """Store detailed iteration results to JSON."""
        filename = f"detailed_{self.experiment_name}_place={place}_day={day}_interval={time_interval}{time_unit}.json"
        filepath = self.output_dir / "detailed" / filename
        
        data = {
            'experiment_name': self.experiment_name,
            'parameters': {
                'place': place,
                'day': day,
                'time_interval': time_interval,
                'time_unit': time_unit
            },
            'iterations': self.iteration_results,
            'kpis': self.kpis
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        self.logger.info(f"Detailed results saved to {filepath}")
        
    def store_summary_csv(self, place: str, day: int, time_interval: int, time_unit: str, 
                         summary_stats: Dict[str, Dict[str, float]]):
        """Store summary statistics in CSV format (compatible with original format)."""
        filename = f"Average_result_{self.experiment_name}_place={place}_day={day}_interval={time_interval}{time_unit}.csv"
        filepath = self.output_dir / "summary" / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write headers
            writer.writerow(['method', 'avg_objective_value', 'avg_computation_time'])
            
            # Write data for each method
            for method, stats in summary_stats.items():
                writer.writerow([
                    method,
                    stats.get('avg_objective_value', 0),
                    stats.get('avg_computation_time', 0)
                ])
                
        self.logger.info(f"Summary results saved to {filepath}")
        
    def calculate_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics across all iterations."""
        if not self.iteration_results:
            return {}
            
        # Aggregate results by method
        method_results = {}
        
        for iteration_data in self.iteration_results:
            results = iteration_data['results']
            for method, data in results.items():
                if method not in method_results:
                    method_results[method] = {
                        'objective_values': [],
                        'computation_times': []
                    }
                
                if isinstance(data, dict):
                    if 'objective_value' in data:
                        method_results[method]['objective_values'].append(data['objective_value'])
                    if 'solve_time' in data or 'computation_time' in data:
                        time_val = data.get('solve_time', data.get('computation_time', 0))
                        method_results[method]['computation_times'].append(time_val)
        
        # Calculate statistics
        summary_stats = {}
        for method, data in method_results.items():
            obj_vals = data['objective_values']
            comp_times = data['computation_times']
            
            summary_stats[method] = {
                'avg_objective_value': np.mean(obj_vals) if obj_vals else 0,
                'std_objective_value': np.std(obj_vals) if obj_vals else 0,
                'avg_computation_time': np.mean(comp_times) if comp_times else 0,
                'std_computation_time': np.std(comp_times) if comp_times else 0,
                'num_iterations': len(obj_vals)
            }
            
        return summary_stats
        
    def store_benchmarks(self, benchmarks: Dict[str, Any]):
        """Store benchmark data for analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmarks_{self.experiment_name}_{timestamp}.json"
        filepath = self.output_dir / "benchmarks" / filename
        
        with open(filepath, 'w') as f:
            json.dump(benchmarks, f, indent=2, default=str)
            
        self.logger.info(f"Benchmarks saved to {filepath}")
        
    def add_kpi(self, name: str, value: Any):
        """Add a KPI to track."""
        self.kpis[name] = value
        self.logger.info(f"KPI {name}: {value}")
        
    def finalize_experiment(self, place: str, day: int, time_interval: int, time_unit: str):
        """Finalize the experiment and save all results."""
        # Calculate summary statistics
        summary_stats = self.calculate_summary_statistics()
        
        # Store results
        self.store_summary_csv(place, day, time_interval, time_unit, summary_stats)
        self.store_detailed_results(place, day, time_interval, time_unit)
        
        # Log final summary
        self.logger.info("=== Experiment Summary ===")
        for method, stats in summary_stats.items():
            self.logger.info(f"{method}:")
            self.logger.info(f"  Avg Objective: {stats['avg_objective_value']:.4f} ± {stats['std_objective_value']:.4f}")
            self.logger.info(f"  Avg Time: {stats['avg_computation_time']:.3f}s ± {stats['std_computation_time']:.3f}s")
            self.logger.info(f"  Iterations: {stats['num_iterations']}")
            
        self.logger.info(f"All results saved to {self.output_dir}")

class ExperimentTimer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.name}")
        return self
        
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        self.logger.debug(f"Completed {self.name} in {elapsed:.3f}s")
        
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0
        end = self.end_time or time.time()
        return end - self.start_time

def validate_results(results: Dict[str, Any]) -> bool:
    """Validate that results contain expected fields."""
    required_fields = ['objective_value', 'solve_time']
    
    for method, data in results.items():
        if not isinstance(data, dict):
            continue
            
        for field in required_fields:
            if field not in data:
                logging.warning(f"Missing field '{field}' in results for method '{method}'")
                return False
                
    return True

def create_benchmark_summary(results_dir: str = "../results") -> Dict[str, Any]:
    """Create a summary of all benchmark results."""
    results_path = Path(results_dir)
    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiments': []
    }
    
    # Scan for detailed result files
    detailed_dir = results_path / "detailed"
    if detailed_dir.exists():
        for file_path in detailed_dir.glob("detailed_*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    summary['experiments'].append({
                        'file': file_path.name,
                        'experiment_name': data.get('experiment_name'),
                        'parameters': data.get('parameters'),
                        'num_iterations': len(data.get('iterations', [])),
                        'kpis': data.get('kpis', {})
                    })
            except Exception as e:
                logging.warning(f"Could not load {file_path}: {e}")
                
    return summary 