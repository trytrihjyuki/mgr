#!/usr/bin/env python3
"""
Benchmark Orchestrator
Main system for systematic benchmarking and comparison of the 4 taxi pricing methods
"""

import numpy as np
import pandas as pd
import json
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.experiment_config import ExperimentConfig
from pricing_methods import (
    HikimaMinMaxCostFlowMethod, HikimaResult,
    MAPSMethod, MAPSResult,
    LinUCBMethod, LinUCBResult,
    LinearProgramMethod, LinearProgramResult
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Complete benchmark results for all methods"""
    experiment_id: str
    experiment_config: Dict[str, Any]
    
    # Individual method results
    hikima_result: HikimaResult = None
    maps_result: MAPSResult = None
    linucb_result: LinUCBResult = None
    linear_program_result: LinearProgramResult = None
    
    # Comparative metrics
    objective_values: Dict[str, float] = None
    computation_times: Dict[str, float] = None
    convergence_info: Dict[str, int] = None
    
    # Statistical analysis
    statistical_tests: Dict[str, Any] = None
    performance_ranking: List[str] = None
    
    # Metadata
    total_computation_time: float = 0.0
    timestamp: str = ""
    data_characteristics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.objective_values is None:
            self.objective_values = {}
        if self.computation_times is None:
            self.computation_times = {}
        if self.convergence_info is None:
            self.convergence_info = {}
        if self.statistical_tests is None:
            self.statistical_tests = {}
        if self.performance_ranking is None:
            self.performance_ranking = []
        if self.data_characteristics is None:
            self.data_characteristics = {}


class DataProcessor:
    """Handles data loading and preprocessing for experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.s3_client = boto3.client('s3') if config.aws_config else None
    
    def load_data_from_s3(self, vehicle_type: str, year: int, month: int) -> pd.DataFrame:
        """Load taxi data from S3"""
        if not self.s3_client:
            raise ValueError("S3 client not configured")
        
        bucket = self.config.aws_config.s3_bucket
        key = f"{self.config.aws_config.s3_data_prefix}/{vehicle_type}/year={year}/month={month:02d}/{vehicle_type}_tripdata_{year}-{month:02d}.parquet"
        
        try:
            logger.info(f"Loading {vehicle_type} data from S3: {key}")
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            df = pd.read_parquet(response['Body'])
            logger.info(f"✅ Loaded {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to load {key}: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data into format suitable for all pricing methods
        
        Args:
            df: Raw taxi data DataFrame
            
        Returns:
            Tuple of (requesters_data, taxis_data) as numpy arrays
        """
        # Filter data based on configuration
        df = df.dropna(subset=['pickup_datetime', 'dropoff_datetime'])
        
        # Apply time filtering (user-configurable, no hardcoded rush hours)
        if 'pickup_datetime' in df.columns:
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
            df['hour'] = df['pickup_datetime'].dt.hour
            
            time_config = self.config.time_config
            df = df[
                (df['hour'] >= time_config.start_hour) & 
                (df['hour'] < time_config.end_hour)
            ]
        
        # Data quality filters (configurable)
        data_config = self.config.data_config
        df = df[
            (df.get('trip_distance', 0) > data_config.min_trip_distance) &
            (df.get('total_amount', 0) > data_config.min_total_amount)
        ]
        
        # Convert distance to km
        if 'trip_distance' in df.columns:
            df['trip_distance_km'] = df['trip_distance'] * data_config.distance_conversion_factor
        
        # Sample data for reasonable computation time
        sample_size = min(len(df), 5000)  # Configurable
        df_sample = df.sample(n=sample_size, random_state=42)
        
        # Prepare requesters data: [borough, location_id, trip_distance, total_amount, destination_id, duration]
        requesters_data = []
        for _, row in df_sample.iterrows():
            requester = [
                0,  # borough (simplified)
                row.get('PULocationID', 161),  # pickup location
                row.get('trip_distance_km', row.get('trip_distance', 1.0)),  # distance in km
                row.get('total_amount', 10.0),  # fare amount
                row.get('DOLocationID', row.get('PULocationID', 161)),  # dropoff location
                300.0  # duration (simplified to 5 minutes)
            ]
            requesters_data.append(requester)
        
        # Prepare taxis data: [location_id, available]
        # For simulation, create taxis based on dropoff locations
        taxis_data = []
        for _, row in df_sample.iterrows():
            taxi = [
                row.get('DOLocationID', row.get('PULocationID', 161)),  # taxi location
                1  # available
            ]
            taxis_data.append(taxi)
        
        return np.array(requesters_data), np.array(taxis_data)


class BenchmarkOrchestrator:
    """Main orchestrator for systematic benchmarking of pricing methods"""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize orchestrator with experiment configuration
        
        Args:
            config: Complete experiment configuration
        """
        self.config = config
        self.data_processor = DataProcessor(config)
        
        # Initialize pricing methods
        self.methods = {}
        
        if "HikimaMinMaxCostFlow" in config.methods_to_run:
            self.methods["HikimaMinMaxCostFlow"] = HikimaMinMaxCostFlowMethod(
                asdict(config.hikima_config)
            )
        
        if "MAPS" in config.methods_to_run:
            self.methods["MAPS"] = MAPSMethod(
                asdict(config.maps_config)
            )
        
        if "LinUCB" in config.methods_to_run:
            self.methods["LinUCB"] = LinUCBMethod(
                asdict(config.linucb_config)
            )
        
        if "LinearProgram" in config.methods_to_run:
            self.methods["LinearProgram"] = LinearProgramMethod(
                asdict(config.lp_config)
            )
        
        logger.info(f"Initialized orchestrator with {len(self.methods)} methods: {list(self.methods.keys())}")
    
    def run_single_method(self, method_name: str, requesters_data: np.ndarray, 
                         taxis_data: np.ndarray) -> Tuple[str, Any, float]:
        """
        Run a single pricing method
        
        Args:
            method_name: Name of the method to run
            requesters_data: Preprocessed requester data
            taxis_data: Preprocessed taxi data
            
        Returns:
            Tuple of (method_name, result, computation_time)
        """
        if method_name not in self.methods:
            raise ValueError(f"Method {method_name} not available")
        
        logger.info(f"🏃 Running {method_name} method...")
        start_time = time.time()
        
        try:
            method = self.methods[method_name]
            result = method.solve(requesters_data, taxis_data)
            computation_time = time.time() - start_time
            
            logger.info(f"✅ {method_name} completed in {computation_time:.3f}s, objective: {result.objective_value:.2f}")
            return method_name, result, computation_time
            
        except Exception as e:
            computation_time = time.time() - start_time
            logger.error(f"❌ {method_name} failed after {computation_time:.3f}s: {e}")
            
            # Return a failed result object
            if method_name == "HikimaMinMaxCostFlow":
                result = HikimaResult(objective_value=0.0, computation_time=computation_time)
            elif method_name == "MAPS":
                result = MAPSResult(objective_value=0.0, computation_time=computation_time)
            elif method_name == "LinUCB":
                result = LinUCBResult(objective_value=0.0, computation_time=computation_time)
            elif method_name == "LinearProgram":
                result = LinearProgramResult(objective_value=0.0, computation_time=computation_time)
            else:
                result = None
            
            return method_name, result, computation_time
    
    def run_parallel_benchmark(self, requesters_data: np.ndarray, 
                              taxis_data: np.ndarray) -> Dict[str, Tuple[Any, float]]:
        """
        Run all methods in parallel for efficient benchmarking
        
        Args:
            requesters_data: Preprocessed requester data
            taxis_data: Preprocessed taxi data
            
        Returns:
            Dictionary mapping method names to (result, computation_time) tuples
        """
        results = {}
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=len(self.methods)) as executor:
            # Submit all methods
            future_to_method = {
                executor.submit(self.run_single_method, method_name, requesters_data, taxis_data): method_name
                for method_name in self.methods.keys()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_method):
                method_name = future_to_method[future]
                try:
                    method_name_result, result, computation_time = future.result()
                    results[method_name] = (result, computation_time)
                except Exception as e:
                    logger.error(f"Method {method_name} failed with exception: {e}")
                    results[method_name] = (None, 0.0)
        
        return results
    
    def calculate_comparative_metrics(self, method_results: Dict[str, Tuple[Any, float]]) -> Dict[str, Any]:
        """
        Calculate comparative metrics across all methods
        
        Args:
            method_results: Dictionary of method results
            
        Returns:
            Dictionary of comparative metrics
        """
        metrics = {
            'objective_values': {},
            'computation_times': {},
            'convergence_iterations': {},
            'acceptance_rates': {},
            'matched_pairs_count': {}
        }
        
        for method_name, (result, comp_time) in method_results.items():
            if result is None:
                metrics['objective_values'][method_name] = 0.0
                metrics['computation_times'][method_name] = comp_time
                metrics['convergence_iterations'][method_name] = 0
                metrics['acceptance_rates'][method_name] = 0.0
                metrics['matched_pairs_count'][method_name] = 0
                continue
            
            metrics['objective_values'][method_name] = result.objective_value
            metrics['computation_times'][method_name] = comp_time
            
            # Method-specific metrics
            if hasattr(result, 'convergence_iterations'):
                metrics['convergence_iterations'][method_name] = result.convergence_iterations
            
            if hasattr(result, 'acceptance_probabilities') and result.acceptance_probabilities is not None:
                metrics['acceptance_rates'][method_name] = np.mean(result.acceptance_probabilities)
            
            if hasattr(result, 'matched_pairs') and result.matched_pairs is not None:
                metrics['matched_pairs_count'][method_name] = len(result.matched_pairs)
        
        return metrics
    
    def perform_statistical_analysis(self, method_results: Dict[str, Tuple[Any, float]]) -> Dict[str, Any]:
        """
        Perform statistical analysis and significance testing
        
        Args:
            method_results: Dictionary of method results
            
        Returns:
            Statistical analysis results
        """
        analysis = {
            'performance_ranking': [],
            'objective_value_stats': {},
            'efficiency_analysis': {}
        }
        
        # Rank methods by objective value
        objective_values = {
            name: result[0].objective_value if result[0] else 0.0
            for name, result in method_results.items()
        }
        
        analysis['performance_ranking'] = sorted(
            objective_values.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Calculate relative performance
        max_objective = max(objective_values.values()) if objective_values.values() else 1.0
        analysis['relative_performance'] = {
            name: (value / max_objective * 100) if max_objective > 0 else 0.0
            for name, value in objective_values.items()
        }
        
        # Efficiency analysis (objective per computation time)
        computation_times = {
            name: result[1] for name, result in method_results.items()
        }
        
        analysis['efficiency_scores'] = {
            name: (objective_values[name] / max(computation_times[name], 0.001))
            for name in objective_values.keys()
        }
        
        return analysis
    
    def run_benchmark(self, vehicle_type: str = "green", year: int = 2019, 
                     month: int = 10) -> BenchmarkResult:
        """
        Run complete benchmark of all pricing methods
        
        Args:
            vehicle_type: Type of vehicle data to use
            year: Year of data
            month: Month of data
            
        Returns:
            Complete benchmark results
        """
        experiment_id = f"benchmark_{vehicle_type}_{year}_{month:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🚀 Starting benchmark experiment: {experiment_id}")
        logger.info(f"📊 Methods to benchmark: {list(self.methods.keys())}")
        
        total_start_time = time.time()
        
        try:
            # Load and preprocess data
            df = self.data_processor.load_data_from_s3(vehicle_type, year, month)
            requesters_data, taxis_data = self.data_processor.preprocess_data(df)
            
            logger.info(f"📈 Data prepared: {len(requesters_data)} requesters, {len(taxis_data)} taxis")
            
            # Run all methods in parallel
            method_results = self.run_parallel_benchmark(requesters_data, taxis_data)
            
            # Calculate comparative metrics
            comparative_metrics = self.calculate_comparative_metrics(method_results)
            
            # Perform statistical analysis
            statistical_analysis = self.perform_statistical_analysis(method_results)
            
            total_computation_time = time.time() - total_start_time
            
            # Create comprehensive benchmark result
            benchmark_result = BenchmarkResult(
                experiment_id=experiment_id,
                experiment_config=self.config.to_dict(),
                total_computation_time=total_computation_time,
                timestamp=datetime.now().isoformat(),
                data_characteristics={
                    'vehicle_type': vehicle_type,
                    'year': year,
                    'month': month,
                    'n_requesters': len(requesters_data),
                    'n_taxis': len(taxis_data),
                    'time_range': f"{self.config.time_config.start_hour:02d}:00-{self.config.time_config.end_hour:02d}:00"
                }
            )
            
            # Store individual method results
            for method_name, (result, comp_time) in method_results.items():
                if method_name == "HikimaMinMaxCostFlow":
                    benchmark_result.hikima_result = result
                elif method_name == "MAPS":
                    benchmark_result.maps_result = result
                elif method_name == "LinUCB":
                    benchmark_result.linucb_result = result
                elif method_name == "LinearProgram":
                    benchmark_result.linear_program_result = result
            
            # Store comparative metrics
            benchmark_result.objective_values = comparative_metrics['objective_values']
            benchmark_result.computation_times = comparative_metrics['computation_times']
            benchmark_result.convergence_info = comparative_metrics['convergence_iterations']
            
            # Store statistical analysis
            benchmark_result.statistical_tests = statistical_analysis
            benchmark_result.performance_ranking = [name for name, _ in statistical_analysis['performance_ranking']]
            
            logger.info(f"🎉 Benchmark completed in {total_computation_time:.3f}s")
            logger.info(f"🏆 Performance ranking: {benchmark_result.performance_ranking}")
            
            # Save results to S3 if configured
            if self.config.aws_config:
                self.save_results_to_s3(benchmark_result)
            
            return benchmark_result
            
        except Exception as e:
            total_computation_time = time.time() - total_start_time
            logger.error(f"❌ Benchmark failed after {total_computation_time:.3f}s: {e}")
            
            # Return error result
            error_result = BenchmarkResult(
                experiment_id=experiment_id,
                experiment_config=self.config.to_dict(),
                total_computation_time=total_computation_time,
                timestamp=datetime.now().isoformat()
            )
            error_result.objective_values = {"error": str(e)}
            
            return error_result
    
    def save_results_to_s3(self, result: BenchmarkResult):
        """Save benchmark results to S3"""
        if not hasattr(self, 's3_client'):
            self.s3_client = boto3.client('s3')
        
        try:
            bucket = self.config.aws_config.s3_bucket
            key = f"{self.config.aws_config.s3_results_prefix}/benchmarks/{result.experiment_id}.json"
            
            # Convert result to JSON-serializable format
            result_dict = asdict(result)
            
            # Handle numpy arrays in results
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return obj
            
            # Recursively convert numpy objects
            def recursive_convert(obj):
                if isinstance(obj, dict):
                    return {k: recursive_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_convert(v) for v in obj]
                else:
                    return convert_numpy(obj)
            
            result_dict = recursive_convert(result_dict)
            
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(result_dict, indent=2, default=str),
                ContentType='application/json'
            )
            
            logger.info(f"📤 Results saved to s3://{bucket}/{key}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results to S3: {e}")


# CLI Interface Functions
def run_hikima_replication_experiment():
    """Run experiment that replicates Hikima paper setup"""
    from config.experiment_config import create_hikima_replication_config
    
    config = create_hikima_replication_config()
    orchestrator = BenchmarkOrchestrator(config)
    
    # Run for 2 days as in original Hikima setup
    results = []
    for day in [1, 2]:  # Two days as in original paper
        logger.info(f"Running Hikima replication experiment - Day {day}")
        result = orchestrator.run_benchmark(vehicle_type="green", year=2019, month=10)
        results.append(result)
    
    return results


def run_extended_benchmark_experiment(days: int = 30):
    """Run extended benchmarking experiment over multiple days"""
    from config.experiment_config import create_extended_benchmark_config
    
    config = create_extended_benchmark_config(days)
    orchestrator = BenchmarkOrchestrator(config)
    
    result = orchestrator.run_benchmark(vehicle_type="green", year=2019, month=10)
    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if len(sys.argv) < 2:
        print("Usage: python benchmark_orchestrator.py [hikima_replication|extended_benchmark]")
        sys.exit(1)
    
    experiment_type = sys.argv[1]
    
    if experiment_type == "hikima_replication":
        results = run_hikima_replication_experiment()
        print(f"Completed Hikima replication with {len(results)} day results")
    elif experiment_type == "extended_benchmark":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        result = run_extended_benchmark_experiment(days)
        print(f"Completed extended benchmark over {days} days")
    else:
        print("Unknown experiment type. Use 'hikima_replication' or 'extended_benchmark'")
        sys.exit(1) 