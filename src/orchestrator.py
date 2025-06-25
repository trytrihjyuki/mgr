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

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Complete benchmark results for all methods"""
    experiment_id: str
    experiment_config: Dict[str, Any]
    
    # Individual method results
    hikima_result: Any = None
    maps_result: Any = None
    linucb_result: Any = None
    linear_program_result: Any = None
    
    # Comparative metrics
    objective_values: Dict[str, float] = None
    computation_times: Dict[str, float] = None
    
    # Metadata
    total_computation_time: float = 0.0
    timestamp: str = ""
    
    def __post_init__(self):
        if self.objective_values is None:
            self.objective_values = {}
        if self.computation_times is None:
            self.computation_times = {}


class BenchmarkOrchestrator:
    """Main orchestrator for systematic benchmarking of pricing methods"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize orchestrator with experiment configuration"""
        self.config = config
        
        # Initialize pricing methods
        self.methods = {}
        
        # Import and initialize methods based on configuration
        from pricing_methods.hikima_method import HikimaMinMaxCostFlowMethod
        from pricing_methods.maps_method import MAPSMethod
        from pricing_methods.linucb_method import LinUCBMethod
        from pricing_methods.linear_program_method import LinearProgramMethod
        
        if "HikimaMinMaxCostFlow" in config.get('methods_to_run', []):
            self.methods["HikimaMinMaxCostFlow"] = HikimaMinMaxCostFlowMethod(config.get('hikima_config', {}))
        
        if "MAPS" in config.get('methods_to_run', []):
            self.methods["MAPS"] = MAPSMethod(config.get('maps_config', {}))
        
        if "LinUCB" in config.get('methods_to_run', []):
            self.methods["LinUCB"] = LinUCBMethod(config.get('linucb_config', {}))
        
        if "LinearProgram" in config.get('methods_to_run', []):
            self.methods["LinearProgram"] = LinearProgramMethod(config.get('lp_config', {}))
        
        logger.info(f"Initialized orchestrator with {len(self.methods)} methods: {list(self.methods.keys())}")
    
    def run_benchmark(self, requesters_data: np.ndarray, taxis_data: np.ndarray) -> BenchmarkResult:
        """
        Run complete benchmark of all pricing methods
        
        Args:
            requesters_data: Preprocessed requester data
            taxis_data: Preprocessed taxi data
            
        Returns:
            Complete benchmark results
        """
        experiment_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🚀 Starting benchmark experiment: {experiment_id}")
        logger.info(f"📊 Methods to benchmark: {list(self.methods.keys())}")
        
        total_start_time = time.time()
        results = {}
        computation_times = {}
        
        # Run each method
        for method_name, method in self.methods.items():
            logger.info(f"🏃 Running {method_name} method...")
            
            try:
                start_time = time.time()
                result = method.solve(requesters_data, taxis_data)
                comp_time = time.time() - start_time
                
                results[method_name] = result
                computation_times[method_name] = comp_time
                
                logger.info(f"✅ {method_name} completed in {comp_time:.3f}s, objective: {result.objective_value:.2f}")
                
            except Exception as e:
                comp_time = time.time() - start_time
                logger.error(f"❌ {method_name} failed after {comp_time:.3f}s: {e}")
                results[method_name] = None
                computation_times[method_name] = comp_time
        
        total_computation_time = time.time() - total_start_time
        
        # Extract objective values
        objective_values = {}
        for method_name, result in results.items():
            if result is not None:
                objective_values[method_name] = result.objective_value
            else:
                objective_values[method_name] = 0.0
        
        # Create benchmark result
        benchmark_result = BenchmarkResult(
            experiment_id=experiment_id,
            experiment_config=self.config,
            total_computation_time=total_computation_time,
            timestamp=datetime.now().isoformat(),
            objective_values=objective_values,
            computation_times=computation_times
        )
        
        # Store individual results
        for method_name, result in results.items():
            if method_name == "HikimaMinMaxCostFlow":
                benchmark_result.hikima_result = result
            elif method_name == "MAPS":
                benchmark_result.maps_result = result
            elif method_name == "LinUCB":
                benchmark_result.linucb_result = result
            elif method_name == "LinearProgram":
                benchmark_result.linear_program_result = result
        
        logger.info(f"🎉 Benchmark completed in {total_computation_time:.3f}s")
        
        # Print results summary
        print("\n📈 BENCHMARK RESULTS SUMMARY:")
        print("=" * 50)
        for method_name in sorted(objective_values.keys(), key=lambda x: objective_values[x], reverse=True):
            obj_val = objective_values[method_name]
            comp_time = computation_times[method_name]
            print(f"{method_name:20s}: Objective={obj_val:8.2f}, Time={comp_time:6.3f}s")
        
        return benchmark_result 