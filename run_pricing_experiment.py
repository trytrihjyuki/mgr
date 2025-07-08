#!/usr/bin/env python3
"""
UNIFIED Ride-Hailing Pricing Experiments Runner (EC2 Version)
============================================================

This script runs pricing experiments on a single machine, leveraging
multiprocessing to parallelize the workload. It is a simplified
version of the original Lambda-based runner.
"""

import json
import time
import logging
import argparse
import signal
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Tuple
import random
import sys
import pandas as pd
import numpy as np
import io
import os
import boto3

from pricing_logic import PricingExperimentRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'experiment_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

# Global shutdown flag
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    if not shutdown_requested:
        print(f"\n🛑 Shutdown requested (signal {signum}). Stopping execution...")
        logging.warning(f"Shutdown requested via signal {signum}")
        shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)

def run_scenario(scenario_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper function to run a single scenario.
    This function is designed to be called by a multiprocessing Pool.
    """
    if shutdown_requested:
        return {'scenario_id': scenario_params['scenario_id'], 'success': False, 'error': 'Shutdown requested'}

    try:
        runner = PricingExperimentRunner(num_eval=scenario_params.get('num_eval', 1000))
        
        # This is the core logic that was previously in the Lambda function
        # We can call it directly now
        # Note: a simplified version of `process_single_scenario_core` is used here
        
        start_time = time.time()
        
        # Extract parameters
        year = scenario_params['year']
        month = scenario_params['month']
        day = scenario_params['day']
        borough = scenario_params['borough']
        vehicle_type = scenario_params['vehicle_type']
        methods = scenario_params['methods']
        acceptance_function = scenario_params['acceptance_function']
        time_window = scenario_params['time_window']

        # Create datetime objects for time window
        time_start = datetime(year, month, day, time_window['hour_start'], time_window['minute_start'])
        time_end = time_start + timedelta(minutes=time_window['time_interval'])

        # Load data
        requesters_df, taxis_df = runner.load_taxi_data(
            vehicle_type, year, month, day, borough, time_start, time_end
        )
        
        data_stats = {
            'num_requesters': len(requesters_df),
            'num_taxis': len(taxis_df)
        }

        if requesters_df.empty or taxis_df.empty:
            return {
                'scenario_id': scenario_params['scenario_id'],
                'success': True,
                'results': [], 
                'data_statistics': data_stats,
                'performance_summary': {'total_runtime': time.time() - start_time}
            }
        
        # Calculate distance matrix and edge weights
        distance_matrix, edge_weights = runner.calculate_distance_matrix_and_edge_weights(requesters_df, taxis_df)

        # Run specified methods
        results = []
        for method in methods:
            if method == 'LP':
                result = runner.run_lp(requesters_df, taxis_df, distance_matrix, edge_weights, acceptance_function)
            elif method == 'MinMaxCostFlow':
                result = runner.run_minmaxcostflow(requesters_df, taxis_df, distance_matrix, edge_weights, acceptance_function)
            elif method == 'MAPS':
                result = runner.run_maps(requesters_df, taxis_df, distance_matrix, edge_weights, acceptance_function)
            elif method == 'LinUCB':
                result = runner.run_linucb(requesters_df, taxis_df, distance_matrix, edge_weights, acceptance_function, borough, vehicle_type, time_start.hour)
            else:
                result = {'method_name': method, 'error': 'Unknown method'}
            results.append(result)

        total_runtime = time.time() - start_time
        performance_summary = {'total_runtime': total_runtime}
        
        # Save results to S3
        s3_location = runner.save_results_to_s3(results, scenario_params, data_stats, performance_summary)

        return {
            'scenario_id': scenario_params['scenario_id'],
            'success': True,
            'results': results,
            'data_statistics': data_stats,
            'performance_summary': performance_summary,
            's3_location': s3_location
        }

    except Exception as e:
        logging.error(f"Error in scenario {scenario_params.get('scenario_id', 'unknown')}: {e}", exc_info=True)
        return {'scenario_id': scenario_params.get('scenario_id', 'unknown'), 'success': False, 'error': str(e)}


def create_scenario_parameters(args, training_id: str) -> List[Dict[str, Any]]:
    """Generate a list of scenario parameters based on the input arguments."""
    # This function is mostly the same as the original, so I'll keep it as is.
    # ... (original implementation of create_scenario_parameters)
    scenarios = []
    days = []
    if args.days:
        days = [int(d) for d in args.days.split(',')]
    elif args.start_day and args.end_day:
        days = range(args.start_day, args.end_day + 1)
    elif args.days_modulo:
        modulo, remainder = [int(d) for d in args.days_modulo.split(',')]
        total_days = args.total_days if args.total_days else 31
        days = [d for d in range(1, total_days + 1) if d % modulo == remainder]
    else:
        days = [args.day] if args.day else []

    time_unit_multiplier = 60 if args.time_unit == 'm' else 1
    total_duration_seconds = (args.hour_end - args.hour_start) * 3600
    interval_seconds = args.time_interval * time_unit_multiplier
    num_scenarios = total_duration_seconds // interval_seconds

    eval_functions = args.eval.split(',')
    methods = args.methods.split(',')

    for day in days:
        for eval_func in eval_functions:
            for i in range(num_scenarios):
                total_seconds_from_start = i * interval_seconds
                hour = args.hour_start + (total_seconds_from_start // 3600)
                minute = (total_seconds_from_start % 3600) // 60
                
                scenario_id = f"day{day:02d}_{eval_func}_s{i:03d}"
                
                scenarios.append({
                    'scenario_id': scenario_id,
                    'year': args.year,
                    'month': args.month,
                    'day': day,
                    'borough': args.borough,
                    'vehicle_type': args.vehicle_type,
                    'eval': eval_func,
                    'methods': methods,
                    'acceptance_function': eval_func,
                    'time_window': {
                        'hour_start': hour,
                        'minute_start': minute,
                        'time_interval': args.time_interval,
                    },
                    'num_eval': args.num_eval,
                    'training_id': training_id,
                    'execution_date': datetime.now().strftime('%Y%m%d')
                })
    return scenarios

def main():
    """Main function to run the experiments."""
    parser = argparse.ArgumentParser(description="Ride-Hailing Pricing Experiment Runner (EC2 Version)")
    # Add all the arguments from the original script
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--month", type=int, default=10)
    parser.add_argument("--day", type=int, help="A single day to run.")
    parser.add_argument("--days", type=str, help="A comma-separated list of specific days to run.")
    parser.add_argument("--start_day", type=int, help="The start day of a range to run.")
    parser.add_argument("--end_day", type=int, help="The end day of a range to run.")
    parser.add_argument("--days_modulo", type=str, help="A 'modulo,remainder' pair to select days.")
    parser.add_argument("--total_days", type=int, help="Total number of days in the month for modulo calculation.")
    parser.add_argument("--borough", type=str, default="Manhattan")
    parser.add_argument("--vehicle_type", type=str, default="yellow")
    parser.add_argument("--eval", type=str, default="PL,Sigmoid")
    parser.add_argument("--methods", type=str, default="LP,MinMaxCostFlow,LinUCB,MAPS")
    parser.add_argument("--hour_start", type=int, default=10)
    parser.add_argument("--hour_end", type=int, default=20)
    parser.add_argument("--time_interval", type=int, default=5)
    parser.add_argument("--time_unit", type=str, default='m', choices=['m', 's'])
    parser.add_argument("--num_eval", type=int, default=1000)
    parser.add_argument("--parallel", type=int, default=cpu_count(), help="Number of parallel processes to use.")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--experiment-id", type=str, help="A unique ID for the experiment run.")
    
    args = parser.parse_args()

    training_id = args.experiment_id or "ec2_experiment_" + datetime.now().strftime('%Y%m%d_%H%M%S')
    
    scenarios = create_scenario_parameters(args, training_id)
    
    if not scenarios:
        logging.error("No scenarios generated. Please check your day selection parameters.")
        return

    logging.info(f"Generated {len(scenarios)} scenarios to run.")
    
    start_time = time.time()
    
    # --- Progress Reporting Setup ---
    total_scenarios = len(scenarios)
    processed_scenarios = 0
    runner = PricingExperimentRunner()
    # Write initial progress file
    runner.save_progress_to_s3(training_id, 0, total_scenarios)
    last_progress_update = time.time()
    
    results = []
    with Pool(processes=args.parallel) as pool:
        # Use imap_unordered to get results as they complete
        for result in pool.imap_unordered(run_scenario, scenarios):
            results.append(result)
            processed_scenarios += 1
            
            # Update progress file periodically (e.g., every 5 seconds) to avoid too many S3 writes
            if time.time() - last_progress_update > 5:
                logging.info(f"Progress: {processed_scenarios}/{total_scenarios} scenarios completed.")
                runner.save_progress_to_s3(training_id, processed_scenarios, total_scenarios)
                last_progress_update = time.time()

    end_time = time.time()
    
    # Final progress update
    runner.save_progress_to_s3(training_id, processed_scenarios, total_scenarios)
    logging.info(f"Finished processing {len(results)} scenarios in {end_time - start_time:.2f} seconds.")
    
    # Process results
    success_count = sum(1 for r in results if r.get('success'))
    error_count = len(results) - success_count
    
    logging.info(f"✅ Success: {success_count}, ❌ Errors: {error_count}")

    # Create a single _SUCCESS file if all scenarios were successful
    if error_count == 0 and success_count > 0:
        try:
            s3_client = boto3.client('s3')
            bucket_name = os.environ.get('S3_BUCKET', 'magisterka')
            
            # The _SUCCESS file is placed at the root of the experiment directory
            success_key = f"experiments/{training_id}/_SUCCESS"
            
            s3_client.put_object(
                Bucket=bucket_name,
                Key=success_key,
                Body=b'',
                ContentType='application/text'
            )
            logging.info(f"✅ Successfully created global _SUCCESS file at s3://{bucket_name}/{success_key}")

        except Exception as e:
            logging.error(f"❌ Failed to create global _SUCCESS file: {e}")

    # Optionally, aggregate and save final results
    # (The aggregation logic can be adapted from the original script if needed)

if __name__ == "__main__":
    main() 