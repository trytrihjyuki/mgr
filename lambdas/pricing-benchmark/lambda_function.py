"""
Ride-Hailing Pricing Benchmark Lambda Function

This Lambda function implements a comprehensive experimental environment for ride-hailing pricing.
It acts as a wrapper around the core logic in pricing_logic.py.
"""

import json
import os
import sys
import logging
import traceback
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Add the parent directory to the path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from pricing_logic import PricingExperimentRunner
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"❌ Failed to import PricingExperimentRunner: {e}")
    IMPORTS_SUCCESSFUL = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Main Lambda entry point.
    Handles different invocation modes:
    - Batch processing of scenarios
    - Single scenario execution
    - LinUCB model training
    - Quick test mode
    """
    if not IMPORTS_SUCCESSFUL:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Failed to import pricing_logic module'})
        }

    # Quick test for warming up or basic validation
    if event.get('test_mode'):
        logger.info("✅ Lambda function is warm.")
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Test successful'})
        }

    # Check if this is a batch scenario request
    if 'batch_scenarios' in event and isinstance(event['batch_scenarios'], list):
        return handle_batch_scenarios(event, context)

    # Check if this is a LinUCB training request
    if event.get('train_linucb', False):
        return handle_training_request(event)

    # Default to single scenario execution
    return handle_single_scenario(event, context)


def handle_batch_scenarios(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Handles batch processing of multiple scenarios in a single Lambda invocation.
    """
    batch_results = []
    runner = PricingExperimentRunner(num_eval=event.get('num_eval', 1000))

    for scenario_event in event['batch_scenarios']:
        try:
            result_body = process_single_scenario_core(scenario_event, runner)
            status = 'error' if 'error' in result_body else 'success'
            batch_results.append({
                'scenario_id': scenario_event.get('scenario_id'),
                'status': status,
                **result_body
            })
        except Exception as e:
            batch_results.append({
                'scenario_id': scenario_event.get('scenario_id'),
                'status': 'error',
                'error': f"Unhandled exception: {e}"
            })

    return {
        'statusCode': 200,
        'body': json.dumps({'batch_mode': True, 'results': batch_results})
    }


def handle_training_request(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handles a request to train a LinUCB model."""
    try:
        runner = PricingExperimentRunner(num_eval=event.get('num_eval', 1000))
        training_result = runner.train_linucb_model(
            vehicle_type=event['vehicle_type'],
            borough=event['borough'],
            training_year=event['training_year'],
            training_month=event['training_month']
        )
        return {'statusCode': 200, 'body': json.dumps(training_result)}
    except Exception as e:
        logger.error(f"❌ LinUCB Training error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e), 'trace': traceback.format_exc()})
        }


def handle_single_scenario(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Handles a single scenario execution.
    """
    try:
        runner = PricingExperimentRunner(num_eval=event.get('num_eval', 1000))
        result_body = process_single_scenario_core(event, runner)
        return {'statusCode': 200, 'body': json.dumps(result_body)}
    except Exception as e:
        logger.error(f"❌ Unhandled exception in single scenario: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e), 'trace': traceback.format_exc()})
        }


def process_single_scenario_core(event: Dict[str, Any], runner: "PricingExperimentRunner") -> Dict[str, Any]:
    """
    Core logic to process a single scenario. This can be called directly.
    """
    start_time = time.time()
    try:
        # Extract parameters
        params = event
        year = params['year']
        month = params['month']
        day = params['day']
        borough = params['borough']
        vehicle_type = params['vehicle_type']
        methods = params['methods']
        acceptance_function = params['acceptance_function']
        time_window = params['time_window']

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
                'results': [], 'data_statistics': data_stats,
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
        s3_location = runner.save_results_to_s3(results, params, data_stats, performance_summary)

        return {
            'results': results,
            'data_statistics': data_stats,
            'performance_summary': performance_summary,
            's3_location': s3_location
        }
    except Exception as e:
        logger.error(f"❌ Scenario processing error: {e}")
        return {'error': str(e), 'trace': traceback.format_exc()}