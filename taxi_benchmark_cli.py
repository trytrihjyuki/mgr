#!/usr/bin/env python3
"""
Taxi Benchmark CLI
==================

Command-line interface for running taxi pricing experiments on AWS Lambda.

Usage:
    python taxi_benchmark_cli.py --start-date 2019-10-06 --end-date 2019-10-12 \
        --vehicle-type yellow --borough Manhattan --method MinMaxCostFlow \
        --eval Sigmoid --num-iter 100 --start-hour 0 --end-hour 23 \
        --time-delta 5m --lambda-size L
"""

import argparse
import json
import sys
import time
import boto3
from datetime import datetime
import pandas as pd


def validate_args(args):
    """Validate command-line arguments."""
    
    # Validate dates
    try:
        start_date = pd.to_datetime(args.start_date)
        end_date = pd.to_datetime(args.end_date)
        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date")
    except:
        raise ValueError("Invalid date format. Use YYYY-MM-DD")
    
    # Validate vehicle type
    valid_vehicles = ["yellow", "green", "fhv"]
    if args.vehicle_type not in valid_vehicles:
        raise ValueError(f"Vehicle type must be one of {valid_vehicles}")
    
    # Validate borough
    valid_boroughs = ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island"]
    if args.borough not in valid_boroughs:
        raise ValueError(f"Borough must be one of {valid_boroughs}")
    
    # Validate method
    valid_methods = ["MinMaxCostFlow", "MAPS", "LinUCB", "LP"]
    if args.method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")
    
    # Validate acceptance function
    valid_evals = ["PL", "Sigmoid"]
    if args.eval not in valid_evals:
        raise ValueError(f"Eval must be one of {valid_evals}")
    
    # Validate hours
    if not (0 <= args.start_hour <= 23):
        raise ValueError("Start hour must be between 0 and 23")
    if not (0 <= args.end_hour <= 23):
        raise ValueError("End hour must be between 0 and 23")
    if args.start_hour >= args.end_hour:
        raise ValueError("Start hour must be less than end hour")
    
    # Validate lambda size
    valid_sizes = ["S", "M", "L", "XL"]
    if args.lambda_size not in valid_sizes:
        raise ValueError(f"Lambda size must be one of {valid_sizes}")


def check_data_availability(args):
    """Check if data exists for the requested experiment."""
    
    print("Checking data availability...")
    
    s3_client = boto3.client("s3")
    bucket = "magisterka"
    
    # Check for area information
    try:
        s3_client.head_object(Bucket=bucket, Key="area_information.csv")
        print("✓ Area information found")
    except:
        print("✗ Area information not found")
        return False
    
    # Check for data files
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    
    months_to_check = set()
    current = start_date
    while current <= end_date:
        months_to_check.add((current.year, current.month))
        current += pd.DateOffset(months=1)
    
    all_available = True
    for year, month in months_to_check:
        key = f"datasets/{args.vehicle_type}/year={year}/month={month:02d}/{args.vehicle_type}_tripdata_{year}-{month:02d}.parquet"
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            print(f"✓ Data found for {year}-{month:02d}")
        except:
            print(f"✗ Data missing for {year}-{month:02d}")
            all_available = False
    
    return all_available


def calculate_experiment_size(args):
    """Calculate the size of the experiment."""
    
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    
    num_days = (end_date - start_date).days + 1
    
    # Calculate time windows per day
    hours_per_day = args.end_hour - args.start_hour
    
    if args.time_delta.endswith("m"):
        minutes = int(args.time_delta[:-1])
        windows_per_hour = 60 / minutes
    elif args.time_delta.endswith("h"):
        hours = int(args.time_delta[:-1])
        windows_per_hour = 1 / hours
    else:
        windows_per_hour = 12  # Default to 5m
    
    time_windows = int(hours_per_day * windows_per_hour * num_days)
    total_calculations = time_windows * args.num_iter
    
    print("\nExperiment size:")
    print(f"  Days: {num_days}")
    print(f"  Time windows: {time_windows}")
    print(f"  Monte Carlo iterations: {args.num_iter}")
    print(f"  Total calculations: {total_calculations:,}")
    
    # Estimate time and cost
    time_per_window = 0.5  # seconds (estimate)
    total_time = time_windows * time_per_window
    
    lambda_costs = {
        "S": 0.0000133334,   # per GB-second
        "M": 0.0000133334,
        "L": 0.0000133334,
        "XL": 0.0000133334
    }
    
    lambda_memory = {
        "S": 0.5,
        "M": 1.0,
        "L": 2.0,
        "XL": 3.0
    }
    
    estimated_cost = total_time * lambda_memory[args.lambda_size] * lambda_costs[args.lambda_size]
    
    print(f"\nEstimated execution time: {total_time/60:.1f} minutes")
    print(f"Estimated AWS Lambda cost: ${estimated_cost:.2f}")
    
    return time_windows


def invoke_lambda(args):
    """Invoke the Lambda function to start the experiment."""
    
    lambda_client = boto3.client("lambda")
    
    # Prepare payload
    payload = {
        "function_type": "orchestrator",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "vehicle_type": args.vehicle_type,
        "borough": args.borough,
        "method": args.method,
        "acceptance_function": args.eval,
        "start_hour": args.start_hour,
        "end_hour": args.end_hour,
        "time_delta": args.time_delta,
        "num_iter": args.num_iter,
        "lambda_size": args.lambda_size,
        "parallel_workers": args.parallel_workers
    }
    
    print("\nInvoking Lambda function...")
    
    try:
        response = lambda_client.invoke(
            FunctionName="taxi-benchmark-orchestrator",
            InvocationType="RequestResponse",
            Payload=json.dumps(payload)
        )
        
        # Parse response
        result = json.loads(response["Payload"].read())
        
        if response["StatusCode"] == 200:
            body = json.loads(result["body"])
            experiment_id = body["experiment_id"]
            print(f"\n✓ Experiment started successfully!")
            print(f"  Experiment ID: {experiment_id}")
            print(f"  Total scenarios: {body['total_scenarios']}")
            print(f"  Batches: {body['batches']}")
            
            # Monitor progress
            if args.monitor:
                monitor_experiment(experiment_id)
            else:
                print(f"\nTo monitor progress, run:")
                print(f"  python taxi_benchmark_cli.py --monitor {experiment_id}")
            
            return experiment_id
        else:
            print(f"\n✗ Failed to start experiment:")
            print(f"  {result}")
            return None
            
    except Exception as e:
        print(f"\n✗ Error invoking Lambda: {str(e)}")
        return None


def monitor_experiment(experiment_id):
    """Monitor the progress of an experiment."""
    
    s3_client = boto3.client("s3")
    bucket = "taxi-benchmark"
    
    print(f"\nMonitoring experiment {experiment_id}...")
    print("Press Ctrl+C to stop monitoring (experiment will continue running)")
    
    try:
        while True:
            # Check for completion
            success_key = f"experiment/run_{experiment_id}/_SUCCESS"
            try:
                s3_client.head_object(Bucket=bucket, Key=success_key)
                print("\n✓ Experiment completed successfully!")
                
                # Download and display summary
                summary_key = f"experiment/run_{experiment_id}/experiment_summary.json"
                response = s3_client.get_object(Bucket=bucket, Key=summary_key)
                summary = json.loads(response["Body"].read())
                
                print("\nExperiment Summary:")
                print(f"  Total scenarios: {summary['experiment_metadata']['total_scenarios']}")
                print(f"  Method: {summary['experiment_metadata']['method']}")
                print(f"  Acceptance function: {summary['experiment_metadata']['acceptance_function']}")
                
                stats = summary["aggregated_statistics"]
                print(f"\nResults:")
                print(f"  Avg objective: {stats['objective_values']['mean']:.2f}")
                print(f"  Avg matching rate: {stats['matching_rates']['mean']:.2%}")
                print(f"  Avg computation time: {stats['computation_times']['mean']:.3f}s")
                
                print(f"\nResults saved to:")
                print(f"  s3://{bucket}/experiment/run_{experiment_id}/")
                
                break
                
            except:
                # Count completed batches
                prefix = f"experiment/run_{experiment_id}/results_batch_"
                response = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix
                )
                
                if "Contents" in response:
                    completed = len(response["Contents"])
                    print(f"\r  Batches completed: {completed}", end="")
                
                time.sleep(5)
                
    except KeyboardInterrupt:
        print("\n\nStopped monitoring (experiment continues running)")


def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="Run taxi pricing experiments on AWS Lambda"
    )
    
    # Required arguments
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--vehicle-type", required=True, choices=["yellow", "green", "fhv"],
                       help="Type of vehicle")
    parser.add_argument("--borough", required=True, 
                       choices=["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island"],
                       help="NYC borough")
    parser.add_argument("--method", required=True,
                       choices=["MinMaxCostFlow", "MAPS", "LinUCB", "LP"],
                       help="Pricing method")
    parser.add_argument("--eval", required=True, choices=["PL", "Sigmoid"],
                       help="Acceptance probability function")
    
    # Optional arguments
    parser.add_argument("--num-iter", type=int, default=100,
                       help="Number of Monte Carlo iterations (default: 100)")
    parser.add_argument("--start-hour", type=int, default=0,
                       help="Start hour (0-23, default: 0)")
    parser.add_argument("--end-hour", type=int, default=23,
                       help="End hour (0-23, default: 23)")
    parser.add_argument("--time-delta", default="5m",
                       help="Time window size (e.g., 5m, 10m, 1h, default: 5m)")
    parser.add_argument("--lambda-size", default="L", choices=["S", "M", "L", "XL"],
                       help="Lambda function size (default: L)")
    parser.add_argument("--parallel-workers", type=int, default=10,
                       help="Number of parallel Lambda workers (default: 10)")
    
    # Monitoring
    parser.add_argument("--monitor", nargs="?", const=True, default=False,
                       help="Monitor experiment progress (optionally provide experiment ID)")
    
    # Other options
    parser.add_argument("--dry-run", action="store_true",
                       help="Show experiment details without running")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip data availability check")
    
    args = parser.parse_args()
    
    # If monitoring a specific experiment
    if args.monitor and args.monitor != True:
        monitor_experiment(args.monitor)
        return
    
    print("=" * 60)
    print("TAXI BENCHMARK FRAMEWORK")
    print("=" * 60)
    
    try:
        # Validate arguments
        validate_args(args)
        
        # Check data availability
        if not args.skip_validation:
            if not check_data_availability(args):
                print("\n✗ Some required data is missing.")
                print("  Use --skip-validation to proceed anyway (synthetic data will be used)")
                return
        
        # Calculate experiment size
        calculate_experiment_size(args)
        
        # Dry run mode
        if args.dry_run:
            print("\n[DRY RUN] Experiment not started.")
            print("Remove --dry-run to execute the experiment.")
            return
        
        # Confirm before starting
        print("\nDo you want to start the experiment? (yes/no): ", end="")
        if input().lower() != "yes":
            print("Experiment cancelled.")
            return
        
        # Invoke Lambda
        experiment_id = invoke_lambda(args)
        
        if experiment_id:
            print("\n" + "=" * 60)
            print("EXPERIMENT LAUNCHED SUCCESSFULLY")
            print("=" * 60)
        
    except ValueError as e:
        print(f"\n✗ Validation error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 