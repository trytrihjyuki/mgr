#!/usr/bin/env python3
"""
UNIFIED Ride-Hailing Pricing Experiments Runner
===============================================

This is the unified, production-ready experiment runner with full Hikima methodology support.
Includes proper rate limiting, timeout handling, and Hikima-consistent time window parametrization.

HIKIMA TIME WINDOW SETUP:
- Standard: 10:00-20:00 (10 hours)
- Default interval: 5 minutes (120 scenarios/day)
- Manhattan special: Can use 30 seconds intervals
- Other boroughs: 5 minutes recommended

Usage Examples:
    # Standard Hikima setup (10:00-20:00, 5min intervals)
    python run_pricing_experiment.py --year=2019 --month=10 --day=1 \
        --borough=Manhattan --vehicle_type=yellow --eval=PL --methods=LP

    # Custom time window (12:00-18:00, 10min intervals)  
    python run_pricing_experiment.py --year=2019 --month=10 --day=1 \
        --borough=Manhattan --vehicle_type=yellow --eval=PL --methods=LP \
        --hour_start=12 --hour_end=18 --time_interval=10

    # Manhattan with 30-second intervals (Hikima intensive)
    python run_pricing_experiment.py --year=2019 --month=10 --day=1 \
        --borough=Manhattan --vehicle_type=yellow --eval=PL --methods=LP \
        --time_interval=30 --time_unit=s
"""

import boto3
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Dict, Any, Tuple
import random
import sys

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduced from INFO
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ExperimentRunner:
    def __init__(self, region='eu-north-1', parallel_workers=5, production_mode=True):
        """Initialize the unified experiment runner with Hikima-consistent defaults"""
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        self.parallel_workers = min(parallel_workers, 5)  # Cap at 5 to avoid rate limits
        self.production_mode = production_mode
        self.lock = threading.Lock()
        self.success_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Improved timeout and retry settings
        self.lambda_timeout = 900  # 15 minutes (Lambda max)
        self.client_timeout = 600  # 10 minutes (client timeout, less than Lambda)
        self.max_retries = 3
        self.base_backoff = 2.0
        
    def invoke_lambda_with_retry(self, payload: Dict[str, Any], scenario_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Invoke Lambda with exponential backoff and proper timeout handling"""
        
        for attempt in range(self.max_retries):
            try:
                # Add jitter to prevent thundering herd  
                backoff_time = (self.base_backoff ** attempt) + random.uniform(0, 2) if attempt > 0 else 0
                if attempt > 0:
                    time.sleep(backoff_time)
                
                # Configure client with proper timeout
                response = self.lambda_client.invoke(
                    FunctionName='rideshare-pricing-benchmark',
                    InvocationType='RequestResponse',
                    Payload=json.dumps(payload)
                )
                
                if response['StatusCode'] == 200:
                    result = json.loads(response['Payload'].read().decode('utf-8'))
                    
                    # Check if Lambda function returned an error (even with 200 status)
                    if isinstance(result, dict) and 'errorMessage' in result:
                        error_msg = f"Lambda function error: {result.get('errorType', 'Unknown')} - {result.get('errorMessage', 'No message')}"
                        return False, {'error': error_msg, 'scenario_id': scenario_id}
                    
                    return True, result
                else:
                    error_msg = f"Lambda returned status {response['StatusCode']}"
                    if not self.production_mode:
                        print(f"⚠️ {scenario_id}: {error_msg}")
                    
            except Exception as e:
                error_str = str(e)
                
                # Handle specific error types
                if 'TooManyRequestsException' in error_str or 'Rate Exceeded' in error_str:
                    if attempt < self.max_retries - 1:
                        next_backoff = (self.base_backoff ** (attempt + 1)) + random.uniform(0, 2)
                        if not self.production_mode:
                            print(f"⏳ {scenario_id}: Rate limited, retrying in {next_backoff:.1f}s...")
                        continue
                    else:
                        return False, {'error': 'Rate limit exceeded after retries', 'scenario_id': scenario_id}
                        
                elif 'timeout' in error_str.lower():
                    if not self.production_mode:
                        print(f"⏰ {scenario_id}: Timeout ({attempt+1}/{self.max_retries})")
                    if attempt < self.max_retries - 1:
                        continue
                    else:
                        return False, {'error': 'Timeout after retries', 'scenario_id': scenario_id}
                
                else:
                    return False, {'error': str(e), 'scenario_id': scenario_id}
        
        return False, {'error': 'Max retries exceeded', 'scenario_id': scenario_id}

    def process_scenario(self, scenario_params: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single scenario with improved error handling"""
        scenario_id = scenario_params['scenario_id']
        
        try:
            success, result = self.invoke_lambda_with_retry(scenario_params, scenario_id)
            
            # Extract S3 location if available - FIXED: Lambda client returns data directly, not in 'body'
            s3_location = None
            if success and isinstance(result, dict):
                # First try direct access (Lambda client invocation)
                s3_location = result.get('s3_location')
                
                # Fallback: check if it's in a 'body' field (HTTP API Gateway format)
                if not s3_location:
                    body_str = result.get('body', '{}')
                    if isinstance(body_str, str):
                        try:
                            body_data = json.loads(body_str)
                            s3_location = body_data.get('s3_location')
                        except json.JSONDecodeError:
                            pass
            
            with self.lock:
                if success:
                    self.success_count += 1
                    if not self.production_mode:
                        if s3_location:
                            print(f"   ✅ {scenario_id}: Success -> {s3_location}")
                        else:
                            print(f"   ✅ {scenario_id}: Success")
                else:
                    self.error_count += 1
                    error_msg = result.get('error', 'Unknown error')
                    print(f"   ❌ {scenario_id}: {error_msg}")
                
                # Progress update
                total_processed = self.success_count + self.error_count
                if total_processed % 10 == 0 or not self.production_mode:
                    elapsed = time.time() - self.start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    print(f"⚡ [{total_processed:3d}/???] ✅{self.success_count} ❌{self.error_count} | Rate: {rate:.1f}/s")
            
            return {
                'scenario_id': scenario_id,
                'success': success,
                'result': result,
                's3_location': s3_location,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            with self.lock:
                self.error_count += 1
            return {
                'scenario_id': scenario_id,
                'success': False,
                'result': {'error': str(e)},
                's3_location': None,
                'timestamp': datetime.now().isoformat()
            }

    def run_experiments(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run experiments with improved concurrency control"""
        total_scenarios = len(scenarios)
        
        print(f"🚀 CLOUD EXECUTION: {total_scenarios} scenarios, {self.parallel_workers} parallel workers")
        
        # Adaptive worker count based on scenario complexity
        if any('methods' in s and len(s.get('methods', [])) >= 3 for s in scenarios):
            # Reduce workers for complex scenarios (3+ methods)
            adaptive_workers = min(self.parallel_workers, 3)
            print(f"🔧 Complex scenarios detected, reducing workers to {adaptive_workers}")
        else:
            adaptive_workers = self.parallel_workers
        
        results = []
        
        with ThreadPoolExecutor(max_workers=adaptive_workers) as executor:
            # Submit all scenarios
            future_to_scenario = {
                executor.submit(self.process_scenario, scenario): scenario 
                for scenario in scenarios
            }
            
            # Process completed scenarios
            for future in as_completed(future_to_scenario):
                result = future.result()
                results.append(result)
        
        # Final summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        elapsed = time.time() - self.start_time
        
        print(f"\n📊 EXECUTION COMPLETE:")
        print(f"   ✅ Successful: {successful}/{total_scenarios}")
        print(f"   ❌ Failed: {failed}/{total_scenarios}")
        print(f"   ⏱️ Total Time: {elapsed:.1f}s")
        print(f"   📈 Average Rate: {total_scenarios/elapsed:.1f} scenarios/s")
        
        return results

def create_scenario_parameters(args, training_id: str) -> List[Dict[str, Any]]:
    """Create scenario parameters with Hikima-consistent time windows"""
    scenarios = []
    execution_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Hikima time window configuration
    hour_start = args.hour_start
    hour_end = args.hour_end
    time_interval = args.time_interval
    time_unit = args.time_unit
    
    # Convert to minutes for calculation
    if time_unit == 's':
        interval_minutes = time_interval / 60.0
    else:  # 'm'
        interval_minutes = time_interval
    
    total_minutes = (hour_end - hour_start) * 60
    num_intervals = int(total_minutes / interval_minutes)
    
    for day in args.days:
        for eval_func in args.eval:
            for scenario_idx in range(num_intervals):
                # Calculate current time within the Hikima window
                minutes_elapsed = scenario_idx * interval_minutes
                current_hour = hour_start + int(minutes_elapsed // 60)
                current_minute = int(minutes_elapsed % 60)
                
                scenario_id = f"day{day:02d}_{eval_func}_s{scenario_idx:03d}"
                
                scenario = {
                    'scenario_id': scenario_id,
                    'year': args.year,
                    'month': args.month,
                    'day': day,
                    'borough': args.borough,
                    'vehicle_type': args.vehicle_type,
                    'acceptance_function': eval_func,
                    'scenario_index': scenario_idx,
                    'time_window': {
                        'hour_start': hour_start,
                        'hour_end': hour_end,
                        'minute_start': 0,
                        'time_interval': time_interval,
                        'time_unit': time_unit,
                        'current_hour': current_hour,
                        'current_minute': current_minute,
                        'current_second': 0
                    },
                    'methods': args.methods,
                    'training_id': training_id,
                    'execution_date': execution_date
                }
                scenarios.append(scenario)
    
    return scenarios

def validate_hikima_consistency(args):
    """Validate that parameters are consistent with Hikima methodology"""
    warnings = []
    
    # Check standard Hikima time window
    if args.hour_start != 10 or args.hour_end != 20:
        warnings.append(f"⚠️ Non-standard time window: {args.hour_start}:00-{args.hour_end}:00 (Hikima standard: 10:00-20:00)")
    
    # Check interval consistency
    if args.time_unit == 'm' and args.time_interval not in [5, 10, 15, 30]:
        warnings.append(f"⚠️ Unusual minute interval: {args.time_interval}m (common: 5m, 10m, 15m, 30m)")
    
    if args.time_unit == 's' and args.time_interval not in [30, 60, 120, 300]:
        warnings.append(f"⚠️ Unusual second interval: {args.time_interval}s (common: 30s, 60s, 120s, 300s)")
    
    # Check borough-specific recommendations
    if args.borough == 'Manhattan' and args.time_unit == 'm' and args.time_interval > 5:
        warnings.append(f"⚠️ Manhattan typically uses ≤5m intervals (current: {args.time_interval}m)")
    
    # Calculate scenarios per day
    if args.time_unit == 's':
        interval_minutes = args.time_interval / 60.0
    else:
        interval_minutes = args.time_interval
    
    scenarios_per_day = int((args.hour_end - args.hour_start) * 60 / interval_minutes)
    
    if scenarios_per_day > 1200:  # More than 30s intervals over 10 hours
        warnings.append(f"⚠️ Very high scenario count: {scenarios_per_day}/day (consider larger intervals)")
    
    return warnings, scenarios_per_day

def main():
    parser = argparse.ArgumentParser(
        description='Unified Ride-Hailing Pricing Experiments - Hikima Methodology',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
HIKIMA TIME WINDOW EXAMPLES:

Standard Hikima setup (10:00-20:00, 5min intervals = 120 scenarios/day):
    --hour_start=10 --hour_end=20 --time_interval=5 --time_unit=m

Manhattan intensive (10:00-20:00, 30sec intervals = 1200 scenarios/day):
    --hour_start=10 --hour_end=20 --time_interval=30 --time_unit=s

Custom business hours (09:00-17:00, 10min intervals = 48 scenarios/day):
    --hour_start=9 --hour_end=17 --time_interval=10 --time_unit=m
        """
    )
    
    # Core experiment parameters
    parser.add_argument('--year', type=int, required=True, help='Year (e.g. 2019)')
    parser.add_argument('--month', type=int, required=True, help='Month (1-12)')
    parser.add_argument('--borough', required=True, choices=['Manhattan', 'Bronx', 'Queens', 'Brooklyn'], 
                       help='Borough name')
    parser.add_argument('--vehicle_type', required=True, choices=['yellow', 'green', 'fhv'], 
                       help='Vehicle type')
    parser.add_argument('--eval', required=True, help='Acceptance functions (comma-separated): PL,Sigmoid')
    parser.add_argument('--methods', required=True, 
                       help='Pricing methods (comma-separated): LP,MinMaxCostFlow,LinUCB,MAPS')
    
    # Day specification (multiple options)
    day_group = parser.add_mutually_exclusive_group(required=True)
    day_group.add_argument('--day', type=int, help='Single day (1-31)')
    day_group.add_argument('--days', help='Multiple days (comma-separated): 1,6,10')
    day_group.add_argument('--start_day', type=int, help='Start day for range (use with --end_day)')
    parser.add_argument('--end_day', type=int, help='End day for range (use with --start_day)')
    
    # HIKIMA TIME WINDOW PARAMETERS
    hikima_group = parser.add_argument_group('Hikima Time Window Configuration')
    hikima_group.add_argument('--hour_start', type=int, default=10, 
                             help='Start hour - Hikima standard: 10 (default: 10)')
    hikima_group.add_argument('--hour_end', type=int, default=20, 
                             help='End hour - Hikima standard: 20 (default: 20)')
    hikima_group.add_argument('--time_interval', type=int, default=5, 
                             help='Time interval value - Hikima standard: 5 (default: 5)')
    hikima_group.add_argument('--time_unit', choices=['m', 's'], default='m', 
                             help='Time unit: m=minutes, s=seconds - Hikima standard: m (default: m)')
    
    # LinUCB training options
    parser.add_argument('--skip_training', action='store_true', help='Skip LinUCB training (use pre-trained models)')
    parser.add_argument('--force_training', action='store_true', help='Force LinUCB retraining')
    
    # Execution options
    parser.add_argument('--parallel', type=int, default=3, help='Parallel workers (default: 3)')
    parser.add_argument('--production', action='store_true', help='Production mode (minimal logging)')
    parser.add_argument('--debug', action='store_true', help='Debug mode (verbose output)')
    parser.add_argument('--dry_run', action='store_true', help='Dry run (show what would be executed)')
    
    args = parser.parse_args()
    
    # Process day arguments
    if args.day:
        args.days = [args.day]
    elif args.days:
        args.days = [int(d.strip()) for d in args.days.split(',')]
    elif args.start_day and args.end_day:
        args.days = list(range(args.start_day, args.end_day + 1))
    else:
        parser.error("Must specify --day, --days, or --start_day/--end_day")
    
    # Process eval and methods
    args.eval = [e.strip() for e in args.eval.split(',')]
    args.methods = [m.strip() for m in args.methods.split(',')]
    
    # Validate methods
    valid_methods = ['LP', 'MinMaxCostFlow', 'LinUCB', 'MAPS']
    for method in args.methods:
        if method not in valid_methods:
            parser.error(f"Invalid method: {method}. Valid: {valid_methods}")
    
    # Adaptive parallel workers based on complexity
    if len(args.methods) >= 3:
        args.parallel = min(args.parallel, 2)  # Cap at 2 for complex scenarios
        print(f"🔧 Complex scenario detected ({len(args.methods)} methods), reducing parallel workers to {args.parallel}")
    
    print("🚀 UNIFIED RIDE-HAILING PRICING EXPERIMENTS")
    print("=" * 70)
    
    # Validate Hikima consistency
    warnings, scenarios_per_day = validate_hikima_consistency(args)
    
    if warnings:
        print("⚠️ HIKIMA CONSISTENCY WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
        print()
    
    # Configuration summary
    print("🎯 EXPERIMENT CONFIGURATION:")
    print(f"   📅 Date(s): {args.year}-{args.month:02d}-{args.days}")
    print(f"   🏙️ Borough: {args.borough}")
    print(f"   🚗 Vehicle: {args.vehicle_type}")
    print(f"   📊 Evaluation: {args.eval}")
    print(f"   🔧 Methods: {args.methods}")
    print(f"   ⚡ Parallel Workers: {args.parallel}")
    print(f"   🎛️ Production Mode: {args.production}")
    
    # Generate training ID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    training_id = f"exp_{args.vehicle_type}_{args.borough}_{timestamp}"
    print(f"   🆔 Training ID: {training_id}")
    
    # Check LinUCB models if needed
    if 'LinUCB' in args.methods:
        linucb_key = f"models/linucb/{args.vehicle_type}_{args.borough}_201907/trained_model.pkl"
        s3_client = boto3.client('s3', region_name='eu-north-1')
        
        try:
            s3_client.head_object(Bucket='magisterka', Key=linucb_key)
            print(f"✅ LinUCB model ready: s3://magisterka/{linucb_key}")
        except:
            if not args.skip_training:
                print(f"⚠️ LinUCB model missing, training will be needed")
            else:
                print(f"❌ LinUCB model missing and training skipped")
                return 1
    
    # Calculate scenarios with Hikima methodology
    hour_start = args.hour_start
    hour_end = args.hour_end 
    time_interval = args.time_interval
    total_scenarios = len(args.days) * len(args.eval) * scenarios_per_day
    
    print(f"\n📊 HIKIMA TIME WINDOW CONFIGURATION:")
    print(f"   ⏰ Time Window: {hour_start:02d}:00-{hour_end:02d}:00 ({hour_end-hour_start} hours)")
    print(f"   📅 Interval: {time_interval}{args.time_unit} ({'standard' if (time_interval == 5 and args.time_unit == 'm') else 'custom'})")
    print(f"   🔢 Scenarios/Day: {scenarios_per_day}")
    print(f"   📈 Total Scenarios: {total_scenarios}")
    
    # Execution plan
    print(f"\n📋 EXECUTION PLAN:")
    print(f"   📊 Total scenarios: {total_scenarios}")
    
    # Time estimation based on method complexity
    if len(args.methods) == 1:
        estimated_time = total_scenarios * 6 / args.parallel  # 6s per scenario for single method
    elif len(args.methods) == 2:
        estimated_time = total_scenarios * 15 / args.parallel  # 15s per scenario for 2 methods
    else:
        estimated_time = total_scenarios * 45 / args.parallel  # 45s per scenario for 3+ methods
    
    print(f"   ⏱️ Estimated time: {estimated_time:.0f}s ({estimated_time/60:.1f} min)")
    print(f"   💰 Estimated cost: ~${total_scenarios * 0.0001:.2f} (Lambda invocations)")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No actual execution")
        return 0
    
    # Create scenarios
    scenarios = create_scenario_parameters(args, training_id)
    
    print(f"\n🚀 STARTING EXPERIMENT EXECUTION...")
    
    # Run experiments with improved runner
    runner = ExperimentRunner(
        parallel_workers=args.parallel,
        production_mode=args.production
    )
    
    results = runner.run_experiments(scenarios)
    
    # Final summary
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   ✅ Successful: {len(successful_results)}/{total_scenarios}")
    print(f"   ❌ Failed: {len(failed_results)}/{total_scenarios}")
    
    # Aggregate and display S3 paths by day
    s3_paths_by_day = {}
    for result in successful_results:
        s3_location = result.get('s3_location')
        if s3_location:
            # Extract day from scenario_id (format: day{day:02d}_{eval_func}_s{scenario_idx:03d})
            scenario_id = result.get('scenario_id', '')
            if scenario_id.startswith('day'):
                day_part = scenario_id.split('_')[0]  # Extract "day01", "day06", etc.
                if day_part not in s3_paths_by_day:
                    s3_paths_by_day[day_part] = s3_location
    
    if s3_paths_by_day:
        print(f"\n💾 S3 EXPERIMENT RESULTS:")
        for day_part, s3_path in sorted(s3_paths_by_day.items()):
            day_num = day_part.replace('day', '').lstrip('0') or '1'  # Remove leading zeros
            print(f"   📅 Day {day_num}: {s3_path}")
    
    if failed_results:
        print(f"\n❌ FAILED SCENARIOS:")
        for result in failed_results[:10]:  # Show first 10
            error = result['result'].get('error', 'Unknown error')
            print(f"   - {result['scenario_id']}: {error}")
        if len(failed_results) > 10:
            print(f"   ... and {len(failed_results) - 10} more")
    
    success_rate = len(successful_results) / total_scenarios * 100 if total_scenarios > 0 else 0
    print(f"\n📈 SUCCESS RATE: {success_rate:.1f}%")
    
    if success_rate < 90:
        print("⚠️ Low success rate - consider reducing --parallel or checking Lambda limits")
        return 1
    
    print(f"\n🎉 Experiment completed! Check S3 paths above for results.")
    return 0

if __name__ == '__main__':
    sys.exit(main()) 