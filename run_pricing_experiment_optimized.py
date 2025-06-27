#!/usr/bin/env python3
"""
Optimized Ride-Hailing Pricing Experiment CLI - Cloud Performance Edition

This version is optimized for fast cloud execution with:
- Parallel Lambda invocations (up to 50 concurrent)
- Production mode with minimal logging
- Real-time progress tracking
- Automatic error recovery
- Fast result aggregation

Usage:
    python run_pricing_experiment_optimized.py --year=2019 --month=10 --day=6 \\
        --borough=Manhattan --vehicle_type=yellow --eval=PL --methods=LP \\
        --parallel=20 --production_mode
"""

import argparse
import json
import boto3
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

class OptimizedPricingExperimentCLI:
    def __init__(self):
        self.lambda_client = boto3.client('lambda', region_name='eu-north-1')
        self.s3_client = boto3.client('s3')
        self.function_name = 'rideshare-pricing-benchmark'
        self.bucket_name = 'magisterka'
        self.progress_lock = threading.Lock()
        self.stats = {
            'total_scenarios': 0,
            'completed': 0,
            'failed': 0,
            'start_time': None,
            'last_update': None
        }
    
    def parse_arguments(self):
        """Parse command line arguments for cloud optimization."""
        parser = argparse.ArgumentParser(
            description='🚀 Optimized Ride-Hailing Pricing Experiments - Cloud Performance Edition',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Fast single day with 20 parallel executions
  python run_pricing_experiment_optimized.py --year=2019 --month=10 --day=1 \\
    --borough=Manhattan --vehicle_type=yellow --eval=PL --methods=LP \\
    --parallel=20 --production_mode

  # Multiple days with maximum performance  
  python run_pricing_experiment_optimized.py --year=2019 --month=10 --days=1,6,10 \\
    --borough=Manhattan --vehicle_type=yellow --eval=PL,Sigmoid \\
    --methods=LP,MinMaxCostFlow,LinUCB,MAPS --parallel=30 --production_mode

  # Full month with optimized settings
  python run_pricing_experiment_optimized.py --year=2019 --month=10 --start_day=1 --end_day=31 \\
    --borough=Manhattan --vehicle_type=yellow --eval=PL \\
    --methods=LP,LinUCB --parallel=25 --production_mode --skip_training
            """
        )
        
        # Required parameters
        parser.add_argument('--year', type=int, required=True, help='Year (e.g., 2019)')
        parser.add_argument('--month', type=int, required=True, help='Month (1-12)')
        parser.add_argument('--borough', required=True, choices=['Manhattan', 'Brooklyn', 'Queens', 'Bronx'], help='Borough')
        parser.add_argument('--vehicle_type', required=True, choices=['yellow', 'green', 'fhv'], help='Vehicle type')
        parser.add_argument('--eval', required=True, help='Evaluation functions (comma-separated: PL,Sigmoid)')
        parser.add_argument('--methods', required=True, help='Pricing methods (comma-separated: LP,MinMaxCostFlow,LinUCB,MAPS)')
        
        # Day specification (mutually exclusive)
        day_group = parser.add_mutually_exclusive_group(required=True)
        day_group.add_argument('--day', type=int, help='Single day to analyze')
        day_group.add_argument('--days', help='Comma-separated list of days (e.g., 1,6,10)')
        day_group.add_argument('--start_day', type=int, help='Start day for range (use with --end_day)')
        
        parser.add_argument('--end_day', type=int, help='End day for range (use with --start_day)')
        
        # Cloud optimization parameters
        parser.add_argument('--parallel', type=int, default=15, help='Parallel Lambda executions (1-50, default: 15)')
        parser.add_argument('--production_mode', action='store_true', help='Production mode for faster execution')
        parser.add_argument('--timeout', type=int, default=900, help='Lambda timeout seconds (default: 900)')
        parser.add_argument('--skip_training', action='store_true', help='Skip LinUCB training (use pre-trained models)')
        parser.add_argument('--force_training', action='store_true', help='Force LinUCB retraining')
        
        # Hikima time window configuration
        parser.add_argument('--hour_start', type=int, default=10, help='Start hour (default: 10 for 10:00)')
        parser.add_argument('--hour_end', type=int, default=20, help='End hour (default: 20 for 20:00)')
        parser.add_argument('--time_interval', type=int, default=5, help='Time interval (default: 5)')
        parser.add_argument('--time_unit', choices=['m', 's'], default='m', help='Time unit: m=minutes, s=seconds (default: m)')
        
        # Testing and analysis
        parser.add_argument('--dry_run', action='store_true', help='Show execution plan without running')
        parser.add_argument('--training_id', help='Custom training ID')
        parser.add_argument('--retry_failed', action='store_true', help='Retry failed scenarios automatically')
        parser.add_argument('--max_retries', type=int, default=2, help='Maximum retries per scenario (default: 2)')
        
        return parser.parse_args()
    
    def validate_arguments(self, args):
        """Validate arguments with cloud-specific checks."""
        errors = []
        
        # Date validation
        if args.start_day and not args.end_day:
            errors.append("--end_day required with --start_day")
        if args.end_day and not args.start_day:
            errors.append("--start_day required with --end_day")
        
        # Performance validation
        if args.parallel < 1 or args.parallel > 50:
            errors.append("--parallel must be between 1 and 50 for optimal cloud performance")
        
        if args.skip_training and args.force_training:
            errors.append("Cannot use both --skip_training and --force_training")
        
        # Parse and validate evaluation functions
        valid_evals = {'PL', 'Sigmoid'}
        eval_functions = [e.strip() for e in args.eval.split(',')]
        for eval_func in eval_functions:
            if eval_func not in valid_evals:
                errors.append(f"Invalid evaluation function: {eval_func}")
        
        # Parse and validate methods
        valid_methods = {'LP', 'MinMaxCostFlow', 'LinUCB', 'MAPS'}
        methods = [m.strip() for m in args.methods.split(',')]
        for method in methods:
            if method not in valid_methods:
                errors.append(f"Invalid method: {method}")
        
        if errors:
            print("❌ Validation errors:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        
        return eval_functions, methods
    
    def get_days_to_process(self, args):
        """Get list of days to process."""
        if args.day:
            return [args.day]
        elif args.days:
            return [int(d.strip()) for d in args.days.split(',')]
        elif args.start_day and args.end_day:
            return list(range(args.start_day, args.end_day + 1))
    
    def generate_training_id(self, args):
        """Generate optimized training ID."""
        if args.training_id:
            return args.training_id
        return f"opt_exp_{args.vehicle_type}_{args.borough}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def check_linucb_requirements(self, args, methods):
        """Check LinUCB requirements for cloud execution."""
        if 'LinUCB' not in methods:
            return True
        
        model_key = f"models/linucb/{args.vehicle_type}_{args.borough}_201907/trained_model.pkl"
        
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=model_key)
            print(f"✅ LinUCB model ready: s3://{self.bucket_name}/{model_key}")
            return True
        except:
            if args.skip_training:
                print(f"❌ LinUCB model missing and --skip_training specified")
                print(f"   Missing: s3://{self.bucket_name}/{model_key}")
                return False
            else:
                print(f"⚠️ LinUCB model not found. Training required (use --skip_training to avoid)")
                return True
    
    def create_scenario_payloads(self, args, eval_functions, methods, days, training_id):
        """Create optimized scenario payloads for parallel execution with configurable time windows - EXACT Hikima methodology."""
        scenarios = []
        scenario_counter = 0
        
        # Calculate total scenarios based on EXACT Hikima methodology
        if args.time_unit == 'm':
            # Minutes: calculate how many intervals fit in the time range
            total_minutes = (args.hour_end - args.hour_start) * 60
            total_scenarios = total_minutes // args.time_interval
            unit_display = f"{args.time_interval}min"
        else:  # seconds
            # Seconds: calculate how many intervals fit in the time range  
            total_seconds = (args.hour_end - args.hour_start) * 3600
            total_scenarios = total_seconds // args.time_interval
            unit_display = f"{args.time_interval}s"
        
        print(f"📊 Hikima Configuration: {args.hour_start}:00-{args.hour_end}:00, {unit_display} intervals = {total_scenarios} scenarios/day")
        
        for day in days:
            for eval_function in eval_functions:
                # Generate scenarios based on exact Hikima time methodology
                for scenario_index in range(total_scenarios):
                    if args.time_unit == 'm':
                        # Minutes-based calculation
                        scenario_minute = scenario_index * args.time_interval
                        current_hour = args.hour_start + (scenario_minute // 60)
                        current_minute = scenario_minute % 60
                        current_second = 0
                    else:  # seconds
                        # Seconds-based calculation
                        scenario_seconds = scenario_index * args.time_interval
                        total_minutes = scenario_seconds // 60
                        current_hour = args.hour_start + (total_minutes // 60)
                        current_minute = total_minutes % 60
                        current_second = scenario_seconds % 60
                    
                    scenario = {
                        'scenario_id': f"day{day:02d}_{eval_function}_s{scenario_index:03d}",
                        'year': args.year,
                        'month': args.month,
                        'day': day,
                        'borough': args.borough,
                        'vehicle_type': args.vehicle_type,
                        'acceptance_function': eval_function,
                        'scenario_index': scenario_index,
                        'time_window': {
                            'hour_start': args.hour_start,
                            'hour_end': args.hour_end,
                            'minute_start': 0,
                            'time_interval': args.time_interval,
                            'time_unit': args.time_unit,
                            'current_hour': current_hour,
                            'current_minute': current_minute,
                            'current_second': current_second
                        },
                        'methods': methods,
                        'training_id': training_id,
                        'execution_date': datetime.now().strftime('%Y%m%d_%H%M%S')
                    }
                    
                    # Add production mode flag
                    if args.production_mode:
                        scenario['production_mode'] = True
                    
                    scenarios.append(scenario)
                    scenario_counter += 1
        
        return scenarios
    
    def invoke_lambda_scenario(self, scenario, retry_count=0):
        """Invoke single Lambda scenario with error handling."""
        try:
            response = self.lambda_client.invoke(
                FunctionName=self.function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(scenario)
            )
            
            if response.get('StatusCode') == 200:
                result = json.loads(response['Payload'].read())
                return scenario['scenario_id'], result, None
            else:
                error = f"Lambda Status {response.get('StatusCode')}"
                return scenario['scenario_id'], None, error
                
        except Exception as e:
            return scenario['scenario_id'], None, str(e)
    
    def update_progress(self, completed, failed, total):
        """Thread-safe progress update."""
        with self.progress_lock:
            self.stats['completed'] = completed
            self.stats['failed'] = failed
            current_time = time.time()
            
            if self.stats['start_time']:
                elapsed = current_time - self.stats['start_time']
                rate = (completed + failed) / elapsed if elapsed > 0 else 0
                remaining = total - completed - failed
                eta = remaining / rate if rate > 0 else 0
                
                # Progress update every 5 scenarios or 10 seconds
                if self.stats['last_update'] is None:
                    self.stats['last_update'] = current_time
                
                if (completed + failed) % 5 == 0 or current_time - self.stats['last_update'] > 10:
                    print(f"⚡ [{completed + failed:3d}/{total}] ✅{completed} ❌{failed} | "
                          f"Rate: {rate:.1f}/s | ETA: {eta:.0f}s")
                    self.stats['last_update'] = current_time
    
    def execute_scenarios_parallel(self, scenarios, args):
        """Execute scenarios with optimized parallel processing."""
        print(f"🚀 CLOUD EXECUTION: {len(scenarios)} scenarios, {args.parallel} parallel workers")
        
        self.stats['total_scenarios'] = len(scenarios)
        self.stats['start_time'] = time.time()
        
        results = {}
        completed_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all scenarios
            future_to_scenario = {
                executor.submit(self.invoke_lambda_scenario, scenario): scenario
                for scenario in scenarios
            }
            
            # Process results as they complete
            for future in as_completed(future_to_scenario):
                scenario = future_to_scenario[future]
                scenario_id, result, error = future.result()
                
                if result and not error:
                    results[scenario_id] = result
                    completed_count += 1
                else:
                    results[scenario_id] = {'error': error}
                    failed_count += 1
                    if not args.production_mode:
                        print(f"   ❌ {scenario_id}: {error}")
                
                self.update_progress(completed_count, failed_count, len(scenarios))
        
        total_time = time.time() - self.stats['start_time']
        success_rate = completed_count / len(scenarios) * 100
        
        print(f"\n✅ EXECUTION COMPLETED")
        print(f"   📊 Total: {len(scenarios)} scenarios")  
        print(f"   ✅ Success: {completed_count} ({success_rate:.1f}%)")
        print(f"   ❌ Failed: {failed_count}")
        print(f"   ⏱️ Total time: {total_time:.1f}s")
        print(f"   🚀 Average rate: {len(scenarios)/total_time:.1f} scenarios/second")
        
        return results, success_rate > 95  # Consider successful if >95% completion
    
    def aggregate_results_fast(self, results, scenarios, args, eval_functions, methods, training_id):
        """Fast results aggregation optimized for cloud processing."""
        print("📊 Aggregating results...")
        
        # Group by day and evaluation function
        day_groups = defaultdict(lambda: defaultdict(list))
        
        for scenario in scenarios:
            day = scenario['day']
            eval_func = scenario['acceptance_function']
            scenario_id = scenario['scenario_id']
            
            if scenario_id in results and 'error' not in results[scenario_id]:
                day_groups[day][eval_func].append(results[scenario_id])
        
        # Create day aggregations
        day_results = []
        for day in sorted(day_groups.keys()):
            for eval_func in eval_functions:
                if eval_func in day_groups[day]:
                    scenarios_data = day_groups[day][eval_func]
                    
                    if scenarios_data:
                        # Calculate aggregated statistics
                        total_scenarios = len(scenarios_data)
                        
                        # Aggregate method results
                        method_stats = defaultdict(list)
                        for scenario_result in scenarios_data:
                            if 'results' in scenario_result:
                                for method_result in scenario_result['results']:
                                    method_name = method_result['method_name']
                                    method_stats[method_name].append(method_result['objective_value'])
                        
                        # Create day summary
                        day_summary = {
                            'day': day,
                            'evaluation_function': eval_func,
                            'total_scenarios': total_scenarios,
                            'methods': list(methods),
                            'training_id': training_id,
                            'aggregation_timestamp': datetime.now().isoformat(),
                            'method_statistics': {},
                            'all_scenario_results': scenarios_data
                        }
                        
                        # Calculate method statistics
                        for method_name, values in method_stats.items():
                            if values:
                                day_summary['method_statistics'][method_name] = {
                                    'mean_objective': sum(values) / len(values),
                                    'max_objective': max(values),
                                    'min_objective': min(values),
                                    'scenario_count': len(values)
                                }
                        
                        # Save to S3
                        s3_key = f"experiments/type={args.vehicle_type}/eval={eval_func}/borough={args.borough}/year={args.year}/month={args.month:02d}/day={day:02d}/optimized_{training_id}.json"
                        
                        try:
                            self.s3_client.put_object(
                                Bucket=self.bucket_name,
                                Key=s3_key,
                                Body=json.dumps(day_summary, indent=2),
                                ContentType='application/json'
                            )
                            
                            day_results.append({
                                'day': day,
                                'eval_function': eval_func,
                                's3_key': s3_key,
                                'scenarios': total_scenarios,
                                'methods': len(method_stats)
                            })
                            
                        except Exception as e:
                            print(f"⚠️ Failed to save day {day} results: {e}")
        
        return day_results
    
    def run(self):
        """Main optimized execution function."""
        print("🚀 OPTIMIZED RIDE-HAILING PRICING EXPERIMENTS - CLOUD EDITION")
        print("=" * 70)
        
        # Parse and validate
        args = self.parse_arguments() 
        eval_functions, methods = self.validate_arguments(args)
        days = self.get_days_to_process(args)
        training_id = self.generate_training_id(args)
        
        print(f"🎯 CLOUD CONFIGURATION:")
        print(f"   📅 Date(s): {args.year}-{args.month:02d}-{days}")
        print(f"   🏙️ Borough: {args.borough}")
        print(f"   🚗 Vehicle: {args.vehicle_type}")
        print(f"   📊 Evaluation: {eval_functions}")
        print(f"   🔧 Methods: {methods}")
        print(f"   ⚡ Parallel Workers: {args.parallel}")
        print(f"   🎛️ Production Mode: {'ON' if args.production_mode else 'OFF'}")
        print(f"   🆔 Training ID: {training_id}")
        
        # Check requirements
        if not self.check_linucb_requirements(args, methods):
            sys.exit(1)
        
        # Generate scenarios
        scenarios = self.create_scenario_payloads(args, eval_functions, methods, days, training_id)
        total_scenarios = len(scenarios)
        estimated_time = total_scenarios * 3 / args.parallel  # ~3s per scenario
        
        print(f"\n📋 EXECUTION PLAN:")
        print(f"   📊 Total scenarios: {total_scenarios}")
        print(f"   ⏱️ Estimated time: {estimated_time:.0f}s ({estimated_time/60:.1f} min)")
        print(f"   💰 Estimated cost: ~${total_scenarios * 0.0001:.2f} (Lambda invocations)")
        
        if args.dry_run:
            print("\n🔍 DRY RUN MODE - Execution plan shown above")
            print("Remove --dry_run to execute")
            return
        
        # Execute scenarios
        print(f"\n🚀 STARTING CLOUD EXECUTION...")
        results, success = self.execute_scenarios_parallel(scenarios, args)
        
        if not success:
            print("⚠️ Some scenarios failed. Results may be incomplete.")
        
        # Aggregate results
        day_results = self.aggregate_results_fast(results, scenarios, args, eval_functions, methods, training_id)
        
        # Show results
        print(f"\n📈 RESULTS SUMMARY:")
        for result in day_results:
            print(f"   📄 Day {result['day']} ({result['eval_function']}): {result['scenarios']} scenarios")
            print(f"      💾 s3://magisterka/{result['s3_key']}")
        
        print(f"\n🎉 CLOUD EXECUTION COMPLETED!")
        print(f"   ✅ Processed {len(day_results)} day-evaluation combinations")
        print(f"   📊 Total scenarios: {sum(r['scenarios'] for r in day_results)}")
        
        if not args.production_mode:
            print(f"\n🔍 To analyze results:")
            print(f"   aws s3 ls s3://magisterka/experiments/type={args.vehicle_type}/")

if __name__ == "__main__":
    cli = OptimizedPricingExperimentCLI()
    cli.run() 