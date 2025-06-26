#!/usr/bin/env python3
"""
Ride-Hailing Pricing Experiment CLI Tool

Usage:
    python run_pricing_experiment.py --year=2019 --month=10 --day=6 --borough=Manhattan --vehicle_type=yellow --eval=PL,Sigmoid --methods=LP,MinMaxCostFlow,LinUCB,MAPS
"""

import argparse
import json
import boto3
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
import os

class PricingExperimentCLI:
    def __init__(self):
        self.lambda_client = boto3.client('lambda')
        self.s3_client = boto3.client('s3')
        self.function_name = 'rideshare-pricing-benchmark'
        self.bucket_name = 'magisterka'
    
    def parse_arguments(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description='Run ride-hailing pricing experiments',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Single day experiment
  python run_pricing_experiment.py --year=2019 --month=10 --day=6 --borough=Manhattan --vehicle_type=yellow --eval=PL --methods=MinMaxCostFlow,MAPS

  # Multiple days
  python run_pricing_experiment.py --year=2019 --month=10 --days=1,6,10 --borough=Manhattan --vehicle_type=green --eval=PL,Sigmoid --methods=LP,MinMaxCostFlow,LinUCB,MAPS

  # Date range
  python run_pricing_experiment.py --year=2019 --month=10 --start_day=1 --end_day=31 --borough=Queens --vehicle_type=yellow --eval=PL --methods=LinUCB
            """
        )
        
        # Required parameters
        parser.add_argument('--year', type=int, required=True, help='Year (e.g., 2019)')
        parser.add_argument('--month', type=int, required=True, help='Month (1-12)')
        parser.add_argument('--borough', required=True, choices=['Manhattan', 'Brooklyn', 'Queens', 'Bronx'], help='Borough to analyze')
        parser.add_argument('--vehicle_type', required=True, choices=['yellow', 'green', 'fhv'], help='Vehicle type')
        parser.add_argument('--eval', required=True, help='Evaluation functions (comma-separated: PL,Sigmoid)')
        parser.add_argument('--methods', required=True, help='Pricing methods (comma-separated: LP,MinMaxCostFlow,LinUCB,MAPS)')
        
        # Day specification (mutually exclusive)
        day_group = parser.add_mutually_exclusive_group(required=True)
        day_group.add_argument('--day', type=int, help='Single day to analyze')
        day_group.add_argument('--days', help='Comma-separated list of days (e.g., 1,6,10)')
        day_group.add_argument('--start_day', type=int, help='Start day for range (use with --end_day)')
        
        parser.add_argument('--end_day', type=int, help='End day for range (use with --start_day)')
        
        # Optional parameters
        parser.add_argument('--training_period', default='2019-07', help='Training period for LinUCB (YYYY-MM format)')
        parser.add_argument('--hour_start', type=int, default=10, help='Start hour (default: 10)')
        parser.add_argument('--hour_end', type=int, default=20, help='End hour (default: 20)')
        parser.add_argument('--time_interval', type=int, default=5, help='Time interval in minutes (default: 5)')
        parser.add_argument('--dry_run', action='store_true', help='Show what would be executed without running')
        parser.add_argument('--parallel', type=int, default=1, help='Number of parallel executions (default: 1)')
        parser.add_argument('--training_id', help='Custom training ID (default: auto-generated)')
        
        return parser.parse_args()
    
    def validate_arguments(self, args):
        """Validate command line arguments."""
        errors = []
        
        # Validate date range
        if args.start_day and not args.end_day:
            errors.append("--end_day is required when using --start_day")
        if args.end_day and not args.start_day:
            errors.append("--start_day is required when using --end_day")
        if args.start_day and args.end_day and args.start_day > args.end_day:
            errors.append("--start_day must be <= --end_day")
        
        # Validate month/day ranges
        if args.month < 1 or args.month > 12:
            errors.append("Month must be between 1 and 12")
        
        max_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Assuming leap year
        max_day = max_days[args.month - 1]
        
        days_to_check = []
        if args.day:
            days_to_check = [args.day]
        elif args.days:
            days_to_check = [int(d.strip()) for d in args.days.split(',')]
        elif args.start_day and args.end_day:
            days_to_check = list(range(args.start_day, args.end_day + 1))
        
        for day in days_to_check:
            if day < 1 or day > max_day:
                errors.append(f"Day {day} is invalid for month {args.month}")
        
        # Validate time range
        if args.hour_start >= args.hour_end:
            errors.append("--hour_start must be < --hour_end")
        if args.hour_start < 0 or args.hour_start > 23:
            errors.append("--hour_start must be between 0 and 23")
        if args.hour_end < 1 or args.hour_end > 24:
            errors.append("--hour_end must be between 1 and 24")
        
        # Validate eval functions
        valid_evals = {'PL', 'Sigmoid'}
        eval_functions = [e.strip() for e in args.eval.split(',')]
        for eval_func in eval_functions:
            if eval_func not in valid_evals:
                errors.append(f"Invalid evaluation function: {eval_func}. Valid options: {', '.join(valid_evals)}")
        
        # Validate methods
        valid_methods = {'LP', 'MinMaxCostFlow', 'LinUCB', 'MAPS'}
        methods = [m.strip() for m in args.methods.split(',')]
        for method in methods:
            if method not in valid_methods:
                errors.append(f"Invalid method: {method}. Valid options: {', '.join(valid_methods)}")
        
        if errors:
            print("❌ Validation errors:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        
        return eval_functions, methods, days_to_check
    
    def get_days_to_process(self, args):
        """Get list of days to process based on arguments."""
        if args.day:
            return [args.day]
        elif args.days:
            return [int(d.strip()) for d in args.days.split(',')]
        elif args.start_day and args.end_day:
            return list(range(args.start_day, args.end_day + 1))
        else:
            raise ValueError("No days specified")
    
    def generate_training_id(self, args):
        """Generate training ID if not provided."""
        if args.training_id:
            return args.training_id
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"pricing_exp_{args.vehicle_type}_{args.borough}_{timestamp}"
    
    def check_linucb_training(self, args, training_id):
        """Check if LinUCB training data exists, create if needed."""
        methods = [m.strip() for m in args.methods.split(',')]
        if 'LinUCB' not in methods:
            return True
        
        # Check if training data exists in S3
        training_key = f"models/linucb/{args.vehicle_type}_{args.borough}_{args.training_period.replace('-', '')}/trained_model.pkl"
        
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=training_key)
            print(f"✅ LinUCB training data found: s3://{self.bucket_name}/{training_key}")
            return True
        except:
            print(f"⚠️ LinUCB training data not found. Triggering training for {args.training_period}...")
            return self.trigger_linucb_training(args, training_id)
    
    def trigger_linucb_training(self, args, training_id):
        """Trigger LinUCB training Lambda function."""
        # Parse training period
        training_year, training_month = args.training_period.split('-')
        training_year = int(training_year)
        training_month = int(training_month)
        
        # Create training event
        training_event = {
            'action': 'train_linucb',
            'vehicle_type': args.vehicle_type,
            'borough': args.borough,
            'training_year': training_year,
            'training_month': training_month,
            'training_id': training_id,
            'base_price': 5.875,
            'price_multipliers': [0.6, 0.8, 1.0, 1.2, 1.4]
        }
        
        print(f"🔧 Starting LinUCB training for {args.vehicle_type} taxis in {args.borough}, {args.training_period}")
        
        try:
            response = self.lambda_client.invoke(
                FunctionName=self.function_name,
                Payload=json.dumps(training_event)
            )
            
            result = json.loads(response['Payload'].read())
            
            if result.get('statusCode') == 200:
                print("✅ LinUCB training completed successfully")
                return True
            else:
                print(f"❌ LinUCB training failed: {result.get('body', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Error triggering LinUCB training: {e}")
            return False
    
    def create_experiment_events(self, args, eval_functions, methods, days_to_process, training_id):
        """Create all Lambda events for the experiment."""
        events = []
        
        for day in days_to_process:
            for eval_function in eval_functions:
                # Calculate total scenarios for this day
                total_scenarios = ((args.hour_end - args.hour_start) * 60) // args.time_interval
                
                for scenario_index in range(total_scenarios):
                    event = {
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
                            'time_interval': args.time_interval
                        },
                        'methods': methods,
                        'training_id': training_id,
                        'execution_date': datetime.now().strftime('%Y%m%d_%H%M%S')
                    }
                    
                    events.append({
                        'event': event,
                        'description': f"Day {day}, {eval_function}, Scenario {scenario_index:03d}"
                    })
        
        return events
    
    def execute_experiments(self, events, dry_run=False, parallel=1):
        """Execute all experiment events."""
        total_events = len(events)
        print(f"🚀 Executing {total_events} scenarios")
        
        if dry_run:
            print("🔍 DRY RUN MODE - Showing what would be executed:")
            for i, event_data in enumerate(events[:10]):  # Show first 10
                event = event_data['event']
                desc = event_data['description']
                print(f"  {i+1:3d}: {desc} - {event['year']}-{event['month']:02d}-{event['day']:02d}")
            
            if total_events > 10:
                print(f"  ... and {total_events - 10} more scenarios")
            return True
        
        successful = 0
        failed = 0
        
        for i, event_data in enumerate(events):
            event = event_data['event']
            desc = event_data['description']
            
            print(f"⏳ [{i+1:3d}/{total_events}] {desc}")
            
            try:
                response = self.lambda_client.invoke(
                    FunctionName=self.function_name,
                    Payload=json.dumps(event)
                )
                
                result = json.loads(response['Payload'].read())
                
                if result.get('statusCode') == 200:
                    successful += 1
                    print(f"   ✅ Success")
                else:
                    failed += 1
                    error_msg = result.get('body', 'Unknown error')
                    print(f"   ❌ Failed: {error_msg}")
                
            except Exception as e:
                failed += 1
                print(f"   ❌ Error: {e}")
            
            # Brief pause to avoid rate limiting
            if i < total_events - 1:
                time.sleep(0.5)
        
        print(f"\n📊 Execution Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📈 Success rate: {successful/total_events*100:.1f}%")
        
        return failed == 0
    
    def show_results_info(self, args, eval_functions, days_to_process, training_id):
        """Show where to find the results."""
        print(f"\n📁 Results Location:")
        
        for day in days_to_process:
            for eval_function in eval_functions:
                s3_path = f"s3://{self.bucket_name}/experiments/type={args.vehicle_type}/eval={eval_function}/borough={args.borough}/year={args.year}/month={args.month:02d}/day={day:02d}/"
                day_file = f"day_summary_{args.year}{args.month:02d}{day:02d}.json"
                print(f"   📄 {s3_path}{day_file}")
        
        print(f"\n🔍 To analyze results:")
        print(f"   python analyze_day_results.py --training_id={training_id}")
    
    def run(self):
        """Main execution function."""
        print("🧪 RIDE-HAILING PRICING EXPERIMENT CLI")
        print("=" * 60)
        
        # Parse and validate arguments
        args = self.parse_arguments()
        eval_functions, methods, days_to_process = self.validate_arguments(args)
        
        # Generate training ID
        training_id = self.generate_training_id(args)
        
        print(f"🎯 Experiment Configuration:")
        print(f"   📅 Date(s): {args.year}-{args.month:02d}-{days_to_process}")
        print(f"   🏙️ Borough: {args.borough}")
        print(f"   🚗 Vehicle: {args.vehicle_type}")
        print(f"   📊 Evaluation: {eval_functions}")
        print(f"   🔧 Methods: {methods}")
        print(f"   🆔 Training ID: {training_id}")
        print()
        
        # Check LinUCB training if needed
        if not self.check_linucb_training(args, training_id):
            print("❌ LinUCB training failed. Exiting.")
            sys.exit(1)
        
        # Create all experiment events
        events = self.create_experiment_events(args, eval_functions, methods, days_to_process, training_id)
        
        print(f"📋 Generated {len(events)} experiment scenarios")
        
        # Execute experiments
        success = self.execute_experiments(events, dry_run=args.dry_run, parallel=args.parallel)
        
        if success and not args.dry_run:
            self.show_results_info(args, eval_functions, days_to_process, training_id)
            print("\n🎉 All experiments completed successfully!")
        elif args.dry_run:
            print("\n🔍 Dry run completed. Use without --dry_run to execute.")
        else:
            print("\n❌ Some experiments failed. Check logs above.")
            sys.exit(1)

if __name__ == "__main__":
    cli = PricingExperimentCLI()
    cli.run() 