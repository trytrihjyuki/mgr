#!/usr/bin/env python3
"""
Ride-Hailing Pricing Benchmark CLI - Hikima Environment Replication

A systematic benchmarking framework that recreates the exact experimental 
environment from Hikima et al. with time windows and full dataset processing.

Examples:
    # Exact Hikima replication (10-20h, 5min windows, 120 scenarios per day)
    python run_experiment.py --year=2019 --month=10 --days=1,6 --hours=10,20 --window=5 --func=PL,Sigmoid --methods=MinMaxCostFlow,MAPS,LinUCB

    # All 4 methods with Hikima setup
    python run_experiment.py --year=2019 --month=10 --days=1 --hours=10,20 --window=5 --func=PL --methods=MinMaxCostFlow,MAPS,LinUCB,LP

    # Custom time range (24h, 30min windows, 48 scenarios per day)
    python run_experiment.py --year=2019 --month=10 --days=1 --hours=0,24 --window=30 --func=PL,Sigmoid --methods=LP,MAPS,LinUCB

    # Multi-day multi-month experiment
    python run_experiment.py --year=2019 --months=10,11 --days=1,6,11,16 --hours=10,20 --window=5 --func=PL --methods=LP,MAPS
"""

import argparse
import json
import boto3
import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RideHailingBenchmarkCLI:
    """
    Command-line interface for ride-hailing pricing benchmarks.
    Recreates exact Hikima et al. experimental environment.
    """
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """Initialize CLI with configuration."""
        self.config = self._load_config(config_path)
        self.lambda_client = boto3.client('lambda')
        self.s3_client = boto3.client('s3')
        
        # Generate unique training ID for this experiment run
        self.training_id = f"{self._generate_training_id()}"
        self.execution_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"🆔 Training ID: {self.training_id}")
        logger.info(f"📅 Execution Date: {self.execution_date}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load experiment configuration."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"❌ Configuration file not found: {config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in configuration: {e}")
            sys.exit(1)
    
    def _generate_training_id(self) -> str:
        """Generate 9-digit training ID as required."""
        import random
        return f"{random.randint(100_000_000, 999_999_999)}"
    
    def run_experiment(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Run the main experiment based on CLI arguments.
        """
        logger.info("🚀 Starting Ride-Hailing Pricing Benchmark - Hikima Environment")
        logger.info(f"📊 Configuration: {args}")
        
        # Parse and validate arguments
        experiment_params = self._parse_experiment_params(args)
        
        # Generate time-window based experiment plan (Hikima style)
        experiment_plan = self._generate_time_window_experiment_plan(experiment_params)
        total_scenarios = sum(len(day_scenarios) for day_scenarios in experiment_plan.values())
        logger.info(f"📋 Generated {total_scenarios} time-window scenarios across {len(experiment_plan)} days")
        
        # Execute experiments by day
        results = []
        day_count = 0
        for (year, month, day), scenarios in experiment_plan.items():
            day_count += 1
            logger.info(f"🔄 Processing day {day_count}/{len(experiment_plan)}: {year}-{month:02d}-{day:02d}")
            logger.info(f"   📊 {len(scenarios)} time windows × {len(experiment_params['acceptance_functions'])} functions = {len(scenarios) * len(experiment_params['acceptance_functions'])} experiments")
            
            day_results = []
            for scenario_idx, scenario in enumerate(scenarios):
                for func_idx, acceptance_function in enumerate(experiment_params['acceptance_functions']):
                    experiment_idx = scenario_idx * len(experiment_params['acceptance_functions']) + func_idx + 1
                    total_day_experiments = len(scenarios) * len(experiment_params['acceptance_functions'])
                    
                    logger.info(f"   🕐 Time window {experiment_idx}/{total_day_experiments}: "
                               f"{scenario['start_hour']:02d}:{scenario['start_minute']:02d}-"
                               f"{scenario['end_hour']:02d}:{scenario['end_minute']:02d} | {acceptance_function}")
                    
                    try:
                        # Create experiment payload
                        experiment_payload = {
                            'training_id': self.training_id,
                            'execution_date': self.execution_date,
                            'year': year,
                            'month': month,
                            'day': day,
                            'scenario_index': scenario_idx,
                            'time_window': scenario,
                            'vehicle_type': experiment_params['vehicle_type'],
                            'borough': experiment_params['borough'],
                            'acceptance_function': acceptance_function,
                            'methods': experiment_params['methods'],
                            'scenario': experiment_params['scenario'],
                            'config_name': 'experiment_config.json'
                        }
                        
                        result = self._run_single_experiment(experiment_payload)
                        day_results.append(result)
                        
                        # Log success
                        if result.get('statusCode') == 200:
                            try:
                                body = json.loads(result['body'])
                                if 'body' in body and isinstance(body['body'], str):
                                    experiment_results = json.loads(body['body'])
                                    results_array = experiment_results.get('results', [])
                                else:
                                    results_array = body.get('results', [])
                                
                                logger.info(f"      ✅ Success: {len(results_array)} method results")
                                if results_array:
                                    for res in results_array:
                                        method = res.get('method_name', 'Unknown')
                                        objective = res.get('objective_value', 0)
                                        logger.info(f"         📊 {method}: Objective={objective:.2f}")
                            except Exception as e:
                                logger.warning(f"      ⚠️ Could not parse results: {e}")
                        else:
                            logger.warning(f"      ⚠️ Experiment returned status {result.get('statusCode')}")
                            
                    except Exception as e:
                        logger.error(f"      ❌ Experiment failed: {e}")
                        continue
                    
                    # Add delay between experiments to avoid rate limiting
                    time.sleep(1)
            
            results.extend(day_results)
            logger.info(f"✅ Completed day {year}-{month:02d}-{day:02d}: {len(day_results)} experiments")
        
        # Generate summary
        summary = self._generate_summary(results, experiment_params, total_scenarios)
        logger.info("🎉 Benchmark completed!")
        logger.info(f"📊 Summary: {summary['total_experiments']} experiments, "
                   f"{summary['successful_experiments']} successful")
        
        return summary
    
    def _parse_experiment_params(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Parse and validate CLI arguments into experiment parameters."""
        params = {}
        
        # Required parameters
        params['year'] = args.year
        
        # Parse months
        if hasattr(args, 'months') and args.months:
            params['months'] = [int(m.strip()) for m in args.months.split(',')]
        else:
            params['months'] = [args.month] if args.month else [10]  # Default to October
        
        # Parse days
        if args.days:
            params['days'] = [int(d.strip()) for d in args.days.split(',')]
        else:
            params['days'] = [1]  # Default to 1st
        
        # Parse time range (hours)
        if args.hours:
            hour_parts = args.hours.split(',')
            if len(hour_parts) != 2:
                raise ValueError("Hours must be specified as 'start,end' (e.g., '10,20')")
            params['start_hour'] = int(hour_parts[0])
            params['end_hour'] = int(hour_parts[1])
        else:
            # Default to Hikima business hours
            params['start_hour'] = 10
            params['end_hour'] = 20
        
        # Parse time window duration
        params['window_minutes'] = args.window if args.window else 5  # Default 5 minutes
        
        # Parse acceptance functions
        if args.func:
            params['acceptance_functions'] = [f.strip() for f in args.func.split(',')]
        else:
            params['acceptance_functions'] = ['PL']  # Default
        
        # Parse methods
        if args.methods:
            params['methods'] = [m.strip() for m in args.methods.split(',')]
        else:
            params['methods'] = ['MinMaxCostFlow', 'MAPS', 'LinUCB']  # Default
        
        # Optional parameters
        params['vehicle_type'] = getattr(args, 'vehicle_type', 'green')
        params['borough'] = getattr(args, 'borough', 'Manhattan')
        params['scenario'] = getattr(args, 'scenario', 'comprehensive')
        
        # Validate parameters
        self._validate_params(params)
        
        return params
    
    def _validate_params(self, params: Dict[str, Any]):
        """Validate experiment parameters."""
        config = self.config
        
        # Validate time parameters
        if params['start_hour'] < 0 or params['start_hour'] > 23:
            raise ValueError(f"Invalid start hour: {params['start_hour']}. Must be 0-23.")
        if params['end_hour'] < 1 or params['end_hour'] > 24:
            raise ValueError(f"Invalid end hour: {params['end_hour']}. Must be 1-24.")
        if params['start_hour'] >= params['end_hour']:
            raise ValueError(f"Start hour ({params['start_hour']}) must be less than end hour ({params['end_hour']})")
        
        if params['window_minutes'] < 1 or params['window_minutes'] > 60:
            raise ValueError(f"Invalid window minutes: {params['window_minutes']}. Must be 1-60.")
        
        # Validate vehicle types
        valid_vehicle_types = config['data_sources']['supported_vehicle_types']
        if params['vehicle_type'] not in valid_vehicle_types:
            raise ValueError(f"Invalid vehicle type: {params['vehicle_type']}. "
                           f"Supported: {valid_vehicle_types}")
        
        # Validate boroughs
        valid_boroughs = config['data_sources']['supported_boroughs']
        if params['borough'] not in valid_boroughs:
            raise ValueError(f"Invalid borough: {params['borough']}. "
                           f"Supported: {valid_boroughs}")
        
        # Validate acceptance functions
        valid_functions = list(config['acceptance_functions'].keys())
        for func in params['acceptance_functions']:
            if func not in valid_functions:
                raise ValueError(f"Invalid acceptance function: {func}. "
                               f"Supported: {valid_functions}")
        
        # Validate methods
        valid_methods = list(config['pricing_methods'].keys())
        for method in params['methods']:
            if method not in valid_methods:
                raise ValueError(f"Invalid pricing method: {method}. "
                               f"Supported: {valid_methods}")
    
    def _generate_time_window_experiment_plan(self, params: Dict[str, Any]) -> Dict[tuple, List[Dict[str, Any]]]:
        """Generate time-window based experiment plan following Hikima methodology."""
        plan = {}
        
        for month in params['months']:
            for day in params['days']:
                # Generate time windows for this day
                scenarios = self._generate_time_windows(
                    params['start_hour'], 
                    params['end_hour'], 
                    params['window_minutes']
                )
                plan[(params['year'], month, day)] = scenarios
        
        return plan
    
    def _generate_time_windows(self, start_hour: int, end_hour: int, window_minutes: int) -> List[Dict[str, Any]]:
        """Generate time windows following Hikima's 5-minute interval approach."""
        scenarios = []
        
        # Calculate total minutes in the time range
        total_minutes = (end_hour - start_hour) * 60
        
        # Generate scenarios every 5 minutes (following Hikima's tt_tmp * 5 approach)
        for tt_tmp in range(0, total_minutes // 5):
            tt = tt_tmp * 5  # 5-minute intervals
            h, m = divmod(tt, 60)
            
            start_scenario_hour = start_hour + h
            start_scenario_minute = m
            
            # End time is start + window_minutes + 30 seconds (following Hikima)
            end_scenario_minute = start_scenario_minute + window_minutes
            end_scenario_hour = start_scenario_hour
            
            # Handle minute overflow
            if end_scenario_minute >= 60:
                end_scenario_hour += end_scenario_minute // 60
                end_scenario_minute = end_scenario_minute % 60
            
            # Stop if we exceed the end hour
            if end_scenario_hour > end_hour or (end_scenario_hour == end_hour and end_scenario_minute > 0):
                break
            
            scenario = {
                'scenario_index': tt_tmp,
                'start_hour': start_scenario_hour,
                'start_minute': start_scenario_minute,
                'start_second': 0,
                'end_hour': end_scenario_hour,
                'end_minute': end_scenario_minute,
                'end_second': 30,  # Hikima adds 30 seconds
                'window_minutes': window_minutes
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def _run_single_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single time-window experiment via AWS Lambda."""
        
        # Invoke Lambda function
        response = self.lambda_client.invoke(
            FunctionName='rideshare-pricing-benchmark',
            InvocationType='RequestResponse',
            Payload=json.dumps(experiment)
        )
        
        # Parse response
        result = {
            'statusCode': response['StatusCode'],
            'payload': experiment,
            'response': response
        }
        
        if response['StatusCode'] == 200:
            result['body'] = response['Payload'].read().decode('utf-8')
        else:
            result['error'] = response.get('FunctionError', 'Unknown error')
        
        return result
    
    def _generate_summary(self, results: List[Dict[str, Any]], 
                         params: Dict[str, Any], total_scenarios: int) -> Dict[str, Any]:
        """Generate experiment summary."""
        successful = [r for r in results if r.get('statusCode') == 200]
        failed = [r for r in results if r.get('statusCode') != 200]
        
        summary = {
            'training_id': self.training_id,
            'execution_date': self.execution_date,
            'timestamp': datetime.now().isoformat(),
            'experiment_parameters': params,
            'total_scenarios': total_scenarios,
            'total_experiments': len(results),
            'successful_experiments': len(successful),
            'failed_experiments': len(failed),
            'results': results,
            's3_pattern': self.config['aws_deployment']['s3_result_pattern']
        }
        
        # Add S3 locations for successful experiments
        s3_locations = []
        for result in successful:
            if result.get('body'):
                try:
                    body = json.loads(result['body'])
                    # Handle nested Lambda response structure
                    if 'body' in body and isinstance(body['body'], str):
                        experiment_results = json.loads(body['body'])
                        if 's3_location' in experiment_results:
                            s3_locations.append(experiment_results['s3_location'])
                    else:
                        if 's3_location' in body:
                            s3_locations.append(body['s3_location'])
                except:
                    pass
        
        summary['s3_locations'] = s3_locations
        
        return summary


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Ride-Hailing Pricing Benchmark CLI - Hikima Environment Replication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Exact Hikima replication (10-20h, 5min windows, 120 scenarios per day)
  python run_experiment.py --year=2019 --month=10 --days=1,6 --hours=10,20 --window=5 --func=PL,Sigmoid --methods=MinMaxCostFlow,MAPS,LinUCB

  # All 4 methods with Hikima setup
  python run_experiment.py --year=2019 --month=10 --days=1 --hours=10,20 --window=5 --func=PL --methods=MinMaxCostFlow,MAPS,LinUCB,LP

  # Custom time range (24h, 30min windows, 48 scenarios per day)
  python run_experiment.py --year=2019 --month=10 --days=1 --hours=0,24 --window=30 --func=PL,Sigmoid --methods=LP,MAPS,LinUCB

  # Multi-day multi-month experiment  
  python run_experiment.py --year=2019 --months=10,11 --days=1,6,11,16 --hours=10,20 --window=5 --func=PL --methods=LP,MAPS
        """
    )
    
    # Required arguments
    parser.add_argument('--year', type=int, required=True,
                       help='Year for the experiment (e.g., 2019)')
    
    # Time specification (month OR months)
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument('--month', type=int,
                           help='Single month for the experiment (1-12)')
    time_group.add_argument('--months', type=str,
                           help='Comma-separated months (e.g., "10,11,12")')
    
    # Day specification
    parser.add_argument('--days', type=str, required=True,
                       help='Comma-separated days (e.g., "1,6" for Hikima replication)')
    
    # Time range specification (Hikima style)
    parser.add_argument('--hours', type=str, required=True,
                       help='Start and end hours as "start,end" (e.g., "10,20" for business hours)')
    
    parser.add_argument('--window', type=int, default=5,
                       help='Time window duration in minutes (default: 5 for Hikima replication)')
    
    # Acceptance functions
    parser.add_argument('--func', type=str, required=True,
                       help='Comma-separated acceptance functions (PL,Sigmoid)')
    
    # Pricing methods
    parser.add_argument('--methods', type=str, required=True,
                       help='Comma-separated methods (MinMaxCostFlow,MAPS,LinUCB,LP)')
    
    # Optional arguments
    parser.add_argument('--vehicle-type', default='green',
                       choices=['green', 'yellow', 'fhv'],
                       help='Vehicle type (default: green)')
    
    parser.add_argument('--borough', default='Manhattan',
                       choices=['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'],
                       help='Borough for analysis (default: Manhattan)')
    
    parser.add_argument('--scenario', default='comprehensive',
                       choices=['hikima_replication', 'comprehensive', 'scalability'],
                       help='Experiment scenario (default: comprehensive)')
    
    parser.add_argument('--config', default='configs/experiment_config.json',
                       help='Path to experiment configuration file')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        cli = RideHailingBenchmarkCLI(args.config)
        summary = cli.run_experiment(args)
        
        # Print summary
        print("\n" + "="*80)
        print("🎉 EXPERIMENT SUMMARY - Hikima Environment Replication")
        print("="*80)
        print(f"Training ID: {summary['training_id']}")
        print(f"Execution Date: {summary['execution_date']}")
        print(f"Total time-window scenarios: {summary['total_scenarios']}")
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Successful: {summary['successful_experiments']}")
        print(f"Failed: {summary['failed_experiments']}")
        
        if summary['s3_locations']:
            print(f"\n📁 Results stored in S3 ({len(summary['s3_locations'])} files):")
            for i, location in enumerate(summary['s3_locations'][:5]):  # Show first 5
                print(f"  {location}")
            if len(summary['s3_locations']) > 5:
                print(f"  ... and {len(summary['s3_locations']) - 5} more files")
        
        print("\n🔍 Use the following pattern to find all results:")
        try:
            pattern_template = summary['s3_pattern']
            # Replace placeholders with wildcards for search pattern
            pattern_template = pattern_template.replace('{execution_date}', '*')
            pattern_template = pattern_template.replace('{training_id}', '*')
            
            if 'month:02d' in pattern_template:
                pattern = pattern_template.replace('{month:02d}', '*').replace('{day:02d}', '*').format(
                    vehicle_type=args.vehicle_type,
                    acceptance_function="*",
                    year=args.year
                )
            else:
                pattern = pattern_template.format(
                    vehicle_type=args.vehicle_type,
                    acceptance_function="*",
                    year=args.year,
                    month="*",
                    day="*"
                )
            print(f"  s3://magisterka/{pattern}")
        except Exception as e:
            logger.error(f"❌ Failed to generate S3 pattern: {e}")
            print(f"  Check S3 bucket manually: s3://magisterka/experiments/")
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 