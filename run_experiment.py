#!/usr/bin/env python3
"""
Ride-Hailing Pricing Benchmark CLI

A systematic benchmarking framework for comparing pricing methods in ride-hailing.
Supports the exact Hikima experimental setup and extended analysis.

Examples:
    # Hikima replication (2 days, business hours)
    python run_experiment.py --year=2019 --month=10 --days=1,6 --func=PL,Sigmoid --methods=MinMaxCostFlow,MAPS,LinUCB

    # Comprehensive analysis (all 4 methods)
    python run_experiment.py --year=2019 --month=10 --days=1 --func=PL --methods=MinMaxCostFlow,MAPS,LinUCB,LP

    # Extended multi-day analysis
    python run_experiment.py --year=2019 --month=10 --days=1,2,3,4,5,6,7 --func=PL,Sigmoid --methods=LP,MAPS,LinUCB
    
    # Multi-month experiment
    python run_experiment.py --year=2019 --months=3,4,5 --days=1,15 --func=PL --methods=LP,MAPS
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
    """
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """Initialize CLI with configuration."""
        self.config = self._load_config(config_path)
        self.lambda_client = boto3.client('lambda')
        self.s3_client = boto3.client('s3')
        
        # Generate unique training ID for this experiment run
        self.training_id = f"{self._generate_training_id()}"
        logger.info(f"🆔 Training ID: {self.training_id}")
    
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
        logger.info("🚀 Starting Ride-Hailing Pricing Benchmark")
        logger.info(f"📊 Configuration: {args}")
        
        # Parse and validate arguments
        experiment_params = self._parse_experiment_params(args)
        
        # Generate experiment plan
        experiment_plan = self._generate_experiment_plan(experiment_params)
        logger.info(f"📋 Generated {len(experiment_plan)} experiment(s)")
        
        # Execute experiments
        results = []
        for i, experiment in enumerate(experiment_plan):
            logger.info(f"🔄 Running experiment {i+1}/{len(experiment_plan)}")
            logger.info(f"   📅 {experiment['year']}-{experiment['month']:02d}-{experiment['day']:02d}")
            logger.info(f"   🚗 {experiment['vehicle_type']} | 🏙️ {experiment['borough']}")
            logger.info(f"   📊 {experiment['acceptance_function']} | 🧮 {experiment['methods']}")
            
            try:
                result = self._run_single_experiment(experiment)
                results.append(result)
                
                # Log success
                if result.get('statusCode') == 200:
                    body = json.loads(result['body'])
                    logger.info(f"   ✅ Success: {len(body.get('results', []))} method results")
                else:
                    logger.warning(f"   ⚠️ Experiment returned status {result.get('statusCode')}")
                    
            except Exception as e:
                logger.error(f"   ❌ Experiment failed: {e}")
                continue
            
            # Add delay between experiments to avoid rate limiting
            if i < len(experiment_plan) - 1:
                time.sleep(2)
        
        # Generate summary
        summary = self._generate_summary(results, experiment_params)
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
        params['time_range'] = getattr(args, 'time_range', 'business_hours')
        
        # Validate parameters
        self._validate_params(params)
        
        return params
    
    def _validate_params(self, params: Dict[str, Any]):
        """Validate experiment parameters."""
        config = self.config
        
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
        
        # Validate time range
        valid_time_ranges = list(config['temporal_config'].keys())
        if params['time_range'] not in valid_time_ranges:
            raise ValueError(f"Invalid time range: {params['time_range']}. "
                           f"Supported: {valid_time_ranges}")
    
    def _generate_experiment_plan(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate a list of individual experiments to run."""
        experiments = []
        
        for month in params['months']:
            for day in params['days']:
                for acceptance_function in params['acceptance_functions']:
                    # Create one experiment per acceptance function
                    # (methods will be run within the same Lambda call)
                    experiment = {
                        'training_id': self.training_id,
                        'year': params['year'],
                        'month': month,
                        'day': day,
                        'vehicle_type': params['vehicle_type'],
                        'borough': params['borough'],
                        'acceptance_function': acceptance_function,
                        'methods': params['methods'],
                        'scenario': params['scenario'],
                        'time_range': params['time_range']
                    }
                    experiments.append(experiment)
        
        return experiments
    
    def _run_single_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment via AWS Lambda."""
        
        # Prepare Lambda payload
        payload = {
            'training_id': experiment['training_id'],
            'year': experiment['year'],
            'month': experiment['month'],
            'day': experiment['day'],
            'vehicle_type': experiment['vehicle_type'],
            'borough': experiment['borough'],
            'acceptance_function': experiment['acceptance_function'],
            'methods': experiment['methods'],
            'scenario': experiment['scenario'],
            'time_range': experiment['time_range'],
            'config_name': 'experiment_config.json'
        }
        
        # Invoke Lambda function
        response = self.lambda_client.invoke(
            FunctionName='rideshare-pricing-benchmark',
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        # Parse response
        result = {
            'statusCode': response['StatusCode'],
            'payload': payload,
            'response': response
        }
        
        if response['StatusCode'] == 200:
            result['body'] = response['Payload'].read().decode('utf-8')
        else:
            result['error'] = response.get('FunctionError', 'Unknown error')
        
        return result
    
    def _generate_summary(self, results: List[Dict[str, Any]], 
                         params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate experiment summary."""
        successful = [r for r in results if r.get('statusCode') == 200]
        failed = [r for r in results if r.get('statusCode') != 200]
        
        summary = {
            'training_id': self.training_id,
            'timestamp': datetime.now().isoformat(),
            'experiment_parameters': params,
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
                    if 's3_location' in body:
                        s3_locations.append(body['s3_location'])
                except:
                    pass
        
        summary['s3_locations'] = s3_locations
        
        return summary


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Ride-Hailing Pricing Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hikima replication (2 days, PL + Sigmoid)
  python run_experiment.py --year=2019 --month=10 --days=1,6 --func=PL,Sigmoid --methods=MinMaxCostFlow,MAPS,LinUCB

  # All 4 methods comparison
  python run_experiment.py --year=2019 --month=10 --days=1 --func=PL --methods=MinMaxCostFlow,MAPS,LinUCB,LP

  # Extended multi-day analysis
  python run_experiment.py --year=2019 --month=10 --days=1,2,3,4,5,6,7 --func=PL,Sigmoid --methods=LP,MAPS,LinUCB

  # Multi-month experiment
  python run_experiment.py --year=2019 --months=3,4,5 --days=1,15 --func=PL --methods=LP,MAPS
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
                           help='Comma-separated months (e.g., "3,4,5")')
    
    # Day specification
    parser.add_argument('--days', type=str, required=True,
                       help='Comma-separated days (e.g., "1,6" for Hikima replication)')
    
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
    
    parser.add_argument('--time-range', default='business_hours',
                       choices=['business_hours', 'full_day'],
                       help='Time range for analysis (default: business_hours)')
    
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
        print("\n" + "="*60)
        print("🎉 EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Training ID: {summary['training_id']}")
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Successful: {summary['successful_experiments']}")
        print(f"Failed: {summary['failed_experiments']}")
        
        if summary['s3_locations']:
            print("\n📁 Results stored in S3:")
            for location in summary['s3_locations']:
                print(f"  {location}")
        
        print("\n🔍 Use the following pattern to find all results:")
        pattern = summary['s3_pattern'].format(
            vehicle_type=args.vehicle_type,
            acceptance_function="*",
            year=args.year,
            month="*"
        )
        print(f"  s3://magisterka/{pattern}")
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 