#!/usr/bin/env python3
"""
Ride-Hailing Pricing Benchmark CLI

A command-line interface for running pricing method benchmarks.

Usage:
    python run_benchmark.py hikima-replication
    python run_benchmark.py comprehensive --borough Manhattan --year 2019 --month 10
    python run_benchmark.py custom --methods HikimaMinMaxCostFlow,MAPS --acceptance PL,Sigmoid
"""

import argparse
import json
import boto3
import sys
from datetime import datetime
from typing import List, Optional


class BenchmarkCLI:
    """Command-line interface for running pricing benchmarks."""
    
    def __init__(self):
        self.lambda_client = boto3.client('lambda')
        self.function_name = 'rideshare-experiment-runner'
    
    def run_experiment(self, payload: dict) -> dict:
        """Invoke the Lambda function with the given payload."""
        print(f"🚀 Starting experiment: {payload.get('scenario', 'custom')}")
        print(f"📋 Configuration: {json.dumps(payload, indent=2)}")
        
        try:
            response = self.lambda_client.invoke(
                FunctionName=self.function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            result = json.loads(response['Payload'].read().decode('utf-8'))
            
            if response['StatusCode'] == 200:
                body = json.loads(result['body'])
                return body
            else:
                print(f"❌ Lambda invocation failed: {result}")
                return None
                
        except Exception as e:
            print(f"❌ Error invoking Lambda: {e}")
            return None
    
    def hikima_replication(self, args) -> dict:
        """Run exact Hikima replication experiment."""
        payload = {
            'scenario': 'hikima_replication',
            'vehicle_type': args.vehicle_type,
            'year': args.year,
            'month': args.month,
            'day': args.day,
            'borough': args.borough,
            'config_name': 'benchmark_config.json'
        }
        return self.run_experiment(payload)
    
    def comprehensive_benchmark(self, args) -> dict:
        """Run comprehensive benchmark with all 4 methods."""
        payload = {
            'scenario': 'comprehensive_benchmark',
            'vehicle_type': args.vehicle_type,
            'year': args.year,
            'month': args.month,
            'day': args.day,
            'borough': args.borough,
            'config_name': 'benchmark_config.json'
        }
        return self.run_experiment(payload)
    
    def extended_analysis(self, args) -> dict:
        """Run extended multi-day analysis."""
        payload = {
            'scenario': 'extended_analysis',
            'vehicle_type': args.vehicle_type,
            'year': args.year,
            'month': args.month,
            'borough': args.borough,
            'config_name': 'benchmark_config.json'
        }
        return self.run_experiment(payload)
    
    def full_day_analysis(self, args) -> dict:
        """Run 24-hour analysis."""
        payload = {
            'scenario': 'full_day_analysis',
            'vehicle_type': args.vehicle_type,
            'year': args.year,
            'month': args.month,
            'day': args.day,
            'borough': args.borough,
            'config_name': 'benchmark_config.json'
        }
        return self.run_experiment(payload)
    
    def custom_experiment(self, args) -> dict:
        """Run custom experiment with user-specified parameters."""
        payload = {
            'vehicle_type': args.vehicle_type,
            'year': args.year,
            'month': args.month,
            'day': args.day,
            'borough': args.borough,
            'start_hour': args.start_hour,
            'end_hour': args.end_hour,
            'config_name': 'benchmark_config.json'
        }
        
        if args.methods:
            payload['methods'] = args.methods.split(',')
        
        if args.acceptance:
            payload['acceptance_functions'] = args.acceptance.split(',')
        
        return self.run_experiment(payload)
    
    def print_results(self, results: dict):
        """Print experiment results in a formatted way."""
        if not results:
            return
        
        print("\n" + "="*80)
        print(f"🎉 EXPERIMENT RESULTS: {results.get('experiment_id', 'Unknown')}")
        print("="*80)
        
        # Print configuration
        config = results.get('configuration', {})
        print(f"📅 Date: {config.get('year')}-{config.get('month'):02d}-{config.get('day'):02d}")
        print(f"🚗 Vehicle: {config.get('vehicle_type')}")
        print(f"🏙️ Borough: {config.get('borough')}")
        print(f"⏰ Time Range: {config.get('time_range')}")
        print(f"🧪 Scenario: {config.get('scenario')}")
        
        # Print data summary
        data_summary = results.get('data_summary', {})
        print(f"📊 Total Records: {data_summary.get('total_records')}")
        print(f"🙋 Requesters: {data_summary.get('requesters')}")
        print(f"🚕 Taxis: {data_summary.get('taxis')}")
        
        # Print method results
        method_results = results.get('results', [])
        if method_results:
            print(f"\n📈 METHOD PERFORMANCE:")
            print("-" * 100)
            print(f"{'Method':<25} {'Acceptance':<12} {'Objective':<12} {'Time(s)':<8} {'Matches':<8} {'Avg Price':<10}")
            print("-" * 100)
            
            for result in method_results:
                method = result.get('method_name', 'Unknown')[:24]
                acceptance = result.get('additional_metrics', {}).get('acceptance_function', 'N/A')[:11]
                objective = f"{result.get('objective_value', 0):.2f}"
                time_val = f"{result.get('computation_time', 0):.3f}"
                matches = str(result.get('n_matches', 0))
                avg_price = f"${result.get('average_price', 0):.2f}"
                
                print(f"{method:<25} {acceptance:<12} {objective:<12} {time_val:<8} {matches:<8} {avg_price:<10}")
        
        print("-" * 100)
        print(f"⏱️ Total Execution Time: {results.get('execution_time_seconds', 0):.2f} seconds")
        
        if 's3_location' in results:
            print(f"💾 Results saved to: {results['s3_location']}")
        
        print("="*80 + "\n")


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description='Run ride-hailing pricing benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hikima replication experiment
  python run_benchmark.py hikima-replication
  
  # Comprehensive benchmark
  python run_benchmark.py comprehensive --borough Brooklyn --year 2019
  
  # Custom experiment
  python run_benchmark.py custom --methods HikimaMinMaxCostFlow,LinearProgram --acceptance PL,Sigmoid
  
  # Full day analysis
  python run_benchmark.py full-day --vehicle-type yellow --month 11
  
  # Extended multi-day analysis
  python run_benchmark.py extended --borough Queens
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Experiment type')
    
    # Common arguments
    def add_common_args(parser):
        parser.add_argument('--vehicle-type', default='green', 
                          choices=['green', 'yellow', 'fhv'],
                          help='Vehicle type (default: green)')
        parser.add_argument('--year', type=int, default=2019,
                          help='Year (default: 2019)')
        parser.add_argument('--month', type=int, default=10,
                          help='Month (default: 10)')
        parser.add_argument('--day', type=int, default=1,
                          help='Day (default: 1)')
        parser.add_argument('--borough', default='Manhattan',
                          choices=['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'],
                          help='NYC borough (default: Manhattan)')
    
    # Hikima replication
    hikima_parser = subparsers.add_parser('hikima-replication', 
                                         help='Exact Hikima et al. replication')
    add_common_args(hikima_parser)
    
    # Comprehensive benchmark
    comp_parser = subparsers.add_parser('comprehensive',
                                       help='All 4 methods benchmark')
    add_common_args(comp_parser)
    
    # Extended analysis
    ext_parser = subparsers.add_parser('extended',
                                      help='Multi-day robustness testing')
    add_common_args(ext_parser)
    
    # Full day analysis
    full_parser = subparsers.add_parser('full-day',
                                       help='24-hour temporal analysis')
    add_common_args(full_parser)
    
    # Custom experiment
    custom_parser = subparsers.add_parser('custom',
                                         help='Custom experiment configuration')
    add_common_args(custom_parser)
    custom_parser.add_argument('--start-hour', type=int, default=10,
                              help='Start hour (default: 10)')
    custom_parser.add_argument('--end-hour', type=int, default=20,
                              help='End hour (default: 20)')
    custom_parser.add_argument('--methods',
                              help='Comma-separated method names (e.g., HikimaMinMaxCostFlow,MAPS)')
    custom_parser.add_argument('--acceptance',
                              help='Comma-separated acceptance functions (e.g., PL,Sigmoid)')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    cli = BenchmarkCLI()
    
    # Map commands to methods
    command_map = {
        'hikima-replication': cli.hikima_replication,
        'comprehensive': cli.comprehensive_benchmark,
        'extended': cli.extended_analysis,
        'full-day': cli.full_day_analysis,
        'custom': cli.custom_experiment
    }
    
    if args.command in command_map:
        print(f"🔬 Running {args.command} experiment...")
        results = command_map[args.command](args)
        
        if results:
            cli.print_results(results)
            
            if results.get('status') == 'failed':
                print(f"❌ Experiment failed: {results.get('error')}")
                sys.exit(1)
            else:
                print("✅ Experiment completed successfully!")
        else:
            print("❌ Experiment failed!")
            sys.exit(1)
    else:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main() 