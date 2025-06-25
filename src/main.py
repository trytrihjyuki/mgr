#!/usr/bin/env python3
"""
Rideshare Pricing Optimization Benchmark Suite

Main CLI interface for running pricing algorithm benchmarks on NYC taxi data.

Usage:
    python -m src.main --help
    python -m src.main run --config configs/default.json
    python -m src.main run --vehicle-type green --year 2019 --month 10 --methods hikima,maps
    python -m src.main validate-config configs/hikima_replication.json
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any

from .utils.config import Config
from .experiments.runner import ExperimentRunner
from .experiments.evaluator import ResultsEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Rideshare Pricing Optimization Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python -m src.main run --config configs/default.json
  
  # Run Hikima replication study
  python -m src.main run --config configs/hikima_replication.json
  
  # Run custom experiment via CLI args
  python -m src.main run \\
    --vehicle-type green \\
    --year 2019 \\
    --month 10 \\
    --borough Manhattan \\
    --start-hour 10 \\
    --end-hour 20 \\
    --methods hikima,maps,linucb,linear_program \\
    --simulation-range 120 \\
    --num-evaluations 100
  
  # Run multi-day study
  python -m src.main run \\
    --config configs/extended_study.json \\
    --start-day 1 \\
    --end-day 7
  
  # Validate configuration file
  python -m src.main validate-config configs/default.json
  
  # List available datasets
  python -m src.main list-data
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run experiment command
    run_parser = subparsers.add_parser('run', help='Run pricing optimization experiment')
    add_run_arguments(run_parser)
    run_parser.set_defaults(func=run_experiment)
    
    # Validate config command
    validate_parser = subparsers.add_parser('validate-config', help='Validate configuration file')
    validate_parser.add_argument('config_file', help='Path to configuration file')
    validate_parser.set_defaults(func=validate_config)
    
    # List data command
    list_parser = subparsers.add_parser('list-data', help='List available datasets')
    list_parser.add_argument('--bucket', default='rideshare-benchmark-data', help='S3 bucket name')
    list_parser.set_defaults(func=list_data)
    
    # Analyze results command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze experiment results')
    analyze_parser.add_argument('results_file', help='Path to results JSON file')
    analyze_parser.add_argument('--output', help='Output file for analysis report')
    analyze_parser.set_defaults(func=analyze_results)
    
    return parser


def add_run_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the run command."""
    
    # Configuration file (optional)
    parser.add_argument('--config', '-c', help='Configuration file path')
    
    # Data parameters
    parser.add_argument('--vehicle-type', choices=['green', 'yellow', 'fhv'], 
                       help='Type of vehicle data')
    parser.add_argument('--year', type=int, help='Data year')
    parser.add_argument('--month', type=int, choices=range(1, 13), help='Data month')
    parser.add_argument('--borough', choices=['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island', 'All'],
                       help='NYC borough to analyze')
    
    # Time parameters
    parser.add_argument('--start-hour', type=int, choices=range(0, 24), 
                       help='Start hour (0-23)')
    parser.add_argument('--end-hour', type=int, choices=range(1, 25), 
                       help='End hour (1-24)')
    parser.add_argument('--start-day', type=int, choices=range(1, 32), 
                       help='Start day for multi-day experiments')
    parser.add_argument('--end-day', type=int, choices=range(1, 32), 
                       help='End day for multi-day experiments')
    
    # Experiment parameters
    parser.add_argument('--methods', help='Comma-separated list of methods: hikima,maps,linucb,linear_program')
    parser.add_argument('--acceptance-function', choices=['PL', 'Sigmoid'], 
                       help='Customer acceptance function type')
    parser.add_argument('--simulation-range', type=int, 
                       help='Number of simulation scenarios')
    parser.add_argument('--num-evaluations', type=int, 
                       help='Number of Monte Carlo evaluations per scenario')
    
    # Output options
    parser.add_argument('--output-dir', default='./results', 
                       help='Output directory for results')
    parser.add_argument('--save-config', help='Save effective configuration to file')
    
    # Logging options
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Suppress non-error output')


def run_experiment(args):
    """Run pricing optimization experiment."""
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Load configuration
    config = load_config(args)
    
    # Validate configuration
    try:
        config.validate()
        logger.info(f"✅ Configuration validated")
    except ValueError as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        sys.exit(1)
    
    # Log experiment info
    experiment_id = config.get_experiment_id()
    logger.info(f"🧪 Starting experiment: {experiment_id}")
    
    if config.is_hikima_replication():
        logger.info("📋 This is a Hikima paper replication study")
    
    # Save effective configuration if requested
    if args.save_config:
        config.to_json(args.save_config)
        logger.info(f"💾 Saved configuration to: {args.save_config}")
    
    # Run experiment
    runner = ExperimentRunner(config)
    results = runner.run_experiment()
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"{experiment_id}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"💾 Results saved to: {results_file}")
    
    # Print summary
    print_experiment_summary(results)


def validate_config(args):
    """Validate a configuration file."""
    try:
        config = Config.from_json(args.config_file)
        config.validate()
        print(f"✅ Configuration file {args.config_file} is valid")
        
        # Print summary
        print(f"\nConfiguration Summary:")
        print(f"  Experiment: {config.experiment.vehicle_type} {config.experiment.year}-{config.experiment.month:02d}")
        print(f"  Borough: {config.experiment.borough}")
        print(f"  Time: {config.experiment.start_hour:02d}:00-{config.experiment.end_hour:02d}:00")
        print(f"  Methods: {', '.join(config.experiment.methods)}")
        print(f"  Scenarios: {config.experiment.simulation_range}")
        print(f"  Evaluations: {config.experiment.num_evaluations}")
        
        if config.is_hikima_replication():
            print(f"  📋 This replicates the original Hikima paper setup")
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)


def list_data(args):
    """List available datasets."""
    from .utils.aws_utils import S3DataManager
    
    s3_manager = S3DataManager(bucket_name=args.bucket)
    datasets = s3_manager.list_available_datasets()
    
    if not datasets:
        print("No datasets found in S3 bucket")
        return
    
    print(f"Available datasets in bucket '{args.bucket}':")
    print(f"{'Vehicle':<8} {'Year':<6} {'Month':<6} {'Size (MB)':<10} {'Last Modified':<20}")
    print("-" * 60)
    
    for dataset in datasets:
        print(f"{dataset['vehicle_type']:<8} {dataset['year']:<6} {dataset['month']:<6} "
              f"{dataset['size_mb']:<10} {dataset['last_modified'][:19]:<20}")


def analyze_results(args):
    """Analyze experiment results."""
    evaluator = ResultsEvaluator()
    
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    analysis = evaluator.analyze_results(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"Analysis saved to: {args.output}")
    else:
        print(json.dumps(analysis, indent=2, default=str))


def load_config(args) -> Config:
    """Load configuration from file and/or CLI arguments."""
    
    if args.config:
        # Load from file
        config = Config.from_json(args.config)
        logger.info(f"📋 Loaded configuration from: {args.config}")
    else:
        # Use default configuration
        config = Config()
        logger.info(f"📋 Using default configuration")
    
    # Override with CLI arguments
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    
    # Remove non-config arguments
    non_config_args = {'config', 'output_dir', 'save_config', 'verbose', 'quiet', 'func', 'command'}
    cli_args = {k: v for k, v in cli_args.items() if k not in non_config_args}
    
    if cli_args:
        override_config = Config.from_cli_args(cli_args)
        
        # Merge configurations (CLI args override file config)
        for field_name in ['experiment', 'data', 'algorithms', 'acceptance']:
            override_field = getattr(override_config, field_name)
            config_field = getattr(config, field_name)
            
            for attr_name in dir(override_field):
                if not attr_name.startswith('_'):
                    override_value = getattr(override_field, attr_name)
                    default_value = getattr(Config(), field_name).__dict__.get(attr_name)
                    
                    if override_value != default_value:
                        setattr(config_field, attr_name, override_value)
                        logger.info(f"🔧 Override: {field_name}.{attr_name} = {override_value}")
    
    return config


def print_experiment_summary(results: Dict[str, Any]):
    """Print a summary of experiment results."""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    experiment_info = results.get('experiment_info', {})
    print(f"Experiment ID: {experiment_info.get('experiment_id', 'N/A')}")
    print(f"Duration: {experiment_info.get('total_duration_seconds', 0):.1f} seconds")
    
    if 'method_results' in results:
        print(f"\nMethod Performance:")
        print(f"{'Method':<15} {'Avg Value':<12} {'Match Rate':<12} {'Comp Time':<12}")
        print("-" * 55)
        
        for method, method_data in results['method_results'].items():
            avg_value = method_data.get('average_objective_value', 0)
            match_rate = method_data.get('average_match_rate', 0) * 100
            comp_time = method_data.get('average_computation_time', 0)
            
            print(f"{method:<15} {avg_value:<12.1f} {match_rate:<12.1f}% {comp_time:<12.3f}s")
    
    print(f"\nResults saved and can be analyzed with:")
    print(f"python -m src.main analyze {experiment_info.get('results_file', 'results.json')}")
    print("="*80)


if __name__ == '__main__':
    main() 