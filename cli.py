#!/usr/bin/env python3
"""
Taxi Pricing Benchmark CLI
Command-line interface for systematic benchmarking of 4 taxi pricing methods
"""

import argparse
import logging
import json
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from config.experiment_config import (
    ExperimentConfig, 
    create_default_config, 
    create_hikima_replication_config, 
    create_extended_benchmark_config
)
from src.orchestrator import BenchmarkOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data(n_requesters: int = 100, n_taxis: int = 80) -> tuple:
    """
    Create sample data for testing and demonstration
    
    Args:
        n_requesters: Number of requesters to generate
        n_taxis: Number of taxis to generate
        
    Returns:
        Tuple of (requesters_data, taxis_data) as numpy arrays
    """
    logger.info(f"Creating sample data: {n_requesters} requesters, {n_taxis} taxis")
    
    # Generate synthetic requester data
    # Format: [borough, area_id, trip_distance_km, total_amount, destination_id, duration]
    requesters_data = []
    np.random.seed(42)  # For reproducible results
    
    for i in range(n_requesters):
        requester = [
            0,  # borough (0 = Manhattan)
            np.random.randint(1, 265),  # pickup area_id (NYC taxi zones)
            np.random.uniform(0.5, 15.0),  # trip_distance in km
            np.random.uniform(8.0, 50.0),  # total_amount in USD
            np.random.randint(1, 265),  # destination area_id
            np.random.uniform(180.0, 1800.0)  # duration in seconds
        ]
        requesters_data.append(requester)
    
    # Generate synthetic taxi data
    # Format: [area_id, available]
    taxis_data = []
    for i in range(n_taxis):
        taxi = [
            np.random.randint(1, 265),  # current area_id
            1  # available (1 = available)
        ]
        taxis_data.append(taxi)
    
    return np.array(requesters_data), np.array(taxis_data)


def run_hikima_replication_experiment(args):
    """Run experiment that replicates Hikima paper setup"""
    logger.info("🔬 Running Hikima replication experiment")
    logger.info("📊 Setup: 2 days, 10:00-20:00, 5-minute windows")
    
    config = create_hikima_replication_config()
    
    # Override config with CLI arguments if provided
    if args.methods:
        config.methods_to_run = args.methods
    if args.acceptance_function:
        config.hikima_config.acceptance_type = args.acceptance_function
        config.maps_config.acceptance_type = args.acceptance_function
        config.linucb_config.acceptance_type = args.acceptance_function
    
    orchestrator = BenchmarkOrchestrator(config.to_dict())
    
    # Use sample data for demonstration (in production, load from S3)
    requesters_data, taxis_data = create_sample_data(
        n_requesters=200,  # Moderate size for Hikima replication
        n_taxis=150
    )
    
    logger.info("🚀 Starting Hikima replication benchmark...")
    result = orchestrator.run_benchmark(requesters_data, taxis_data)
    
    # Save results
    output_file = f"results/hikima_replication_{result.experiment_id}.json"
    os.makedirs("results", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(result.__dict__, f, indent=2, default=str)
    
    logger.info(f"✅ Results saved to {output_file}")
    
    return result


def run_extended_benchmark_experiment(args):
    """Run extended benchmarking experiment over multiple days"""
    days = args.days if hasattr(args, 'days') else 30
    
    logger.info(f"🔬 Running extended benchmark experiment over {days} days")
    logger.info("📊 Setup: 24-hour coverage, configurable time windows")
    
    config = create_extended_benchmark_config(days)
    
    # Override config with CLI arguments
    if args.methods:
        config.methods_to_run = args.methods
    if args.time_window:
        config.time_config.time_window_minutes = args.time_window
    if args.start_hour is not None:
        config.time_config.start_hour = args.start_hour
    if args.end_hour is not None:
        config.time_config.end_hour = args.end_hour
    
    orchestrator = BenchmarkOrchestrator(config.to_dict())
    
    # Use larger sample data for extended benchmark
    requesters_data, taxis_data = create_sample_data(
        n_requesters=500,  # Larger dataset for extended analysis
        n_taxis=400
    )
    
    logger.info("🚀 Starting extended benchmark...")
    result = orchestrator.run_benchmark(requesters_data, taxis_data)
    
    # Save results
    output_file = f"results/extended_benchmark_{days}days_{result.experiment_id}.json"
    os.makedirs("results", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(result.__dict__, f, indent=2, default=str)
    
    logger.info(f"✅ Results saved to {output_file}")
    
    return result


def run_custom_experiment(args):
    """Run custom experiment with user-specified parameters"""
    logger.info("🔬 Running custom experiment")
    
    # Create base configuration
    config = create_default_config()
    
    # Apply custom parameters
    config.experiment_name = args.name if args.name else "custom_experiment"
    config.methods_to_run = args.methods if args.methods else config.methods_to_run
    
    # Time configuration
    if args.start_hour is not None:
        config.time_config.start_hour = args.start_hour
    if args.end_hour is not None:
        config.time_config.end_hour = args.end_hour
    if args.time_window:
        config.time_config.time_window_minutes = args.time_window
    
    # Method-specific configuration
    if args.acceptance_function:
        config.hikima_config.acceptance_type = args.acceptance_function
        config.maps_config.acceptance_type = args.acceptance_function
        config.linucb_config.acceptance_type = args.acceptance_function
    
    # Data size
    n_requesters = args.requesters if args.requesters else 100
    n_taxis = args.taxis if args.taxis else 80
    
    logger.info(f"📊 Custom setup: {n_requesters} requesters, {n_taxis} taxis")
    logger.info(f"🕐 Time range: {config.time_config.start_hour:02d}:00-{config.time_config.end_hour:02d}:00")
    
    orchestrator = BenchmarkOrchestrator(config.to_dict())
    
    # Generate data
    requesters_data, taxis_data = create_sample_data(n_requesters, n_taxis)
    
    logger.info("🚀 Starting custom benchmark...")
    result = orchestrator.run_benchmark(requesters_data, taxis_data)
    
    # Save results
    output_file = f"results/custom_{config.experiment_name}_{result.experiment_id}.json"
    os.makedirs("results", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(result.__dict__, f, indent=2, default=str)
    
    logger.info(f"✅ Results saved to {output_file}")
    
    return result


def validate_config(args):
    """Validate and display configuration"""
    logger.info("🔍 Validating experiment configuration")
    
    if args.config_file:
        try:
            config = ExperimentConfig.from_file(args.config_file)
            logger.info(f"✅ Loaded configuration from {args.config_file}")
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            return False
    else:
        config = create_default_config()
        logger.info("✅ Using default configuration")
    
    # Validate configuration
    errors = config.validate()
    if errors:
        logger.error(f"❌ Configuration validation failed:")
        for error in errors:
            logger.error(f"   - {error}")
        return False
    else:
        logger.info("✅ Configuration is valid")
    
    # Display configuration summary
    logger.info("\n📋 Configuration Summary:")
    logger.info(f"   Experiment: {config.experiment_name}")
    logger.info(f"   Methods: {config.methods_to_run}")
    logger.info(f"   Time Range: {config.time_config.start_hour:02d}:00-{config.time_config.end_hour:02d}:00")
    logger.info(f"   Window Size: {config.time_config.time_window_minutes} minutes")
    
    if args.save_config:
        output_file = args.save_config
        config.save_to_file(output_file)
        logger.info(f"💾 Configuration saved to {output_file}")
    
    return True


def list_methods():
    """List available pricing methods"""
    methods = [
        ("HikimaMinMaxCostFlow", "Hikima's MinMax Cost Flow method"),
        ("MAPS", "Multi-Area Pricing Strategy method"),
        ("LinUCB", "Linear Upper Confidence Bound method"),
        ("LinearProgram", "Linear Programming (Gupta-Nagarajan) method")
    ]
    
    print("\n📚 Available Pricing Methods:")
    print("=" * 50)
    for name, description in methods:
        print(f"{name:20s}: {description}")
    print()


def create_example_configs():
    """Create example configuration files"""
    logger.info("📝 Creating example configuration files")
    
    os.makedirs("configs", exist_ok=True)
    
    # Hikima replication config
    hikima_config = create_hikima_replication_config()
    hikima_config.save_to_file("configs/hikima_replication.json")
    logger.info("✅ Created configs/hikima_replication.json")
    
    # Extended benchmark config
    extended_config = create_extended_benchmark_config(100)
    extended_config.save_to_file("configs/extended_benchmark_100days.json")
    logger.info("✅ Created configs/extended_benchmark_100days.json")
    
    # Default config
    default_config = create_default_config()
    default_config.save_to_file("configs/default.json")
    logger.info("✅ Created configs/default.json")
    
    logger.info("📁 Example configurations created in ./configs/ directory")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Taxi Pricing Benchmark System - Systematic comparison of 4 pricing methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Hikima replication experiment
  python cli.py hikima-replication
  
  # Run extended benchmark over 30 days
  python cli.py extended-benchmark --days 30
  
  # Run custom experiment with specific parameters
  python cli.py custom --methods HikimaMinMaxCostFlow MAPS --start-hour 8 --end-hour 18
  
  # Validate configuration file
  python cli.py validate --config configs/hikima_replication.json
  
  # List available methods
  python cli.py list-methods
  
  # Create example configurations
  python cli.py create-examples
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Hikima replication experiment
    hikima_parser = subparsers.add_parser('hikima-replication', help='Run Hikima paper replication experiment')
    hikima_parser.add_argument('--methods', nargs='+', 
                              choices=['HikimaMinMaxCostFlow', 'MAPS', 'LinUCB', 'LinearProgram'],
                              help='Methods to run (default: all)')
    hikima_parser.add_argument('--acceptance-function', choices=['PL', 'Sigmoid'],
                              help='Acceptance function type')
    
    # Extended benchmark experiment
    extended_parser = subparsers.add_parser('extended-benchmark', help='Run extended benchmark experiment')
    extended_parser.add_argument('--days', type=int, default=30, help='Number of days to simulate')
    extended_parser.add_argument('--methods', nargs='+',
                                choices=['HikimaMinMaxCostFlow', 'MAPS', 'LinUCB', 'LinearProgram'],
                                help='Methods to run (default: all)')
    extended_parser.add_argument('--time-window', type=int, help='Time window in minutes')
    extended_parser.add_argument('--start-hour', type=int, help='Start hour (0-23)')
    extended_parser.add_argument('--end-hour', type=int, help='End hour (1-24)')
    
    # Custom experiment
    custom_parser = subparsers.add_parser('custom', help='Run custom experiment')
    custom_parser.add_argument('--name', help='Experiment name')
    custom_parser.add_argument('--methods', nargs='+',
                              choices=['HikimaMinMaxCostFlow', 'MAPS', 'LinUCB', 'LinearProgram'],
                              help='Methods to run')
    custom_parser.add_argument('--requesters', type=int, help='Number of requesters')
    custom_parser.add_argument('--taxis', type=int, help='Number of taxis')
    custom_parser.add_argument('--start-hour', type=int, help='Start hour (0-23)')
    custom_parser.add_argument('--end-hour', type=int, help='End hour (1-24)')
    custom_parser.add_argument('--time-window', type=int, help='Time window in minutes')
    custom_parser.add_argument('--acceptance-function', choices=['PL', 'Sigmoid'],
                              help='Acceptance function type')
    
    # Configuration validation
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('--config', dest='config_file', help='Configuration file to validate')
    validate_parser.add_argument('--save-config', help='Save validated config to file')
    
    # List methods
    subparsers.add_parser('list-methods', help='List available pricing methods')
    
    # Create examples
    subparsers.add_parser('create-examples', help='Create example configuration files')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'hikima-replication':
            result = run_hikima_replication_experiment(args)
            print(f"\n🎉 Hikima replication completed! Experiment ID: {result.experiment_id}")
            
        elif args.command == 'extended-benchmark':
            result = run_extended_benchmark_experiment(args)
            print(f"\n🎉 Extended benchmark completed! Experiment ID: {result.experiment_id}")
            
        elif args.command == 'custom':
            result = run_custom_experiment(args)
            print(f"\n🎉 Custom experiment completed! Experiment ID: {result.experiment_id}")
            
        elif args.command == 'validate':
            if validate_config(args):
                print("\n✅ Configuration is valid and ready to use!")
            else:
                print("\n❌ Configuration validation failed!")
                sys.exit(1)
                
        elif args.command == 'list-methods':
            list_methods()
            
        elif args.command == 'create-examples':
            create_example_configs()
            print("\n✅ Example configurations created!")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 