#!/usr/bin/env python3
"""
Main CLI entry point for the taxi pricing benchmark framework.

Usage:
    python run_experiment.py --processing-date 2019-10-06 \
                            --vehicle-type green \
                            --boroughs Manhattan Brooklyn \
                            --methods LP MinMaxCostFlow MAPS LinUCB \
                            --num-iter 100 \
                            --start-hour 6 \
                            --end-hour 22 \
                            --time-delta 30
"""

import argparse
import sys
import os
from datetime import datetime, date
from pathlib import Path
import json
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core import (
    ExperimentConfig, setup_logger, get_logger,
    VehicleType, Borough, PricingMethod, AcceptanceFunction
)
from src.data import DataValidator
from src.experiments import ExperimentRunner


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run taxi pricing experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiment for a single day with all methods
  python run_experiment.py --processing-date 2019-10-06 --vehicle-type green \\
      --boroughs Manhattan --methods LP MinMaxCostFlow MAPS LinUCB

  # Run with specific time window
  python run_experiment.py --processing-date 2019-10-06 --vehicle-type yellow \\
      --boroughs Manhattan Brooklyn --methods LP --start-hour 8 --end-hour 12 \\
      --time-delta 15

  # Run with sigmoid acceptance function
  python run_experiment.py --processing-date 2019-10-06 --vehicle-type green \\
      --boroughs Queens --methods MAPS --num-iter 50
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--processing-date',
        type=str,
        required=True,
        help='Date to process (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--vehicle-type',
        type=str,
        required=True,
        choices=['green', 'yellow', 'fhv', 'fhvhv'],
        help='Type of vehicle'
    )
    
    parser.add_argument(
        '--boroughs',
        type=str,
        nargs='+',
        required=True,
        help='Boroughs to process (e.g., Manhattan Brooklyn Queens)'
    )
    
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        required=True,
        choices=['LP', 'MinMaxCostFlow', 'MAPS', 'LinUCB'],
        help='Pricing methods to evaluate'
    )
    
    # Optional arguments
    parser.add_argument(
        '--num-iter',
        type=int,
        default=100,
        help='Number of Monte Carlo iterations per time window (default: 100)'
    )
    
    parser.add_argument(
        '--start-hour',
        type=int,
        default=0,
        help='Start hour of experiment window (default: 0)'
    )
    
    parser.add_argument(
        '--end-hour',
        type=int,
        default=23,
        help='End hour of experiment window (default: 23)'
    )
    
    parser.add_argument(
        '--time-delta',
        type=int,
        default=5,
        help='Time window size in minutes (default: 5)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--s3-base',
        type=str,
        default=None,
        help='S3 base bucket (default: from env S3_BASE or "magisterka")'
    )
    
    parser.add_argument(
        '--s3-results',
        type=str,
        default='taxi-benchmark',
        help='S3 results bucket (default: taxi-benchmark)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without running experiment'
    )
    
    parser.add_argument(
        '--local-mode',
        action='store_true',
        help='Run in local mode (no S3, save results locally)'
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> ExperimentConfig:
    """Create experiment configuration from command line arguments."""
    # Parse date
    try:
        processing_date = datetime.strptime(args.processing_date, '%Y-%m-%d').date()
    except ValueError:
        raise ValueError(f"Invalid date format: {args.processing_date}. Use YYYY-MM-DD")
    
    # Parse enums
    vehicle_type = VehicleType(args.vehicle_type)
    
    boroughs = []
    for b in args.boroughs:
        try:
            boroughs.append(Borough(b))
        except ValueError:
            # Try alternative parsing
            try:
                boroughs.append(Borough.from_string(b))
            except ValueError:
                raise ValueError(f"Invalid borough: {b}")
    
    methods = [PricingMethod(m) for m in args.methods]
    
    # Get S3 base
    s3_base = args.s3_base
    if not s3_base:
        s3_base = os.getenv('S3_BASE', 'magisterka')
    
    # Create config
    config = ExperimentConfig(
        processing_date=processing_date,
        vehicle_type=vehicle_type,
        boroughs=boroughs,
        methods=methods,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        time_delta=args.time_delta,
        time_unit='m',  # Always use minutes for consistency
        num_iter=args.num_iter,
        num_workers=args.num_workers,
        s3_base=s3_base,
        s3_results_bucket=args.s3_results
    )
    
    return config


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    log_file = None
    if not args.dry_run:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"experiment_{timestamp}.log"
    
    logger = setup_logger(
        level=args.log_level,
        log_file=log_file,
        colorize=True
    )
    
    logger.info("=" * 80)
    logger.info("TAXI PRICING BENCHMARK FRAMEWORK")
    logger.info("=" * 80)
    
    try:
        # Create configuration
        logger.info("Creating experiment configuration...")
        config = create_config_from_args(args)
        
        # Log configuration
        logger.info("Experiment Configuration:")
        logger.info(f"  Processing Date: {config.processing_date}")
        logger.info(f"  Vehicle Type: {config.vehicle_type.value}")
        logger.info(f"  Boroughs: {[b.value for b in config.boroughs]}")
        logger.info(f"  Methods: {[m.value for m in config.methods]}")
        logger.info(f"  Evaluation: Both PL and Sigmoid acceptance functions")
        logger.info(f"  Time Window: {config.start_hour:02d}:00 - {config.end_hour:02d}:00")
        logger.info(f"  Time Delta: {config.time_delta} minutes")
        logger.info(f"  Monte Carlo Iterations: {config.num_iter}")
        logger.info(f"  Parallel Workers: {config.num_workers}")
        
        # Calculate total scenarios
        num_windows = len(config.get_time_windows())
        total_scenarios = config.get_total_scenarios()
        logger.info(f"  Number of Time Windows: {num_windows}")
        logger.info(f"  Total Scenarios: {total_scenarios} " 
                   f"({num_windows} windows × {config.num_iter} iterations)")
        
        # Validate data availability
        if not args.local_mode:
            logger.info("\nValidating data availability...")
            validator = DataValidator(s3_bucket=config.s3_base)
            
            # Check data
            is_valid, issues = validator.validate_experiment_data(config)
            if not is_valid:
                logger.error("Data validation failed!")
                for issue in issues:
                    logger.error(f"  - {issue}")
                sys.exit(1)
            
            # Check configuration
            is_valid, issues = validator.validate_configuration(config)
            if not is_valid:
                logger.error("Configuration validation failed!")
                for issue in issues:
                    logger.error(f"  - {issue}")
                sys.exit(1)
            
            # Estimate data size
            estimates = validator.estimate_data_size(config)
            logger.info(f"  Input data size: {estimates.get('input_data_size_mb', 0):.1f} MB")
            logger.info(f"  Estimated output size: {estimates.get('estimated_output_size_mb', 0):.1f} MB")
        
        # Dry run check
        if args.dry_run:
            logger.info("\n✅ Dry run completed successfully. Configuration is valid.")
            
            # Save configuration
            config_file = Path('configs') / f"config_{config.get_experiment_id()}.json"
            config_file.parent.mkdir(exist_ok=True)
            config.save(config_file)
            logger.info(f"Configuration saved to: {config_file}")
            
            sys.exit(0)
        
        # Run experiment
        logger.info("\n" + "=" * 80)
        logger.info("STARTING EXPERIMENT")
        logger.info("=" * 80)
        
        runner = ExperimentRunner(config, local_mode=args.local_mode)
        results_path = runner.run()
        
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {results_path}")
        
    except KeyboardInterrupt:
        logger.warning("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nExperiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()