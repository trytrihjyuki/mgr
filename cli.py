#!/usr/bin/env python3
"""
Taxi Pricing Benchmark CLI
Unified command-line interface for systematic benchmarking of 4 taxi pricing methods
"""

import argparse
import logging
import json
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from config.experiment_config import (
    ExperimentConfig, 
    create_default_config
)
from src.orchestrator import BenchmarkOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_methods(methods_arg: str) -> List[str]:
    """Parse methods argument with shorthand support"""
    if methods_arg == "-1" or methods_arg.lower() == "all":
        return ["HikimaMinMaxCostFlow", "MAPS", "LinUCB", "LinearProgram"]
    
    method_map = {
        "hikima": "HikimaMinMaxCostFlow",
        "maps": "MAPS", 
        "linucb": "LinUCB",
        "lp": "LinearProgram"
    }
    
    methods = []
    for m in methods_arg.split(","):
        m = m.strip().lower()
        if m in method_map:
            methods.append(method_map[m])
        elif m in ["HikimaMinMaxCostFlow", "MAPS", "LinUCB", "LinearProgram"]:
            methods.append(m)
        else:
            logger.warning(f"Unknown method: {m}")
    
    return methods if methods else ["HikimaMinMaxCostFlow", "MAPS", "LinUCB", "LinearProgram"]


def parse_days(days_arg: str, month: int, year: int) -> List[int]:
    """Parse days argument with shorthand support"""
    if days_arg == "-1" or days_arg.lower() == "all":
        # Return all days in the month
        if month in [1, 3, 5, 7, 8, 10, 12]:
            return list(range(1, 32))
        elif month in [4, 6, 9, 11]:
            return list(range(1, 31))
        else:  # February
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                return list(range(1, 30))  # Leap year
            else:
                return list(range(1, 29))
    
    days = []
    for d in days_arg.split(","):
        d = d.strip()
        if "-" in d:  # Range like "10-15"
            start, end = map(int, d.split("-"))
            days.extend(range(start, end + 1))
        else:
            days.append(int(d))
    
    return days


def parse_months(months_arg: str) -> List[int]:
    """Parse months argument with shorthand support"""
    if months_arg == "-1" or months_arg.lower() == "all":
        return list(range(1, 13))
    
    months = []
    for m in months_arg.split(","):
        m = m.strip()
        if "-" in m:  # Range like "1-6"
            start, end = map(int, m.split("-"))
            months.extend(range(start, end + 1))
        else:
            months.append(int(m))
    
    return months


def parse_acceptance_functions(func_arg: str) -> List[str]:
    """Parse acceptance functions argument"""
    functions = []
    for f in func_arg.split(","):
        f = f.strip().upper()
        if f in ["PL", "SIGMOID"]:
            functions.append(f)
        else:
            logger.warning(f"Unknown acceptance function: {f}")
    
    return functions if functions else ["PL"]


def parse_time_window(window_arg: str) -> int:
    """Parse time window argument (e.g., '5m', '30s', '1h')"""
    if window_arg.endswith('m'):
        return int(window_arg[:-1])
    elif window_arg.endswith('s'):
        return int(window_arg[:-1]) // 60  # Convert seconds to minutes
    elif window_arg.endswith('h'):
        return int(window_arg[:-1]) * 60  # Convert hours to minutes
    else:
        return int(window_arg)  # Assume minutes


def create_sample_data(n_requesters: int = 100, n_taxis: int = 80, location: str = None) -> tuple:
    """
    Create sample data for testing and demonstration
    
    Args:
        n_requesters: Number of requesters to generate
        n_taxis: Number of taxis to generate
        location: Geographic location filter
        
    Returns:
        Tuple of (requesters_data, taxis_data) as numpy arrays
    """
    logger.info(f"Creating sample data: {n_requesters} requesters, {n_taxis} taxis")
    if location:
        logger.info(f"Location filter: {location}")
    
    # Geographic bounds for NYC boroughs
    location_bounds = {
        "manhattan": {"lat": (40.7, 40.8), "lon": (-74.02, -73.93), "zones": range(1, 69)},
        "brooklyn": {"lat": (40.57, 40.74), "lon": (-74.04, -73.83), "zones": range(70, 159)},
        "queens": {"lat": (40.54, 40.8), "lon": (-73.96, -73.7), "zones": range(160, 229)},
        "bronx": {"lat": (40.79, 40.92), "lon": (-73.93, -73.77), "zones": range(230, 254)},
        "staten_island": {"lat": (40.477, 40.651), "lon": (-74.26, -74.05), "zones": range(255, 264)},
    }
    
    # Generate synthetic requester data
    requesters_data = []
    np.random.seed(42)  # For reproducible results
    
    for i in range(n_requesters):
        # Select location zones based on filter
        if location and location.lower() in location_bounds:
            bounds = location_bounds[location.lower()]
            zone_range = bounds["zones"]
        else:
            zone_range = range(1, 265)  # All NYC zones
        
        requester = [
            0,  # borough (0 = default)
            np.random.choice(list(zone_range)),  # pickup area_id
            np.random.uniform(0.5, 15.0),  # trip_distance in km
            np.random.uniform(8.0, 50.0),  # total_amount in USD
            np.random.choice(list(zone_range)),  # destination area_id
            np.random.uniform(180.0, 1800.0)  # duration in seconds
        ]
        requesters_data.append(requester)
    
    # Generate synthetic taxi data
    taxis_data = []
    for i in range(n_taxis):
        if location and location.lower() in location_bounds:
            bounds = location_bounds[location.lower()]
            zone_range = bounds["zones"]
        else:
            zone_range = range(1, 265)
        
        taxi = [
            np.random.choice(list(zone_range)),  # current area_id
            1  # available (1 = available)
        ]
        taxis_data.append(taxi)
    
    return np.array(requesters_data), np.array(taxis_data)


def run_experiment(args) -> None:
    """Run taxi pricing benchmark experiment with parsed arguments"""
    
    # Parse arguments
    methods = parse_methods(args.methods)
    days = parse_days(args.days, args.month, args.year) if args.days != "1" else [1, 6]  # Default Hikima days
    months = parse_months(args.months) if hasattr(args, 'months') and args.months else [args.month]
    acceptance_functions = parse_acceptance_functions(args.func)
    time_window = parse_time_window(args.window)
    
    # Log experiment setup
    logger.info("🚀 Starting Taxi Pricing Benchmark")
    logger.info(f"📊 Methods: {methods}")
    logger.info(f"📅 Year: {args.year}, Months: {months}, Days: {days}")
    logger.info(f"🕐 Time: {args.start_hour:02d}:00-{args.end_hour:02d}:00, Window: {time_window}min")
    logger.info(f"🔧 Functions: {acceptance_functions}")
    if args.location:
        logger.info(f"📍 Location: {args.location}")
    
    # Calculate total experiments
    total_experiments = len(methods) * len(days) * len(months) * len(acceptance_functions)
    logger.info(f"🧮 Total experiments to run: {total_experiments}")
    
    all_results = []
    experiment_count = 0
    
    # Run experiments for each combination
    for month in months:
        for day in days:
            for acceptance_func in acceptance_functions:
                experiment_count += 1
                logger.info(f"\n🔬 Experiment {experiment_count}/{len(days) * len(months) * len(acceptance_functions)}")
                logger.info(f"📅 Date: {args.year}-{month:02d}-{day:02d}, Function: {acceptance_func}")
                
                # Create configuration
                config = create_default_config()
                config.experiment_name = f"benchmark_{args.year}{month:02d}{day:02d}_{acceptance_func}"
                config.methods_to_run = methods
                
                # Time configuration
                config.time_config.start_hour = args.start_hour
                config.time_config.end_hour = args.end_hour
                config.time_config.time_window_minutes = time_window
                config.time_config.start_date = f"{args.year}-{month:02d}-{day:02d}"
                config.time_config.end_date = f"{args.year}-{month:02d}-{day:02d}"
                
                # Set acceptance function for all methods
                config.hikima_config.acceptance_type = acceptance_func
                config.maps_config.acceptance_type = acceptance_func
                config.linucb_config.acceptance_type = acceptance_func
                
                # Create orchestrator
                orchestrator = BenchmarkOrchestrator(config.to_dict())
                
                # Generate data
                requesters_data, taxis_data = create_sample_data(
                    n_requesters=args.requesters,
                    n_taxis=args.taxis,
                    location=args.location
                )
                
                # Run benchmark
                result = orchestrator.run_benchmark(requesters_data, taxis_data)
                result.experiment_config['date'] = f"{args.year}-{month:02d}-{day:02d}"
                result.experiment_config['acceptance_function'] = acceptance_func
                all_results.append(result)
                
                # Save individual result
                output_file = f"results/{result.experiment_id}.json"
                os.makedirs("results", exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(result.__dict__, f, indent=2, default=str)
    
    # Create summary report
    if len(all_results) > 1:
        create_summary_report(all_results, args)
    
    logger.info(f"\n🎉 All experiments completed! Total: {len(all_results)} results")


def create_summary_report(results: List[Any], args) -> None:
    """Create a summary report of all experiments"""
    
    summary_data = {
        'experiment_summary': {
            'total_experiments': len(results),
            'methods': parse_methods(args.methods),
            'acceptance_functions': parse_acceptance_functions(args.func),
            'time_range': f"{args.start_hour:02d}:00-{args.end_hour:02d}:00",
            'location': args.location,
            'year': args.year
        },
        'performance_summary': {},
        'detailed_results': []
    }
    
    # Aggregate performance by method
    method_performance = {}
    for result in results:
        for method, objective in result.objective_values.items():
            if method not in method_performance:
                method_performance[method] = []
            method_performance[method].append(objective)
    
    # Calculate statistics
    for method, values in method_performance.items():
        summary_data['performance_summary'][method] = {
            'mean_objective': float(np.mean(values)),
            'std_objective': float(np.std(values)),
            'min_objective': float(np.min(values)),
            'max_objective': float(np.max(values)),
            'experiments_count': len(values)
        }
    
    # Add detailed results
    for result in results:
        summary_data['detailed_results'].append({
            'experiment_id': result.experiment_id,
            'date': result.experiment_config.get('date'),
            'acceptance_function': result.experiment_config.get('acceptance_function'),
            'objective_values': result.objective_values,
            'computation_times': result.computation_times
        })
    
    # Save summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f"results/SUMMARY_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    logger.info(f"📊 Summary report saved to {summary_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("📈 EXPERIMENT SUMMARY")
    print("="*60)
    for method, stats in summary_data['performance_summary'].items():
        print(f"{method:20s}: Mean={stats['mean_objective']:8.2f}, Std={stats['std_objective']:6.2f}")
    print("="*60)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Taxi Pricing Benchmark System - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all methods for full year 2019 with both acceptance functions
  python cli.py --methods=-1 --days=-1 --months=-1 --year=2019 --func=PL,Sigmoid --start-hour=0 --end-hour=24 --window=5m
  
  # Hikima replication setup (2 specific days in October)
  python cli.py --methods=-1 --days=1,6 --month=10 --year=2019 --func=PL,Sigmoid --start-hour=10 --end-hour=20 --window=5m --location=Manhattan
  
  # Quick test with 2 methods
  python cli.py --methods=hikima,maps --days=1 --month=10 --year=2019 --func=PL --requesters=50 --taxis=40
  
  # Extended analysis for Q1 2019
  python cli.py --methods=-1 --days=-1 --months=1,2,3 --year=2019 --func=Sigmoid --window=30m
  
  # Brooklyn-specific analysis
  python cli.py --methods=maps,lp --days=1-7 --month=10 --year=2019 --func=PL --location=Brooklyn

Method shortcuts: hikima, maps, linucb, lp (or use -1 or 'all' to select all methods)
Acceptance functions: PL, Sigmoid (comma-separated)
Time windows: 5m, 30s, 1h (minutes/seconds/hours)
Locations: Manhattan, Brooklyn, Queens, Bronx, Staten_Island
"""
    )
    
    # Core experiment parameters
    parser.add_argument('--methods', default='hikima,maps', 
                       help='Methods to run: hikima,maps,linucb,lp or -1 for all')
    parser.add_argument('--days', default='1,6',
                       help='Days to run: 1,6 or 1-7 or -1 for all')
    parser.add_argument('--month', type=int, default=10,
                       help='Month to run (1-12)')
    parser.add_argument('--months', default=None,
                       help='Multiple months: 1,2,3 or 1-6 or -1 for all (overrides --month)')
    parser.add_argument('--year', type=int, default=2019,
                       help='Year to run')
    parser.add_argument('--func', default='PL',
                       help='Acceptance functions: PL,Sigmoid')
    
    # Time parameters
    parser.add_argument('--start-hour', type=int, default=10,
                       help='Start hour (0-23)')
    parser.add_argument('--end-hour', type=int, default=20,
                       help='End hour (1-24)')
    parser.add_argument('--window', default='5m',
                       help='Time window: 5m, 30s, 1h')
    
    # Data parameters
    parser.add_argument('--requesters', type=int, default=200,
                       help='Number of requesters to simulate')
    parser.add_argument('--taxis', type=int, default=150,
                       help='Number of taxis to simulate')
    parser.add_argument('--location', default=None,
                       help='Geographic filter: Manhattan, Brooklyn, Queens, Bronx, Staten_Island')
    
    # Utility commands
    parser.add_argument('--list-methods', action='store_true',
                       help='List available pricing methods')
    parser.add_argument('--validate', metavar='CONFIG_FILE',
                       help='Validate configuration file')
    parser.add_argument('--create-examples', action='store_true',
                       help='Create example configuration files')
    
    args = parser.parse_args()
    
    try:
        if args.list_methods:
            print("\n📚 Available Pricing Methods:")
            print("=" * 50)
            print("hikima               : Hikima's MinMax Cost Flow method")
            print("maps                 : Multi-Area Pricing Strategy method") 
            print("linucb               : Linear Upper Confidence Bound method")
            print("lp                   : Linear Programming (Gupta-Nagarajan) method")
            print("\nShorthand: Use -1 or 'all' to select all methods")
            
        elif args.validate:
            try:
                from config.experiment_config import ExperimentConfig
                config = ExperimentConfig.from_file(args.validate)
                errors = config.validate()
                if errors:
                    print(f"❌ Configuration validation failed:")
                    for error in errors:
                        print(f"   - {error}")
                    sys.exit(1)
                else:
                    print("✅ Configuration is valid!")
            except Exception as e:
                print(f"❌ Failed to validate configuration: {e}")
                sys.exit(1)
                
        elif args.create_examples:
            from config.experiment_config import create_hikima_replication_config, create_extended_benchmark_config, create_default_config
            
            os.makedirs("configs", exist_ok=True)
            
            # Create example configurations
            configs = [
                (create_hikima_replication_config(), "configs/hikima_replication.json"),
                (create_extended_benchmark_config(100), "configs/extended_benchmark.json"),
                (create_default_config(), "configs/default.json")
            ]
            
            for config, filename in configs:
                config.save_to_file(filename)
                print(f"✅ Created {filename}")
            
            print("📁 Example configurations created in ./configs/ directory")
            
        else:
            # Run main experiment
            run_experiment(args)
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 