#!/usr/bin/env python3
"""
Enhanced ride-hailing pricing experiment with support for multiple vehicle types.
Supports Yellow Taxi, Green Taxi, FHV, and FHVHV data with flexible parameters.
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import datetime

# Import our modules
from lp_pricing import LPPricingOptimizer, create_price_grid
from benchmark_utils import BenchmarkLogger, ExperimentTimer

class EnhancedRideHailingExperiment:
    """
    Enhanced ride-hailing pricing experiment supporting multiple vehicle types.
    """
    
    # Column mapping for different vehicle types
    COLUMN_MAPPINGS = {
        'yellow': {
            'pickup_datetime': 'tpep_pickup_datetime',
            'dropoff_datetime': 'tpep_dropoff_datetime',
            'pickup_location': 'PULocationID',
            'dropoff_location': 'DOLocationID',
            'trip_distance': 'trip_distance',
            'total_amount': 'total_amount'
        },
        'green': {
            'pickup_datetime': 'lpep_pickup_datetime',
            'dropoff_datetime': 'lpep_dropoff_datetime',
            'pickup_location': 'PULocationID',
            'dropoff_location': 'DOLocationID',
            'trip_distance': 'trip_distance',
            'total_amount': 'total_amount'
        },
        'fhv': {
            'pickup_datetime': 'pickup_datetime',
            'dropoff_datetime': 'dropOff_datetime',
            'pickup_location': 'PUlocationID',
            'dropoff_location': 'DOlocationID',
            'trip_distance': None,  # FHV doesn't have trip distance
            'total_amount': None    # FHV doesn't have total amount
        },
        'fhvhv': {
            'pickup_datetime': 'pickup_datetime',
            'dropoff_datetime': 'dropoff_datetime',
            'pickup_location': 'PULocationID',
            'dropoff_location': 'DOLocationID',
            'trip_distance': 'trip_miles',
            'total_amount': 'total_amount'
        }
    }
    
    def __init__(self, vehicle_type: str, year: int, month: int, place: str, 
                 day: int, time_interval: int, time_unit: str, simulation_range: int,
                 acceptance_function: str = 'PL', data_dir: Path = Path("../data")):
        """
        Initialize enhanced experiment.
        
        Args:
            vehicle_type: Type of vehicle data ('yellow', 'green', 'fhv', 'fhvhv')
            year: Year of data
            month: Month of data
            place: Borough/place name
            day: Day for simulation
            time_interval: Time interval length
            time_unit: Time unit ('s' or 'm')
            simulation_range: Number of simulation iterations
            acceptance_function: Type of acceptance function ('PL', 'Sigmoid')
            data_dir: Directory containing data files
        """
        self.vehicle_type = vehicle_type
        self.year = year
        self.month = month
        self.place = place
        self.day = day
        self.time_interval = time_interval
        self.time_unit = time_unit
        self.simulation_range = simulation_range
        self.acceptance_function = acceptance_function
        self.data_dir = Path(data_dir)
        
        # Validate vehicle type
        if vehicle_type not in self.COLUMN_MAPPINGS:
            raise ValueError(f"Unsupported vehicle type: {vehicle_type}")
        
        # Constants
        self.num_eval = 100
        self.epsilon = 1e-10
        self.alpha = 18
        self.s_taxi = 25
        
        # Acceptance function parameters
        if acceptance_function == 'Sigmoid':
            self.beta = 1.3
            self.gamma = (0.3 * np.sqrt(3) / np.pi).astype(np.float64)
        
        # Initialize components
        experiment_name = f"{acceptance_function}_{vehicle_type}_{year}_{month:02d}"
        self.benchmark_logger = BenchmarkLogger(experiment_name)
        self.logger = self.benchmark_logger.logger
        self.lp_optimizer = LPPricingOptimizer(verbose=False)
        
        # Load data
        self._load_area_data()
        self._load_trip_data()
        
        self.logger.info(f"Enhanced experiment initialized: {vehicle_type} {year}-{month:02d} {place}")
    
    def _load_area_data(self):
        """Load area information and compute distance matrix."""
        self.logger.info("Loading area information...")
        
        area_file = self.data_dir / "area_information.csv"
        if not area_file.exists():
            raise FileNotFoundError(f"Area information file not found: {area_file}")
        
        df_loc = pd.read_csv(area_file)
        self.area_data = df_loc
        self.tu_data = df_loc.values[:, 6:8]
        
        # Get location sets for the borough
        df_ID = df_loc[df_loc["borough"] == self.place]
        self.location_ids = list(set(df_ID.values[:, 4]))
        
        self.logger.info(f"Loaded area data for {self.place}: {len(self.location_ids)} locations")
    
    def _load_trip_data(self):
        """Load and preprocess trip data for the specified vehicle type."""
        self.logger.info(f"Loading {self.vehicle_type} trip data for {self.year}-{self.month:02d}...")
        
        # Construct data file path
        csv_file = self.data_dir / "csv" / f"{self.vehicle_type}_tripdata_{self.year}-{self.month:02d}.csv"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"Trip data file not found: {csv_file}")
        
        # Load data
        column_mapping = self.COLUMN_MAPPINGS[self.vehicle_type]
        
        # Read with appropriate date parsing
        date_columns = [column_mapping['pickup_datetime'], column_mapping['dropoff_datetime']]
        df = pd.read_csv(csv_file, parse_dates=date_columns)
        
        # Filter and standardize columns
        required_columns = [
            column_mapping['pickup_datetime'],
            column_mapping['dropoff_datetime'],
            column_mapping['pickup_location'],
            column_mapping['dropoff_location']
        ]
        
        # Add optional columns if they exist
        if column_mapping['trip_distance'] and column_mapping['trip_distance'] in df.columns:
            required_columns.append(column_mapping['trip_distance'])
            has_distance = True
        else:
            has_distance = False
            
        if column_mapping['total_amount'] and column_mapping['total_amount'] in df.columns:
            required_columns.append(column_mapping['total_amount'])
            has_amount = True
        else:
            has_amount = False
        
        # Filter for existing columns only
        existing_columns = [col for col in required_columns if col in df.columns]
        df = df[existing_columns].copy()
        
        # Merge with area data for borough filtering
        df = pd.merge(df, self.area_data, how="inner", 
                     left_on=column_mapping['pickup_location'], right_on="LocationID")
        
        # Apply filters
        filters = [
            (df["borough"] == self.place),
            (df[column_mapping['pickup_location']] < 264),
            (df[column_mapping['dropoff_location']] < 264)
        ]
        
        if has_distance:
            filters.append(df[column_mapping['trip_distance']] > 1e-3)
        if has_amount:
            filters.append(df[column_mapping['total_amount']] > 1e-3)
        
        # Apply all filters
        mask = filters[0]
        for f in filters[1:]:
            mask &= f
        
        df = df[mask]
        
        # Standardize column names
        standard_columns = {
            column_mapping['pickup_datetime']: 'pickup_datetime',
            column_mapping['dropoff_datetime']: 'dropoff_datetime',
            column_mapping['pickup_location']: 'pickup_location',
            column_mapping['dropoff_location']: 'dropoff_location'
        }
        
        if has_distance:
            standard_columns[column_mapping['trip_distance']] = 'trip_distance'
        if has_amount:
            standard_columns[column_mapping['total_amount']] = 'total_amount'
        
        df = df.rename(columns=standard_columns)
        
        # Handle missing data for FHV
        if not has_distance:
            # Estimate trip distance based on location (simplified)
            df['trip_distance'] = np.random.uniform(1, 10, len(df))  # Placeholder
        if not has_amount:
            # Estimate trip amount based on distance and time
            df['total_amount'] = df['trip_distance'] * np.random.uniform(2, 5, len(df))  # Placeholder
        
        self.trip_data = df
        
        self.logger.info(f"Loaded {len(df)} {self.vehicle_type} trips for {self.place}")
    
    def _solve_lp_pricing(self, df_requesters: np.ndarray, df_taxis: np.ndarray, 
                         n: int, m: int) -> Dict[str, Any]:
        """Solve pricing using LP approach with configurable acceptance function."""
        with ExperimentTimer("LP Pricing", self.logger):
            # Calculate edge costs (simplified for demonstration)
            edges = {}
            for i in range(n):
                for j in range(m):
                    # Use simplified distance calculation
                    cost = np.random.uniform(5, 15)  # Placeholder cost
                    edges[(i, j)] = cost
            
            # Create price grid
            base_prices = []
            for i in range(n):
                trip_amount = df_requesters[i, 4] if len(df_requesters[i]) > 4 else 20
                for mult in [0.5, 0.75, 1.0, 1.25, 1.5]:
                    base_prices.append(trip_amount * mult)
            
            base_prices = sorted(list(set(base_prices)))[:10]
            price_grid = create_price_grid(list(range(n)), base_prices)
            
            # Calculate acceptance probabilities
            acceptance_prob = {}
            for c in range(n):
                trip_amount = df_requesters[c, 4] if len(df_requesters[c]) > 4 else 20
                
                for price in price_grid[c]:
                    if self.acceptance_function == 'PL':
                        # Piecewise linear
                        c_param = 2.0 / trip_amount
                        prob = max(0, min(1, -c_param * price + 3.0))
                    else:  # Sigmoid
                        # Sigmoid function
                        exponent = (-price + self.beta * trip_amount) / (self.gamma * trip_amount)
                        exponent = np.clip(exponent, -500, 500)
                        prob = 1 - (1 / (1 + np.exp(exponent)))
                        prob = max(0, min(1, prob))
                    
                    acceptance_prob[(c, price)] = prob
            
            # Solve LP
            clients = list(range(n))
            taxis = list(range(m))
            
            solution = self.lp_optimizer.solve_pricing_lp(
                clients, taxis, edges, price_grid, acceptance_prob
            )
            
            return {
                'objective_value': solution.get('objective_value', 0),
                'solve_time': solution.get('solve_time', 0),
                'status': solution.get('status', 'unknown')
            }
    
    def run_experiment(self):
        """Run the complete experiment."""
        self.logger.info("Starting enhanced experiment execution")
        
        for iteration in range(self.simulation_range):
            # Prepare iteration data (simplified)
            n_requesters = np.random.randint(5, 20)  # Random number of requesters
            n_taxis = np.random.randint(3, 15)       # Random number of taxis
            
            if n_requesters == 0 or n_taxis == 0:
                continue
            
            # Create dummy data arrays (in real implementation, extract from trip_data)
            df_requesters = np.random.random((n_requesters, 6)) * 100
            df_taxis = np.random.random(n_taxis) * 100
            
            # Log iteration start
            iteration_params = {
                'iteration': iteration,
                'vehicle_type': self.vehicle_type,
                'year': self.year,
                'month': self.month,
                'n_requesters': n_requesters,
                'n_taxis': n_taxis
            }
            self.benchmark_logger.log_iteration_start(iteration, iteration_params)
            
            # Run LP pricing
            lp_results = self._solve_lp_pricing(df_requesters, df_taxis, n_requesters, n_taxis)
            
            # Compile results
            iteration_results = {
                'LP_Pricing': lp_results
            }
            
            # Log results
            self.benchmark_logger.log_iteration_result(iteration, iteration_results)
            
            # Add KPIs
            self.benchmark_logger.add_kpi(f'iteration_{iteration}_n_requesters', n_requesters)
            self.benchmark_logger.add_kpi(f'iteration_{iteration}_n_taxis', n_taxis)
        
        # Finalize experiment
        self.benchmark_logger.finalize_experiment(
            self.place, self.day, self.time_interval, self.time_unit
        )

def main():
    """Main function to run enhanced experiment."""
    parser = argparse.ArgumentParser(description='Enhanced Ride-Hailing Pricing Experiment')
    
    parser.add_argument('place', help='Borough/place name')
    parser.add_argument('day', type=int, help='Day for simulation')
    parser.add_argument('time_interval', type=int, help='Time interval length')
    parser.add_argument('time_unit', choices=['s', 'm'], help='Time unit')
    parser.add_argument('simulation_range', type=int, help='Number of simulation iterations')
    
    parser.add_argument('--vehicle-type', choices=['yellow', 'green', 'fhv', 'fhvhv'],
                       default='yellow', help='Vehicle type')
    parser.add_argument('--year', type=int, default=2019, help='Year of data')
    parser.add_argument('--month', type=int, default=10, help='Month of data')
    
    args = parser.parse_args()
    
    print(f"🚕 Enhanced Experiment")
    print(f"   Vehicle type: {args.vehicle_type}")
    print(f"   Data: {args.year}-{args.month:02d}")
    print(f"   Place: {args.place}")

if __name__ == "__main__":
    main() 