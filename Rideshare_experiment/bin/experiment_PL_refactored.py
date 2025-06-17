"""
Refactored ride-hailing pricing experiment with Piecewise Linear acceptance function.
Uses Linear Programming approach based on Gupta-Nagarajan reduction.
"""

import pandas as pd
import numpy as np
import random
import math
import networkx as nx
import time
import pyproj
import datetime
import pickle
import sys
from typing import Dict, List, Tuple, Any
import logging

# Import our new modules
from lp_pricing import LPPricingOptimizer, create_price_grid
from benchmark_utils import BenchmarkLogger, ExperimentTimer

# Initialize GRS80 geodetic system
grs80 = pyproj.Geod(ellps='GRS80')

class RideHailingExperimentPL:
    """
    Ride-hailing pricing experiment with Piecewise Linear acceptance function.
    """
    
    def __init__(self, place: str, day: int, time_interval: int, time_unit: str, simulation_range: int):
        """Initialize experiment parameters."""
        self.place = place
        self.day = day
        self.time_interval = time_interval
        self.time_unit = time_unit
        self.simulation_range = simulation_range
        
        # Constants
        self.year = 2019
        self.month = 10
        self.num_eval = 100  # Number of simulations for objective evaluation
        self.epsilon = 1e-10
        self.alpha = 18  # Edge weight parameter
        self.s_taxi = 25  # Taxi speed parameter
        
        # LinUCB parameters
        self.UCB_alpha = 0.5
        self.base_price = 5.875
        self.a = np.array([0.6, 0.8, 1, 1.2, 1.4])
        self.arm_price = self.base_price * self.a
        
        # Initialize benchmark logger
        self.benchmark_logger = BenchmarkLogger("PL_LP")
        self.logger = self.benchmark_logger.logger
        
        # Initialize LP optimizer
        self.lp_optimizer = LPPricingOptimizer(verbose=False)
        
        # Load data
        self._load_area_data()
        self._load_trip_data()
        
        self.logger.info(f"Experiment initialized: {place}, day {day}, interval {time_interval}{time_unit}")
        
    def _load_area_data(self):
        """Load area information and compute distance matrix."""
        self.logger.info("Loading area information...")
        
        df_loc = pd.read_csv("../data/area_information.csv")
        self.tu_data = df_loc.values[:, 6:8]
        
        # Compute distance matrix
        num_areas = self.tu_data.shape[0]
        self.dist_matrix = np.zeros([num_areas, num_areas])
        
        for i in range(num_areas):
            for j in range(num_areas):
                azimuth, bkw_azimuth, distance = grs80.inv(
                    self.tu_data[i, 0], self.tu_data[i, 1], 
                    self.tu_data[j, 0], self.tu_data[j, 1]
                )
                self.dist_matrix[i, j] = distance
        
        self.dist_matrix = self.dist_matrix * 0.001  # Convert to km
        
        df_ID = df_loc[df_loc["borough"] == self.place]
        self.PUID_set = list(set(df_ID.values[:, 4]))
        self.DOID_set = list(set(df_ID.values[:, 4]))
        
        self.logger.info(f"Loaded {num_areas} areas for {self.place}")
        
    def _load_trip_data(self):
        """Load and preprocess trip data."""
        self.logger.info("Loading trip data...")
        
        # Time window for data extraction
        hour_start, hour_end = 10, 20
        day_start_time = datetime.datetime(self.year, self.month, self.day, hour_start-1, 55, 0)
        day_end_time = datetime.datetime(self.year, self.month, self.day, hour_end, 5, 0)
        
        # Load green taxi data
        df_green = pd.read_csv("../data/green_tripdata_2019-10.csv", 
                               parse_dates=['lpep_pickup_datetime', 'lpep_dropoff_datetime'])
        df_green = df_green[['lpep_pickup_datetime', 'lpep_dropoff_datetime', 
                            'PULocationID', 'DOLocationID', 'trip_distance', 'total_amount']]
        
        # Load area data for merging
        df_loc = pd.read_csv("../data/area_information.csv")
        df_green = pd.merge(df_green, df_loc, how="inner", left_on="PULocationID", right_on="LocationID")
        
        # Filter data
        df_green = df_green[
            (df_green["trip_distance"] > 1e-3) & 
            (df_green["total_amount"] > 1e-3) & 
            (df_green["borough"] == self.place) & 
            (df_green["PULocationID"] < 264) & 
            (df_green["DOLocationID"] < 264) & 
            (df_green["lpep_pickup_datetime"] > day_start_time) & 
            (df_green["lpep_pickup_datetime"] < day_end_time)
        ]
        
        self.green_tripdata = df_green[['borough', 'PULocationID', 'DOLocationID', 
                                       'trip_distance', 'total_amount', 
                                       'lpep_pickup_datetime', 'lpep_dropoff_datetime']]
        
        # Similar processing for yellow taxi data
        df_yellow = pd.read_csv("../data/yellow_tripdata_2019-10.csv",
                               parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
        df_yellow = pd.merge(df_yellow, df_loc, how="inner", left_on="PULocationID", right_on="LocationID")
        
        df_yellow = df_yellow[
            (df_yellow["trip_distance"] > 1e-3) & 
            (df_yellow["total_amount"] > 1e-3) & 
            (df_yellow["borough"] == self.place) & 
            (df_yellow["PULocationID"] < 264) & 
            (df_yellow["DOLocationID"] < 264) & 
            (df_yellow["tpep_pickup_datetime"] > day_start_time) & 
            (df_yellow["tpep_pickup_datetime"] < day_end_time)
        ]
        
        self.yellow_tripdata = df_yellow[['borough', 'PULocationID', 'DOLocationID', 
                                         'trip_distance', 'total_amount', 
                                         'tpep_pickup_datetime', 'tpep_dropoff_datetime']]
        
        self.logger.info(f"Loaded {len(self.green_tripdata)} green trips and {len(self.yellow_tripdata)} yellow trips")
    
    def _prepare_iteration_data(self, iteration: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Prepare data for a single iteration."""
        # Calculate time window
        tt = iteration * 5
        h, m = divmod(tt, 60)
        hour = 10 + h
        minute = m
        second = 0
        
        set_time = datetime.datetime(self.year, self.month, self.day, hour, minute, second)
        
        if self.time_unit == 'm':
            if minute + self.time_interval < 60:
                after_time = datetime.datetime(self.year, self.month, self.day, hour, 
                                             minute + self.time_interval, second)
            else:
                after_time = datetime.datetime(self.year, self.month, self.day, hour + 1, 
                                             minute + self.time_interval - 60, second)
        elif self.time_unit == 's':
            after_time = datetime.datetime(self.year, self.month, self.day, hour, minute, 
                                         second + self.time_interval)
        
        # Extract requesters and taxis for this time window
        green_requesters = self.green_tripdata[
            (self.green_tripdata["lpep_pickup_datetime"] > set_time) & 
            (self.green_tripdata["lpep_pickup_datetime"] < after_time)
        ]
        yellow_requesters = self.yellow_tripdata[
            (self.yellow_tripdata["tpep_pickup_datetime"] > set_time) & 
            (self.yellow_tripdata["tpep_pickup_datetime"] < after_time)
        ]
        
        green_taxis = self.green_tripdata[
            (self.green_tripdata["lpep_dropoff_datetime"] > set_time) & 
            (self.green_tripdata["lpep_dropoff_datetime"] < after_time)
        ]
        yellow_taxis = self.yellow_tripdata[
            (self.yellow_tripdata["tpep_dropoff_datetime"] > set_time) & 
            (self.yellow_tripdata["tpep_dropoff_datetime"] < after_time)
        ]
        
        # Combine data
        df_requesters = np.concatenate([green_requesters.values, yellow_requesters.values])
        df_taxis = np.concatenate([green_taxis.values, yellow_taxis.values])[:, 2]
        
        # Calculate trip duration
        time_consume = np.zeros([df_requesters.shape[0], 1])
        for i in range(df_requesters.shape[0]):
            time_consume[i] = (df_requesters[i, 6] - df_requesters[i, 5]).seconds
            
        df_requesters = np.hstack([df_requesters[:, 0:5], time_consume])
        
        # Filter and sort data
        df_requesters = df_requesters[df_requesters[:, 3] > 1e-3]
        df_requesters = df_requesters[df_requesters[:, 4] > 1e-3]
        df_requesters = df_requesters[np.argsort(df_requesters[:, 3])]  # Sort by distance
        df_requesters[:, 3] = df_requesters[:, 3] * 1.60934  # Convert to km
        
        n, m = df_requesters.shape[0], df_taxis.size
        
        return df_requesters, df_taxis, n, m
    
    def _solve_lp_pricing(self, df_requesters: np.ndarray, df_taxis: np.ndarray, 
                         n: int, m: int) -> Dict[str, Any]:
        """Solve pricing using LP approach."""
        with ExperimentTimer("LP Pricing", self.logger):
            # Generate random locations with noise
            requester_locations_x = (self.tu_data[df_requesters[:, 1].astype('int64') - 1, 0].reshape((n, 1)) + 
                                   np.random.normal(0, 0.00306, (n, 1)))
            requester_locations_y = (self.tu_data[df_requesters[:, 1].astype('int64') - 1, 1].reshape((n, 1)) + 
                                   np.random.normal(0, 0.000896, (n, 1)))
            taxi_locations_x = (self.tu_data[df_taxis.astype('int64') - 1, 0].reshape((m, 1)) + 
                               np.random.normal(0, 0.00306, (m, 1)))
            taxi_locations_y = (self.tu_data[df_taxis.astype('int64') - 1, 1].reshape((m, 1)) + 
                               np.random.normal(0, 0.000896, (m, 1)))
            
            # Calculate distances and costs
            edges = {}
            for i in range(n):
                for j in range(m):
                    azimuth, bkw_azimuth, distance = grs80.inv(
                        requester_locations_x[i, 0], requester_locations_y[i, 0],
                        taxi_locations_x[j, 0], taxi_locations_y[j, 0]
                    )
                    distance_km = distance * 0.001
                    cost = (distance_km + df_requesters[i, 3]) / self.s_taxi * self.alpha
                    edges[(i, j)] = cost
            
            # Create price grid - using reasonable price levels
            base_prices = [df_requesters[i, 4] / df_requesters[i, 3] * mult 
                          for i in range(n) for mult in [0.5, 0.75, 1.0, 1.25, 1.5]]
            base_prices = sorted(list(set(base_prices)))[:10]  # Limit to 10 price levels
            
            price_grid = create_price_grid(list(range(n)), base_prices)
            
            # Calculate acceptance probabilities using piecewise linear function
            acceptance_prob = {}
            for c in range(n):
                c_param = 2.0 / df_requesters[c, 4]  # Piecewise linear parameter
                d_param = 3.0  # Constant term
                
                for price in price_grid[c]:
                    # Piecewise linear acceptance: -c*price + d
                    prob = max(0, min(1, -c_param * price + d_param))
                    acceptance_prob[(c, price)] = prob
            
            # Solve LP
            clients = list(range(n))
            taxis = list(range(m))
            
            solution = self.lp_optimizer.solve_pricing_lp(
                clients, taxis, edges, price_grid, acceptance_prob
            )
            
            # Extract deterministic prices
            deterministic_prices = self.lp_optimizer.extract_deterministic_prices(solution)
            price_array = np.zeros(n)
            for i in range(n):
                price_array[i] = deterministic_prices.get(i, base_prices[0])
            
            return {
                'prices': price_array,
                'objective_value': solution.get('objective_value', 0),
                'solve_time': solution.get('solve_time', 0),
                'status': solution.get('status', 'unknown'),
                'acceptance_prob': acceptance_prob,
                'price_grid': price_grid
            }
    
    def run_experiment(self):
        """Run the complete experiment."""
        self.logger.info("Starting experiment execution")
        
        results_per_iteration = []
        
        for iteration in range(self.simulation_range):
            # Log iteration start
            iteration_params = {
                'iteration': iteration,
                'time_offset_minutes': iteration * 5,
                'place': self.place,
                'day': self.day
            }
            self.benchmark_logger.log_iteration_start(iteration, iteration_params)
            
            # Prepare data
            df_requesters, df_taxis, n, m = self._prepare_iteration_data(iteration)
            
            if n == 0 or m == 0:
                self.logger.warning(f"Iteration {iteration}: No requesters ({n}) or taxis ({m})")
                continue
                
            self.logger.info(f"Iteration {iteration}: {n} requesters, {m} taxis")
            
            # Run LP pricing method
            lp_results = self._solve_lp_pricing(df_requesters, df_taxis, n, m)
            
            # Compile iteration results
            iteration_results = {
                'LP_Pricing': {
                    'objective_value': lp_results['objective_value'],
                    'solve_time': lp_results['solve_time'],
                    'status': lp_results['status']
                }
            }
            
            # Log results
            self.benchmark_logger.log_iteration_result(iteration, iteration_results)
            results_per_iteration.append(iteration_results)
            
            # Add KPIs
            self.benchmark_logger.add_kpi(f'iteration_{iteration}_n_requesters', n)
            self.benchmark_logger.add_kpi(f'iteration_{iteration}_m_taxis', m)
            
        # Finalize experiment
        self.benchmark_logger.finalize_experiment(
            self.place, self.day, self.time_interval, self.time_unit
        )
        
        return results_per_iteration

def main():
    """Main function to run experiment."""
    if len(sys.argv) != 6:
        print("Usage: python experiment_PL_refactored.py <place> <day> <time_interval> <time_unit> <simulation_range>")
        print("Example: python experiment_PL_refactored.py Manhattan 6 30 s 5")
        sys.exit(1)
    
    place = sys.argv[1]
    day = int(sys.argv[2])
    time_interval = int(sys.argv[3])
    time_unit = sys.argv[4]
    simulation_range = int(sys.argv[5])
    
    # Create and run experiment
    experiment = RideHailingExperimentPL(place, day, time_interval, time_unit, simulation_range)
    results = experiment.run_experiment()
    
    print(f"Experiment completed. Results saved to results directory.")
    print(f"Total iterations: {len(results)}")

if __name__ == "__main__":
    main() 