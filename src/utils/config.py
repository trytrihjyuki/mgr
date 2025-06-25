"""
Configuration management for rideshare pricing benchmark experiments.

This module handles all configuration parameters, removing hardcoded values
and making experiments fully reproducible and configurable.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class DataConfig:
    """Data source and processing configuration."""
    s3_bucket: str = "rideshare-benchmark-data"
    area_info_key: str = "reference/area_info.csv"
    base_url: str = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    cache_dir: str = "./data/cache"
    
    # Data filtering parameters
    min_trip_distance: float = 1e-3  # Minimum valid trip distance
    min_total_amount: float = 1e-3   # Minimum valid fare amount
    max_location_id: int = 264       # Valid LocationID range


@dataclass
class AlgorithmParams:
    """Algorithm-specific parameters for all methods."""
    
    # Hikima MinMaxCost Flow parameters
    hikima_alpha: float = 18.0           # Opportunity cost parameter
    hikima_taxi_speed: float = 25.0      # km/h
    hikima_base_price: float = 5.875     # USD
    hikima_epsilon: float = 1e-10        # Numerical precision
    
    # MAPS algorithm parameters  
    maps_price_discretization: float = 0.05  # Price grid step size
    maps_max_distance: float = 2.0           # Max taxi-rider matching distance (km)
    maps_s0_rate: float = 1.5                # Base acceptance rate multiplier
    
    # LinUCB bandit parameters
    linucb_alpha: float = 0.5                # Exploration parameter
    linucb_arm_multipliers: List[float] = None  # Price arm multipliers
    
    # Linear Program parameters
    lp_solver: str = "CBC"                   # Default LP solver
    lp_time_limit: int = 300                 # Max solve time in seconds
    
    def __post_init__(self):
        if self.linucb_arm_multipliers is None:
            self.linucb_arm_multipliers = [0.6, 0.8, 1.0, 1.2, 1.4]


@dataclass
class AcceptanceFunctionParams:
    """Parameters for customer acceptance probability functions."""
    
    # Piecewise Linear (PL) parameters
    pl_alpha: float = 1.5                    # Slope change point multiplier
    
    # Sigmoid parameters
    sigmoid_beta: float = 1.3                # Price sensitivity
    sigmoid_gamma: float = 0.17              # Approximately 0.3*sqrt(3)/π


@dataclass
class ExperimentConfig:
    """Core experiment configuration."""
    
    # Data selection
    vehicle_type: str = "green"              # green, yellow, or fhv
    year: int = 2019
    month: int = 10
    borough: str = "Manhattan"               # Manhattan, Brooklyn, Queens, Bronx, Staten Island
    
    # Time configuration
    start_hour: int = 10                     # 24-hour format
    end_hour: int = 20                       # 24-hour format  
    time_interval_minutes: int = 5           # Scenario time window
    
    # Experiment scope
    simulation_range: int = 120              # Number of scenarios (Hikima default: 120 for 10h with 5min windows)
    num_evaluations: int = 100               # Monte Carlo evaluations per scenario
    
    # Methods to benchmark
    methods: List[str] = None                # ["hikima", "maps", "linucb", "linear_program"]
    acceptance_function: str = "PL"          # "PL" or "Sigmoid"
    
    # Multi-day experiments
    start_day: int = 1
    end_day: int = 1                         # Set > start_day for multi-day experiments
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = ["hikima", "maps", "linucb", "linear_program"]


@dataclass
class Config:
    """Main configuration container for all experiment parameters."""
    
    data: DataConfig = None
    algorithms: AlgorithmParams = None
    acceptance: AcceptanceFunctionParams = None
    experiment: ExperimentConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.algorithms is None:
            self.algorithms = AlgorithmParams()
        if self.acceptance is None:
            self.acceptance = AcceptanceFunctionParams()  
        if self.experiment is None:
            self.experiment = ExperimentConfig()
    
    @classmethod
    def from_json(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return cls(
            data=DataConfig(**config_data.get('data', {})),
            algorithms=AlgorithmParams(**config_data.get('algorithms', {})),
            acceptance=AcceptanceFunctionParams(**config_data.get('acceptance', {})),
            experiment=ExperimentConfig(**config_data.get('experiment', {}))
        )
    
    @classmethod
    def from_cli_args(cls, args: Dict[str, Any]) -> 'Config':
        """Create configuration from command line arguments."""
        config = cls()
        
        # Map CLI args to config fields
        if 'vehicle_type' in args:
            config.experiment.vehicle_type = args['vehicle_type']
        if 'year' in args:
            config.experiment.year = int(args['year'])
        if 'month' in args:
            config.experiment.month = int(args['month'])
        if 'borough' in args:
            config.experiment.borough = args['borough']
        if 'start_hour' in args:
            config.experiment.start_hour = int(args['start_hour'])
        if 'end_hour' in args:
            config.experiment.end_hour = int(args['end_hour'])
        if 'methods' in args:
            config.experiment.methods = args['methods'].split(',') if isinstance(args['methods'], str) else args['methods']
        if 'acceptance_function' in args:
            config.experiment.acceptance_function = args['acceptance_function']
        if 'simulation_range' in args:
            config.experiment.simulation_range = int(args['simulation_range'])
        if 'num_evaluations' in args:
            config.experiment.num_evaluations = int(args['num_evaluations'])
        if 'start_day' in args:
            config.experiment.start_day = int(args['start_day'])
        if 'end_day' in args:
            config.experiment.end_day = int(args['end_day'])
            
        return config
    
    def to_json(self, output_path: str):
        """Save configuration to JSON file."""
        config_dict = {
            'data': asdict(self.data),
            'algorithms': asdict(self.algorithms),
            'acceptance': asdict(self.acceptance),
            'experiment': asdict(self.experiment)
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []
        
        # Validate time range
        if self.experiment.start_hour >= self.experiment.end_hour:
            errors.append(f"start_hour ({self.experiment.start_hour}) must be less than end_hour ({self.experiment.end_hour})")
        
        if self.experiment.start_hour < 0 or self.experiment.start_hour > 23:
            errors.append(f"start_hour must be 0-23, got {self.experiment.start_hour}")
            
        if self.experiment.end_hour < 1 or self.experiment.end_hour > 24:
            errors.append(f"end_hour must be 1-24, got {self.experiment.end_hour}")
        
        # Validate methods
        valid_methods = {"hikima", "maps", "linucb", "linear_program"}
        invalid_methods = set(self.experiment.methods) - valid_methods
        if invalid_methods:
            errors.append(f"Invalid methods: {invalid_methods}. Valid methods: {valid_methods}")
        
        # Validate acceptance function
        if self.experiment.acceptance_function not in ["PL", "Sigmoid"]:
            errors.append(f"acceptance_function must be 'PL' or 'Sigmoid', got '{self.experiment.acceptance_function}'")
        
        # Validate vehicle type
        if self.experiment.vehicle_type not in ["green", "yellow", "fhv"]:
            errors.append(f"vehicle_type must be 'green', 'yellow', or 'fhv', got '{self.experiment.vehicle_type}'")
        
        # Validate borough
        valid_boroughs = {"Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"}
        if self.experiment.borough not in valid_boroughs:
            errors.append(f"borough must be one of {valid_boroughs}, got '{self.experiment.borough}'")
        
        # Validate simulation range
        total_hours = self.experiment.end_hour - self.experiment.start_hour
        if self.experiment.simulation_range < 1:
            errors.append(f"simulation_range must be at least 1, got {self.experiment.simulation_range}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
        
        return True
    
    def get_experiment_id(self) -> str:
        """Generate unique experiment identifier."""
        methods_str = "_".join(sorted(self.experiment.methods))
        time_range = f"{self.experiment.start_hour:02d}-{self.experiment.end_hour:02d}"
        
        if self.experiment.start_day == self.experiment.end_day:
            day_str = f"day{self.experiment.start_day:02d}"
        else:
            day_str = f"days{self.experiment.start_day:02d}-{self.experiment.end_day:02d}"
        
        return (f"{self.experiment.vehicle_type}_{self.experiment.year}{self.experiment.month:02d}_"
                f"{self.experiment.borough}_{time_range}_{day_str}_"
                f"{methods_str}_{self.experiment.acceptance_function}_"
                f"sim{self.experiment.simulation_range}_eval{self.experiment.num_evaluations}")
    
    def is_hikima_replication(self) -> bool:
        """Check if this configuration replicates the original Hikima paper setup."""
        return (
            self.experiment.start_hour == 10 and
            self.experiment.end_hour == 20 and
            self.experiment.time_interval_minutes == 5 and
            self.experiment.simulation_range == 120 and  # 10 hours * 12 (5-min intervals per hour)
            self.experiment.num_evaluations == 100 and
            self.experiment.acceptance_function == "PL"
        ) 