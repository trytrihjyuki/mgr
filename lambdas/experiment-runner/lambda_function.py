"""
Pricing Methods Benchmark Lambda Function

This Lambda function runs systematic benchmarks of 4 pricing methods:
1. HikimaMinMaxCostFlow - Extracted from the provided Hikima source code
2. MAPS - Area-based pricing algorithm 
3. LinUCB - Contextual bandit learning
4. LinearProgram - Gupta-Nagarajan LP approach

The function loads data from S3, runs experiments using real NYC taxi data,
and saves results back to S3 for analysis.
"""

import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import os
import io
import traceback
import pyproj
from geopy.distance import geodesic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our pricing methods
import sys
sys.path.append('/var/task/src')

from pricing_methods import HikimaMinMaxCostFlow, MAPS, LinUCB, LinearProgram
from pricing_methods.base_method import PricingResult


class PricingBenchmarkRunner:
    """
    Main class for running pricing method benchmarks.
    """
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.environ.get('S3_BUCKET', 'taxi-pricing-benchmark')
        self.geod = pyproj.Geod(ellps='WGS84')
        
        # Initialize pricing methods
        self.pricing_methods = {}
        
    def load_config(self, config_name: str = 'benchmark_config.json') -> Dict[str, Any]:
        """Load experiment configuration from S3."""
        try:
            config_key = f"configs/{config_name}"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=config_key)
            config = json.loads(response['Body'].read().decode('utf-8'))
            logger.info(f"✅ Loaded configuration: {config_name}")
            return config
        except Exception as e:
            logger.error(f"❌ Failed to load config {config_name}: {e}")
            # Return default config
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if loading fails."""
        return {
            "methods_config": {
                "HikimaMinMaxCostFlow": {"enabled": True, "parameters": {}},
                "MAPS": {"enabled": True, "parameters": {}},
                "LinUCB": {"enabled": True, "parameters": {}},
                "LinearProgram": {"enabled": True, "parameters": {}}
            },
            "data_config": {
                "min_trip_distance": 0.001,
                "min_total_amount": 0.001,
                "distance_conversion_factor": 1.60934
            },
            "time_config": {
                "business_hours": {"start_hour": 10, "end_hour": 20}
            }
        }
    
    def initialize_pricing_methods(self, config: Dict[str, Any]):
        """Initialize all pricing methods based on configuration."""
        methods_config = config.get('methods_config', {})
        
        for method_name, method_config in methods_config.items():
            if not method_config.get('enabled', False):
                continue
                
            parameters = method_config.get('parameters', {})
            
            try:
                if method_name == "HikimaMinMaxCostFlow":
                    self.pricing_methods[method_name] = HikimaMinMaxCostFlow(**parameters)
                elif method_name == "MAPS":
                    self.pricing_methods[method_name] = MAPS(**parameters)
                elif method_name == "LinUCB":
                    self.pricing_methods[method_name] = LinUCB(**parameters)
                elif method_name == "LinearProgram":
                    self.pricing_methods[method_name] = LinearProgram(**parameters)
                else:
                    logger.warning(f"⚠️ Unknown pricing method: {method_name}")
                    
                logger.info(f"✅ Initialized pricing method: {method_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize {method_name}: {e}")
    
    def load_data_from_s3(self, vehicle_type: str, year: int, month: int, day: int = None) -> pd.DataFrame:
        """Load trip data from S3."""
        if day:
            s3_key = f"datasets/{vehicle_type}/year={year}/month={month:02d}/day={day:02d}/{vehicle_type}_tripdata_{year}-{month:02d}-{day:02d}.parquet"
        else:
            s3_key = f"datasets/{vehicle_type}/year={year}/month={month:02d}/{vehicle_type}_tripdata_{year}-{month:02d}.parquet"
        
        try:
            logger.info(f"📥 Loading data from s3://{self.bucket_name}/{s3_key}")
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            df = pd.read_parquet(io.BytesIO(response['Body'].read()))
            logger.info(f"✅ Loaded {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to load data from {s3_key}: {e}")
            raise
    
    def load_taxi_zones(self) -> pd.DataFrame:
        """Load taxi zone reference data from S3."""
        try:
            s3_key = "reference_data/area_info.csv"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            zones_df = pd.read_csv(io.BytesIO(response['Body'].read()))
            logger.info(f"✅ Loaded {len(zones_df)} taxi zones")
            return zones_df
        except Exception as e:
            logger.warning(f"⚠️ Could not load taxi zones: {e}")
            # Return minimal zones data
            return pd.DataFrame({
                'LocationID': [1, 161, 162, 230],
                'zone': ['Newark Airport', 'Midtown Center', 'Midtown East', 'Times Square'],
                'borough': ['EWR', 'Manhattan', 'Manhattan', 'Manhattan'],
                'latitude': [40.689, 40.758, 40.757, 40.760],
                'longitude': [-74.172, -73.978, -73.972, -73.984]
            })
    
    def preprocess_data(self, df: pd.DataFrame, config: Dict[str, Any], 
                       borough: str = None, start_hour: int = 10, end_hour: int = 20) -> Dict[str, Any]:
        """
        Preprocess trip data for experiments.
        
        This follows the exact methodology from the provided Hikima source code.
        """
        logger.info(f"🔄 Preprocessing data for {borough or 'all boroughs'}")
        
        # Load taxi zones
        zones_df = self.load_taxi_zones()
        
        # Data cleaning following Hikima methodology
        data_config = config.get('data_config', {})
        min_distance = data_config.get('min_trip_distance', 0.001)
        min_amount = data_config.get('min_total_amount', 0.001)
        
        # Handle different column name formats
        pickup_col = None
        dropoff_col = None
        pu_location_col = None
        do_location_col = None
        
        for col in df.columns:
            if 'pickup_datetime' in col.lower():
                pickup_col = col
            elif 'dropoff_datetime' in col.lower():
                dropoff_col = col
            elif 'pulocationid' in col.lower():
                pu_location_col = col
            elif 'dolocationid' in col.lower():
                do_location_col = col
        
        if pickup_col and dropoff_col:
            df[pickup_col] = pd.to_datetime(df[pickup_col])
            df[dropoff_col] = pd.to_datetime(df[dropoff_col])
        
        # Filter data following Hikima criteria
        df = df[
            (df.get('trip_distance', 0) > min_distance) &
            (df.get('total_amount', 0) > min_amount)
        ]
        
        # Time filtering
        if pickup_col:
            df['hour'] = df[pickup_col].dt.hour
            df = df[(df['hour'] >= start_hour) & (df['hour'] < end_hour)]
        
        # Add zone information
        if pu_location_col and pu_location_col in df.columns:
            df = df.merge(zones_df, left_on=pu_location_col, right_on='LocationID', how='left')
            if borough:
                df = df[df['borough'] == borough]
        
        # Sort by distance (required by MAPS algorithm)
        df = df.sort_values('trip_distance', ascending=True)
        
        # Convert distance to km
        conversion_factor = data_config.get('distance_conversion_factor', 1.60934)
        df['trip_distance_km'] = df['trip_distance'] * conversion_factor
        
        # Sample data for reasonable Lambda execution time
        max_sample = data_config.get('max_sample_size', 1000)
        if len(df) > max_sample:
            df = df.sample(n=max_sample, random_state=42)
        
        logger.info(f"✅ Preprocessed data: {len(df)} records")
        
        return {
            'data': df,
            'original_size': len(df),
            'time_range': f"{start_hour:02d}:00-{end_hour:02d}:00",
            'borough': borough,
            'preprocessing_complete': True
        }
    
    def calculate_distance_matrix(self, requesters_data: pd.DataFrame, 
                                 taxis_data: pd.DataFrame, zones_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate distance matrix between requesters and taxis.
        
        This follows the distance calculation from the Hikima source code.
        """
        n = len(requesters_data)
        m = len(taxis_data)
        
        distance_matrix = np.zeros((n, m))
        
        # Get coordinates for requesters and taxis
        requester_coords = self._get_coordinates(requesters_data, zones_df)
        taxi_coords = self._get_coordinates(taxis_data, zones_df)
        
        # Calculate distances using geodesic distance
        for i in range(n):
            for j in range(m):
                try:
                    distance_km = geodesic(requester_coords[i], taxi_coords[j]).kilometers
                    distance_matrix[i, j] = distance_km
                except Exception:
                    # Fallback to simple Euclidean distance
                    lat_diff = requester_coords[i][0] - taxi_coords[j][0]
                    lon_diff = requester_coords[i][1] - taxi_coords[j][1]
                    distance_matrix[i, j] = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Rough km conversion
        
        return distance_matrix
    
    def _get_coordinates(self, data: pd.DataFrame, zones_df: pd.DataFrame) -> List[tuple]:
        """Get coordinates for trip data."""
        coordinates = []
        
        for _, row in data.iterrows():
            # Try to get from LocationID first
            if 'PULocationID' in row and not pd.isna(row['PULocationID']):
                zone_info = zones_df[zones_df['LocationID'] == row['PULocationID']]
                if len(zone_info) > 0:
                    lat = zone_info.iloc[0]['latitude']
                    lon = zone_info.iloc[0]['longitude']
                    coordinates.append((lat, lon))
                    continue
            
            # Try direct coordinates
            if 'pickup_latitude' in row and 'pickup_longitude' in row:
                if not pd.isna(row['pickup_latitude']) and not pd.isna(row['pickup_longitude']):
                    coordinates.append((row['pickup_latitude'], row['pickup_longitude']))
                    continue
            
            # Default to Manhattan center
            coordinates.append((40.7589, -73.9851))
        
        return coordinates
    
    def run_experiment(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the main pricing benchmark experiment.
        """
        start_time = datetime.now()
        experiment_id = f"benchmark_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🧪 Starting experiment: {experiment_id}")
        
        try:
            # Load configuration
            config_name = event.get('config_name', 'benchmark_config.json')
            config = self.load_config(config_name)
            
            # Initialize pricing methods
            self.initialize_pricing_methods(config)
            
            # Extract experiment parameters
            vehicle_type = event.get('vehicle_type', 'green')
            year = event.get('year', 2019)
            month = event.get('month', 10)
            day = event.get('day', 1)
            borough = event.get('borough', 'Manhattan')
            scenario_name = event.get('scenario', 'comprehensive_benchmark')
            
            # Get scenario configuration
            scenarios_config = config.get('experiment_scenarios', {})
            if scenario_name in scenarios_config:
                scenario = scenarios_config[scenario_name]
                start_hour = config['time_config'][scenario['time_range']]['start_hour']
                end_hour = config['time_config'][scenario['time_range']]['end_hour']
                methods_to_run = scenario.get('methods', list(self.pricing_methods.keys()))
                acceptance_functions = scenario.get('acceptance_functions', ['PL'])
            else:
                # Default parameters
                start_hour = event.get('start_hour', 10)
                end_hour = event.get('end_hour', 20)
                methods_to_run = event.get('methods', list(self.pricing_methods.keys()))
                acceptance_functions = event.get('acceptance_functions', ['PL'])
            
            # Load and preprocess data
            df = self.load_data_from_s3(vehicle_type, year, month, day)
            preprocessed = self.preprocess_data(df, config, borough, start_hour, end_hour)
            
            if len(preprocessed['data']) == 0:
                raise ValueError("No data available after preprocessing")
            
            # Split data into requesters and taxis (following Hikima methodology)
            data = preprocessed['data']
            requesters_data = data.copy()  # All records as potential requesters
            taxis_data = data.copy()       # All records as potential taxis
            
            # Load taxi zones for distance calculation
            zones_df = self.load_taxi_zones()
            
            # Calculate distance matrix
            distance_matrix = self.calculate_distance_matrix(requesters_data, taxis_data, zones_df)
            
            # Run experiments for each method and acceptance function
            results = []
            
            for method_name in methods_to_run:
                if method_name not in self.pricing_methods:
                    logger.warning(f"⚠️ Method {method_name} not available")
                    continue
                
                for acceptance_function in acceptance_functions:
                    logger.info(f"🔄 Running {method_name} with {acceptance_function} acceptance")
                    
                    try:
                        # Update method parameters for this acceptance function
                        method = self.pricing_methods[method_name]
                        method.acceptance_function = acceptance_function
                        
                        # Run the pricing method
                        result = method.calculate_prices(
                            requesters_data=requesters_data,
                            taxis_data=taxis_data,
                            distance_matrix=distance_matrix,
                            current_hour=(start_hour + end_hour) // 2,
                            acceptance_function=acceptance_function
                        )
                        
                        # Add experiment metadata
                        result.additional_metrics.update({
                            'experiment_id': experiment_id,
                            'method_name': method_name,
                            'acceptance_function': acceptance_function,
                            'vehicle_type': vehicle_type,
                            'borough': borough,
                            'year': year,
                            'month': month,
                            'day': day,
                            'time_range': f"{start_hour:02d}:00-{end_hour:02d}:00"
                        })
                        
                        results.append(result)
                        
                        logger.info(f"✅ {method_name} ({acceptance_function}): "
                                  f"Objective={result.objective_value:.2f}, "
                                  f"Time={result.computation_time:.3f}s")
                        
                    except Exception as e:
                        logger.error(f"❌ Error in {method_name} with {acceptance_function}: {e}")
                        logger.error(traceback.format_exc())
            
            # Aggregate results
            experiment_results = {
                'experiment_id': experiment_id,
                'timestamp': start_time.isoformat(),
                'configuration': {
                    'vehicle_type': vehicle_type,
                    'year': year,
                    'month': month,
                    'day': day,
                    'borough': borough,
                    'time_range': f"{start_hour:02d}:00-{end_hour:02d}:00",
                    'scenario': scenario_name,
                    'methods_tested': methods_to_run,
                    'acceptance_functions': acceptance_functions
                },
                'data_summary': {
                    'total_records': len(data),
                    'requesters': len(requesters_data),
                    'taxis': len(taxis_data),
                    'preprocessing_info': preprocessed
                },
                'results': [self._result_to_dict(r) for r in results],
                'execution_time_seconds': (datetime.now() - start_time).total_seconds()
            }
            
            # Save results to S3
            self._save_results_to_s3(experiment_results)
            
            logger.info(f"🎉 Experiment {experiment_id} completed successfully")
            return experiment_results
            
        except Exception as e:
            logger.error(f"❌ Experiment failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'experiment_id': experiment_id,
                'status': 'failed',
                'error': str(e),
                'timestamp': start_time.isoformat()
            }
    
    def _result_to_dict(self, result: PricingResult) -> Dict[str, Any]:
        """Convert PricingResult to dictionary for JSON serialization."""
        return {
            'method_name': result.method_name,
            'objective_value': float(result.objective_value),
            'computation_time': float(result.computation_time),
            'n_prices': len(result.prices),
            'average_price': float(np.mean(result.prices)) if len(result.prices) > 0 else 0.0,
            'average_acceptance_probability': float(np.mean(result.acceptance_probabilities)) if len(result.acceptance_probabilities) > 0 else 0.0,
            'n_matches': len(result.matches),
            'match_rate': len(result.matches) / len(result.prices) if len(result.prices) > 0 else 0.0,
            'additional_metrics': result.additional_metrics or {}
        }
    
    def _save_results_to_s3(self, results: Dict[str, Any]):
        """Save experiment results to S3."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"experiments/results/{results['experiment_id']}/{timestamp}_results.json"
            
            results_json = json.dumps(results, indent=2, default=str)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=results_json,
                ContentType='application/json'
            )
            
            logger.info(f"💾 Results saved to s3://{self.bucket_name}/{s3_key}")
            results['s3_location'] = f"s3://{self.bucket_name}/{s3_key}"
            
        except Exception as e:
            logger.error(f"❌ Failed to save results to S3: {e}")


def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Expected event format:
    {
        "vehicle_type": "green",
        "year": 2019,
        "month": 10, 
        "day": 1,
        "borough": "Manhattan",
        "scenario": "comprehensive_benchmark",
        "config_name": "benchmark_config.json"
    }
    """
    logger.info(f"📥 Received event: {json.dumps(event, default=str)}")
    
    runner = PricingBenchmarkRunner()
    results = runner.run_experiment(event)
    
    return {
        'statusCode': 200,
        'body': json.dumps(results, default=str),
        'headers': {
            'Content-Type': 'application/json'
        }
    } 