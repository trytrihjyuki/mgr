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
import os
import sys
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all imports to help debug dependency issues."""
    test_results = {}
    
    # Test basic imports
    try:
        import boto3
        test_results['boto3'] = f"✅ Success: {boto3.__version__}"
    except Exception as e:
        test_results['boto3'] = f"❌ Failed: {str(e)}"
    
    # Test numpy specifically
    try:
        import numpy as np
        test_results['numpy'] = f"✅ Success: {np.__version__}"
        # Test basic numpy operations
        arr = np.array([1, 2, 3])
        test_results['numpy_ops'] = f"✅ Array test: {arr}"
    except Exception as e:
        test_results['numpy'] = f"❌ Failed: {str(e)}"
        test_results['numpy_traceback'] = traceback.format_exc()
    
    # Test pandas
    try:
        import pandas as pd
        test_results['pandas'] = f"✅ Success: {pd.__version__}"
    except Exception as e:
        test_results['pandas'] = f"❌ Failed: {str(e)}"
    
    # Test other scientific packages
    for package_name in ['scipy', 'networkx', 'geopy', 'pulp']:
        try:
            if package_name == 'scipy':
                import scipy
                test_results[package_name] = f"✅ Success: {scipy.__version__}"
            elif package_name == 'networkx':
                import networkx as nx
                test_results[package_name] = f"✅ Success: {nx.__version__}"
            elif package_name == 'geopy':
                import geopy
                test_results[package_name] = f"✅ Success: {geopy.__version__}"
            elif package_name == 'pulp':
                import pulp
                test_results[package_name] = f"✅ Success: Available"
        except Exception as e:
            test_results[package_name] = f"❌ Failed: {str(e)}"
    
    # Test system info
    test_results['python_version'] = sys.version
    test_results['python_path'] = sys.path
    test_results['environment'] = dict(os.environ)
    
    return test_results

# Now try the main imports with better error handling
try:
    import boto3
    import pandas as pd
    import numpy as np
    import io
    # Import our pricing methods only if numpy works
    from src.pricing_methods import HikimaMinMaxCostFlow, MAPS, LinUCB, LinearProgram
    from src.pricing_methods.base_method import PricingResult
    IMPORTS_SUCCESSFUL = True
except Exception as e:
    logger.error(f"❌ Import error: {e}")
    logger.error(traceback.format_exc())
    IMPORTS_SUCCESSFUL = False


class PricingBenchmarkRunner:
    """
    Main class for running pricing method benchmarks.
    """
    
    def __init__(self):
        if not IMPORTS_SUCCESSFUL:
            raise RuntimeError("Critical imports failed")
        
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.environ.get('S3_BUCKET', 'magisterka')
        
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
                "distance_conversion_factor": 1.60934,
                "max_sample_size": 200
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
        ].copy()  # Make a copy to avoid SettingWithCopyWarning
        
        # Time filtering
        if pickup_col:
            df.loc[:, 'hour'] = df[pickup_col].dt.hour
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
        
        # Calculate distances using simple haversine formula
        for i in range(n):
            for j in range(m):
                # Use haversine formula for distance calculation
                lat1, lon1 = requester_coords[i]
                lat2, lon2 = taxi_coords[j]
                
                # Convert to radians
                lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                
                # Haversine formula
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                r = 6371  # Earth's radius in kilometers
                distance_km = r * c
                
                distance_matrix[i, j] = distance_km
        
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
        # Generate a unique ID for this training/experiment run
        training_id = f"{random.randint(100_000_000, 999_999_999)}"
        experiment_id = f"benchmark_{training_id}_{start_time.strftime('%Y%m%d')}"
        
        logger.info(f"🧪 Starting experiment: {experiment_id} (Training ID: {training_id})")
        
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
            day = event.get('day', None)  # Default to None for monthly data
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
            logger.info(f"📥 Loading data for {vehicle_type} {year}-{month:02d}")
            df = self.load_data_from_s3(vehicle_type, year, month, day)
            logger.info(f"📊 Raw data loaded: {len(df)} records")
            
            preprocessed = self.preprocess_data(df, config, borough, start_hour, end_hour)
            logger.info(f"🔄 Data preprocessed: {len(preprocessed['data'])} records")
            
            if len(preprocessed['data']) == 0:
                raise ValueError("No data available after preprocessing")
            
            # Split data into requesters and taxis (following Hikima methodology)
            data = preprocessed['data']
            requesters_data = data.copy()  # All records as potential requesters
            taxis_data = data.copy()       # All records as potential taxis
            
            # Load taxi zones for distance calculation
            zones_df = self.load_taxi_zones()
            
            # Calculate distance matrix
            logger.info(f"🔢 Calculating distance matrix: {len(requesters_data)}x{len(taxis_data)}")
            distance_matrix = self.calculate_distance_matrix(requesters_data, taxis_data, zones_df)
            logger.info(f"✅ Distance matrix calculated")
            
            # Run experiments for each method and acceptance function
            results = []
            
            for method_name in methods_to_run:
                if method_name not in self.pricing_methods:
                    logger.warning(f"⚠️ Method {method_name} not available")
                    continue
                
                for acceptance_function in acceptance_functions:
                    logger.info(f"🔄 Running {method_name} with {acceptance_function} acceptance")
                    method_start_time = datetime.now()
                    
                    try:
                        # Update method parameters for this acceptance function
                        method = self.pricing_methods[method_name]
                        method.acceptance_function = acceptance_function
                        
                        # Run the pricing method with timeout protection
                        logger.info(f"   📊 Data size: {len(requesters_data)} requesters, {len(taxis_data)} taxis")
                        
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
                        
                        method_duration = (datetime.now() - method_start_time).total_seconds()
                        logger.info(f"✅ {method_name} ({acceptance_function}): "
                                  f"Objective={result.objective_value:.2f}, "
                                  f"Time={result.computation_time:.3f}s, "
                                  f"Total={method_duration:.3f}s")
                        
                        # Check if we're approaching Lambda timeout
                        total_elapsed = (datetime.now() - start_time).total_seconds()
                        if total_elapsed > 800:  # 13.3 minutes, leave buffer for cleanup
                            logger.warning(f"⚠️ Approaching Lambda timeout ({total_elapsed:.1f}s), stopping early")
                            break
                        
                    except Exception as e:
                        method_duration = (datetime.now() - method_start_time).total_seconds()
                        logger.error(f"❌ Error in {method_name} with {acceptance_function} after {method_duration:.1f}s: {e}")
                        logger.error(traceback.format_exc())
                        
                        # Don't let one method failure stop the entire experiment
                        continue
                
                # Check timeout after each method completes
                total_elapsed = (datetime.now() - start_time).total_seconds()
                if total_elapsed > 800:  # 13.3 minutes
                    logger.warning(f"⚠️ Approaching Lambda timeout ({total_elapsed:.1f}s), stopping experiment")
                    break
            
            # Aggregate results
            experiment_results = {
                'experiment_id': experiment_id,
                'training_id': training_id,
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
                'training_id': training_id,
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
        """Save experiment results to S3 using the new structured path."""
        try:
            config = results['configuration']
            training_id = results['training_id']
            
            # Create the structured S3 path
            s3_key = (f"experiments/type={config['vehicle_type']}"
                      f"/year={config['year']}/month={config['month']:02d}"
                      f"/day={config['day']:02d}/{training_id}.json")
            
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
    
    Or for testing:
    {
        "test_mode": "numpy_only"
    }
    """
    logger.info(f"📥 Received event: {json.dumps(event, default=str)}")
    
    # Handle test mode
    if event.get('test_mode') == 'numpy_only':
        logger.info("🧪 Running numpy test mode")
        test_results = test_imports()
        return {
            'statusCode': 200,
            'body': json.dumps({
                'test_mode': 'numpy_only',
                'results': test_results,
                'imports_successful': IMPORTS_SUCCESSFUL
            }, default=str),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
    
    # Check if imports were successful before proceeding
    if not IMPORTS_SUCCESSFUL:
        error_details = test_imports()
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Import failure - cannot proceed',
                'details': error_details
            }, default=str),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
    
    try:
        runner = PricingBenchmarkRunner()
        results = runner.run_experiment(event)
        
        # This is returned to the client (e.g., run_benchmark.py)
        return {
            'statusCode': 200,
            'body': json.dumps(results, default=str),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
    except Exception as e:
        logger.error(f"❌ Lambda handler error: {e}")
        logger.error(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'traceback': traceback.format_exc()
            }),
            'headers': {
                'Content-Type': 'application/json'
            }
        } 