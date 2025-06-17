#!/usr/bin/env python3
"""
Experiment Runner Lambda Function
Runs bipartite matching experiments on rideshare data stored in S3.
"""

import json
import boto3
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import os
import io
from urllib.parse import urlparse
import traceback
import random
import math

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class BipartiteMatchingExperiment:
    """
    Runs bipartite matching experiments on rideshare data.
    """
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.environ.get('S3_BUCKET', 'magisterka')
    
    def load_data_from_s3(self, vehicle_type: str, year: int, month: int) -> pd.DataFrame:
        """Load rideshare data from S3."""
        s3_key = f"datasets/{vehicle_type}/year={year}/month={month:02d}/{vehicle_type}_tripdata_{year}-{month:02d}.parquet"
        
        try:
            logger.info(f"Loading data from s3://{self.bucket_name}/{s3_key}")
            
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            df = pd.read_parquet(io.BytesIO(response['Body'].read()))
            
            logger.info(f"✅ Loaded {len(df)} records from {s3_key}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load data from {s3_key}: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame, simulation_range: int = 5) -> Dict[str, Any]:
        """
        Preprocess rideshare data for bipartite matching experiment.
        
        Args:
            df: Raw rideshare data
            simulation_range: Number of simulation scenarios
            
        Returns:
            Preprocessed data for experiments
        """
        logger.info("🔄 Preprocessing rideshare data...")
        
        # Basic data cleaning
        df = df.dropna(subset=['pickup_datetime', 'dropoff_datetime'])
        
        # Convert datetime columns
        if 'pickup_datetime' in df.columns:
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        if 'dropoff_datetime' in df.columns:
            df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
        
        # Extract time-based features
        df['hour'] = df['pickup_datetime'].dt.hour
        df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
        
        # Create simplified location zones (for demo purposes)
        # In real implementation, this would use actual taxi zones
        if 'pickup_longitude' in df.columns and 'pickup_latitude' in df.columns:
            df['pickup_zone'] = (
                (df['pickup_longitude'] // 0.01).astype(int).astype(str) + "_" +
                (df['pickup_latitude'] // 0.01).astype(int).astype(str)
            )
        
        if 'dropoff_longitude' in df.columns and 'dropoff_latitude' in df.columns:
            df['dropoff_zone'] = (
                (df['dropoff_longitude'] // 0.01).astype(int).astype(str) + "_" +
                (df['dropoff_latitude'] // 0.01).astype(int).astype(str)
            )
        
        # Sample data for simulation
        sample_size = min(len(df), 10000)  # Limit for Lambda execution
        df_sample = df.sample(n=sample_size, random_state=42)
        
        # Create demand/supply scenarios
        scenarios = []
        for i in range(simulation_range):
            # Simulate different demand/supply ratios
            demand_factor = random.uniform(0.5, 1.5)
            supply_factor = random.uniform(0.5, 1.5)
            
            scenarios.append({
                'scenario_id': i,
                'demand_factor': demand_factor,
                'supply_factor': supply_factor,
                'data_sample': df_sample.sample(frac=demand_factor, replace=True).reset_index(drop=True)
            })
        
        return {
            'original_size': len(df),
            'processed_size': len(df_sample),
            'scenarios': scenarios,
            'preprocessing_time': datetime.now().isoformat()
        }
    
    def run_bipartite_matching(self, scenario_data: Dict[str, Any], 
                              acceptance_function: str = 'PL') -> Dict[str, Any]:
        """
        Run bipartite matching algorithm on scenario data.
        
        Args:
            scenario_data: Preprocessed scenario data
            acceptance_function: Type of acceptance function ('PL' or 'Sigmoid')
            
        Returns:
            Matching results
        """
        df = scenario_data['data_sample']
        
        # Simplified bipartite matching simulation
        # In real implementation, this would use the actual algorithm from the paper
        
        # Create riders and drivers
        num_rides = len(df)
        num_drivers = int(num_rides * scenario_data['supply_factor'])
        
        # Simulate acceptance probabilities
        if acceptance_function == 'PL':
            # Piecewise Linear acceptance function
            base_acceptance = [random.uniform(0.6, 0.9) for _ in range(num_rides)]
            distance_penalty = [random.uniform(0.0, 0.3) for _ in range(num_rides)]
            acceptance_probs = [max(0.1, base - penalty) for base, penalty in zip(base_acceptance, distance_penalty)]
        else:
            # Sigmoid acceptance function
            logits = [random.gauss(0, 1) for _ in range(num_rides)]
            acceptance_probs = [1 / (1 + math.exp(-logit)) for logit in logits]
        
        # Simulate matching decisions
        matched = [1 if random.random() < prob else 0 for prob in acceptance_probs]
        
        # Calculate metrics
        total_requests = num_rides
        successful_matches = sum(matched)
        match_rate = successful_matches / total_requests if total_requests > 0 else 0
        avg_acceptance_prob = sum(acceptance_probs) / len(acceptance_probs) if acceptance_probs else 0
        
        return {
            'scenario_id': scenario_data['scenario_id'],
            'total_requests': total_requests,
            'available_drivers': num_drivers,
            'successful_matches': int(successful_matches),
            'match_rate': float(match_rate),
            'avg_acceptance_probability': float(avg_acceptance_prob),
            'acceptance_function': acceptance_function,
            'supply_demand_ratio': scenario_data['supply_factor'] / scenario_data['demand_factor']
        }
    
    def run_experiment(self, vehicle_type: str, year: int, month: int,
                      place: str = "Manhattan", simulation_range: int = 5,
                      acceptance_function: str = 'PL') -> Dict[str, Any]:
        """
        Run complete rideshare experiment.
        
        Args:
            vehicle_type: Type of vehicle data
            year: Year of data
            month: Month of data
            place: Location (for metadata)
            simulation_range: Number of simulation scenarios
            acceptance_function: Type of acceptance function
            
        Returns:
            Complete experiment results
        """
        start_time = datetime.now()
        experiment_id = f"rideshare_{vehicle_type}_{year}_{month:02d}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🧪 Starting experiment: {experiment_id}")
        
        try:
            # Load and preprocess data
            df = self.load_data_from_s3(vehicle_type, year, month)
            preprocessed = self.preprocess_data(df, simulation_range)
            
            # Run experiments on all scenarios
            scenario_results = []
            for scenario in preprocessed['scenarios']:
                result = self.run_bipartite_matching(scenario, acceptance_function)
                scenario_results.append(result)
            
            # Aggregate results
            total_requests = sum(r['total_requests'] for r in scenario_results)
            total_matches = sum(r['successful_matches'] for r in scenario_results)
            match_rates = [r['match_rate'] for r in scenario_results]
            acceptance_probs = [r['avg_acceptance_probability'] for r in scenario_results]
            avg_match_rate = sum(match_rates) / len(match_rates) if match_rates else 0
            avg_acceptance_prob = sum(acceptance_probs) / len(acceptance_probs) if acceptance_probs else 0
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            results = {
                'experiment_id': experiment_id,
                'experiment_type': 'rideshare',
                'parameters': {
                    'vehicle_type': vehicle_type,
                    'year': year,
                    'month': month,
                    'place': place,
                    'simulation_range': simulation_range,
                    'acceptance_function': acceptance_function
                },
                'data_info': {
                    'original_data_size': preprocessed['original_size'],
                    'processed_data_size': preprocessed['processed_size']
                },
                'results': {
                    'total_scenarios': len(scenario_results),
                    'total_requests': total_requests,
                    'total_successful_matches': total_matches,
                    'average_match_rate': float(avg_match_rate),
                    'average_acceptance_probability': float(avg_acceptance_prob),
                    'scenario_details': scenario_results
                },
                'execution_time_seconds': execution_time,
                'timestamp': start_time.isoformat(),
                'status': 'completed'
            }
            
            # Upload results to S3
            self._upload_results_to_s3(results)
            
            logger.info(f"✅ Experiment completed: {experiment_id}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Experiment failed: {e}")
            logger.error(traceback.format_exc())
            
            error_results = {
                'experiment_id': experiment_id,
                'experiment_type': 'rideshare',
                'parameters': {
                    'vehicle_type': vehicle_type,
                    'year': year,
                    'month': month,
                    'place': place,
                    'simulation_range': simulation_range,
                    'acceptance_function': acceptance_function
                },
                'error': str(e),
                'status': 'failed',
                'timestamp': start_time.isoformat()
            }
            
            self._upload_results_to_s3(error_results)
            return error_results
    
    def _upload_results_to_s3(self, results: Dict[str, Any]):
        """Upload experiment results to S3."""
        experiment_id = results['experiment_id']
        s3_key = f"experiments/results/rideshare/{experiment_id}_results.json"
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json.dumps(results, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"📤 Results uploaded to s3://{self.bucket_name}/{s3_key}")
            
        except Exception as e:
            logger.error(f"❌ Failed to upload results: {e}")

def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Expected event format:
    {
        "vehicle_type": "green|yellow|fhv",
        "year": 2019,
        "month": 3,
        "place": "Manhattan",
        "simulation_range": 5,
        "acceptance_function": "PL|Sigmoid"
    }
    """
    
    try:
        # Extract parameters
        vehicle_type = event['vehicle_type']
        year = event['year']
        month = event['month']
        place = event.get('place', 'Manhattan')
        simulation_range = event.get('simulation_range', 5)
        acceptance_function = event.get('acceptance_function', 'PL')
        
        # Run experiment
        experiment = BipartiteMatchingExperiment()
        results = experiment.run_experiment(
            vehicle_type=vehicle_type,
            year=year,
            month=month,
            place=place,
            simulation_range=simulation_range,
            acceptance_function=acceptance_function
        )
        
        return {
            'statusCode': 200 if results['status'] == 'completed' else 500,
            'body': json.dumps(results)
        }
        
    except Exception as e:
        logger.error(f"Lambda execution failed: {e}")
        logger.error(traceback.format_exc())
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'status': 'failed'
            })
        }

if __name__ == "__main__":
    # Test the function locally
    test_event = {
        "vehicle_type": "green",
        "year": 2019,
        "month": 3,
        "place": "Manhattan",
        "simulation_range": 3,
        "acceptance_function": "PL"
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2)) 