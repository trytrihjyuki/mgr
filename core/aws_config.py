#!/usr/bin/env python3
"""
AWS Configuration and Settings for Bipartite Matching Optimization Project.
Manages S3 bucket structure and AWS service configurations.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class AWSConfig:
    """Centralized AWS configuration management."""
    
    # S3 Bucket Configuration
    BUCKET_NAME = "magisterka"
    
    # S3 Key Prefixes (folder structure)
    S3_PREFIXES = {
        'datasets': {
            'root': 'datasets',
            'rideshare': 'datasets/{vehicle_type}/year={year}/month={month:02d}/'
        },
        'experiments': {
            'results': 'experiments/results',
            'analysis': 'experiments/analysis',
            'logs': 'experiments/logs'
        }
    }
    
    # AWS Region
    REGION = os.getenv('AWS_REGION', 'us-east-1')
    
    # Compute Configuration
    COMPUTE_CONFIG = {
        'instance_type': 't3.medium',  # Simple, cost-effective for experiments
        'timeout_minutes': 60,         # Max experiment runtime
        'max_concurrent_jobs': 3       # Parallel execution limit
    }
    
    @classmethod
    def get_s3_dataset_key(cls, vehicle_type: str, year: int, month: int, filename: str) -> str:
        """Generate S3 key for dataset files."""
        prefix = cls.S3_PREFIXES['datasets']['rideshare'].format(
            vehicle_type=vehicle_type, year=year, month=month
        )
        return f"{prefix}{filename}"
    
    @classmethod
    def get_s3_results_key(cls, experiment_type: str, filename: str) -> str:
        """Generate S3 key for experiment results."""
        return f"{cls.S3_PREFIXES['experiments']['results']}/{experiment_type}/{filename}"
    
    @classmethod
    def get_s3_analysis_key(cls, analysis_type: str, filename: str) -> str:
        """Generate S3 key for analysis outputs."""
        return f"{cls.S3_PREFIXES['experiments']['analysis']}/{analysis_type}/{filename}"
    
    @classmethod
    def get_s3_logs_key(cls, experiment_id: str, filename: str) -> str:
        """Generate S3 key for log files."""
        return f"{cls.S3_PREFIXES['experiments']['logs']}/{experiment_id}/{filename}"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate AWS configuration."""
        required_env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
        
        # Check if running on EC2 (has IAM role) or has explicit credentials
        has_credentials = all(os.getenv(var) for var in required_env_vars)
        is_ec2 = os.path.exists('/proc/xen') or os.path.exists('/sys/hypervisor/uuid')
        
        if not (has_credentials or is_ec2):
            print("⚠️  AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            return False
        
        return True

# AWS Service Endpoints for different regions
AWS_ENDPOINTS = {
    'us-east-1': {
        's3': 'https://s3.amazonaws.com',
        'ec2': 'https://ec2.us-east-1.amazonaws.com'
    },
    'eu-west-1': {
        's3': 'https://s3.eu-west-1.amazonaws.com',
        'ec2': 'https://ec2.eu-west-1.amazonaws.com'
    }
} 