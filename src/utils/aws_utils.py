"""
AWS utilities for data management and storage.

Handles S3 operations for loading NYC taxi data and storing experiment results.
"""

import boto3
import pandas as pd
import io
import logging
from typing import Dict, Optional, List
from botocore.exceptions import ClientError, NoCredentialsError
import os
from pathlib import Path
import requests

logger = logging.getLogger(__name__)


class S3DataManager:
    """Manages data operations with AWS S3."""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        """
        Initialize S3 data manager.
        
        Args:
            bucket_name: S3 bucket name for data storage
            region: AWS region for S3 operations
        """
        self.bucket_name = bucket_name
        self.region = region
        
        try:
            self.s3_client = boto3.client('s3', region_name=region)
            self.s3_resource = boto3.resource('s3', region_name=region)
            self.bucket = self.s3_resource.Bucket(bucket_name)
            logger.info(f"✅ Connected to S3 bucket: {bucket_name}")
        except NoCredentialsError:
            logger.warning("⚠️ AWS credentials not found. Using local data only.")
            self.s3_client = None
            self.s3_resource = None
            self.bucket = None
    
    def load_taxi_data(self, vehicle_type: str, year: int, month: int, 
                      local_cache_dir: str = "./data/cache") -> pd.DataFrame:
        """
        Load NYC taxi data from S3 or download if not available.
        
        Args:
            vehicle_type: Type of vehicle (green, yellow, fhv)
            year: Data year
            month: Data month
            local_cache_dir: Local directory for caching data
            
        Returns:
            DataFrame with taxi trip data
        """
        # Create cache directory if needed
        cache_dir = Path(local_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        filename = f"{vehicle_type}_tripdata_{year}-{month:02d}.parquet"
        s3_key = f"datasets/{vehicle_type}/year={year}/month={month:02d}/{filename}"
        local_path = cache_dir / filename
        
        # Try to load from local cache first
        if local_path.exists():
            logger.info(f"📁 Loading cached data: {local_path}")
            return pd.read_parquet(local_path)
        
        # Try to load from S3
        if self.s3_client:
            try:
                logger.info(f"☁️ Loading data from S3: s3://{self.bucket_name}/{s3_key}")
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
                df = pd.read_parquet(io.BytesIO(response['Body'].read()))
                
                # Cache locally for future use
                df.to_parquet(local_path)
                logger.info(f"💾 Cached data locally: {local_path}")
                return df
                
            except ClientError as e:
                logger.warning(f"⚠️ Could not load from S3: {e}")
        
        # Download from NYC TLC if not in S3
        logger.info(f"🌐 Downloading data from NYC TLC...")
        df = self._download_nyc_data(vehicle_type, year, month)
        
        # Cache locally
        df.to_parquet(local_path)
        logger.info(f"💾 Cached data locally: {local_path}")
        
        # Upload to S3 if available
        if self.s3_client:
            try:
                self.s3_client.upload_file(str(local_path), self.bucket_name, s3_key)
                logger.info(f"☁️ Uploaded to S3: s3://{self.bucket_name}/{s3_key}")
            except Exception as e:
                logger.warning(f"⚠️ Could not upload to S3: {e}")
        
        return df
    
    def _download_nyc_data(self, vehicle_type: str, year: int, month: int) -> pd.DataFrame:
        """Download NYC taxi data from official TLC website."""
        base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
        filename = f"{vehicle_type}_tripdata_{year}-{month:02d}.parquet"
        url = f"{base_url}/{filename}"
        
        logger.info(f"🌐 Downloading: {url}")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            df = pd.read_parquet(io.BytesIO(response.content))
            logger.info(f"✅ Downloaded {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to download data: {e}")
            raise
    
    def load_area_info(self, local_path: str = "./area_info.csv") -> pd.DataFrame:
        """
        Load NYC taxi zone area information.
        
        Args:
            local_path: Local path to area_info.csv
            
        Returns:
            DataFrame with taxi zone information
        """
        # Try to load from S3 first
        if self.s3_client:
            try:
                s3_key = "reference/area_info.csv"
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
                df = pd.read_csv(io.BytesIO(response['Body'].read()))
                logger.info(f"✅ Loaded area info from S3: {len(df)} zones")
                return df
            except Exception as e:
                logger.warning(f"⚠️ Could not load area info from S3: {e}")
        
        # Load from local file
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
            logger.info(f"✅ Loaded area info locally: {len(df)} zones")
            return df
        
        logger.error(f"❌ Area info file not found: {local_path}")
        raise FileNotFoundError(f"Area info file not found: {local_path}")
    
    def save_results(self, results: Dict, experiment_id: str, 
                    local_results_dir: str = "./results") -> str:
        """
        Save experiment results to S3 and locally.
        
        Args:
            results: Experiment results dictionary
            experiment_id: Unique experiment identifier
            local_results_dir: Local directory for results
            
        Returns:
            S3 URL of saved results (if uploaded) or local path
        """
        # Create local results directory
        results_dir = Path(local_results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save locally
        local_path = results_dir / f"{experiment_id}.json"
        import json
        with open(local_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Saved results locally: {local_path}")
        
        # Upload to S3 if available
        if self.s3_client:
            try:
                s3_key = f"results/{experiment_id}.json"
                self.s3_client.upload_file(str(local_path), self.bucket_name, s3_key)
                s3_url = f"s3://{self.bucket_name}/{s3_key}"
                logger.info(f"☁️ Uploaded results to S3: {s3_url}")
                return s3_url
            except Exception as e:
                logger.warning(f"⚠️ Could not upload results to S3: {e}")
        
        return str(local_path)
    
    def list_available_datasets(self) -> List[Dict[str, str]]:
        """List all available datasets in S3."""
        if not self.s3_client:
            return []
        
        datasets = []
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix='datasets/')
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith('.parquet'):
                            # Parse dataset info from key
                            parts = key.split('/')
                            if len(parts) >= 4:
                                vehicle_type = parts[1]
                                year_part = parts[2].split('=')[1] if '=' in parts[2] else parts[2]
                                month_part = parts[3].split('=')[1] if '=' in parts[3] else parts[3]
                                
                                datasets.append({
                                    'vehicle_type': vehicle_type,
                                    'year': year_part,
                                    'month': month_part,
                                    's3_key': key,
                                    'size_mb': round(obj['Size'] / (1024 * 1024), 2),
                                    'last_modified': obj['LastModified'].isoformat()
                                })
        
        except Exception as e:
            logger.error(f"❌ Error listing datasets: {e}")
        
        return datasets
    
    def check_bucket_exists(self) -> bool:
        """Check if the S3 bucket exists and is accessible."""
        if not self.s3_client:
            return False
        
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except ClientError:
            return False
    
    def create_bucket_if_not_exists(self) -> bool:
        """Create S3 bucket if it doesn't exist."""
        if not self.s3_client:
            return False
        
        if self.check_bucket_exists():
            return True
        
        try:
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            logger.info(f"✅ Created S3 bucket: {self.bucket_name}")
            return True
        except ClientError as e:
            logger.error(f"❌ Failed to create bucket: {e}")
            return False 