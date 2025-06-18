#!/usr/bin/env python3
"""
S3 Data Manager for Bipartite Matching Optimization Project.
Handles all S3 operations including dataset uploads, result storage, and analysis outputs.
"""

import boto3
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
from botocore.exceptions import ClientError, NoCredentialsError
import time
from aws_config import AWSConfig

class S3DataManager:
    """
    Manages all S3 operations for the bipartite matching project.
    Handles datasets, experiment results, and analysis outputs.
    """
    
    def __init__(self, profile_name: Optional[str] = None, region: str = None):
        """
        Initialize S3 data manager.
        
        Args:
            profile_name: AWS profile name (optional)
            region: AWS region (optional, defaults to AWSConfig.REGION)
        """
        self.region = region or AWSConfig.REGION
        self.bucket_name = AWSConfig.BUCKET_NAME
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize S3 client
        try:
            session = boto3.Session(profile_name=profile_name, region_name=self.region)
            self.s3_client = session.client('s3')
            self.s3_resource = session.resource('s3')
            self.bucket = self.s3_resource.Bucket(self.bucket_name)
            
            # Test connection
            self._test_connection()
            
        except (NoCredentialsError, ClientError) as e:
            self.logger.error(f"Failed to initialize S3 connection: {e}")
            raise
    
    def _test_connection(self):
        """Test S3 connection and bucket access."""
        try:
            # Try to access the bucket
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            self.logger.info(f"✅ Connected to S3 bucket: {self.bucket_name}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket doesn't exist, try to create it
                self.logger.info(f"🔧 Bucket {self.bucket_name} not found, creating...")
                self._create_bucket()
            else:
                raise
    
    def _create_bucket(self):
        """Create the S3 bucket if it doesn't exist."""
        try:
            if self.region == 'us-east-1':
                # us-east-1 doesn't need location constraint
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            
            self.logger.info(f"✅ Created S3 bucket: {self.bucket_name}")
            
            # Set up bucket folders by uploading placeholder files
            self._setup_bucket_structure()
            
        except ClientError as e:
            self.logger.error(f"Failed to create bucket: {e}")
            raise
    
    def _setup_bucket_structure(self):
        """Set up the folder structure in S3."""
        folders = [
            'datasets/',
            'datasets/green/',
            'datasets/yellow/',
            'datasets/fhv/',
            'experiments/',
            'experiments/rideshare/',
            'experiments/crowdsourcing/',
            'experiments/analysis/',
            'experiments/logs/'
        ]
        
        for folder in folders:
            # Create folder by uploading empty file
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=folder + '.gitkeep',
                Body=b'# S3 folder placeholder\n'
            )
        
        self.logger.info("📁 Set up S3 bucket folder structure")
    
    def upload_dataset(self, local_file: Union[str, Path], vehicle_type: str, 
                      year: int, month: int, progress_callback=None) -> str:
        """
        Upload dataset file to S3.
        
        Args:
            local_file: Path to local file
            vehicle_type: Type of vehicle data ('green', 'yellow', etc.)
            year: Year of data
            month: Month of data
            progress_callback: Optional callback for upload progress
            
        Returns:
            S3 key of uploaded file
        """
        local_file = Path(local_file)
        if not local_file.exists():
            raise FileNotFoundError(f"File not found: {local_file}")
        
        # Generate S3 key
        s3_key = AWSConfig.get_s3_dataset_key(vehicle_type, year, month, local_file.name)
        
        self.logger.info(f"📤 Uploading {local_file.name} to s3://{self.bucket_name}/{s3_key}")
        
        try:
            # Upload with progress tracking
            file_size = local_file.stat().st_size
            uploaded = 0
            
            def upload_progress(bytes_transferred):
                nonlocal uploaded
                uploaded += bytes_transferred
                if progress_callback:
                    progress_callback(uploaded, file_size)
                
                # Log progress every 10MB
                if uploaded % (10 * 1024 * 1024) == 0 or uploaded == file_size:
                    percent = (uploaded / file_size) * 100
                    mb_uploaded = uploaded / (1024 * 1024)
                    mb_total = file_size / (1024 * 1024)
                    self.logger.info(f"   📊 Progress: {percent:.1f}% ({mb_uploaded:.1f}/{mb_total:.1f} MB)")
            
            self.s3_client.upload_file(
                str(local_file), 
                self.bucket_name, 
                s3_key,
                Callback=upload_progress
            )
            
            self.logger.info(f"✅ Upload complete: s3://{self.bucket_name}/{s3_key}")
            return s3_key
            
        except ClientError as e:
            self.logger.error(f"Upload failed: {e}")
            raise
    
    def download_dataset(self, vehicle_type: str, year: int, month: int, 
                        filename: str, local_dir: Union[str, Path]) -> Path:
        """
        Download dataset from S3.
        
        Args:
            vehicle_type: Type of vehicle data
            year: Year of data
            month: Month of data
            filename: Name of file to download
            local_dir: Local directory to save file
            
        Returns:
            Path to downloaded file
        """
        s3_key = AWSConfig.get_s3_dataset_key(vehicle_type, year, month, filename)
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_dir / filename
        
        self.logger.info(f"📥 Downloading s3://{self.bucket_name}/{s3_key}")
        
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_file))
            self.logger.info(f"✅ Downloaded to {local_file}")
            return local_file
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"File not found in S3: {s3_key}")
            else:
                self.logger.error(f"Download failed: {e}")
                raise
    
    def upload_experiment_results(self, results_data: Dict[str, Any], 
                                 experiment_type: str, experiment_id: str) -> str:
        """
        Upload experiment results to S3.
        
        Args:
            results_data: Dictionary containing experiment results
            experiment_type: Type of experiment ('rideshare', 'crowdsourcing')
            experiment_id: Unique experiment identifier
            
        Returns:
            S3 key of uploaded results
        """
        # Add metadata
        results_data['upload_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'experiment_id': experiment_id,
            'experiment_type': experiment_type
        }
        
        filename = f"{experiment_id}_results.json"
        s3_key = AWSConfig.get_s3_results_key(experiment_type, filename)
        
        self.logger.info(f"📤 Uploading results: s3://{self.bucket_name}/{s3_key}")
        
        try:
            # Convert to JSON and upload
            json_content = json.dumps(results_data, indent=2, default=str)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_content.encode('utf-8'),
                ContentType='application/json'
            )
            
            self.logger.info(f"✅ Results uploaded: s3://{self.bucket_name}/{s3_key}")
            return s3_key
            
        except ClientError as e:
            self.logger.error(f"Results upload failed: {e}")
            raise
    
    def upload_experiment_logs(self, log_file: Union[str, Path], 
                              experiment_id: str) -> str:
        """
        Upload experiment log file to S3.
        
        Args:
            log_file: Path to log file
            experiment_id: Unique experiment identifier
            
        Returns:
            S3 key of uploaded log
        """
        log_file = Path(log_file)
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
        
        s3_key = AWSConfig.get_s3_logs_key(experiment_id, log_file.name)
        
        self.logger.info(f"📤 Uploading log: s3://{self.bucket_name}/{s3_key}")
        
        try:
            self.s3_client.upload_file(str(log_file), self.bucket_name, s3_key)
            self.logger.info(f"✅ Log uploaded: s3://{self.bucket_name}/{s3_key}")
            return s3_key
            
        except ClientError as e:
            self.logger.error(f"Log upload failed: {e}")
            raise
    
    def upload_analysis(self, analysis_data: Union[Dict, str, Path], 
                       analysis_type: str, filename: str) -> str:
        """
        Upload analysis results to S3.
        
        Args:
            analysis_data: Analysis data (dict, file path, or JSON string)
            analysis_type: Type of analysis ('performance', 'comparison', etc.)
            filename: Name for the uploaded file
            
        Returns:
            S3 key of uploaded analysis
        """
        s3_key = AWSConfig.get_s3_analysis_key(analysis_type, filename)
        
        self.logger.info(f"📤 Uploading analysis: s3://{self.bucket_name}/{s3_key}")
        
        try:
            if isinstance(analysis_data, (str, Path)) and Path(analysis_data).exists():
                # Upload file
                self.s3_client.upload_file(str(analysis_data), self.bucket_name, s3_key)
            elif isinstance(analysis_data, dict):
                # Upload JSON data
                json_content = json.dumps(analysis_data, indent=2, default=str)
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=json_content.encode('utf-8'),
                    ContentType='application/json'
                )
            else:
                # Upload string content
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=str(analysis_data).encode('utf-8'),
                    ContentType='text/plain'
                )
            
            self.logger.info(f"✅ Analysis uploaded: s3://{self.bucket_name}/{s3_key}")
            return s3_key
            
        except ClientError as e:
            self.logger.error(f"Analysis upload failed: {e}")
            raise
    
    def list_datasets(self, vehicle_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available datasets in S3.
        
        Args:
            vehicle_type: Filter by vehicle type (optional)
            
        Returns:
            List of dataset information
        """
        prefix = 'datasets/'
        if vehicle_type:
            prefix = f'datasets/{vehicle_type}/'
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            datasets = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('.csv') or key.endswith('.parquet'):
                    # Parse key to extract metadata
                    parts = key.split('/')
                    if len(parts) >= 4:
                        datasets.append({
                            'key': key,
                            'filename': Path(key).name,
                            'vehicle_type': parts[1] if len(parts) > 1 else None,
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'],
                            'url': f"s3://{self.bucket_name}/{key}"
                        })
            
            return datasets
            
        except ClientError as e:
            self.logger.error(f"Failed to list datasets: {e}")
            raise
    
    def list_experiment_results(self, experiment_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List experiment results in S3.
        
        Args:
            experiment_type: Filter by experiment type (optional)
            
        Returns:
            List of experiment result information
        """
        prefix = 'experiments/'
        if experiment_type:
            prefix = f'experiments/{experiment_type}/'
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            results = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('.json') or key.endswith('.csv'):
                    results.append({
                        'key': key,
                        'filename': Path(key).name,
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'],
                        'url': f"s3://{self.bucket_name}/{key}"
                    })
            
            return results
            
        except ClientError as e:
            self.logger.error(f"Failed to list results: {e}")
            raise
    
    def download_results(self, experiment_type: str, filename: str, 
                        local_dir: Union[str, Path]) -> Path:
        """
        Download experiment results from S3.
        
        Args:
            experiment_type: Type of experiment
            filename: Name of results file
            local_dir: Local directory to save file
            
        Returns:
            Path to downloaded file
        """
        s3_key = AWSConfig.get_s3_results_key(experiment_type, filename)
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_dir / filename
        
        self.logger.info(f"📥 Downloading results: s3://{self.bucket_name}/{s3_key}")
        
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_file))
            self.logger.info(f"✅ Downloaded to {local_file}")
            return local_file
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"Results not found in S3: {s3_key}")
            else:
                self.logger.error(f"Download failed: {e}")
                raise
    
    def get_signed_url(self, s3_key: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for S3 object access.
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds
            
        Returns:
            Presigned URL
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            self.logger.error(f"Failed to generate presigned URL: {e}")
            raise
    
    def cleanup_old_results(self, days_old: int = 30):
        """
        Clean up old experiment results and logs.
        
        Args:
            days_old: Remove files older than this many days
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        prefixes = ['experiments/results/', 'experiments/logs/']
        
        for prefix in prefixes:
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix
                )
                
                objects_to_delete = []
                for obj in response.get('Contents', []):
                    if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                        objects_to_delete.append({'Key': obj['Key']})
                
                if objects_to_delete:
                    self.s3_client.delete_objects(
                        Bucket=self.bucket_name,
                        Delete={'Objects': objects_to_delete}
                    )
                    self.logger.info(f"🗑️  Cleaned up {len(objects_to_delete)} old files from {prefix}")
                
            except ClientError as e:
                self.logger.error(f"Cleanup failed for {prefix}: {e}")

def main():
    """Command line interface for S3 data manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description='S3 Data Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List datasets
    list_cmd = subparsers.add_parser('list-datasets', help='List available datasets')
    list_cmd.add_argument('--vehicle-type', help='Filter by vehicle type')
    
    # Upload dataset
    upload_cmd = subparsers.add_parser('upload-dataset', help='Upload dataset')
    upload_cmd.add_argument('file', help='Local file to upload')
    upload_cmd.add_argument('vehicle_type', help='Vehicle type')
    upload_cmd.add_argument('year', type=int, help='Year')
    upload_cmd.add_argument('month', type=int, help='Month')
    
    # List results
    results_cmd = subparsers.add_parser('list-results', help='List experiment results')
    results_cmd.add_argument('--experiment-type', help='Filter by experiment type')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize S3 manager
    if not AWSConfig.validate_config():
        return
    
    s3_manager = S3DataManager()
    
    if args.command == 'list-datasets':
        datasets = s3_manager.list_datasets(args.vehicle_type)
        print(f"📊 Found {len(datasets)} datasets:")
        for dataset in datasets:
            print(f"  {dataset['url']} ({dataset['size']} bytes)")
    
    elif args.command == 'upload-dataset':
        s3_key = s3_manager.upload_dataset(
            args.file, args.vehicle_type, args.year, args.month
        )
        print(f"✅ Uploaded: s3://{AWSConfig.BUCKET_NAME}/{s3_key}")
    
    elif args.command == 'list-results':
        results = s3_manager.list_experiment_results(args.experiment_type)
        print(f"📊 Found {len(results)} result files:")
        for result in results:
            print(f"  {result['url']} ({result['size']} bytes)")

if __name__ == "__main__":
    main() 