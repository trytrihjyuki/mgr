#!/usr/bin/env python3
"""
AWS Deployment and Setup Script for Bipartite Matching Experiments.
Handles data upload, resource setup, and experiment management.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import time

from aws_config import AWSConfig
from aws_s3_manager import S3DataManager

class AWSDeploymentManager:
    """
    Manages AWS deployment for bipartite matching experiments.
    """
    
    def __init__(self):
        """Initialize deployment manager."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Validate AWS configuration
        if not AWSConfig.validate_config():
            raise ValueError("AWS configuration validation failed")
        
        self.s3_manager = S3DataManager()
        self.logger.info("✅ AWS deployment manager initialized")
    
    def upload_datasets(self, data_dir: str = "data", vehicle_types: List[str] = None) -> Dict[str, Any]:
        """
        Upload all datasets from local directory to S3.
        
        Args:
            data_dir: Local directory containing datasets
            vehicle_types: List of vehicle types to upload (default: all)
            
        Returns:
            Dictionary with upload results
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            # Try Rideshare_experiment/data as fallback
            data_path = Path("Rideshare_experiment/data")
            if not data_path.exists():
                raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        self.logger.info(f"📤 Starting dataset upload from {data_path}")
        
        vehicle_types = vehicle_types or ['green', 'yellow', 'fhv', 'fhvhv']
        upload_results = {
            'successful': [],
            'failed': [],
            'total_size': 0
        }
        
        # Find all data files
        data_files = []
        for vehicle_type in vehicle_types:
            pattern = f"{vehicle_type}_tripdata_*.csv"
            found_files = list(data_path.glob(pattern))
            data_files.extend([(f, vehicle_type) for f in found_files])
            
            # Also check for parquet files
            pattern = f"{vehicle_type}_tripdata_*.parquet"
            found_files = list(data_path.glob(pattern))
            data_files.extend([(f, vehicle_type) for f in found_files])
        
        self.logger.info(f"📊 Found {len(data_files)} data files to upload")
        
        for file_path, vehicle_type in data_files:
            try:
                # Parse year and month from filename
                # Expected format: vehicle_tripdata_YYYY-MM.csv
                filename_parts = file_path.stem.split('_')
                if len(filename_parts) >= 3:
                    date_part = filename_parts[2]  # YYYY-MM
                    if '-' in date_part:
                        year_str, month_str = date_part.split('-')
                        year = int(year_str)
                        month = int(month_str)
                        
                        self.logger.info(f"📤 Uploading {file_path.name}...")
                        
                        s3_key = self.s3_manager.upload_dataset(
                            file_path, vehicle_type, year, month
                        )
                        
                        file_size = file_path.stat().st_size
                        upload_results['successful'].append({
                            'file': file_path.name,
                            'vehicle_type': vehicle_type,
                            'year': year,
                            'month': month,
                            's3_key': s3_key,
                            'size': file_size
                        })
                        upload_results['total_size'] += file_size
                        
            except Exception as e:
                self.logger.error(f"❌ Failed to upload {file_path.name}: {e}")
                upload_results['failed'].append({
                    'file': file_path.name,
                    'error': str(e)
                })
        
        # Summary
        successful_count = len(upload_results['successful'])
        failed_count = len(upload_results['failed'])
        total_size_mb = upload_results['total_size'] / (1024 * 1024)
        
        self.logger.info(f"📊 Upload Summary:")
        self.logger.info(f"   Successful: {successful_count}")
        self.logger.info(f"   Failed: {failed_count}")
        self.logger.info(f"   Total Size: {total_size_mb:.1f} MB")
        
        return upload_results
    
    def upload_crowdsourcing_data(self, data_dir: str = "Crowd_sourcing_experiment/work") -> bool:
        """
        Upload crowdsourcing datasets to S3.
        
        Args:
            data_dir: Directory containing crowdsourcing data
            
        Returns:
            True if successful, False otherwise
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            self.logger.error(f"Crowdsourcing data directory not found: {data_dir}")
            return False
        
        # Look for the main crowdsourcing data file
        data_file = data_path / "trec-rf10-data.csv"
        if not data_file.exists():
            self.logger.error(f"Crowdsourcing data file not found: {data_file}")
            return False
        
        try:
            s3_key = f"datasets/crowdsourcing/2010/trec-rf10-data.csv"
            
            self.logger.info(f"📤 Uploading crowdsourcing data...")
            self.s3_manager.s3_client.upload_file(
                str(data_file),
                self.s3_manager.bucket_name,
                s3_key
            )
            
            self.logger.info(f"✅ Crowdsourcing data uploaded: s3://{self.s3_manager.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to upload crowdsourcing data: {e}")
            return False
    
    def list_s3_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all data currently in S3.
        
        Returns:
            Dictionary with datasets organized by type
        """
        datasets = self.s3_manager.list_datasets()
        results = self.s3_manager.list_experiment_results()
        
        organized_data = {
            'rideshare_datasets': [d for d in datasets if d['vehicle_type'] in ['green', 'yellow', 'fhv', 'fhvhv']],
            'crowdsourcing_datasets': [d for d in datasets if 'crowdsourcing' in d['key']],
            'experiment_results': results
        }
        
        self.logger.info(f"📊 S3 Data Summary:")
        self.logger.info(f"   Rideshare datasets: {len(organized_data['rideshare_datasets'])}")
        self.logger.info(f"   Crowdsourcing datasets: {len(organized_data['crowdsourcing_datasets'])}")
        self.logger.info(f"   Experiment results: {len(organized_data['experiment_results'])}")
        
        return organized_data
    
    def run_ec2_experiment(self, experiment_type: str, **kwargs) -> str:
        """
        Run experiment on EC2 instance (simplified implementation).
        
        Args:
            experiment_type: Type of experiment to run
            **kwargs: Experiment parameters
            
        Returns:
            Experiment ID
        """
        self.logger.info(f"🚀 Starting {experiment_type} experiment on EC2...")
        
        # This is a simplified implementation
        # In a real deployment, this would:
        # 1. Launch EC2 instance with the Docker image
        # 2. Run the experiment
        # 3. Terminate the instance
        
        # For now, we'll simulate this by running locally but using S3 for storage
        from aws_experiment_runner import AWSExperimentRunner
        
        runner = AWSExperimentRunner(experiment_type, use_s3=True)
        
        if experiment_type == 'rideshare':
            results = runner.run_rideshare_experiment(**kwargs)
        else:
            results = runner.run_crowdsourcing_experiment(**kwargs)
        
        # Save results
        runner.save_results(results)
        runner.upload_logs()
        runner.cleanup()
        
        self.logger.info(f"✅ Experiment completed: {runner.experiment_id}")
        return runner.experiment_id
    
    def cleanup_old_data(self, days_old: int = 30):
        """
        Clean up old experiment results and logs.
        
        Args:
            days_old: Remove files older than this many days
        """
        self.logger.info(f"🧹 Cleaning up data older than {days_old} days...")
        self.s3_manager.cleanup_old_results(days_old)
        self.logger.info("✅ Cleanup completed")
    
    def create_analysis_dashboard(self) -> str:
        """
        Create a simple analysis dashboard of all results.
        
        Returns:
            S3 key of the dashboard file
        """
        self.logger.info("📊 Creating analysis dashboard...")
        
        # Get all experiment results
        results = self.s3_manager.list_experiment_results()
        
        dashboard_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_experiments': len(results),
            'experiments': results,
            'summary': {
                'rideshare_experiments': len([r for r in results if 'rideshare' in r['key']]),
                'crowdsourcing_experiments': len([r for r in results if 'crowdsourcing' in r['key']]),
                'total_data_size': sum(r['size'] for r in results)
            }
        }
        
        dashboard_filename = f"experiment_dashboard_{int(time.time())}.json"
        s3_key = self.s3_manager.upload_analysis(
            dashboard_data, 'dashboard', dashboard_filename
        )
        
        self.logger.info(f"✅ Dashboard created: s3://{self.s3_manager.bucket_name}/{s3_key}")
        return s3_key

def main():
    """Command line interface for AWS deployment manager."""
    parser = argparse.ArgumentParser(description='AWS Deployment Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Upload datasets
    upload_cmd = subparsers.add_parser('upload-datasets', help='Upload datasets to S3')
    upload_cmd.add_argument('--data-dir', default='data', help='Local data directory')
    upload_cmd.add_argument('--vehicle-types', nargs='+', 
                           choices=['green', 'yellow', 'fhv', 'fhvhv'],
                           help='Vehicle types to upload')
    
    # Upload crowdsourcing data
    crowd_cmd = subparsers.add_parser('upload-crowdsourcing', help='Upload crowdsourcing data')
    crowd_cmd.add_argument('--data-dir', default='Crowd_sourcing_experiment/work',
                          help='Crowdsourcing data directory')
    
    # List S3 data
    subparsers.add_parser('list-data', help='List data in S3')
    
    # Run experiment
    run_cmd = subparsers.add_parser('run-experiment', help='Run experiment on AWS')
    run_cmd.add_argument('experiment_type', choices=['rideshare', 'crowdsourcing'])
    run_cmd.add_argument('--vehicle-type', default='green')
    run_cmd.add_argument('--year', type=int, default=2019)
    run_cmd.add_argument('--month', type=int, default=3)
    run_cmd.add_argument('--place', default='Manhattan')
    run_cmd.add_argument('--phi', type=float, default=0.8)
    run_cmd.add_argument('--psi', type=float, default=0.6)
    
    # Cleanup
    cleanup_cmd = subparsers.add_parser('cleanup', help='Clean up old data')
    cleanup_cmd.add_argument('--days-old', type=int, default=30, help='Days old threshold')
    
    # Dashboard
    subparsers.add_parser('create-dashboard', help='Create analysis dashboard')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        manager = AWSDeploymentManager()
        
        if args.command == 'upload-datasets':
            manager.upload_datasets(args.data_dir, args.vehicle_types)
        
        elif args.command == 'upload-crowdsourcing':
            manager.upload_crowdsourcing_data(args.data_dir)
        
        elif args.command == 'list-data':
            data = manager.list_s3_data()
            print("\n📊 S3 Data Summary:")
            for category, items in data.items():
                print(f"  {category}: {len(items)} items")
                for item in items[:3]:  # Show first 3 items
                    print(f"    - {item.get('filename', 'N/A')} ({item.get('size', 0)} bytes)")
                if len(items) > 3:
                    print(f"    ... and {len(items) - 3} more")
        
        elif args.command == 'run-experiment':
            kwargs = {}
            if args.experiment_type == 'rideshare':
                kwargs = {
                    'vehicle_type': args.vehicle_type,
                    'year': args.year,
                    'month': args.month,
                    'place': args.place
                }
            else:
                kwargs = {
                    'phi': args.phi,
                    'psi': args.psi
                }
            
            experiment_id = manager.run_ec2_experiment(args.experiment_type, **kwargs)
            print(f"✅ Experiment completed: {experiment_id}")
        
        elif args.command == 'cleanup':
            manager.cleanup_old_data(args.days_old)
        
        elif args.command == 'create-dashboard':
            dashboard_key = manager.create_analysis_dashboard()
            print(f"✅ Dashboard created: {dashboard_key}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ Operation failed: {e}")
        logging.exception("Detailed error information:")

if __name__ == "__main__":
    main() 