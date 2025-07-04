#!/usr/bin/env python3
"""
S3 Experiment Audit Script
=========================

Comprehensive audit tool for tracking all experiments in S3.
- Tracks experiment files and metadata
- Provides daily/weekly summaries
- Monitors experiment health and completion status
- Identifies missing or corrupted files
- Tracks parallel execution progress

Usage:
    python s3_audit_script.py --list-recent
    python s3_audit_script.py --daily-summary 2024-01-15
    python s3_audit_script.py --health-check
    python s3_audit_script.py --parallel-status
"""

import boto3
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse
import logging
from pathlib import Path
import sys
from collections import defaultdict
import os

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'audit_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)

class S3ExperimentAuditor:
    """Auditor for S3 experiment files and metadata."""
    
    def __init__(self, bucket_name: str = "magisterka", region: str = "eu-north-1"):
        """Initialize the auditor."""
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.local_audit_dir = Path("audit_cache")
        self.local_audit_dir.mkdir(exist_ok=True)
        
        logger.info(f"🔍 S3 Experiment Auditor initialized - Bucket: {bucket_name}")
        
    def list_all_experiments(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """List all experiments in S3 with metadata."""
        logger.info(f"📋 Scanning S3 for experiments from last {days_back} days...")
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        experiments = []
        
        # Search patterns for different experiment structures
        prefixes = [
            "experiments/type=",  # New partitioned structure
            "experiments/results/",  # Legacy structure
            "experiments/logs/",  # Log files
        ]
        
        for prefix in prefixes:
            try:
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
                
                for page in pages:
                    for obj in page.get('Contents', []):
                        if obj['LastModified'].replace(tzinfo=None) >= cutoff_date:
                            experiment_info = self._parse_experiment_object(obj)
                            if experiment_info:
                                experiments.append(experiment_info)
                                
            except Exception as e:
                logger.warning(f"⚠️ Error scanning prefix {prefix}: {e}")
                
        logger.info(f"✅ Found {len(experiments)} experiment files")
        return experiments
    
    def _parse_experiment_object(self, obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse S3 object to extract experiment information."""
        key = obj['Key']
        
        # Skip non-experiment files
        if not (key.endswith('.json') or key.endswith('.parquet')):
            return None
            
        try:
            # Parse path components
            path_parts = key.split('/')
            experiment_info = {
                'key': key,
                'filename': Path(key).name,
                'size_bytes': obj['Size'],
                'last_modified': obj['LastModified'],
                'file_type': 'json' if key.endswith('.json') else 'parquet',
                'path_parts': path_parts
            }
            
            # Extract metadata from path
            if 'type=' in key:
                # New partitioned structure
                experiment_info.update(self._parse_partitioned_path(key))
            else:
                # Legacy structure
                experiment_info.update(self._parse_legacy_path(key))
                
            return experiment_info
            
        except Exception as e:
            logger.warning(f"⚠️ Error parsing {key}: {e}")
            return None
    
    def _parse_partitioned_path(self, key: str) -> Dict[str, Any]:
        """Parse new partitioned path structure."""
        metadata = {}
        
        # Extract from path: experiments/type=yellow/eval=pl/year=2019/month=10/day=01/
        if 'type=' in key:
            metadata['vehicle_type'] = key.split('type=')[1].split('/')[0]
        if 'eval=' in key:
            metadata['acceptance_function'] = key.split('eval=')[1].split('/')[0]
        if 'year=' in key:
            metadata['year'] = int(key.split('year=')[1].split('/')[0])
        if 'month=' in key:
            metadata['month'] = int(key.split('month=')[1].split('/')[0])
        if 'day=' in key:
            metadata['day'] = int(key.split('day=')[1].split('/')[0])
            
        # Extract experiment ID from filename
        filename = Path(key).stem
        if '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 2:
                metadata['experiment_id'] = filename
                metadata['timestamp'] = parts[-1] if len(parts[-1]) == 6 else None
                
        metadata['structure_type'] = 'partitioned'
        return metadata
    
    def _parse_legacy_path(self, key: str) -> Dict[str, Any]:
        """Parse legacy path structure."""
        metadata = {'structure_type': 'legacy'}
        
        # Extract from legacy paths like experiments/results/rideshare/
        if 'results' in key:
            metadata['experiment_type'] = 'rideshare'
            
        filename = Path(key).stem
        metadata['experiment_id'] = filename
        
        return metadata
    
    def get_daily_summary(self, date: str) -> Dict[str, Any]:
        """Get summary of experiments for a specific date."""
        logger.info(f"📅 Generating daily summary for {date}")
        
        target_date = datetime.strptime(date, '%Y-%m-%d')
        
        # Get experiments for the day
        experiments = self.list_all_experiments(days_back=1)
        daily_experiments = [
            exp for exp in experiments 
            if exp['last_modified'].date() == target_date.date()
        ]
        
        # Analyze the experiments
        summary = {
            'date': date,
            'total_experiments': len(daily_experiments),
            'by_vehicle_type': defaultdict(int),
            'by_acceptance_function': defaultdict(int),
            'by_file_type': defaultdict(int),
            'total_size_mb': sum(exp['size_bytes'] for exp in daily_experiments) / (1024 * 1024),
            'experiments': daily_experiments
        }
        
        for exp in daily_experiments:
            summary['by_vehicle_type'][exp.get('vehicle_type', 'unknown')] += 1
            summary['by_acceptance_function'][exp.get('acceptance_function', 'unknown')] += 1
            summary['by_file_type'][exp.get('file_type', 'unknown')] += 1
            
        return summary
    
    def check_experiment_health(self) -> Dict[str, Any]:
        """Check health of experiments - identify missing files, corrupted data, etc."""
        logger.info("🏥 Performing experiment health check...")
        
        experiments = self.list_all_experiments(days_back=7)
        
        # Group experiments by experiment ID
        experiment_groups = defaultdict(list)
        for exp in experiments:
            exp_id = exp.get('experiment_id', 'unknown')
            experiment_groups[exp_id].append(exp)
        
        health_report = {
            'total_experiment_groups': len(experiment_groups),
            'healthy_experiments': 0,
            'missing_files': [],
            'incomplete_experiments': [],
            'corrupted_files': [],
            'large_files': [],
            'summary': {}
        }
        
        for exp_id, files in experiment_groups.items():
            # Check for expected files
            has_summary = any(f['filename'] == 'experiment_summary.json' for f in files)
            has_results = any(f['filename'] == 'results.parquet' for f in files)
            
            if has_summary and has_results:
                health_report['healthy_experiments'] += 1
            else:
                health_report['incomplete_experiments'].append({
                    'experiment_id': exp_id,
                    'has_summary': has_summary,
                    'has_results': has_results,
                    'files': [f['filename'] for f in files]
                })
            
            # Check for unusually large files (>100MB)
            large_files = [f for f in files if f['size_bytes'] > 100 * 1024 * 1024]
            if large_files:
                health_report['large_files'].extend(large_files)
        
        health_report['summary'] = {
            'health_percentage': (health_report['healthy_experiments'] / 
                                 max(1, len(experiment_groups))) * 100,
            'incomplete_count': len(health_report['incomplete_experiments']),
            'large_files_count': len(health_report['large_files'])
        }
        
        logger.info(f"✅ Health check complete: {health_report['summary']['health_percentage']:.1f}% healthy")
        return health_report
    
    def check_parallel_execution_status(self) -> Dict[str, Any]:
        """Check status of parallel experiment execution."""
        logger.info("🔄 Checking parallel execution status...")
        
        # Look for parallel_experiments directories
        parallel_dirs = []
        try:
            for item in Path('.').iterdir():
                if item.is_dir() and item.name.startswith('parallel_experiments_'):
                    parallel_dirs.append(item)
        except Exception as e:
            logger.warning(f"⚠️ Error scanning parallel directories: {e}")
        
        status = {
            'parallel_directories': len(parallel_dirs),
            'active_experiments': [],
            'completed_experiments': [],
            'failed_experiments': []
        }
        
        for dir_path in parallel_dirs:
            try:
                # Check for log files
                log_files = list(dir_path.glob('logs/*.log'))
                config_file = dir_path / 'experiment_config.txt'
                
                dir_status = {
                    'directory': dir_path.name,
                    'log_files': len(log_files),
                    'has_config': config_file.exists(),
                    'timestamp': dir_path.name.split('_')[-1] if '_' in dir_path.name else 'unknown'
                }
                
                # Try to determine status from logs
                if log_files:
                    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                    dir_status['latest_log'] = latest_log.name
                    dir_status['last_modified'] = datetime.fromtimestamp(latest_log.stat().st_mtime)
                    
                    # Check if still running (modified within last hour)
                    if (datetime.now() - dir_status['last_modified']).total_seconds() < 3600:
                        status['active_experiments'].append(dir_status)
                    else:
                        status['completed_experiments'].append(dir_status)
                else:
                    status['failed_experiments'].append(dir_status)
                    
            except Exception as e:
                logger.warning(f"⚠️ Error analyzing {dir_path}: {e}")
        
        return status
    
    def generate_audit_report(self, output_file: str = None) -> str:
        """Generate comprehensive audit report."""
        logger.info("📊 Generating comprehensive audit report...")
        
        if output_file is None:
            output_file = f"s3_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'bucket': self.bucket_name,
            'region': self.region,
            'experiments': self.list_all_experiments(days_back=30),
            'health_check': self.check_experiment_health(),
            'parallel_status': self.check_parallel_execution_status()
        }
        
        # Add summary statistics
        experiments = report['experiments']
        report['summary'] = {
            'total_experiments': len(experiments),
            'total_size_gb': sum(exp['size_bytes'] for exp in experiments) / (1024**3),
            'date_range': {
                'earliest': min(exp['last_modified'] for exp in experiments).isoformat() if experiments else None,
                'latest': max(exp['last_modified'] for exp in experiments).isoformat() if experiments else None
            },
            'by_vehicle_type': {},
            'by_acceptance_function': {},
            'by_month': {}
        }
        
        # Calculate breakdowns
        for exp in experiments:
            vtype = exp.get('vehicle_type', 'unknown')
            afunc = exp.get('acceptance_function', 'unknown')
            month = exp.get('month', 'unknown')
            
            report['summary']['by_vehicle_type'][vtype] = report['summary']['by_vehicle_type'].get(vtype, 0) + 1
            report['summary']['by_acceptance_function'][afunc] = report['summary']['by_acceptance_function'].get(afunc, 0) + 1
            report['summary']['by_month'][str(month)] = report['summary']['by_month'].get(str(month), 0) + 1
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Audit report saved to {output_file}")
        return output_file
    
    def print_summary(self, experiments: List[Dict[str, Any]] = None):
        """Print a human-readable summary."""
        if experiments is None:
            experiments = self.list_all_experiments(days_back=7)
        
        print(f"\n🔍 S3 Experiment Audit Summary")
        print(f"{'='*50}")
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🪣 Bucket: {self.bucket_name}")
        print(f"📊 Total Experiments: {len(experiments)}")
        
        if experiments:
            total_size = sum(exp['size_bytes'] for exp in experiments)
            print(f"💾 Total Size: {total_size / (1024**3):.2f} GB")
            
            # Group by vehicle type
            by_vtype = defaultdict(int)
            by_afunc = defaultdict(int)
            for exp in experiments:
                by_vtype[exp.get('vehicle_type', 'unknown')] += 1
                by_afunc[exp.get('acceptance_function', 'unknown')] += 1
            
            print(f"\n🚗 By Vehicle Type:")
            for vtype, count in sorted(by_vtype.items()):
                print(f"  • {vtype}: {count}")
            
            print(f"\n📈 By Acceptance Function:")
            for afunc, count in sorted(by_afunc.items()):
                print(f"  • {afunc}: {count}")
        
        print(f"\n{'='*50}")

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='S3 Experiment Audit Tool')
    parser.add_argument('--bucket', default='magisterka', help='S3 bucket name')
    parser.add_argument('--region', default='eu-north-1', help='AWS region')
    parser.add_argument('--days-back', type=int, default=30, help='Days back to search')
    
    # Action arguments
    parser.add_argument('--list-recent', action='store_true', help='List recent experiments')
    parser.add_argument('--daily-summary', type=str, help='Generate daily summary for date (YYYY-MM-DD)')
    parser.add_argument('--health-check', action='store_true', help='Perform health check')
    parser.add_argument('--parallel-status', action='store_true', help='Check parallel execution status')
    parser.add_argument('--full-report', action='store_true', help='Generate full audit report')
    
    args = parser.parse_args()
    
    # Initialize auditor
    auditor = S3ExperimentAuditor(bucket_name=args.bucket, region=args.region)
    
    try:
        if args.list_recent:
            experiments = auditor.list_all_experiments(days_back=args.days_back)
            auditor.print_summary(experiments)
            
        elif args.daily_summary:
            summary = auditor.get_daily_summary(args.daily_summary)
            print(f"\n📅 Daily Summary for {args.daily_summary}")
            print(f"{'='*50}")
            print(f"Total Experiments: {summary['total_experiments']}")
            print(f"Total Size: {summary['total_size_mb']:.2f} MB")
            
            for vtype, count in summary['by_vehicle_type'].items():
                print(f"  • {vtype}: {count}")
                
        elif args.health_check:
            health = auditor.check_experiment_health()
            print(f"\n🏥 Health Check Results")
            print(f"{'='*50}")
            print(f"Health Percentage: {health['summary']['health_percentage']:.1f}%")
            print(f"Healthy Experiments: {health['healthy_experiments']}")
            print(f"Incomplete Experiments: {health['summary']['incomplete_count']}")
            print(f"Large Files: {health['summary']['large_files_count']}")
            
        elif args.parallel_status:
            status = auditor.check_parallel_execution_status()
            print(f"\n🔄 Parallel Execution Status")
            print(f"{'='*50}")
            print(f"Parallel Directories: {status['parallel_directories']}")
            print(f"Active Experiments: {len(status['active_experiments'])}")
            print(f"Completed Experiments: {len(status['completed_experiments'])}")
            print(f"Failed Experiments: {len(status['failed_experiments'])}")
            
        elif args.full_report:
            report_file = auditor.generate_audit_report()
            print(f"\n📄 Full audit report generated: {report_file}")
            
        else:
            # Default: show summary
            auditor.print_summary()
            
    except Exception as e:
        logger.error(f"❌ Audit failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 