#!/usr/bin/env python3
"""
Improved Parallel Experiment Runner
===================================

Enhanced version of the parallel experiment runner with:
- Better tracking of parallel lambda executions
- Daily save logic with check-if-saved-then-proceed
- Comprehensive timestamps in all logs
- Status monitoring and health checks
- Reduced spam for broken lambda functions
- Progress tracking and recovery

Usage:
    python improved_parallel_runner.py --config experiments.json
    python improved_parallel_runner.py --resume experiment_id
    python improved_parallel_runner.py --status
"""

import os
import json
import time
import logging
import argparse
import boto3
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import threading
import signal
import sys
from dataclasses import dataclass, asdict
from collections import defaultdict
import pickle
import hashlib

# Configure logging with detailed timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'parallel_runner_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentStatus:
    """Track experiment execution status."""
    experiment_id: str
    scenario_id: str
    status: str  # 'pending', 'running', 'completed', 'failed', 'skipped'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    lambda_execution_id: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    s3_location: Optional[str] = None
    
    def duration(self) -> float:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

@dataclass
class DailyExperimentSummary:
    """Daily experiment summary for tracking."""
    date: str
    experiment_id: str
    total_scenarios: int
    completed_scenarios: int
    failed_scenarios: int
    skipped_scenarios: int
    total_execution_time: float
    s3_paths: List[str]
    saved_to_s3: bool = False
    
class ImprovedParallelRunner:
    """Enhanced parallel experiment runner with comprehensive tracking."""
    
    def __init__(self, config_file: str = None, resume_experiment: str = None):
        """Initialize the improved runner."""
        self.config = self._load_config(config_file)
        self.bucket_name = self.config.get('aws', {}).get('bucket', 'magisterka')
        self.region = self.config.get('aws', {}).get('region', 'eu-north-1')
        
        # Initialize AWS clients
        self.s3_client = boto3.client('s3', region_name=self.region)
        self.lambda_client = boto3.client('lambda', region_name=self.region)
        
        # Tracking
        self.experiment_id = resume_experiment or self._generate_experiment_id()
        self.status_tracker: Dict[str, ExperimentStatus] = {}
        self.daily_summaries: Dict[str, DailyExperimentSummary] = {}
        
        # Execution control
        self.max_concurrent_lambdas = self.config.get('execution', {}).get('max_concurrent', 3)
        self.lambda_timeout = self.config.get('execution', {}).get('lambda_timeout', 900)
        self.max_retries = self.config.get('execution', {}).get('max_retries', 3)
        self.backoff_multiplier = self.config.get('execution', {}).get('backoff_multiplier', 2.0)
        
        # Status file for persistence
        self.status_file = Path(f"experiment_status_{self.experiment_id}.json")
        self.daily_cache_dir = Path(f"daily_cache_{self.experiment_id}")
        self.daily_cache_dir.mkdir(exist_ok=True)
        
        # Load existing status if resuming
        if resume_experiment:
            self._load_status()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.shutdown_requested = False
        self.lock = threading.Lock()
        
        logger.info(f"🚀 Improved Parallel Runner initialized - Experiment: {self.experiment_id}")
        logger.info(f"📊 Config: {self.max_concurrent_lambdas} concurrent lambdas, {self.lambda_timeout}s timeout")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'aws': {
                'bucket': 'magisterka',
                'region': 'eu-north-1'
            },
            'execution': {
                'max_concurrent': 3,
                'lambda_timeout': 900,
                'max_retries': 3,
                'backoff_multiplier': 2.0
            },
            'experiments': {
                'daily_save_enabled': True,
                'health_check_interval': 300,  # 5 minutes
                'progress_report_interval': 60  # 1 minute
            }
        }
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"parallel_exp_{timestamp}"
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.warning(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        self._save_status()
        
    def _save_status(self):
        """Save current status to file."""
        status_data = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'status_tracker': {k: asdict(v) for k, v in self.status_tracker.items()},
            'daily_summaries': {k: asdict(v) for k, v in self.daily_summaries.items()}
        }
        
        with open(self.status_file, 'w') as f:
            json.dump(status_data, f, indent=2, default=str)
        
        logger.info(f"💾 Status saved to {self.status_file}")
    
    def _load_status(self):
        """Load status from file for resuming."""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    status_data = json.load(f)
                
                # Restore status tracker
                for scenario_id, status_dict in status_data.get('status_tracker', {}).items():
                    status = ExperimentStatus(**status_dict)
                    # Convert datetime strings back to datetime objects
                    if status.start_time:
                        status.start_time = datetime.fromisoformat(status.start_time)
                    if status.end_time:
                        status.end_time = datetime.fromisoformat(status.end_time)
                    self.status_tracker[scenario_id] = status
                
                # Restore daily summaries
                for date, summary_dict in status_data.get('daily_summaries', {}).items():
                    self.daily_summaries[date] = DailyExperimentSummary(**summary_dict)
                
                logger.info(f"📂 Loaded status with {len(self.status_tracker)} scenarios")
                
            except Exception as e:
                logger.error(f"❌ Failed to load status: {e}")
    
    def check_daily_save_status(self, date: str) -> bool:
        """Check if daily results are already saved to S3."""
        cache_file = self.daily_cache_dir / f"daily_save_{date}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    save_info = json.load(f)
                    if save_info.get('saved_to_s3', False):
                        logger.info(f"📅 Daily save for {date} already completed - skipping")
                        return True
            except Exception as e:
                logger.warning(f"⚠️ Error checking daily save status: {e}")
        
        return False
    
    def save_daily_results(self, date: str, scenarios: List[str]) -> bool:
        """Save daily results to S3 and mark as completed."""
        logger.info(f"📅 Saving daily results for {date}...")
        
        # Check if already saved
        if self.check_daily_save_status(date):
            return True
        
        # Collect completed scenarios for this date
        completed_scenarios = []
        failed_scenarios = []
        
        for scenario_id in scenarios:
            status = self.status_tracker.get(scenario_id)
            if status:
                if status.status == 'completed':
                    completed_scenarios.append(status)
                elif status.status == 'failed':
                    failed_scenarios.append(status)
        
        if not completed_scenarios:
            logger.warning(f"⚠️ No completed scenarios found for {date}")
            return False
        
        # Create daily summary
        summary = DailyExperimentSummary(
            date=date,
            experiment_id=self.experiment_id,
            total_scenarios=len(scenarios),
            completed_scenarios=len(completed_scenarios),
            failed_scenarios=len(failed_scenarios),
            skipped_scenarios=len(scenarios) - len(completed_scenarios) - len(failed_scenarios),
            total_execution_time=sum(s.duration() for s in completed_scenarios),
            s3_paths=[s.s3_location for s in completed_scenarios if s.s3_location]
        )
        
        # Save to S3
        try:
            s3_key = f"daily_summaries/{self.experiment_id}/{date}_summary.json"
            summary_data = asdict(summary)
            summary_data['generated_at'] = datetime.now().isoformat()
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json.dumps(summary_data, indent=2, default=str),
                ContentType='application/json'
            )
            
            summary.saved_to_s3 = True
            self.daily_summaries[date] = summary
            
            # Mark as saved locally
            cache_file = self.daily_cache_dir / f"daily_save_{date}.json"
            with open(cache_file, 'w') as f:
                json.dump({'saved_to_s3': True, 's3_key': s3_key}, f)
            
            logger.info(f"✅ Daily results saved for {date}: s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save daily results for {date}: {e}")
            return False
    
    def execute_lambda_with_tracking(self, scenario_params: Dict[str, Any]) -> ExperimentStatus:
        """Execute lambda with comprehensive tracking."""
        scenario_id = scenario_params.get('scenario_id', 'unknown')
        
        # Check if already completed
        if scenario_id in self.status_tracker:
            existing_status = self.status_tracker[scenario_id]
            if existing_status.status == 'completed':
                logger.info(f"⏭️ Skipping already completed scenario: {scenario_id}")
                return existing_status
        
        # Initialize status
        status = ExperimentStatus(
            experiment_id=self.experiment_id,
            scenario_id=scenario_id,
            status='pending',
            start_time=datetime.now()
        )
        
        with self.lock:
            self.status_tracker[scenario_id] = status
        
        logger.info(f"🚀 Starting lambda execution for {scenario_id}")
        
        for attempt in range(self.max_retries):
            if self.shutdown_requested:
                status.status = 'skipped'
                status.end_time = datetime.now()
                status.error_message = 'Shutdown requested'
                break
            
            try:
                status.status = 'running'
                status.retry_count = attempt + 1
                
                # Generate unique execution ID
                execution_id = f"{scenario_id}_{attempt}_{int(time.time())}"
                status.lambda_execution_id = execution_id
                
                # Add execution ID to lambda payload
                lambda_payload = scenario_params.copy()
                lambda_payload['execution_id'] = execution_id
                lambda_payload['retry_attempt'] = attempt + 1
                
                logger.info(f"📤 Lambda invocation {execution_id} (attempt {attempt + 1}/{self.max_retries})")
                
                # Invoke lambda
                response = self.lambda_client.invoke(
                    FunctionName='rideshare-pricing-benchmark',
                    InvocationType='RequestResponse',
                    Payload=json.dumps(lambda_payload)
                )
                
                if response['StatusCode'] == 200:
                    result = json.loads(response['Payload'].read().decode('utf-8'))
                    
                    # Check for lambda function errors
                    if 'errorMessage' in result:
                        raise Exception(f"Lambda function error: {result['errorMessage']}")
                    
                    # Success
                    status.status = 'completed'
                    status.end_time = datetime.now()
                    status.s3_location = result.get('s3_location')
                    
                    logger.info(f"✅ Lambda completed for {scenario_id} in {status.duration():.2f}s")
                    logger.info(f"📁 S3 location: {status.s3_location}")
                    break
                    
                else:
                    raise Exception(f"Lambda returned status {response['StatusCode']}")
                    
            except Exception as e:
                error_msg = str(e)
                status.error_message = error_msg
                
                if attempt < self.max_retries - 1:
                    backoff_time = self.backoff_multiplier ** attempt
                    logger.warning(f"⚠️ Lambda failed for {scenario_id} (attempt {attempt + 1}): {error_msg}")
                    logger.info(f"⏳ Retrying in {backoff_time:.1f}s...")
                    time.sleep(backoff_time)
                else:
                    status.status = 'failed'
                    status.end_time = datetime.now()
                    logger.error(f"❌ Lambda failed for {scenario_id} after {self.max_retries} attempts: {error_msg}")
        
        with self.lock:
            self.status_tracker[scenario_id] = status
        
        return status
    
    def run_parallel_experiments(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run experiments in parallel with improved tracking."""
        logger.info(f"🎯 Starting parallel experiment execution...")
        logger.info(f"📊 Total scenarios: {len(scenarios)}")
        logger.info(f"🔧 Concurrent lambdas: {self.max_concurrent_lambdas}")
        
        start_time = datetime.now()
        
        # Group scenarios by date for daily saves
        scenarios_by_date = defaultdict(list)
        for scenario in scenarios:
            # Extract date from scenario (assuming scenario has date info)
            date = scenario.get('date', start_time.strftime('%Y-%m-%d'))
            scenarios_by_date[date].append(scenario['scenario_id'])
        
        # Progress tracking
        completed_count = 0
        failed_count = 0
        total_count = len(scenarios)
        
        # Progress reporting thread
        def progress_reporter():
            while not self.shutdown_requested:
                time.sleep(60)  # Report every minute
                with self.lock:
                    completed = sum(1 for s in self.status_tracker.values() if s.status == 'completed')
                    failed = sum(1 for s in self.status_tracker.values() if s.status == 'failed')
                    running = sum(1 for s in self.status_tracker.values() if s.status == 'running')
                    
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = completed / elapsed if elapsed > 0 else 0
                    
                    logger.info(f"📊 Progress: {completed}/{total_count} completed, {failed} failed, {running} running | Rate: {rate:.2f}/s")
        
        progress_thread = threading.Thread(target=progress_reporter, daemon=True)
        progress_thread.start()
        
        # Execute scenarios in parallel
        with ThreadPoolExecutor(max_workers=self.max_concurrent_lambdas) as executor:
            # Submit all scenarios
            future_to_scenario = {
                executor.submit(self.execute_lambda_with_tracking, scenario): scenario
                for scenario in scenarios
            }
            
            # Process completed scenarios
            for future in as_completed(future_to_scenario):
                if self.shutdown_requested:
                    break
                    
                scenario = future_to_scenario[future]
                try:
                    status = future.result()
                    
                    if status.status == 'completed':
                        completed_count += 1
                    elif status.status == 'failed':
                        failed_count += 1
                    
                    # Save status periodically
                    if (completed_count + failed_count) % 10 == 0:
                        self._save_status()
                        
                except Exception as e:
                    logger.error(f"❌ Unexpected error processing scenario {scenario.get('scenario_id', 'unknown')}: {e}")
                    failed_count += 1
        
        # Save daily results
        for date, scenario_ids in scenarios_by_date.items():
            if not self.shutdown_requested:
                self.save_daily_results(date, scenario_ids)
        
        # Final status save
        self._save_status()
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Generate final report
        report = {
            'experiment_id': self.experiment_id,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': total_duration,
            'total_scenarios': total_count,
            'completed_scenarios': completed_count,
            'failed_scenarios': failed_count,
            'success_rate': completed_count / total_count if total_count > 0 else 0,
            'average_duration': sum(s.duration() for s in self.status_tracker.values() if s.status == 'completed') / max(1, completed_count),
            'daily_summaries': {k: asdict(v) for k, v in self.daily_summaries.items()},
            'shutdown_requested': self.shutdown_requested
        }
        
        logger.info(f"🎉 Experiment completed!")
        logger.info(f"📊 Results: {completed_count}/{total_count} completed ({report['success_rate']:.1%})")
        logger.info(f"⏱️ Duration: {total_duration:.2f}s")
        
        return report
    
    def get_experiment_status(self) -> Dict[str, Any]:
        """Get current experiment status."""
        with self.lock:
            status_counts = defaultdict(int)
            for status in self.status_tracker.values():
                status_counts[status.status] += 1
            
            return {
                'experiment_id': self.experiment_id,
                'timestamp': datetime.now().isoformat(),
                'total_scenarios': len(self.status_tracker),
                'status_breakdown': dict(status_counts),
                'daily_summaries': len(self.daily_summaries),
                'last_save': self.status_file.stat().st_mtime if self.status_file.exists() else None
            }
    
    def print_status_summary(self):
        """Print human-readable status summary."""
        status = self.get_experiment_status()
        
        print(f"\n🔍 Experiment Status Summary")
        print(f"{'='*50}")
        print(f"📅 Timestamp: {status['timestamp']}")
        print(f"🧪 Experiment ID: {status['experiment_id']}")
        print(f"📊 Total Scenarios: {status['total_scenarios']}")
        print(f"\n📈 Status Breakdown:")
        for status_type, count in status['status_breakdown'].items():
            print(f"  • {status_type}: {count}")
        print(f"\n📅 Daily Summaries: {status['daily_summaries']}")
        print(f"{'='*50}")

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Improved Parallel Experiment Runner')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--resume', type=str, help='Resume experiment with given ID')
    parser.add_argument('--status', action='store_true', help='Show experiment status')
    parser.add_argument('--scenarios', type=str, help='Scenarios JSON file')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without actual execution')
    
    args = parser.parse_args()
    
    try:
        runner = ImprovedParallelRunner(
            config_file=args.config,
            resume_experiment=args.resume
        )
        
        if args.status:
            runner.print_status_summary()
            return
        
        if args.scenarios:
            with open(args.scenarios, 'r') as f:
                scenarios = json.load(f)
            
            if args.dry_run:
                logger.info("🔍 Dry run mode - scenarios loaded but not executed")
                logger.info(f"📊 Would execute {len(scenarios)} scenarios")
                return
            
            # Run experiments
            report = runner.run_parallel_experiments(scenarios)
            
            # Save final report
            report_file = f"final_report_{runner.experiment_id}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 Final report saved to {report_file}")
            
        else:
            logger.error("❌ No scenarios file provided. Use --scenarios flag.")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Runner failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 