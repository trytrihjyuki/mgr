#!/usr/bin/env python3
"""
UNIFIED Ride-Hailing Pricing Experiments Runner
===============================================

This is the unified, production-ready experiment runner with full Hikima methodology support.
Includes proper rate limiting, timeout handling, and Hikima-consistent time window parametrization.

HIKIMA TIME WINDOW SETUP:
- Standard: 10:00-20:00 (10 hours)
- Default interval: 5 minutes (120 scenarios/day)
- Manhattan special: Can use 30 seconds intervals
- Other boroughs: 5 minutes recommended

Usage Examples:
    # Standard Hikima setup (10:00-20:00, 5min intervals)
    python run_pricing_experiment.py --year=2019 --month=10 --day=1 \
        --borough=Manhattan --vehicle_type=yellow --eval=PL --methods=LP

    # Custom time window (12:00-18:00, 10min intervals)  
    python run_pricing_experiment.py --year=2019 --month=10 --day=1 \
        --borough=Manhattan --vehicle_type=yellow --eval=PL --methods=LP \
        --hour_start=12 --hour_end=18 --time_interval=10

    # Manhattan with 30-second intervals (Hikima intensive)
    python run_pricing_experiment.py --year=2019 --month=10 --day=1 \
        --borough=Manhattan --vehicle_type=yellow --eval=PL --methods=LP \
        --time_interval=30 --time_unit=s
"""

import boto3
import json
import time
import logging
import argparse
import signal
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading
from typing import List, Dict, Any, Tuple
import random
import sys
import pandas as pd
import numpy as np
import io

# Configure logging with enhanced timestamps
logging.basicConfig(
    level=logging.INFO,  # Restored INFO level for better tracking
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'experiment_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

# Global flags for shutdown handling
shutdown_requested = False
force_exit_count = 0

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully with immediate force exit on second press"""
    global shutdown_requested, force_exit_count
    force_exit_count += 1
    
    if force_exit_count == 1:
        print(f"\n🛑 Shutdown requested (signal {signum}). Stopping execution...")
        print("   Press Ctrl+C again to FORCE EXIT immediately.")
        shutdown_requested = True
        logging.warning(f"Shutdown requested via signal {signum}")
    else:
        print(f"\n💥 FORCE EXIT! Terminating NOW...")
        logging.error("Force exit requested - terminating immediately")
        import os
        import sys
        try:
            # Kill any child processes if psutil is available
            import psutil
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            for child in children:
                child.kill()
            logging.info(f"Killed {len(children)} child processes")
        except ImportError:
            logging.warning("psutil not available - cannot kill child processes automatically")
        except Exception as e:
            logging.warning(f"Error killing child processes: {e}")
        
        try:
            # Try graceful exit first
            sys.exit(130)  # Standard exit code for SIGINT
        except:
            # Force exit if graceful fails
            os._exit(130)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

class ExperimentRunner:
    def __init__(self, region='eu-north-1', parallel_workers=5, production_mode=True, batch_size=5, max_experiment_duration=0, max_lambda_concurrency=700):
        """Initialize the unified experiment runner with Hikima-consistent defaults"""
        # Configure clients with proper timeouts
        import botocore.config
        config = botocore.config.Config(
            read_timeout=850,  # 14+ minutes (close to Lambda timeout)
            connect_timeout=60,
            retries={'max_attempts': 0}  # We handle retries ourselves
        )
        
        self.lambda_client = boto3.client('lambda', region_name=region, config=config)
        self.s3_client = boto3.client('s3', region_name=region)
        self.parallel_workers = min(parallel_workers, 5)  # Cap at 5 to avoid rate limits
        self.production_mode = production_mode
        self.lock = threading.Lock()
        self.success_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Improved timeout and retry settings  
        self.lambda_timeout = 900  # 15 minutes (Lambda max)
        self.client_timeout = 850  # 14+ minutes (close to Lambda timeout)
        self.max_retries = 3
        self.base_backoff = 2.0
        
        # Batch processing settings for improved performance
        self.batch_size = max(1, min(batch_size, 10))  # 1-10 scenarios per batch
        self.use_batch_processing = batch_size > 1
        
        # Configurable maximum execution time (0 = no timeout)
        self.max_experiment_duration = max_experiment_duration if max_experiment_duration else 0
        
        # Lambda concurrency management - increased to 700 concurrent limit
        self.max_lambda_concurrency = max(1, min(max_lambda_concurrency, 700))  # Updated cap to 700
        self.lambda_semaphore = threading.Semaphore(self.max_lambda_concurrency)
        self.active_lambda_count = 0
        self.lambda_count_lock = threading.Lock()  # FIXED: Add back the missing lock
        
        if self.max_lambda_concurrency > 500:
            print(f"⚠️  High concurrency set: {self.max_lambda_concurrency} concurrent Lambda invocations")
            print(f"   Make sure your AWS account can handle this load!")
        
        print(f"🔧 Lambda concurrency limit: {self.max_lambda_concurrency} (max supported: 700)")
        print(f"🔧 Using batch processing: {self.use_batch_processing} (batch size: {self.batch_size})")
        print(f"🔧 Max experiment duration: {self.max_experiment_duration if self.max_experiment_duration else 'unlimited'}")
        print(f"🔧 Production mode: {self.production_mode}")
        
        # Initialize metrics tracking
        self.metrics = {
            'total_lambda_invocations': 0,
            'total_lambda_duration': 0,
            'failed_lambda_invocations': 0,
            'concurrency_waits': 0,
            'max_concurrent_reached': 0
        }
        
        # Circuit breaker for lambda failures to prevent spam
        self.circuit_breaker = {
            'failure_count': 0,
            'failure_threshold': 5,  # Open circuit after 5 consecutive failures (more aggressive)
            'last_failure_time': 0,
            'timeout_duration': 180,  # 3 minutes (shorter timeout)
            'state': 'closed',  # closed, open, half-open
            'rate_limit_count': 0,  # Track rate limit failures specifically
            'rate_limit_threshold': 3  # Open after 3 rate limit failures
        }
        
        # Error tracking for better reporting
        self.error_patterns = {}
        self.lambda_health_status = 'healthy'  # healthy, degraded, unhealthy
        
    def check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows lambda invocation."""
        current_time = time.time()
        
        if self.circuit_breaker['state'] == 'open':
            # Check if timeout has elapsed
            if current_time - self.circuit_breaker['last_failure_time'] > self.circuit_breaker['timeout_duration']:
                self.circuit_breaker['state'] = 'half-open'
                print(f"🔄 Circuit breaker moving to half-open state")
                return True
            else:
                remaining_time = self.circuit_breaker['timeout_duration'] - (current_time - self.circuit_breaker['last_failure_time'])
                print(f"🚫 Circuit breaker OPEN - {remaining_time:.0f}s remaining")
                return False
        
        return True
    
    def record_lambda_success(self):
        """Record successful lambda invocation."""
        if self.circuit_breaker['state'] == 'half-open':
            self.circuit_breaker['state'] = 'closed'
            self.circuit_breaker['failure_count'] = 0
            print(f"✅ Circuit breaker closed - lambda healthy")
        elif self.circuit_breaker['state'] == 'closed':
            # Reset failure count on success
            self.circuit_breaker['failure_count'] = max(0, self.circuit_breaker['failure_count'] - 1)
        
        self.lambda_health_status = 'healthy'
    
    def record_lambda_failure(self, error_message: str):
        """Record lambda failure and update circuit breaker."""
        self.circuit_breaker['failure_count'] += 1
        self.circuit_breaker['last_failure_time'] = time.time()
        
        # Track error patterns
        error_type = self._classify_error(error_message)
        self.error_patterns[error_type] = self.error_patterns.get(error_type, 0) + 1
        
        # Special handling for rate limit errors (more aggressive)
        if error_type == 'rate_limit':
            self.circuit_breaker['rate_limit_count'] += 1
            print(f"⚠️ Rate limit failure #{self.circuit_breaker['rate_limit_count']}: {error_message}")
            
            # Open circuit immediately on rate limit threshold
            if self.circuit_breaker['rate_limit_count'] >= self.circuit_breaker['rate_limit_threshold']:
                self.circuit_breaker['state'] = 'open'
                self.lambda_health_status = 'unhealthy'
                print(f"🚫 Circuit breaker OPENED due to rate limiting ({self.circuit_breaker['rate_limit_count']} rate limit failures)")
                return
        
        # Update health status for general failures
        if self.circuit_breaker['failure_count'] >= self.circuit_breaker['failure_threshold']:
            self.circuit_breaker['state'] = 'open'
            self.lambda_health_status = 'unhealthy'
            print(f"⚠️ Circuit breaker OPENED - too many failures ({self.circuit_breaker['failure_count']})")
        elif self.circuit_breaker['failure_count'] >= self.circuit_breaker['failure_threshold'] // 2:
            self.lambda_health_status = 'degraded'
            print(f"⚠️ Lambda health degraded - {self.circuit_breaker['failure_count']} failures")
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error type for tracking with enhanced rate limit detection."""
        error_lower = error_message.lower()
        
        # More comprehensive rate limiting detection
        rate_limit_indicators = [
            'rate limit', 'throttl', 'throttling', 'rate exceeded', 
            'too many requests', 'limit exceeded', 'quota exceeded',
            'provisioned concurrency', 'concurrent executions exceeded',
            'service unavailable', 'busy', 'lambda is scaling',
            'toomanyrequestsexception', 'throttlingexception'
        ]
        
        if any(indicator in error_lower for indicator in rate_limit_indicators):
            return 'rate_limit'
        elif 'timeout' in error_lower or 'time out' in error_lower:
            return 'timeout'
        elif 'memory' in error_lower or 'out of memory' in error_lower:
            return 'memory'
        elif 'permission' in error_lower or 'access denied' in error_lower:
            return 'permission'
        elif 'invalid' in error_lower or 'bad request' in error_lower:
            return 'invalid_request'
        else:
            return 'unknown'
    
    def get_lambda_health_summary(self) -> Dict[str, Any]:
        """Get lambda health summary."""
        return {
            'status': self.lambda_health_status,
            'circuit_breaker_state': self.circuit_breaker['state'],
            'failure_count': self.circuit_breaker['failure_count'],
            'error_patterns': self.error_patterns,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': self.success_count / max(1, self.success_count + self.error_count)
        }
        
    def invoke_lambda_with_retry(self, payload: Dict[str, Any], scenario_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Invoke Lambda with exponential backoff and proper timeout handling"""
        
        # Check circuit breaker before attempting
        if not self.check_circuit_breaker():
            return False, {'error': 'Circuit breaker is open - lambda service unavailable', 'scenario_id': scenario_id}
        
        for attempt in range(self.max_retries):
            # Check for shutdown before each attempt
            if shutdown_requested:
                return False, {'error': 'Shutdown requested during Lambda invocation', 'scenario_id': scenario_id}
            
            # Acquire semaphore to respect concurrency limit
            acquired = self.lambda_semaphore.acquire(blocking=True, timeout=300)  # 5 minute timeout
            if not acquired:
                return False, {'error': 'Lambda concurrency limit timeout', 'scenario_id': scenario_id}
                
            try:
                with self.lambda_count_lock:
                    self.active_lambda_count += 1
                    if not self.production_mode and self.active_lambda_count % 10 == 1:
                        print(f"⚡ Active Lambda invocations: {self.active_lambda_count}/{self.max_lambda_concurrency}")
                
                # Add jitter to prevent thundering herd  
                backoff_time = (self.base_backoff ** attempt) + random.uniform(0, 2) if attempt > 0 else 0
                if attempt > 0:
                    if not self.production_mode:
                        print(f"⏳ {scenario_id}: Retrying (attempt {attempt+1}) after {backoff_time:.1f}s...")
                    time.sleep(backoff_time)
                
                if not self.production_mode:
                    print(f"📤 {scenario_id}: Invoking Lambda (attempt {attempt+1}/{self.max_retries})")
                
                # Track Lambda invocation time
                lambda_start_time = time.time()
                
                # Configure client with proper timeout
                response = self.lambda_client.invoke(
                    FunctionName='rideshare-pricing-benchmark',
                    InvocationType='RequestResponse',
                    Payload=json.dumps(payload)
                )
                
                lambda_duration = time.time() - lambda_start_time
                
                if not self.production_mode:
                    print(f"📥 {scenario_id}: Lambda responded in {lambda_duration:.1f}s (status: {response['StatusCode']})")
                
                if response['StatusCode'] == 200:
                    result = json.loads(response['Payload'].read().decode('utf-8'))
                    
                    # Check if Lambda function returned an error (even with 200 status)
                    if isinstance(result, dict) and 'errorMessage' in result:
                        error_msg = f"Lambda function error: {result.get('errorType', 'Unknown')} - {result.get('errorMessage', 'No message')}"
                        self.record_lambda_failure(error_msg)
                        return False, {'error': error_msg, 'scenario_id': scenario_id}
                    
                    # Success
                    self.record_lambda_success()
                    return True, result
                else:
                    error_msg = f"Lambda returned status {response['StatusCode']}"
                    if not self.production_mode:
                        print(f"⚠️ {scenario_id}: {error_msg}")
                    
            except Exception as e:
                error_str = str(e)
                
                # Handle specific error types
                if 'TooManyRequestsException' in error_str or 'Rate Exceeded' in error_str:
                    if attempt < self.max_retries - 1:
                        next_backoff = (self.base_backoff ** (attempt + 1)) + random.uniform(0, 2)
                        if not self.production_mode:
                            print(f"⏳ {scenario_id}: Rate limited, retrying in {next_backoff:.1f}s...")
                        continue
                    else:
                        return False, {'error': 'Rate limit exceeded after retries', 'scenario_id': scenario_id}
                        
                elif 'timeout' in error_str.lower() or 'timed out' in error_str.lower():
                    if not self.production_mode:
                        print(f"⏰ {scenario_id}: Timeout ({attempt+1}/{self.max_retries}) - Client timeout: {self.client_timeout}s")
                    if attempt < self.max_retries - 1:
                        continue
                    else:
                        return False, {'error': f'Timeout after retries (client timeout: {self.client_timeout}s)', 'scenario_id': scenario_id}
                
                else:
                    self.record_lambda_failure(str(e))
                    return False, {'error': str(e), 'scenario_id': scenario_id}
            
            finally:
                # Always release semaphore and update count
                with self.lambda_count_lock:
                    self.active_lambda_count = max(0, self.active_lambda_count - 1)
                self.lambda_semaphore.release()
        
        # Max retries exceeded
        self.record_lambda_failure('Max retries exceeded')
        return False, {'error': 'Max retries exceeded', 'scenario_id': scenario_id}
    
    def invoke_lambda_batch_with_retry(self, batch_payload: Dict[str, Any], batch_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Invoke Lambda with batch processing for improved performance"""
        
        for attempt in range(self.max_retries):
            # Check for shutdown before each attempt
            if shutdown_requested:
                return False, {'error': 'Shutdown requested during Lambda batch invocation', 'batch_id': batch_id}
                
            # Acquire semaphore to respect concurrency limit
            acquired = self.lambda_semaphore.acquire(blocking=True, timeout=300)  # 5 minute timeout
            if not acquired:
                return False, {'error': 'Lambda concurrency limit timeout', 'batch_id': batch_id}
            
            try:
                with self.lambda_count_lock:
                    self.active_lambda_count += 1
                    if not self.production_mode and self.active_lambda_count % 10 == 1:
                        print(f"⚡ Active Lambda invocations: {self.active_lambda_count}/{self.max_lambda_concurrency}")
                
                # Add jitter to prevent thundering herd  
                backoff_time = (self.base_backoff ** attempt) + random.uniform(0, 2) if attempt > 0 else 0
                if attempt > 0:
                    if not self.production_mode:
                        print(f"⏳ {batch_id}: Retrying batch (attempt {attempt+1}) after {backoff_time:.1f}s...")
                    time.sleep(backoff_time)
                
                if not self.production_mode:
                    print(f"📤 {batch_id}: Invoking Lambda batch ({len(batch_payload['batch_scenarios'])} scenarios, attempt {attempt+1}/{self.max_retries})")
                
                # Track Lambda invocation time
                lambda_start_time = time.time()
                
                # Configure client with proper timeout
                response = self.lambda_client.invoke(
                    FunctionName='rideshare-pricing-benchmark',
                    InvocationType='RequestResponse',
                    Payload=json.dumps(batch_payload)
                )
                
                lambda_duration = time.time() - lambda_start_time
                
                if not self.production_mode:
                    print(f"📥 {batch_id}: Lambda responded in {lambda_duration:.1f}s (status: {response['StatusCode']})")
                
                if response['StatusCode'] == 200:
                    result = json.loads(response['Payload'].read().decode('utf-8'))
                    
                    # Check if Lambda function returned an error (even with 200 status)
                    if isinstance(result, dict) and 'errorMessage' in result:
                        error_msg = f"Lambda function error: {result.get('errorType', 'Unknown')} - {result.get('errorMessage', 'No message')}"
                        return False, {'error': error_msg, 'batch_id': batch_id}
                    
                    return True, result
                else:
                    error_msg = f"Lambda returned status {response['StatusCode']}"
                    if not self.production_mode:
                        print(f"⚠️ {batch_id}: {error_msg}")
                    
            except Exception as e:
                error_str = str(e)
                
                # Handle specific error types
                if 'TooManyRequestsException' in error_str or 'Rate Exceeded' in error_str:
                    if attempt < self.max_retries - 1:
                        next_backoff = (self.base_backoff ** (attempt + 1)) + random.uniform(0, 2)
                        if not self.production_mode:
                            print(f"⏳ {batch_id}: Rate limited, retrying in {next_backoff:.1f}s...")
                        continue
                    else:
                        return False, {'error': 'Rate limit exceeded after retries', 'batch_id': batch_id}
                        
                elif 'timeout' in error_str.lower() or 'timed out' in error_str.lower():
                    if not self.production_mode:
                        print(f"⏰ {batch_id}: Timeout ({attempt+1}/{self.max_retries}) - Client timeout: {self.client_timeout}s")
                    if attempt < self.max_retries - 1:
                        continue
                    else:
                        return False, {'error': f'Timeout after retries (client timeout: {self.client_timeout}s)', 'batch_id': batch_id}
                
                else:
                    return False, {'error': str(e), 'batch_id': batch_id}
            
            finally:
                # Always release semaphore and update count
                with self.lambda_count_lock:
                    self.active_lambda_count = max(0, self.active_lambda_count - 1)
                self.lambda_semaphore.release()
        
        return False, {'error': 'Max retries exceeded', 'batch_id': batch_id}

    def process_scenario(self, scenario_params: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single scenario with improved error handling"""
        scenario_id = scenario_params['scenario_id']
        
        # Check for shutdown at the start
        if shutdown_requested:
            return {
                'scenario_id': scenario_id,
                'success': False,
                'result': {'error': 'Shutdown requested'},
                's3_location': None,
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            if not self.production_mode:
                print(f"🚀 Starting scenario: {scenario_id}")
            
            success, result = self.invoke_lambda_with_retry(scenario_params, scenario_id)
            
            # Extract S3 location if available - FIXED: Lambda client returns data directly, not in 'body'
            s3_location = None
            if success and isinstance(result, dict):
                # First try direct access (Lambda client invocation)
                s3_location = result.get('s3_location')
                
                # Fallback: check if it's in a 'body' field (HTTP API Gateway format)
                if not s3_location:
                    body_str = result.get('body', '{}')
                    if isinstance(body_str, str):
                        try:
                            body_data = json.loads(body_str)
                            s3_location = body_data.get('s3_location')
                        except json.JSONDecodeError:
                            pass
            
            with self.lock:
                if success:
                    self.success_count += 1
                    if not self.production_mode:
                        if s3_location:
                            print(f"   ✅ {scenario_id}: Success -> {s3_location}")
                        else:
                            print(f"   ✅ {scenario_id}: Success")
                else:
                    self.error_count += 1
                    error_msg = result.get('error', 'Unknown error')
                    print(f"   ❌ {scenario_id}: {error_msg}")
                
                # Progress update
                total_processed = self.success_count + self.error_count
                if total_processed % 10 == 0 or not self.production_mode:
                    elapsed = time.time() - self.start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    print(f"⚡ [{total_processed:3d}/???] ✅{self.success_count} ❌{self.error_count} | Rate: {rate:.1f}/s")
            
            return {
                'scenario_id': scenario_id,
                'success': success,
                'result': result,
                's3_location': s3_location,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            with self.lock:
                self.error_count += 1
            return {
                'scenario_id': scenario_id,
                'success': False,
                'result': {'error': str(e)},
                's3_location': None,
                'timestamp': datetime.now().isoformat()
            }
    
    def process_batch(self, batch_scenarios: List[Dict[str, Any]], batch_id: str) -> List[Dict[str, Any]]:
        """Process a batch of scenarios with improved performance"""
        
        # Check for shutdown at the start
        if shutdown_requested:
            return [{
                'scenario_id': scenario.get('scenario_id', f'batch_{batch_id}_scenario_{i}'),
                'success': False,
                'result': {'error': 'Shutdown requested'},
                's3_location': None,
                'timestamp': datetime.now().isoformat()
            } for i, scenario in enumerate(batch_scenarios)]
        
        try:
            if not self.production_mode:
                print(f"🚀 Starting batch {batch_id} with {len(batch_scenarios)} scenarios")
            
            # Create batch payload
            batch_payload = {
                'batch_scenarios': batch_scenarios,
                'num_eval': batch_scenarios[0].get('num_eval', 1000) if batch_scenarios else 1000
            }
            
            success, result = self.invoke_lambda_batch_with_retry(batch_payload, batch_id)
            
            if success and isinstance(result, dict):
                # Check if Lambda function returned an error (even with 200 status)
                if 'errorMessage' in result:
                    error_msg = f"Lambda function error: {result.get('errorType', 'Unknown')} - {result.get('errorMessage', 'No message')}"
                    return [{
                        'scenario_id': scenario.get('scenario_id', f'batch_{batch_id}_scenario_{i}'),
                        'success': False,
                        'result': {'error': error_msg},
                        's3_location': None,
                        'timestamp': datetime.now().isoformat()
                    } for i, scenario in enumerate(batch_scenarios)]
                
                # Parse Lambda response - handle both direct response and body wrapper
                batch_data = None
                if 'body' in result:
                    # Lambda response wrapped in body (direct invocation)
                    try:
                        batch_data = json.loads(result['body'])
                    except json.JSONDecodeError as e:
                        if not self.production_mode:
                            print(f"   ❌ JSON decode error in batch response: {e}")
                        batch_data = None
                else:
                    # Direct response (no body wrapper)
                    batch_data = result
                
                # Process batch data
                if batch_data and batch_data.get('batch_mode') and 'results' in batch_data:
                    batch_results = []
                    
                    for i, scenario_result in enumerate(batch_data['results']):
                        scenario_id = scenario_result.get('scenario_id', f'batch_{batch_id}_scenario_{i}')
                        
                        # Check if scenario was successful
                        if scenario_result.get('status') == 'success':
                            s3_location = scenario_result.get('s3_location')
                            batch_results.append({
                                'scenario_id': scenario_id,
                                'success': True,
                                'result': scenario_result,
                                's3_location': s3_location,
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            with self.lock:
                                self.success_count += 1
                                if not self.production_mode:
                                    if s3_location:
                                        print(f"   ✅ {scenario_id}: Success -> {s3_location}")
                                    else:
                                        print(f"   ✅ {scenario_id}: Success")
                        else:
                            error_msg = scenario_result.get('error', 'Unknown error')
                            batch_results.append({
                                'scenario_id': scenario_id,
                                'success': False,
                                'result': {'error': error_msg},
                                's3_location': None,
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            with self.lock:
                                self.error_count += 1
                                print(f"   ❌ {scenario_id}: {error_msg}")
                    
                    with self.lock:
                        # Progress update
                        total_processed = self.success_count + self.error_count
                        elapsed = time.time() - self.start_time
                        rate = total_processed / elapsed if elapsed > 0 else 0
                        print(f"⚡ Batch {batch_id}: [{total_processed:3d}/???] ✅{self.success_count} ❌{self.error_count} | Rate: {rate:.1f}/s")
                    
                    return batch_results
                else:
                    # Handle non-batch response format
                    error_details = "Unknown response format"
                    if batch_data:
                        error_details = f"batch_mode={batch_data.get('batch_mode')}, has_results={'results' in batch_data}"
                    elif 'body' in result:
                        error_details = f"body content (first 200 chars): {result['body'][:200]}"
                    
                    if not self.production_mode:
                        print(f"   🔍 Batch response debug: {error_details}")
                    
                    return [{
                        'scenario_id': scenario.get('scenario_id', f'batch_{batch_id}_scenario_{i}'),
                        'success': False,
                        'result': {'error': f'Invalid batch response format: {error_details}'},
                        's3_location': None,
                        'timestamp': datetime.now().isoformat()
                    } for i, scenario in enumerate(batch_scenarios)]
            else:
                # Batch failed
                error_msg = result.get('error', 'Batch processing failed')
                with self.lock:
                    self.error_count += len(batch_scenarios)
                
                return [{
                    'scenario_id': scenario.get('scenario_id', f'batch_{batch_id}_scenario_{i}'),
                    'success': False,
                    'result': {'error': error_msg},
                    's3_location': None,
                    'timestamp': datetime.now().isoformat()
                } for i, scenario in enumerate(batch_scenarios)]
                
        except Exception as e:
            with self.lock:
                self.error_count += len(batch_scenarios)
            
            return [{
                'scenario_id': scenario.get('scenario_id', f'batch_{batch_id}_scenario_{i}'),
                'success': False,
                'result': {'error': str(e)},
                's3_location': None,
                'timestamp': datetime.now().isoformat()
            } for i, scenario in enumerate(batch_scenarios)]

    def run_experiments(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run experiments with improved concurrency control and optional batch processing"""
        total_scenarios = len(scenarios)
        
        # Determine processing mode
        if self.use_batch_processing:
            print(f"🚀 CLOUD EXECUTION: {total_scenarios} scenarios, {self.parallel_workers} parallel workers, BATCH mode (size={self.batch_size})")
        else:
            print(f"🚀 CLOUD EXECUTION: {total_scenarios} scenarios, {self.parallel_workers} parallel workers, SINGLE mode")
        
        # Adaptive worker count based on scenario complexity and Monte Carlo simulations
        complex_scenarios = any('methods' in s and len(s.get('methods', [])) >= 3 for s in scenarios)
        high_simulation_count = any('num_eval' in s and s.get('num_eval', 1000) >= 1000 for s in scenarios)
        
        if complex_scenarios or high_simulation_count:
            # Reduce workers for complex scenarios (3+ methods) or high simulation counts (1000+)
            adaptive_workers = min(self.parallel_workers, 2)  # Even more conservative
            complexity_reason = []
            if complex_scenarios:
                complexity_reason.append("3+ methods")
            if high_simulation_count:
                complexity_reason.append("1000+ simulations")
            print(f"🔧 Complex scenarios detected ({', '.join(complexity_reason)}), reducing workers to {adaptive_workers}")
        else:
            adaptive_workers = self.parallel_workers
        
        # Use batch processing if enabled
        if self.use_batch_processing:
            return self.run_experiments_batch(scenarios, adaptive_workers)
        else:
            return self.run_experiments_single(scenarios, adaptive_workers)
    
    def run_experiments_batch(self, scenarios: List[Dict[str, Any]], adaptive_workers: int) -> List[Dict[str, Any]]:
        """Run experiments using batch processing for improved performance"""
        total_scenarios = len(scenarios)
        
        # Split scenarios into batches
        batches = []
        for i in range(0, len(scenarios), self.batch_size):
            batch = scenarios[i:i + self.batch_size]
            batches.append(batch)
        
        print(f"📦 Created {len(batches)} batches of up to {self.batch_size} scenarios each")
        
        results = []
        submitted_count = 0
        completed_count = 0
        
        print(f"🎯 Starting batch submission...")
        
        with ThreadPoolExecutor(max_workers=adaptive_workers) as executor:
            # Submit all batches with progress tracking
            future_to_batch = {}
            for batch_idx, batch in enumerate(batches):
                # Check for shutdown during submission
                if shutdown_requested:
                    print(f"🛑 Shutdown requested during submission, stopping at batch {batch_idx}/{len(batches)}")
                    break
                    
                try:
                    batch_id = f"batch_{batch_idx:03d}"
                    future = executor.submit(self.process_batch, batch, batch_id)
                    future_to_batch[future] = (batch_idx, batch)
                    submitted_count += len(batch)
                    if (batch_idx + 1) % 5 == 0 or batch_idx == len(batches) - 1:
                        print(f"   📤 Submitted {batch_idx + 1}/{len(batches)} batches ({submitted_count} scenarios)")
                except Exception as e:
                    print(f"   ❌ Failed to submit batch {batch_idx}: {e}")
            
            print(f"✅ Submitted {len(future_to_batch)} batches to {adaptive_workers} workers")
            print(f"⏳ Waiting for batch completions...")
            
            # Early exit if all submission was cancelled
            if len(future_to_batch) == 0:
                print("🛑 No batches submitted, exiting...")
                return []
            
            # Process completed batches with timeout and progress tracking
            start_time = time.time()
            last_progress_time = start_time
            
            # Monitor execution with timeout and shutdown handling
            execution_timeout = 7200  # 2 hours max execution
            
            try:
                for future in as_completed(future_to_batch, timeout=execution_timeout):
                    # Check for shutdown request first
                    if shutdown_requested:
                        print(f"🛑 Shutdown requested, cancelling remaining batches...")
                        # Cancel all remaining futures
                        cancelled_count = 0
                        for remaining_future in future_to_batch:
                            if not remaining_future.done():
                                if remaining_future.cancel():
                                    cancelled_count += 1
                        print(f"   ✅ Cancelled {cancelled_count} pending batches")
                        break
                    
                    try:
                        batch_results = future.result(timeout=120)  # 2 minutes timeout per batch result
                        results.extend(batch_results)
                        completed_count += len(batch_results)
                        
                        # Progress reporting
                        current_time = time.time()
                        if current_time - last_progress_time >= 30 or completed_count >= total_scenarios:  # Every 30s or final
                            elapsed = current_time - start_time
                            rate = completed_count / elapsed if elapsed > 0 else 0
                            remaining = total_scenarios - completed_count
                            eta = remaining / rate if rate > 0 else "unknown"
                            eta_str = f"{eta:.0f}s" if isinstance(eta, (int, float)) else eta
                            
                            print(f"⚡ Progress: {completed_count}/{total_scenarios} ({100*completed_count/total_scenarios:.1f}%) | Rate: {rate:.2f}/s | ETA: {eta_str}")
                            
                            # Debug info for hanging batches
                            pending_count = len(future_to_batch) - len([f for f in future_to_batch if f.done()])
                            if pending_count > 0:
                                print(f"   📊 Active: {pending_count} batches still running")
                            
                            last_progress_time = current_time
                            
                    except TimeoutError:
                        print(f"⏰ Timeout waiting for batch result (120s)")
                        batch_idx, batch = future_to_batch.get(future, (-1, []))
                        print(f"   ⚠️ Timed out batch: batch_{batch_idx:03d}")
                        
                        # Add timeout results for all scenarios in the batch
                        for i, scenario in enumerate(batch):
                            results.append({
                                'scenario_id': scenario.get('scenario_id', f'batch_{batch_idx}_scenario_{i}'),
                                'success': False,
                                'result': {'error': 'Batch processing timeout (120s)'},
                                's3_location': None,
                                'timestamp': datetime.now().isoformat()
                            })
                        completed_count += len(batch)
                        
                    except Exception as e:
                        print(f"❌ Error processing batch result: {e}")
                        batch_idx, batch = future_to_batch.get(future, (-1, []))
                        
                        # Add error results for all scenarios in the batch
                        for i, scenario in enumerate(batch):
                            results.append({
                                'scenario_id': scenario.get('scenario_id', f'batch_{batch_idx}_scenario_{i}'),
                                'success': False,
                                'result': {'error': str(e)},
                                's3_location': None,
                                'timestamp': datetime.now().isoformat()
                            })
                        completed_count += len(batch)
                        
            except TimeoutError:
                print(f"🚨 EXECUTION TIMEOUT: Maximum execution time ({execution_timeout}s) exceeded!")
                print(f"   📊 Completed: {completed_count}/{total_scenarios} scenarios")
                
                # Mark remaining scenarios as failed
                for future, (batch_idx, batch) in future_to_batch.items():
                    if not future.done():
                        for i, scenario in enumerate(batch):
                            results.append({
                                'scenario_id': scenario.get('scenario_id', f'batch_{batch_idx}_scenario_{i}'),
                                'success': False,
                                'result': {'error': f'Execution timeout ({execution_timeout}s)'},
                                's3_location': None,
                                'timestamp': datetime.now().isoformat()
                            })
                        completed_count += len(batch)
        
        return results
    
    def run_experiments_single(self, scenarios: List[Dict[str, Any]], adaptive_workers: int) -> List[Dict[str, Any]]:
        """Run experiments using single scenario processing (original method)"""
        total_scenarios = len(scenarios)
        
        results = []
        submitted_count = 0
        completed_count = 0
        
        print(f"🎯 Starting scenario submission...")
        
        with ThreadPoolExecutor(max_workers=adaptive_workers) as executor:
            # Submit all scenarios with progress tracking
            future_to_scenario = {}
            for scenario in scenarios:
                # Check for shutdown during submission
                if shutdown_requested:
                    print(f"🛑 Shutdown requested during submission, stopping at {submitted_count}/{len(scenarios)}")
                    break
                    
                try:
                    future = executor.submit(self.process_scenario, scenario)
                    future_to_scenario[future] = scenario
                    submitted_count += 1
                    if submitted_count % 10 == 0 or submitted_count == len(scenarios):
                        print(f"   📤 Submitted {submitted_count}/{len(scenarios)} scenarios")
                except Exception as e:
                    print(f"   ❌ Failed to submit scenario {scenario.get('scenario_id', 'unknown')}: {e}")
            
            print(f"✅ Submitted {submitted_count}/{len(scenarios)} scenarios to {adaptive_workers} workers")
            print(f"⏳ Waiting for completions...")
            
            # Early exit if all submission was cancelled
            if submitted_count == 0:
                print("🛑 No scenarios submitted, exiting...")
                return []
            
            # Process completed scenarios with timeout and progress tracking
            start_time = time.time()
            last_progress_time = start_time
            
            # Monitor execution with timeout and shutdown handling
            execution_timeout = self.max_experiment_duration if self.max_experiment_duration > 0 else 7200  # Use configured timeout or default 2 hours
            
            try:
                for future in as_completed(future_to_scenario, timeout=execution_timeout):
                    # Check for shutdown request first
                    if shutdown_requested:
                        print(f"🛑 Shutdown requested, cancelling remaining scenarios...")
                        # Cancel all remaining futures
                        cancelled_count = 0
                        for remaining_future in future_to_scenario:
                            if not remaining_future.done():
                                if remaining_future.cancel():
                                    cancelled_count += 1
                        print(f"   ✅ Cancelled {cancelled_count} pending scenarios")
                        break
                    
                    # Check if experiment is taking too long (only if timeout is configured)
                    if self.max_experiment_duration > 0:
                        elapsed_time = time.time() - start_time
                        if elapsed_time > self.max_experiment_duration:
                            print(f"⏰ Experiment timeout reached ({self.max_experiment_duration}s), stopping execution...")
                            # Cancel remaining futures
                            cancelled_count = 0
                            for remaining_future in future_to_scenario:
                                if not remaining_future.done():
                                    if remaining_future.cancel():
                                        cancelled_count += 1
                            print(f"   ✅ Cancelled {cancelled_count} pending scenarios due to timeout")
                            break
                    
                    try:
                        result = future.result(timeout=60)  # Increased to 60s timeout per result
                        results.append(result)
                        completed_count += 1
                        
                        # Progress reporting
                        current_time = time.time()
                        if current_time - last_progress_time >= 30 or completed_count == len(scenarios):  # Every 30s or final
                            elapsed = current_time - start_time
                            rate = completed_count / elapsed if elapsed > 0 else 0
                            remaining = len(scenarios) - completed_count
                            eta = remaining / rate if rate > 0 else "unknown"
                            eta_str = f"{eta:.0f}s" if isinstance(eta, (int, float)) else eta
                            
                            print(f"⚡ Progress: {completed_count}/{len(scenarios)} ({100*completed_count/len(scenarios):.1f}%) | Rate: {rate:.2f}/s | ETA: {eta_str}")
                            
                            # Debug info for hanging scenarios
                            pending_count = len(future_to_scenario) - completed_count
                            if pending_count > 0:
                                print(f"   📊 Active: {pending_count} scenarios still running")
                            
                            last_progress_time = current_time
                            
                    except TimeoutError:
                        print(f"⏰ Timeout waiting for scenario result (60s)")
                        scenario = future_to_scenario.get(future, {})
                        scenario_id = scenario.get('scenario_id', 'unknown')
                        print(f"   ⚠️ Timed out scenario: {scenario_id}")
                        results.append({
                            'scenario_id': scenario_id,
                            'success': False,
                            'result': {'error': 'Result processing timeout (60s)'},
                            's3_location': None,
                            'timestamp': datetime.now().isoformat()
                        })
                        completed_count += 1
                    except Exception as e:
                        print(f"❌ Error processing scenario result: {e}")
                        scenario = future_to_scenario.get(future, {})
                        scenario_id = scenario.get('scenario_id', 'unknown')
                        results.append({
                            'scenario_id': scenario_id,
                            'success': False,
                            'result': {'error': str(e)},
                            's3_location': None,
                            'timestamp': datetime.now().isoformat()
                        })
                        completed_count += 1
                        
            except TimeoutError:
                print(f"🚨 EXECUTION TIMEOUT: Maximum execution time ({execution_timeout}s) exceeded!")
                print(f"   📊 Completed: {completed_count}/{len(scenarios)} scenarios")
                
                # Mark remaining scenarios as failed
                for future, scenario in future_to_scenario.items():
                    if future not in [f for f in future_to_scenario.keys() if f.done()]:
                        results.append({
                            'scenario_id': scenario.get('scenario_id', 'unknown'),
                            'success': False,
                            'result': {'error': f'Execution timeout ({execution_timeout}s)'},
                            's3_location': None,
                            'timestamp': datetime.now().isoformat()
                        })
                        completed_count += 1
        
        # Final summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        elapsed = time.time() - self.start_time
        
        print(f"\n📊 EXECUTION COMPLETE:")
        print(f"   ✅ Successful: {successful}/{total_scenarios}")
        print(f"   ❌ Failed: {failed}/{total_scenarios}")
        print(f"   ⏱️ Total Time: {elapsed:.1f}s")
        print(f"   📈 Average Rate: {total_scenarios/elapsed:.1f} scenarios/s")
        
        return results
    
    def check_daily_save_status(self, day: int, eval_func: str, args) -> bool:
        """Check if daily results are already saved to S3."""
        import os
        
        # Check local cache first
        cache_dir = f"daily_cache_{args.year}_{args.month:02d}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"day_{day:02d}_{eval_func}_saved.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    save_info = json.load(f)
                    if save_info.get('saved_to_s3', False):
                        print(f"   ⏭️ Daily save for Day {day} {eval_func} already completed - skipping")
                        return True
            except Exception as e:
                print(f"   ⚠️ Error checking daily save cache: {e}")
        
        # Check S3 directly
        try:
            s3_prefix = f"experiments/type={args.vehicle_type}/eval={eval_func}/borough={args.borough}/year={args.year}/month={args.month:02d}/day={day:02d}/"
            response = self.s3_client.list_objects_v2(
                Bucket='magisterka',
                Prefix=s3_prefix,
                MaxKeys=1
            )
            
            if response.get('Contents'):
                print(f"   ⏭️ Daily save for Day {day} {eval_func} found in S3 - skipping")
                # Update local cache
                save_info = {
                    'saved_to_s3': True,
                    's3_prefix': s3_prefix,
                    'timestamp': datetime.now().isoformat()
                }
                with open(cache_file, 'w') as f:
                    json.dump(save_info, f)
                return True
                
        except Exception as e:
            print(f"   ⚠️ Error checking S3 for daily save: {e}")
        
        return False
    
    def mark_daily_save_complete(self, day: int, eval_func: str, args, s3_location: str):
        """Mark daily save as complete in local cache."""
        import os
        
        cache_dir = f"daily_cache_{args.year}_{args.month:02d}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"day_{day:02d}_{eval_func}_saved.json")
        
        save_info = {
            'saved_to_s3': True,
            's3_location': s3_location,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(save_info, f)
        
        print(f"   ✅ Marked Day {day} {eval_func} as saved to S3")

    def aggregate_and_save_results(self, results: List[Dict[str, Any]], args, training_id: str) -> Dict[str, str]:
        """Aggregate all scenario results and save them by day to S3 with daily save checking."""
        print(f"\n📊 AGGREGATING RESULTS BY DAY (with daily save checking)...")
        
        # Group results by day and eval function
        results_by_day_eval = {}
        
        for result in results:
            if not result['success']:
                continue
                
            scenario_id = result['scenario_id']
            lambda_result = result['result']
            
            # Parse scenario_id: day{day:02d}_{eval_func}_s{scenario_idx:03d}
            parts = scenario_id.split('_')
            if len(parts) >= 3:
                day_part = parts[0]  # day01, day06, etc.
                eval_func = parts[1]  # PL, Sigmoid
                scenario_idx = int(parts[2][1:])  # s000 -> 0
                
                day_num = int(day_part[3:])  # day01 -> 1
                
                key = (day_num, eval_func)
                if key not in results_by_day_eval:
                    results_by_day_eval[key] = []
                
                # Add scenario data with Lambda result
                scenario_data = {
                    'scenario_id': scenario_id,
                    'scenario_index': scenario_idx,
                    'lambda_result': lambda_result,
                    'timestamp': result['timestamp']
                }
                results_by_day_eval[key].append(scenario_data)
        
        saved_paths = {}
        
        # Save aggregated results for each day+eval combination
        for (day_num, eval_func), day_scenarios in results_by_day_eval.items():
            if not day_scenarios:
                continue
                
            print(f"   📅 Processing Day {day_num}, Eval: {eval_func} ({len(day_scenarios)} scenarios)")
            
            # Check if already saved
            if self.check_daily_save_status(day_num, eval_func, args):
                # Still add to saved_paths for reporting
                s3_location = f"s3://magisterka/experiments/type={args.vehicle_type}/eval={eval_func}/borough={args.borough}/year={args.year}/month={args.month:02d}/day={day_num:02d}/"
                saved_paths[f"day{day_num:02d}_{eval_func}"] = s3_location
                continue
            
            # Generate experiment ID and timestamp for this day's experiment
            experiment_id = f"{random.randint(10000, 99999)}"
            execution_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create S3 path
            base_path = f"experiments/type={args.vehicle_type}/eval={eval_func}/borough={args.borough}/year={args.year}/month={args.month:02d}/day={day_num:02d}/{experiment_id}_{execution_timestamp}"
            
            # Aggregate experiment metadata
            experiment_summary = {
                'experiment_metadata': {
                    'experiment_id': experiment_id,
                    'execution_timestamp': execution_timestamp,
                    'seed': random.getstate()[1][0] if hasattr(random.getstate()[1], '__getitem__') else None,
                    'vehicle_type': args.vehicle_type,
                    'acceptance_function': eval_func,
                    'borough': args.borough,
                    'year': args.year,
                    'month': args.month,
                    'day': day_num,
                    'training_id': training_id,
                    'total_scenarios': len(day_scenarios)
                },
                'experiment_setup': {
                    'methods': args.methods,
                    'hour_start': args.hour_start,
                    'hour_end': args.hour_end,
                    'time_interval': args.time_interval,
                    'time_unit': args.time_unit,
                    'parallel_workers': args.parallel,
                    'production_mode': args.production,
                    'execution_date': datetime.now().strftime('%Y%m%d_%H%M%S'),
                    'num_eval': args.num_eval  # Number of Monte Carlo simulations per scenario
                },
                'aggregated_statistics': {},
                'results_basic_analysis': {}
            }
            
            # Collect all results data for parquet
            results_data = []
            all_data_stats = []
            all_matching_stats = []
            method_results = {}
            
            for scenario_data in day_scenarios:
                lambda_result = scenario_data['lambda_result']
                
                # Extract data from lambda result
                if isinstance(lambda_result, dict) and 'body' in lambda_result:
                    # Parse the body if it's a string
                    body_data = json.loads(lambda_result['body']) if isinstance(lambda_result['body'], str) else lambda_result['body']
                else:
                    body_data = lambda_result
                
                # Extract statistics
                data_statistics = body_data.get('data_statistics', {})
                experiment_metadata = body_data.get('experiment_metadata', {})
                performance_summary = body_data.get('performance_summary', {})
                
                all_data_stats.append(data_statistics)
                
                # Extract individual method results
                method_results_this_scenario = body_data.get('results', [])
                for method_result in method_results_this_scenario:
                    method_name = method_result.get('method_name', 'Unknown')
                    if method_name not in method_results:
                        method_results[method_name] = []
                    
                    # Add scenario context to method result
                    result_row = {
                        'experiment_id': experiment_id,
                        'execution_timestamp': execution_timestamp,
                        'scenario_index': scenario_data['scenario_index'],
                        'scenario_id': scenario_data['scenario_id'],
                        'method_name': method_name,
                        'objective_value': method_result.get('objective_value', 0),
                        'computation_time': method_result.get('computation_time', 0),
                        'success': 'error' not in method_result,
                        'error_message': method_result.get('error', None),
                        'num_requests': method_result.get('num_requests', 0),
                        'num_taxis': method_result.get('num_taxis', 0),
                        'avg_acceptance_rate': method_result.get('avg_acceptance_rate', 0),
                        'vehicle_type': args.vehicle_type,
                        'acceptance_function': eval_func,
                        'borough': args.borough,
                        'year': args.year,
                        'month': args.month,
                        'day': day_num,
                        **{f'method_{k}': v for k, v in method_result.items() 
                           if k not in ['method_name', 'objective_value', 'computation_time', 'error', 'num_requests', 'num_taxis', 'avg_acceptance_rate']}
                    }
                    
                    method_results[method_name].append(result_row)
                    results_data.append(result_row)
            
            # Calculate comprehensive aggregated statistics
            if all_data_stats:
                
                # Extract all numerical data for statistical analysis
                requesters_counts = [d.get('num_requesters', 0) for d in all_data_stats]
                taxis_counts = [d.get('num_taxis', 0) for d in all_data_stats]
                trip_distances = [d.get('avg_trip_distance_km', 0) for d in all_data_stats]
                trip_amounts = [d.get('avg_trip_amount', 0) for d in all_data_stats]
                ratios = [d.get('ratio_requests_to_taxis', 0) for d in all_data_stats]
                
                def safe_stats(data_list):
                    """Calculate comprehensive statistics safely for countable data."""
                    if not data_list or all(x == 0 for x in data_list):
                        return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0, 'sum': 0}
                    return {
                        'mean': float(np.mean(data_list)),
                        'median': float(np.median(data_list)),
                        'std': float(np.std(data_list)),
                        'min': float(np.min(data_list)),
                        'max': float(np.max(data_list)),
                        'sum': float(np.sum(data_list))
                    }
                
                def safe_ratio_stats(data_list):
                    """Calculate statistics for ratios/rates (sum doesn't make sense)."""
                    if not data_list or all(x == 0 for x in data_list):
                        return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
                    return {
                        'mean': float(np.mean(data_list)),
                        'median': float(np.median(data_list)),
                        'std': float(np.std(data_list)),
                        'min': float(np.min(data_list)),
                        'max': float(np.max(data_list))
                    }
                
                experiment_summary['aggregated_statistics'] = {
                    'total_scenarios': len(day_scenarios),
                    'requesters': safe_stats(requesters_counts),
                    'taxis': safe_stats(taxis_counts),
                    'trip_distances_km': safe_stats(trip_distances),
                    'trip_amounts': safe_stats(trip_amounts),
                    'request_taxi_ratios': safe_ratio_stats(ratios),
                    'data_availability': {
                        'scenarios_with_data': sum(1 for d in all_data_stats if d.get('num_requesters', 0) > 0),
                        'scenarios_without_data': sum(1 for d in all_data_stats if d.get('num_requesters', 0) == 0),
                        'data_coverage_percentage': (sum(1 for d in all_data_stats if d.get('num_requesters', 0) > 0) / len(all_data_stats) * 100) if all_data_stats else 0
                    }
                }
                
                # Calculate comprehensive method performance summary
                experiment_summary['results_basic_analysis'] = {
                    'total_methods': len(args.methods),
                    'total_method_executions': len(results_data),
                    'successful_executions': sum(1 for r in results_data if r['success']),
                    'failed_executions': sum(1 for r in results_data if not r['success']),
                    'methods_performance': {}
                }
                
                # Overall objective value analysis across all methods
                all_objectives = [r['objective_value'] for r in results_data if r['success']]
                all_computation_times = [r['computation_time'] for r in results_data if r['success']]
                all_acceptance_rates = [r['avg_acceptance_rate'] for r in results_data if r['success']]
                
                if all_objectives:
                    experiment_summary['results_basic_analysis']['overall_performance'] = {
                        'objective_values': safe_stats(all_objectives),
                        'computation_times': safe_stats(all_computation_times),
                        'acceptance_rates': safe_ratio_stats(all_acceptance_rates)
                    }
                
                # Detailed per-method analysis
                for method_name, method_data in method_results.items():
                    if method_data:
                        objectives = [r['objective_value'] for r in method_data]
                        times = [r['computation_time'] for r in method_data]
                        successes = [r['success'] for r in method_data]
                        acceptance_rates = [r['avg_acceptance_rate'] for r in method_data]
                        num_requests_list = [r['num_requests'] for r in method_data]
                        num_taxis_list = [r['num_taxis'] for r in method_data]
                        
                        # Calculate pairing/matching statistics
                        matching_ratios = []
                        efficiency_ratios = []
                        for r in method_data:
                            if r['num_requests'] > 0 and r['num_taxis'] > 0:
                                max_possible_matches = min(r['num_requests'], r['num_taxis'])
                                estimated_matches = int(r['num_requests'] * r['avg_acceptance_rate']) if r['avg_acceptance_rate'] > 0 else 0
                                estimated_matches = min(estimated_matches, r['num_taxis'])
                                
                                matching_ratio = estimated_matches / r['num_requests'] if r['num_requests'] > 0 else 0
                                efficiency_ratio = estimated_matches / max_possible_matches if max_possible_matches > 0 else 0
                                
                                matching_ratios.append(matching_ratio)
                                efficiency_ratios.append(efficiency_ratio)
                        
                        experiment_summary['results_basic_analysis']['methods_performance'][method_name] = {
                            'objective_values': safe_stats(objectives),
                            'computation_times': safe_stats(times),
                            'acceptance_rates': safe_ratio_stats(acceptance_rates),
                            'matching_performance': {
                                'matching_ratios': safe_ratio_stats(matching_ratios),
                                'efficiency_ratios': safe_ratio_stats(efficiency_ratios),
                                'scenarios_with_matches': sum(1 for r in matching_ratios if r > 0),
                                'scenarios_without_matches': sum(1 for r in matching_ratios if r == 0)
                            },
                            'data_characteristics': {
                                'num_requests': safe_stats(num_requests_list),
                                'num_taxis': safe_stats(num_taxis_list)
                            },
                            'success_rate': sum(successes) / len(successes) if successes else 0,
                            'scenarios_completed': len(method_data),
                            'relative_performance': {
                                'rank_by_objective': 0,  # Will be calculated below
                                'rank_by_efficiency': 0  # Will be calculated below
                            }
                        }
                
                # Calculate relative performance rankings
                method_names = list(experiment_summary['results_basic_analysis']['methods_performance'].keys())
                if len(method_names) > 1:
                    # Rank by average objective value
                    obj_rankings = sorted(method_names, 
                                        key=lambda m: experiment_summary['results_basic_analysis']['methods_performance'][m]['objective_values']['mean'], 
                                        reverse=True)
                    for i, method in enumerate(obj_rankings):
                        experiment_summary['results_basic_analysis']['methods_performance'][method]['relative_performance']['rank_by_objective'] = i + 1
                    
                    # Rank by average efficiency ratio
                    eff_rankings = sorted(method_names, 
                                        key=lambda m: experiment_summary['results_basic_analysis']['methods_performance'][m]['matching_performance']['efficiency_ratios']['mean'], 
                                        reverse=True)
                    for i, method in enumerate(eff_rankings):
                        experiment_summary['results_basic_analysis']['methods_performance'][method]['relative_performance']['rank_by_efficiency'] = i + 1
            
            # Save to S3
            try:
                # 1. Save experiment_summary.json
                summary_key = f"{base_path}/experiment_summary.json"
                self.s3_client.put_object(
                    Bucket='magisterka',
                    Key=summary_key,
                    Body=json.dumps(experiment_summary, indent=2, default=str),
                    ContentType='application/json'
                )
                
                # 2. Save results.parquet (if we have results)
                if results_data:
                    results_df = pd.DataFrame(results_data)
                    
                    # Convert DataFrame to parquet bytes
                    parquet_buffer = io.BytesIO()
                    results_df.to_parquet(parquet_buffer, index=False, engine='pyarrow')
                    parquet_bytes = parquet_buffer.getvalue()
                    
                    # Upload results.parquet
                    results_key = f"{base_path}/results.parquet"
                    self.s3_client.put_object(
                        Bucket='magisterka',
                        Key=results_key,
                        Body=parquet_bytes,
                        ContentType='application/octet-stream'
                    )
                
                s3_location = f"s3://magisterka/{base_path}/"
                saved_paths[f"day{day_num:02d}_{eval_func}"] = s3_location
                
                # Mark as saved in cache
                self.mark_daily_save_complete(day_num, eval_func, args, s3_location)
                
                print(f"   ✅ Saved Day {day_num} {eval_func}: {s3_location}")
                print(f"      📁 Files: experiment_summary.json, results.parquet")
                print(f"      🔢 Experiment ID: {experiment_id}")
                print(f"      📊 Total scenarios: {len(day_scenarios)}, Total method executions: {len(results_data)}")
                
            except Exception as e:
                print(f"   ❌ Failed to save Day {day_num} {eval_func}: {e}")
        
        return saved_paths

def create_scenario_parameters(args, training_id: str) -> List[Dict[str, Any]]:
    """Create scenario parameters with Hikima-consistent time windows"""
    scenarios = []
    execution_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Hikima time window configuration
    hour_start = args.hour_start
    hour_end = args.hour_end
    time_interval = args.time_interval
    time_unit = args.time_unit
    
    # Convert to minutes for calculation
    if time_unit == 's':
        interval_minutes = time_interval / 60.0
    else:  # 'm'
        interval_minutes = time_interval
    
    total_minutes = (hour_end - hour_start) * 60
    num_intervals = int(total_minutes / interval_minutes)
    
    for day in args.days:
        for eval_func in args.eval:
            for scenario_idx in range(num_intervals):
                # Calculate current time within the Hikima window
                minutes_elapsed = scenario_idx * interval_minutes
                current_hour = hour_start + int(minutes_elapsed // 60)
                current_minute = int(minutes_elapsed % 60)
                
                scenario_id = f"day{day:02d}_{eval_func}_s{scenario_idx:03d}"
                
                scenario = {
                    'scenario_id': scenario_id,
                    'year': args.year,
                    'month': args.month,
                    'day': day,
                    'borough': args.borough,
                    'vehicle_type': args.vehicle_type,
                    'acceptance_function': eval_func,
                    'scenario_index': scenario_idx,
                    'time_window': {
                        'hour_start': hour_start,
                        'hour_end': hour_end,
                        'minute_start': 0,
                        'time_interval': time_interval,
                        'time_unit': time_unit,
                        'current_hour': current_hour,
                        'current_minute': current_minute,
                        'current_second': 0
                    },
                    'methods': args.methods,
                    'training_id': training_id,
                    'execution_date': execution_date,
                    'skip_s3_save': True,  # Skip individual saves, aggregate at end
                    'num_eval': args.num_eval  # Configurable number of Monte Carlo simulations
                }
                scenarios.append(scenario)
    
    return scenarios

def validate_hikima_consistency(args):
    """Validate that parameters are consistent with Hikima methodology"""
    warnings = []
    
    # Check standard Hikima time window
    if args.hour_start != 10 or args.hour_end != 20:
        warnings.append(f"⚠️ Non-standard time window: {args.hour_start}:00-{args.hour_end}:00 (Hikima standard: 10:00-20:00)")
    
    # Check interval consistency
    if args.time_unit == 'm' and args.time_interval not in [5, 10, 15, 30]:
        warnings.append(f"⚠️ Unusual minute interval: {args.time_interval}m (common: 5m, 10m, 15m, 30m)")
    
    if args.time_unit == 's' and args.time_interval not in [30, 60, 120, 300]:
        warnings.append(f"⚠️ Unusual second interval: {args.time_interval}s (common: 30s, 60s, 120s, 300s)")
    
    # Check borough-specific recommendations
    if args.borough == 'Manhattan' and args.time_unit == 'm' and args.time_interval > 5:
        warnings.append(f"⚠️ Manhattan typically uses ≤5m intervals (current: {args.time_interval}m)")
    
    # Calculate scenarios per day
    if args.time_unit == 's':
        interval_minutes = args.time_interval / 60.0
    else:
        interval_minutes = args.time_interval
    
    scenarios_per_day = int((args.hour_end - args.hour_start) * 60 / interval_minutes)
    
    if scenarios_per_day > 1200:  # More than 30s intervals over 10 hours
        warnings.append(f"⚠️ Very high scenario count: {scenarios_per_day}/day (consider larger intervals)")
    
    return warnings, scenarios_per_day

def main():
    parser = argparse.ArgumentParser(
        description='Unified Ride-Hailing Pricing Experiments - Hikima Methodology',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
HIKIMA TIME WINDOW EXAMPLES:

Standard Hikima setup (10:00-20:00, 5min intervals = 120 scenarios/day):
    --hour_start=10 --hour_end=20 --time_interval=5 --time_unit=m

Manhattan intensive (10:00-20:00, 30sec intervals = 1200 scenarios/day):
    --hour_start=10 --hour_end=20 --time_interval=30 --time_unit=s

Custom business hours (09:00-17:00, 10min intervals = 48 scenarios/day):
    --hour_start=9 --hour_end=17 --time_interval=10 --time_unit=m
        """
    )
    
    # Core experiment parameters
    parser.add_argument('--year', type=int, required=True, help='Year (e.g. 2019)')
    parser.add_argument('--month', type=int, required=True, help='Month (1-12)')
    parser.add_argument('--borough', required=True, choices=['Manhattan', 'Bronx', 'Queens', 'Brooklyn'], 
                       help='Borough name')
    parser.add_argument('--vehicle_type', required=True, choices=['yellow', 'green', 'fhv'], 
                       help='Vehicle type')
    parser.add_argument('--eval', required=True, help='Acceptance functions (comma-separated): PL,Sigmoid')
    parser.add_argument('--methods', required=True, 
                       help='Pricing methods (comma-separated): LP,MinMaxCostFlow,LinUCB,MAPS')
    
    # Day specification (multiple options)
    day_group = parser.add_mutually_exclusive_group(required=True)
    day_group.add_argument('--day', type=int, help='Single day (1-31)')
    day_group.add_argument('--days', help='Multiple days (comma-separated): 1,6,10')
    day_group.add_argument('--start_day', type=int, help='Start day for range (use with --end_day)')
    day_group.add_argument('--days_modulo', help='Days by modulo (DIVISOR,REMAINDER): 5,1 = days 1,6,11,16,21,26,31')
    parser.add_argument('--end_day', type=int, help='End day for range (use with --start_day)')
    parser.add_argument('--total_days', type=int, default=31, help='Total days in month for modulo calculation (default: 31)')
    
    # HIKIMA TIME WINDOW PARAMETERS
    hikima_group = parser.add_argument_group('Hikima Time Window Configuration')
    hikima_group.add_argument('--hour_start', type=int, default=10, 
                             help='Start hour - Hikima standard: 10 (default: 10)')
    hikima_group.add_argument('--hour_end', type=int, default=20, 
                             help='End hour - Hikima standard: 20 (default: 20)')
    hikima_group.add_argument('--time_interval', type=int, default=5, 
                             help='Time interval value - Hikima standard: 5 (default: 5)')
    hikima_group.add_argument('--time_unit', choices=['m', 's'], default='m', 
                             help='Time unit: m=minutes, s=seconds - Hikima standard: m (default: m)')
    
    # LinUCB training options
    parser.add_argument('--skip_training', action='store_true', help='Skip LinUCB training (use pre-trained models)')
    parser.add_argument('--force_training', action='store_true', help='Force LinUCB retraining')
    
    # Hikima simulation parameters
    parser.add_argument('--num_eval', type=int, default=1000, 
                       help='Number of Monte Carlo simulations (default: 1000, Hikima standard)')
    
    # Execution options
    parser.add_argument('--parallel', type=int, default=3, help='Parallel workers (default: 3)')
    parser.add_argument('--timeout', type=int, default=0, help='Maximum experiment duration in seconds (0=no timeout, default: 0)')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for Lambda processing (1-10, default: 5)')
    parser.add_argument('--no_batch', action='store_true', help='Disable batch processing (use single scenario mode)')
    parser.add_argument('--max_lambda_concurrency', type=int, default=700,
                       help='Maximum concurrent Lambda invocations (default: 700, max: 700). Controls AWS concurrency limit to prevent hanging.')
    
    # Mode and output options
    parser.add_argument('--production', action='store_true', help='Production mode (minimal logging)')
    parser.add_argument('--debug', action='store_true', help='Debug mode (verbose output)')
    parser.add_argument('--dry_run', action='store_true', help='Dry run (show what would be executed)')
    
    args = parser.parse_args()
    
    # Process day arguments
    if args.day:
        args.days = [args.day]
    elif args.days:
        args.days = [int(d.strip()) for d in args.days.split(',')]
    elif args.start_day and args.end_day:
        args.days = list(range(args.start_day, args.end_day + 1))
    elif args.days_modulo:
        try:
            divisor, remainder = map(int, args.days_modulo.split(','))
            if remainder >= divisor:
                parser.error(f"Remainder ({remainder}) must be less than divisor ({divisor})")
            
            args.days = []
            for day in range(1, args.total_days + 1):
                if day % divisor == remainder:
                    args.days.append(day)
            
            if not args.days:
                parser.error(f"No days found for modulo {divisor} with remainder {remainder}")
                
            print(f"🎯 Modulo selection: divisor={divisor}, remainder={remainder}")
            print(f"📅 Selected days: {args.days}")
        except ValueError:
            parser.error("Invalid days_modulo format. Use: DIVISOR,REMAINDER (e.g., 5,1)")
    else:
        parser.error("Must specify --day, --days, --start_day/--end_day, or --days_modulo")
    
    # Process eval and methods
    args.eval = [e.strip() for e in args.eval.split(',')]
    args.methods = [m.strip() for m in args.methods.split(',')]
    
    # Validate methods
    valid_methods = ['LP', 'MinMaxCostFlow', 'LinUCB', 'MAPS']
    for method in args.methods:
        if method not in valid_methods:
            parser.error(f"Invalid method: {method}. Valid: {valid_methods}")
    
    # Validate and process batch size
    if args.no_batch:
        batch_size = 1  # Disable batch processing
    else:
        batch_size = max(1, min(args.batch_size, 10))  # Clamp to 1-10
        if batch_size != args.batch_size:
            print(f"🔧 Adjusted batch size from {args.batch_size} to {batch_size} (valid range: 1-10)")
    
    # Adaptive parallel workers based on complexity
    if len(args.methods) >= 3:
        args.parallel = min(args.parallel, 2)  # Cap at 2 for complex scenarios
        print(f"🔧 Complex scenario detected ({len(args.methods)} methods), reducing parallel workers to {args.parallel}")
    
    print("🚀 UNIFIED RIDE-HAILING PRICING EXPERIMENTS")
    print("=" * 70)
    
    # Validate Hikima consistency
    warnings, scenarios_per_day = validate_hikima_consistency(args)
    
    if warnings:
        print("⚠️ HIKIMA CONSISTENCY WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
        print()
    
    # Configuration summary
    print("🎯 EXPERIMENT CONFIGURATION:")
    print(f"   📅 Date(s): {args.year}-{args.month:02d}-{args.days}")
    print(f"   🏙️ Borough: {args.borough}")
    print(f"   🚗 Vehicle: {args.vehicle_type}")
    print(f"   📊 Evaluation: {args.eval}")
    print(f"   🔧 Methods: {args.methods}")
    print(f"   ⚡ Parallel Workers: {args.parallel}")
    print(f"   📦 Batch Processing: {'Enabled' if batch_size > 1 else 'Disabled'} (size={batch_size})")
    print(f"   🎛️ Production Mode: {args.production}")
    print(f"   🎲 Monte Carlo Simulations: {args.num_eval} ({'Hikima standard' if args.num_eval == 1000 else 'custom'})")
    
    # Generate training ID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    training_id = f"exp_{args.vehicle_type}_{args.borough}_{timestamp}"
    print(f"   🆔 Training ID: {training_id}")
    
    # Check LinUCB models if needed
    if 'LinUCB' in args.methods:
        linucb_key = f"models/linucb/{args.vehicle_type}_{args.borough}_201907/trained_model.pkl"
        s3_client = boto3.client('s3', region_name='eu-north-1')
        
        try:
            s3_client.head_object(Bucket='magisterka', Key=linucb_key)
            print(f"✅ LinUCB model ready: s3://magisterka/{linucb_key}")
        except:
            if not args.skip_training:
                print(f"⚠️ LinUCB model missing, training will be needed")
            else:
                print(f"❌ LinUCB model missing and training skipped")
                return 1
    
    # Calculate scenarios with Hikima methodology
    hour_start = args.hour_start
    hour_end = args.hour_end 
    time_interval = args.time_interval
    total_scenarios = len(args.days) * len(args.eval) * scenarios_per_day
    
    print(f"\n📊 HIKIMA TIME WINDOW CONFIGURATION:")
    print(f"   ⏰ Time Window: {hour_start:02d}:00-{hour_end:02d}:00 ({hour_end-hour_start} hours)")
    print(f"   📅 Interval: {time_interval}{args.time_unit} ({'standard' if (time_interval == 5 and args.time_unit == 'm') else 'custom'})")
    print(f"   🔢 Scenarios/Day: {scenarios_per_day}")
    print(f"   📈 Total Scenarios: {total_scenarios}")
    
    # Timeout analysis and recommendations
    print(f"\n🔧 TIMEOUT ANALYSIS:")
    complexity_factor = 1.0
    if len(args.methods) >= 3:
        complexity_factor *= 2.0  # Reduced from 3x to 2x
        print(f"   ⚠️ High method complexity: {len(args.methods)} methods (+2x time)")
    if args.num_eval >= 1000:
        complexity_factor *= (args.num_eval / 500)  # More realistic: 1000 simulations = 2x baseline
        print(f"   ⚠️ High simulation count: {args.num_eval} simulations (+{args.num_eval/500:.1f}x time)")
    
    estimated_scenario_time = 5 * complexity_factor  # Base: 5s per scenario (more realistic)
    print(f"   ⏱️ Estimated per-scenario time: {estimated_scenario_time:.1f}s")
    
    # More aggressive warnings for Lambda timeout (900s = 15 minutes)
    if estimated_scenario_time >= 900:  # 15+ minutes (Lambda max)
        print(f"   🚨 CRITICAL: Scenarios WILL timeout (Lambda max: 15min)!")
        print(f"      💡 REQUIRED fixes:")
        print(f"         • Reduce --num_eval to <500 (current: {args.num_eval})")
        print(f"         • Use 1-2 methods max (current: {len(args.methods)})")
        print(f"         • Or run methods separately")
        print(f"      ⚠️  Current config will fail!")
        
        # Give user a chance to abort
        if not args.production and not args.dry_run:
            try:
                print("\n❓ Continue anyway? This will likely fail and hang. (y/N): ", end='', flush=True)
                response = input()
                if response.lower() not in ['y', 'yes']:
                    print("🛑 Experiment aborted. Please adjust parameters.")
                    return 1
            except (KeyboardInterrupt, EOFError):
                print("\n🛑 Experiment aborted by user.")
                return 1
                
    elif estimated_scenario_time >= 600:  # 10+ minutes
        print(f"   🚨 WARNING: Scenarios may timeout! Consider:")
        print(f"      • Reducing --num_eval (current: {args.num_eval})")
        print(f"      • Using fewer methods (current: {len(args.methods)})")
        print(f"      • Running methods separately")
    
    # Execution plan
    print(f"\n📋 EXECUTION PLAN:")
    print(f"   📊 Total scenarios: {total_scenarios}")
    
    # Time estimation based on method complexity and simulations
    estimated_time = total_scenarios * estimated_scenario_time / args.parallel
    
    print(f"   ⏱️ Estimated time: {estimated_time:.0f}s ({estimated_time/60:.1f} min)")
    print(f"   💰 Estimated cost: ~${total_scenarios * 0.0001:.2f} (Lambda invocations)")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No actual execution")
        return 0
    
    # Create scenarios
    scenarios = create_scenario_parameters(args, training_id)
    
    print(f"\n🚀 STARTING EXPERIMENT EXECUTION...")
    
    # Run experiments with improved runner
    runner = ExperimentRunner(
        parallel_workers=args.parallel,
        production_mode=args.production,
        batch_size=batch_size,
        max_experiment_duration=args.timeout,
        max_lambda_concurrency=args.max_lambda_concurrency
    )
    
    results = runner.run_experiments(scenarios)
    
    # Final summary
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   ✅ Successful: {len(successful_results)}/{total_scenarios}")
    print(f"   ❌ Failed: {len(failed_results)}/{total_scenarios}")
    
    # Aggregate and save results by day
    saved_paths = {}
    if successful_results:
        saved_paths = runner.aggregate_and_save_results(successful_results, args, training_id)
    
    if saved_paths:
        print(f"\n💾 S3 EXPERIMENT RESULTS:")
        for day_eval, s3_path in sorted(saved_paths.items()):
            print(f"   📅 {day_eval}: {s3_path}")
    
    if failed_results:
        print(f"\n❌ FAILED SCENARIOS:")
        for result in failed_results[:10]:  # Show first 10
            error = result['result'].get('error', 'Unknown error')
            print(f"   - {result['scenario_id']}: {error}")
        if len(failed_results) > 10:
            print(f"   ... and {len(failed_results) - 10} more")
    
    success_rate = len(successful_results) / total_scenarios * 100 if total_scenarios > 0 else 0
    print(f"\n📈 SUCCESS RATE: {success_rate:.1f}%")
    
    if success_rate < 90:
        print("⚠️ Low success rate - consider reducing --parallel or checking Lambda limits")
        return 1
    
    print(f"\n🎉 Experiment completed! Check S3 paths above for results.")
    return 0

if __name__ == '__main__':
    sys.exit(main()) 