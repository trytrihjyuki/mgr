# Technical Guide

Detailed technical documentation for data operations, experiment execution, and system validation.

## System Setup

### Prerequisites
```bash
# AWS CLI configured with appropriate permissions
aws configure

# Python dependencies 
pip install boto3 pandas numpy pulp networkx

# Docker (for Lambda deployment)
docker --version
```

### Lambda Deployment
```bash
cd lambdas/pricing-benchmark
./deploy.sh
```

## Enhanced Parallel Execution

### Overview
The enhanced parallel execution system (`enhanced_parallel_experiments.sh`) provides:
- **Multi-Process**: 3 parallel processes handling different days
- **Smart Timeout**: Progress-based timeout (not time-based)
- **Daily Saves**: Automatic S3 upload after each day
- **Circuit Breaker**: Prevents spam on broken lambdas
- **Recovery**: Resume from interruptions

### Configuration
```bash
# Edit script variables at top of enhanced_parallel_experiments.sh
YEAR=2019
MONTH=10
TOTAL_DAYS=31
BOROUGH="Manhattan"
VEHICLE_TYPE="yellow"
METHODS="LP,MinMaxCostFlow,LinUCB,MAPS"
ACCEPTANCE_FUNC="PL,Sigmoid"
PARALLEL_WORKERS=1
NUM_EVAL=20
```

### Execution
```bash
# Run enhanced parallel execution
./enhanced_parallel_experiments.sh

# Monitor progress in real-time
tail -f parallel_experiments_*/logs/master.log

# Check individual process logs
tail -f parallel_experiments_*/logs/process_1.log
```

### Monitoring Features
- **Progress Tracking**: Real-time batch completion monitoring
- **Health Checks**: Memory, disk, and process monitoring
- **Smart Timeout**: Only kills processes with no progress for 20+ minutes
- **Circuit Breaker**: Stops after 5 consecutive failures
- **Process Coordination**: Staggered starts to prevent AWS rate limiting

## Data Operations

### 1. Data Check

#### **Verify Data Availability**
```bash
# Check if parquet files exist for target period
aws s3 ls s3://magisterka/datasets/yellow/year=2019/month=10/
aws s3 ls s3://magisterka/datasets/green/year=2019/month=10/ 
aws s3 ls s3://magisterka/datasets/fhv/year=2019/month=10/

# Expected output:
# 2024-06-26 18:15:32  25847392 yellow_tripdata_2019-10.parquet
```

#### **Validate Area Information**
```bash
# Check area mapping file exists
aws s3 ls s3://magisterka/datasets/area_information.csv

# Sample area data structure:
# LocationID,Zone,Borough,latitude,longitude,service_zone
# 1,Newark Airport,EWR,40.6895,-74.1745,EWR
```

#### **Check LinUCB Models**
```bash
# Verify pre-trained models exist
aws s3 ls s3://magisterka/models/linucb/yellow_Manhattan_201907/
aws s3 ls s3://magisterka/models/linucb/green_Bronx_201907/

# Expected files:
# trained_model.pkl (contains A matrices, b vectors, feature dimensions)
```

### 2. Data Download

#### **TLC Data Sources**
Original NYC Taxi & Limousine Commission data:
```bash
# Yellow taxi data (2019-10 example)
curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-10.parquet

# Green taxi data  
curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2019-10.parquet

# FHV data
curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/fhv_tripdata_2019-10.parquet
```

#### **Upload to S3**
```bash
# Upload with correct structure
aws s3 cp yellow_tripdata_2019-10.parquet s3://magisterka/datasets/yellow/year=2019/month=10/
aws s3 cp green_tripdata_2019-10.parquet s3://magisterka/datasets/green/year=2019/month=10/
aws s3 cp fhv_tripdata_2019-10.parquet s3://magisterka/datasets/fhv/year=2019/month=10/
```

### 3. LinUCB Model Preparation
```bash
# Generate pre-trained models (optional, speeds up experiments)
python prepare_hikima_matrices.py

# This creates models for:
# - 3 vehicle types × 4 boroughs × 3 months = 36 combinations
# - Stored in s3://magisterka/models/linucb/{vehicle}_{borough}_{period}/
```

## Single Day Experiments

### **Single Method Testing**
```bash
# Test LP method only (fastest validation)
python run_pricing_experiment.py \
  --year=2019 --month=10 --day=1 \
  --borough=Manhattan --vehicle_type=yellow \
  --eval=PL --methods=LP \
  --parallel=3 --skip_training

# Expected output:
# 📊 Hikima Configuration: 10:00-20:00, 5min intervals = 120 scenarios/day
# ✅ Success: 120 (100.0%) ⚡ Rate: 2.0 scenarios/second
```

### **Full Algorithm Comparison**
```bash
# Run all 4 methods with both evaluation functions
python run_pricing_experiment.py \
  --year=2019 --month=10 --day=6 \
  --borough=Manhattan --vehicle_type=yellow \
  --eval=PL,Sigmoid --methods=LP,MinMaxCostFlow,LinUCB,MAPS \
  --parallel=2 --skip_training

# Performance expectations:
# - LP: ~3-5 scenarios/second
# - MinMaxCostFlow: ~1-2 scenarios/second  
# - LinUCB: ~2-3 scenarios/second
# - MAPS: ~1-2 scenarios/second
```

### **Custom Time Windows**
```bash
# High-frequency analysis (30-second intervals)
python run_pricing_experiment.py \
  --year=2019 --month=10 --day=1 \
  --borough=Manhattan --vehicle_type=green \
  --eval=PL --methods=LP \
  --hour_start=14 --hour_end=16 --time_interval=30 --time_unit=s \
  --parallel=3 --skip_training

# Result: 240 scenarios (2 hours × 120 per hour)
```

## Debugging Common Issues

### **Lambda Invocation Hanging**
If the Python process starts but makes no progress:

1. **Check Lambda Function Status**
```bash
# Verify Lambda function exists and is deployable
aws lambda get-function --function-name rideshare-pricing-benchmark

# Check recent CloudWatch logs
aws logs filter-log-events \
  --log-group-name /aws/lambda/rideshare-pricing-benchmark \
  --start-time $(date -d '1 hour ago' +%s)000
```

2. **Debug Lambda Invocation**
```bash
# Test Lambda function directly
aws lambda invoke \
  --function-name rideshare-pricing-benchmark \
  --payload '{"year": 2019, "month": 10, "day": 1, "borough": "Manhattan", "vehicle_type": "yellow", "acceptance_function": "PL", "methods": ["LP"]}' \
  --cli-binary-format raw-in-base64-out \
  response.json

# Check response
cat response.json
```

3. **Check Data Access**
```bash
# Verify S3 permissions
aws s3 ls s3://magisterka/datasets/yellow/year=2019/month=10/

# Test data download
aws s3 cp s3://magisterka/datasets/yellow/year=2019/month=10/yellow_tripdata_2019-10.parquet ./test_data.parquet
```

### **Progress Detection Issues**
If progress monitoring shows "0 scenarios completed":

1. **Check Log Files**
```bash
# Look for actual progress in log files
grep -E "(scenarios|Batch|completed)" parallel_experiments_*/logs/process_*.log

# Check for error patterns
grep -E "(ERROR|Failed|Timeout)" parallel_experiments_*/logs/process_*.log
```

2. **Verify Progress Extraction**
```bash
# Test progress extraction function manually
log_file="parallel_experiments_*/logs/process_1.log"
if [ -f "$log_file" ]; then
    progress=$(tail -50 "$log_file" | grep -E "(scenarios|Batch.*completed)" | tail -1)
    echo "Latest progress: $progress"
fi
```

### **Circuit Breaker Activation**
If processes are being killed due to circuit breaker:

1. **Check Error Patterns**
```bash
# Look for rate limiting errors
grep -E "(Rate.*limit|Throttling|429)" parallel_experiments_*/logs/process_*.log

# Check for AWS service errors
grep -E "(ServiceException|ClientError)" parallel_experiments_*/logs/process_*.log
```

2. **Adjust Circuit Breaker Settings**
```bash
# Edit enhanced_parallel_experiments.sh
CIRCUIT_BREAKER_THRESHOLD=10  # Increase from 5 to 10
```

## Result Validation

### **Check S3 Output**
```bash
# Verify experiment results were saved
aws s3 ls s3://magisterka/experiments/type=yellow/eval=PL/borough=Manhattan/year=2019/month=10/day=01/

# Download and inspect results
aws s3 cp s3://magisterka/experiments/type=yellow/eval=PL/borough=Manhattan/year=2019/month=10/day=01/$(date +%Y%m%d)_experiment.json ./

# Quick results summary
python -c "
import json
with open('$(date +%Y%m%d)_experiment.json') as f:
    data = json.load(f)
    
print(f'Scenarios: {len(data[\"scenarios\"])}')
print(f'Methods: {list(data[\"method_performance_summary\"].keys())}')

for method, perf in data['method_performance_summary'].items():
    obj_val = perf['objective_value']['mean']
    success_rate = perf['success_rate']
    print(f'{method}: {obj_val:.2f} (success: {success_rate:.1%})')
"
```

### **Performance Monitoring**
```bash
# Monitor Lambda function performance
aws logs filter-log-events --log-group-name /aws/lambda/rideshare-pricing-benchmark \
  --start-time $(date -d '1 hour ago' +%s)000 \
  --filter-pattern "✅"

# Check for errors
aws logs filter-log-events --log-group-name /aws/lambda/rideshare-pricing-benchmark \
  --start-time $(date -d '1 hour ago' +%s)000 \
  --filter-pattern "ERROR"
```

## System Optimization

### **Lambda Configuration**
```bash
# High-performance configuration
aws lambda update-function-configuration \
  --function-name rideshare-pricing-benchmark \
  --memory-size 10240 \
  --timeout 900 \
  --environment Variables='{
    "OMP_NUM_THREADS":"8",
    "OPENBLAS_NUM_THREADS":"8",
    "MKL_NUM_THREADS":"8"
  }'
```

### **Parallel Execution Tuning**
```bash
# Conservative settings (stable)
--parallel=2   # For complex scenarios with 4 methods
--num_eval=20  # Reduced Monte Carlo simulations

# Aggressive settings (faster)
--parallel=5   # For simple scenarios with 1-2 methods
--num_eval=100 # More simulations for better statistics
```

### **Circuit Breaker Configuration**
```bash
# Edit enhanced_parallel_experiments.sh for custom settings
CIRCUIT_BREAKER_THRESHOLD=5      # Failures before stopping
PROGRESS_TIMEOUT=1200            # 20 minutes without progress
STAGGER_DELAY=10                 # Seconds between process starts
```

## System Limits

- **Lambda Execution**: 15 minutes maximum per scenario
- **Memory**: 10GB maximum per Lambda instance  
- **Concurrent Executions**: 400 per account (critical bottleneck)
- **S3 Storage**: Unlimited (pay per use)
- **Scenario Complexity**: Tested up to 500 requests, 300 taxis per scenario
- **Parallel Processes**: 2 maximum recommended (3+ processes hit concurrency limits)
- **Progress Timeout**: 20 minutes without batch progress triggers termination

## Critical AWS Lambda Concurrency Issue

**✅ FIXED: Lambda Concurrency Management System**

**Previous Issue (RESOLVED):**
- Each experiment process submitted ~100+ batches simultaneously  
- 3 processes = ~300+ concurrent Lambda functions
- AWS account limit = 400 concurrent executions total
- **Result**: Processes got stuck waiting for Lambda responses that never came

**✅ NEW SOLUTION: Concurrency Control Parameters**

1. **`--max_lambda_concurrency` parameter** (default: 150)
   - Controls maximum concurrent Lambda invocations per process
   - Prevents hitting AWS account limits
   - Uses semaphore-based throttling in Python code

2. **`MAX_LAMBDA_CONCURRENCY` in shell script** (default: 150)
   - Environment variable for batch experiments
   - Total limit = `NUM_PROCESSES × MAX_LAMBDA_CONCURRENCY`

**Recommended Settings:**
```bash
# Conservative (safest):
NUM_PROCESSES=1 MAX_LAMBDA_CONCURRENCY=200     # Total: 200 concurrent
./enhanced_parallel_experiments.sh

# Standard (recommended):
NUM_PROCESSES=2 MAX_LAMBDA_CONCURRENCY=150     # Total: 300 concurrent  
./enhanced_parallel_experiments.sh

# Aggressive (for testing):
NUM_PROCESSES=2 MAX_LAMBDA_CONCURRENCY=180     # Total: 360 concurrent
./enhanced_parallel_experiments.sh

# Single experiment:
python run_pricing_experiment.py --max_lambda_concurrency=200 [other args]
```

**Key Benefits:**
- ✅ **No more hanging**: Processes can't overwhelm Lambda
- ✅ **Predictable performance**: Progress continues steadily  
- ✅ **Resource control**: Stay within AWS limits
- ✅ **Automatic throttling**: Built-in semaphore protection

**Monitoring:**
```bash
# Check current concurrency usage
grep "Active Lambda invocations" parallel_experiments_*/logs/process_*.log | tail -5

# Monitor progress without hanging
tail -f parallel_experiments_*/logs/master.log
``` 