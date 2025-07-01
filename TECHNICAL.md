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

#### **Data Validation Script**
```bash
# Quick validation of data integrity
python -c "
import boto3
s3 = boto3.client('s3')

# Check critical files
files = [
    'datasets/yellow/year=2019/month=10/yellow_tripdata_2019-10.parquet',
    'datasets/area_information.csv',
    'models/linucb/yellow_Manhattan_201907/trained_model.pkl'
]

for file in files:
    try:
        s3.head_object(Bucket='magisterka', Key=file)
        print(f'✅ {file}')
    except:
        print(f'❌ {file} - MISSING')
"
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

#### **Data Processing Pipeline**
```bash
# Automated download script for multiple months
for month in {07..10}; do
    echo "Processing 2019-${month}"
    
    # Download
    curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-${month}.parquet
    curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2019-${month}.parquet
    
    # Upload to S3 with partitioning
    aws s3 cp yellow_tripdata_2019-${month}.parquet s3://magisterka/datasets/yellow/year=2019/month=${month}/
    aws s3 cp green_tripdata_2019-${month}.parquet s3://magisterka/datasets/green/year=2019/month=${month}/
    
    # Cleanup
    rm *_tripdata_2019-${month}.parquet
done
```

#### **LinUCB Model Preparation**
```bash
# Generate pre-trained models (optional, speeds up experiments)
python prepare_hikima_matrices.py

# This creates models for:
# - 3 vehicle types × 4 boroughs × 3 months = 36 combinations
# - Stored in s3://magisterka/models/linucb/{vehicle}_{borough}_{period}/
```

### 3. Run Experiment

#### **Single Method Testing**
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
gut
#### **Full Algorithm Comparison**
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

#### **Custom Time Windows**
```bash
# High-frequency analysis (30-second intervals)
python run_pricing_experiment.py \
  --year=2019 --month=10 --day=1 \
  --borough=Manhattan --vehicle_type=green \
  --eval=PL --methods=LP \
  --hour_start=14 --hour_end=16 --time_interval=30 --time_unit=s \
  --parallel=3 --skip_training

# Result: 240 scenarios (2 hours × 120 per hour)

# Low-frequency analysis (15-minute intervals)
python run_pricing_experiment.py \
  --year=2019 --month=10 --day=1 \
  --borough=Queens --vehicle_type=yellow \
  --eval=Sigmoid --methods=MAPS \
  --hour_start=8 --hour_end=20 --time_interval=15 --time_unit=m \
  --parallel=3 --skip_training

# Result: 48 scenarios (12 hours × 4 per hour)
```

#### **Production Batch Processing**
```bash
# Process multiple days efficiently
for day in {1..7}; do
    echo "Processing day ${day}"
    python run_pricing_experiment.py \
      --year=2019 --month=10 --day=${day} \
      --borough=Manhattan --vehicle_type=yellow \
      --eval=PL,Sigmoid --methods=LP,LinUCB \
      --parallel=3 --skip_training \
      --production
done
```

## Result Validation

#### **Check S3 Output**
```bash
# Verify experiment results were saved
aws s3 ls s3://magisterka/experiments/type=yellow/eval=PL/borough=Manhattan/year=2019/month=10/day=01/

# Download and inspect results
aws s3 cp s3://magisterka/experiments/type=yellow/eval=PL/borough=Manhattan/year=2019/month=10/day=01/20250627_experiment.json ./

# Quick results summary
python -c "
import json
with open('20250627_experiment.json') as f:
    data = json.load(f)
    
print(f'Scenarios: {len(data["scenarios"])}')
print(f'Methods: {list(data["method_performance_summary"].keys())}')

for method, perf in data['method_performance_summary'].items():
    obj_val = perf['objective_value']['mean']
    success_rate = perf['success_rate']
    print(f'{method}: {obj_val:.2f} (success: {success_rate:.1%})')
"
```

#### **Performance Monitoring**
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

## Troubleshooting

#### **Common Issues**

**1. Lambda Timeout**
```bash
# Increase timeout for complex scenarios
aws lambda update-function-configuration \
  --function-name rideshare-pricing-benchmark \
  --timeout 900
```

**2. Memory Errors**
```bash
# Increase memory allocation
aws lambda update-function-configuration \
  --function-name rideshare-pricing-benchmark \
  --memory-size 10240
```

**3. Data Not Found**
```bash
# Check exact S3 paths
aws s3 ls s3://magisterka/datasets/ --recursive | grep "2019-10"

# Verify file permissions
aws s3api head-object --bucket magisterka --key datasets/yellow/year=2019/month=10/yellow_tripdata_2019-10.parquet
```

**4. LinUCB Model Missing**
```bash
# Check if pre-trained model exists
aws s3 ls s3://magisterka/models/linucb/yellow_Manhattan_201907/

# Force training if needed (takes 10-20 minutes)
python run_pricing_experiment.py \
  --year=2019 --month=10 --day=1 \
  --borough=Manhattan --vehicle_type=yellow \
  --eval=PL --methods=LinUCB \
  --force_training
```

#### **Performance Optimization**

**1. Parallel Execution Tuning**
```bash
# Adjust based on AWS limits and data complexity
--parallel=10   # Conservative (simple scenarios)
--parallel=25   # Aggressive (complex scenarios)
--parallel=50   # Maximum (simple scenarios, good data)
```

**2. Resource Allocation**
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

#### **Data Quality Issues**

**1. Empty Scenarios**
- Common in low-demand periods (early morning, late night)
- Expected behavior, results in 0 objective value
- Consider adjusting time windows for analysis periods

**2. Large Datasets**
- May cause memory issues for some boroughs/periods
- Use smaller time windows or filter by geographic area
- Consider data sampling for exploratory analysis

**3. Model Performance**
- LinUCB requires sufficient training data
- MAPS sensitive to area granularity
- MinMaxCostFlow may converge slowly with large instances

## System Limits

- **Lambda Execution**: 15 minutes maximum per scenario
- **Memory**: 10GB maximum per Lambda instance  
- **Concurrent Executions**: 1000 default AWS limit
- **S3 Storage**: Unlimited (pay per use)
- **Scenario Complexity**: Tested up to 500 requests, 300 taxis per scenario 