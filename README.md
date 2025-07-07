# Ride-Hailing Pricing Experiment Framework

Cloud-based system for running pricing experiments on NYC taxi data using Hikima methodology. Supports multiple pricing algorithms (LP, MinMaxCostFlow, LinUCB, MAPS) with configurable time windows and acceptance functions.

## Quick Start

### 1. Data Check
```bash
# Check if data exists for specific month
aws s3 ls s3://magisterka/datasets/yellow/year=2019/month=10/

# Verify LinUCB models are available
aws s3 ls s3://magisterka/models/linucb/yellow_Manhattan_201907/
```

### 2. Single Day Experiment
```bash
# Standard Hikima experiment (120 scenarios: 10:00-20:00, 5min intervals)
python run_pricing_experiment.py \
  --year=2019 --month=10 --day=6 \
  --borough=Manhattan --vehicle_type=yellow \
  --eval=PL,Sigmoid --methods=LP,MinMaxCostFlow,LinUCB,MAPS \
  --parallel=3 --skip_training

# Custom time window (48 scenarios: 8:00-20:00, 15min intervals)  
python run_pricing_experiment.py \
  --year=2019 --month=10 --day=1 \
  --borough=Manhattan --vehicle_type=green \
  --eval=PL --methods=LP \
  --hour_start=8 --hour_end=20 --time_interval=15 --time_unit=m \
  --parallel=2 --skip_training
```

### 3. Multi-Day Parallel Execution
```bash
# Enhanced parallel execution for entire month
./enhanced_parallel_experiments.sh

# This runs:
# - 2 parallel processes handling different days (reduced from 3 due to AWS limits)
# - Smart progress monitoring with 20-minute timeout
# - Daily S3 saves with automatic resume capability
# - Circuit breaker to prevent spam on broken lambdas
# - Full experiment tracking and recovery
```

## Core Features

### **Enhanced Parallel Execution**
- **Multi-Process**: 2 parallel processes handling different days (AWS limit: 400 concurrent Lambdas)
- **Smart Timeout**: Only kills processes with no progress for 20+ minutes
- **Daily Saves**: Automatic S3 upload after each day completes
- **Circuit Breaker**: Prevents spam on broken lambdas (5 failures = halt)
- **Progress Tracking**: Real-time monitoring of batch progress
- **Recovery**: Automatic resume from interruptions

### **Pricing Algorithms**
- **LP**: Gupta-Nagarajan Linear Program optimization
- **MinMaxCostFlow**: Capacity scaling min-cost flow algorithm  
- **LinUCB**: Contextual bandit learning with pre-trained models
- **MAPS**: Area-based pricing with bipartite matching

### **Acceptance Functions**
- **PL**: Piecewise Linear (`acceptance = -2.0/trip_amount * price + 3.0`)
- **Sigmoid**: Sigmoid function with Hikima parameters (`β=1.3`, `γ=0.3*√3/π`)

### **Vehicle Types**
- **yellow**: Yellow taxi data (largest dataset)
- **green**: Green taxi data (outer boroughs)
- **fhv**: For-hire vehicle data

### **Boroughs**
- **Manhattan**: Highest density, uses 30s time intervals in Hikima
- **Bronx/Queens/Brooklyn**: Lower density, uses 300s time intervals in Hikima

## Configuration Options

### **Day Selection**
```bash
# Single day
--day=6

# Multiple specific days
--days=1,6,11,16,21,26,31

# Day range
--start_day=1 --end_day=7

# Modulo pattern (every 5th day starting from day 1)
--days_modulo=5,1 --total_days=31
```

### **Time Window Configuration**

#### **Standard Hikima**
```bash
--hour_start=10 --hour_end=20 --time_interval=5 --time_unit=m
# Result: 120 scenarios (10 hours × 12 intervals/hour)
```

#### **High-Frequency Analysis**
```bash
--hour_start=14 --hour_end=16 --time_interval=30 --time_unit=s  
# Result: 240 scenarios (2 hours × 120 intervals/hour)
```

#### **Custom Research**
```bash
--hour_start=8 --hour_end=22 --time_interval=15 --time_unit=m
# Result: 56 scenarios (14 hours × 4 intervals/hour)
```

## Results Structure

Results are saved to S3 with day-level aggregation:
```
s3://magisterka/experiments/
  type={vehicle_type}/
    eval={acceptance_function}/
      borough={borough}/
        year={year}/month={month}/day={day}/
          {execution_date}_{training_id}.json
```

Each file contains:
- **scenarios**: Individual scenario results with time windows
- **day_statistics**: Aggregated statistics across all scenarios  
- **method_performance_summary**: Algorithm performance metrics

## Performance

- **Cloud Execution**: AWS Lambda with 10GB RAM, 8+ virtual cores
- **Parallel Processing**: Up to 50 concurrent Lambda executions
- **Speed**: 1-7 scenarios/second depending on algorithm complexity
- **Cost**: ~$0.01-0.05 per day experiment
- **Reliability**: Smart timeout and circuit breaker prevent hanging

## Monitoring

The enhanced parallel execution provides comprehensive monitoring:
- **Real-time Progress**: Batch completion tracking
- **Health Checks**: Memory, disk, and process monitoring
- **Timeout Management**: Progress-based timeout (not time-based)
- **Error Classification**: Rate limit detection and handling
- **Recovery**: Automatic resume from last completed day

## See Also

- **TECHNICAL.md**: Detailed setup, validation, and troubleshooting guide 