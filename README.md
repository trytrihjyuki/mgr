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

### 2. Data Download
```bash
# Download parquet files to S3 (if needed)
# Data structure: s3://magisterka/datasets/{vehicle_type}/year={year}/month={month:02d}/
# Example: s3://magisterka/datasets/yellow/year=2019/month=10/yellow_tripdata_2019-10.parquet
```

### 3. Run Experiment
```bash
# Standard Hikima experiment (120 scenarios: 10:00-20:00, 5min intervals)
python run_pricing_experiment_optimized.py \
  --year=2019 --month=10 --day=6 \
  --borough=Manhattan --vehicle_type=yellow \
  --eval=PL,Sigmoid --methods=LP,MinMaxCostFlow,LinUCB,MAPS \
  --parallel=20 --skip_training

# Custom time window (240 scenarios: 10:00-12:00, 30s intervals)  
python run_pricing_experiment_optimized.py \
  --year=2019 --month=10 --day=1 \
  --borough=Manhattan --vehicle_type=green \
  --eval=PL --methods=LP \
  --hour_start=10 --hour_end=12 --time_interval=30 --time_unit=s \
  --parallel=10 --skip_training
```

## Framework Components

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

## Time Window Configuration

### **Standard Hikima**
```bash
--hour_start=10 --hour_end=20 --time_interval=5 --time_unit=m
# Result: 120 scenarios (10 hours × 12 intervals/hour)
```

### **High-Frequency Analysis**
```bash
--hour_start=14 --hour_end=16 --time_interval=30 --time_unit=s  
# Result: 240 scenarios (2 hours × 120 intervals/hour)
```

### **Custom Research**
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

## See Also

- **TECHNICAL.md**: Detailed setup, validation, and troubleshooting guide 