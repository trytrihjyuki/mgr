# 🚗 Ride-Hailing Pricing Benchmark Framework

A systematic benchmarking framework for comparing 4 pricing methods in ride-hailing platforms using real NYC taxi data. This framework is designed to be **parameter-agnostic** with **no hardcoded values**, supporting both Hikima experimental replication and extended analysis.

## 🎯 **Core Features**

- **4 Pricing Methods**: MinMaxCostFlow (Hikima), MAPS, LinUCB, and Linear Program (Gupta-Nagarajan)
- **No Hardcoded Parameters**: All values configurable via JSON config or CLI arguments
- **AWS Lambda Container Support**: Handles large dependencies (PuLP, NetworkX, SciPy)
- **Proper S3 Organization**: Results stored with training IDs for grouping
- **Flexible Experiments**: Single day to multi-month analysis
- **Hikima Compliance**: Exact replication of Hikima et al. experimental setup

## 🏗️ **Architecture**

```
├── 🚀 run_experiment.py              # Main CLI interface
├── 📋 configs/experiment_config.json # Configuration (no hardcoded values)
├── ⚡ lambdas/pricing-benchmark/     # AWS Lambda function (container-based)
│   ├── lambda_function.py           # Main benchmark logic
│   ├── Dockerfile                   # Container for large dependencies
│   ├── requirements.txt             # Scientific packages
│   └── deploy.sh                    # Deployment script
└── 📊 Results stored in S3          # experiments/type=*/eval=*/year=*/month=*/{training_id}.json
```

## 🚀 **Quick Start**

### 1. **Deploy the System**
```bash
# Deploy Lambda function with container support
cd lambdas/pricing-benchmark
./deploy.sh
```

### 2. **Upload Configuration**
```bash
# Upload experiment configuration to S3
aws s3 cp configs/experiment_config.json s3://magisterka/configs/
```

### 3. **Run Experiments**

#### **Hikima Replication (2 days, business hours)**
```bash
python run_experiment.py \
  --year=2019 --month=10 --days=1,6 \
  --func=PL,Sigmoid \
  --methods=MinMaxCostFlow,MAPS,LinUCB
```

#### **Comprehensive Analysis (All 4 methods)**
```bash
python run_experiment.py \
  --year=2019 --month=10 --days=1 \
  --func=PL \
  --methods=MinMaxCostFlow,MAPS,LinUCB,LP
```

#### **Extended Multi-day Analysis**
```bash
python run_experiment.py \
  --year=2019 --month=10 --days=1,2,3,4,5,6,7 \
  --func=PL,Sigmoid \
  --methods=LP,MAPS,LinUCB
```

#### **Multi-month Experiments**
```bash
python run_experiment.py \
  --year=2019 --months=3,4,5 --days=1,15 \
  --func=PL \
  --methods=LP,MAPS
```

## 📊 **Pricing Methods**

### 1. **MinMaxCostFlow** (Hikima et al.)
- **Source**: Extracted from the provided Hikima source code
- **Algorithm**: Min-cost flow with delta-scaling
- **Acceptance Functions**: PL (Piecewise Linear), Sigmoid
- **Performance**: High computational complexity, optimal theoretical results

### 2. **MAPS** (Area-based Pricing)
- **Algorithm**: Bipartite matching with area-based pricing
- **Strategy**: Price based on trip amount percentiles and geographic zones
- **Performance**: Medium complexity, practical results

### 3. **LinUCB** (Contextual Bandit)
- **Algorithm**: Linear Upper Confidence Bound with contextual features
- **Strategy**: Distance-based price multipliers with learning
- **Performance**: Low complexity, adaptive pricing

### 4. **LP** (Gupta-Nagarajan Linear Program)
- **Algorithm**: Exact LP formulation using PuLP
- **Implementation**: Point-to-point implementation of the provided theory
- **Performance**: Medium complexity, theoretical optimum

## 🔧 **Configuration**

All parameters are configurable in `configs/experiment_config.json`:

### **Sampling Strategy**
```json
{
  "sampling_strategy": {
    "method": "configurable",
    "hikima_max_records": 8000,  # Exact Hikima replication
    "default_max_records": null,  # No limit for extended analysis
    "random_seed": 42
  }
}
```

### **Acceptance Functions**
```json
{
  "acceptance_functions": {
    "PL": {
      "formula": "-2.0/trip_amount * price + 3.0",
      "parameters": {"c_multiplier": 2.0, "d_constant": 3.0}
    },
    "Sigmoid": {
      "formula": "1 - 1/(1 + exp((-price + beta*trip_amount)/(gamma*trip_amount)))",
      "parameters": {"beta": 1.3, "gamma": 0.16539880833293433}
    }
  }
}
```

### **Pricing Method Parameters**
```json
{
  "pricing_methods": {
    "MinMaxCostFlow": {
      "parameters": {"epsilon": 1e-10, "alpha": 18.0, "s_taxi": 25.0}
    },
    "LP": {
      "parameters": {
        "min_price_factor": 0.5,
        "max_price_factor": 2.0,
        "price_grid_size": 10,
        "solver_timeout": 300
      }
    }
  }
}
```

## 📁 **S3 Data Organization**

### **Input Data**
```
s3://magisterka/datasets/
├── green/year=2019/month=10/green_tripdata_2019-10.parquet
├── yellow/year=2019/month=10/yellow_tripdata_2019-10.parquet
└── fhv/year=2015/month=01/fhv_tripdata_2015-01.parquet
```

### **Results Output**
```
s3://magisterka/experiments/
├── type=green/eval=pl/year=2019/month=10/{training_id}.json
├── type=green/eval=sigmoid/year=2019/month=10/{training_id}.json
└── type=yellow/eval=pl/year=2019/month=11/{training_id}.json
```

Each experiment file contains:
- **Training ID**: 9-digit unique identifier for grouping experiments
- **Configuration**: All experiment parameters
- **Results**: Performance metrics for each pricing method
- **Metadata**: Timestamps, data sizes, execution times

## 🧪 **Experiment Scenarios**

### **Hikima Replication**
- **Purpose**: Exact replication of Hikima et al. experimental setup
- **Time Range**: Business hours (10:00-20:00)
- **Sample Size**: 8000 records (as in original)
- **Methods**: MinMaxCostFlow, MAPS, LinUCB
- **Acceptance Functions**: PL, Sigmoid

### **Comprehensive Benchmark**
- **Purpose**: All 4 methods comparison
- **Time Range**: Configurable
- **Sample Size**: No limit (full dataset)
- **Methods**: All 4 methods
- **Acceptance Functions**: Both PL and Sigmoid

### **Scalability Testing**
- **Purpose**: Large-scale experiments
- **Time Range**: Full day (00:00-24:00)
- **Sample Size**: No limit
- **Methods**: Excludes MinMaxCostFlow (computationally intensive)

## 🔍 **Results Analysis**

### **Finding Results**
```bash
# List all experiments for a training session
aws s3 ls s3://magisterka/experiments/type=green/eval=pl/year=2019/month=10/ --recursive

# Download specific results
aws s3 cp s3://magisterka/experiments/type=green/eval=pl/year=2019/month=10/123456789.json results.json

# Search by pattern
aws s3 ls s3://magisterka/experiments/ --recursive | grep "123456789"
```

### **Result Structure**
```json
{
  "training_id": "123456789",
  "timestamp": "2024-01-15T10:30:00",
  "configuration": {
    "vehicle_type": "green",
    "acceptance_function": "PL",
    "methods_tested": ["MinMaxCostFlow", "MAPS", "LinUCB", "LP"]
  },
  "results": [
    {
      "method_name": "LP",
      "objective_value": 1250.75,
      "computation_time": 15.3,
      "average_price": 8.45,
      "match_rate": 0.87
    }
  ]
}
```

## 📈 **Performance Metrics**

- **Objective Value**: Total expected profit from pricing
- **Computation Time**: Algorithm execution time
- **Match Rate**: Percentage of requesters matched to taxis
- **Average Price**: Mean price across all requests
- **Acceptance Rate**: Percentage of prices accepted by customers

## 🔬 **Advanced Usage**

### **Custom Scenarios**
```bash
# Full day analysis
python run_experiment.py \
  --year=2019 --month=10 --days=1 \
  --func=PL --methods=LP,MAPS \
  --time-range=full_day

# Multi-borough comparison
python run_experiment.py \
  --year=2019 --month=10 --days=1 \
  --func=PL --methods=LP \
  --borough=Brooklyn

# High-frequency sampling
python run_experiment.py \
  --year=2019 --month=10 --days=1,2,3,4,5,6,7,8,9,10 \
  --func=PL,Sigmoid --methods=LP,MAPS,LinUCB
```

### **Direct Lambda Invocation**
```bash
aws lambda invoke \
  --function-name rideshare-pricing-benchmark \
  --payload '{
    "training_id": "123456789",
    "year": 2019,
    "month": 10,
    "day": 1,
    "vehicle_type": "green",
    "borough": "Manhattan",
    "acceptance_function": "PL",
    "methods": ["MinMaxCostFlow", "MAPS", "LinUCB", "LP"],
    "scenario": "comprehensive"
  }' \
  output.json
```

## 🛠️ **Development**

### **Adding New Pricing Methods**
1. Extend `BasePricingMethod` class
2. Implement `calculate_prices` method
3. Add configuration in `experiment_config.json`
4. Update CLI options

### **Modifying Acceptance Functions**
1. Add function definition in config
2. Update `_calculate_acceptance_probability` method
3. Test with existing methods

### **Custom Data Sources**
1. Update S3 paths in configuration
2. Modify data loading logic in `load_data_from_s3`
3. Ensure column naming compatibility

## 🚨 **Important Notes**

- **Training IDs**: Each experiment run generates a unique 9-digit training ID for result grouping
- **Container Images**: Lambda uses container images for large dependencies (PuLP, SciPy)
- **Timeout Management**: Lambda functions have 15-minute timeout; large experiments may need splitting
- **Data Sampling**: Configurable sampling strategy for performance vs. accuracy trade-offs
- **No Hardcoding**: All parameters configurable; no hardcoded rush hours or Hikima-specific setups

## 📚 **Research Applications**

### **Temporal Analysis**
```bash
# Seasonal patterns (multiple months)
python run_experiment.py --year=2019 --months=3,6,9,12 --days=1,15 --func=PL --methods=LP,MAPS

# Weekly patterns (7 consecutive days)
python run_experiment.py --year=2019 --month=10 --days=1,2,3,4,5,6,7 --func=PL,Sigmoid --methods=LP,MAPS,LinUCB
```

### **Comparative Studies**
```bash
# Method comparison (all 4 methods)
python run_experiment.py --year=2019 --month=10 --days=1 --func=PL --methods=MinMaxCostFlow,MAPS,LinUCB,LP

# Acceptance function comparison
python run_experiment.py --year=2019 --month=10 --days=1 --func=PL,Sigmoid --methods=LP
```

### **Scalability Testing**
```bash
# Large-scale analysis (avoid MinMaxCostFlow for performance)
python run_experiment.py --year=2019 --month=10 --days=1,2,3,4,5 --func=PL --methods=LP,MAPS,LinUCB --time-range=full_day
```

---

**🏆 This framework provides a clean, systematic approach to ride-hailing pricing research with real NYC data, supporting both exact Hikima replication and extended analysis!** 