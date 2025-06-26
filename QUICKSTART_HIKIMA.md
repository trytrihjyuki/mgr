# 🚀 Quick Start: Hikima Experimental Replication

This guide provides step-by-step instructions to **exactly replicate** the Hikima et al. experimental setup using the cleaned benchmarking framework.

## 📋 **What This Replicates**

The original Hikima experimental setup:
- **Data**: NYC Green/Yellow taxi data from October 2019
- **Days**: Day 1 and Day 6 (as in original paper)
- **Time Range**: Business hours (10:00-20:00)
- **Sample Size**: Maximum 8000 records per experiment
- **Methods**: MinMaxCostFlow (Hikima), MAPS, LinUCB
- **Acceptance Functions**: Both PL (Piecewise Linear) and Sigmoid
- **Evaluation**: 100 Monte Carlo iterations

## 🛠️ **Prerequisites**

1. **AWS CLI configured** with access to S3 bucket `magisterka`
2. **Docker installed** for Lambda container deployment
3. **Python 3.9+** with required packages

## ⚡ **1-Minute Setup**

```bash
# Clone and setup
git clone <repository>
cd ride-hailing-benchmark

# Setup environment
make setup

# Deploy the system
make deploy-all
```

## 🔬 **Run Hikima Replication**

### **Option 1: Using Make (Recommended)**
```bash
make run-hikima
```

### **Option 2: Using CLI Directly**
```bash
python run_experiment.py \
  --year=2019 --month=10 --days=1,6 \
  --func=PL,Sigmoid \
  --methods=MinMaxCostFlow,MAPS,LinUCB \
  --scenario=hikima_replication
```

### **Option 3: Individual Experiments**
```bash
# Day 1 with PL acceptance function
python run_experiment.py \
  --year=2019 --month=10 --days=1 \
  --func=PL \
  --methods=MinMaxCostFlow,MAPS,LinUCB

# Day 1 with Sigmoid acceptance function
python run_experiment.py \
  --year=2019 --month=10 --days=1 \
  --func=Sigmoid \
  --methods=MinMaxCostFlow,MAPS,LinUCB

# Day 6 with both functions
python run_experiment.py \
  --year=2019 --month=10 --days=6 \
  --func=PL,Sigmoid \
  --methods=MinMaxCostFlow,MAPS,LinUCB
```

## 📊 **Expected Results Structure**

Results will be stored in S3 with the pattern:
```
s3://magisterka/experiments/
├── type=green/eval=pl/year=2019/month=10/{training_id}.json
├── type=green/eval=sigmoid/year=2019/month=10/{training_id}.json
└── type=yellow/eval=pl/year=2019/month=10/{training_id}.json
```

Each result file contains:
```json
{
  "training_id": "123456789",
  "configuration": {
    "year": 2019,
    "month": 10,
    "day": 1,
    "scenario": "hikima_replication",
    "max_sample_size": 8000
  },
  "results": [
    {
      "method_name": "MinMaxCostFlow",
      "objective_value": 1250.75,
      "computation_time": 45.2,
      "average_price": 8.45,
      "match_rate": 0.87
    },
    {
      "method_name": "MAPS",
      "objective_value": 1180.30,
      "computation_time": 12.1,
      "average_price": 7.95,
      "match_rate": 0.83
    },
    {
      "method_name": "LinUCB",
      "objective_value": 1095.60,
      "computation_time": 3.5,
      "average_price": 7.20,
      "match_rate": 0.79
    }
  ]
}
```

## 🔍 **Accessing Results**

### **Find Your Results**
```bash
# List all experiments for your training ID
aws s3 ls s3://magisterka/experiments/ --recursive | grep "123456789"

# Download specific result
aws s3 cp s3://magisterka/experiments/type=green/eval=pl/year=2019/month=10/123456789.json results.json

# View results
cat results.json | jq .
```

### **Compare Multiple Experiments**
```bash
# Download all results for a training session
aws s3 sync s3://magisterka/experiments/type=green/eval=pl/year=2019/month=10/ ./results/ --exclude "*" --include "*123456789*"
```

## 📈 **Performance Comparison**

Expected performance characteristics (approximate):

| Method | Computation Time | Objective Value | Description |
|--------|------------------|-----------------|-------------|
| **MinMaxCostFlow** | 30-60s | Highest | Hikima's optimal algorithm |
| **MAPS** | 10-20s | Medium-High | Area-based practical approach |
| **LinUCB** | 2-5s | Medium | Fast contextual learning |

## 🎯 **Key Differences from Original**

### **✅ Improvements**
- **No Hardcoded Parameters**: All values configurable
- **AWS Lambda Containers**: Handles large dependencies automatically
- **Proper S3 Organization**: Results grouped by training ID
- **Clean Error Handling**: Robust failure recovery
- **Multiple Acceptance Functions**: Both PL and Sigmoid in one run

### **🔄 Maintained Compatibility**
- **Same Sample Size**: 8000 records maximum (configurable)
- **Same Time Range**: Business hours (10:00-20:00)
- **Same Data Source**: NYC taxi data October 2019
- **Same Methods**: MinMaxCostFlow, MAPS, LinUCB
- **Same Evaluation**: Monte Carlo simulation

## 🚨 **Troubleshooting**

### **Lambda Timeout**
If experiments fail due to timeout:
```bash
# Run methods individually
python run_experiment.py --year=2019 --month=10 --days=1 --func=PL --methods=MinMaxCostFlow
python run_experiment.py --year=2019 --month=10 --days=1 --func=PL --methods=MAPS
python run_experiment.py --year=2019 --month=10 --days=1 --func=PL --methods=LinUCB
```

### **Data Not Found**
Ensure data is properly uploaded to S3:
```bash
aws s3 ls s3://magisterka/datasets/green/year=2019/month=10/
```

### **Import Errors**
Test Lambda function:
```bash
make test-lambda
```

## 🔬 **Next Steps: Extended Analysis**

After replicating Hikima, try extended analysis:

```bash
# Add the 4th method (Linear Program)
python run_experiment.py \
  --year=2019 --month=10 --days=1,6 \
  --func=PL,Sigmoid \
  --methods=MinMaxCostFlow,MAPS,LinUCB,LP

# Multi-day analysis
python run_experiment.py \
  --year=2019 --month=10 --days=1,2,3,4,5,6,7 \
  --func=PL,Sigmoid \
  --methods=LP,MAPS,LinUCB

# Multi-month patterns
python run_experiment.py \
  --year=2019 --months=3,6,9,12 --days=1,15 \
  --func=PL \
  --methods=LP,MAPS
```

---

**🎉 You now have a clean, systematic way to replicate and extend the Hikima experimental setup!** 