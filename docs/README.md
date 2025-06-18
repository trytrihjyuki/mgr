# Unified Rideshare Experiment System

## 📋 Overview

This system extends the original `experiment_PL.py` to benchmark **all methods** (Hikima, MAPS, LinUCB, Linear Program) in a unified framework. It supports the original experimental format while adding multi-temporal capabilities and enhanced analysis.

## 🚀 Quick Start

### **Basic Experiment (Original Format)**
```bash
# Format: start_hour end_hour time_interval place time_step month day year [vehicle_type] [methods] [acceptance_func]
./run_experiment.sh run-experiment 10 20 5m Manhattan 30s 10 6 2019

# With all methods
./run_experiment.sh run-experiment 10 20 5m Manhattan 30s 10 6 2019 green "hikima,maps,linucb,linear_program" PL
```

### **Multi-Month Analysis**
```bash
# Compare performance across multiple months
./run_experiment.sh run-multi-month 10 20 5m Manhattan 30s "3,4,5" "6,10" 2019 green "hikima,maps" PL
```

### **Data Download**
```bash
# Download historical data (now supports 2013-2023!)
./run_experiment.sh download-single green 2016 3
./run_experiment.sh download-bulk 2019 1 12 green,yellow
```

## 🎯 Key Features

### ✅ **Unified Architecture**
- **Single Lambda function** handles all experiments (no over-engineering)
- **Original format** extended from `experiment_PL.py`
- **All methods** supported: hikima, maps, linucb, linear_program

### ✅ **Multi-Temporal Support**
- **Single days**: Test specific market conditions
- **Multiple days**: Weekly patterns (e.g., Sunday vs Thursday)  
- **Multiple months**: Seasonal analysis

### ✅ **Clean Results Structure**
- **No duplication** of common parameters
- **Monthly summaries** for trend analysis
- **Daily summaries** for detailed plotting
- **Performance ranking** across all methods

### ✅ **Original Compliance**
- **Same parameters** as experiment_PL.py (place, day, time_interval, time_unit, simulation_range)
- **Same methodology** (scenarios vs num_eval distinction)
- **Same results** but extended to all methods

## 📊 Methods Supported

| Method | Description | Algorithm Type |
|--------|-------------|----------------|
| **hikima** | Original proposed method from paper | Min-cost flow |
| **maps** | Area-based pricing approximation | Heuristic |
| **linucb** | Contextual bandit learning | Machine learning |
| **linear_program** | Our optimal LP solution | Mathematical optimization |

## 📈 Understanding Scenarios vs num_eval

**Important distinction from original paper:**

- **scenarios** = Different time periods (120 = every 5 min from 10:00-20:00)
- **num_eval** = Monte Carlo evaluations per scenario (100 = reduce randomness)

👉 **See [Scenarios vs num_eval Explanation](SCENARIOS_VS_NUM_EVAL_EXPLANATION.md)** for detailed analysis.

## 🗂️ Documentation Structure

- **[README.md](README.md)** - This overview (start here)
- **[SCENARIOS_VS_NUM_EVAL_EXPLANATION.md](SCENARIOS_VS_NUM_EVAL_EXPLANATION.md)** - Scenarios vs num_eval distinction
- **[HIKIMA_COMPLIANCE_DOCUMENTATION.md](HIKIMA_COMPLIANCE_DOCUMENTATION.md)** - Original refactoring details
- **[DATA_AVAILABILITY_SOLUTION.md](DATA_AVAILABILITY_SOLUTION.md)** - Data availability fixes
- **[COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md)** - Legacy comprehensive docs

## 🔧 System Architecture

### **Simplified Lambda Structure**
```
lambdas/experiment-runner/
├── lambda_function.py              # Legacy experiment runner
├── lambda_function_unified.py      # New unified runner (recommended)
└── lambda-package/                 # Deployment package
```

### **Command Structure**
```bash
# Original experiment_PL.py format (extended)
./run_experiment.sh run-experiment <start_hour> <end_hour> <time_interval> <place> <time_step> <month> <day> <year> [options]

# Multi-temporal extensions
./run_experiment.sh run-multi-month <start_hour> <end_hour> <time_interval> <place> <time_step> <months> <days> <year> [options]
```

### **Data Storage (Cleaned Up)**
```
s3://magisterka/
├── datasets/                       # Raw taxi data
│   ├── green/year=2019/month=03/
│   ├── yellow/year=2019/month=03/
│   └── fhv/year=2019/month=03/
└── experiments/                    # Experiment results (no redundant /results)
    └── rideshare/type=green/eval=pl/year=2019/months=03/
```

## 🧪 Example Experiments

### **Reproduce Original Paper Results**
```bash
# Manhattan experiment (30s time step, as per paper)
./run_experiment.sh run-experiment 10 20 5m Manhattan 30s 10 6 2019 green "hikima,maps,linucb"

# Bronx experiment (300s time step, as per paper) 
./run_experiment.sh run-experiment 10 20 5m Bronx 300s 10 6 2019 green "hikima,maps,linucb"
```

### **Test Our Linear Program Method**
```bash
# Compare our LP method against original methods
./run_experiment.sh run-experiment 10 20 5m Manhattan 30s 10 6 2019 green "hikima,maps,linucb,linear_program"
```

### **Seasonal Analysis**
```bash
# Compare performance across seasons
./run_experiment.sh run-multi-month 10 20 5m Manhattan 30s "3,6,9,12" "6,10" 2019 green "hikima,linear_program"
```

### **Historical Data Analysis**
```bash
# Test with historical data (now supported!)
./run_experiment.sh run-experiment 10 20 5m Manhattan 30s 10 6 2016 green "hikima,maps,linucb,linear_program"
```

## 📊 Results Analysis

### **Daily Performance**
```bash
# Analyze single experiment
python local-manager/results_manager.py analyze unified_green_manhattan_2019_10_20250618_123456

# Compare experiments
python local-manager/results_manager.py compare exp1_id exp2_id
```

### **Multi-Month Trends**
```json
{
  "monthly_summaries": {
    "2019-03": {
      "hikima": {"avg_objective_value": 1250.45},
      "linear_program": {"avg_objective_value": 1345.67}
    },
    "2019-04": {
      "hikima": {"avg_objective_value": 1180.32},
      "linear_program": {"avg_objective_value": 1267.89}
    }
  }
}
```

## 🔍 Troubleshooting

### **Common Issues**

1. **"Data not available" errors**
   - ✅ **Fixed!** Now supports 2013-2023 data
   - Use: `./run_experiment.sh check-availability green 2016 3`

2. **Lambda timeout**
   - Reduce `simulation_range` for testing
   - Use fewer methods initially

3. **S3 permissions**
   - Check AWS credentials: `aws s3 ls s3://magisterka/`

### **Performance Tips**

- **Start small**: Use `simulation_range=10` for testing
- **Single method**: Test one method first
- **Check data**: Verify data exists before experiments

## 🚀 Next Steps

1. **Deploy** unified Lambda function
2. **Test** basic experiments
3. **Scale up** to multi-month analysis
4. **Compare** results against original paper
5. **Optimize** Linear Program method

## 📞 Support

- **Issues**: Check logs in CloudWatch
- **Data problems**: Use data availability checker
- **Results questions**: See analysis documentation

---

**🎉 The system now provides a clean, unified approach to rideshare pricing optimization experiments with full original paper compliance and modern enhancements!**
