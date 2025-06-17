# Enhanced Rideshare Experiment Usage Examples

## 🎯 **Easy Method: Use Helper Script (Recommended)**

The `run_experiment.sh` script provides a clean interface for sophisticated multi-method experiments:

```bash
# Make executable (one time)
chmod +x run_experiment.sh

# Show all available commands
./run_experiment.sh
```

## 🧪 **Sophisticated Experiment Capabilities**

### **1. Multi-Method Comparative Experiments**

Run all 4 bipartite matching methods simultaneously for direct comparison:

```bash
# Compare all methods on green taxi data
./run_experiment.sh run-comparative green 2019 3 PL 5

# Results show:
# • Proposed Method (Min-Cost Flow)
# • MAPS (Market-Aware Pricing Strategy)  
# • LinUCB (Multi-Armed Bandit)
# • Linear Program (New Method)
```

**Example Output:**
```
🔬 Comparative Experiment Analysis: rideshare_green_2019_03_proposed_maps_linucb_linear_program_pl_...
================================================================================
Methods Tested: proposed, maps, linucb, linear_program
🏆 Best Performing Methods:
  • Objective Value: LINEAR_PROGRAM (165416.502)
  • Match Rate: LINEAR_PROGRAM (54.96%)
  • Revenue: LINEAR_PROGRAM (165416.502)
📈 Performance Ranking:
  1. LINEAR_PROGRAM: 165416.50
  2. PROPOSED: 73565.84
  3. LINUCB: 38240.65
  4. MAPS: 21829.66
```

### **2. Meta-Parameter Testing**

Test different algorithm parameters to optimize performance:

```bash
# Test different window times
./run_experiment.sh test-window-time green 2019 3 linear_program 600
./run_experiment.sh test-window-time green 2019 3 linear_program 300
./run_experiment.sh test-window-time green 2019 3 linear_program 900

# Test retry mechanisms
./run_experiment.sh test-retry-count green 2019 3 proposed 5
./run_experiment.sh test-retry-count green 2019 3 proposed 10

# Compare acceptance functions
./run_experiment.sh test-acceptance-functions green 2019 3 maps
```

**Acceptance Function Comparison Results:**
```
📄 PL Results: Match Rate: 45.24%, Revenue: 65,365.87
📄 Sigmoid Results: Match Rate: 72.01%, Revenue: 105,936.55
```

### **3. Comprehensive Benchmarking**

Run extensive benchmarks across all methods and acceptance functions:

```bash
# Full benchmark suite
./run_experiment.sh run-benchmark green 2019 3 5

# Tests all combinations:
# • 4 Methods × 2 Acceptance Functions = 8 experiments
# • Automatic comparative analysis
# • Performance visualizations
```

### **4. Single Method Testing**

Focus on specific algorithms for detailed analysis:

```bash
# Test individual methods
./run_experiment.sh run-single green 2019 3 proposed PL 5
./run_experiment.sh run-single green 2019 3 linear_program Sigmoid 3
./run_experiment.sh run-single yellow 2019 6 maps PL 7
```

## 📊 **Advanced Analysis & Visualization**

### **Comparative Analysis**

```bash
# Analyze comparative experiments
python local-manager/results_manager.py analyze rideshare_green_2019_03_proposed_maps_linucb_linear_program_pl_...

# Generates:
# • Method performance comparison tables
# • Best performer identification  
# • Performance ranking
# • Detailed statistical analysis
# • 3 visualization plots automatically
```

### **Experiment Management**

```bash
# List recent experiments  
./run_experiment.sh list-experiments 7

# Show specific experiment details
./run_experiment.sh show-experiment <experiment_id>

# Generate comprehensive reports
python local-manager/results_manager.py report --days 7 --output analysis_report.txt
```

## 🔄 **Complete Workflow Examples**

### **Example 1: Algorithm Development Workflow**

```bash
# 1. Download datasets
./run_experiment.sh download-bulk 2019 1 3 green,yellow,fhv

# 2. Run comparative baseline
./run_experiment.sh run-comparative green 2019 3 PL 5

# 3. Test meta-parameters for best performer  
./run_experiment.sh test-window-time green 2019 3 linear_program 300
./run_experiment.sh test-window-time green 2019 3 linear_program 600
./run_experiment.sh test-window-time green 2019 3 linear_program 900

# 4. Compare acceptance functions
./run_experiment.sh test-acceptance-functions green 2019 3 linear_program

# 5. Full benchmark with optimized parameters
./run_experiment.sh run-benchmark green 2019 3 7

# 6. Analyze and visualize results
python local-manager/results_manager.py analyze <comparative_experiment_id>
```

### **Example 2: Multi-Dataset Analysis**

```bash
# Test across different vehicle types and time periods
for vehicle in green yellow fhv; do
    for month in 1 2 3; do
        ./run_experiment.sh run-comparative $vehicle 2019 $month PL 3
        echo "Completed $vehicle taxi analysis for 2019-$month"
    done
done

# Generate comprehensive comparison report
python local-manager/results_manager.py report --days 30
```

### **Example 3: Parameter Sensitivity Analysis**

```bash
# Test linear_program method with different parameters
for window in 180 300 600 900 1200; do
    ./run_experiment.sh test-window-time green 2019 3 linear_program $window
done

for retries in 3 5 10 15; do  
    ./run_experiment.sh test-retry-count green 2019 3 linear_program $retries
done

# Compare all results
python local-manager/results_manager.py compare <experiment_id_1> <experiment_id_2> <experiment_id_3>
```

## 📈 **S3 Data Organization**

Results are automatically organized with method-specific paths:

```
s3://magisterka/
├── datasets/
│   ├── green/year=2019/month=03/
│   ├── yellow/year=2019/month=03/
│   └── fhv/year=2020/month=01/
└── experiments/results/rideshare/
    ├── pl/          # PL acceptance function results
    └── sigmoid/     # Sigmoid acceptance function results
```

## 🎯 **Key Features Achieved**

✅ **4 Sophisticated Methods**: Proposed, MAPS, LinUCB, Linear Program  
✅ **Meta-Parameter Testing**: Window time, retry count, acceptance functions  
✅ **Comparative Analysis**: Side-by-side method comparison  
✅ **Automatic Visualization**: 3 plot types per comparative experiment  
✅ **Statistical Analysis**: Performance ranking, best performers  
✅ **Path Organization**: Method-specific S3 organization  
✅ **Scalable Architecture**: Cloud-native, serverless execution

## 🚀 **Next Steps**

1. **Algorithm Development**: Use comparative results to improve methods
2. **Parameter Optimization**: Use meta-parameter testing to tune performance  
3. **Scalability Testing**: Run experiments on larger datasets
4. **Real-time Integration**: Deploy best-performing methods in production
5. **Research Publication**: Use comprehensive analysis for academic papers

The system now provides **publication-quality experiment infrastructure** matching the sophistication of the original `experiment_PL.py` while adding cloud-native scalability and automated analysis capabilities. 