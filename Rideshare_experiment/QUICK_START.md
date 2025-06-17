# Quick Start Guide

## 🚀 Test the Setup in 2 Minutes

### 1. Install Dependencies
```bash
cd Rideshare_experiment
pip install -r requirements.txt
```

### 2. Download Data
```bash
bash setup.sh
```

**Note**: This downloads ~150MB of NYC taxi data. You'll see progress like:
```
🔄 Downloading green_tripdata_2019-10.parquet...
📦 File size: 149 MB
📥 Progress: 23.4% (35 MB) - 2.1 MB/s
```

### 3. Verify Setup (Optional)
```bash
python3 check_setup.py
```

This checks all files and dependencies are ready.

### 4. Run Quick Test
```bash
bash Experiments_test_refactored.sh
```

This will:
- ✅ Run 5 iterations of PL experiment (Manhattan, day 6, 30s intervals)
- ✅ Run 5 iterations of Sigmoid experiment (same parameters)
- ✅ Create results in `results/` directory
- ✅ Show logs with LP solve times and objective values

### 5. Check Results
```bash
# View summary CSV
cat results/summary/Average_result_*_place=Manhattan_day=6_interval=30s.csv

# Analyze detailed results
cd bin
python3 analyze_results.py ../results/detailed/detailed_*_place=Manhattan_day=6_interval=30s.json
```

## Expected Output

You should see:
```
LP solved optimally in 0.045s with objective 234.5678
```

And results like:
```
Method: LP_Pricing
Average Objective Value: 234.57 ± 45.23
Average Solve Time: 0.0450s ± 0.0123s
Success Rate: 100.00%
```

## What's Different from Original

✅ **Linear Programming**: Uses Gupta-Nagarajan LP instead of min-cost flow  
✅ **Clean Code**: Modular design with separate pricing, logging, benchmarking  
✅ **Rich Logging**: Timestamps, performance metrics, detailed iteration logs  
✅ **Multiple Outputs**: CSV summaries + JSON details + benchmark data  
✅ **Easy Analysis**: Built-in analysis script with plots  

## Next Steps

- **Full experiments**: Run `bash Experiments_refactored.sh` for all boroughs
- **Custom experiments**: Modify parameters in scripts  
- **Analysis**: Use `analyze_results.py` for detailed analysis
- **Compare**: Run multiple experiments and compare results

## Troubleshooting

**Problem**: Import errors  
**Solution**: `pip install -r requirements.txt`

**Problem**: Data not found  
**Solution**: `bash setup.sh`

**Problem**: Slow LP solving  
**Solution**: Normal for first run; CBC needs to initialize

**Problem**: Empty iterations  
**Solution**: Normal; some time windows have no riders/taxis 