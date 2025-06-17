# Multi-Dataset Quick Usage Guide

This guide covers the new multi-dataset and parallel processing features for the ride-hailing pricing optimization experiments.

## 🚀 Quick Start

### 1. Check Your Current Data
```bash
ls -la data/
# You should see: yellow_tripdata_2019-10.parquet, green_tripdata_2019-10.parquet, area_information.csv
```

### 2. Test with Existing Data
```bash
# Test parallel experiments with your current Yellow/Green taxi data
python3 run_experiments_parallel.py --dry-run --vehicle-types yellow green --years 2019 --months 10
```

### 3. Run Quick Parallel Experiments
```bash
# Quick test scenario (uses your existing data)
./run_multi_experiments.sh --scenario quick

# Custom configuration with your data
./run_multi_experiments.sh --vehicle-types "yellow green" --years "2019" --months "10" --boroughs "Manhattan Queens"
```

### 4. Download Additional Datasets (Optional)
```bash
# Check what else is available online
python3 bin/data_manager.py --check-only --vehicle-types yellow green fhv --years 2018 2019 2020

# Download additional datasets
python3 setup_multi.py --vehicle-types fhv --years 2019 --months 10 11
```

### 5. Aggregate Results
```bash
# Basic aggregation
python3 aggregate_results.py --create-plots

# Results will be in results/aggregated/ with visualizations
```

## 📊 Supported Vehicle Types

| Type | Years | Description |
|------|-------|-------------|
| **yellow** | 2009-2024 | Traditional NYC yellow taxis |
| **green** | 2013-2024 | Boro taxis (outer boroughs) |
| **fhv** | 2015-2024 | For-hire vehicles (Uber, Lyft, etc.) |
| **fhvhv** | 2019-2024 | High-volume for-hire vehicles |

## 🛠️ Common Use Cases

### Compare Vehicle Types
```bash
# Compare all vehicle types for same time period
./run_multi_experiments.sh --scenario comparison --max-workers 6
```

### Time Series Analysis
```bash
# Multiple months for trend analysis
python3 setup_multi.py --vehicle-types yellow --years 2019 --months 6 7 8 9 10 11 12
./run_multi_experiments.sh --vehicle-types yellow --months "6 7 8 9 10 11 12" --simulation-range 10
```

### Comprehensive Study
```bash
# Full comparison across vehicle types and time periods (warning: many experiments!)
./run_multi_experiments.sh --scenario comprehensive --max-workers 8
```

## 📁 File Organization

**Current Setup (Compatible Mode):**
```
data/
├── yellow_tripdata_2019-10.parquet    # Your existing files
├── green_tripdata_2019-10.parquet     # Your existing files  
├── area_information.csv               # Your existing files
└── metadata/                          # New: Download reports
    ├── availability_report.json
    └── download_summary.json
```

**New Multi-Dataset Mode (Optional):**
```
data/
├── parquet/                    # Original parquet files
├── csv/                       # Converted CSV files  
└── metadata/                  # Download reports
```

results/
├── summary/                   # Summary CSV files (compatible with original)
├── detailed/                  # Detailed JSON results
├── aggregated/               # Cross-experiment analysis
│   ├── detailed_performance_*.csv
│   ├── summary_tables_*.txt
│   └── figures/              # Visualization plots
└── parallel_runs/            # Parallel execution logs
```

## ⚙️ Configuration Options

### Data Manager
```bash
python3 bin/data_manager.py \
    --vehicle-types yellow green fhv \
    --years 2019 2020 \
    --months 10 11 12 \
    --max-workers 6 \
    --check-only  # Optional: just check availability
```

### Parallel Experiments
```bash
python3 run_experiments_parallel.py \
    --vehicle-types yellow green \
    --years 2019 2020 \
    --months 10 11 \
    --days 6 7 \
    --experiment-types PL Sigmoid \
    --boroughs Manhattan Queens \
    --simulation-range 10 \
    --max-workers 4 \
    --dry-run  # Optional: show configs without running
```

## 📈 Analysis Features

### Performance Comparison
The aggregation script creates:
- **Summary tables** by vehicle type, experiment type, method, borough
- **Performance metrics** (avg objective, solve time, success rate)
- **Visualizations** (box plots, heatmaps, time series)

### Key Metrics
- **Objective Value**: Expected revenue from optimal pricing
- **Solve Time**: LP optimization time (typically <0.1s)
- **Success Rate**: Percentage of successful iterations
- **Scalability**: Performance across different data sizes

## 🎯 Predefined Scenarios

### Quick Test (3 min)
```bash
./run_multi_experiments.sh --scenario quick
```
- Vehicle types: Yellow, Green
- Time period: Oct 2019
- Boroughs: Manhattan only
- Iterations: 3 per experiment

### Comprehensive Analysis (30-60 min)
```bash
./run_multi_experiments.sh --scenario comprehensive
```
- Vehicle types: Yellow, Green, FHV
- Time periods: Oct-Nov 2019-2020
- Boroughs: Manhattan, Queens
- Iterations: 10 per experiment

### Vehicle Comparison (15 min)
```bash
./run_multi_experiments.sh --scenario comparison
```
- Vehicle types: All (Yellow, Green, FHV, FHVHV)
- Time period: Oct 2019
- Borough: Manhattan
- Both PL and Sigmoid experiments

## 🔧 Troubleshooting

### Dataset Not Available
```bash
# Check what's actually available
python3 bin/data_manager.py --check-only --vehicle-types fhv --years 2020 2021

# Some vehicle types have limited year ranges
# FHV: 2015+, FHVHV: 2019+, Green: 2013+
```

### Parallel Execution Issues
```bash
# Reduce workers if memory issues
./run_multi_experiments.sh --max-workers 2

# Use dry run to check configurations
./run_multi_experiments.sh --dry-run --scenario comprehensive
```

### Missing Dependencies
```bash
# Install additional dependencies for multi-dataset features
pip install matplotlib seaborn requests pyarrow

# Verify setup
python3 verify_multi_setup.py
```

## 💡 Best Practices

1. **Start Small**: Use `--scenario quick` or `--dry-run` first
2. **Check Availability**: Always run `--check-only` before downloading
3. **Monitor Resources**: Large experiments can use significant CPU/memory
4. **Save Results**: Aggregated results provide the most insight
5. **Use Parallel Workers**: 4-8 workers optimal for most systems

## 📞 Getting Help

Run any script with `--help` for detailed options:
```bash
python3 setup_multi.py --help
python3 run_experiments_parallel.py --help
./run_multi_experiments.sh --help
python3 aggregate_results.py --help
``` 