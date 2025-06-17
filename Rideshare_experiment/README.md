# Ride-Hailing Pricing Optimization Experiments

This repository contains refactored experiments for ride-hailing pricing optimization using **Linear Programming** based on the Gupta-Nagarajan reduction of the Myerson revenue maximization problem.

## Overview

The experiments compare pricing strategies for ride-hailing platforms using two acceptance function models:
- **Piecewise Linear (PL)**: Linear acceptance probability function
- **Sigmoid**: Sigmoid acceptance probability function

The optimization is performed using the **Gupta-Nagarajan Linear Program** which provides a polynomial-time approximation to the originally intractable revenue maximization problem.

## Key Features

✅ **Clean LP Implementation**: Uses PuLP for linear programming with CBC solver  
✅ **Comprehensive Logging**: Detailed logs with timestamps and performance metrics  
✅ **Benchmark Data Storage**: JSON and CSV outputs for analysis  
✅ **Modular Design**: Separates pricing logic, benchmarking, and experiments  
✅ **Real NYC Data**: Uses NYC taxi trip data from 2019  
✅ **Multiple Boroughs**: Supports Manhattan, Queens, Bronx, Brooklyn experiments  

## Directory Structure

```
Rideshare_experiment/
├── bin/                          # Experiment code
│   ├── lp_pricing.py            # LP pricing optimization module
│   ├── benchmark_utils.py       # Logging and benchmarking utilities
│   ├── experiment_PL_refactored.py      # PL acceptance experiments
│   ├── experiment_Sigmoid_refactored.py # Sigmoid acceptance experiments
│   └── data_download.py         # Data download script
├── data/                        # NYC taxi and area data
├── results/                     # Experiment outputs (created automatically)
│   ├── summary/                 # Summary CSV files
│   ├── detailed/                # Detailed JSON results
│   ├── logs/                    # Experiment logs
│   └── benchmarks/              # Benchmark data
├── requirements.txt             # Python dependencies
├── Experiments_refactored.sh    # Full experiment suite
├── Experiments_test_refactored.sh # Quick test
└── README.md                    # This file
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `pulp>=2.6.0` - Linear programming solver
- `pandas>=1.3.0` - Data processing
- `numpy>=1.21.0` - Numerical computing
- `networkx>=2.6.0` - Graph algorithms
- `pyproj>=3.2.0` - Geographic projections

### 2. Download Data

```bash
bash setup.sh
```

This downloads NYC taxi trip data (green and yellow taxi datasets from October 2019).  
**Note**: Downloads ~300MB total with progress tracking - takes 2-5 minutes depending on connection.

## Running Experiments

### Quick Test (Recommended First)

Test the setup with a small number of iterations:

```bash
bash Experiments_test_refactored.sh
```

This runs 5 iterations for Manhattan with both PL and Sigmoid models.

### Full Experiment Suite

Run the complete experiment suite across all boroughs:

```bash
bash Experiments_refactored.sh
```

This executes:
- **PL Experiments**: Manhattan (30s intervals), Queens/Bronx/Brooklyn (5m intervals)
- **Sigmoid Experiments**: Same configuration
- **120 iterations** per experiment (10 hours of simulated time in 5-minute increments)

### Custom Experiments

Run individual experiments with custom parameters:

```bash
cd bin

# PL experiment: <borough> <day> <interval> <unit> <iterations>
python3 experiment_PL_refactored.py Manhattan 6 30 s 10

# Sigmoid experiment
python3 experiment_Sigmoid_refactored.py Queens 6 5 m 20
```

**Parameters:**
- `borough`: Manhattan, Queens, Bronx, Brooklyn
- `day`: Day of October 2019 (1-31)
- `interval`: Time interval length
- `unit`: s (seconds) or m (minutes)  
- `iterations`: Number of time windows to simulate

## Multi-Dataset & Parallel Processing

### Supported Vehicle Types

The system now supports multiple NYC vehicle types:

- **Yellow Taxi** (2009-2024): Traditional NYC yellow taxis
- **Green Taxi** (2013-2024): Boro taxis serving outer boroughs  
- **For-Hire Vehicles (FHV)** (2015-2024): Uber, Lyft, and other app-based services
- **High Volume FHV (FHVHV)** (2019-2024): High-volume for-hire vehicles

### Data Sources
- [Yellow Taxi Data](https://data.cityofnewyork.us/Transportation/2018-Yellow-Taxi-Trip-Data/t29m-gskq/about_data)
- [Green Taxi Data](https://data.cityofnewyork.us/Transportation/2015-Green-Taxi-Trip-Data/gi8d-wdg5/about_data)  
- [FHV Data](https://data.cityofnewyork.us/Transportation/2020-For-Hire-Vehicles-Trip-Data/m3yx-mvk4/about_data)

### Multi-Dataset Setup

Download data for multiple vehicle types, years, and months:

```bash
# Download specific datasets
python3 setup_multi.py --vehicle-types yellow green --years 2019 2020 --months 10 11

# Check availability without downloading
python3 setup_multi.py --check-only --vehicle-types yellow green fhv --years 2018 2019 2020

# Download with 8 parallel workers
python3 setup_multi.py --max-workers 8 --vehicle-types yellow green fhv fhvhv
```

### Parallel Experiment Execution

Run experiments across multiple datasets in parallel:

```bash
# Run experiments on multiple vehicle types and time periods
python3 run_experiments_parallel.py \
    --vehicle-types yellow green \
    --years 2019 2020 \
    --months 10 11 \
    --days 6 7 \
    --experiment-types PL Sigmoid \
    --boroughs Manhattan Queens \
    --max-workers 4

# Quick test across datasets
python3 run_experiments_parallel.py \
    --vehicle-types yellow green fhv \
    --simulation-range 3 \
    --max-workers 6

# Show what would run without executing
python3 run_experiments_parallel.py --dry-run --vehicle-types yellow green
```

### Results Aggregation

Aggregate and analyze results across multiple experiments:

```bash
# Basic aggregation
python3 aggregate_results.py

# Create visualizations and detailed analysis
python3 aggregate_results.py --create-plots --output-dir results/analysis

# Process specific results directory
python3 aggregate_results.py --results-dir custom_results --create-plots
```

### Data Management Tools

**Check dataset availability:**
```bash
cd bin
python3 data_manager.py --check-only --vehicle-types yellow green fhv --years 2018 2019 2020
```

**Download specific datasets:**
```bash
cd bin
python3 data_manager.py --vehicle-types yellow --years 2019 --months 1 2 3 --max-workers 6
```

### Enhanced Experiment Features

**Vehicle-type specific parameters:**
- Column mapping handles different data schemas automatically
- Missing fields (like trip distance in FHV) are estimated
- Acceptance functions adapt to vehicle type characteristics

**Current file organization:**
```
data/
├── yellow_tripdata_2019-10.parquet    # Your existing data
├── green_tripdata_2019-10.parquet     # Your existing data
├── area_information.csv               # Your existing data
└── metadata/                          # New: Download reports (created as needed)

results/
├── summary/          # CSV summaries (original format)
├── detailed/         # JSON detailed results  
├── aggregated/       # Cross-experiment analysis (new)
└── parallel_runs/    # Parallel execution logs (new)
```

## Algorithm Details

### Gupta-Nagarajan Linear Program

The core optimization solves:

```
maximize  Σ (π - w_c,t) * x_c,t,π
subject to:
  Σ_π y_c,π ≤ 1                           ∀c (offer ≤1 price per rider)
  Σ_t x_c,t,π ≤ p_c(π) * y_c,π           ∀c,π (acceptance constraint)  
  Σ_c,π x_c,t,π ≤ 1                       ∀t (taxi capacity)
  x,y ≥ 0
```

Where:
- `y_c,π` = probability of offering price π to rider c
- `x_c,t,π` = probability rider c accepts π and matches taxi t  
- `p_c(π)` = acceptance probability function
- `w_c,t` = travel cost from c to t

### Acceptance Functions

**Piecewise Linear:**
```
p_c(π) = max(0, min(1, -c_param * π + d_param))
where c_param = 2.0 / total_amount_c
```

**Sigmoid:**
```
p_c(π) = 1 - 1/(1 + exp((-π + β * total_amount_c)/(γ * total_amount_c)))
where β = 1.3, γ = 0.3√3/π
```

## Output Files

### Summary Results (CSV)

Location: `results/summary/Average_result_{model}_place={borough}_day={day}_interval={interval}{unit}.csv`

Format:
```csv
method,avg_objective_value,avg_computation_time
LP_Pricing,1234.56,0.123
```

### Detailed Results (JSON)

Location: `results/detailed/detailed_{model}_place={borough}_day={day}_interval={interval}{unit}.json`

Contains:
- Complete iteration-by-iteration results
- LP solver status and objective values
- Timing information
- KPIs (number of riders/taxis per iteration)

### Logs

Location: `results/logs/{model}_{timestamp}.log`

Real-time experiment logs with:
- Iteration progress
- LP solve times and objective values
- Warning messages for empty iterations
- Summary statistics

## Analysis & Interpretation

### Key Metrics

1. **Objective Value**: Expected revenue from optimal pricing
2. **Solve Time**: Time to solve LP (should be <1 second typically)
3. **Success Rate**: Percentage of iterations with valid solutions

### Example Analysis

```python
import json
import pandas as pd

# Load detailed results
with open('results/detailed/detailed_PL_LP_place=Manhattan_day=6_interval=30s.json') as f:
    data = json.load(f)

# Extract objective values per iteration
objectives = [iter_data['results']['LP_Pricing']['objective_value'] 
              for iter_data in data['iterations']]

print(f"Average objective: {np.mean(objectives):.2f}")
print(f"Std deviation: {np.std(objectives):.2f}")
```

### Performance Benchmarks

**Expected Performance:**
- **LP Solve Time**: 0.01-0.5 seconds per iteration
- **Memory Usage**: <500MB peak  
- **Scalability**: Handles 50+ riders, 50+ taxis per iteration

## Troubleshooting

### Common Issues

1. **"No module named 'pulp'"**
   ```bash
   pip install pulp
   ```

2. **"Data files not found"**
   ```bash
   bash setup.sh
   ```

   If download is slow or stuck:
   - Check internet connection
   - Files are ~150MB each (green & yellow taxi data)  
   - Progress is shown every 2 seconds
   - Script automatically resumes if files partially downloaded

3. **"CBC solver not found"**
   - CBC is bundled with PuLP, but if issues persist:
   ```bash
   pip uninstall pulp
   pip install pulp
   ```

4. **Empty iterations (no riders/taxis)**
   - Normal for some time windows
   - Check logs for frequency

### Performance Issues

- **Slow LP solving**: Try smaller price grids (modify `base_prices` in code)
- **Memory issues**: Reduce simulation range or use smaller time intervals
- **Disk space**: Results can grow large; clean old outputs periodically

## Technical Notes

### Solver Configuration

The experiments use **CBC** (COIN-OR Branch and Cut) solver by default:
- Open-source, no license required
- Bundled with PuLP installation
- Good performance for medium-scale problems

For larger instances, consider Gurobi or CPLEX (requires licenses).

### Data Processing

- **Geographic calculations** using WGS84/GRS80 ellipsoid
- **Random location noise** added for privacy (±300m standard deviation)
- **Trip filtering** removes unrealistic trips (distance <1mm, amount <$0.001)

### Reproducibility

- Random seed is not fixed by default
- For reproducible results, add `np.random.seed(42)` in main functions
- Geographic noise will still cause small variations

## Citation

If you use this code in research, please cite:

```
@article{gupta2021revenue,
  title={Revenue Maximization in Ride-Hailing: A Linear Programming Approach},
  author={Gupta, Anupam and Nagarajan, Viswanath},
  journal={AAAI 2021 Integrated Optimization for Bipartite Matching},
  year={2021}
}
```

## License

MIT License - see LICENSE file for details.

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review log files in `results/logs/`
3. Verify data files exist in `data/` directory
4. Test with the small example first (`Experiments_test_refactored.sh`) 