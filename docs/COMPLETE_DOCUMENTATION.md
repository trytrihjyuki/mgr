# 🚀 Rideshare Bipartite Matching - Complete Documentation

## 📋 Table of Contents

1. [🌟 Overview](#overview)
2. [🚀 Quick Start](#quick-start)
3. [📊 Running Experiments](#experiments)
4. [📋 Parameter Reference](#parameters)
5. [📈 JSON Output Reference](#json-output)
6. [🧰 Helper Commands](#commands)
7. [🐛 Troubleshooting](#troubleshooting)

---

## 🌟 Overview {#overview}

This system implements **bipartite matching algorithms** for rideshare optimization using real NYC taxi data. Built on **AWS Lambda architecture** with four advanced algorithms:

### 🔬 **Four Algorithms**

| Algorithm | Description | Efficiency | Best For |
|-----------|-------------|------------|----------|
| **PROPOSED** | Min-cost flow bipartite matching | 85-91% | Balanced performance |
| **LINEAR_PROGRAM** | Novel LP optimization method | 88-91% | Maximum revenue |
| **MAPS** | Market-Aware Pricing Strategy | 65-77% | Market dynamics |
| **LINUCB** | Multi-Armed Bandit approach | 75-84% | Learning scenarios |

---

## 🚀 Quick Start {#quick-start}

### ⚡ **One-Command Setup**
```bash
# Check data availability first
./run_experiment.sh check-availability green 2019 3

# Download data if needed
./run_experiment.sh download-single green 2019 3

# Run comprehensive comparison (all 4 algorithms)
./run_experiment.sh run-comparative green 2019 3 PL 5
```

---

## 📊 Running Experiments {#experiments}

### 🧪 **Experiment Types**

#### **1. Single Method**
```bash
./run_experiment.sh run-single green 2019 3 linear_program PL 5
#                              ↑     ↑    ↑       ↑         ↑  ↑
#                          vehicle year month  algorithm  func scenarios
```

#### **2. Comparative Analysis** 
```bash
./run_experiment.sh run-comparative green 2019 3 PL 5
# Runs all 4 algorithms: PROPOSED, MAPS, LINUCB, LINEAR_PROGRAM
```

#### **3. Parameter Testing**
```bash
# Test different time windows
./run_experiment.sh test-window-time green 2019 3 proposed 600

# Test meta-parameters (num_eval, alpha, window_time)
./run_experiment.sh test-meta-params green 2019 3 maps

# Comprehensive parameter sweep (54 combinations)
./run_experiment.sh parameter-sweep green 2019 3 linear_program
```

---

## 📋 Parameter Reference {#parameters}

### 🎯 **Required Parameters**

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `vehicle_type` | string | `green`, `yellow`, `fhv` | NYC taxi vehicle type |
| `year` | integer | `2017-2023` | Data year (2018-2023 recommended) |
| `month` | integer | `1-12` | Data month |
| `method` | string | `proposed`, `maps`, `linucb`, `linear_program` | Algorithm to use |

### ⚙️ **Algorithm Parameters**

| Parameter | Default | Range | Description | Impact |
|-----------|---------|-------|-------------|--------|
| `simulation_range` | 3-5 | 1-100 | Number of scenarios per method | Higher = more robust results |
| `num_eval` | 100 | 10-1000 | Monte Carlo evaluations per scenario | Higher = more accurate |
| `window_time` | 300 | 30-600 | Matching time window (seconds) | Higher = better matches, slower |
| `alpha` | 18 | 5-50 | Algorithm parameter for w_ij calculation | Affects matching efficiency |
| `s_taxi` | 25 | 10-50 | Taxi speed parameter (km/h) | Affects distance calculations |
| `retry_count` | 3 | 1-10 | Retry attempts for failed matches | Higher = more resilient |
| `ucb_alpha` | 0.5 | 0.1-2.0 | LinUCB exploration parameter | Higher = more exploration |
| `base_price` | 5.875 | 3.0-15.0 | Base trip price (USD) | Affects revenue calculations |

### 📊 **Acceptance Functions**

#### **1. Piecewise Linear (PL)**
```
P(accept) = max(0, min(1, -c × price + b))
```
- **Fast computation**
- **Linear relationship**
- **Good for basic modeling**

#### **2. Sigmoid**
```
P(accept) = 1 - 1/(1 + exp((βR - price)/(γR)))
```
- **Realistic S-curve behavior**
- **Models human psychology**
- **More computational overhead**

---

## 📈 JSON Output Reference {#json-output}

### 📋 **Complete Output Structure**

```json
{
  "experiment_id": "rideshare_green_2019_03_proposed_maps_linucb_linear_program_pl_20250618_002357",
  "experiment_type": "rideshare_comparative",
  "status": "completed",
  "timestamp": "2025-06-18T00:23:58.007707",
  "execution_time_seconds": 0.44,
  "parameters": { /* Input parameters */ },
  "data_info": { /* Source data details */ },
  "method_results": { /* Per-algorithm results */ },
  "execution_times": { /* Performance timing */ },
  "comparative_stats": { /* Cross-algorithm comparison */ }
}
```

### 🎯 **Input Parameters Section**

| Field | Type | Description |
|-------|------|-------------|
| `vehicle_type` | string | Taxi type used for experiment |
| `year`, `month` | integer | Data period |
| `methods` | array | List of algorithms executed |
| `acceptance_function` | string | PL or Sigmoid |
| `simulation_range` | integer | Number of scenarios |
| `window_time` | integer | Matching window (seconds) |
| `num_eval` | integer | Monte Carlo evaluations |
| `alpha` | number | Algorithm parameter |
| `s_taxi` | number | Taxi speed (km/h) |
| `ucb_alpha` | number | LinUCB exploration parameter |
| `base_price` | number | Base price per trip |

### 📊 **Data Info Section**

| Field | Type | Description |
|-------|------|-------------|
| `exists` | boolean | Whether source data exists |
| `data_key` | string | S3 path to source PARQUET file |
| `data_size_bytes` | integer | Size of source data |
| `last_modified` | string | When data was last updated |

### 🔬 **Method Results (Per Algorithm)**

Each algorithm returns detailed results:

```json
"proposed": {
  "method": "proposed",
  "algorithm": "min_cost_flow", 
  "scenarios": [ /* Individual scenario results */ ],
  "summary": { /* Aggregated statistics */ },
  "parameters": { /* Algorithm-specific parameters */ }
}
```

#### **Scenario-Level KPIs**

| Field | Type | Description | Typical Range |
|-------|------|-------------|---------------|
| `scenario_id` | integer | Scenario identifier | 0, 1, 2... |
| `total_requests` | integer | Total ride requests | 10,000-50,000 |
| `total_drivers` | integer | Available drivers | 5,000-25,000 |
| `supply_demand_ratio` | number | Drivers/requests ratio | 0.3-0.8 |
| `successful_matches` | integer | Completed matches | 3,000-20,000 |
| `match_rate` | number | Success rate (0-1) | 0.25-0.70 |
| `total_revenue` | number | Total revenue (USD) | $50,000-$200,000 |
| `objective_value` | number | Optimization objective | 50,000-200,000 |
| `avg_trip_value` | number | Revenue per trip | $8-$20 |
| `algorithm_efficiency` | number | Algorithm performance (0-1) | 0.65-0.95 |
| `num_evaluations` | integer | Monte Carlo iterations | 50-1000 |
| `evaluation_std` | number | Standard deviation | Very small (10^-10) |

#### **Summary Statistics**

| Field | Type | Description |
|-------|------|-------------|
| `total_scenarios` | integer | Number of scenarios run |
| `avg_objective_value` | number | Mean objective across scenarios |
| `avg_match_rate` | number | Mean match rate |
| `avg_revenue` | number | Mean total revenue |
| `avg_algorithm_efficiency` | number | Mean algorithm efficiency |
| `total_requests_processed` | integer | Sum of all requests |
| `total_successful_matches` | integer | Sum of all matches |
| `overall_match_rate` | number | Global match rate |
| `match_rate_std` | number | Match rate standard deviation |

### 🏆 **Comparative Statistics**

#### **Best Performing**
```json
"best_performing": {
  "objective_value": {
    "method": "linear_program",
    "value": 117300.16
  },
  "match_rate": {
    "method": "linear_program", 
    "value": 0.453
  }
}
```

#### **Performance Ranking**
```json
"performance_ranking": {
  "1": {"method": "linear_program", "score": 117300.16},
  "2": {"method": "proposed", "score": 112646.37},
  "3": {"method": "maps", "score": 107123.81},
  "4": {"method": "linucb", "score": 85888.29}
}
```

### 📊 **Key Performance Indicators (KPIs)**

#### **Primary KPIs**
1. **Objective Value**: Main optimization target (revenue/utility)
2. **Match Rate**: Percentage of successful matches (0-1)
3. **Algorithm Efficiency**: Performance score (0-1)
4. **Revenue**: Total monetary value generated

#### **KPI Interpretation**

| KPI Range | Performance | Interpretation |
|-----------|-------------|----------------|
| **Match Rate > 0.5** | Excellent | High success rate |
| **Match Rate 0.3-0.5** | Good | Acceptable performance |
| **Match Rate < 0.3** | Poor | Low efficiency |
| **Efficiency > 0.85** | Excellent | Optimal algorithm performance |
| **Efficiency 0.7-0.85** | Good | Reasonable performance |
| **Efficiency < 0.7** | Poor | Sub-optimal results |

---

## 🧰 Helper Commands {#commands}

### 📊 **Data Management**

```bash
# Check if data exists before downloading
./run_experiment.sh check-availability yellow 2017 8
# Output: ✅ Data available - attempting download from NYC Open Data
#         💡 Suggestion: Use years 2018-2023

# Download single dataset
./run_experiment.sh download-single green 2019 3

# Bulk download with pre-validation  
./run_experiment.sh download-bulk 2019 1 3 green,yellow,fhv

# List available data
./run_experiment.sh list-data
```

### 🔬 **Experiment Management**

```bash
# List recent experiments (last 30 days)
./run_experiment.sh list-experiments 30

# Show specific experiment details
./run_experiment.sh show-experiment <experiment_id>

# Detailed analysis with visualizations
./run_experiment.sh analyze <experiment_id>

# Compare two experiments
./run_experiment.sh compare-methods <exp_id_1> <exp_id_2>
```

### 📈 **Advanced Testing**

```bash
# Test different time windows
./run_experiment.sh test-window-time green 2019 3 proposed 600

# Test both acceptance functions
./run_experiment.sh test-acceptance-functions green 2019 3 linear_program

# Comprehensive parameter testing
./run_experiment.sh test-meta-params green 2019 3 maps

# Parameter sensitivity sweep (54 combinations)
./run_experiment.sh parameter-sweep green 2019 3 proposed
```

---

## 🐛 Troubleshooting {#troubleshooting}

### ❌ **Common Issues**

#### **1. Data Not Available**
```bash
# Error: Data availability check failed
Solution: Use recommended years
./run_experiment.sh check-availability green 2019 3  # ✅ Works
./run_experiment.sh check-availability yellow 2017 8  # ❌ Fails
```

#### **2. AWS Lambda Timeout**
```
Error: Lambda timeout after 15 minutes
Solutions:
- Reduce simulation_range (use 2-5 instead of 10+)
- Reduce num_eval (use 50-100 instead of 500+)
- Use smaller datasets
```

### ✅ **Data Availability Guide**

| Vehicle Type | Available Years | Coverage | Best Performance |
|--------------|----------------|----------|------------------|
| **Green** | 2017-2023 | Complete ✅ | All years work |
| **Yellow** | 2018-2023 | Nearly complete ✅ | 2019-2023 optimal |
| **FHV** | 2018-2023 | Complete ✅ | 2019-2023 optimal |

### 🎯 **Best Practices**

1. **Always check availability first**:
   ```bash
   ./run_experiment.sh check-availability <type> <year> <month>
   ```

2. **Use 2019 for guaranteed success**:
   ```bash
   ./run_experiment.sh download-bulk 2019 1 3 green,yellow,fhv  # ✅ Always works
   ```

3. **Pre-validate bulk operations** - system warns about failures

---

## 🎯 **Quick Reference Card**

### 🚀 **Essential Commands**
```bash
# Check → Download → Experiment → Analyze
./run_experiment.sh check-availability green 2019 3
./run_experiment.sh download-single green 2019 3  
./run_experiment.sh run-comparative green 2019 3 PL 5
./run_experiment.sh analyze <experiment_id>
```

### 📊 **Parameter Defaults**
- **simulation_range**: 3-5 scenarios
- **num_eval**: 100 Monte Carlo evaluations  
- **window_time**: 300 seconds
- **acceptance_function**: PL (Piecewise Linear)
- **alpha**: 18 (algorithm parameter)

### 🏆 **Algorithm Performance**
- **LINEAR_PROGRAM**: Highest revenue (88-91% efficiency)
- **PROPOSED**: Best balance (85-91% efficiency)  
- **MAPS**: Market dynamics (65-77% efficiency)
- **LINUCB**: Learning scenarios (75-84% efficiency)

---

**🎉 This comprehensive documentation provides everything needed to run sophisticated bipartite matching experiments with detailed understanding of all parameters and output metrics.** 