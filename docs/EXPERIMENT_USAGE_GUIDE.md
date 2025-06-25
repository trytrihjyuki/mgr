# Experiment Usage Guide

## 🚀 **Quick Start**

This guide shows you how to run different types of rideshare pricing experiments using real NYC taxi data.

## 🏃‍♂️ **Ready-to-Run Examples**

**Copy these commands and run them directly!**

### **🟢 Basic Examples (5 minutes each)**

```bash
# 1. Simple Hikima Paper Replication (Manhattan, Business Hours)
./run_experiment.sh run-experiment 10 20 5m Manhattan 30s 10 6 2019 green "hikima,maps,linucb,linear_program" PL

# 2. Fast Test Run (2 scenarios only)
./run_experiment.sh run-experiment 15 17 30m Manhattan 30s 3 15 2019 yellow "hikima,maps" PL

# 3. Different Borough - Brooklyn Analysis
./run_experiment.sh run-experiment 10 20 5m Brooklyn 30s 10 6 2019 green "hikima,maps,linucb" PL

# 4. Sigmoid Acceptance Function Test
./run_experiment.sh run-experiment 10 20 10m Queens 60s 10 6 2019 green "hikima,maps" Sigmoid
```

### **⚡ Rush Hour Analysis**

```bash
# Morning Rush Hour (7-10 AM)
./run_experiment.sh run-experiment 7 10 15m Manhattan 30s 1 6 2019 yellow "hikima,maps,linucb" PL

# Evening Rush Hour (17-20 PM) 
./run_experiment.sh run-experiment 17 20 10m Manhattan 30s 1 6 2019 yellow "hikima,maps,linucb" PL

# Extended Rush Hours (7-10 AM + 17-20 PM combined analysis)
./run_experiment.sh run-experiment 7 10 15m Manhattan 30s 1 6 2019 yellow "hikima,maps" PL
./run_experiment.sh run-experiment 17 20 15m Manhattan 30s 1 6 2019 yellow "hikima,maps" PL
```

### **🌙 24-Hour Analysis**

```bash
# Full Day Analysis (all 24 hours)
./run_experiment.sh run-experiment 0 24 60m Manhattan 60s 10 6 2019 green "hikima,maps" PL

# Night-time Analysis (22:00-06:00)
./run_experiment.sh run-experiment 22 24 30m Manhattan 60s 10 6 2019 green "hikima,maps" PL
./run_experiment.sh run-experiment 0 6 30m Manhattan 60s 10 6 2019 green "hikima,maps" PL

# Business Hours Extended (8 AM - 10 PM)
./run_experiment.sh run-experiment 8 22 30m Manhattan 30s 10 6 2019 yellow "hikima,maps,linucb" PL
```

### **🗓️ Multi-Day Experiments** 

```bash
# Week Analysis (Monday-Sunday, March 2019)
./run_experiment.sh run-multi-month 10 20 30m Manhattan 60s "3" "1,2,3,4,5,6,7" 2019 green "hikima,maps" PL

# Weekend vs Weekday Comparison
# Weekdays (Mon-Fri)
./run_experiment.sh run-multi-month 0 24 60m Manhattan 60s "3" "2,3,4,5,6" 2019 green "hikima,maps" PL
# Weekend (Sat-Sun)
./run_experiment.sh run-multi-month 0 24 60m Manhattan 60s "3" "7,1" 2019 green "hikima,maps" PL

# Multiple Months Analysis (March, April, May)
./run_experiment.sh run-multi-month 10 20 30m Manhattan 60s "3,4,5" "10,15,20" 2019 green "hikima,maps" PL
```

### **🏙️ All Boroughs Comparison**

```bash
# Manhattan
./run_experiment.sh run-experiment 10 20 15m Manhattan 30s 10 6 2019 yellow "hikima,maps,linucb" PL

# Brooklyn  
./run_experiment.sh run-experiment 10 20 15m Brooklyn 30s 10 6 2019 green "hikima,maps,linucb" PL

# Queens
./run_experiment.sh run-experiment 10 20 15m Queens 30s 10 6 2019 green "hikima,maps,linucb" PL

# Bronx
./run_experiment.sh run-experiment 10 20 15m Bronx 30s 10 6 2019 green "hikima,maps,linucb" PL
```

### **🔬 Algorithm Comparison Studies**

```bash
# All 4 Algorithms Head-to-Head
./run_experiment.sh run-experiment 10 20 15m Manhattan 30s 10 6 2019 green "hikima,maps,linucb,linear_program" PL

# Hikima vs MAPS Only
./run_experiment.sh run-experiment 10 20 10m Manhattan 30s 10 6 2019 green "hikima,maps" PL

# PL vs Sigmoid Acceptance Function Comparison
./run_experiment.sh run-experiment 10 20 15m Manhattan 30s 10 6 2019 green "hikima,maps" PL
./run_experiment.sh run-experiment 10 20 15m Manhattan 30s 10 6 2019 green "hikima,maps" Sigmoid
```

### **📅 Seasonal Analysis (Different Months)**

```bash
# Winter Analysis (January)
./run_experiment.sh run-experiment 10 20 20m Manhattan 60s 1 15 2019 green "hikima,maps" PL

# Spring Analysis (April)  
./run_experiment.sh run-experiment 10 20 20m Manhattan 60s 4 15 2019 green "hikima,maps" PL

# Summer Analysis (July)
./run_experiment.sh run-experiment 10 20 20m Manhattan 60s 7 15 2019 green "hikima,maps" PL

# Fall Analysis (October)
./run_experiment.sh run-experiment 10 20 20m Manhattan 60s 10 15 2019 green "hikima,maps" PL
```

### **🚀 High-Performance/Large-Scale Experiments**

```bash
# High-Resolution Analysis (1-minute intervals)
./run_experiment.sh run-experiment 15 18 1m Manhattan 30s 10 6 2019 yellow "hikima,maps" PL

# Large Scenario Count (30 scenarios)
./run_experiment.sh run-experiment 10 20 6m Manhattan 30s 30 6 2019 green "hikima,maps" PL

# Extended Time Range (6 AM - 11 PM)
./run_experiment.sh run-experiment 6 23 30m Manhattan 60s 15 6 2019 yellow "hikima,maps,linucb" PL
```

## 📊 **Command Format Explanation**

```bash
./run_experiment.sh run-experiment [start_hour] [end_hour] [time_interval] [place] [time_step] [month] [day] [year] [vehicle_type] [methods] [acceptance_func]
```

**Parameters**:
- `start_hour`: 0-23 (e.g., 10 = 10:00 AM)
- `end_hour`: 1-24 (e.g., 20 = 8:00 PM) 
- `time_interval`: 5m, 10m, 30m, 60m (time window size)
- `place`: Manhattan, Brooklyn, Queens, Bronx
- `time_step`: 30s, 60s (simulation granularity)
- `month`: 1-12 (data month)
- `day`: 1-31 (specific day)
- `year`: 2019, 2020, etc. (data year)
- `vehicle_type`: green, yellow, fhv
- `methods`: "hikima,maps,linucb,linear_program" (algorithms to test)
- `acceptance_func`: PL, Sigmoid (acceptance function type)

## ⚡ **Performance Expectations**

| Experiment Type | Duration | Scenarios | Est. Time | Memory |
|-----------------|----------|-----------|-----------|---------|
| **Basic (10-20h)** | 10 hours | 5-10 | 3-5 min | 512MB |
| **Rush Hour** | 3 hours | 12 | 2-3 min | 512MB |
| **24-Hour** | 24 hours | 24 | 8-12 min | 1024MB |
| **Multi-Day** | 7 days | 50+ | 15+ min | 2048MB |
| **High-Res** | 3 hours (1m) | 180 | 10-15 min | 1024MB |

## 🔧 **Quick Reference - Common Patterns**

### **Most Used Commands**
```bash
# Quick Test (3 minutes)
./run_experiment.sh run-experiment 15 17 30m Manhattan 30s 3 15 2019 yellow "hikima,maps" PL

# Standard Research (Hikima Paper)
./run_experiment.sh run-experiment 10 20 5m Manhattan 30s 10 6 2019 green "hikima,maps,linucb" PL

# Full Day Analysis
./run_experiment.sh run-experiment 0 24 60m Manhattan 60s 10 6 2019 green "hikima,maps" PL
```

### **Parameter Quick Reference**
```bash
# Time Ranges (Popular)
start_hour=7  end_hour=10    # Morning Rush
start_hour=17 end_hour=20    # Evening Rush  
start_hour=10 end_hour=20    # Business Hours (Hikima)
start_hour=0  end_hour=24    # Full Day

# Boroughs
Manhattan  # Yellow taxis primarily
Brooklyn   # Green taxis primarily
Queens     # Green taxis primarily
Bronx      # Green taxis primarily

# Vehicle Types
yellow     # Manhattan focused
green      # Outer boroughs
fhv        # For hire vehicles

# Methods Combinations
"hikima,maps"                      # Fast comparison
"hikima,maps,linucb"               # 3-way comparison
"hikima,maps,linucb,linear_program" # Full comparison
```

## 🚨 **Troubleshooting Common Issues**

### **❌ Command Not Found**
```bash
# Problem: ./run_experiment.sh: command not found
# Solution: Make script executable
chmod +x run_experiment.sh
```

### **❌ AWS Lambda Timeout**
```bash
# Problem: Function timed out after X seconds
# Solution: Reduce experiment size or increase timeout

# Instead of:
./run_experiment.sh run-experiment 0 24 1m Manhattan 30s 30 6 2019 green "hikima,maps,linucb,linear_program" PL

# Use:
./run_experiment.sh run-experiment 10 20 10m Manhattan 30s 10 6 2019 green "hikima,maps" PL
```

### **❌ Memory Issues**
```bash
# Problem: Memory exceeded
# Solution: Reduce scenarios or time range

# Instead of large scenarios:
simulation_range=30

# Use smaller:
simulation_range=5
```

### **❌ No Data Found**
```bash
# Problem: No data available for specified parameters
# Solutions:
# 1. Check if data exists for that month/year
# 2. Use known good dates:

# Good dates for testing:
month=3  day=15 year=2019  # March 15, 2019 (Friday)
month=6  day=10 year=2019  # June 10, 2019 (Monday)
month=10 day=5  year=2019  # October 5, 2019 (Saturday)
```

### **❌ S3 Access Issues**
```bash
# Problem: Access denied to S3 bucket
# Solution: Check AWS credentials
aws sts get-caller-identity  # Check if AWS CLI is configured
aws s3 ls s3://magisterka/   # Test S3 access
```

## 💡 **Pro Tips**

### **🚀 Speed Up Development**
```bash
# Use small test runs during development
./run_experiment.sh run-experiment 15 17 30m Manhattan 30s 2 15 2019 yellow "hikima" PL

# Then scale up for production
./run_experiment.sh run-experiment 10 20 10m Manhattan 30s 10 6 2019 yellow "hikima,maps,linucb" PL
```

### **📊 Batch Processing**
```bash
# Run multiple experiments in sequence
./run_experiment.sh run-experiment 10 20 10m Manhattan 30s 10 6 2019 green "hikima,maps" PL && \
./run_experiment.sh run-experiment 10 20 10m Brooklyn 30s 10 6 2019 green "hikima,maps" PL && \
./run_experiment.sh run-experiment 10 20 10m Queens 30s 10 6 2019 green "hikima,maps" PL
```

### **💾 Save Results**
```bash
# Redirect output to file for later analysis
./run_experiment.sh run-experiment 10 20 10m Manhattan 30s 10 6 2019 green "hikima,maps" PL | tee experiment_results.log
```

### **⏱️ Track Execution Time**
```bash
# Time your experiments
time ./run_experiment.sh run-experiment 10 20 10m Manhattan 30s 10 6 2019 green "hikima,maps" PL
```

## 📋 **Prerequisites**

1. **AWS Account** with Lambda and S3 access
2. **NYC TLC Data** uploaded to S3 bucket
3. **Taxi Zone Data** available (uploaded via `upload_area_info.py`)
4. **Lambda Function** deployed with experiment code

## 🧪 **Experiment Types**

### **1. Standard Business Hours Experiment**

**Use Case**: Replicate Hikima paper methodology exactly
**Duration**: 10:00-20:00 (business hours only)
**Best For**: Academic research, paper comparisons

```json
{
  "vehicle_type": "green",
  "year": 2019,
  "month": 3,
  "place": "Manhattan",
  "simulation_range": 5,
  "acceptance_function": "PL"
}
```

**Expected Results**:
```json
{
  "experiment_type": "hikima_compliant",
  "data_info": {
    "business_hours_filtered": true,
    "time_range": "10:00-20:00"
  },
  "results": {
    "average_match_rate": 0.75,
    "average_objective_value": 1250.45
  }
}
```

### **2. Custom Time Range Experiments**

**Use Case**: Analyze specific time periods (fully user-controlled)
**Duration**: Any time range you specify
**Best For**: Targeted analysis, rush hour studies, night patterns

```json
{
  "vehicle_type": "yellow", 
  "year": 2019,
  "month": 6,
  "start_hour": 0,     // Start at midnight
  "end_hour": 24,      // End at midnight (24-hour experiment)
  "simulation_range": 6,
  "acceptance_function": "Sigmoid"
}
```

**Other Time Range Examples**:
```json
// Rush hour only (7 AM - 10 AM)
{"start_hour": 7, "end_hour": 10}

// Business hours (9 AM - 6 PM)  
{"start_hour": 9, "end_hour": 18}

// Evening only (6 PM - 11 PM)
{"start_hour": 18, "end_hour": 23}
```

**Time Periods Analyzed**:
- **Night (00:00-06:00)**: Low demand, limited drivers
- **Early Morning (06:00-10:00)**: Commuter rush patterns
- **Morning Rush (10:00-14:00)**: High business demand
- **Afternoon (14:00-18:00)**: Steady mid-day patterns
- **Evening Rush (18:00-22:00)**: Peak demand/supply mismatch
- **Late Night (22:00-24:00)**: Entertainment/nightlife

**Expected Results**:
```json
{
  "time_analysis": {
    "experiment_duration": "24_hours",
    "time_periods": [
      {
        "period": "Evening Rush (18:00-22:00)",
        "avg_objective_value": 1580.32,
        "match_rate": 0.82,
        "demand_factor": 1.6
      },
      {
        "period": "Night (00:00-06:00)",
        "avg_objective_value": 890.45,
        "match_rate": 0.65,
        "demand_factor": 0.4
      }
    ]
  }
}
```

### **3. Multi-Day Comparative Study**

**Use Case**: Analyze patterns across multiple days
**Duration**: Specified day range (e.g., 1 week)
**Best For**: Temporal pattern analysis, weekly trends

```json
{
  "vehicle_type": "green",
  "year": 2019, 
  "month": 3,
  "multi_day_experiment": true,
  "start_day": 1,
  "end_day": 7,
  "start_hour": 0,      // 24-hour analysis
  "end_hour": 24,       // Full day coverage
  "simulation_range": 6
}
```

**Analysis Scope**:
- **7 days** of data aggregated
- **24-hour patterns** for each day
- **Daily variations** identified
- **Weekly trends** analyzed

**Expected Results**:
```json
{
  "time_analysis": {
    "days_analyzed": 7,
    "daily_patterns": {
      "day_1": {"avg_revenue": 1250.45, "peak_hour": "19:00"},
      "day_7": {"avg_revenue": 980.32, "peak_hour": "21:00"}
    },
    "weekly_trends": {
      "weekday_avg": 1180.45,
      "weekend_avg": 1350.67
    }
  }
}
```

## 🕐 **Time-Based Analysis Patterns**

### **Rush Hour Analysis**
```json
{
  "vehicle_type": "yellow",
  "year": 2019,
  "month": 3,
  "start_hour": 0,
  "end_hour": 24,
  "simulation_range": 6,
  "focus": "rush_hours"
}
```

**Key Insights**:
- **Morning Rush (07:00-10:00)**: Commuter patterns, predictable demand
- **Evening Rush (17:00-20:00)**: Highest revenue potential, supply shortages
- **Off-Peak (10:00-17:00)**: Steady demand, optimal efficiency

### **Weekend vs Weekday Patterns**
```json
{
  "vehicle_type": "green",
  "year": 2019,
  "month": 3,
  "multi_day_experiment": true,
  "start_day": 1,  // Monday
  "end_day": 7,    // Sunday
  "start_hour": 0,
  "end_hour": 24
}
```

**Pattern Differences**:
- **Weekdays**: Business-driven, rush hour peaks
- **Weekends**: Entertainment-driven, evening peaks, later night activity

## 🌍 **Geographic Analysis**

### **Borough-Specific Experiments**

**Manhattan Focus**:
```json
{
  "vehicle_type": "yellow",
  "year": 2019,
  "month": 3,
  "place": "Manhattan",
  "start_hour": 0,
  "end_hour": 24
}
```

**Outer Boroughs**:
```json
{
  "vehicle_type": "green",  // Green taxis serve outer boroughs
  "year": 2019,
  "month": 3,
  "place": "Brooklyn",
  "start_hour": 0,
  "end_hour": 24
}
```

**Airport Analysis**:
```json
{
  "vehicle_type": "yellow",
  "year": 2019,
  "month": 3,
  "focus_zones": ["JFK Airport", "LaGuardia Airport"],
  "start_hour": 0,
  "end_hour": 24
}
```

## 🔬 **Algorithm Comparison Studies**

### **All Three Methods**
```json
{
  "vehicle_type": "yellow",
  "year": 2019,
  "month": 3,
  "methods": ["hikima", "maps", "linucb"],
  "acceptance_function": "PL",
  "simulation_range": 5
}
```

### **Acceptance Function Comparison**
```json
[
  {
    "vehicle_type": "green",
    "year": 2019,
    "month": 3,
    "acceptance_function": "PL",
    "experiment_id": "pl_comparison"
  },
  {
    "vehicle_type": "green", 
    "year": 2019,
    "month": 3,
    "acceptance_function": "Sigmoid",
    "experiment_id": "sigmoid_comparison"
  }
]
```

## ⚡ **Lambda Configuration by Experiment Type**

### **Standard Experiments**
```yaml
Memory: 512 MB
Timeout: 5 minutes
Concurrency: 5
Use Case: Development, quick tests
```

### **Custom Time Range Experiments**
```yaml
Memory: 1024 MB
Timeout: 10 minutes
Concurrency: 10
Use Case: Any time range analysis
```

### **Multi-Day Experiments**
```yaml
Memory: 2048-3008 MB
Timeout: 15 minutes
Concurrency: 5
Use Case: Week/month analysis
```

### **Production Batch Processing**
```yaml
Memory: 3008 MB
Timeout: 15 minutes
Concurrency: 100+
Use Case: Large-scale research
```

## 📊 **Data Requirements by Experiment**

### **File Size Estimates** (from [NYC TLC data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page))

| Experiment Type | Data Size | Processing Time | Memory Needed |
|----------------|-----------|-----------------|---------------|
| **Single Day** | 5-15 MB | 2-3 minutes | 512 MB |
| **Custom Range** | 50-150 MB | 5-8 minutes | 1024 MB |
| **Multi-Day (7 days)** | 300-1000 MB | 10-15 minutes | 2048 MB |
| **Full Month** | 1-3 GB | 15+ minutes | 3008 MB |

### **Data Quality Considerations**

**2009 Data** (Larger files):
- Legacy CSV format
- More data quality issues
- Requires extra processing time

**2019+ Data** (Optimized):
- Parquet format (efficient)
- Clean, standardized data
- Faster processing

**2025+ Data** (Enhanced):
- Includes congestion pricing
- Additional metadata fields
- Modern compression

## 🔍 **Result Analysis Patterns**

### **Revenue Optimization Results**
```json
{
  "algorithm_comparison": {
    "hikima": {
      "avg_revenue": 1250.45,
      "rank": 1,
      "efficiency": "95%"
    },
    "maps": {
      "avg_revenue": 1180.32, 
      "rank": 2,
      "efficiency": "87%"
    },
    "linucb": {
      "avg_revenue": 1145.78,
      "rank": 3,
      "efficiency": "83%"
    }
  }
}
```

### **Geographic Performance Insights**
```json
{
  "geographic_analysis": {
    "Manhattan": {
      "avg_match_rate": 0.82,
      "peak_revenue_hour": "19:00",
      "optimal_algorithm": "hikima"
    },
    "Brooklyn": {
      "avg_match_rate": 0.71,
      "peak_revenue_hour": "20:00", 
      "optimal_algorithm": "maps"
    }
  }
}
```

### **Temporal Demand Patterns**
```json
{
  "temporal_analysis": {
    "rush_hour_performance": {
      "morning_rush": {"multiplier": 1.4, "duration": "07:00-10:00"},
      "evening_rush": {"multiplier": 1.8, "duration": "17:00-20:00"}
    },
    "off_peak_efficiency": {
      "midday": {"efficiency": 0.95, "time": "10:00-17:00"},
      "late_night": {"efficiency": 0.65, "time": "22:00-06:00"}
    }
  }
}
```

## 🚀 **Running Experiments**

### **Via API (Production)**
```bash
curl -X POST https://api.your-domain.com/experiment \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_type": "green",
    "year": 2019,
    "month": 3,
    "start_hour": 0,
    "end_hour": 24,
    "simulation_range": 6
  }'
```

### **Via AWS CLI**
```bash
aws lambda invoke \
  --function-name rideshare-experiment-runner \
  --payload '{"vehicle_type":"yellow","year":2019,"month":3,"multi_day_experiment":true,"start_day":1,"end_day":7}' \
  response.json
```

### **Local Testing**
```python
from lambda_function_heavy import lambda_handler

event = {
    "vehicle_type": "green",
    "year": 2019, 
    "month": 3,
    "start_hour": 0,
    "end_hour": 24,
    "simulation_range": 6
}

result = lambda_handler(event, None)
print(json.dumps(result, indent=2))
```

## 📈 **Best Practices**

### **Experiment Design**
1. **Start small**: Begin with standard experiments
2. **Scale gradually**: Move to larger time ranges, then multi-day
3. **Compare systematically**: Use same parameters across methods
4. **Document thoroughly**: Record all parameter choices

### **Performance Optimization**
1. **Memory sizing**: Match Lambda memory to experiment size
2. **Data sampling**: Use samples for development
3. **Batch processing**: Group related experiments
4. **Result caching**: Store intermediate results

### **Cost Management**
1. **Development**: Use smaller datasets and shorter time ranges
2. **Production**: Optimize Lambda memory allocation
3. **Bulk processing**: Use provisioned concurrency for regular runs
4. **Monitoring**: Track costs per experiment type

## 🎯 **Common Use Cases**

### **Academic Research**
- **Paper replication**: Standard Hikima experiments
- **Algorithm comparison**: All three methods on same data
- **Parameter sensitivity**: Vary acceptance functions
- **Temporal analysis**: Custom time ranges and multi-day patterns

### **Industry Applications**
- **Market analysis**: Geographic demand patterns
- **Pricing optimization**: Revenue maximization strategies
- **Operational planning**: Driver allocation optimization
- **Business intelligence**: Demand forecasting

### **Policy Research**
- **Congestion impact**: 2025+ data with congestion pricing
- **Geographic equity**: Cross-borough fairness analysis
- **Economic effects**: Revenue distribution studies
- **Transportation planning**: Integration with public transit

---

## 🏆 **Success Metrics**

### **Experiment Quality Indicators**
✅ **Match Rate > 70%**: Good demand/supply balance  
✅ **Objective Value Growth**: Algorithm effectiveness  
✅ **Geographic Coverage**: All boroughs represented  
✅ **Temporal Consistency**: Stable patterns across time  
✅ **Algorithm Convergence**: Stable results across runs  

This guide provides the foundation for conducting comprehensive, scientifically rigorous rideshare pricing optimization research using real-world NYC taxi data. 🚀 