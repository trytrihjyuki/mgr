# Experiment Usage Guide

## 🚀 **Quick Start**

This guide shows you how to run different types of rideshare pricing experiments using real NYC taxi data.

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

### **2. 24-Hour Full Day Experiment**

**Use Case**: Understand demand patterns across entire day
**Duration**: 00:00-23:59 (full 24 hours)
**Best For**: Market analysis, demand forecasting

```json
{
  "vehicle_type": "yellow", 
  "year": 2019,
  "month": 6,
  "full_day_experiment": true,
  "simulation_range": 6,
  "acceptance_function": "Sigmoid"
}
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
  "full_day_experiment": true,
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
  "full_day_experiment": true,
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
  "full_day_experiment": true
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
  "full_day_experiment": true
}
```

**Outer Boroughs**:
```json
{
  "vehicle_type": "green",  // Green taxis serve outer boroughs
  "year": 2019,
  "month": 3,
  "place": "Brooklyn",
  "full_day_experiment": true
}
```

**Airport Analysis**:
```json
{
  "vehicle_type": "yellow",
  "year": 2019,
  "month": 3,
  "focus_zones": ["JFK Airport", "LaGuardia Airport"],
  "full_day_experiment": true
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

### **24-Hour Experiments**
```yaml
Memory: 1024 MB
Timeout: 10 minutes
Concurrency: 10
Use Case: Full day analysis
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
| **24-Hour** | 50-150 MB | 5-8 minutes | 1024 MB |
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
    "full_day_experiment": true,
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
    "full_day_experiment": True,
    "simulation_range": 6
}

result = lambda_handler(event, None)
print(json.dumps(result, indent=2))
```

## 📈 **Best Practices**

### **Experiment Design**
1. **Start small**: Begin with standard experiments
2. **Scale gradually**: Move to 24-hour, then multi-day
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
- **Temporal analysis**: 24-hour and multi-day patterns

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