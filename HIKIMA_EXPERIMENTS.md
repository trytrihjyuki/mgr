# Hikima Pricing Benchmark Lambda Function

## Overview

This Lambda function implements the complete experimental environment from Hikima et al. paper on ride-hailing pricing optimization. It loads real NYC taxi data from S3 and implements 4 pricing methods:

1. **MinMaxCostFlow** - Exact Hikima et al. min-cost flow algorithm
2. **MAPS** - Area-based pricing with bipartite matching  
3. **LinUCB** - Contextual bandit learning with Upper Confidence Bound
4. **LP** - Gupta-Nagarajan Linear Program optimization

## Data Structure

The function loads real taxi data from S3 with the following structure:
```
s3://magisterka/datasets/{taxi_type}/year={year}/month={month:02d}/{taxi_type}_tripdata_{year}-{month:02d}.parquet
```

### Supported Taxi Types
- `yellow` - Yellow taxi data
- `green` - Green taxi data  
- `fhv` - For-hire vehicle data

### Supported Boroughs
- `Manhattan`
- `Brooklyn`
- `Queens`
- `Bronx`
- `Staten Island`

## Event Parameters

### Basic Parameters
```json
{
  "year": 2019,
  "month": 10,
  "day": 1,
  "borough": "Manhattan",
  "vehicle_type": "yellow",
  "acceptance_function": "PL",  // or "Sigmoid"
  "methods": ["MinMaxCostFlow", "MAPS", "LinUCB", "LP"]
}
```

### Time Window Configuration
```json
{
  "time_window": {
    "hour_start": 10,     // Start hour (24h format)
    "hour_end": 20,       // End hour (24h format)  
    "minute_start": 0,    // Start minute
    "time_interval": 5    // Interval in minutes
  },
  "scenario_index": 0     // Which 5-minute scenario (0-119 for 10h period)
}
```

### Experiment Metadata
```json
{
  "execution_date": "20241201_120000",
  "training_id": "experiment_12345"
}
```

## Example Experiments

### 1. Manhattan Yellow Taxi Experiment (Hikima Original)
```json
{
  "year": 2019,
  "month": 10,
  "day": 1,
  "borough": "Manhattan",
  "vehicle_type": "yellow",
  "acceptance_function": "PL",
  "time_window": {
    "hour_start": 10,
    "hour_end": 20,
    "minute_start": 0,
    "time_interval": 5
  },
  "scenario_index": 6,
  "methods": ["MinMaxCostFlow", "MAPS", "LinUCB", "LP"],
  "execution_date": "20241201_120000",
  "training_id": "manhattan_yellow_pl_001"
}
```

### 2. Multi-Day Brooklyn Green Taxi Experiment  
```json
{
  "year": 2019,
  "month": 10,
  "day": 6,
  "borough": "Brooklyn", 
  "vehicle_type": "green",
  "acceptance_function": "Sigmoid",
  "time_window": {
    "hour_start": 10,
    "hour_end": 20,
    "minute_start": 0,
    "time_interval": 5
  },
  "scenario_index": 24,
  "methods": ["MinMaxCostFlow", "MAPS"],
  "execution_date": "20241201_120000", 
  "training_id": "brooklyn_green_sigmoid_001"
}
```

## Response Structure

### Successful Response
```json
{
  "statusCode": 200,
  "body": {
    "training_id": "experiment_12345",
    "execution_date": "20241201_120000",
    "timestamp": "2024-12-01T12:00:00.000Z",
    "status": "success",
    "execution_time_seconds": 45.2,
    "s3_location": "s3://magisterka/experiments/...",
    
    "experiment_metadata": {
      "year": 2019,
      "month": 10,
      "day": 1,
      "borough": "Manhattan",
      "time_window": {
        "start": "2019-10-01T10:30:00",
        "end": "2019-10-01T10:35:00", 
        "duration_minutes": 5,
        "scenario_index": 6
      },
      "vehicle_type": "yellow",
      "acceptance_function": "PL",
      "methods": ["MinMaxCostFlow", "MAPS", "LinUCB", "LP"]
    },
    
    "data_statistics": {
      "num_requesters": 42,
      "num_taxis": 38,
      "ratio_requests_to_taxis": 1.11,
      "avg_trip_distance_km": 3.24,
      "avg_trip_amount": 12.45,
      "avg_trip_duration_seconds": 875
    },
    
    "performance_summary": {
      "total_objective_value": 156.78,
      "total_computation_time": 23.4,
      "avg_computation_time": 5.85,
      "methods": {
        "MinMaxCostFlow": {
          "objective_value": 45.23,
          "computation_time": 8.2,
          "success": true
        },
        "MAPS": {
          "objective_value": 42.11,
          "computation_time": 6.1,
          "success": true
        },
        "LinUCB": {
          "objective_value": 38.44,
          "computation_time": 4.8,
          "success": true
        },
        "LP": {
          "objective_value": 31.00,
          "computation_time": 4.3,
          "success": true
        }
      }
    },
    
    "results": [
      {
        "method_name": "MinMaxCostFlow",
        "objective_value": 45.23,
        "computation_time": 8.2,
        "num_requests": 42,
        "num_taxis": 38,
        "avg_acceptance_rate": 0.73
      }
      // ... other methods
    ]
  }
}
```

## S3 Output Structure

Results are saved to S3 with the following key pattern:
```
experiments/type={vehicle_type}/eval={acceptance_function}/borough={borough}/year={year}/month={month:02d}/day={day:02d}/{execution_date}_{training_id}_scenario{scenario_index}.json
```

Example:
```
experiments/type=yellow/eval=PL/borough=Manhattan/year=2019/month=10/day=01/20241201_120000_manhattan_yellow_pl_001_scenario6.json
```

## Acceptance Functions

### Piecewise Linear (PL)
```
p(price) = -2.0/trip_amount * price + 3.0
```

### Sigmoid
```  
p(price) = 1 - 1/(1 + exp((-price + β*trip_amount)/(γ*trip_amount)))
```
Where β = 1.3, γ = 0.3√3/π

## Multi-Scenario Experiments

To run the full Hikima experiment (2019-10-01 to 2019-10-06, 10:00-20:00, 5min intervals):

### Time Scenarios
- Total scenarios per day: 120 (10 hours × 12 intervals/hour)
- Day 1 scenarios: 0-119
- Day 2 scenarios: 120-239  
- Day 6 scenarios: 600-719

### Example: Run scenario 6 (10:30-10:35) on day 1
```json
{
  "day": 1,
  "scenario_index": 6,
  // ... other parameters
}
```

### Example: Run scenario 24 (12:00-12:05) on day 1  
```json
{
  "day": 1,
  "scenario_index": 24,
  // ... other parameters
}
```

## Testing

Use test mode to validate setup:
```json
{
  "test_mode": true
}
```

This returns import status and basic configuration without running experiments.

## Error Handling

The function includes comprehensive fallbacks:
- Missing parquet files → CSV fallback → synthetic data
- Missing area information → generated coordinates
- Failed pricing methods → zero objective value with error details
- S3 failures → local result storage

## Dependencies

### Required
- boto3
- pandas  
- numpy
- networkx
- pulp
- scipy

### Optional
- pyarrow (for parquet support)

## Deployment

Deploy as AWS Lambda with:
- Runtime: Python 3.9+
- Memory: 1024MB+
- Timeout: 15 minutes
- Environment: S3_BUCKET=magisterka 