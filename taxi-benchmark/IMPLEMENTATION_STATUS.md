# Implementation Status - Taxi Benchmark Framework

## ✅ Complete Implementation of Hikima et al. Pricing Methods

This repository now contains a **fully functional implementation** of all pricing methods from the Hikima et al. paper "Dynamic Pricing in Ride-Hailing Platforms: A Reinforcement Learning Approach" and its experimental baselines.

## Implemented Methods

### 1. **MinMaxCostFlow Method** (Main Hikima Contribution) ✓
- **Location**: `src/methods/minmax_costflow.py`
- **Description**: The core algorithm from Hikima et al. using min-cost flow with delta-scaling
- **Key Features**:
  - Delta-scaling algorithm for efficient computation
  - Supports both PL (piecewise linear) and Sigmoid acceptance functions
  - Handles flow-based price optimization
  - Dijkstra's algorithm for shortest path computation

### 2. **MAPS Method** (Baseline) ✓
- **Location**: `src/methods/maps.py`
- **Description**: Matching And Pricing in Shared economy (Tong et al.)
- **Key Features**:
  - Greedy matching algorithm
  - Area-based pricing strategy
  - Supply-demand ratio optimization
  - Bipartite matching within radius constraints

### 3. **LinUCB Method** (Baseline) ✓
- **Location**: `src/methods/linucb.py`
- **Description**: Linear Upper Confidence Bound (Chu et al.)
- **Key Features**:
  - Multi-armed bandit approach
  - Online learning with parameter updates
  - Feature extraction (time, location, trip characteristics)
  - Confidence-based exploration
  - **S3 Integration**: Automatically loads pre-trained matrices from S3
    - Path: `s3://taxi-benchmark/models/work/learned_matrix_PL/{month}_{borough}/`
    - Supports combining matrices from multiple months (201907, 201908, 201909)

### 4. **LP Method** (Additional) ✓
- **Location**: `src/methods/lp.py`
- **Description**: Linear Programming approach (Gupta-Nagarajan)
- **Note**: Not in original Hikima paper but included as additional method
- **Key Features**:
  - Price grid discretization
  - PuLP solver integration
  - Probing variables for rider-price pairs

## Framework Alignment with Hikima Experiments

### Data Processing Pipeline ✓
- NYC taxi data loader (`src/data/loader.py`)
- Scenario generation matching Hikima's setup
- Support for green and yellow taxi data
- Time window-based sampling (5-minute intervals)

### Dual Acceptance Function Evaluation ✓ (NEW)
**The framework now automatically evaluates BOTH acceptance functions for every experiment:**
1. **Piecewise Linear (PL/ReLU)**: `-2/amount * price + 3`
2. **Sigmoid**: `1 / (1 + exp(-(beta*amount - price)/(gamma*amount)))`

This ensures comprehensive comparison across both acceptance models without needing to run separate experiments.

### Parameter Alignment ✓
All parameters match the original paper:
- `alpha = 18.0` (cost parameter)
- `s_taxi = 25.0` km/h (taxi speed)
- `epsilon = 1e-10` (numerical tolerance)
- `beta = 1.3` (sigmoid parameter)
- `gamma = 0.276` (≈ 0.3*√3/π)

### Evaluation Framework ✓
- Monte Carlo simulations for performance evaluation
- Bipartite matching for taxi-requester assignment
- Revenue calculation considering acceptance probabilities
- Computation time tracking

## Testing

Run the test script to verify all methods are working:
```bash
python3 test_methods.py
```

This will test:
- All method implementations
- Factory pattern creation
- Edge cases (empty scenarios, no taxis)
- Price computation correctness

## Project Structure

```
taxi-benchmark/
├── src/
│   ├── methods/
│   │   ├── base.py              # Abstract base class
│   │   ├── minmax_costflow.py   # Hikima main method
│   │   ├── maps.py              # MAPS baseline
│   │   ├── linucb.py           # LinUCB baseline
│   │   ├── lp.py               # LP method (additional)
│   │   └── factory.py          # Method factory
│   ├── data/
│   │   ├── loader.py           # Data loading
│   │   ├── processor.py        # Data processing
│   │   └── validator.py        # Data validation
│   └── core/
│       ├── config.py           # Configuration management
│       ├── types.py            # Type definitions
│       └── logging.py          # Logging utilities
├── test_methods.py             # Test all implementations
├── run_experiment.py           # Main experiment runner
└── requirements.txt            # Python dependencies
```

## Running Experiments

### Basic Usage
```python
from src.methods import MinMaxCostFlowMethod, MAPSMethod, LinUCBMethod

# Configuration (no acceptance_function needed - both are evaluated)
config = {
    'alpha': 18.0,
    's_taxi': 25.0,
    'sigmoid_beta': 1.3,
    'sigmoid_gamma': 0.276,
    # ... other parameters
}

# Scenario data
scenario_data = {
    'num_requesters': n_requesters,
    'num_taxis': n_taxis,
    'trip_amounts': trip_valuations,
    'edge_weights': edge_weights,
    # ... other data
}

# Run method - automatically evaluates both PL and Sigmoid
method = MinMaxCostFlowMethod(config)
results = method.execute(scenario_data, num_simulations=100)

# Access dual results
pl_revenue = results['PL']['avg_revenue']
sigmoid_revenue = results['Sigmoid']['avg_revenue']
prices = results['prices']  # Same prices used for both evaluations
```

### Full Experiment
```bash
python3 run_experiment.py --config configs/experiment.yaml
```

## Key Improvements from Original

1. **Modular Design**: Clean separation of methods with common base class
2. **Type Safety**: Strong typing with dataclasses and enums
3. **Factory Pattern**: Easy method instantiation
4. **Comprehensive Testing**: Test suite for all methods
5. **Documentation**: Extensive inline documentation
6. **Edge Case Handling**: Robust handling of empty/edge scenarios

## Dependencies

Main requirements:
- numpy: Numerical computations
- networkx: Graph algorithms (matching)
- pulp: Linear programming (LP method)
- pandas: Data processing
- pyarrow: Parquet file support
- boto3: S3 integration

## Notes

- All empty directories have been removed for cleaner structure
- Methods are fully aligned with Hikima et al.'s experimental setup
- Ready for benchmarking and comparative analysis
- Supports both local and cloud (EC2/S3) execution

## References

1. Hikima et al. "Dynamic Pricing in Ride-Hailing Platforms: A Reinforcement Learning Approach"
2. Tong et al. "Dynamic Pricing in Spatial Crowdsourcing: A Matching-Based Approach" (MAPS)
3. Chu et al. "A Contextual-Bandit Approach to Personalized News Article Recommendation" (LinUCB)
4. Gupta & Nagarajan "Approximation Algorithms for Dynamic Resource Allocation" (LP) 