# 📁 Taxi Pricing Benchmark - Project Structure
git 
## 🎯 **Refactored System Overview**

This is a completely refactored taxi pricing benchmarking system designed for systematic comparison of 4 pricing methods with full configurability and AWS cloud integration.

## 🚀 **Quick Start**

```bash
# Install dependencies
pip install -r requirements.txt

# Create example configurations
python cli.py create-examples

# Run Hikima replication experiment
python cli.py hikima-replication

# Run extended benchmark
python cli.py extended-benchmark --days 30

# Run custom experiment
python cli.py custom --methods HikimaMinMaxCostFlow MAPS --start-hour 8 --end-hour 18
```

## 🏗️ **Core System Structure**

```
taxi-pricing-benchmark/
├── 🎯 cli.py                          # MAIN CLI INTERFACE
├── 📋 requirements.txt                # Python dependencies
├── 📖 README.md                       # Complete documentation
│
├── 📁 config/                         # Configuration System
│   ├── experiment_config.py           # Complete configuration framework
│   └── __init__.py
│
├── 📁 src/                           # Core Implementation
│   ├── __init__.py
│   ├── orchestrator.py                # Main benchmarking orchestrator
│   └── pricing_methods/               # 4 Pricing Method Implementations
│       ├── __init__.py
│       ├── hikima_method.py           # Hikima MinMax Cost Flow
│       ├── maps_method.py             # Multi-Area Pricing Strategy
│       ├── linucb_method.py           # Linear Upper Confidence Bound
│       └── linear_program_method.py   # Linear Programming (Gupta-Nagarajan)
│
├── 📁 configs/                       # Configuration Files
│   ├── hikima_replication.json       # Exact Hikima setup
│   ├── extended_benchmark_100days.json # Extended analysis
│   └── default.json                  # Default configuration
│
├── 📁 results/                       # Experiment Results
│   └── [Generated experiment files]
│
├── 📁 lambdas/                       # AWS Lambda Functions (Optional)
│   ├── data-ingestion/               # NYC data download
│   └── experiment-runner/            # Cloud-based experiments
│
└── 📁 docs/                          # Legacy documentation (archived)
```

## 🎯 **Key Entry Points**

### **1. Main CLI Interface**
```bash
python cli.py <command> [options]
```
- **Primary interface** for all experiments
- **Complete help system** with examples
- **Configuration validation**
- **Result management**

### **2. Configuration System**
```python
from config.experiment_config import create_hikima_replication_config
config = create_hikima_replication_config()
```
- **No hardcoded values** - everything configurable
- **Multiple pre-defined setups** (Hikima, extended, custom)
- **JSON-based configuration files**
- **Comprehensive validation**

### **3. Benchmarking Orchestrator**
```python
from src.orchestrator import BenchmarkOrchestrator
orchestrator = BenchmarkOrchestrator(config)
result = orchestrator.run_benchmark(requesters_data, taxis_data)
```
- **Parallel execution** of pricing methods
- **Comprehensive result collection**
- **Statistical analysis**
- **Performance ranking**

## 📊 **The 4 Benchmarked Methods**

### **1. Hikima MinMax Cost Flow** (`hikima_method.py`)
- **Source**: Extracted from `experiment_PL.py`
- **Algorithm**: Min-cost flow with delta-scaling
- **Features**: Exact mathematical implementation
- **Acceptance Functions**: Piecewise Linear, Sigmoid

### **2. MAPS** (`maps_method.py`)
- **Source**: Extracted from `experiment_PL.py` and `experiment_sigmoid.py`
- **Algorithm**: Multi-area bipartite matching with iterative improvement
- **Features**: Geographic area-based pricing
- **Acceptance Functions**: Piecewise Linear, Sigmoid

### **3. LinUCB** (`linucb_method.py`)
- **Source**: Extracted from experiment sources
- **Algorithm**: Contextual bandits with upper confidence bounds
- **Features**: Feature-based learning, confidence intervals
- **Parameters**: Configurable α, feature sets

### **4. Linear Programming** (`linear_program_method.py`)
- **Source**: Gupta-Nagarajan theoretical formulation
- **Algorithm**: Exact LP formulation using PuLP
- **Features**: Optimal solution with discrete price grids
- **Fallback**: Greedy approximation when PuLP unavailable

## ⚙️ **Configuration Features**

### **No Hardcoded Values**
- ✅ Time ranges (start_hour, end_hour) - user controlled
- ✅ Acceptance function parameters - configurable
- ✅ Method-specific parameters - adjustable
- ✅ Data processing thresholds - customizable
- ✅ AWS settings - optional

### **Experiment Types**
1. **Hikima Replication**: Exact paper setup (2 days, 10:00-20:00, 5-min)
2. **Extended Benchmark**: Long-term analysis (30-365+ days)
3. **Custom Experiments**: User-defined parameters

### **Method Configurations**
```json
{
  "hikima_config": {
    "alpha": 18.0,
    "s_taxi": 25.0,
    "acceptance_type": "PL"
  },
  "maps_config": {
    "s_0_rate": 1.5,
    "price_discretization_rate": 0.05
  },
  "linucb_config": {
    "ucb_alpha": 0.5,
    "base_price": 5.875
  },
  "lp_config": {
    "price_grid_size": 10,
    "solver_timeout": 300
  }
}
```

## 📈 **Experiment Results**

### **Output Structure**
Each experiment generates comprehensive JSON results:

```json
{
  "experiment_id": "benchmark_20241201_143022",
  "objective_values": {
    "HikimaMinMaxCostFlow": 1250.75,
    "MAPS": 1180.32,
    "LinUCB": 1145.89,
    "LinearProgram": 1195.44
  },
  "computation_times": {
    "HikimaMinMaxCostFlow": 2.145,
    "MAPS": 1.876,
    "LinUCB": 0.532,
    "LinearProgram": 3.221
  },
  "performance_ranking": ["HikimaMinMaxCostFlow", "LinearProgram", "MAPS", "LinUCB"]
}
```

### **Comparative Analysis**
- **Performance ranking** by objective value
- **Efficiency analysis** (objective/time)
- **Statistical significance** testing
- **Method-specific metrics** (convergence, acceptance rates)

## 🛠️ **Development Workflow**

### **1. Quick Testing**
```bash
# Test all methods with small dataset
python cli.py custom --requesters 50 --taxis 40

# Test specific methods
python cli.py custom --methods HikimaMinMaxCostFlow LinearProgram
```

### **2. Configuration Development**
```bash
# Validate configuration
python cli.py validate --config configs/my_config.json

# Create from examples
python cli.py create-examples
```

### **3. Research Experiments**
```bash
# Academic replication
python cli.py hikima-replication

# Extended analysis
python cli.py extended-benchmark --days 100 --time-window 30
```

## 🔧 **Advanced Features**

### **Parallel Execution**
- Methods run in parallel using `ThreadPoolExecutor`
- Independent failure handling
- Comprehensive error reporting

### **AWS Integration** (Optional)
- S3 data storage and retrieval
- Lambda-based cloud computing
- CloudWatch monitoring

### **Extensibility**
- Add new pricing methods easily
- Custom acceptance functions
- Pluggable data sources

## 📚 **Academic Compliance**

### **Hikima Methodology**
- ✅ **Exact replication** of experimental setup
- ✅ **Same parameters** (α=18, s_taxi=25)
- ✅ **Both acceptance functions** (PL, Sigmoid)
- ✅ **No artificial data manipulation**

### **Research Extensions**
- 📊 **Scalable experiments** (100+ days)
- 🌍 **Geographic flexibility**
- ⚙️ **Parameter sensitivity analysis**
- 📈 **Statistical significance testing**

## 🎯 **Key Advantages of Refactored System**

1. **🚫 No Hardcoded Values**: Everything user-configurable
2. **🔬 Exact Academic Compliance**: Faithful to original Hikima methodology
3. **📊 Comprehensive Benchmarking**: 4 methods with detailed comparison
4. **⚡ Parallel Execution**: Efficient computation
5. **🛠️ Easy Extension**: Add new methods or parameters
6. **☁️ Cloud Ready**: Optional AWS integration
7. **📈 Rich Analysis**: Statistical testing and performance ranking

---

**🏆 This refactored system provides a robust, scalable, and academically compliant platform for taxi pricing optimization research!** 