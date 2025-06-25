# 🚕 Taxi Pricing Benchmark System

Systematic benchmarking and evaluation of taxi pricing optimization methods in artificial environments.

## 🎯 Project Purpose

This project provides a comprehensive benchmarking framework to systematically compare **4 different taxi pricing methods** against the same task of proposing optimal prices for clients in artificial ride-hailing environments. The framework is designed to replicate the methodology presented by Hikima et al. while extending it to enable large-scale comparative analysis.

## 📋 Benchmarked Methods

### 1. **Hikima MinMax Cost Flow** 
- Implementation of Hikima et al.'s min-cost flow approach
- Exact mathematical formulation from the original paper
- Supports both Piecewise Linear and Sigmoid acceptance functions

### 2. **MAPS (Multi-Area Pricing Strategy)**
- Area-based pricing optimization method
- Bipartite matching with geographic considerations
- Iterative improvement algorithm

### 3. **LinUCB (Linear Upper Confidence Bound)**
- Contextual bandits approach to pricing
- Feature-based learning with confidence bounds
- Real-time adaptation to market conditions

### 4. **Linear Programming (Gupta-Nagarajan)**
- Linear programming formulation for ride-hailing
- Exact implementation of the theoretical framework
- PuLP-based solver with fallback approximation

## 🏗️ System Architecture

```
taxi-pricing-benchmark/
├── 📁 config/                  # Configuration system
│   ├── experiment_config.py    # Complete configuration framework
│   └── examples/              # Example configuration files
├── 📁 src/                    # Core implementation
│   ├── pricing_methods/       # 4 pricing method implementations
│   │   ├── hikima_method.py
│   │   ├── maps_method.py
│   │   ├── linucb_method.py
│   │   └── linear_program_method.py
│   └── orchestrator.py        # Main benchmarking orchestrator
├── 📁 lambdas/               # AWS Lambda functions (optional)
├── 📁 results/               # Experiment results
├── 📁 configs/               # Configuration files
├── cli.py                    # Command-line interface
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd taxi-pricing-benchmark

# Install dependencies
pip install -r requirements.txt

# Create example configurations
python cli.py create-examples
```

### Basic Usage

```bash
# Run Hikima replication experiment (2 days, 10:00-20:00, 5-min windows)
python cli.py hikima-replication

# Run extended benchmark over 30 days
python cli.py extended-benchmark --days 30

# Run custom experiment with specific parameters
python cli.py custom --methods HikimaMinMaxCostFlow MAPS \
                     --start-hour 8 --end-hour 18 \
                     --requesters 500 --taxis 400

# List available methods
python cli.py list-methods

# Validate configuration
python cli.py validate --config configs/hikima_replication.json
```

## 📊 Experiment Types

### 1. **Hikima Replication Experiment**
Exact replication of the original Hikima et al. experimental setup:
- **Duration**: 2 days
- **Time Range**: 10:00-20:00 (business hours)
- **Time Windows**: 5-minute intervals
- **Purpose**: Validate implementation against original results

```bash
python cli.py hikima-replication --acceptance-function PL
```

### 2. **Extended Benchmark Experiment**
Comprehensive evaluation over longer periods:
- **Duration**: Configurable (30-365+ days)
- **Time Range**: 24-hour coverage
- **Time Windows**: Configurable intervals
- **Purpose**: Deep performance analysis and seasonal patterns

```bash
python cli.py extended-benchmark --days 100 --time-window 30
```

### 3. **Custom Experiments**
Flexible experimentation with user-defined parameters:
- **All parameters configurable**
- **Subset of methods**
- **Custom data sizes**
- **Specific time ranges**

```bash
python cli.py custom --name rush_hour_analysis \
                     --start-hour 7 --end-hour 10 \
                     --methods LinUCB LinearProgram
```

## ⚙️ Configuration System

The system uses a comprehensive configuration framework that **eliminates all hardcoded values**:

### Key Configuration Features
- **No hardcoded rush hours** - all time ranges user-controlled
- **Configurable acceptance functions** (Piecewise Linear, Sigmoid)
- **Method-specific parameters** fully configurable
- **Data processing parameters** adjustable
- **AWS integration** optional and configurable

### Example Configuration

```json
{
  "experiment_name": "taxi_pricing_benchmark",
  "methods_to_run": ["HikimaMinMaxCostFlow", "MAPS", "LinUCB", "LinearProgram"],
  "time_config": {
    "start_hour": 10,
    "end_hour": 20,
    "time_window_minutes": 5,
    "start_date": "2019-10-01",
    "end_date": "2019-10-02"
  },
  "hikima_config": {
    "alpha": 18.0,
    "s_taxi": 25.0,
    "acceptance_type": "PL"
  }
}
```

## 📈 Results and Analysis

### Output Structure
Each experiment produces comprehensive results:

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
  "performance_ranking": ["HikimaMinMaxCostFlow", "LinearProgram", "MAPS", "LinUCB"],
  "data_characteristics": {
    "n_requesters": 200,
    "n_taxis": 150,
    "time_range": "10:00-20:00"
  }
}
```

### Comparative Metrics
- **Objective Value**: Revenue/profit optimization
- **Computation Time**: Algorithm efficiency
- **Convergence**: Algorithm stability
- **Acceptance Rates**: Market response
- **Matching Efficiency**: Supply-demand balance

## 🛠️ Development and Extension

### Adding New Methods

1. Create method implementation in `src/pricing_methods/`
2. Implement standard interface with `solve()` method
3. Add to configuration system
4. Update CLI and orchestrator

### Custom Acceptance Functions

```python
def custom_acceptance_function(price, customer_data):
    # Implement custom logic
    return acceptance_probability
```

### AWS Integration (Optional)

The system supports AWS deployment for large-scale experiments:
- **S3** for data storage
- **Lambda** for compute
- **CloudWatch** for monitoring

## 📚 Academic Compliance

### Hikima Methodology Compliance
- ✅ **Exact mathematical formulation** from original paper
- ✅ **Same experimental setup** (2 days, 10:00-20:00, 5-min windows)
- ✅ **Identical parameters** (α=18, s_taxi=25, etc.)
- ✅ **Both acceptance functions** (PL and Sigmoid)
- ✅ **No artificial scaling** - uses raw data counts

### Research Extensions
- 📊 **Scalable to 100+ days** for comprehensive analysis
- 🌍 **Geographic flexibility** - any city/region
- ⚙️ **Parameter sensitivity** analysis
- 📈 **Statistical significance** testing

## 🔧 Technical Requirements

### Dependencies
```txt
numpy>=1.21.0
pandas>=1.3.0
networkx>=2.6.0
pulp>=2.5.0              # For Linear Programming method
boto3>=1.18.0            # For AWS integration (optional)
```

### System Requirements
- **Python**: 3.8+
- **Memory**: 4GB+ recommended for large experiments
- **Storage**: Varies with experiment size
- **AWS Account**: Optional for cloud deployment

## 📖 Examples and Tutorials

### Example 1: Quick Comparison
```bash
# Compare all methods on default parameters
python cli.py custom --methods HikimaMinMaxCostFlow MAPS LinUCB LinearProgram
```

### Example 2: Parameter Sensitivity
```bash
# Test different acceptance functions
python cli.py custom --acceptance-function PL --name "piecewise_linear_test"
python cli.py custom --acceptance-function Sigmoid --name "sigmoid_test"
```

### Example 3: Scalability Test
```bash
# Large-scale experiment
python cli.py custom --requesters 1000 --taxis 800 --name "scalability_test"
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/new-method`)
3. **Implement** changes with tests
4. **Submit** pull request

### Code Style
- Follow PEP 8
- Add type hints
- Include docstrings
- Write unit tests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hikima et al.** for the original pricing optimization methodology
- **Gupta & Nagarajan** for the linear programming formulation
- **NYC TLC** for providing taxi trip data
- **Research community** for methodological foundations

## 📞 Support

For questions, issues, or contributions:
- 🐛 **Issues**: Use GitHub Issues
- 📧 **Email**: [contact information]
- 📚 **Documentation**: See `/docs` directory
- 💬 **Discussions**: GitHub Discussions

---

**🏆 This framework provides a clean, scalable platform for rigorous ride-hailing pricing optimization research with real-world applicability!** 